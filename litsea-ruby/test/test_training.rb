# frozen_string_literal: true

require_relative 'test_helper'

# Feature extraction, training, cancellation, and GVL behaviour.
class TestTraining < LitseaTest
  def test_extract_then_train_round_trip
    Dir.mktmpdir do |dir|
      corpus = write_corpus(File.join(dir, 'corpus.txt'), SENTENCES)
      features = File.join(dir, 'features.txt')
      model = File.join(dir, 'trained.model')

      Litsea::Extractor.new(:japanese).extract(corpus, features)
      assert_operator File.size(features), :>, 0

      metrics = Litsea::Trainer.new(0.01, 20, features).train(model)
      assert_operator metrics.num_instances, :>, 0
      assert_includes 0.0..100.0, metrics.accuracy

      # The CLI is the independent check that the file is a valid model.
      line = run_cli(['segment', '-l', 'japanese', model], "これはテストです。\n").first
      refute_empty line
      assert_equal line, Litsea::Segmenter.open(:japanese, model).segment('これはテストです。').join(' ')
    end
  end

  def test_tag_free_extraction_is_smaller
    Dir.mktmpdir do |dir|
      corpus = write_corpus(File.join(dir, 'corpus.txt'), SENTENCES)
      extractor = Litsea::Extractor.new(:japanese)

      extractor.extract(corpus, File.join(dir, 'full.txt'))
      extractor.extract(corpus, File.join(dir, 'lean.txt'), tag_free: true)

      assert_operator File.size(File.join(dir, 'lean.txt')), :<, File.size(File.join(dir, 'full.txt'))
    end
  end

  def test_two_stage_training_round_trip
    Dir.mktmpdir do |dir|
      corpus = write_corpus(File.join(dir, 'corpus_pos.txt'), POS_SENTENCES)
      prefix = File.join(dir, 'features')
      model = File.join(dir, 'two_stage.model')

      Litsea::Extractor.new(:japanese).extract_two_stage(corpus, prefix, feature_set: 'fast')
      %w[stage1 stage2 lexicon].each { |suffix| assert_path_exists "#{prefix}.#{suffix}" }

      trainer = Litsea::TwoStageTrainer.new(3, prefix)
      assert trainer.available?
      metrics = trainer.train(model)
      assert_operator metrics.stage1.num_instances, :>, 0
      assert_operator metrics.stage2.num_instances, :>, 0

      seg = Litsea::Segmenter.open(:japanese, model)
      assert seg.has_pos?
      tokens = seg.segment_with_pos('これはテストです。')
      refute_empty tokens
      assert(tokens.all? { |token| !token.pos.nil? })
    end
  end

  def test_two_stage_trainer_cannot_be_reused
    Dir.mktmpdir do |dir|
      corpus = write_corpus(File.join(dir, 'corpus_pos.txt'), POS_SENTENCES)
      prefix = File.join(dir, 'features')
      model = File.join(dir, 'two_stage.model')

      Litsea::Extractor.new(:japanese).extract_two_stage(corpus, prefix)
      trainer = Litsea::TwoStageTrainer.new(1, prefix)
      trainer.train(model)
      refute trainer.available?

      error = assert_raises(Litsea::InvalidArgumentError) { trainer.train(model) }
      assert_match(/already been used/, error.message)
    end
  end

  def test_perceptron_trainer
    Dir.mktmpdir do |dir|
      corpus = write_corpus(File.join(dir, 'corpus_pos.txt'), POS_SENTENCES)
      prefix = File.join(dir, 'features')
      model = File.join(dir, 'perceptron.model')

      Litsea::Extractor.new(:japanese).extract_two_stage(corpus, prefix)
      metrics = Litsea::PerceptronTrainer.new(2, "#{prefix}.stage2").train(model)

      assert_operator metrics.num_instances, :>, 0
      refute_empty metrics.gold_per_class
      assert_path_exists model
    end
  end

  def test_cancelling_before_training_still_writes_a_model
    Dir.mktmpdir do |dir|
      corpus = write_corpus(File.join(dir, 'corpus.txt'), SENTENCES)
      features = File.join(dir, 'features.txt')
      model = File.join(dir, 'cancelled.model')

      Litsea::Extractor.new(:japanese).extract(corpus, features)

      token = Litsea::CancelToken.new
      token.cancel
      assert token.cancelled?

      # Cancelling is cooperative, not an error.
      metrics = Litsea::Trainer.new(0.01, 100_000, features).train(model, cancel: token)
      assert_operator metrics.num_instances, :>, 0
      assert_path_exists model
    end
  end

  def test_cancel_token_reset
    token = Litsea::CancelToken.new
    refute token.cancelled?
    token.cancel
    assert token.cancelled?
    token.reset
    refute token.cancelled?
  end

  def test_training_releases_the_gvl
    # If training held the GVL, no other Ruby thread could run while it
    # worked and `ticks` would stay at 0. Neither magnus nor rb-sys wraps
    # `rb_thread_call_without_gvl`, so this is what proves the hand-written
    # FFI in `src/gvl.rs` does its job.
    Dir.mktmpdir do |dir|
      corpus = write_corpus(File.join(dir, 'corpus.txt'), SENTENCES, repeats: 400)
      features = File.join(dir, 'features.txt')
      Litsea::Extractor.new(:japanese).extract(corpus, features)

      ticks = 0
      ticker = Thread.new do
        loop do
          ticks += 1
          sleep 0.005
        end
      end

      begin
        Litsea::Trainer.new(0.0, 40, features).train(File.join(dir, 'trained.model'))
      ensure
        ticker.kill
      end

      assert_operator ticks, :>, 0, "the GVL was held during training (ticks=#{ticks})"
    end
  end

  def test_a_ruby_thread_can_cancel_training_in_flight
    Dir.mktmpdir do |dir|
      corpus = write_corpus(File.join(dir, 'corpus.txt'), SENTENCES, repeats: 400)
      features = File.join(dir, 'features.txt')
      Litsea::Extractor.new(:japanese).extract(corpus, features)

      token = Litsea::CancelToken.new
      cancelled_at = nil
      canceller = Thread.new do
        sleep 0.15
        cancelled_at = Time.now
        token.cancel
      end

      started_at = Time.now
      # A high iteration count so the run cannot finish before the cancel.
      metrics = Litsea::Trainer.new(0.0, 2000, features).train(File.join(dir, 'm.model'), cancel: token)
      finished_at = Time.now
      canceller.join

      assert_operator metrics.num_instances, :>, 0
      assert token.cancelled?
      # The cancel must land inside the training window, not before or after
      # it -- that is what makes this a test of in-flight cancellation.
      assert_operator cancelled_at, :>, started_at
      assert_operator cancelled_at, :<, finished_at
    end
  end
end
