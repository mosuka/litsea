# frozen_string_literal: true

require_relative 'test_helper'

# Segmentation, POS tagging, and the exception hierarchy.
class TestSegmenter < LitseaTest
  SEGMENTATION_CASES = [
    ['japanese', 'japanese.model', 'これはテストです。'],
    ['chinese', 'chinese.model', '我喜欢吃中国菜。'],
    ['korean', 'korean.model', '안녕하세요 반갑습니다'],
    ['english', 'english.model', 'The quick brown fox jumps over the lazy dog.']
  ].freeze

  POS_CASES = [
    ['japanese', 'japanese_pos.model', 'これはテストです。'],
    ['korean', 'korean_pos.model', '안녕하세요 반갑습니다']
  ].freeze

  def test_segment_matches_the_cli
    SEGMENTATION_CASES.each do |language, model, sentence|
      # Compare the rendered line rather than a re-split of it: the CLI joins
      # tokens with a space, so for Korean and English -- where whitespace is
      # its own token -- splitting the output again cannot recover them.
      expected = run_cli(['segment', '-l', language, model_path(model)], "#{sentence}\n").first
      seg = Litsea::Segmenter.open(language, model_path(model))
      assert_equal expected, seg.segment(sentence).join(' '), language
    end
  end

  def test_segment_with_pos_matches_the_cli
    POS_CASES.each do |language, model, sentence|
      expected = run_cli(['segment', '-l', language, '--pos', model_path(model)], "#{sentence}\n").first
      seg = Litsea::Segmenter.open(language, model_path(model))
      rendered = seg.segment_with_pos(sentence).map { |t| "#{t.surface}/#{t.pos}" }.join(' ')
      assert_equal expected, rendered, language
    end
  end

  def test_byte_offsets_reconstruct_the_input
    SEGMENTATION_CASES.each do |language, model, sentence|
      seg = Litsea::Segmenter.open(language, model_path(model))
      tokens = seg.segment_tokens(sentence)

      refute_empty tokens
      expected_start = 0
      tokens.each do |token|
        assert_equal expected_start, token.start, "#{language}: tokens must tile the input"
        # Ruby's String#[] counts characters, so slice by bytes.
        assert_equal token.surface, sentence.byteslice(token.start, token.end - token.start)
        assert_nil token.pos
        expected_start = token.end
      end
      assert_equal sentence.bytesize, expected_start
      assert_equal sentence, tokens.map(&:surface).join
    end
  end

  def test_whitespace_is_its_own_token
    seg = Litsea::Segmenter.open(:korean, model_path('korean.model'))
    assert_equal ['안녕하세요', ' ', '반갑습니다'], seg.segment('안녕하세요 반갑습니다')
  end

  def test_segment_batch_matches_single_calls
    seg = Litsea::Segmenter.open(:japanese, model_path('japanese.model'))
    sentences = ['これはテストです。', '', '東京都から神奈川県へ引っ越した']

    batched = seg.segment_batch(sentences)
    assert_equal sentences.map { |s| seg.segment(s) }, batched
    assert_empty batched[1]
  end

  def test_segment_with_pos_batch_matches_single_calls
    seg = Litsea::Segmenter.open(:japanese, model_path('japanese_pos.model'))
    sentences = ['これはテストです。', '東京都から神奈川県へ引っ越した']

    batched = seg.segment_with_pos_batch(sentences)
    assert_equal 2, batched.length
    sentences.each_with_index do |sentence, index|
      expected = seg.segment_with_pos(sentence).map { |t| "#{t.surface}/#{t.pos}" }
      assert_equal(expected, batched[index].map { |t| "#{t.surface}/#{t.pos}" })
    end
  end

  def test_model_kind_is_detected
    refute Litsea::Segmenter.open(:ja, model_path('japanese.model')).has_pos?
    assert Litsea::Segmenter.open(:ja, model_path('japanese_pos.model')).has_pos?
  end

  def test_language_accepts_symbols_and_strings
    expected = Litsea::Segmenter.open('japanese', model_path('japanese.model')).segment('これはテストです。')
    [:ja, :japanese, 'ja', 'JA', 'Japanese'].each do |name|
      seg = Litsea::Segmenter.open(name, model_path('japanese.model'))
      assert_equal expected, seg.segment('これはテストです。'), name.inspect
    end
  end

  def test_loading_sources_agree
    path = model_path('japanese.model')
    sentence = 'これはテストです。'

    from_path = Litsea::Segmenter.open(:japanese, path)
    from_bytes = Litsea::Segmenter.from_bytes(:japanese, File.binread(path))
    from_uri = Litsea::Segmenter.from_uri(:japanese, path)

    assert_equal from_path.segment(sentence), from_bytes.segment(sentence)
    assert_equal from_path.segment(sentence), from_uri.segment(sentence)
  end

  def test_language_accessor
    assert_equal 'korean', Litsea::Segmenter.open(:korean, model_path('korean.model')).language
  end

  def test_pos_on_segmentation_model_raises
    seg = Litsea::Segmenter.open(:japanese, model_path('japanese.model'))
    error = assert_raises(Litsea::PosUnavailableError) { seg.segment_with_pos('これはテストです。') }
    assert_match(/two-stage POS model/, error.message)
  end

  def test_unknown_language_raises
    error = assert_raises(Litsea::InvalidArgumentError) do
      Litsea::Segmenter.open(:klingon, model_path('japanese.model'))
    end
    assert_match(/klingon/, error.message)
  end

  def test_missing_model_raises
    assert_raises(Litsea::IoError) { Litsea::Segmenter.open(:japanese, model_path('nope.model')) }
  end

  def test_malformed_model_raises
    Dir.mktmpdir do |dir|
      path = File.join(dir, 'broken.model')
      File.write(path, "this is not a model\n")
      assert_raises(Litsea::ParseError) { Litsea::Segmenter.open(:japanese, path) }
    end
  end

  def test_legacy_joint_model_raises
    Dir.mktmpdir do |dir|
      path = File.join(dir, 'joint.model')
      # A bare integer first line is the joint class-count header.
      File.write(path, "17\nfoo\t1.0\n")
      error = assert_raises(Litsea::ModelError) { Litsea::Segmenter.open(:japanese, path) }
      assert_match(/no longer supported/, error.message)
    end
  end

  def test_every_error_derives_from_the_base
    [
      Litsea::InvalidArgumentError,
      Litsea::IoError,
      Litsea::ModelError,
      Litsea::ParseError,
      Litsea::UnsupportedError,
      Litsea::PosUnavailableError
    ].each do |klass|
      assert_operator klass, :<, Litsea::Error, klass.name
    end

    # One rescue is enough for anything the binding raises.
    assert_raises(Litsea::Error) { Litsea::Segmenter.open(:klingon, model_path('japanese.model')) }
  end

  def test_module_functions
    assert_match(/\A\d+\.\d+\.\d+\z/, Litsea.version)
    assert_equal %w[japanese chinese korean english], Litsea.supported_languages
    assert_equal Litsea::VERSION, Litsea.version
  end
end
