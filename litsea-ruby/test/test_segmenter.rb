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

  # User dictionary cases (#263): the model splits each registered word
  # without the dictionary, and the English word contains a space.
  USER_WORD_CASES = [
    ['japanese', 'japanese.model', %w[雪だるま 新幹線], '雪だるまを作る'],
    ['japanese', 'japanese.model', %w[雪だるま 新幹線], '新幹線で帰る'],
    ['english', 'english.model', ['New York'], 'I live in New York.']
  ].freeze

  # Writes a `--user-words` file for the CLI into +dir+ and returns its path.
  def write_words(dir, words)
    path = File.join(dir, 'words.txt')
    File.write(path, words.map { |word| "#{word}\n" }.join)
    path
  end

  def test_user_words_match_the_cli
    USER_WORD_CASES.each do |language, model, words, sentence|
      Dir.mktmpdir do |dir|
        expected = run_cli(
          ['segment', '-l', language, '--user-words', write_words(dir, words), model_path(model)], "#{sentence}\n"
        ).first
        seg = Litsea::Segmenter.open(language, model_path(model), user_words: words)
        assert_equal expected, seg.segment(sentence).join(' '), sentence
        # The dictionary must actually matter for these cases.
        refute_equal Litsea::Segmenter.open(language, model_path(model)).segment(sentence), seg.segment(sentence)
      end
    end
  end

  def test_user_words_with_pos_match_the_cli
    sentence = '雪だるまを作る'
    words = ['雪だるま']
    Dir.mktmpdir do |dir|
      expected = run_cli(
        ['segment', '-l', 'japanese', '--pos', '--user-words', write_words(dir, words),
         model_path('japanese_pos.model')], "#{sentence}\n"
      ).first
      tokens = Litsea::Segmenter.open(:japanese, model_path('japanese_pos.model'), user_words: words)
                                .segment_with_pos(sentence)
      assert_equal expected, tokens.map { |t| "#{t.surface}/#{t.pos}" }.join(' ')
      assert_equal '雪だるま', tokens.first.surface
    end
  end

  def test_user_words_on_every_constructor
    path = model_path('japanese.model')
    sentence = '雪だるまを作る'
    expected = %w[雪だるま を 作る]

    assert_equal expected, Litsea::Segmenter.open(:japanese, path, user_words: ['雪だるま']).segment(sentence)
    assert_equal expected,
                 Litsea::Segmenter.from_bytes(:japanese, File.binread(path), user_words: ['雪だるま']).segment(sentence)
    assert_equal expected, Litsea::Segmenter.from_uri(:japanese, path, user_words: ['雪だるま']).segment(sentence)
  end

  def test_user_words_gate
    path = model_path('japanese.model')
    sentence = '雪だるまを作る'
    plain = Litsea::Segmenter.open(:japanese, path).segment(sentence)

    # Gate 0 applies a word only where the model already agrees with it.
    gated = Litsea::Segmenter.open(:japanese, path, user_words: ['雪だるま'], user_words_gate: 0.0)
    assert_equal plain, gated.segment(sentence)
    open_gate = Litsea::Segmenter.open(:japanese, path, user_words: ['雪だるま'], user_words_gate: 1.0)
    assert_equal %w[雪だるま を 作る], open_gate.segment(sentence)
    # An empty array or nil is no dictionary at all.
    assert_equal plain, Litsea::Segmenter.open(:japanese, path, user_words: []).segment(sentence)
    assert_equal plain, Litsea::Segmenter.open(:japanese, path, user_words: nil, user_words_gate: nil).segment(sentence)
  end

  def test_user_words_invalid_arguments
    path = model_path('japanese.model')
    [['雪だるま'], nil].each do |words|
      error = assert_raises(Litsea::InvalidArgumentError) do
        Litsea::Segmenter.open(:japanese, path, user_words: words, user_words_gate: 1.5)
      end
      assert_match(/0\.0\.\.=1\.0/, error.message)
    end
    assert_raises(ArgumentError) { Litsea::Segmenter.open(:japanese, path, user_word: ['雪だるま']) }
    assert_raises(ArgumentError) { Litsea::Segmenter.open(:japanese) }
    assert_raises(TypeError) { Litsea::Segmenter.open(:japanese, path, user_words: '雪だるま') }
  end

  # Tagged user dictionary entries (#264): the tagger gives these words NOUN
  # without a tag (`ポケモン` through the lexicon, `チャットGPT` through the
  # full argmax). `楓` is a one-character, tag-only entry (#276).
  USER_WORD_TAGS = { 'チャットGPT' => 'PROPN', 'ポケモン' => 'PROPN', '楓' => 'PROPN' }.freeze
  TAGGED_SENTENCES = %w[チャットGPTに聞く ポケモンを集める 雪だるまを作る 楓が笑った。].freeze

  # Renders `segment_with_pos` the way `litsea segment --pos` prints it.
  def tagged(segmenter, sentence)
    segmenter.segment_with_pos(sentence).map { |t| "#{t.surface}/#{t.pos}" }.join(' ')
  end

  def test_user_word_tags_match_the_cli
    model = model_path('japanese_pos.model')
    lines = USER_WORD_TAGS.map { |word, tag| "#{word}\t#{tag}" } + ['雪だるま']
    Dir.mktmpdir do |dir|
      expected = run_cli(
        ['segment', '-l', 'japanese', '--pos', '--user-words', write_words(dir, lines), model],
        TAGGED_SENTENCES.map { |sentence| "#{sentence}\n" }.join
      )
      seg = Litsea::Segmenter.open(:japanese, model, user_words: ['雪だるま'], user_word_tags: USER_WORD_TAGS)
      assert_equal(expected, TAGGED_SENTENCES.map { |sentence| tagged(seg, sentence) })
      assert tagged(seg, 'チャットGPTに聞く').start_with?('チャットGPT/PROPN ')
      # The tags must actually matter: untagged, the same words come out as NOUN.
      untagged = Litsea::Segmenter.open(:japanese, model, user_words: ['雪だるま', *USER_WORD_TAGS.keys])
      assert tagged(untagged, 'チャットGPTに聞く').start_with?('チャットGPT/NOUN ')
      refute_equal tagged(untagged, 'ポケモンを集める'), tagged(seg, 'ポケモンを集める')
      assert tagged(seg, '楓が笑った。').start_with?('楓/PROPN ')
      assert tagged(untagged, '楓が笑った。').start_with?('楓/NOUN ')
    end
  end

  def test_user_word_tags_on_every_constructor
    path = model_path('japanese_pos.model')
    sentence = 'チャットGPTに聞く'

    from_path = Litsea::Segmenter.open(:japanese, path, user_word_tags: USER_WORD_TAGS)
    from_bytes = Litsea::Segmenter.from_bytes(:japanese, File.binread(path), user_word_tags: USER_WORD_TAGS)
    from_uri = Litsea::Segmenter.from_uri(:japanese, path, user_word_tags: USER_WORD_TAGS)

    token = from_path.segment_with_pos(sentence).first
    assert_equal %w[チャットGPT PROPN], [token.surface, token.pos]
    assert_equal tagged(from_path, sentence), tagged(from_bytes, sentence)
    assert_equal tagged(from_path, sentence), tagged(from_uri, sentence)
    # Segmentation alone ignores the tags.
    assert_equal %w[チャットGPT に 聞く], from_path.segment(sentence)
  end

  def test_user_word_tags_invalid_arguments
    path = model_path('japanese_pos.model')
    error = assert_raises(Litsea::InvalidArgumentError) do
      Litsea::Segmenter.open(:japanese, path, user_word_tags: { 'ポケモン' => 'Noun' })
    end
    assert_match(/unknown UPOS tag "Noun"/, error.message)
    # A bad gate is still reported when only tagged words are given.
    error = assert_raises(Litsea::InvalidArgumentError) do
      Litsea::Segmenter.open(:japanese, path, user_word_tags: USER_WORD_TAGS, user_words_gate: 1.5)
    end
    assert_match(/0\.0\.\.=1\.0/, error.message)
    assert_raises(TypeError) { Litsea::Segmenter.open(:japanese, path, user_word_tags: ['ポケモン']) }
  end

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
