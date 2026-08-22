# frozen_string_literal: true

require 'minitest/autorun'
require 'open3'
require 'tmpdir'
require 'litsea'

# Shared helpers: model paths and the CLI the parity tests compare against.
class LitseaTest < Minitest::Test
  REPO_ROOT = File.expand_path('../..', __dir__)

  # Absolute path to a bundled model.
  def model_path(name)
    File.join(REPO_ROOT, 'models', name)
  end

  # Builds the `litsea` CLI once and returns the path to the binary.
  #
  # The parity tests compare against the CLI rather than hardcoded output,
  # so the reference implementation decides what is correct.
  def litsea_cli
    binary = File.join(REPO_ROOT, 'target', 'debug', 'litsea')
    unless File.exist?(binary)
      _, stderr, status = Open3.capture3('cargo', 'build', '--quiet', '-p', 'litsea-cli', chdir: REPO_ROOT)
      raise "failed to build the litsea CLI: #{stderr}" unless status.success?
    end
    binary
  end

  # Runs the CLI over +input+ and returns its output lines.
  def run_cli(args, input)
    stdout, stderr, status = Open3.capture3(litsea_cli, *args, stdin_data: input)
    raise "the CLI failed: #{stderr}" unless status.success?

    stdout.split("\n").reject(&:empty?)
  end

  # Writes a small corpus, repeated so training has something to learn.
  def write_corpus(path, sentences, repeats: 20)
    File.write(path, "#{(sentences * repeats).join("\n")}\n")
    path
  end

  SENTENCES = [
    'これ は テスト です 。',
    '隣 の 客 は よく 柿 食う 客 だ',
    '東京 都 から 神奈川 県 へ 引っ越し た'
  ].freeze

  POS_SENTENCES = [
    'これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT',
    '隣/NOUN の/ADP 客/NOUN は/ADP よく/ADV 柿/NOUN 食う/VERB 客/NOUN だ/AUX',
    '東京/PROPN 都/NOUN から/ADP 神奈川/PROPN 県/NOUN へ/ADP 引っ越し/VERB た/AUX'
  ].freeze
end
