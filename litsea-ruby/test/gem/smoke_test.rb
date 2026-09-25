# frozen_string_literal: true

# Smoke test for an installed litsea gem: checks that it was loaded through
# RubyGems and compiled against the litsea crate of its own version, then
# segments and tags the README example with it.
#
# Usage: ruby smoke_test.rb MODEL   (a two-stage POS model: models/japanese_pos.model)
#
# Run by install_and_test.sh; `rake test` does not pick it up.

require 'litsea'

spec = Gem.loaded_specs['litsea']
abort 'litsea was not loaded from an installed gem' unless spec
puts "litsea gem #{spec.version} (#{spec.full_gem_path}), Litsea.version #{Litsea.version}"

# Cargo.toml pins the litsea crate to the gem's version, so anything else
# means the gem compiled against another release.
abort "the gem #{spec.version} was compiled against litsea #{Litsea.version}" unless Litsea.version == spec.version.to_s

model = ARGV.fetch(0) { abort 'usage: ruby smoke_test.rb MODEL' }
segmenter = Litsea::Segmenter.open(:japanese, model)
text = 'これはテストです。'

tokens = segmenter.segment(text)
puts tokens.join(' / ')
expected_tokens = %w[これ は テスト です 。]
abort "expected #{expected_tokens.inspect}, got #{tokens.inspect}" unless tokens == expected_tokens

tags = segmenter.segment_with_pos(text).map { |token| "#{token.surface}/#{token.pos}" }
puts tags.join(' ')
expected_tags = %w[これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT]
abort "expected #{expected_tags.inspect}, got #{tags.inspect}" unless tags == expected_tags

puts 'OK'
