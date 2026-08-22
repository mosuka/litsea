# frozen_string_literal: true

# Segment a sentence, with and without POS tags.
#
# Usage:
#   bundle exec ruby -Ilib examples/segment.rb ../models/japanese.model "これはテストです。"

require 'litsea'

model_path, text = ARGV
if model_path.nil? || text.nil?
  warn 'usage: ruby examples/segment.rb <model> <text>'
  exit 2
end

# The model file identifies its own kind, so nothing here declares whether
# this is a POS model - has_pos? reports what was loaded.
segmenter = Litsea::Segmenter.open(:japanese, model_path)
puts "model: #{model_path} (has_pos?=#{segmenter.has_pos?})"
puts "tokens: #{segmenter.segment(text).join(' ')}"

return unless segmenter.has_pos?

puts 'tagged:'
segmenter.segment_with_pos(text).each do |token|
  puts format("  %<surface>s\t%<pos>s\t[%<start>d:%<end>d]",
              surface: token.surface, pos: token.pos, start: token.start, end: token.end)
end
