# frozen_string_literal: true

# Train a segmentation model, cancelling it from another thread.
#
# Usage:
#   bundle exec ruby -Ilib examples/train.rb corpus.txt out.model
#
# The corpus is one sentence per line, with words separated by spaces:
#
#   これ は テスト です 。

require 'litsea'
require 'tmpdir'

corpus, model = ARGV
if corpus.nil? || model.nil?
  warn 'usage: ruby examples/train.rb <corpus> <model>'
  exit 2
end

work_dir = Dir.mktmpdir('litsea-')
features = File.join(work_dir, 'features.txt')

puts "extracting features from #{corpus} ..."
Litsea::Extractor.new(:japanese).extract(corpus, features)

# Training releases the GVL, so this thread keeps running and can stop the
# run. Cancelling is not an error: the partial model is still written.
cancel = Litsea::CancelToken.new
watchdog = Thread.new do
  sleep 60
  cancel.cancel
end

puts 'training (stops after 60s if it has not converged) ...'
metrics = Litsea::Trainer.new(0.01, 10_000, features).train(model, cancel: cancel)
watchdog.kill

puts "wrote #{model}"
puts format('  accuracy:  %.2f%%', metrics.accuracy)
puts format('  precision: %.2f%%', metrics.precision)
puts format('  recall:    %.2f%%', metrics.recall)
puts "  instances: #{metrics.num_instances}"
puts '  (training was cancelled; the model is partially trained)' if cancel.cancelled?
