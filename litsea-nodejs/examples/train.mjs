/**
 * Train a segmentation model without blocking the event loop.
 *
 * Usage:
 *   node examples/train.mjs corpus.txt out.model
 *
 * The corpus is one sentence per line, with words separated by spaces:
 *
 *   これ は テスト です 。
 */

import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { CancelToken, Extractor, Trainer } from '../index.js'

const [, , corpus, model] = process.argv

if (!corpus || !model) {
  console.error('usage: node examples/train.mjs <corpus> <model>')
  process.exit(2)
}

const workDir = mkdtempSync(join(tmpdir(), 'litsea-'))
const features = join(workDir, 'features.txt')

console.log(`extracting features from ${corpus} ...`)
await new Extractor('japanese').extract(corpus, features)

// Training runs on a worker thread, so timers keep firing and the token can
// stop it. Cancelling is not an error: the partial model is still written.
const cancel = new CancelToken()
const deadline = setTimeout(() => cancel.cancel(), 60_000)
const progress = setInterval(() => process.stdout.write('.'), 500)

console.log('training (stops after 60s if it has not converged) ...')
try {
  const metrics = await new Trainer(0.01, 10_000, features).train(model, cancel)
  process.stdout.write('\n')
  console.log(`wrote ${model}`)
  console.log(`  accuracy:  ${metrics.accuracy.toFixed(2)}%`)
  console.log(`  precision: ${metrics.precision.toFixed(2)}%`)
  console.log(`  recall:    ${metrics.recall.toFixed(2)}%`)
  console.log(`  instances: ${metrics.numInstances}`)
  if (cancel.cancelled) {
    console.log('  (training was cancelled; the model is partially trained)')
  }
} finally {
  clearTimeout(deadline)
  clearInterval(progress)
}
