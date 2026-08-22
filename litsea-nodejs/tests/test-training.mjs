/** Feature extraction, training, cancellation, and event-loop behaviour. */

import assert from 'node:assert/strict'
import { mkdtempSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'

import { CancelToken, Extractor, Segmenter, Trainer, TwoStageTrainer, PerceptronTrainer } from '../index.js'
import { runCli } from './helpers.mjs'

const SENTENCES = [
  'これ は テスト です 。',
  '隣 の 客 は よく 柿 食う 客 だ',
  '東京 都 から 神奈川 県 へ 引っ越し た',
]

const POS_SENTENCES = [
  'これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT',
  '隣/NOUN の/ADP 客/NOUN は/ADP よく/ADV 柿/NOUN 食う/VERB 客/NOUN だ/AUX',
  '東京/PROPN 都/NOUN から/ADP 神奈川/PROPN 県/NOUN へ/ADP 引っ越し/VERB た/AUX',
]

/**
 * Creates a temporary directory for a test's artifacts.
 *
 * @returns {string} The directory path.
 */
function tempDir() {
  return mkdtempSync(join(tmpdir(), 'litsea-nodejs-'))
}

/**
 * Writes a small corpus, repeated so training has something to learn.
 *
 * @param {string} path - Where to write the corpus.
 * @param {string[]} sentences - The sentences to repeat.
 * @param {number} repeats - How many times to repeat them.
 * @returns {string} The corpus path.
 */
function writeCorpus(path, sentences, repeats = 20) {
  const lines = []
  for (let i = 0; i < repeats; i += 1) {
    lines.push(...sentences)
  }
  writeFileSync(path, `${lines.join('\n')}\n`)
  return path
}

test('extract then train produces a model the CLI can load', async () => {
  const dir = tempDir()
  const corpus = writeCorpus(join(dir, 'corpus.txt'), SENTENCES)
  const features = join(dir, 'features.txt')
  const model = join(dir, 'trained.model')

  await new Extractor('japanese').extract(corpus, features)
  assert.ok(statSync(features).size > 0)

  const metrics = await new Trainer(0.01, 20, features).train(model)
  assert.ok(metrics.numInstances > 0)
  assert.ok(metrics.accuracy >= 0 && metrics.accuracy <= 100)

  // The CLI is the independent check that the file is a valid model.
  const [line] = runCli(['segment', '-l', 'japanese', model], 'これはテストです。\n')
  assert.ok(line.length > 0)
  assert.equal(Segmenter.open('japanese', model).segment('これはテストです。').join(' '), line)
})

test('tag-free extraction writes fewer features', async () => {
  const dir = tempDir()
  const corpus = writeCorpus(join(dir, 'corpus.txt'), SENTENCES)
  const full = join(dir, 'full.txt')
  const lean = join(dir, 'lean.txt')
  const extractor = new Extractor('japanese')

  await extractor.extract(corpus, full)
  await extractor.extract(corpus, lean, false, true)

  assert.ok(statSync(lean).size < statSync(full).size)
})

test('two-stage training produces a POS-capable model', async () => {
  const dir = tempDir()
  const corpus = writeCorpus(join(dir, 'corpus_pos.txt'), POS_SENTENCES)
  const prefix = join(dir, 'features')
  const model = join(dir, 'two_stage.model')

  await new Extractor('japanese').extractTwoStage(corpus, prefix, 'fast')
  for (const suffix of ['stage1', 'stage2', 'lexicon']) {
    assert.ok(statSync(`${prefix}.${suffix}`).size > 0)
  }

  const trainer = new TwoStageTrainer(3, prefix)
  assert.equal(trainer.available, true)
  const metrics = await trainer.train(model)
  assert.ok(metrics.stage1.numInstances > 0)
  assert.ok(metrics.stage2.numInstances > 0)

  const seg = Segmenter.open('japanese', model)
  assert.equal(seg.hasPos, true)
  const tokens = seg.segmentWithPos('これはテストです。')
  assert.ok(tokens.length > 0)
  assert.ok(tokens.every((token) => typeof token.pos === 'string'))
})

test('a two-stage trainer cannot be reused', async () => {
  const dir = tempDir()
  const corpus = writeCorpus(join(dir, 'corpus_pos.txt'), POS_SENTENCES)
  const prefix = join(dir, 'features')
  const model = join(dir, 'two_stage.model')

  await new Extractor('japanese').extractTwoStage(corpus, prefix)
  const trainer = new TwoStageTrainer(1, prefix)
  await trainer.train(model)
  assert.equal(trainer.available, false)

  await assert.rejects(trainer.train(model), (error) => {
    assert.equal(error.code, 'invalid_argument')
    assert.match(error.message, /already been used/)
    return true
  })
})

test('the perceptron trainer trains from stage-2 features', async () => {
  const dir = tempDir()
  const corpus = writeCorpus(join(dir, 'corpus_pos.txt'), POS_SENTENCES)
  const prefix = join(dir, 'features')
  const model = join(dir, 'perceptron.model')

  await new Extractor('japanese').extractTwoStage(corpus, prefix)
  const metrics = await new PerceptronTrainer(2, `${prefix}.stage2`).train(model)

  assert.ok(metrics.numInstances > 0)
  assert.ok(Object.keys(metrics.goldPerClass).length > 0)
})

test('cancelling before training still writes a model', async () => {
  const dir = tempDir()
  const corpus = writeCorpus(join(dir, 'corpus.txt'), SENTENCES)
  const features = join(dir, 'features.txt')
  const model = join(dir, 'cancelled.model')

  await new Extractor('japanese').extract(corpus, features)

  const cancel = new CancelToken()
  cancel.cancel()
  assert.equal(cancel.cancelled, true)

  // Cancelling is cooperative, not an error.
  const metrics = await new Trainer(0.01, 100000, features).train(model, cancel)
  assert.ok(metrics.numInstances > 0)
  assert.ok(statSync(model).size > 0)
})

test('training does not block the event loop', async () => {
  // If `train` ran on the main thread, no timer could fire while it worked
  // and `ticks` would stay at 0. The iteration count is bounded so a
  // regression fails quickly instead of hanging.
  const dir = tempDir()
  const corpus = writeCorpus(join(dir, 'corpus.txt'), SENTENCES, 400)
  const features = join(dir, 'features.txt')
  const model = join(dir, 'trained.model')

  await new Extractor('japanese').extract(corpus, features)

  let ticks = 0
  const timer = setInterval(() => {
    ticks += 1
  }, 5)

  try {
    await new Trainer(0.0, 40, features).train(model)
  } finally {
    clearInterval(timer)
  }

  assert.ok(ticks > 0, `the event loop was blocked during training (ticks=${ticks})`)
})

test('a cancel token can stop training that is already running', async () => {
  const dir = tempDir()
  const corpus = writeCorpus(join(dir, 'corpus.txt'), SENTENCES, 400)
  const features = join(dir, 'features.txt')
  const model = join(dir, 'cancelled.model')

  await new Extractor('japanese').extract(corpus, features)

  const cancel = new CancelToken()
  const started = Date.now()
  const training = new Trainer(0.0, 400, features).train(model, cancel)
  setTimeout(() => cancel.cancel(), 100)

  const metrics = await training
  assert.ok(metrics.numInstances > 0)
  assert.equal(cancel.cancelled, true)
  assert.ok(Date.now() - started < 30000)
})
