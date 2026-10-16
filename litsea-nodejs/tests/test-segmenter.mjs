/** Segmentation, POS tagging, and error codes. */

import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'

import { Segmenter, supportedLanguages, version } from '../index.js'
import { modelPath, runCli } from './helpers.mjs'

// One sentence per language, with the model that handles it.
const SEGMENTATION_CASES = [
  ['japanese', 'japanese.model', 'これはテストです。'],
  ['chinese', 'chinese.model', '我喜欢吃中国菜。'],
  ['korean', 'korean.model', '안녕하세요 반갑습니다'],
  ['english', 'english.model', 'The quick brown fox jumps over the lazy dog.'],
]

const POS_CASES = [
  ['japanese', 'japanese_pos.model', 'これはテストです。'],
  ['korean', 'korean_pos.model', '안녕하세요 반갑습니다'],
]

// User dictionary cases (#263): the model splits each registered word
// without the dictionary, and the English word contains a space.
const USER_WORD_CASES = [
  ['japanese', 'japanese.model', ['雪だるま', '新幹線'], '雪だるまを作る'],
  ['japanese', 'japanese.model', ['雪だるま', '新幹線'], '新幹線で帰る'],
  ['english', 'english.model', ['New York'], 'I live in New York.'],
]

/**
 * Writes a `--user-words` file for the CLI.
 *
 * @param {string[]} words - The words, one per line.
 * @returns {string} The file path.
 */
function writeWords(words) {
  const path = join(mkdtempSync(join(tmpdir(), 'litsea-words-')), 'words.txt')
  writeFileSync(path, words.map((word) => `${word}\n`).join(''))
  return path
}

for (const [language, model, words, sentence] of USER_WORD_CASES) {
  test(`userWords matches the CLI (${language}: ${sentence})`, () => {
    const [expected] = runCli(
      ['segment', '-l', language, '--user-words', writeWords(words), modelPath(model)],
      `${sentence}\n`,
    )
    const seg = Segmenter.open(language, modelPath(model), words)
    assert.equal(seg.segment(sentence).join(' '), expected)
    // The dictionary must actually matter for these cases.
    assert.notDeepEqual(seg.segment(sentence), Segmenter.open(language, modelPath(model)).segment(sentence))
  })
}

test('userWords with POS matches the CLI', () => {
  const sentence = '雪だるまを作る'
  const words = ['雪だるま']
  const [expected] = runCli(
    ['segment', '-l', 'japanese', '--pos', '--user-words', writeWords(words), modelPath('japanese_pos.model')],
    `${sentence}\n`,
  )
  const seg = Segmenter.open('japanese', modelPath('japanese_pos.model'), words)
  const tokens = seg.segmentWithPos(sentence)
  assert.equal(tokens.map((token) => `${token.surface}/${token.pos}`).join(' '), expected)
  assert.equal(tokens[0].surface, '雪だるま')
})

test('userWords on every constructor', async () => {
  const path = modelPath('japanese.model')
  const sentence = '雪だるまを作る'
  const words = ['雪だるま']
  const expected = ['雪だるま', 'を', '作る']

  assert.deepEqual(Segmenter.open('japanese', path, words).segment(sentence), expected)
  assert.deepEqual(Segmenter.fromBytes('japanese', readFileSync(path), words).segment(sentence), expected)
  assert.deepEqual((await Segmenter.fromUri('japanese', path, words)).segment(sentence), expected)
})

test('userWordsGate 0 applies a word only where the model agrees', () => {
  const path = modelPath('japanese.model')
  const sentence = '雪だるまを作る'
  const plain = Segmenter.open('japanese', path).segment(sentence)

  assert.deepEqual(Segmenter.open('japanese', path, ['雪だるま'], 0).segment(sentence), plain)
  assert.deepEqual(Segmenter.open('japanese', path, ['雪だるま'], 1).segment(sentence), ['雪だるま', 'を', '作る'])
  // An empty array, undefined, or null is no dictionary at all.
  assert.deepEqual(Segmenter.open('japanese', path, []).segment(sentence), plain)
  assert.deepEqual(Segmenter.open('japanese', path, undefined, undefined).segment(sentence), plain)
  assert.deepEqual(Segmenter.open('japanese', path, null, null).segment(sentence), plain)
})

test('an out-of-range userWordsGate throws with a code', async () => {
  const path = modelPath('japanese.model')
  for (const words of [['雪だるま'], undefined]) {
    assert.throws(
      () => Segmenter.open('japanese', path, words, 1.5),
      (error) => {
        assert.equal(error.code, 'invalid_argument')
        assert.match(error.message, /0\.0\.\.=1\.0/)
        return true
      },
    )
  }
  // fromUri validates the gate before scheduling any work.
  assert.throws(
    () => Segmenter.fromUri('japanese', path, ['雪だるま'], -0.5),
    (error) => {
      assert.equal(error.code, 'invalid_argument')
      return true
    },
  )
})

for (const [language, model, sentence] of SEGMENTATION_CASES) {
  test(`segment matches the CLI (${language})`, () => {
    // Compare the rendered line, not a re-split of it: the CLI joins tokens
    // with a space, so for Korean and English -- where whitespace is its own
    // token -- splitting the output again cannot recover the tokens.
    const [expected] = runCli(['segment', '-l', language, modelPath(model)], `${sentence}\n`)
    const seg = Segmenter.open(language, modelPath(model))
    assert.equal(seg.segment(sentence).join(' '), expected)
  })

  test(`byte offsets reconstruct the input (${language})`, () => {
    const seg = Segmenter.open(language, modelPath(model))
    const tokens = seg.segmentTokens(sentence)
    const raw = Buffer.from(sentence, 'utf8')

    assert.ok(tokens.length > 0)
    let expectedStart = 0
    for (const token of tokens) {
      assert.equal(token.start, expectedStart, 'tokens must tile the input')
      assert.equal(raw.subarray(token.start, token.end).toString('utf8'), token.surface)
      assert.equal(token.pos, undefined, 'napi maps a missing tag to undefined')
      expectedStart = token.end
    }
    assert.equal(expectedStart, raw.length)
    assert.equal(tokens.map((t) => t.surface).join(''), sentence)
  })
}

for (const [language, model, sentence] of POS_CASES) {
  test(`segmentWithPos matches the CLI (${language})`, () => {
    const [expected] = runCli(
      ['segment', '-l', language, '--pos', modelPath(model)],
      `${sentence}\n`,
    )
    const seg = Segmenter.open(language, modelPath(model))
    const actual = seg
      .segmentWithPos(sentence)
      .map((token) => `${token.surface}/${token.pos}`)
      .join(' ')
    assert.equal(actual, expected)
  })
}

test('whitespace is its own token', () => {
  const seg = Segmenter.open('korean', modelPath('korean.model'))
  assert.deepEqual(seg.segment('안녕하세요 반갑습니다'), ['안녕하세요', ' ', '반갑습니다'])
})

test('segmentBatch matches single calls', () => {
  const seg = Segmenter.open('japanese', modelPath('japanese.model'))
  const sentences = ['これはテストです。', '', '東京都から神奈川県へ引っ越した']

  assert.deepEqual(
    seg.segmentBatch(sentences),
    sentences.map((s) => seg.segment(s)),
  )
  assert.deepEqual(seg.segmentBatch(sentences)[1], [])
})

test('segmentWithPosBatch matches single calls', () => {
  const seg = Segmenter.open('japanese', modelPath('japanese_pos.model'))
  const sentences = ['これはテストです。', '東京都から神奈川県へ引っ越した']

  assert.deepEqual(
    seg.segmentWithPosBatch(sentences),
    sentences.map((s) => seg.segmentWithPos(s)),
  )
})

test('the model kind is detected, with no flag', () => {
  assert.equal(Segmenter.open('ja', modelPath('japanese.model')).hasPos, false)
  assert.equal(Segmenter.open('ja', modelPath('japanese_pos.model')).hasPos, true)
})

test('language names and codes are interchangeable', () => {
  const expected = Segmenter.open('japanese', modelPath('japanese.model')).segment('これはテストです。')
  for (const name of ['ja', 'JA', 'japanese', 'Japanese']) {
    assert.deepEqual(Segmenter.open(name, modelPath('japanese.model')).segment('これはテストです。'), expected)
  }
})

test('fromBytes agrees with open', () => {
  const path = modelPath('japanese.model')
  const sentence = 'これはテストです。'
  assert.deepEqual(
    Segmenter.fromBytes('japanese', readFileSync(path)).segment(sentence),
    Segmenter.open('japanese', path).segment(sentence),
  )
})

test('fromUri resolves asynchronously and agrees with open', async () => {
  const path = modelPath('japanese.model')
  const sentence = 'これはテストです。'
  const seg = await Segmenter.fromUri('japanese', path)
  assert.deepEqual(seg.segment(sentence), Segmenter.open('japanese', path).segment(sentence))
})

test('POS on a segmentation-only model throws with a code', () => {
  const seg = Segmenter.open('japanese', modelPath('japanese.model'))
  assert.throws(
    () => seg.segmentWithPos('これはテストです。'),
    (error) => {
      assert.equal(error.code, 'pos_unavailable')
      assert.match(error.message, /two-stage POS model/)
      return true
    },
  )
})

test('an unknown language throws with a code', () => {
  assert.throws(
    () => Segmenter.open('klingon', modelPath('japanese.model')),
    (error) => {
      assert.equal(error.code, 'invalid_argument')
      assert.match(error.message, /klingon/)
      return true
    },
  )
})

test('a missing model file throws with a code', () => {
  assert.throws(
    () => Segmenter.open('japanese', modelPath('does-not-exist.model')),
    (error) => {
      assert.equal(error.code, 'io')
      return true
    },
  )
})

test('a rejected promise carries the same code as a thrown error', async () => {
  // The async path cannot use napi's typed status, so the code is re-thrown
  // in Task::reject; this pins that the two paths agree.
  await assert.rejects(
    Segmenter.fromUri('japanese', modelPath('does-not-exist.model')),
    (error) => {
      assert.equal(error.code, 'io')
      return true
    },
  )
})

test('version and supportedLanguages', () => {
  assert.match(version(), /^\d+\.\d+\.\d+$/)
  assert.deepEqual(supportedLanguages(), ['japanese', 'chinese', 'korean', 'english'])
})
