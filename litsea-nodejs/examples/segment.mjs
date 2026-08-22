/**
 * Segment a sentence, with and without POS tags.
 *
 * Usage:
 *   node examples/segment.mjs ../models/japanese.model "これはテストです。"
 *   node examples/segment.mjs ../models/japanese_pos.model "これはテストです。"
 */

import { Segmenter } from '../index.js'

const [, , modelPath, text] = process.argv

if (!modelPath || !text) {
  console.error('usage: node examples/segment.mjs <model> <text>')
  process.exit(2)
}

// The model file identifies its own kind, so nothing here declares whether
// this is a POS model - `hasPos` reports what was loaded.
const segmenter = Segmenter.open('japanese', modelPath)
console.log(`model: ${modelPath} (hasPos=${segmenter.hasPos})`)
console.log('tokens:', segmenter.segment(text).join(' '))

if (segmenter.hasPos) {
  console.log('tagged:')
  for (const token of segmenter.segmentWithPos(text)) {
    console.log(`  ${token.surface}\t${token.pos}\t[${token.start}:${token.end}]`)
  }
}
