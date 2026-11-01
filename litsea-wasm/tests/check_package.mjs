// Checks the file list of the litsea-wasm npm package (#286).
//
// Reads the JSON that `npm pack --dry-run --json` prints for litsea-wasm/pkg/
// on stdin and exits non-zero unless the package ships the files below. Run
// by `make check-litsea-wasm-package` after `make build-litsea-wasm`.

import { readFileSync } from 'node:fs'

/** Files the published package must contain, as paths inside the tarball. */
const REQUIRED = [
  'litsea_wasm_bg.wasm',
  'litsea_wasm.js',
  'litsea_wasm.d.ts',
  'cache.js',
  'cache.d.ts',
]

const [pack] = JSON.parse(readFileSync(0, 'utf8'))
const shipped = pack.files.map((file) => file.path)
const missing = REQUIRED.filter((path) => !shipped.includes(path))

if (missing.length > 0) {
  console.error(`${pack.name}@${pack.version} is missing: ${missing.join(', ')}`)
  console.error(`it ships: ${shipped.join(', ')}`)
  process.exit(1)
}
console.log(`${pack.name}@${pack.version} ships: ${shipped.join(', ')}`)
