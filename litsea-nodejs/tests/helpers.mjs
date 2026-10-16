/** Shared helpers for the Node.js binding tests. */

import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

export const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..')
export const MODELS_DIR = join(REPO_ROOT, 'models')

/**
 * Returns the path to a bundled model.
 *
 * @param {string} name - The model file name.
 * @returns {string} The absolute path.
 */
export function modelPath(name) {
  return join(MODELS_DIR, name)
}

/** Whether this process has already built the CLI. */
let cliBuilt = false

/**
 * Builds the `litsea` CLI once per process and returns the path to the
 * binary.
 *
 * The parity tests compare against the CLI rather than hardcoded output, so
 * the reference implementation decides what is correct. The build always
 * runs, even when a binary exists: one left in `target/` (a CI cache
 * restored by its `Cargo.lock` key, or an older checkout) can predate the
 * source under test, and Cargo makes the build a no-op when it is current.
 *
 * @returns {string} The absolute path to the CLI binary.
 */
export function litseaCli() {
  const binary = join(REPO_ROOT, 'target', 'debug', 'litsea')
  if (!cliBuilt) {
    execFileSync('cargo', ['build', '--quiet', '-p', 'litsea-cli'], { cwd: REPO_ROOT })
    if (!existsSync(binary)) {
      throw new Error(`cargo build did not produce ${binary}`)
    }
    cliBuilt = true
  }
  return binary
}

/**
 * Runs the CLI over `input` and returns its output lines.
 *
 * @param {string[]} args - Arguments after the binary name.
 * @param {string} input - Text to feed on standard input.
 * @returns {string[]} The output lines.
 */
export function runCli(args, input) {
  const stdout = execFileSync(litseaCli(), args, { input, encoding: 'utf8' })
  return stdout.split('\n').filter((line) => line.length > 0)
}
