/**
 * Optional model caching for the browser.
 *
 * Models are 84 KB to 8 MB and cross the network once per visitor, which is
 * the one cost this binding has that the native ones do not. This helper
 * keeps a fetched model in Cache Storage so reloads and repeat visits skip
 * the download. It is plain JavaScript and lives outside the wasm module, so
 * a page that does not want it pays nothing.
 *
 * Cache Storage requires a secure context (https, or localhost).
 */

const CACHE_NAME = 'litsea-models-v1'

/**
 * Fetches a model, serving it from Cache Storage when possible.
 *
 * Falls back to a plain fetch when Cache Storage is unavailable (an
 * insecure context, or a browser with it disabled), so callers do not need
 * to branch.
 *
 * @param {string} url - The model URL.
 * @param {RequestInit} [init] - Passed through to `fetch` on a cache miss.
 * @returns {Promise<Uint8Array>} The model bytes, ready for `Segmenter.fromBytes`.
 */
export async function fetchModel(url, init) {
  if (typeof caches === 'undefined') {
    const response = await fetch(url, init)
    return toBytes(response, url)
  }

  const cache = await caches.open(CACHE_NAME)
  const cached = await cache.match(url)
  if (cached) {
    return toBytes(cached, url)
  }

  const response = await fetch(url, init)
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: HTTP ${response.status}`)
  }
  // Store a clone: a Response body can only be read once.
  await cache.put(url, response.clone())
  return toBytes(response, url)
}

/**
 * Removes every cached model.
 *
 * @returns {Promise<boolean>} Whether a cache was deleted.
 */
export async function clearModelCache() {
  if (typeof caches === 'undefined') {
    return false
  }
  return caches.delete(CACHE_NAME)
}

/**
 * Reads a response body as bytes.
 *
 * @param {Response} response - The response to read.
 * @param {string} url - The URL, for the error message.
 * @returns {Promise<Uint8Array>} The body bytes.
 */
async function toBytes(response, url) {
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: HTTP ${response.status}`)
  }
  return new Uint8Array(await response.arrayBuffer())
}
