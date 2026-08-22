/** Optional model caching for the browser. See cache.js. */

/**
 * Fetches a model, serving it from Cache Storage when possible.
 *
 * @param url The model URL.
 * @param init Passed through to `fetch` on a cache miss.
 * @returns The model bytes, ready for `Segmenter.fromBytes`.
 */
export declare function fetchModel(url: string, init?: RequestInit): Promise<Uint8Array>

/**
 * Removes every cached model.
 *
 * @returns Whether a cache was deleted.
 */
export declare function clearModelCache(): Promise<boolean>
