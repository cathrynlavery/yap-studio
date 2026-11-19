// Offline review: save videos on this device and hold new notes until the server is reachable.
import {versionUrl} from './timeline.mjs';

const MEDIA_CACHE = 'yap-media';
const DATA_CACHE = 'yap-data';
const OUTBOX_DB = 'yap-review-outbox';
const SYNC_INTERVAL_MS = 20000;
const CLAIM_WAIT_MS = 5000;

/** A random id the server uses to save a retried note only once. */
export function newClientId(source = globalThis.crypto) {
  const bytes = new Uint8Array(16);
  source.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export function bytesText(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '';
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes < 1e6) return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
  const megabytes = bytes / 1e6;
  return `${megabytes < 10 ? megabytes.toFixed(1) : Math.round(megabytes)} MB`;
}

/** Describe what this site stores and whether the browser promised to keep it. */
export function storageText(estimate, persisted) {
  if (!estimate || !Number.isFinite(estimate.usage)) return '';
  const room = Number.isFinite(estimate.quota) ? ` of about ${bytesText(estimate.quota)}` : '';
  const kept = persisted ? 'Kept until you remove it.'
    : 'The phone may clear it if space runs low; add this page to your Home Screen to keep it.';
  return `Using ${bytesText(estimate.usage)}${room} on this device. ${kept}`;
}

/** The data requests a video page makes on load, and its video address. */
export function offlineUrls(prefix, token, review, pageVersion = null) {
  const current = review.series?.current ?? null;
  const data = [versionUrl(`${prefix}api/review`, pageVersion)];
  if (review.series?.project) {
    for (const name of ['removed', 'words', 'video-map', 'captions']) {
      data.push(versionUrl(`${prefix}api/${name}`, current));
    }
  }
  return {data, media: versionUrl(`${prefix}media`, current, token)};
}

function requestResult(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/** Notes waiting for the server, kept in IndexedDB so they survive a reload or a closed app. */
export async function openOutbox(factory = globalThis.indexedDB) {
  if (!factory) return null;
  const opening = factory.open(OUTBOX_DB, 1);
  opening.onupgradeneeded = () => opening.result.createObjectStore('notes', {keyPath: 'id'});
  const db = await requestResult(opening);
  const run = (mode, work) => {
    const transaction = db.transaction('notes', mode);
    const result = requestResult(work(transaction.objectStore('notes')));
    return new Promise((resolve, reject) => {
      transaction.oncomplete = () => resolve(result);
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  };
  return {
    all: () => run('readonly', (notes) => notes.getAll()),
    put: (item) => run('readwrite', (notes) => notes.put(item)),
    remove: (id) => run('readwrite', (notes) => notes.delete(id)),
  };
}

/** POST one waiting note with its client id; a network failure throws. */
export async function postNote(item, fetcher = globalThis.fetch) {
  const response = await fetcher(item.url, {
    method: 'POST',
    headers: {'X-Yap-Token': item.token, 'Content-Type': 'application/json'},
    body: JSON.stringify({...item.body, client_id: item.id}),
  });
  let message = '';
  if (!response.ok) {
    try { message = (await response.json()).error?.message ?? ''; } catch { /* Keep the status. */ }
  }
  return {status: response.status, message};
}

/**
 * Send waiting notes oldest first. Stop at the first network failure or server error and
 * keep the rest for the next try; a refused note keeps its reason and waits for the reviewer.
 */
export async function syncOutbox(outbox, send = postNote) {
  const result = {sent: 0, failed: 0, waiting: 0};
  const items = (await outbox.all()).sort((a, b) => a.created - b.created);
  for (const [index, item] of items.entries()) {
    if (item.error) {
      result.failed += 1;
      continue;
    }
    let outcome = null;
    try {
      outcome = await send(item);
    } catch { /* Unreachable: counted as waiting below. */ }
    if (outcome && outcome.status >= 200 && outcome.status < 300) {
      await outbox.remove(item.id);
      result.sent += 1;
    } else if (outcome && outcome.status >= 400 && outcome.status < 500
      && ![408, 429].includes(outcome.status)) {
      await outbox.put({...item, error: outcome.message || `The server refused this note (${outcome.status}).`});
      result.failed += 1;
    } else {
      const rest = items.slice(index);
      result.waiting += rest.filter((note) => !note.error).length;
      result.failed += rest.filter((note) => note.error).length;
      break;
    }
  }
  return result;
}

/** Retry on reconnect, on focus, when the page shows and on a timer; one sync at a time. */
export function keepSyncing(outbox, onSynced, {win = globalThis.window, doc = globalThis.document} = {}) {
  let running = null;
  const flush = () => {
    running ??= syncOutbox(outbox).then(onSynced, () => {}).finally(() => { running = null; });
    return running;
  };
  win.addEventListener('online', flush);
  win.addEventListener('focus', flush);
  doc.addEventListener('visibilitychange', () => { if (doc.visibilityState === 'visible') flush(); });
  setInterval(flush, SYNC_INTERVAL_MS);
  flush();
  return flush;
}

/** The comment fields of a waiting note, shaped like a saved comment for the list. */
export function pendingComment(item) {
  return {start_frame: null, end_frame: null, start_s: 0, end_s: null, target: null,
    request: null, pin: null, ...item.body, id: item.id, pending: true, error: item.error ?? null};
}

export function offlineSupported(nav = globalThis.navigator) {
  return Boolean(globalThis.isSecureContext && nav?.serviceWorker && globalThis.caches);
}

/** Start the worker when this origin already has one, so a new build replaces the old. */
export async function refreshWorker(nav = globalThis.navigator) {
  if (!offlineSupported(nav)) return false;
  const existing = await nav.serviceWorker.getRegistration('/');
  if (existing) await nav.serviceWorker.register('/sw.js', {scope: '/'});
  return Boolean(existing);
}

async function controlledWorker(nav = globalThis.navigator) {
  await nav.serviceWorker.register('/sw.js', {scope: '/'});
  await nav.serviceWorker.ready;
  if (!nav.serviceWorker.controller) {
    await new Promise((resolve) => {
      nav.serviceWorker.addEventListener('controllerchange', resolve, {once: true});
      setTimeout(resolve, CLAIM_WAIT_MS);
    });
  }
  if (!nav.serviceWorker.controller) throw new Error('Reload this page, then press Download again.');
  return nav.serviceWorker.controller;
}

async function savedMedia(url) {
  const saved = await (await caches.open(MEDIA_CACHE)).match(new URL(url, location.href).href);
  return saved ? Number(saved.headers.get('Content-Length')) || 0 : null;
}

/** Save one page and its data through the worker; a copy from the worker's cache is not new. */
async function saveData(url, token) {
  const response = await fetch(url, {headers: token ? {'X-Yap-Token': token} : {}, cache: 'no-store'});
  if (!response.ok || response.headers.get('X-Yap-Offline')) {
    throw new Error(response.ok ? 'Connect to the review server, then press Download again.'
      : `Could not save this review (${response.status}).`);
  }
  await response.arrayBuffer();
}

/** Stream the whole video into the cache, counting bytes; a cut-off download is not kept. */
async function saveMedia(url, onBytes) {
  const address = new URL(url, location.href).href;
  const response = await fetch(address, {cache: 'no-store'});
  if (response.status !== 200 || !response.body) {
    throw new Error(`Could not download the video (${response.status}).`);
  }
  const total = Number(response.headers.get('Content-Length'));
  if (!(total > 0)) throw new Error('The server did not say how big the video is.');
  let loaded = 0;
  const counted = response.body.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      loaded += chunk.byteLength;
      onBytes(loaded, total);
      controller.enqueue(chunk);
    },
  }));
  const cache = await caches.open(MEDIA_CACHE);
  await cache.put(address, new Response(counted, {headers: {
    'Content-Type': response.headers.get('Content-Type') || 'video/mp4',
    'Content-Length': String(total),
  }}));
  if (loaded !== total) {
    await cache.delete(address);
    throw new Error('The download was cut off. Press Download again.');
  }
  return loaded;
}

async function mediaSize(url) {
  try {
    const response = await fetch(url, {method: 'HEAD', cache: 'no-store'});
    return response.ok ? Number(response.headers.get('Content-Length')) || 0 : 0;
  } catch {
    return 0;
  }
}

/** The folder and token a page or its video belongs to: '/v/2/|token'. */
function owner(address) {
  const url = new URL(address, location.href);
  return `${url.pathname.replace(/media$/, '')}|${url.searchParams.get('t')}`;
}

/** Sizes of the videos saved on this device, by owner; reads only the local cache. */
async function savedSizes() {
  const cache = await caches.open(MEDIA_CACHE);
  const sizes = new Map();
  for (const request of await cache.keys()) {
    const saved = await cache.match(request);
    sizes.set(owner(request.url), Number(saved?.headers.get('Content-Length')) || 0);
  }
  return sizes;
}

function button(text, className = '') {
  const node = document.createElement('button');
  node.type = 'button';
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

/**
 * Add the Download for offline controls. `pages()` lists the shown videos without asking the
 * server: [{page, show(text)}]. `entries()` runs only on Download and resolves to
 * [{page, token, data: [urls], media, extra: [urls]}]; `shell` lists [url, token] pairs saved
 * first, such as the list page itself.
 */
export function mountOfflineTools(container, {pages, entries}, {all = false, shell = []} = {}) {
  if (!container || !offlineSupported()) return null;
  const label = all ? 'Download all for offline' : 'Download for offline';
  const save = button(label, 'offline-save');
  const remove = button('Remove downloads', 'offline-remove');
  remove.hidden = true;
  const status = document.createElement('p');
  status.className = 'offline-status muted';
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  const storage = document.createElement('p');
  storage.className = 'offline-storage muted';
  container.append(save, remove, status, storage);
  container.hidden = false;
  let busy = false;

  const refresh = async () => {
    const shown = pages();
    let sizes = new Map();
    try { sizes = await savedSizes(); } catch { /* No cache: nothing is saved. */ }
    let total = 0;
    let ready = 0;
    for (const item of shown) {
      const size = sizes.get(owner(item.page));
      if (size !== undefined) {
        ready += 1;
        total += size;
      }
      item.show(size === undefined ? '' : `Ready offline · ${bytesText(size)}`);
    }
    remove.hidden = ready === 0;
    if (!busy) {
      status.textContent = !shown.length || ready < shown.length ? ''
        : `${shown.length === 1 ? 'Ready offline' : `All ${shown.length} videos ready offline`}`
          + ` · ${bytesText(total)}.`;
    }
    try {
      const [estimate, persisted] = await Promise.all([
        navigator.storage?.estimate?.(), navigator.storage?.persisted?.()]);
      storage.textContent = ready ? storageText(estimate, persisted) : '';
    } catch { storage.textContent = ''; }
  };

  save.addEventListener('click', async () => {
    if (busy) return;
    busy = true;
    save.disabled = true;
    remove.disabled = true;
    try {
      const worker = await controlledWorker();
      // Ask once; a Home Screen app is usually granted, Safari tabs usually are not.
      await Promise.resolve(navigator.storage?.persist?.()).catch(() => false);
      status.textContent = 'Getting the review ready…';
      for (const [url, token] of shell) await saveData(url, token);
      const list = await entries();
      const shows = new Map(pages().map((item) => [owner(item.page), item.show]));
      const sizes = await savedSizes();
      for (const entry of list) entry.size = sizes.get(owner(entry.media)) ?? await mediaSize(entry.media);
      const total = list.reduce((sum, entry) => sum + entry.size, 0);
      let done = 0;
      for (const [index, entry] of list.entries()) {
        const name = list.length > 1 ? `Video ${index + 1} of ${list.length}: ` : '';
        const show = shows.get(owner(entry.page)) ?? (() => {});
        for (const url of [entry.page, ...entry.data, ...(entry.extra ?? [])]) await saveData(url, entry.token);
        if (await savedMedia(entry.media) === null) {
          await saveMedia(entry.media, (loaded, size) => {
            const percent = total ? Math.min(100, Math.floor(((done + loaded) / total) * 100)) : 0;
            status.textContent = `${name}${bytesText(loaded)} of ${bytesText(size)}`
              + (list.length > 1 ? ` · ${percent}% of ${bytesText(total)}` : '');
            show(`Downloading ${Math.floor((loaded / size) * 100)}%`);
          });
          worker.postMessage({type: 'media-changed'});
        }
        done += entry.size;
      }
      busy = false;
      await refresh();
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : String(error);
    } finally {
      busy = false;
      save.disabled = false;
      remove.disabled = false;
    }
  });

  remove.addEventListener('click', async () => {
    if (busy) return;
    await Promise.all([caches.delete(MEDIA_CACHE), caches.delete(DATA_CACHE)]);
    navigator.serviceWorker.controller?.postMessage({type: 'media-changed'});
    await refresh();
    status.textContent = 'Downloads removed.';
  });

  void refresh();
  return refresh;
}
