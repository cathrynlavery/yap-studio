/* Offline copy of a review: the page shell, its data and downloaded videos. */
const VERSION = "__VERSION__";
const STATIC_FILES = ["__STATIC__"];
const SHELL_CACHE = `yap-shell-${VERSION}`;
const DATA_CACHE = 'yap-data';
const MEDIA_CACHE = 'yap-media';
// A phone with no signal can hold a request open for a minute; past this, use the saved copy.
const NETWORK_WAIT_MS = 6000;
const OFFLINE_HOLD_MS = 30000;
const STATIC_PATH = /^\/(?:v\/[1-9][0-9]*\/)?static\/([A-Za-z0-9_-]+\.(?:css|mjs|svg))$/;
const MEDIA_PATH = /^\/(?:v\/[1-9][0-9]*\/)?media$/;
const DATA_PATH = /^\/(?:v\/[1-9][0-9]*\/)?api\/(?:review|list|removed|words|captions|video-map|thumbnail)$/;
const PAGE_PATH = /^\/(?:v\/[1-9][0-9]*\/)?$/;

let offlineUntil = 0;
let mediaKeys = null;

function loadMediaKeys() {
  return caches.open(MEDIA_CACHE).then((cache) => cache.keys())
    .then((keys) => { mediaKeys = new Set(keys.map((request) => request.url)); })
    .catch(() => { mediaKeys = null; });
}

/** Parse one HTTP byte range against a size: {start, end}, or null when unsatisfiable. */
function parseRange(header, size) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(String(header).trim());
  if (!match || (!match[1] && !match[2]) || size <= 0) return null;
  let start;
  let end;
  if (match[1]) {
    start = Number(match[1]);
    end = match[2] ? Math.min(Number(match[2]), size - 1) : size - 1;
  } else {
    const count = Number(match[2]);
    if (count === 0) return null;
    start = Math.max(0, size - count);
    end = size - 1;
  }
  return start < size && start <= end ? {start, end} : null;
}

/** Answer a media request from a saved whole file, as 206 for a range (iOS asks for ranges). */
async function rangedResponse(saved, rangeHeader) {
  const blob = await saved.blob();
  const type = saved.headers.get('Content-Type') || 'video/mp4';
  const size = blob.size;
  if (!rangeHeader) {
    return new Response(blob, {status: 200, headers: {
      'Content-Type': type, 'Content-Length': String(size), 'Accept-Ranges': 'bytes'}});
  }
  const range = parseRange(rangeHeader, size);
  if (range === null) {
    return new Response('', {status: 416, headers: {'Content-Range': `bytes */${size}`}});
  }
  return new Response(blob.slice(range.start, range.end + 1), {
    status: 206,
    statusText: 'Partial Content',
    headers: {
      'Content-Type': type,
      'Content-Length': String(range.end - range.start + 1),
      'Content-Range': `bytes ${range.start}-${range.end}/${size}`,
      'Accept-Ranges': 'bytes',
    },
  });
}

/**
 * Key saved data by its address and token, so another review on this port never sees it.
 * A page is the same file for every version, so its key keeps only the token.
 */
function dataKey(request) {
  const url = new URL(request.url);
  url.hash = '';
  const token = request.headers.get('X-Yap-Token') || url.searchParams.get('t');
  if (PAGE_PATH.test(url.pathname)) url.search = '';
  if (token && !url.searchParams.has('t')) url.searchParams.set('t', token);
  url.searchParams.sort();
  return url.href;
}

function fromCache(response) {
  const headers = new Headers(response.headers);
  headers.set('X-Yap-Offline', '1');
  return new Response(response.body, {status: response.status, statusText: response.statusText, headers});
}

function networkFirst(event, cacheName, key) {
  const network = fetch(event.request).then(async (response) => {
    offlineUntil = 0;
    if (!response.ok) return response;
    // A full cache must not cost the page a good answer; Download reads X-Yap-Saved instead.
    const saved = await caches.open(cacheName).then((cache) => cache.put(key, response.clone()))
      .then(() => true, () => false);
    const headers = new Headers(response.headers);
    if (saved) headers.set('X-Yap-Saved', '1');
    return new Response(response.body, {status: response.status, statusText: response.statusText, headers});
  });
  event.waitUntil(network.then(() => {}, () => {}));
  return (async () => {
    const saved = await caches.open(cacheName).then((cache) => cache.match(key));
    if (!saved) return network;
    if (Date.now() < offlineUntil) return fromCache(saved);
    let timer;
    const wait = new Promise((resolve) => { timer = setTimeout(resolve, NETWORK_WAIT_MS, null); });
    try {
      const response = await Promise.race([network, wait]);
      if (response) return response;
    } catch { /* Unreachable: answer from the saved copy below. */ } finally {
      clearTimeout(timer);
    }
    offlineUntil = Date.now() + OFFLINE_HOLD_MS;
    return fromCache(saved);
  })();
}

self.addEventListener('install', (event) => {
  // One file at a time: WebKit drops the install when addAll saves this many files over HTTPS.
  event.waitUntil(caches.open(SHELL_CACHE).then(async (cache) => {
    for (const path of STATIC_FILES) await cache.add(new Request(path, {cache: 'no-store'}));
  }).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(caches.keys()
    .then((names) => Promise.all(names.filter((name) => name.startsWith('yap-shell-')
      && name !== SHELL_CACHE).map((name) => caches.delete(name))))
    .then(() => self.clients.claim()));
});

self.addEventListener('message', (event) => {
  if (event.data?.type === 'media-changed') event.waitUntil(loadMediaKeys());
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  const path = url.pathname;
  const staticName = STATIC_PATH.exec(path);
  if (staticName) {
    event.respondWith(networkFirst(event, SHELL_CACHE, `${self.location.origin}/static/${staticName[1]}`));
  } else if (MEDIA_PATH.test(path)) {
    const key = new URL(request.url);
    key.hash = '';
    // Leave a video this phone has not saved to the browser, exactly as without the worker.
    if (mediaKeys !== null && !mediaKeys.has(key.href)) return;
    event.respondWith((async () => {
      if (mediaKeys === null) await loadMediaKeys();
      const saved = await caches.open(MEDIA_CACHE).then((cache) => cache.match(key.href, {ignoreVary: true}));
      return saved ? rangedResponse(saved, request.headers.get('Range')) : fetch(request);
    })());
  } else if (DATA_PATH.test(path) || PAGE_PATH.test(path)) {
    // Pages are saved both when opened and when Download fetches them.
    if (request.mode === 'navigate') event.waitUntil(loadMediaKeys());
    event.respondWith(networkFirst(event, DATA_CACHE, dataKey(request)));
  }
});

loadMediaKeys();
