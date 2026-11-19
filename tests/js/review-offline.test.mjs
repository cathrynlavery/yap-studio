import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import {
  bytesText, newClientId, offlineUrls, pendingComment, postNote, storageText, syncOutbox,
} from '../../src/yap_studio/review/static/offline.mjs';

function memoryOutbox(items) {
  const notes = new Map(items.map((item) => [item.id, structuredClone(item)]));
  return {
    notes,
    all: async () => [...notes.values()].map((item) => structuredClone(item)),
    put: async (item) => { notes.set(item.id, structuredClone(item)); },
    remove: async (id) => { notes.delete(id); },
  };
}

const note = (id, created, extra = {}) => ({id, created, url: `https://review.test/v/1/api/comments`,
  token: 'video-token', body: {kind: 'fix', text: `note ${id}`, start_frame: created}, error: null, ...extra});

test('sync sends oldest first and removes each saved note', async () => {
  const outbox = memoryOutbox([note('b', 2), note('a', 1)]);
  const sent = [];
  const result = await syncOutbox(outbox, async (item) => { sent.push(item.id); return {status: 201}; });
  assert.deepEqual(sent, ['a', 'b']);
  assert.deepEqual(result, {sent: 2, failed: 0, waiting: 0});
  assert.equal(outbox.notes.size, 0);
});

test('a lost connection stops the sync and keeps every unsent note', async () => {
  const outbox = memoryOutbox([note('a', 1), note('b', 2), note('c', 3)]);
  let calls = 0;
  const result = await syncOutbox(outbox, async () => {
    calls += 1;
    if (calls === 2) throw new TypeError('Load failed');
    return {status: 201};
  });
  assert.deepEqual(result, {sent: 1, failed: 0, waiting: 2});
  assert.equal(calls, 2, 'no request after the connection dropped');
  assert.deepEqual([...outbox.notes.keys()], ['b', 'c']);
  for (const status of [500, 503, 408, 429]) {
    const later = memoryOutbox([note('x', 1)]);
    assert.deepEqual(await syncOutbox(later, async () => ({status})), {sent: 0, failed: 0, waiting: 1});
    assert.equal((await later.all())[0].error, null);
  }
});

test('a refused note keeps its reason, is not retried, and does not block the rest', async () => {
  const outbox = memoryOutbox([note('a', 1), note('b', 2)]);
  const sent = [];
  const first = await syncOutbox(outbox, async (item) => {
    sent.push(item.id);
    return item.id === 'a' ? {status: 409, message: 'The video changed.'} : {status: 200};
  });
  assert.deepEqual(first, {sent: 1, failed: 1, waiting: 0});
  assert.equal(outbox.notes.get('a').error, 'The video changed.');
  const second = await syncOutbox(outbox, async (item) => { sent.push(item.id); return {status: 201}; });
  assert.deepEqual(second, {sent: 0, failed: 1, waiting: 0});
  assert.deepEqual(sent, ['a', 'b']);
});

test('a retry after a lost response sends the same client id, so the server saves once', async () => {
  const saved = new Map();
  const server = async (url, options) => {
    const body = JSON.parse(options.body);
    assert.equal(options.headers['X-Yap-Token'], 'video-token');
    const fresh = !saved.has(body.client_id);
    if (fresh) saved.set(body.client_id, body);
    return {ok: true, status: fresh ? 201 : 200, json: async () => ({})};
  };
  const outbox = memoryOutbox([note('0123456789abcdef', 1)]);
  let dropReply = true;
  const send = async (item) => {
    const reply = await postNote(item, server);
    if (dropReply) { dropReply = false; throw new TypeError('response lost'); }
    return reply;
  };
  assert.deepEqual(await syncOutbox(outbox, send), {sent: 0, failed: 0, waiting: 1});
  assert.deepEqual(await syncOutbox(outbox, send), {sent: 1, failed: 0, waiting: 0});
  assert.equal(saved.size, 1);
  assert.deepEqual(saved.get('0123456789abcdef'),
    {kind: 'fix', text: 'note 0123456789abcdef', start_frame: 1, client_id: '0123456789abcdef'});
});

test('postNote reports the server message for a refusal', async () => {
  const reply = await postNote(note('a', 1), async () => ({ok: false, status: 400,
    json: async () => ({error: {message: 'text must contain 1 to 5000 characters'}})}));
  assert.deepEqual(reply, {status: 400, message: 'text must contain 1 to 5000 characters'});
});

test('client ids, sizes, storage text, page addresses and pending rows', () => {
  const id = newClientId();
  assert.match(id, /^[0-9a-f]{32}$/);
  assert.notEqual(newClientId(), id);
  assert.equal(bytesText(21_400_000), '21 MB');
  assert.equal(bytesText(4_250_000), '4.3 MB');
  assert.equal(bytesText(1_500_000_000), '1.5 GB');
  assert.equal(bytesText(31_400), '31 KB');
  assert.equal(storageText({usage: 42e6, quota: 1.2e9}, true),
    'Using 42 MB of about 1.2 GB on this device. Kept until you remove it.');
  assert.match(storageText({usage: 42e6}, false), /may clear it/);
  assert.equal(storageText(undefined, false), '');
  assert.deepEqual(offlineUrls('v/2/', 'tok', {series: null}),
    {data: ['v/2/api/review'], media: 'v/2/media?t=tok'});
  assert.deepEqual(offlineUrls('', 'tok', {series: {current: 3, project: 'p'}}, '2'), {
    data: ['api/review?version=2', 'api/removed?version=3', 'api/words?version=3',
      'api/video-map?version=3', 'api/captions?version=3'],
    media: 'media?version=3&t=tok',
  });
  const row = pendingComment(note('a', 1, {error: 'refused'}));
  assert.equal(row.end_frame, null);
  assert.equal(row.start_frame, 1);
  assert.equal(row.pending, true);
  assert.equal(row.error, 'refused');
});

function worker() {
  const listeners = {};
  const context = vm.createContext({
    self: {addEventListener: (name, handler) => { listeners[name] = handler; }, location: {origin: 'https://review.test'}},
    caches: {open: async () => ({keys: async () => [], match: async () => undefined})},
    Response, Headers, Blob, URL, setTimeout, clearTimeout, Promise, Date,
  });
  const source = readFileSync(new URL('../../src/yap_studio/review/static/sw.js', import.meta.url), 'utf8');
  vm.runInContext(`${source}\nglobalThis.exported = {parseRange, rangedResponse, dataKey};`, context);
  return {...context.exported, listeners};
}

test('the worker answers byte ranges from a saved video the way iOS asks for them', async () => {
  const {rangedResponse} = worker();
  // The worker runs in its own realm; copy results so deepEqual compares plain objects.
  const parseRange = (header, size) => {
    const range = worker().parseRange(header, size);
    return range && {...range};
  };
  assert.deepEqual(parseRange('bytes=0-1', 10), {start: 0, end: 1});
  assert.deepEqual(parseRange('bytes=4-', 10), {start: 4, end: 9});
  assert.deepEqual(parseRange('bytes=-3', 10), {start: 7, end: 9});
  assert.deepEqual(parseRange('bytes=8-100', 10), {start: 8, end: 9});
  for (const bad of ['bytes=10-', 'bytes=5-2', 'bytes=-0', 'bytes=-', 'items=0-1', 'bytes=0-1,3-4']) {
    assert.equal(parseRange(bad, 10), null, bad);
  }
  const saved = () => new Response(new Blob(['0123456789']), {headers: {'Content-Type': 'video/mp4'}});
  const probe = await rangedResponse(saved(), 'bytes=0-1');
  assert.equal(probe.status, 206);
  assert.equal(probe.headers.get('Content-Range'), 'bytes 0-1/10');
  assert.equal(probe.headers.get('Content-Length'), '2');
  assert.equal(probe.headers.get('Accept-Ranges'), 'bytes');
  assert.equal(probe.headers.get('Content-Type'), 'video/mp4');
  assert.equal(await probe.text(), '01');
  const tail = await rangedResponse(saved(), 'bytes=6-');
  assert.equal(await tail.text(), '6789');
  const whole = await rangedResponse(saved(), null);
  assert.equal(whole.status, 200);
  assert.equal(whole.headers.get('Content-Length'), '10');
  const refused = await rangedResponse(saved(), 'bytes=20-');
  assert.equal(refused.status, 416);
  assert.equal(refused.headers.get('Content-Range'), 'bytes */10');
});

test('the worker keys saved data by address and token', () => {
  const {dataKey} = worker();
  const key = (url, token) => dataKey({url, headers: new Headers(token ? {'X-Yap-Token': token} : {})});
  assert.equal(key('https://review.test/v/1/api/review?version=2', 'abc'),
    'https://review.test/v/1/api/review?t=abc&version=2');
  assert.equal(key('https://review.test/v/1/?t=abc#list=x'), 'https://review.test/v/1/?t=abc');
  assert.equal(key('https://review.test/v/1/?t=abc', 'abc'), key('https://review.test/v/1/?t=abc'));
  assert.notEqual(key('https://review.test/api/review', 'one'), key('https://review.test/api/review', 'two'));
});
