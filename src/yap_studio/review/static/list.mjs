// The review list: download every video for offline review and sync notes saved offline.
import {keepSyncing, mountOfflineTools, offlineUrls, openOutbox, refreshWorker} from './offline.mjs';

const listToken = new URLSearchParams(location.search).get('t');

async function getJson(url, token) {
  const response = await fetch(url, {headers: {'X-Yap-Token': token}});
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}

function rows() {
  return [...document.querySelectorAll('.review-list li')].map((row) => {
    let label = row.querySelector('.offline-state');
    if (!label) {
      label = document.createElement('span');
      label.className = 'offline-state muted';
      row.querySelector('.review-details')?.append(label);
    }
    return {page: row.querySelector('a').href.split('#')[0],
      show: (text) => { label.textContent = text; }};
  });
}

async function entries() {
  const listing = await getJson('api/list', listToken);
  return Promise.all(listing.videos.map(async (video) => {
    const page = video.url.split('#')[0];
    const token = new URL(page, location.href).searchParams.get('t');
    const prefix = `v/${video.number}/`;
    const review = await getJson(`${prefix}api/review`, token);
    return {page, token, ...offlineUrls(prefix, token, review),
      extra: [`${prefix}api/thumbnail?t=${encodeURIComponent(token)}`]};
  }));
}

async function start() {
  if (!listToken) return;
  await refreshWorker().catch(() => false);
  mountOfflineTools(document.getElementById('offline-tools'), {pages: rows, entries}, {all: true, shell: [
    [`${location.pathname}${location.search}`, listToken], ['api/list', listToken]]});
  const outbox = await openOutbox().catch(() => null);
  if (!outbox) return;
  const pending = document.getElementById('offline-pending');
  keepSyncing(outbox, async () => {
    const waiting = (await outbox.all()).length;
    pending.hidden = waiting === 0;
    pending.textContent = `${waiting} ${waiting === 1 ? 'note' : 'notes'} saved on this phone, waiting to sync.`;
  });
}

void start();
