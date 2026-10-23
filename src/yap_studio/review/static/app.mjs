import {renderTextHook} from './text-hook.mjs';
import {
  decisionBody, metaText, pendingCount, sentenceText, sourceRows, splitProposals,
  stateText, tasteActions,
} from './taste.mjs';
import {
  assignLanes, fpsLabel, frameAt, frameCount, frameMid, frameStart, frameTimeText,
  normalizeSelection, normalizeTimeSelection, parseFps, secondsText, sortComments, spanText,
  TARGET_NAMES, targetName, mapBlockPosition,
  defaultCompareVersion, mapStatusMessage, threadStateLabel, versionUrl,
  replyEvidence, replyHeading, replyMomentText, keepCheckText, changeText,
  draftNotesText, roundStatusText, sendStateLabel, reviewPollInterval, projectTitle, homeUrl, phoneLinkLabel, sendNextText, storyCutStep,
  questionsForComment, questionAnswerBody,
  spellingAvailable, spellingRequest, requestText, editedRequest,
  wordsInRange, cutWordsText, cutState, cutBody, rangeLengthText,
  removedKindName, visibleRemovedRanges, rangeRequested,
  versionBadges, reportStatusLabel, offlineState,
  findingKindLabel, findingSourceLabel, findingEvidenceText, findingNoteBody, findingAdded,
  pictureRect, pinAt, pinPoint, pinText, pinBody, pinInSpan, pinShown, captionAt,
} from './timeline.mjs';

const $ = (id) => document.getElementById(id);
const el = {
  fixCaption: $('fix-caption'), captionFix: $('caption-fix'), captionWords: $('caption-words'),
  captionTo: $('caption-to'), captionEvery: $('caption-every'),
  captionEveryLabel: $('caption-every-label'), captionError: $('caption-error'),
  captionSave: $('caption-save'), captionCancel: $('caption-cancel'),
  textHook: $('text-hook-panel'),
  name: $('media-name'), facts: $('media-facts'), status: $('status'),
  video: $('video'), codecError: $('codec-error'), scrubber: $('scrubber'),
  playerMessage: $('player-message'), playerRetry: $('player-retry'),
  offlinePlayer: $('offline-player'), offlinePlayerMessage: $('offline-player-message'),
  offlinePlayerRetry: $('offline-player-retry'),
  videoMap: $('video-map'), videoMapToggle: $('video-map-toggle'),
  videoMapLanes: $('video-map-lanes'), videoMapPlayhead: $('video-map-playhead'),
  pinLayer: $('pin-layer'), pinMarker: $('pin-marker'),
  composePinText: $('compose-pin-text'), composePinRemove: $('compose-pin-remove'),
  markers: $('markers'), ranges: $('ranges'), playhead: $('playhead'), readout: $('time-readout'),
  selection: $('selection'), selectionStart: $('selection-start'), selectionEnd: $('selection-end'),
  selectionReadout: $('selection-readout'), selectionText: $('selection-text'),
  selectionClear: $('selection-clear'), composeClearRange: $('compose-clear-range'),
  markStart: $('mark-start'), markEnd: $('mark-end'), rangePanel: $('range-panel'),
  rangeLength: $('range-length'), rangeWords: $('range-words'), rangePlay: $('range-play'),
  rangeCut: $('range-cut'), rangeError: $('range-error'), rangeNote: $('range-note'),
  rangeChange: $('range-change'), rangeKeep: $('range-keep'),
  playPause: $('play-pause'), previousFrame: $('previous-frame'), nextFrame: $('next-frame'),
  shortcutHint: $('shortcut-hint'),
  newComment: $('new-comment'), copy: $('copy-feedback'), copyHint: $('copy-notes-hint'),
  previousBanner: $('previous-banner'),
  previousMessage: $('previous-message'), previousView: $('previous-view'),
  previousCopy: $('previous-copy'), previousText: $('previous-text'),
  count: $('comment-count'), empty: $('empty-state'), list: $('comment-list'),
  compose: $('compose'), composeTime: $('compose-time'), composeText: $('compose-text'),
  composeError: $('compose-error'), composeSubmit: $('compose-submit'),
  composeCancel: $('compose-cancel'), composeFix: $('compose-fix'),
  composeKeep: $('compose-keep'),
  composeSpelling: $('compose-spelling'), spellingFrom: $('compose-spelling-from'),
  spellingTo: $('compose-spelling-to'),
  toast: $('toast'), copyFallback: $('copy-fallback'),
  copyFallbackText: $('copy-fallback-text'), copyFallbackClose: $('copy-fallback-close'),
  versionBar: $('version-bar'), versionButtons: $('version-buttons'),
  versionActions: $('version-actions'), rollback: $('rollback-version'), approve: $('approve-version'),
  makeFinal: $('make-final'), approveHint: $('approve-hint'),
  downloadVideo: $('download-video'), downloadReport: $('download-report'), offline: $('offline-banner'),
  compareToggle: $('compare-toggle'), players: $('players'), comparePanel: $('compare-panel'),
  compareVersion: $('compare-version'), compareVideo: $('compare-video'),
  matchMoment: $('match-moment'), matchReverse: $('match-reverse'),
  mapStatus: $('map-status'), mapCandidates: $('map-candidates'),
  earlierThreads: $('earlier-threads'), threadList: $('thread-list'),
  sendBar: $('send-bar'), draftCount: $('draft-count'), sendNotes: $('send-notes'),
  roundList: $('round-list'), homeLink: $('home-link'), sendNext: $('send-next'),
  versionNote: $('version-note'),
  removed: $('removed-section'), removedCount: $('removed-count'),
  removedList: $('removed-list'), removedEmpty: $('removed-empty'), showPauses: $('show-pauses'),
  taste: $('taste-section'), tasteCount: $('taste-count'), tasteError: $('taste-error'),
  tasteList: $('taste-list'),
  tasteEmpty: $('taste-empty'), tasteDecided: $('taste-decided'),
  tasteDecidedCount: $('taste-decided-count'), tasteDecidedList: $('taste-decided-list'),
  findings: $('findings-section'), findingsCount: $('findings-count'),
  findingsList: $('findings-list'), findingsStatus: $('findings-status'),
  findingMarkers: $('finding-markers'),
};
const composeTargetButtons = Object.keys(TARGET_NAMES).map((target) =>
  ({target, button: $(`compose-target-${target}`)}));

const token = new URLSearchParams(location.search).get('t');
let review = null;
let fps = null;
let displayedTime = 0;
let selectedId = null;
let composeKind = 'fix';
let composeTarget = null;
let composeBusy = false;
let editState = null;
let composePin = null;
let pinDrag = null;
let ignorePinClick = false;
const pinPosition = {left: null, top: null};
let manualSeek = false;
let toastTimer = null;
let dragPointer = null;
let dragFraction = null;
let dragFrame = null;
let selection = null;
let rangePlayback = null;
let wordsState = {available: false, version: null, words: []};
let captionsState = {available: false, version: null, phrases: []};
let captionEdit = null;
let rangeWordsCache = null;
let cutBusy = false;
let cutError = '';
let refreshPending = false;
let refreshTimer = null;
let roundBusy = false;
let releaseBusy = false;
let offline = false;
const MEDIA_METADATA_DEADLINE_MS = 10000;
const MEDIA_FIRST_FRAME_DEADLINE_MS = 15000;
const MEDIA_REACHABILITY_DEADLINE_MS = 3000;
let playerGeneration = 0;
let metadataTimer = null;
let firstFrameTimer = null;
let playerResumeTime = null;
let playerNetworkRetried = false;
let playerMessage = '';
let playerRecoveryPending = false;
let playerMetadataLoaded = false;
let answerBusy = false;
let activeWrites = 0;
const addedFindings = new Map();
let findingControls = [];
let layerChecksSummary = null;
let layerStaleSummary = null;
const questionDrafts = new Map();
const tasteDrafts = new Map();
const tasteErrors = new Map();
const tasteSaving = new Set();
let tasteDecidedOpen = false;
let removed = {available: false, ranges: []};
const restorePending = new Set();
let refreshStatusError = false;
let loadVersion = 0;
let selectedVersion = new URLSearchParams(location.search).get('version');
let switchingVersion = false;
let compareVersion = null;
let compareOpen = false;
let matchTimer = null;
let mapGeneration = 0;
let suppressMatchUntil = 0;
let openingReplyEvidence = false;
let replyEvidenceTime = null;
const mediaChangedMessage = 'This video changed after the page opened. Ask Claude to open the review again to see the new one.';

function showStatus(message) {
  refreshStatusError = false;
  el.status.textContent = review?.media_changed ? mediaChangedMessage : message;
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

async function api(path, options = {}) {
  const writing = options.method && options.method !== 'GET';
  if (writing) {
    activeWrites += 1;
    updateFindingControls();
  }
  try {
    return await request(path, options);
  } finally {
    if (writing) {
      activeWrites -= 1;
      updateFindingControls();
    }
  }
}

async function request(path, options) {
  const headers = {'X-Yap-Token': token};
  if (options.body !== undefined) headers['Content-Type'] = 'application/json';
  let response;
  try {
    response = await fetch(path, {...options, headers, body: options.body === undefined
      ? undefined : JSON.stringify(options.body)});
  } catch (error) {
    setOffline(offlineState(offline, 'network_error'));
    throw error;
  }
  const reconnecting = offline;
  setOffline(offlineState(offline, 'response'));
  if (reconnecting && !path.startsWith('api/review')) void refreshReview();
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      message = payload.error?.message || message;
    } catch { /* Keep the HTTP status when the body is unavailable. */ }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  try {
    return response.status === 204 ? null : await response.json();
  } catch (error) {
    if (error instanceof TypeError) setOffline(offlineState(offline, 'network_error'));
    throw error;
  }
}

function schedulePoll() {
  if (refreshTimer !== null) clearInterval(refreshTimer);
  refreshTimer = setInterval(refreshReview, reviewPollInterval(review?.series?.rounds, offline));
}

function setOffline(value) {
  if (offline === value) return;
  offline = value;
  renderPin();
  el.offline.hidden = !offline;
  renderPlayerRecovery();
  if (!offline) showStatus('');
  el.newComment.disabled = offline || !review || Boolean(review.media_changed);
  updateComposeSubmit();
  if (review) {
    renderVersions();
    renderThreads();
    renderSendBar();
    updateTasteControls();
  }
  schedulePoll();
}

function renderPlayerRecovery() {
  el.playerMessage.textContent = playerMessage;
  el.offlinePlayerMessage.textContent = playerMessage;
  el.codecError.hidden = !playerMessage || offline;
  el.offlinePlayer.hidden = !playerMessage;
}

function clearPlayerDeadlines() {
  clearTimeout(metadataTimer);
  clearTimeout(firstFrameTimer);
}

function beginPlayerLoad(time = null, resetNetworkRetry = true) {
  clearPlayerDeadlines();
  playerGeneration += 1;
  playerRecoveryPending = false;
  playerMetadataLoaded = false;
  playerResumeTime = time;
  if (resetNetworkRetry) {
    playerNetworkRetried = false;
    playerMessage = '';
  }
  renderPlayerRecovery();
  metadataTimer = setTimeout(() => recoverPlayer('metadata'), MEDIA_METADATA_DEADLINE_MS);
  firstFrameTimer = setTimeout(() => recoverPlayer('frame'), MEDIA_FIRST_FRAME_DEADLINE_MS);
}

function retryPlayer(automatic = false) {
  const time = el.video.readyState === 0 && playerResumeTime !== null
    ? playerResumeTime : el.video.currentTime;
  beginPlayerLoad(Number.isFinite(time) ? time : 0, !automatic);
  el.video.load();
}

async function recoverPlayer(reason) {
  if (playerRecoveryPending) return;
  playerRecoveryPending = true;
  if (reason === 'error') clearPlayerDeadlines();
  const generation = playerGeneration;
  const code = el.video.error?.code;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), MEDIA_REACHABILITY_DEADLINE_MS);
  let fresh;
  let unreachable = false;
  try {
    fresh = await api(versionUrl('api/review', selectedVersion), {signal: controller.signal});
  } catch {
    unreachable = offline;
  } finally {
    clearTimeout(timer);
  }
  if (generation !== playerGeneration) return;
  playerRecoveryPending = false;
  if (reason === 'metadata' && playerMetadataLoaded) return;
  if (fresh?.media_changed) {
    renderReview(fresh);
    playerMessage = 'The file is still being written or has changed. Wait, then retry.';
  } else if (unreachable) {
    playerMessage = 'The video could not reach the Mac. Check the connection, then retry.';
  } else if (code === 3 || code === 4) {
    playerMessage = 'This browser cannot play this file. Retry, or try another browser.';
  } else if (code === 2) {
    playerMessage = 'The video could not load from the Mac. Check the connection, then retry.';
  } else {
    playerMessage = reason === 'frame'
      ? 'The first frame has not loaded. The video load stalled. Retry to load it again.'
      : 'The video load stalled before its details arrived. Retry to load it again.';
  }
  renderPlayerRecovery();
  if (!fresh?.media_changed && (code === 2 || unreachable) && !playerNetworkRetried) {
    playerNetworkRetried = true;
    retryPlayer(true);
  }
}

function duration() {
  const reported = review?.media.duration_s;
  if (Number.isFinite(reported) && reported > 0) return reported;
  return Number.isFinite(el.video.duration) && el.video.duration > 0 ? el.video.duration : null;
}

function lastFrame() {
  const picture = review?.media.duration_s;  // frame bounds use the picture duration only
  return Number.isFinite(picture) && picture > 0 && fps ? frameCount(picture, fps) - 1 : Infinity;
}

function currentFrame() {
  return Math.min(lastFrame(), frameAt(displayedTime, fps));
}

function currentPositionText() {
  return fps ? frameTimeText(currentFrame(), fps) : secondsText(Math.max(0, displayedTime));
}

function completeSelection() {
  if (!selection) return null;
  if (fps && selection.inFrame !== undefined && selection.outFrame !== undefined) {
    return normalizeSelection(selection.inFrame, selection.outFrame);
  }
  if (!fps && selection.inTime !== undefined && selection.outTime !== undefined) {
    return normalizeTimeSelection(selection.inTime, selection.outTime);
  }
  return null;
}

function canonicalizeSelection() {
  const marked = completeSelection();
  if (!marked) return;
  selection = fps
    ? {inFrame: marked.start_frame, outFrame: (marked.end_frame ?? marked.start_frame + 1) - 1}
    : {inTime: marked.start_s, outTime: marked.end_s ?? marked.start_s};
}

function timelinePercent(time) {
  const length = duration();
  return length === null ? 0 : Math.max(0, Math.min(100, time / length * 100));
}

function renderSelection() {
  const marked = completeSelection();
  renderRangePanel();
  const hasMark = selection && Object.keys(selection).length > 0;
  el.selection.hidden = !hasMark;
  el.selectionReadout.hidden = marked === null;
  el.composeClearRange.hidden = marked === null;
  if (!hasMark) {
    el.composeTime.textContent = currentPositionText();
    return;
  }
  let start;
  let end;
  if (fps) {
    start = marked ? frameStart(marked.start_frame, fps)
      : frameStart(selection.inFrame ?? selection.outFrame, fps);
    end = marked ? frameStart(marked.end_frame ?? marked.start_frame + 1, fps) : start;
  } else {
    start = marked ? marked.start_s : selection.inTime ?? selection.outTime;
    end = marked ? marked.end_s ?? marked.start_s : start;
  }
  el.selection.style.setProperty('--start', `${timelinePercent(start)}%`);
  el.selection.style.setProperty('--end', `${timelinePercent(end)}%`);
  el.selection.classList.toggle('incomplete', marked === null);
  if (marked) {
    const span = spanText(fps ? marked : {start_frame: null, ...marked}, fps);
    el.selectionText.textContent = `Marked ${span}`;
    el.composeTime.textContent = span;
  } else {
    el.composeTime.textContent = currentPositionText();
  }
}

function clearSelection() {
  selection = null;
  cutError = '';
  renderSelection();
}

function markSelection(edge) {
  if (!review || cutBusy) return;
  cutError = '';
  canonicalizeSelection();
  selection ||= {};
  if (fps) selection[edge === 'in' ? 'inFrame' : 'outFrame'] = currentFrame();
  else selection[edge === 'in' ? 'inTime' : 'outTime'] = displayedTime;
  canonicalizeSelection();
  renderSelection();
}

function selectionWords(marked) {
  if (!marked || !wordsState.available || wordsState.version !== selectedVersion) return [];
  const start = fps ? marked.start_frame : marked.start_s;
  const end = fps ? marked.end_frame : marked.end_s;
  if (rangeWordsCache?.state !== wordsState || rangeWordsCache.start !== start
    || rangeWordsCache.end !== end) {
    const covered = wordsInRange(wordsState.words,
      fps ? {num: start * fps.den, den: fps.num} : start,
      fps ? {num: end * fps.den, den: fps.num} : end);
    rangeWordsCache = {state: wordsState, start, end, covered};
  }
  return rangeWordsCache.covered;
}

function selectedRange() {
  const marked = completeSelection();
  return marked && (fps ? marked.end_frame !== null : marked.end_s !== null) ? marked : null;
}

function renderRangePanel() {
  const blocked = !review || offline || Boolean(review.media_changed);
  el.markStart.disabled = blocked || cutBusy;
  el.markEnd.disabled = blocked || cutBusy;
  const marked = selectedRange();
  el.rangePanel.hidden = !marked;
  if (!marked) return;
  const covered = selectionWords(marked);
  const state = cutState(wordsState.version === selectedVersion ? wordsState : null, covered);
  const start = fps ? frameStart(marked.start_frame, fps) : marked.start_s;
  const end = fps ? frameStart(marked.end_frame, fps) : marked.end_s;
  el.rangeLength.textContent = rangeLengthText(start, end);
  el.rangeWords.textContent = state.reason || (state.show ? cutWordsText(covered) : '');
  el.rangeWords.hidden = !state.show;
  el.rangePlay.disabled = blocked;
  el.rangeCut.hidden = !state.show;
  el.rangeCut.disabled = blocked || !state.enabled || composeBusy || Boolean(editState)
    || switchingVersion || roundBusy || releaseBusy || answerBusy || cutBusy;
  for (const button of [el.rangeChange, el.rangeKeep]) {
    if (button) button.disabled = blocked || composeBusy || cutBusy || Boolean(editState) || switchingVersion;
  }
  el.rangeError.textContent = cutError.replace(/^Bad value for [^:]+: /u, '');
  el.rangeError.hidden = !cutError;
}

async function cutSelection() {
  const marked = selectedRange();
  if (!marked || !review || offline || review.media_changed || editState || composeBusy
    || switchingVersion || roundBusy || releaseBusy || answerBusy || cutBusy
    || wordsState.version !== selectedVersion || !cutState(wordsState, selectionWords(marked)).enabled) return;
  const version = selectedVersion;
  const note = String(el.rangeNote?.value ?? '').trim();
  const body = cutBody(marked, fps, note || (el.compose.hidden ? '' : el.composeText.value));
  cutBusy = true;
  composeBusy = true;
  cutError = '';
  updateComposeSubmit();
  renderVersions();
  renderSendBar();
  try {
    const created = await api(versionUrl('api/cut', version), {method: 'POST', body});
    selectedId = created.id;
    review.comments.push(created);
    if (!el.compose.hidden) closeCompose();
    if (el.rangeNote) el.rangeNote.value = '';
    clearSelection();
    renderHeader();
    renderComments();
    renderMarkers();
    try {
      await loadReview(version);
    } catch { /* Keep the saved comment visible when the refresh cannot reconnect. */ }
    showToast('Cut saved');
  } catch (error) {
    cutError = errorMessage(error);
    showStatus(cutError);
  } finally {
    cutBusy = false;
    composeBusy = false;
    updateComposeSubmit();
    renderVersions();
    renderSendBar();
  }
}

async function commentOnRange(kind) {
  // Change this and Keep this save a note on the marked part, right under the video.
  const marked = selectedRange();
  if (!marked || !review || offline || review.media_changed || composeBusy || cutBusy || editState) return;
  let text = String(el.rangeNote?.value ?? '').trim();
  if (!text && kind === 'fix') {
    cutError = 'Say what should change, then press Change this.';
    renderRangePanel();
    el.rangeNote?.focus();
    return;
  }
  if (!text) text = 'Keep this part as it is.';
  composeBusy = true;
  cutError = '';
  renderRangePanel();
  try {
    const created = await api(versionUrl('api/comments', selectedVersion),
      {method: 'POST', body: {kind, target: null, text, ...marked}});
    selectedId = created.id;
    if (el.rangeNote) el.rangeNote.value = '';
    clearSelection();
    await loadReview();
    showToast(kind === 'keep' ? 'Keep saved' : 'Note saved');
  } catch (error) {
    cutError = errorMessage(error);
    showStatus(cutError);
  } finally {
    composeBusy = false;
    updateComposeSubmit();
    renderVersions();
    renderSendBar();
  }
}

function updateTimeline() {
  if (review === null) return;
  const length = duration();
  const current = currentPositionText();
  const total = length === null ? 'unknown' : secondsText(length);
  el.readout.textContent = `${current} / ${total}`;
  el.readout.title = fps ? `Frame ${currentFrame()}` : '';
  const position = timelinePercent(displayedTime);
  el.playhead.style.setProperty('--position', `${position}%`);
  el.videoMapPlayhead.style.setProperty('--position', `${position}%`);
  el.scrubber.setAttribute('aria-valuenow', String(Math.round(position)));
  el.scrubber.setAttribute('aria-valuetext', current);
  renderSelection();
  renderPin();
}

function setDisplayedTime(time) {
  if (!Number.isFinite(time)) return;
  displayedTime = Math.max(0, time);
  updateTimeline();
}

function seekFrame(frame) {
  if (!fps) return;
  rangePlayback = null;
  const target = Math.max(0, Math.min(lastFrame(), frame));
  const middle = frameMid(target, fps);
  manualSeek = true;
  el.video.pause();
  el.video.currentTime = middle;
  setDisplayedTime(middle);
}

function seekTime(time) {
  rangePlayback = null;
  const length = duration();
  const target = Math.max(0, length === null ? time : Math.min(length, time));
  manualSeek = true;
  el.video.pause();
  el.video.currentTime = target;
  setDisplayedTime(target);
}

function seekComment(comment) {
  selectedId = comment.id;
  if (fps && comment.start_frame !== null) seekFrame(comment.pin?.frame ?? comment.start_frame);
  else seekTime(comment.pin?.time_s ?? comment.start_s);
  renderComments();
}

function pinMode() {
  return Boolean(review && !replyNavigationBlocked() && !activeWrites && !cutBusy
    && (editState || !el.compose.hidden));
}

function renderPin() {
  renderCaptionAccess();
  const draft = editState ? editState.pin : composePin;
  const composeText = composePin ? pinText(composePin)
    : 'Tap the picture to pin a spot (optional).';
  if (el.composePinText.textContent !== composeText) el.composePinText.textContent = composeText;
  if (el.composePinRemove.hidden !== !composePin) el.composePinRemove.hidden = !composePin;
  if (editState?.pinText) {
    const editText = draft ? pinText(draft)
      : 'Tap the picture to pin a spot (optional).';
    if (editState.pinText.textContent !== editText) editState.pinText.textContent = editText;
    if (editState.pinRemove.hidden !== !draft) editState.pinRemove.hidden = !draft;
  }
  const comment = editState?.comment ?? review?.comments.find((item) => item.id === selectedId);
  const isDraft = pinMode() && (editState ? editState.pinTouched : !el.compose.hidden);
  const pin = isDraft ? draft : (editState ? editState.pin : comment?.pin);
  const visible = pin && !offline && !switchingVersion && !review?.media_changed && el.video.paused
    && (isDraft || pinShown({...comment, pin}, fps, fps ? currentFrame() : null, displayedTime));
  if (el.pinLayer.hidden !== !visible) el.pinLayer.hidden = !visible;
  if (!visible) return;
  const box = el.video.getBoundingClientRect();
  const rect = pictureRect(box, el.video.videoWidth, el.video.videoHeight);
  const centre = pinPoint(rect, pin);
  const wrap = el.video.closest('.video-wrap').getBoundingClientRect();
  const left = `${centre.left - wrap.left}px`;
  const top = `${centre.top - wrap.top}px`;
  if (pinPosition.left !== left) {
    el.pinMarker.style.setProperty('--pin-left', left);
    pinPosition.left = left;
  }
  if (pinPosition.top !== top) {
    el.pinMarker.style.setProperty('--pin-top', top);
    pinPosition.top = top;
  }
}

function setDraftPin(pin) {
  if (!pinMode()) return;
  el.video.pause();
  if (editState) {
    editState.pin = pin;
    editState.pinTouched = true;
    editState.errorNode.textContent = '';
  } else {
    composePin = pin;
    el.composeError.textContent = '';
  }
  renderPin();
}

function currentCaption() {
  if (offline || switchingVersion || review?.media_changed || !el.video.paused
    || captionsState.version !== selectedVersion || !captionsState.available) return null;
  return captionsState.phrases.find((phrase) => phrase.start <= displayedTime
    && displayedTime < phrase.end) ?? null;
}

function renderCaptionAccess() {
  if (!el.fixCaption) return;
  el.fixCaption.hidden = !currentCaption();
  el.fixCaption.disabled = replyNavigationBlocked() || activeWrites || cutBusy
    || !el.compose.hidden || Boolean(editState);
}

function closeCaptionFix(focus = true) {
  if (composeBusy) return;
  captionEdit = null;
  el.captionFix.hidden = true;
  if (focus) el.fixCaption.focus();
}

function openCaptionFix(phrase = currentCaption()) {
  if (!phrase || replyNavigationBlocked() || activeWrites || cutBusy
    || editState || !el.compose.hidden || !currentCaption()) return;
  captionEdit = {word: null, version: selectedVersion};
  el.captionFix.hidden = false;
  el.captionTo.value = '';
  el.captionTo.disabled = true;
  el.captionSave.disabled = true;
  el.captionEvery.checked = false;
  el.captionEveryLabel.hidden = true;
  el.captionError.textContent = '';
  el.captionWords.replaceChildren();
  let firstWord = null;
  for (const word of phrase.words) {
    const chip = makeButton(word.text, '', () => {
      if (composeBusy || word.editable === false) return;
      captionEdit.word = word;
      for (const button of el.captionWords.querySelectorAll('button')) {
        button.setAttribute('aria-pressed', 'false');
      }
      chip.setAttribute('aria-pressed', 'true');
      el.captionTo.disabled = false;
      el.captionTo.value = word.text;
      el.captionEvery.checked = false;
      el.captionEveryLabel.hidden = word.repeats < 2;
      el.captionSave.disabled = false;
      el.captionTo.focus();
      el.captionTo.setSelectionRange(0, word.text.length);
    });
    chip.setAttribute('aria-pressed', 'false');
    chip.disabled = word.editable === false;
    el.captionWords.append(chip);
    if (chip.disabled) {
      const reason = document.createElement('span');
      reason.id = `caption-word-${word.id}-reason`;
      reason.textContent = word.reason;
      chip.setAttribute('aria-describedby', reason.id);
      el.captionWords.append(reason);
    } else firstWord ??= chip;
  }
  (firstWord ?? el.captionCancel).focus();
}

async function saveCaptionFix() {
  if (!captionEdit?.word || replyNavigationBlocked() || activeWrites || cutBusy) return;
  const {word, version} = captionEdit;
  const to = el.captionTo.value.trim();
  if (!to || [...to].length > 80 || /[\r\n\v\f\u0085\u2028\u2029]/u.test(to)) {
    el.captionError.textContent = 'Use 1 to 80 characters with no newlines.';
    return;
  }
  composeBusy = true;
  el.captionSave.disabled = true;
  el.captionCancel.disabled = true;
  el.captionTo.disabled = true;
  el.captionEvery.disabled = true;
  el.captionError.textContent = '';
  try {
    await api(versionUrl('api/caption-fixes', version), {method: 'POST',
      body: {word: word.id, start: word.start, to,
        every_time: word.repeats > 1 && el.captionEvery.checked}});
    await loadReview(version);
    composeBusy = false;
    closeCaptionFix();
    showToast('Caption fix saved. Send your notes when ready.');
  } catch (error) {
    el.captionError.textContent = errorMessage(error);
  } finally {
    composeBusy = false;
    el.captionSave.disabled = false;
    el.captionCancel.disabled = false;
    el.captionTo.disabled = false;
    el.captionEvery.disabled = false;
    renderCaptionAccess();
    renderVersions();
    renderSendBar();
  }
}

async function undoCaptionFix(id) {
  if (composeBusy || offline || switchingVersion) return;
  composeBusy = true;
  try {
    await api(versionUrl('api/caption-fixes', selectedVersion), {method: 'POST', body: {undo: id}});
    await loadReview(selectedVersion);
    showToast('Caption fix undone.');
  } catch (error) {
    showStatus(errorMessage(error));
  } finally {
    composeBusy = false;
    renderCaptionAccess();
    renderVersions();
    renderSendBar();
  }
}

function placePin(event) {
  if (ignorePinClick) { ignorePinClick = false; return; }
  if (!el.video.paused) { el.video.pause(); return; }
  const rect = pictureRect(el.video.getBoundingClientRect(), el.video.videoWidth, el.video.videoHeight);
  const pin = pinAt(rect, event.clientX, event.clientY);
  const caption = currentCaption() && captionAt(captionsState, displayedTime, rect,
    event.clientX, event.clientY, fps ? currentFrame() : null);
  if (caption && !pinMode()) { openCaptionFix(caption); return; }
  if (pinMode()) {
    if (pin) setDraftPin(pin);
  } else if (!pin) togglePlayback();
  else openCompose({pin});
}

function movePin(event) {
  if (pinDrag?.id !== event.pointerId || !pinMode()) return;
  const rect = pictureRect(el.video.getBoundingClientRect(), el.video.videoWidth, el.video.videoHeight);
  const x = Math.max(rect.left, Math.min(rect.left + rect.width, event.clientX));
  const y = Math.max(rect.top, Math.min(rect.top + rect.height, event.clientY));
  if (event.clientX !== pinDrag.x || event.clientY !== pinDrag.y) pinDrag.moved = true;
  if (pinDrag.moved) setDraftPin(pinAt(rect, x, y));
}

function playRange(span) {
  if (span.id) selectedId = span.id;
  if (fps && span.end_frame !== null) {
    const target = frameMid(span.start_frame, fps);
    const atStart = !el.video.seeking && Math.abs(el.video.currentTime - target) < 0.00001;
    seekFrame(span.start_frame);
    rangePlayback = {endFrame: span.end_frame, starting: true};
    if (atStart) startRangePlayback();
  } else if (span.end_s !== null && span.end_s !== undefined) {
    const target = span.start_s;
    const atStart = !el.video.seeking && Math.abs(el.video.currentTime - target) < 0.00001;
    seekTime(span.start_s);
    rangePlayback = {endTime: span.end_s, starting: true};
    if (atStart) startRangePlayback();
  } else return;
  renderComments();
}

function startRangePlayback() {
  if (!rangePlayback?.starting || rangePlayback.playRequested || el.video.seeking) return;
  const playback = rangePlayback;
  playback.playRequested = true;
  el.video.play().catch((error) => {
    if (rangePlayback !== playback) return;
    rangePlayback = null;
    showStatus(errorMessage(error));
  });
}

function checkRangeStop(time) {
  if (!rangePlayback || rangePlayback.starting) return;
  if (rangePlayback.endFrame !== undefined && frameAt(time, fps) >= rangePlayback.endFrame - 1) {
    seekFrame(rangePlayback.endFrame - 1);
  } else if (rangePlayback.endTime !== undefined && time >= rangePlayback.endTime) {
    seekTime(rangePlayback.endTime);
  }
}

function stepFrame(direction) {
  if (!el.video.paused) {
    el.video.pause();
  }
  if (fps) seekFrame(currentFrame() + direction);
  else seekTime(displayedTime + direction / 30);
}

function togglePlayback() {
  rangePlayback = null;
  if (el.video.paused) {
    el.video.play().catch((error) => showStatus(errorMessage(error)));
  } else {
    el.video.pause();
  }
}

function snapPausedFrame() {
  if (!fps || manualSeek) return;
  const middle = frameMid(currentFrame(), fps);
  if (Math.abs(el.video.currentTime - middle) > 0.00001) el.video.currentTime = middle;
  setDisplayedTime(middle);
}

function copyFallback(text) {
  el.copyFallbackText.value = text;
  if (!el.copyFallback.open) el.copyFallback.showModal();
  el.copyFallbackText.focus();
  el.copyFallbackText.select();
}

function showToast(message) {
  el.toast.textContent = message;
  el.toast.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.remove('visible'), 2500);
}

function copyText(text, count) {
  // Start the clipboard request within the click handler's user gesture.
  try {
    if (!navigator.clipboard?.writeText) {
      copyFallback(text);
      return;
    }
    navigator.clipboard.writeText(text).then(
      () => showToast(count === 0 ? 'Copied feedback' :
        `Copied ${count} ${count === 1 ? 'comment' : 'comments'}`),
      () => copyFallback(text),
    );
  } catch {
    copyFallback(text);
  }
}

function fitVideo(video) {
  // Size the player to the video's own shape so a vertical video fills a phone.
  if (video?.style && video.videoWidth && video.videoHeight) {
    video.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  }
}

function renderHeader() {
  const media = review.media;
  const title = review.series?.project ? (review.series.title || projectTitle(media.name)) : media.name;
  el.name.textContent = title;
  document.title = `${title} · Yap Studio review`;
  const length = duration();
  const facts = [length === null ? 'Unknown length' : secondsText(length)];
  if (review.series?.project) facts.unshift(`Version ${selectedVersion}`);
  el.facts.textContent = facts.join(' · ');
  const note = review.series?.project ? seriesVersion(selectedVersion)?.note : null;
  if (el.versionNote) {
    el.versionNote.hidden = !note;
    el.versionNote.textContent = note ? `Claude's notes: ${note}` : '';
  }
  const home = homeUrl(review.home, location.hostname);
  if (el.homeLink) {
    el.homeLink.hidden = !home;
    if (home) el.homeLink.href = home;
    const label = home && home === review.home?.phone_url ? phoneLinkLabel(review.home.phone_via) : '';
    el.homeLink.textContent = 'Back to Yap Studio' + (label ? ` · ${label}` : '');
  }
  el.copy.disabled = review.comments.length === 0;
  const phoneLink = $('phone-link');
  if (phoneLink) {
    phoneLink.hidden = !review.phone;
    if (review.phone) {
      phoneLink.href = review.phone.url;
      phoneLink.textContent = `Open on your phone: ${phoneLinkLabel(review.phone.via)}`;
    }
  }
  el.copyHint.textContent = el.copy.disabled ? 'Add a note to copy.' : '';
  el.copyHint.hidden = !el.copy.disabled;
  if (fps === null) {
    el.previousFrame.textContent = '← 1/30 s';
    el.previousFrame.setAttribute('aria-label', 'Back one thirtieth of a second');
    el.nextFrame.textContent = '1/30 s →';
    el.nextFrame.setAttribute('aria-label', 'Forward one thirtieth of a second');
    el.shortcutHint.textContent = 'Space play/pause · C comment · I/O mark range · X clear · ← → nudge 1/30 s · click the paused picture to pin a spot';
  } else {
    el.previousFrame.textContent = '← Frame';
    el.previousFrame.setAttribute('aria-label', 'Previous frame');
    el.nextFrame.textContent = 'Frame →';
    el.nextFrame.setAttribute('aria-label', 'Next frame');
    el.shortcutHint.textContent = 'Space play/pause · C comment · I/O mark range · X clear · ← → step one frame · click the paused picture to pin a spot';
  }
}

function seriesVersion(number) {
  return review?.series?.versions.find((item) => item.number === number);
}

function renderVersions() {
  updateFindingControls();
  const series = review.series;
  el.versionBar.hidden = !series;
  if (!series) return;
  el.versionButtons.replaceChildren();
  for (const item of series.versions) {
    const button = makeButton(`Version ${item.number} · ${reportStatusLabel(item.report_status)}`,
      '', () => selectVersion(item.number));
    button.setAttribute('aria-pressed', String(item.number === selectedVersion));
    button.disabled = offline || switchingVersion || composeBusy || roundBusy || answerBusy || releaseBusy
      || tasteSaving.size > 0
      || Boolean(editState?.busy);
    for (const text of versionBadges(item.number, series)) {
      const badge = document.createElement('span');
      badge.className = `version-badge ${text.toLowerCase()}`;
      badge.textContent = text;
      button.append(badge);
    }
    el.versionButtons.append(button);
  }
  el.versionActions.hidden = !series.project;
  const approved = selectedVersion === series.approved?.version;
  el.rollback.hidden = selectedVersion === series.current_version;
  const story = storyCutStep(series, selectedVersion, review.home);
  el.approve.hidden = approved || story.hideApprove;
  if (el.makeFinal) {
    el.makeFinal.hidden = !story.offerFinal;
    el.makeFinal.disabled = Boolean(story.requested);
    el.makeFinal.textContent = story.requested ? 'Claude is making the final' : 'Make the final video';
  }
  if (el.approveHint) el.approveHint.textContent = story.hint;
  const blocked = offline || releaseBusy || switchingVersion || composeBusy || roundBusy
    || answerBusy || tasteSaving.size > 0 || Boolean(editState?.busy) || Boolean(review.media_changed);
  el.rollback.disabled = blocked;
  el.approve.disabled = blocked;
  if (el.makeFinal && blocked) el.makeFinal.disabled = true;
  for (const [link, path] of [[el.downloadVideo, 'api/download'],
    [el.downloadReport, 'api/download/report']]) {
    link.hidden = !series.project || !approved;
    link.setAttribute('aria-disabled', String(offline));
    link.tabIndex = offline ? -1 : 0;
    if (approved && !offline) link.href = versionUrl(path, selectedVersion, token);
    else link.removeAttribute('href');
  }
  el.compareToggle.disabled = series.versions.length < 2 || switchingVersion;
  if (compareVersion === null || compareVersion === selectedVersion) {
    compareVersion = defaultCompareVersion(selectedVersion, series.versions);
    mapGeneration += 1;
  }
  el.compareVersion.replaceChildren();
  for (const item of series.versions.filter((item) => item.number !== selectedVersion)) {
    const option = document.createElement('option');
    option.value = String(item.number);
    option.textContent = `v${item.number}${item.number === series.newest ? ' · Latest' : ''}`;
    el.compareVersion.append(option);
  }
  el.compareVersion.value = String(compareVersion);
  history.replaceState(null, '', versionUrl(
    `${location.pathname}${location.search}${location.hash}`, selectedVersion));
  if (compareOpen) setCompareOpen(true);
}

async function releaseVersion(action) {
  if (offline || releaseBusy || switchingVersion || composeBusy || roundBusy
    || answerBusy || tasteSaving.size > 0 || editState?.busy || !review?.series?.project) return;
  const version = selectedVersion;
  const message = action === 'rollback'
    ? `Make v${version} current? The next notes build from v${version}.`
    : `Approve v${version}? This allows its video and check report to be downloaded.`;
  if (!confirm(message)) return;
  releaseBusy = true;
  renderVersions();
  renderSendBar();
  try {
    const result = await api(`api/${action}`, {method: 'POST', body: {version}});
    await loadReview(version);
    if (action === 'rollback' && result.not_restored?.length) {
      showStatus(`Current version: v${version}. Not restored: ${result.not_restored.join(', ')}.`);
    }
  } catch (error) {
    showStatus(errorMessage(error));
  } finally {
    releaseBusy = false;
    renderVersions();
    renderSendBar();
  }
}

async function requestFinal() {
  if (offline || releaseBusy || !review?.series?.project) return;
  releaseBusy = true;
  renderVersions();
  try {
    await api('api/final', {method: 'POST', body: {version: selectedVersion}});
    await loadReview(selectedVersion);
    showStatus('');
  } catch (error) {
    showStatus(errorMessage(error));
  } finally {
    releaseBusy = false;
    renderVersions();
  }
}

async function selectVersion(number) {
  if (offline || number === selectedVersion || switchingVersion || openingReplyEvidence
    || composeBusy || roundBusy || releaseBusy
    || answerBusy || tasteSaving.size > 0 || editState?.busy) return;
  switchingVersion = true;
  renderPin();
  updateReplyEvidenceButtons();
  replyEvidenceTime = null;
  mapGeneration += 1;
  el.video.pause();
  el.players.closest('main').inert = true;
  renderVersions();
  try {
    await loadReview(number);
  } catch { /* loadReview displays the error while retaining the selected version. */ }
  finally {
    switchingVersion = false;
    renderPin();
    el.players.closest('main').inert = false;
    renderVersions();
    renderThreads();
    renderSendBar();
    updateTasteControls();
  }
}

function setCompareOpen(open, number = compareVersion) {
  if (!review?.series || review.series.versions.length < 2) return;
  if (!open || number !== compareVersion) replyEvidenceTime = null;
  compareOpen = open;
  compareVersion = number;
  el.comparePanel.hidden = !open;
  el.players.classList.toggle('compare-open', open);
  el.compareToggle.setAttribute('aria-expanded', String(open));
  el.compareVersion.value = String(number);
  if (!open) {
    el.compareVideo.pause();
    mapGeneration += 1;
    clearTimeout(matchTimer);
    return;
  }
  const source = versionUrl('media', number, token);
  if (el.compareVideo.getAttribute('src') !== source) {
    el.compareVideo.pause();
    el.compareVideo.src = source;
    el.mapStatus.textContent = '';
    el.mapCandidates.replaceChildren();
    mapGeneration += 1;
  }
}

async function seekPlayer(player, time, navigationCurrent = null) {
  if (player.readyState < 1) {
    await new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timer);
        player.removeEventListener('loadedmetadata', ready);
        player.removeEventListener('error', failed);
      };
      const ready = () => { cleanup(); resolve(); };
      const failed = () => { cleanup(); reject(new Error('Could not load this version.')); };
      const timer = setTimeout(failed, 10000);
      player.addEventListener('loadedmetadata', ready, {once: true});
      player.addEventListener('error', failed, {once: true});
    });
  }
  if (navigationCurrent && !navigationCurrent()) return;
  player.pause();
  player.currentTime = Math.max(0, Math.min(time, player.duration || Infinity));
}

async function matchMoment(reverse = false) {
  if (!compareOpen || switchingVersion) return;
  const generation = ++mapGeneration;
  const from = reverse ? compareVersion : selectedVersion;
  const to = reverse ? selectedVersion : compareVersion;
  const source = reverse ? el.compareVideo : el.video;
  const destination = reverse ? el.video : el.compareVideo;
  const sourceFacts = seriesVersion(from);
  const targetFacts = seriesVersion(to);
  if (!sourceFacts || !targetFacts) return;
  const rate = sourceFacts.cfr ? parseFps(sourceFacts.fps) : null;
  const params = new URLSearchParams({from: String(from), to: String(to)});
  if (rate) params.set('frame', String(frameAt(source.currentTime, rate)));
  else params.set('time', String(source.currentTime));
  try {
    const mapped = await api(`api/map?${params}`);
    if (generation !== mapGeneration || !compareOpen) return;
    const targetRate = targetFacts.cfr ? parseFps(targetFacts.fps) : null;
    const last = targetRate && Number.isFinite(targetFacts.duration_s)
      ? frameCount(targetFacts.duration_s, targetRate) - 1 : Infinity;
    const position = targetRate && mapped.frame !== null
      ? frameMid(Math.min(mapped.frame, last), targetRate) : mapped.time_s;
    if (reverse) suppressMatchUntil = performance.now() + 350;
    await seekPlayer(destination, position);
    if (generation !== mapGeneration || !compareOpen) return;
    if (reverse) setDisplayedTime(destination.currentTime);
    el.mapStatus.textContent = mapStatusMessage(mapped.status, to);
    el.mapCandidates.replaceChildren();
    for (const candidate of mapped.candidates) {
      const button = makeButton(`v${to} · ${secondsText(candidate.start)} · ${candidate.segment}`,
        '', async () => {
          if (reverse) suppressMatchUntil = performance.now() + 350;
          const time = targetRate ? frameMid(frameAt(candidate.start, targetRate), targetRate)
            : candidate.start;
          try {
            await seekPlayer(destination, time);
            if (reverse) setDisplayedTime(destination.currentTime);
          } catch (error) { el.mapStatus.textContent = errorMessage(error); }
        });
      el.mapCandidates.append(button);
    }
  } catch (error) {
    if (generation === mapGeneration) el.mapStatus.textContent = errorMessage(error);
  }
}

function scheduleMatch() {
  if (!compareOpen || switchingVersion || openingReplyEvidence
    || performance.now() < suppressMatchUntil) return;
  // Keep the measured pair until the creator moves away from this evidence moment.
  if (replyEvidenceTime !== null && el.video.paused
    && Math.abs(el.video.currentTime - replyEvidenceTime) < 0.001) return;
  replyEvidenceTime = null;
  clearTimeout(matchTimer);
  matchTimer = setTimeout(() => matchMoment(), 150);
}

async function openThread(thread) {
  if (switchingVersion) return;
  setCompareOpen(true, thread.version);
  const facts = seriesVersion(thread.version);
  const rate = facts.cfr ? parseFps(facts.fps) : null;
  const frame = thread.pin?.frame ?? thread.start_frame;
  const time = rate && frame !== null
    ? frameMid(frame, rate) : thread.pin?.time_s ?? thread.start_s;
  try {
    await seekPlayer(el.compareVideo, time);
    await matchMoment(true);
  } catch (error) { el.mapStatus.textContent = errorMessage(error); }
}

function replyNavigationBlocked() {
  return offline || switchingVersion || openingReplyEvidence || composeBusy || roundBusy
    || releaseBusy || answerBusy || tasteSaving.size > 0
    || Boolean(editState?.busy) || Boolean(review?.media_changed);
}

function updateReplyEvidenceButtons() {
  if (!review?.series?.threads?.length) return;
  const disabled = replyNavigationBlocked();
  for (const button of el.threadList.querySelectorAll('.reply-evidence')) {
    button.disabled = disabled;
  }
}

async function openReplyEvidence(thread) {
  const reply = thread.reply;
  const moment = replyEvidence(reply)?.moment;
  if (replyNavigationBlocked() || !moment) return;
  const targetVersion = reply.version;
  // Start the reply's switch before blocking further version requests.
  const versionSelection = selectedVersion !== targetVersion ? selectVersion(targetVersion) : null;
  const navigation = loadVersion;
  let generation = null;
  const navigationCurrent = () => selectedVersion === targetVersion && navigation === loadVersion
    && (generation === null || generation === mapGeneration);
  openingReplyEvidence = true;
  clearTimeout(matchTimer);
  mapGeneration += 1;
  el.players.closest('main').inert = true;
  renderThreads();
  try {
    if (versionSelection) await versionSelection;
    // A failed or blocked switch keeps its status message and cannot use the old timeline.
    if (!navigationCurrent() || offline || review.media_changed) return;
    el.players.closest('main').inert = true;
    const oldFacts = seriesVersion(thread.version);
    const newFacts = seriesVersion(targetVersion);
    if (!oldFacts || !newFacts) throw new Error('Could not find the reply versions.');
    setCompareOpen(true, thread.version);
    clearTimeout(matchTimer);
    generation = ++mapGeneration;
    const oldRate = oldFacts.cfr ? parseFps(oldFacts.fps) : null;
    const newRate = newFacts.cfr ? parseFps(newFacts.fps) : null;
    const oldTime = oldRate && Number.isInteger(thread.start_frame)
      ? frameMid(thread.start_frame, oldRate) : thread.start_s;
    const newTime = newRate && Number.isInteger(moment.start_frame)
      ? frameMid(moment.start_frame, newRate) : moment.start_s;
    rangePlayback = null;
    manualSeek = true;
    el.video.pause();
    await seekPlayer(el.compareVideo, oldTime, navigationCurrent);
    if (!navigationCurrent() || offline || review.media_changed) return;
    await seekPlayer(el.video, newTime, navigationCurrent);
    if (!navigationCurrent() || offline || review.media_changed) return;
    replyEvidenceTime = el.video.currentTime;
    setDisplayedTime(el.video.currentTime);
    el.mapStatus.textContent = replyMomentText(reply, newFacts);
    el.mapCandidates.replaceChildren();
    showStatus('');
    el.players.scrollIntoView({block: 'start'});
  } catch (error) {
    if (navigationCurrent()) showStatus(errorMessage(error));
  } finally {
    openingReplyEvidence = false;
    el.players.closest('main').inert = false;
    renderThreads();
  }
}

function renderThreads() {
  const threads = review.series?.threads ?? [];
  el.earlierThreads.hidden = threads.length === 0;
  el.threadList.replaceChildren();
  for (const thread of threads) {
    const facts = seriesVersion(thread.version);
    const rate = facts.cfr ? parseFps(facts.fps) : null;
    const row = document.createElement('li');
    const button = makeButton('', 'thread-row', () => openThread(thread));
    const fields = [
      ['timestamp', `v${thread.version} · ${spanText(thread, rate)}`],
      ['thread-tag', `${thread.kind === 'keep' ? 'Keep' : 'Fix'}${thread.target ? `: ${targetName(thread.target)}` : ''}`],
      ['thread-text', thread.text],
      [`state-pill ${thread.state}`, threadStateLabel(thread.state)],
    ];
    if (thread.request) fields.push(['request-text', requestText(thread.request)]);
    for (const [className, text] of fields) {
      const span = document.createElement('span');
      span.className = className;
      span.textContent = text;
      button.append(span);
    }
    row.append(button);
    if (thread.pin) {
      const spot = document.createElement('p');
      spot.className = 'pin-text';
      spot.textContent = pinText(thread.pin);
      button.append(spot);
    }
    if (thread.reply) {
      const reply = thread.reply;
      const block = document.createElement('div');
      block.className = 'thread-reply';
      for (const [className, text] of [
        ['reply-heading', replyHeading(reply)], ['reply-text', reply.text],
        ['reply-detail', changeText(reply)], ['reply-detail', keepCheckText(reply)],
      ]) {
        if (text === null) continue;
        const line = document.createElement('p');
        line.className = className;
        line.textContent = text;
        block.append(line);
      }
      const momentText = replyMomentText(reply, seriesVersion(reply.version));
      if (momentText !== null) {
        const evidenceButton = makeButton(momentText, 'reply-evidence', () => openReplyEvidence(thread));
        evidenceButton.disabled = replyNavigationBlocked();
        block.append(evidenceButton);
      }
      row.append(block);
    }
    appendQuestions(row, thread.version, thread.id);
    el.threadList.append(row);
  }
}

function renderPrevious() {
  const previous = review.previous;
  el.previousBanner.hidden = previous === null;
  if (previous === null) return;
  el.previousMessage.textContent = `This file changed since it was reviewed. ${previous.count} ${previous.count === 1 ? 'comment was' : 'comments were'} left on the earlier version.`;
  el.previousText.textContent = previous.text;
}

function makeButton(text, className, handler) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = text;
  if (className) button.className = className;
  button.addEventListener('click', (event) => {
    event.stopPropagation();
    return handler();
  });
  return button;
}

function updateTasteControls() {
  if (!review?.series?.taste) return;
  for (const card of [...el.tasteList.querySelectorAll('.taste-card'),
    ...el.tasteDecidedList.querySelectorAll('.taste-card')]) {
    const busy = tasteSaving.has(card.dataset.proposal);
    const blocked = busy || offline || switchingVersion || Boolean(review?.media_changed);
    card.setAttribute('aria-busy', String(busy));
    const textarea = card.querySelector('textarea');
    for (const control of card.querySelectorAll('button, textarea')) {
      control.disabled = blocked || (control.dataset.wording === 'true' && !textarea.value.trim());
    }
  }
}

async function goToTasteSource(source) {
  if (offline || review.media_changed || tasteSaving.size > 0) return;
  if (selectedVersion !== source.version) await selectVersion(source.version);
  if (selectedVersion !== source.version || review.media_changed) return;
  if (fps) seekFrame(source.frame);
  else if (source.time_s !== null) seekTime(source.time_s);
}

function renderTaste() {
  const taste = review?.series?.taste;
  el.tasteList.replaceChildren();
  el.tasteDecidedList.replaceChildren();
  el.tasteEmpty.hidden = true;
  el.tasteDecided.hidden = true;
  el.taste.hidden = !review?.series?.project || !taste;
  if (el.taste.hidden) return;
  const items = taste.proposals ?? [];
  // Hide an empty inbox; show no waiting rules only when proposals exist and none is pending.
  el.taste.hidden = !items.length && !taste.error;
  el.tasteCount.textContent = String(pendingCount(items));
  el.tasteError.hidden = taste?.available !== false;
  el.tasteError.textContent = taste?.available === false
    ? taste.error?.message ?? 'The taste rules could not be read.' : '';
  if (el.taste.hidden || taste.available === false) return;
  const {pending, decided} = splitProposals(items);
  el.tasteEmpty.hidden = pending.length !== 0;
  el.tasteDecided.hidden = decided.length === 0;
  el.tasteDecidedCount.textContent = String(decided.length);
  el.tasteDecided.open = tasteDecidedOpen;
  for (const item of [...pending, ...decided]) {
    const card = document.createElement('li');
    card.className = 'taste-card';
    card.dataset.state = item.state;
    card.dataset.proposal = item.id;
    const sentence = document.createElement('p');
    sentence.className = 'taste-sentence';
    sentence.textContent = sentenceText(item);
    card.append(sentence);
    if (item.edited) {
      const original = document.createElement('p');
      original.className = 'taste-wording muted';
      original.textContent = `Your wording. Proposed: ${item.sentence}`;
      card.append(original);
    }
    const meta = document.createElement('p');
    meta.className = 'taste-meta muted';
    meta.textContent = metaText(item);
    const heading = document.createElement('strong');
    heading.className = 'taste-notes-heading';
    heading.textContent = 'From your notes';
    const sources = document.createElement('ul');
    sources.className = 'taste-sources';
    for (const source of sourceRows(item, review.series.versions)) {
      const row = document.createElement('li');
      row.className = 'taste-source';
      const label = document.createElement('p');
      label.className = 'muted';
      label.textContent = source.location ? `${source.label}. ${source.location}` : source.label;
      const quote = document.createElement('p');
      quote.textContent = source.quote;
      row.append(label, quote);
      if (source.canGo) row.append(makeButton('Go to', '', () => goToTasteSource(source)));
      sources.append(row);
    }
    card.append(meta, heading, sources);
    const state = stateText(item);
    if (state) {
      const saved = document.createElement('p');
      saved.className = 'taste-state';
      saved.textContent = state;
      card.append(saved);
    }
    const errorNode = document.createElement('p');
    errorNode.className = 'form-error';
    errorNode.setAttribute('role', 'alert');
    errorNode.textContent = tasteErrors.get(item.id) ?? '';
    card.append(errorNode);
    const actions = document.createElement('div');
    actions.className = 'taste-actions';
    if (tasteDrafts.has(item.id) && item.state === 'pending') {
      const textarea = document.createElement('textarea');
      textarea.rows = 3;
      textarea.maxLength = 280;
      textarea.value = tasteDrafts.get(item.id);
      textarea.setAttribute('aria-label', 'Rule wording');
      textarea.addEventListener('input', () => {
        tasteDrafts.set(item.id, textarea.value);
        updateTasteControls();
      });
      const approve = makeButton('Approve this wording', 'primary', () =>
        saveTasteDecision(item, 'approve', textarea.value));
      approve.dataset.wording = 'true';
      const cancel = makeButton('Cancel', '', () => {
        tasteDrafts.delete(item.id);
        tasteErrors.delete(item.id);
        renderTaste();
      });
      actions.append(approve, cancel);
      card.append(textarea);
    } else {
      for (const {action, label} of tasteActions(item)) {
        const button = makeButton(label, action === 'approve' ? 'primary' : '', () => {
          if (action === 'edit') {
            tasteDrafts.set(item.id, item.sentence);
            renderTaste();
            el.tasteList.querySelector(`[data-proposal="${item.id}"] textarea`)?.focus();
          } else void saveTasteDecision(item, action);
        });
        actions.append(button);
      }
    }
    if (actions.childElementCount) card.append(actions);
    (item.state === 'pending' ? el.tasteList : el.tasteDecidedList).append(card);
  }
  updateTasteControls();
}

async function saveTasteDecision(item, action, sentence) {
  if (offline || review.media_changed || switchingVersion || tasteSaving.has(item.id)) return;
  tasteSaving.add(item.id);
  tasteErrors.delete(item.id);
  loadVersion += 1;
  updateTasteControls();
  renderVersions();
  updateReplyEvidenceButtons();
  renderSendBar();
  try {
    const result = await api('api/taste', {method: 'POST', body: decisionBody(item, action, sentence)});
    review.series.taste = result.taste;
    tasteDrafts.delete(item.id);
  } catch (error) {
    tasteErrors.set(item.id, errorMessage(error));
  } finally {
    tasteSaving.delete(item.id);
    renderTaste();
    renderVersions();
    updateReplyEvidenceButtons();
    renderSendBar();
  }
}

function renderRemoved() {
  updateFindingControls();
  el.removed.hidden = !review?.series?.project;
  const ranges = removed.version === selectedVersion ? (removed.ranges ?? []) : [];
  el.removedCount.textContent = String(ranges.length);
  el.removedList.replaceChildren();
  const visible = visibleRemovedRanges(ranges, el.showPauses.checked);
  el.removedEmpty.hidden = visible.length !== 0;
  el.removedEmpty.textContent = ranges.length ? 'Show pauses to see removed pauses.'
    : 'No removed parts to show.';
  for (const range of visible) {
    const row = document.createElement('li');
    row.className = 'removed-row';
    const heading = document.createElement('div');
    heading.className = 'removed-heading';
    const time = document.createElement('span');
    time.className = 'timestamp';
    time.textContent = fps ? frameTimeText(range.at_frame, fps) : secondsText(range.at_s);
    const kind = document.createElement('span');
    kind.textContent = removedKindName(range.kind);
    heading.append(time, kind);
    const text = document.createElement('p');
    text.className = 'removed-text';
    text.textContent = range.text ?? 'Removed audio';
    const actions = document.createElement('div');
    actions.className = 'removed-actions';
    const go = makeButton('Go to', '', () => fps ? seekFrame(range.at_frame) : seekTime(range.at_s));
    go.disabled = review.media_changed;
    const requested = rangeRequested(range, review.comments);
    const ask = makeButton(requested ? 'Requested' : 'Ask for it back', '', () => restoreRange(range));
    ask.disabled = requested || review.media_changed
      || restorePending.has(`${selectedVersion}:${range.id}`);
    actions.append(go, ask);
    row.append(heading, text, actions);
    el.removedList.append(row);
  }
}

function findingKey(finding) {
  return `${selectedVersion}:${review.media.sha256}:${finding.id}`;
}

function isFindingAdded(finding) {
  const id = addedFindings.get(findingKey(finding));
  return (id !== undefined && review.comments.some((comment) => comment.id === id))
    || findingAdded(finding, review.comments, review.media.duration_s);
}

function findingsBlocked() {
  return !review || offline || switchingVersion || openingReplyEvidence || composeBusy
    || roundBusy || releaseBusy || answerBusy || Boolean(editState?.busy)
    || activeWrites > 0 || restorePending.size > 0 || Boolean(review.media_changed);
}

function updateFindingControls() {
  const blocked = findingsBlocked();
  for (const {button, finding, add} of findingControls) {
    const added = Boolean(add && isFindingAdded(finding));
    if (add) button.textContent = added ? 'Added' : 'Add as my note';
    button.disabled = blocked || added;
  }
}

function seekFinding(finding) {
  if (findingsBlocked()) return;
  if (fps) seekFrame(finding.start_frame);
  else seekTime(finding.start_s);
}

function findingLabel(kind) {
  if (kind === 'behind_hidden') return 'Hidden behind speaker';
  if (kind === 'caption_contrast') return 'Hard to read';
  return kind === 'layer_check' ? 'Layer check' : findingKindLabel(kind);
}

function renderFindings() {
  const findings = review?.findings;
  el.findings.hidden = !findings?.available;
  const message = findings?.error ? "The agent's notes could not be read."
    : !findings?.available && findings?.stale?.length
      ? "The agent's notes are for an earlier render of this version." : '';
  el.findingsStatus.textContent = message;
  el.findingsStatus.hidden = !message;
  const items = findings?.available ? findings.items : [];
  el.findingsCount.textContent = String(items.length);
  el.findingsList.replaceChildren();
  findingControls = findingControls.filter(({add}) => add === undefined);
  for (const finding of items) {
    const row = document.createElement('li');
    row.className = 'finding-row';
    const heading = document.createElement('div');
    heading.className = 'finding-heading';
    const time = document.createElement('span');
    time.className = 'timestamp';
    time.textContent = spanText(finding, fps);
    const kind = document.createElement('span');
    kind.textContent = findingLabel(finding.kind);
    const severity = document.createElement('span');
    severity.textContent = `Severity: ${finding.severity}`;
    const source = document.createElement('span');
    source.textContent = findingSourceLabel(finding);
    heading.append(time, kind, severity, source);
    if (finding.kind === 'layer_check') {
      const pass = document.createElement('span');
      pass.textContent = finding.pass;
      heading.append(pass);
    }
    const reason = document.createElement('p');
    reason.className = 'finding-text';
    reason.textContent = finding.reason;
    const fix = document.createElement('p');
    fix.className = 'finding-fix';
    fix.textContent = `Suggested fix: ${finding.fix}`;
    row.append(heading, reason, fix);
    const evidenceText = findingEvidenceText(finding.evidence);
    if (evidenceText) {
      const evidence = document.createElement('p');
      evidence.className = 'finding-evidence';
      evidence.textContent = evidenceText;
      row.append(evidence);
    }
    const actions = document.createElement('div');
    actions.className = 'finding-actions';
    const go = makeButton('Go to', '', () => seekFinding(finding));
    const add = makeButton('Add as my note', '', () => addFinding(finding));
    findingControls.push({button: go, finding, add: false}, {button: add, finding, add: true});
    actions.append(go, add);
    row.append(actions);
    el.findingsList.append(row);
  }
  const skipped = (findings?.layer_checks ?? []).flatMap((layer) =>
    (layer.skipped ?? []).map((check) => `${layer.path}: ${check.pass} (${check.reason})`));
  if (skipped.length) {
    if (!layerChecksSummary) {
      layerChecksSummary = document.createElement('p');
      layerChecksSummary.className = 'finding-evidence';
      // Engineering detail stays inside the closed findings fold.
      el.findings.append(layerChecksSummary);
    }
    layerChecksSummary.textContent = `Some layer checks did not run: ${skipped.join('; ')}.`;
  } else {
    layerChecksSummary?.remove();
    layerChecksSummary = null;
  }
  const layerStale = findings?.layer_stale ?? [];
  if (layerStale.length) {
    if (!layerStaleSummary) {
      layerStaleSummary = document.createElement('p');
      layerStaleSummary.className = 'finding-evidence';
      el.findings.append(layerStaleSummary);
    }
    layerStaleSummary.textContent = `Layer check receipts do not match this render: ${layerStale.join(', ')}.`;
  } else {
    layerStaleSummary?.remove();
    layerStaleSummary = null;
  }
  updateFindingControls();
}

async function addFinding(finding) {
  if (findingsBlocked() || isFindingAdded(finding)) return;
  const version = selectedVersion;
  const key = findingKey(finding);
  const body = findingNoteBody(finding, review.media.cfr, review.media.duration_s);
  composeBusy = true;
  updateComposeSubmit();
  renderVersions();
  renderSendBar();
  try {
    const created = await api(versionUrl('api/comments', version), {method: 'POST', body});
    addedFindings.set(key, created.id);
    // Keep the saved draft visible even if the subsequent refresh cannot reconnect.
    review.comments.push(created);
    selectedId = created.id;
    renderHeader();
    renderComments();
    renderMarkers();
    renderFindings();
    await loadReview(version);
  } catch (error) {
    showStatus(errorMessage(error));
  } finally {
    composeBusy = false;
    updateComposeSubmit();
    renderVersions();
    renderSendBar();
  }
}

async function restoreRange(range) {
  const version = selectedVersion;
  const key = `${version}:${range.id}`;
  if (restorePending.has(key) || rangeRequested(range, review.comments)) return;
  restorePending.add(key);
  renderRemoved();
  try {
    await api(versionUrl('api/restore', version), {method: 'POST', body: {range: range.id}});
    if (version === selectedVersion) await loadReview(version);
  } catch (error) {
    showStatus(errorMessage(error));
  } finally {
    restorePending.delete(key);
    renderRemoved();
  }
}

function appendQuestions(row, version, id) {
  for (const question of questionsForComment(review.series?.questions, version, id)) {
    const card = document.createElement('section');
    card.className = 'question-card';
    card.setAttribute('aria-label', 'Question');
    card.addEventListener('click', (event) => event.stopPropagation());
    const label = document.createElement('strong');
    label.textContent = 'Question';
    const text = document.createElement('p');
    text.textContent = question.question;
    card.append(label, text);
    if (question.state === 'answered') {
      const saved = document.createElement('p');
      saved.className = 'question-answered';
      saved.textContent = 'Answered; this goes with your next notes.';
      card.append(saved);
    } else {
      const key = `${question.batch}:${question.comment_version}:${id}`;
      const choices = document.createElement('div');
      choices.className = 'question-options';
      const textarea = document.createElement('textarea');
      textarea.rows = 2;
      textarea.maxLength = 2000;
      textarea.value = questionDrafts.get(key) ?? '';
      textarea.setAttribute('aria-label', 'Your answer');
      textarea.placeholder = 'Your answer';
      const errorNode = document.createElement('p');
      errorNode.className = 'form-error';
      errorNode.setAttribute('role', 'alert');
      const submit = makeButton('Answer', 'primary', () =>
        saveAnswer(question, textarea.value, null, card, errorNode, key));
      submit.disabled = answerBusy || review.media_changed || !textarea.value.trim();
      textarea.disabled = answerBusy || review.media_changed;
      textarea.addEventListener('input', () => {
        questionDrafts.set(key, textarea.value);
        submit.disabled = answerBusy || review.media_changed || !textarea.value.trim();
      });
      for (const option of question.options) {
        const button = makeButton(option.label, '', () =>
          saveAnswer(question, null, option.value, card, errorNode, key));
        button.disabled = answerBusy || review.media_changed;
        choices.append(button);
      }
      const actions = document.createElement('div');
      actions.className = 'form-actions';
      actions.append(submit);
      card.append(choices, textarea, errorNode, actions);
    }
    row.append(card);
  }
}

async function saveAnswer(question, answer, option, card, errorNode, key) {
  if (answerBusy || roundBusy || switchingVersion) return;
  answerBusy = true;
  const controls = [...card.querySelectorAll('button, textarea')]
    .map((control) => ({control, disabled: control.disabled}));
  controls.forEach(({control}) => { control.disabled = true; });
  renderSendBar();
  renderVersions();
  let saved = false;
  try {
    await api(versionUrl('api/questions/answer', selectedVersion),
      {method: 'POST', body: questionAnswerBody(question, answer, option)});
    questionDrafts.delete(key);
    saved = true;
    await loadReview();
  } catch (error) {
    errorNode.textContent = errorMessage(error);
  } finally {
    answerBusy = false;
    controls.forEach(({control, disabled}) => { control.disabled = disabled; });
    if (saved) {
      renderComments();
      renderThreads();
    }
    renderSendBar();
    renderVersions();
  }
}

function renderSendBar() {
  updateReplyEvidenceButtons();
  updateFindingControls();
  const series = review?.series;
  el.sendBar.hidden = !series?.project;
  if (!series?.project) return;
  el.draftCount.textContent = draftNotesText(review.comments, series.questions);
  el.sendNotes.disabled = offline || releaseBusy || roundBusy || answerBusy || composeBusy
    || Boolean(editState) || switchingVersion || tasteSaving.size > 0
    || review.media_changed || (!review.comments.some((item) => item.send?.state === 'draft')
      && !series.questions?.some((item) => item.state === 'answered'))
    || series.rounds.some((item) => item.version === selectedVersion && item.status === 'sending');
  el.roundList.replaceChildren();
  for (const round of series.rounds) {
    const row = document.createElement('div');
    row.className = `round-row ${round.status}`;
    const text = document.createElement('span');
    text.textContent = `${round.version === selectedVersion ? '' : `Version ${round.version}: `}${roundStatusText(round)}`;
    row.append(text);
    if (round.status === 'failed') {
      const retry = makeButton('Try again', '', () => submitRound(round));
      retry.disabled = offline || releaseBusy || roundBusy || answerBusy || switchingVersion
        || composeBusy || Boolean(editState)
        || (review.media_changed && round.version === selectedVersion);
      row.append(retry);
    }
    el.roundList.append(row);
  }
  const next = sendNextText(series, review.home);
  if (el.sendNext) {
    el.sendNext.hidden = !next;
    if (next) {
      el.sendNext.className = `send-next ${next.state}`;
      el.sendNext.textContent = next.text;
    }
  }
}

async function submitRound(round = null) {
  if (offline || releaseBusy || roundBusy || answerBusy || composeBusy || editState || switchingVersion
    || !review?.series?.project) return;
  roundBusy = true;
  renderSendBar();
  renderVersions();
  const version = round?.version ?? selectedVersion;
  const route = round ? `api/rounds/${encodeURIComponent(round.id)}/retry` : 'api/rounds';
  try {
    await api(versionUrl(route, version), {method: 'POST', body: {}});
    await loadReview();
  } catch (error) {
    showStatus(errorMessage(error));
  } finally {
    roundBusy = false;
    renderSendBar();
    renderVersions();
  }
}

function commentOnMapItem(item) {
  if (!review || replyNavigationBlocked() || activeWrites || cutBusy) return;
  seekTime(item.start);
  if (!el.compose.hidden || editState) return;
  clearSelection();
  composePin = null;
  openCompose();
  setComposeTarget(item.target);
  el.composeText.value = `"${item.label}" `;
  updateComposeSubmit();
  el.composeText.focus();
}

function renderVideoMap(lanes = [], status = '') {
  el.videoMapLanes.replaceChildren();
  const count = lanes.reduce((total, lane) => total + lane.items.length, 0);
  el.videoMapToggle.textContent = status || `Video map · ${count} ${count === 1 ? 'item' : 'items'}`;
  for (const [index, lane] of lanes.entries()) {
    const row = document.createElement('div');
    row.className = 'video-map-row';
    row.dataset.lane = String(index);
    row.dataset.optionalEmpty = String(lane.items.length === 0
      && !['Story', 'Captions'].includes(lane.name));
    const name = document.createElement('span');
    name.className = 'video-map-name';
    name.textContent = lane.name;
    const track = document.createElement('div');
    track.className = 'video-map-track';
    track.setAttribute('role', 'group');
    track.setAttribute('aria-label', lane.name);
    const ranges = assignLanes(lane.items);
    const rows = ranges.reduce((count, item) => Math.max(count, item.lane + 1), 1);
    track.style.setProperty('--rows', String(rows));
    if (!ranges.length) {
      const empty = document.createElement('span');
      empty.className = 'video-map-empty';
      empty.textContent = 'No items';
      track.append(empty);
    }
    for (const item of ranges) {
      const {left, width} = mapBlockPosition(item, duration());
      const block = makeButton('', 'video-map-block', () => commentOnMapItem(item));
      const label = document.createElement('span');
      label.className = 'video-map-label';
      label.textContent = item.label;
      block.append(label);
      if (item.start === item.end) block.classList.add('video-map-point');
      const description = `${lane.name}: ${item.label}, ${secondsText(item.start)} to ${secondsText(item.end)}. Comment on ${targetName(item.target)}.`;
      block.setAttribute('aria-label', description);
      block.title = description;
      block.style.setProperty('--start', `${left}%`);
      block.style.setProperty('--width', `${width}%`);
      block.style.setProperty('--row', String(item.lane));
      track.append(block);
    }
    row.append(name, track);
    el.videoMapLanes.append(row);
  }
}

function renderMarkers() {
  el.markers.replaceChildren();
  el.ranges.replaceChildren();
  el.findingMarkers.replaceChildren();
  findingControls = findingControls.filter(({add}) => add !== undefined);
  el.scrubber.style.setProperty('--finding-height', '0px');
  const length = duration();
  if (length === null) {
    el.scrubber.style.setProperty('--lane-height', '0px');
    return;
  }
  for (const comment of sortComments(review.comments)) {
    if (comment.end_frame !== null || comment.end_s !== null) continue;
    const time = fps && comment.start_frame !== null
      ? frameMid(comment.start_frame, fps) : comment.start_s;
    const marker = makeButton('', `marker ${comment.kind}`, () => seekComment(comment));
    marker.setAttribute('aria-label', `${spanText(comment, fps)} ${comment.kind}: ${comment.text}`);
    marker.title = `${spanText(comment, fps)} · ${comment.kind}`;
    marker.style.setProperty('--position', `${Math.max(0, Math.min(100, time / length * 100))}%`);
    marker.addEventListener('pointerdown', (event) => event.stopPropagation());
    el.markers.append(marker);
  }
  const ranges = sortComments(review.comments).filter((comment) =>
    comment.end_frame !== null || comment.end_s !== null).map((comment) => ({
    comment,
    start: fps && comment.start_frame !== null
      ? frameStart(comment.start_frame, fps) : comment.start_s,
    end: fps && comment.end_frame !== null
      ? frameStart(comment.end_frame, fps) : comment.end_s,
  }));
  const placed = assignLanes(ranges);
  const count = placed.reduce((highest, range) => Math.max(highest, range.lane + 1), 0);
  el.scrubber.style.setProperty('--lane-height', `${count * 15}px`);
  for (const range of placed) {
    const {comment} = range;
    const bar = makeButton('', `range-bar ${comment.kind}`, () => seekComment(comment));
    bar.setAttribute('aria-label', `${spanText(comment, fps)} ${comment.kind}: ${comment.text}`);
    bar.title = `${spanText(comment, fps)} · ${comment.kind}`;
    bar.style.setProperty('--start', `${timelinePercent(range.start)}%`);
    bar.style.setProperty('--end', `${timelinePercent(range.end)}%`);
    bar.style.setProperty('--lane-top', `${range.lane * 15}px`);
    bar.addEventListener('pointerdown', (event) => event.stopPropagation());
    el.ranges.append(bar);
  }
  const findings = review.findings?.available ? review.findings.items : [];
  const findingRanges = [];
  for (const finding of findings) {
    const label = `${findingSourceLabel(finding)} finding · ${spanText(finding, fps)} · ${findingLabel(finding.kind)} · ${finding.severity}: ${finding.reason}`;
    const ranged = fps ? finding.end_frame !== null : finding.end_s !== null;
    const marker = makeButton('', ranged ? 'range-bar finding-range' : 'marker finding-marker',
      () => seekFinding(finding));
    marker.setAttribute('aria-label', label);
    marker.title = label;
    marker.addEventListener('pointerdown', (event) => event.stopPropagation());
    if (ranged) {
      findingRanges.push({marker,
        start: fps ? frameStart(finding.start_frame, fps) : finding.start_s,
        end: fps ? frameStart(finding.end_frame, fps) : finding.end_s});
    } else {
      const time = fps ? frameMid(finding.start_frame, fps) : finding.start_s;
      marker.style.setProperty('--position', `${timelinePercent(time)}%`);
      el.findingMarkers.append(marker);
    }
    findingControls.push({button: marker, finding});
  }
  const findingLanes = assignLanes(findingRanges);
  const findingLaneCount = findingLanes.reduce((highest, range) => Math.max(highest, range.lane + 1), 0);
  const pointHeight = findings.length > findingRanges.length ? 44 : 0;
  el.scrubber.style.setProperty('--finding-point-height', `${pointHeight}px`);
  el.scrubber.style.setProperty('--finding-height', `${pointHeight + findingLaneCount * 44}px`);
  for (const {marker, start, end, lane} of findingLanes) {
    marker.style.setProperty('--start', `${timelinePercent(start)}%`);
    marker.style.setProperty('--end', `${timelinePercent(end)}%`);
    marker.style.setProperty('--lane-top', `${lane * 44}px`);
    el.findingMarkers.append(marker);
  }
  updateFindingControls();
}

async function changeComment(id, fields) {
  try {
    await api(versionUrl(`api/comments/${encodeURIComponent(id)}`, selectedVersion),
      {method: 'PATCH', body: fields});
    await loadReview();
  } catch (error) {
    showStatus(errorMessage(error));
  }
}

async function deleteComment(id) {
  if (!confirm('Delete this comment?')) return;
  try {
    await api(versionUrl(`api/comments/${encodeURIComponent(id)}`, selectedVersion), {method: 'DELETE'});
    if (selectedId === id) selectedId = null;
    await loadReview();
  } catch (error) {
    showStatus(errorMessage(error));
  }
}

function cancelEdit() {
  editState = null;
  clearSelection();
  renderComments();
  renderSendBar();
  renderPin();
}

async function saveEdit() {
  if (!editState || editState.busy) return;
  const state = editState;
  const {id, comment, textarea, spelling, errorNode} = state;
  if (!textarea.value.trim()) {
    errorNode.textContent = 'Comment text is required.';
    return;
  }
  try {
    const fields = {text: textarea.value, request: editedRequest(comment,
      spelling.wrong.value, spelling.right.value)};
    const cut = comment.request?.type === 'cut';
    const movedPoint = state.pinTouched && state.pin && !completeSelection() && !cut;
    if (fps) Object.assign(fields, (cut ? null : completeSelection())
      ?? {start_frame: movedPoint ? currentFrame() : comment.start_frame,
        end_frame: cut ? comment.end_frame : null});
    else Object.assign(fields, (cut ? null : completeSelection())
      ?? {start_s: movedPoint ? displayedTime : comment.start_s, end_s: cut ? comment.end_s : null});
    const pinFrame = state.pinTouched ? (fps ? currentFrame() : null) : state.pin?.frame;
    const pinTime = state.pinTouched ? displayedTime : state.pin?.time_s;
    if (state.pin && !pinInSpan(fields, fps, pinFrame, pinTime)) {
      errorNode.textContent = 'Move to a frame inside the range, or remove the pin.';
      return;
    }
    if (state.pinTouched) {
      fields.pin = pinBody(state.pin, fps, fps ? currentFrame() : null, displayedTime);
    }
    state.busy = true;
    updateReplyEvidenceButtons();
    updateFindingControls();
    await api(versionUrl(`api/comments/${encodeURIComponent(id)}`, selectedVersion),
      {method: 'PATCH', body: fields});
    editState = null;
    clearSelection();
    await loadReview();
  } catch (error) {
    errorNode.textContent = errorMessage(error);
    showStatus(errorMessage(error));
  } finally {
    state.busy = false;
    updateReplyEvidenceButtons();
    updateFindingControls();
  }
}

function mountEdit(row, state) {
  row.querySelector('.comment-text').hidden = true;
  row.querySelector('.comment-actions').hidden = true;
  row.querySelector('.kind-button').disabled = true;
  row.querySelector('.target-select').disabled = true;
  const request = row.querySelector('.request-text');
  if (request) request.hidden = true;
  const box = document.createElement('div');
  box.className = 'edit-box';
  const actions = document.createElement('div');
  actions.className = 'form-actions';
  actions.append(makeButton('Cancel', '', cancelEdit), makeButton('Save', 'primary', saveEdit));
  box.append(state.spelling.box, state.pinRow, state.textarea, state.errorNode, actions);
  row.append(box);
}

function openEdit(comment, row) {
  if (editState?.id === comment.id) {
    editState.textarea.focus();
    return;
  }
  if (editState) cancelEdit();
  if (!el.compose.hidden) closeCompose();
  el.video.pause();
  const target = el.list.querySelector(`[data-comment-id="${comment.id}"]`) || row;
  const textarea = document.createElement('textarea');
  textarea.value = comment.text;
  textarea.rows = 4;
  textarea.maxLength = 5000;
  textarea.setAttribute('aria-label', 'Edit comment text');
  const errorNode = document.createElement('p');
  errorNode.className = 'form-error';
  errorNode.setAttribute('role', 'alert');
  const box = document.createElement('fieldset');
  box.className = 'spelling-fields';
  box.hidden = !spellingAvailable(comment.kind, comment.target);
  const legend = document.createElement('legend');
  legend.textContent = 'Correct spelling';
  box.append(legend);
  const inputs = ['Shown as', 'Should be'].map((labelText, index) => {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'text';
    input.maxLength = 80;
    input.id = `edit-spelling-${index}-${comment.id}`;
    input.value = comment.request?.[index === 0 ? 'from' : 'to'] ?? '';
    label.htmlFor = input.id;
    label.textContent = labelText;
    box.append(label, input);
    return input;
  });
  const spelling = {box, wrong: inputs[0], right: inputs[1]};
  const updateRequired = () => {
    const required = Boolean(spelling.wrong.value.trim() || spelling.right.value.trim());
    for (const input of inputs) input.required = required;
  };
  for (const input of inputs) input.addEventListener('input', updateRequired);
  updateRequired();
  const pinRow = document.createElement('div');
  pinRow.className = 'compose-pin';
  const pinLine = document.createElement('p');
  pinLine.className = 'pin-line';
  const pinRemove = makeButton('Remove pin', '', () => setDraftPin(null));
  pinRow.append(pinLine, pinRemove);
  editState = {id: comment.id, comment, textarea, spelling, errorNode, busy: false,
    pin: comment.pin ?? null, pinTouched: false, pinRow, pinText: pinLine, pinRemove};
  selectedId = comment.id;
  if (comment.pin) {
    if (fps) seekFrame(comment.pin.frame);
    else seekTime(comment.pin.time_s);
  }
  selection = comment.request?.type === 'cut' ? null : fps
    ? (comment.end_frame === null ? null
      : {inFrame: comment.start_frame, outFrame: comment.end_frame - 1})
    : (comment.end_s === null ? null
      : {inTime: comment.start_s, outTime: comment.end_s});
  renderSelection();
  mountEdit(target, editState);
  renderSendBar();
  renderPin();
  textarea.focus();
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

function renderComments() {
  if (!review) return;
  const comments = sortComments(review.comments);
  const editFocus = editState && [editState.textarea, editState.spelling.wrong,
    editState.spelling.right].includes(document.activeElement) ? document.activeElement : null;
  if (editState && !comments.some((comment) => comment.id === editState.id
    && !['sending', 'sent'].includes(comment.send?.state))) {
    editState = null;
    clearSelection();
  }
  el.count.textContent = String(review.comments.length);
  el.empty.hidden = review.comments.length !== 0;
  el.list.replaceChildren();
  for (const comment of comments) {
    const locked = ['sending', 'sent'].includes(comment.send?.state);
    const row = document.createElement('li');
    row.className = `comment${selectedId === comment.id ? ' selected' : ''}`;
    row.dataset.commentId = comment.id;
    const top = document.createElement('div');
    top.className = 'comment-top';
    const timestamp = makeButton(spanText(comment, fps), 'timestamp', () => seekComment(comment));
    const kind = makeButton(comment.kind === 'keep' ? 'Keep' : 'Fix',
      `kind-button ${comment.kind}`,
      () => changeComment(comment.id, {kind: comment.kind === 'keep' ? 'fix' : 'keep'}));
    kind.setAttribute('aria-label', `Switch ${comment.kind} to ${comment.kind === 'keep' ? 'fix' : 'keep'}`);
    kind.disabled = locked || comment.request?.type === 'caption_fix';
    top.append(timestamp, kind);
    const target = document.createElement('select');
    target.className = 'target-select';
    target.setAttribute('aria-label', `About for comment ${comment.id}`);
    for (const value of [null, ...Object.keys(TARGET_NAMES)]) {
      const option = document.createElement('option');
      option.value = value ?? '';
      option.textContent = targetName(value);
      target.append(option);
    }
    target.value = comment.target ?? '';
    target.disabled = locked || comment.request?.type === 'caption_fix';
    target.addEventListener('change', () => changeComment(comment.id, {target: target.value || null}));
    target.addEventListener('click', (event) => event.stopPropagation());
    const content = document.createElement('p');
    content.className = 'comment-text';
    content.textContent = comment.text;
    const actions = document.createElement('div');
    actions.className = 'comment-actions';
    if (comment.end_frame !== null || comment.end_s !== null) {
      const play = makeButton('Play range', '', () => playRange(comment));
      play.setAttribute('aria-label', 'Play this range');
      actions.append(play);
    }
    if (!locked && comment.request?.type === 'caption_fix') {
      actions.append(makeButton('Undo caption fix', '', () => undoCaptionFix(comment.id)));
    } else if (!locked) {
      actions.append(makeButton('Edit', '', () => openEdit(comment, row)),
        makeButton('Delete', '', () => deleteComment(comment.id)));
      if (['restore', 'cut'].includes(comment.request?.type)) {
        actions.append(makeButton('Clear request', '', () => changeComment(comment.id, {request: null})));
      }
    }
    const sendState = document.createElement('p');
    sendState.className = `send-state ${comment.send?.state ?? 'draft'}`;
    sendState.textContent = sendStateLabel(comment.send);
    row.append(top, target, sendState, content, actions);
    if (comment.pin) {
      const spot = document.createElement('p');
      spot.className = 'pin-text';
      spot.textContent = pinText(comment.pin);
      row.insertBefore(spot, content);
    }
    if (comment.request && comment.request.type !== 'caption_fix') {
      const spelling = document.createElement('p');
      spelling.className = `request-text ${comment.request.type}-text`;
      spelling.textContent = requestText(comment.request);
      row.insertBefore(spelling, content);
    }
    appendQuestions(row, selectedVersion, comment.id);
    row.addEventListener('click', (event) => {
      if (!event.target.closest('button, textarea, input, select, label')) seekComment(comment);
    });
    el.list.append(row);
    if (editState?.id === comment.id) mountEdit(row, editState);
  }
  if (editFocus && editState) editFocus.focus();
  renderPin();
}

function renderReview(fresh, preserveStatus = false) {
  const freshFps = fresh.media.cfr ? parseFps(fresh.media.fps) : null;
  if (fresh.media.cfr && !freshFps) throw new Error('The video frame rate could not be read.');
  const changed = review === null || fresh.media.sha256 !== review.media.sha256;
  if (fresh.series?.current !== selectedVersion || !fresh.series?.project) {
    wordsState = {available: false, version: null, words: []};
    captionsState = {available: false, version: null, phrases: []};
    closeCaptionFix(false);
    rangeWordsCache = null;
    cutError = '';
    // An unsent note belongs to the version it was written on.
    if (el.rangeNote) el.rangeNote.value = '';
  }
  review = fresh;
  selectedVersion = fresh.series?.current ?? null;
  fps = freshFps;
  if (changed) {
    selection = null;
    if (el.rangeNote) el.rangeNote.value = '';
    rangePlayback = null;
    selectedId = null;
    editState = null;
    composePin = null;
    pinDrag = null;
    el.pinLayer.hidden = true;
    el.compose.hidden = true;
    el.composeText.value = '';
    displayedTime = 0;
    beginPlayerLoad();
    el.video.src = versionUrl('media', selectedVersion, token);
  }
  if (!preserveStatus || fresh.media_changed) showStatus('');
  renderHeader();
  renderPrevious();
  renderMarkers();
  renderComments();
  renderVersions();
  renderThreads();
  renderSendBar();
  renderRemoved();
  renderTaste();
  renderFindings();
  if (el.textHook) {
    if (fresh.text_hook) renderTextHook(el.textHook, fresh.text_hook, async (body) => {
      if (offline || switchingVersion || composeBusy || roundBusy || releaseBusy || answerBusy) {
        throw new Error('Finish the current action, then choose your Text hook.');
      }
      composeBusy = true;
      renderVersions();
      renderSendBar();
      try {
        const created = await api(versionUrl('api/comments', selectedVersion), {method: 'POST', body});
        review.comments.push(created);
        review.text_hook = {...review.text_hook, chosen: body.text_hook.text, source: body.text_hook.source};
        renderComments();
        await loadReview(selectedVersion);
        showToast('Text hook saved. Send your notes when ready.');
      } finally {
        composeBusy = false;
        renderVersions();
        renderSendBar();
      }
    }, offline || Boolean(fresh.media_changed)
      || fresh.series?.current !== fresh.series?.newest);
    else el.textHook.hidden = true;
  }
  el.newComment.disabled = offline || Boolean(review.media_changed);
  schedulePoll();
  updateTimeline();
}

async function loadReview(version = selectedVersion) {
  const generation = ++loadVersion;
  try {
    const fresh = await api(versionUrl('api/review', version));
    if (generation !== loadVersion) return;
    renderReview(fresh);
    el.videoMap.hidden = !fresh.series?.project;
    if (el.videoMap.hidden) return;
    renderVideoMap([], 'Video map loading');
    const [removedResult, wordsResult, result, captionsResult] = await Promise.allSettled([
      api(versionUrl('api/removed', fresh.series.current)),
      api(versionUrl('api/words', fresh.series.current)),
      api(versionUrl('api/video-map', fresh.series.current)),
      api(versionUrl('api/captions', fresh.series.current)),
    ]);
    if (generation !== loadVersion) return;
    removed = removedResult.status === 'fulfilled' ? removedResult.value
      : {available: false, ranges: []};
    wordsState = wordsResult.status === 'fulfilled' && Array.isArray(wordsResult.value.words)
      ? wordsResult.value : {available: false, version: fresh.series.current, words: []};
    captionsState = captionsResult.status === 'fulfilled'
      && Array.isArray(captionsResult.value.phrases) ? captionsResult.value
      : {available: false, version: fresh.series.current, phrases: []};
    renderCaptionAccess();
    rangeWordsCache = null;
    renderRemoved();
    renderRangePanel();
    if (removedResult.status === 'rejected') showStatus(errorMessage(removedResult.reason));
    const lanes = result.status === 'fulfilled' && Array.isArray(result.value.lanes)
      ? result.value.lanes : null;
    renderVideoMap(lanes ?? [], lanes === null ? 'Video map unavailable' : '');
  } catch (error) {
    showStatus(errorMessage(error));
    throw error;
  }
}

function tasteWordingFocused() {
  return document.activeElement?.tagName === 'TEXTAREA'
    && Boolean(document.activeElement.closest('.taste-card'));
}

async function refreshReview() {
  if ((!offline && document.visibilityState !== 'visible') || composeBusy
    || editState?.busy || answerBusy || tasteSaving.size > 0 || refreshPending || switchingVersion
    || tasteWordingFocused() || document.activeElement?.closest('.text-hook-panel')
    || (!offline && document.activeElement?.closest('.question-card'))) return;
  refreshPending = true;
  const version = loadVersion;
  const reconnecting = offline;
  try {
    const fresh = await api(versionUrl('api/review', selectedVersion));
    if (version !== loadVersion || (!reconnecting && document.visibilityState !== 'visible')
      || composeBusy || answerBusy || tasteSaving.size > 0 || editState?.busy
      || tasteWordingFocused() || document.activeElement?.closest('.text-hook-panel')
      || (!offline && document.activeElement?.closest('.question-card'))) return;
    if (refreshStatusError) showStatus('');
    if (!reconnecting && review && fresh.text === review.text
      && fresh.media_changed === review.media_changed
      && JSON.stringify(fresh.comments) === JSON.stringify(review.comments)
      && JSON.stringify(fresh.previous) === JSON.stringify(review.previous)
      && JSON.stringify(fresh.findings) === JSON.stringify(review.findings)
      && JSON.stringify(fresh.series) === JSON.stringify(review.series)
      && JSON.stringify(fresh.home) === JSON.stringify(review.home)
      && JSON.stringify(fresh.text_hook) === JSON.stringify(review.text_hook)) return;
    renderReview(fresh, !reconnecting);
    if (reconnecting && fresh.series?.project) await loadReview(selectedVersion);
  } catch (error) {
    if (version === loadVersion && document.visibilityState === 'visible'
      && !composeBusy && !editState?.busy) {
      showStatus(errorMessage(error));
      refreshStatusError = !review?.media_changed;
    }
  } finally {
    refreshPending = false;
  }
}

function setComposeKind(kind) {
  composeKind = kind;
  el.composeFix.setAttribute('aria-pressed', String(kind === 'fix'));
  el.composeKeep.setAttribute('aria-pressed', String(kind === 'keep'));
  updateComposeSpelling();
}

function setComposeTarget(target) {
  composeTarget = target;
  for (const chip of composeTargetButtons) {
    chip.button.setAttribute('aria-pressed', String(chip.target === target));
  }
  updateComposeSpelling();
}

function updateComposeSpelling() {
  const available = spellingAvailable(composeKind, composeTarget);
  el.composeSpelling.hidden = !available;
  const required = available && Boolean(el.spellingFrom.value.trim() || el.spellingTo.value.trim());
  for (const input of [el.spellingFrom, el.spellingTo]) {
    input.disabled = !available;
    input.required = required;
  }
}

function updateComposeSubmit() {
  el.composeSubmit.disabled = offline || composeBusy || !el.composeText.value.trim();
  renderRangePanel();
}

function openCompose({pin = null, focus = true} = {}) {
  if (!review || replyNavigationBlocked() || activeWrites || cutBusy) return;
  closeCaptionFix(false);
  if (!el.compose.hidden) {
    if (focus) el.composeText.focus();
    return;
  }
  if (editState) cancelEdit();
  el.video.pause();
  composePin = pin;
  el.compose.hidden = false;
  el.composeText.value = '';
  el.composeError.textContent = '';
  el.spellingFrom.value = '';
  el.spellingTo.value = '';
  setComposeKind('fix');
  setComposeTarget(null);
  updateTimeline();
  updateComposeSubmit();
  if (focus) el.composeText.focus();
}

function closeCompose() {
  el.compose.hidden = true;
  composePin = null;
  el.composeError.textContent = '';
  el.composeText.value = '';
  updateComposeSubmit();
  el.newComment.focus();
  renderPin();
}

async function submitCompose() {
  if (offline || composeBusy || !el.composeText.value.trim() || !review) return;
  const fields = {kind: composeKind, target: composeTarget, text: el.composeText.value};
  try {
    const request = spellingRequest(composeKind, composeTarget,
      el.spellingFrom.value, el.spellingTo.value);
    if (request !== null) fields.request = request;
  } catch (error) {
    el.composeError.textContent = errorMessage(error);
    return;
  }
  const marked = completeSelection();
  if (fps) Object.assign(fields, marked ?? {start_frame: currentFrame()});
  else Object.assign(fields, marked ?? {start_s: displayedTime});
  if (composePin) {
    if (!pinInSpan(fields, fps, fps ? currentFrame() : null, displayedTime)) {
      el.composeError.textContent = 'Move to a frame inside the range, or remove the pin.';
      return;
    }
    fields.pin = pinBody(composePin, fps, fps ? currentFrame() : null, displayedTime);
  }
  composeBusy = true;
  updateComposeSubmit();
  renderSendBar();
  try {
    const created = await api(versionUrl('api/comments', selectedVersion),
      {method: 'POST', body: fields});
    selectedId = created.id;
    closeCompose();
    clearSelection();
    await loadReview();
  } catch (error) {
    el.composeError.textContent = errorMessage(error);
    showStatus(errorMessage(error));
  } finally {
    composeBusy = false;
    updateComposeSubmit();
    renderVersions();
    renderSendBar();
  }
}

function scrubFraction(fraction) {
  const length = duration();
  if (length === null) return;
  const time = Math.max(0, Math.min(1, fraction)) * length;
  if (fps) seekFrame(Math.min(lastFrame(), frameAt(time, fps)));
  else seekTime(time);
}

function pointerFraction(event) {
  const rect = el.scrubber.getBoundingClientRect();
  return Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
}

function markAtPointer(edge, event, clampEdge = false) {
  const length = duration();
  if (length === null) return;
  cutError = '';
  const time = pointerFraction(event) * length;
  selection ||= {};
  if (fps) {
    let frame = Math.min(lastFrame(), frameAt(time, fps));
    if (clampEdge && edge === 'in') frame = Math.min(frame, selection.outFrame);
    if (clampEdge && edge === 'out') frame = Math.max(frame, selection.inFrame);
    selection[edge === 'in' ? 'inFrame' : 'outFrame'] = frame;
  } else {
    let point = time;
    if (clampEdge && edge === 'in') point = Math.min(point, selection.outTime);
    if (clampEdge && edge === 'out') point = Math.max(point, selection.inTime);
    selection[edge === 'in' ? 'inTime' : 'outTime'] = point;
  }
  renderSelection();
}

function nudgeSelectionEdge(edge, direction) {
  canonicalizeSelection();
  if (!completeSelection()) return;
  if (fps) {
    const name = edge === 'in' ? 'inFrame' : 'outFrame';
    const other = edge === 'in' ? selection.outFrame : selection.inFrame;
    const moved = Math.max(0, Math.min(lastFrame(), selection[name] + direction));
    selection[name] = edge === 'in' ? Math.min(moved, other) : Math.max(moved, other);
    seekFrame(selection[name]);
  } else {
    const name = edge === 'in' ? 'inTime' : 'outTime';
    const other = edge === 'in' ? selection.outTime : selection.inTime;
    const moved = Math.max(0, Math.min(duration() ?? Infinity, selection[name] + direction / 30));
    selection[name] = edge === 'in' ? Math.min(moved, other) : Math.max(moved, other);
    seekTime(selection[name]);
  }
  renderSelection();
}

function updatePointerDrag(event) {
  if (!dragPointer || dragPointer.id !== event.pointerId) return;
  if (dragPointer.kind === 'scrub') {
    queueScrub(event);
  } else if (dragPointer.kind === 'select') {
    markAtPointer('out', event);
  } else {
    const edge = dragPointer.kind === 'edge-start' ? 'in' : 'out';
    markAtPointer(edge, event, true);
    if (fps) seekFrame(selection[edge === 'in' ? 'inFrame' : 'outFrame']);
    else seekTime(selection[edge === 'in' ? 'inTime' : 'outTime']);
  }
}

function queueScrub(event) {
  dragFraction = pointerFraction(event);
  if (dragFrame !== null) return;
  dragFrame = requestAnimationFrame(() => {
    dragFrame = null;
    scrubFraction(dragFraction);
  });
}

function finishScrub(event) {
  if (dragPointer?.id !== event.pointerId) return;
  updatePointerDrag(event);
  if (dragPointer.kind === 'select') {
    canonicalizeSelection();
    renderSelection();
  }
  if (el.scrubber.hasPointerCapture(event.pointerId)) el.scrubber.releasePointerCapture(event.pointerId);
  dragPointer = null;
}

function handleKey(event) {
  if (captionEdit) {
    if (event.code === 'Escape') { event.preventDefault(); closeCaptionFix(); }
    else if (event.code === 'Enter' && event.target === el.captionTo) {
      event.preventDefault();
      saveCaptionFix();
    }
    return;
  }
  if (event.target instanceof Element && event.target.closest('.taste-card')) return;
  if (event.metaKey || event.ctrlKey) return;
  if (event.code === 'Escape' && (!el.compose.hidden || editState)) {
    event.preventDefault();
    if (editState) cancelEdit();
    else closeCompose();
    return;
  }
  const typing = event.target instanceof Element
    && event.target.closest('textarea, input, select, [contenteditable="true"]');
  if (typing) {
    if (event.altKey && (event.code === 'KeyI' || event.code === 'KeyO')
      && (event.target === el.composeText || event.target === editState?.textarea)) {
      event.preventDefault();
      markSelection(event.code === 'KeyI' ? 'in' : 'out');
    } else if (event.altKey && (event.code === 'ArrowLeft' || event.code === 'ArrowRight')) {
      event.preventDefault();
      stepFrame(event.code === 'ArrowLeft' ? -1 : 1);
    } else if (event.altKey && event.code === 'KeyL' && event.target === el.composeText) {
      event.preventDefault();
      setComposeKind(composeKind === 'fix' ? 'keep' : 'fix');
    } else if (event.code === 'Enter' || event.code === 'NumpadEnter') {
      if (event.shiftKey) return;
      event.preventDefault();
      if (editState) saveEdit();
      else if (!el.compose.hidden) submitCompose();
    }
    return;
  }
  if (!el.compose.hidden || editState) return;
  if (event.altKey) return;
  if (event.code === 'Space') {
    if (event.target instanceof Element
      && event.target.closest('button, a, [role="button"]')) return;
    event.preventDefault();
    togglePlayback();
  } else if (event.code === 'KeyC') {
    event.preventDefault();
    openCompose();
  } else if (event.code === 'KeyI' || event.code === 'KeyO') {
    event.preventDefault();
    markSelection(event.code === 'KeyI' ? 'in' : 'out');
  } else if (event.code === 'KeyX') {
    event.preventDefault();
    clearSelection();
  } else if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') {
    event.preventDefault();
    stepFrame(event.code === 'ArrowLeft' ? -1 : 1);
  }
}

function wireEvents() {
  el.videoMap.setAttribute('data-expanded', 'false');
  el.videoMapToggle.setAttribute('aria-expanded', 'false');
  el.videoMapToggle.addEventListener('click', () => {
    const expanded = el.videoMap.getAttribute('data-expanded') !== 'true';
    el.videoMap.setAttribute('data-expanded', String(expanded));
    el.videoMapToggle.setAttribute('aria-expanded', String(expanded));
  });
  el.video.addEventListener('timeupdate', () => {
    if (!('requestVideoFrameCallback' in el.video)) {
      setDisplayedTime(el.video.currentTime);
      checkRangeStop(el.video.currentTime);
    }
  });
  el.video.addEventListener('seeked', () => {
    manualSeek = false;
    if (!('requestVideoFrameCallback' in el.video)) setDisplayedTime(el.video.currentTime);
    startRangePlayback();
    scheduleMatch();
  });
  el.video.addEventListener('pause', () => {
    el.playPause.textContent = 'Play';
    if (!rangePlayback?.starting) rangePlayback = null;
    snapPausedFrame();
    scheduleMatch();
    renderPin();
  });
  el.video.addEventListener('play', () => {
    manualSeek = false;
    replyEvidenceTime = null;
    if (rangePlayback) rangePlayback.starting = false;
    el.playPause.textContent = 'Pause';
    renderPin();
  });
  el.video.addEventListener('loadedmetadata', () => {
    playerMetadataLoaded = true;
    clearTimeout(metadataTimer);
    if (playerResumeTime !== null) {
      el.video.currentTime = Math.min(playerResumeTime, duration() ?? playerResumeTime);
      playerResumeTime = null;
    }
    fitVideo(el.video);
    renderHeader();
    renderMarkers();
    updateTimeline();
  });
  el.video.addEventListener('loadeddata', () => {
    clearPlayerDeadlines();
    playerGeneration += 1;
    playerRecoveryPending = false;
    playerMessage = '';
    renderPlayerRecovery();
  });
  el.video.addEventListener('error', () => recoverPlayer('error'));
  el.playerRetry.addEventListener('click', () => retryPlayer());
  el.offlinePlayerRetry.addEventListener('click', () => retryPlayer());
  if ('requestVideoFrameCallback' in el.video) {
    const framePresented = (_now, metadata) => {
      setDisplayedTime(metadata.mediaTime);
      checkRangeStop(metadata.mediaTime);
      el.video.requestVideoFrameCallback(framePresented);
    };
    el.video.requestVideoFrameCallback(framePresented);
  }
  el.video.addEventListener('click', placePin);
  el.fixCaption?.addEventListener('click', () => openCaptionFix());
  el.captionCancel?.addEventListener('click', () => closeCaptionFix());
  el.captionSave?.addEventListener('click', saveCaptionFix);
  el.video.addEventListener('pointerdown', () => { ignorePinClick = false; });
  el.composePinRemove.addEventListener('click', () => setDraftPin(null));
  el.pinMarker.addEventListener('click', (event) => {
    event.stopPropagation();
    ignorePinClick = false;
  });
  el.pinMarker.addEventListener('pointerdown', (event) => {
    event.stopPropagation();
    if (!pinMode() || event.button !== 0) return;
    event.preventDefault();
    pinDrag = {id: event.pointerId, x: event.clientX, y: event.clientY, moved: false};
    el.pinMarker.setPointerCapture(event.pointerId);
  });
  el.pinMarker.addEventListener('pointermove', movePin);
  el.pinMarker.addEventListener('pointerup', (event) => {
    event.stopPropagation();
    if (pinDrag?.id !== event.pointerId) return;
    movePin(event);
    ignorePinClick = pinDrag.moved;
    pinDrag = null;
    if (el.pinMarker.hasPointerCapture(event.pointerId)) el.pinMarker.releasePointerCapture(event.pointerId);
  });
  for (const type of ['pointercancel', 'lostpointercapture']) {
    el.pinMarker.addEventListener(type, () => { pinDrag = null; });
  }
  el.pinMarker.addEventListener('keydown', (event) => {
    const move = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[event.code];
    if (!move) return;
    event.preventDefault();
    event.stopPropagation();
    const pin = editState ? editState.pin : composePin;
    if (!pinMode() || !pin) return;
    const step = event.shiftKey ? 0.05 : 0.01;
    const clamp = (value) => Math.round(Math.max(0, Math.min(1, value)) * 10000) / 10000;
    setDraftPin({x: clamp(pin.x + move[0] * step), y: clamp(pin.y + move[1] * step)});
  });
  if (typeof ResizeObserver !== 'undefined') {
    const observer = new ResizeObserver(renderPin);
    observer.observe(el.video);
  } else window.addEventListener('resize', renderPin);
  el.video.addEventListener('emptied', () => { el.pinLayer.hidden = true; });
  el.compareToggle.addEventListener('click', () => setCompareOpen(!compareOpen));
  el.compareVersion.addEventListener('change', () => {
    setCompareOpen(true, Number(el.compareVersion.value));
    scheduleMatch();
  });
  el.matchMoment.addEventListener('click', () => matchMoment());
  el.matchReverse.addEventListener('click', () => matchMoment(true));
  el.compareVideo.addEventListener('resize', () => fitVideo(el.compareVideo));
  el.compareVideo.addEventListener('error', () => {
    el.mapStatus.textContent = 'Could not play this version. Check that its media has not changed.';
  });
  el.playPause.addEventListener('click', togglePlayback);
  el.previousFrame.addEventListener('click', () => stepFrame(-1));
  el.nextFrame.addEventListener('click', () => stepFrame(1));
  el.newComment.addEventListener('click', () => openCompose());
  el.markStart.addEventListener('click', () => markSelection('in'));
  el.markEnd.addEventListener('click', () => markSelection('out'));
  el.rangePlay.addEventListener('click', () => {
    const marked = selectedRange();
    if (marked && !offline && !review.media_changed) playRange(marked);
  });
  el.rangeCut.addEventListener('click', cutSelection);
  el.rangeChange?.addEventListener('click', () => commentOnRange('fix'));
  el.rangeKeep?.addEventListener('click', () => commentOnRange('keep'));
  el.findings?.addEventListener?.('toggle', () => document.body?.classList?.toggle('show-flags', Boolean(el.findings.open)));
  el.composeFix.addEventListener('click', () => setComposeKind('fix'));
  el.composeKeep.addEventListener('click', () => setComposeKind('keep'));
  for (const chip of composeTargetButtons) {
    chip.button.addEventListener('click', () =>
      setComposeTarget(composeTarget === chip.target ? null : chip.target));
  }
  el.composeText.addEventListener('input', updateComposeSubmit);
  el.spellingFrom.addEventListener('input', updateComposeSpelling);
  el.spellingTo.addEventListener('input', updateComposeSpelling);
  el.composeSubmit.addEventListener('click', submitCompose);
  el.composeCancel.addEventListener('click', closeCompose);
  el.composeClearRange.addEventListener('click', clearSelection);
  el.selectionClear.addEventListener('click', clearSelection);
  el.copy.addEventListener('click', () => {
    if (review?.comments.length) copyText(review.text, review.comments.length);
  });
  el.previousCopy.addEventListener('click', () => {
    if (review?.previous) copyText(review.previous.text, review.previous.count);
  });
  el.previousView.addEventListener('click', () => {
    el.previousText.hidden = !el.previousText.hidden;
    el.previousView.textContent = el.previousText.hidden ? 'View' : 'Hide';
    el.previousView.setAttribute('aria-expanded', String(!el.previousText.hidden));
  });
  el.copyFallbackClose.addEventListener('click', () => el.copyFallback.close());
  el.scrubber.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || event.target.closest('.marker, .range-bar')) return;
    event.preventDefault();
    const handle = event.target.closest('.selection-handle');
    const kind = handle === el.selectionStart ? 'edge-start'
      : handle === el.selectionEnd ? 'edge-end' : event.shiftKey ? 'select' : 'scrub';
    if (handle) canonicalizeSelection();
    dragPointer = {id: event.pointerId, kind};
    el.scrubber.setPointerCapture(event.pointerId);
    if (kind === 'select') {
      selection = null;
      markAtPointer('in', event);
      markAtPointer('out', event);
    } else updatePointerDrag(event);
  });
  el.scrubber.addEventListener('pointermove', (event) => {
    updatePointerDrag(event);
  });
  el.scrubber.addEventListener('pointerup', finishScrub);
  el.scrubber.addEventListener('pointercancel', (event) => {
    if (dragPointer?.id === event.pointerId) {
      if (dragPointer.kind === 'select') canonicalizeSelection();
      dragPointer = null;
      renderSelection();
    }
  });
  for (const [handle, edge] of [[el.selectionStart, 'in'], [el.selectionEnd, 'out']]) {
    handle.addEventListener('keydown', (event) => {
      if (event.code !== 'ArrowLeft' && event.code !== 'ArrowRight') return;
      event.preventDefault();
      event.stopPropagation();
      nudgeSelectionEdge(edge, event.code === 'ArrowLeft' ? -1 : 1);
    });
  }
  el.scrubber.addEventListener('keydown', (event) => {
    if (event.code === 'Home') { event.preventDefault(); fps ? seekFrame(0) : seekTime(0); }
    if (event.code === 'End') {
      event.preventDefault();
      if (duration() !== null) fps ? seekFrame(lastFrame()) : seekTime(duration());
    }
  });
  document.addEventListener('keydown', handleKey);
  el.sendNotes.addEventListener('click', () => submitRound());
  el.rollback.addEventListener('click', () => releaseVersion('rollback'));
  el.approve.addEventListener('click', () => releaseVersion('approve'));
  el.makeFinal?.addEventListener('click', requestFinal);
  for (const link of [el.downloadVideo, el.downloadReport]) {
    link.addEventListener('click', (event) => { if (offline) event.preventDefault(); });
  }
  el.showPauses.addEventListener('change', renderRemoved);
  el.tasteDecided.addEventListener('toggle', () => {
    tasteDecidedOpen = el.tasteDecided.open;
  });
  window.addEventListener('focus', refreshReview);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') refreshReview();
  });
}

async function start() {
  if (!token) {
    el.name.textContent = 'Open this page from the link yap-studio printed.';
    showStatus('Open this page from the link yap-studio printed.');
    return;
  }
  wireEvents();
  try {
    await loadReview();
  } catch { /* loadReview already displayed the API error. */ }
}

start();
