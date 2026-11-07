// Yap Studio pilot 5: the tidy review screen. A note moves the playhead to its moment and lights its marker;
// Copy feedback puts the notes on the clipboard in the same plain format the real review page uses.
(() => {
  "use strict";

  const rv = document.getElementById("rv");
  if (!rv) return;
  const LENGTH = 12;
  const notes = Array.from(rv.querySelectorAll(".rv-notes li"));
  const marks = Array.from(rv.querySelectorAll(".rv-mark"));
  const head = document.getElementById("rv-head");
  const time = document.getElementById("rv-time");
  const toast = document.getElementById("rv-toast");
  const copy = document.getElementById("rv-copy");
  const FEEDBACK = [
    "Feedback on morning-routine-take.mp4 (0:12.0, 30 fps)",
    "3 comments: 2 fixes, 1 keep",
    "",
    "0:00.8 [Keep: story] Great opening. Keep this energy.",
    "0:03.2-0:05.4 [Fix: caption] Captions a bit smaller through here.",
    "0:07.6 [Fix: graphic] No card here, just me talking.",
  ].join("\n");

  function focusNote(li) {
    const n = li.dataset.n;
    const t = Number(li.dataset.t);
    notes.forEach((x) => x.classList.toggle("on", x === li));
    marks.forEach((m) => m.classList.toggle("on", m.dataset.n === n));
    head.style.left = `${(t / LENGTH) * 100}%`;
    time.textContent = `0:${t.toFixed(1).padStart(4, "0")}`;
  }
  for (const li of notes) {
    li.addEventListener("mouseenter", () => focusNote(li));
    li.addEventListener("focus", () => focusNote(li));
    li.addEventListener("click", () => focusNote(li));
  }

  let timer = 0;
  function say(text, ok) {
    toast.textContent = text;
    toast.classList.add("on");
    copy.classList.toggle("is-done", ok);
    copy.querySelector("span").textContent = ok ? "Copied" : "Copy feedback";
    clearTimeout(timer);
    timer = setTimeout(() => {
      toast.classList.remove("on");
      copy.classList.remove("is-done");
      copy.querySelector("span").textContent = "Copy feedback";
    }, 2200);
  }
  copy.addEventListener("click", () => {
    if (!navigator.clipboard?.writeText) { say("This browser blocked copying.", false); return; }
    navigator.clipboard.writeText(FEEDBACK).then(() => say("Copied 3 comments", true), () => say("This browser blocked copying.", false));
  });
})();
