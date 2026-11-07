// Yap Studio pilot 5 hero reel: three finished videos, each cut a different way.
// One clock drives everything, so the reel pauses cleanly when it is off screen or the tab is hidden.
(() => {
  "use strict";

  const reel = document.getElementById("reel");
  if (!reel) return;
  const { byId, wear, esc } = window.Yap;
  const motion = document.documentElement.classList.contains("motion");
  const $ = (sel, el = reel) => el.querySelector(sel);
  const $$ = (sel, el = reel) => Array.from(el.querySelectorAll(sel));

  // [ms, phase, caption]. *word* is highlighted. The last entry's ms is the scene length.
  const SCENES = [
    { steps: [[0, "a", "You only need"], [900, "b", "*one take.*"], [2600, "c", "I posted *twelve*"], [3700, "c", "videos this *month.*"], [6200]],
      labels: ["Behind your head", "Every word captioned", "Counts up", "Restarts cut"],
      note: { kind: "fix", t: "0:01.2", text: "Make “one take” bigger.", reply: "Done in version 2." } },
    { steps: [[0, "a", "It's all on"], [800, "b", "*my* website."], [2500, "c", "Do *one* take,"], [3500, "d", "cut the *dead air,*"], [4500, "e", "then *post* it."], [6600]],
      labels: ["Your B-roll", "Card on the word", "Ticks as you say it", "Music under you"],
      note: { kind: "keep", t: "0:00.8", text: "Love the website cut.", reply: "Kept, and added to your taste." } },
    { steps: [[0, "a", "Let me show you"], [1000, "b", "*my* style page."], [3000, "c", "Pick your look *once.*"], [5800]],
      labels: ["Something behind you", "Clear of your face", "Your caption style", "Checked first"],
      note: { kind: "fix", t: "0:02.4", text: "Shrink me a bit more here.", reply: "Done in version 2." } },
  ];

  const scenes = $$(".scene");
  const bars = $$(".sc-bars i");
  const tags = $$(".callout .tag");
  const picks = $$(".reel-pick button");
  const badge = $(".ph-badge");
  const note = $(".note-float");
  const capHtml = (text) => text.split(" ").map((w) => `<span class="w${/^\*.*\*[.,]?$/.test(w) ? " on" : ""}">${esc(w.replace(/\*/g, ""))}</span>`).join(" ");
  scenes.forEach((s) => wear(s, byId[s.dataset.preset]));

  function countUp(el) {
    const end = Number(el.dataset.count);
    if (!motion) { el.textContent = String(end); return; }
    const t0 = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - t0) / 1100);
      el.textContent = String(Math.round(end * (1 - Math.pow(1 - k, 3))));
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  let index = 0, stepAt = 0, elapsed = 0, last = 0, started = false, visible = true;

  function setStep(i, k) {
    const [, phase, text] = SCENES[i].steps[k];
    const scene = scenes[i];
    const num = scene.querySelector("[data-count]");
    if (num && phase === "c" && scene.dataset.phase !== "c") countUp(num);
    scene.dataset.phase = phase;
    scene.querySelector(".ph-cap").innerHTML = capHtml(text);
  }

  function enter(i, fromUser) {
    index = (i + SCENES.length) % SCENES.length;
    const data = SCENES[index];
    scenes.forEach((s, k) => s.classList.toggle("on", k === index));
    picks.forEach((b, k) => b.setAttribute("aria-pressed", String(k === index)));
    bars.forEach((b, k) => b.style.setProperty("--p", k < index ? "1" : "0"));
    const num = scenes[index].querySelector("[data-count]");
    if (num) num.textContent = "0";
    tags.forEach((t, k) => { t.textContent = data.labels[k]; t.classList.remove("swap"); void t.offsetWidth; t.classList.add("swap"); t.style.animationDelay = `${k * 90}ms`; });
    note.querySelector(".pill").className = `pill ${data.note.kind}`;
    note.querySelector(".pill").textContent = data.note.kind === "keep" ? "Keep" : "Fix";
    note.querySelector(".nt-time").textContent = data.note.t;
    note.querySelector(".nt-text").textContent = data.note.text;
    note.querySelector(".reply").textContent = data.note.reply;
    reel.classList.remove("is-replied");
    badge.hidden = true;
    if (fromUser !== "first" && motion) { reel.classList.remove("is-switch"); void reel.offsetWidth; reel.classList.add("is-switch"); }
    stepAt = 0;
    elapsed = 0;
    setStep(index, 0);
  }

  function tick(now) {
    const dt = last ? Math.min(100, now - last) : 0;
    last = now;
    if (started && visible && !document.hidden) {
      elapsed += dt;
      const steps = SCENES[index].steps;
      const length = steps[steps.length - 1][0];
      while (stepAt + 1 < steps.length - 1 && elapsed >= steps[stepAt + 1][0]) setStep(index, ++stepAt);
      bars[index].style.setProperty("--p", String(Math.min(1, elapsed / length)));
      if (elapsed > length * 0.45 && badge.hidden) badge.hidden = false;
      if (elapsed > length * 0.3) reel.classList.add("is-noted");
      if (elapsed > length * 0.45) reel.classList.add("is-replied");
      if (elapsed >= length) enter(index + 1);
    }
    requestAnimationFrame(tick);
  }

  picks.forEach((b) => b.addEventListener("click", () => {
    enter(Number(b.dataset.i), true);
    if (!motion) showFinal();
  }));

  // Without motion: each version shows its finished state, and the picker switches between them.
  function showFinal() {
    const steps = SCENES[index].steps;
    setStep(index, steps.length - 2);
    badge.hidden = false;
    reel.classList.add("is-edited", "is-noted", "is-replied");
  }

  enter(0, "first");
  if (!motion) { showFinal(); return; }

  new IntersectionObserver((entries) => { visible = entries.some((e) => e.isIntersecting); }, { threshold: 0.2 }).observe(reel);
  // The first time, the raw take grades into the finished video before the reel starts.
  reel.classList.add("is-raw");
  setTimeout(() => {
    reel.classList.add("is-grading", "is-edited");
    reel.classList.remove("is-raw");
    setTimeout(() => { started = true; }, 500);
  }, 1500);
  requestAnimationFrame(tick);
})();
