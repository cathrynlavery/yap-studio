// Yap Studio pilot 5: six looping card demos. Each one steps through phases in time with its caption,
// so the effect lands on the spoken word. Demos only run while they are on screen.
(() => {
  "use strict";

  const { byId, wear, esc, reduced } = window.Yap;

  // [ms, phase, caption]. *word* is the highlighted word. The last entry's ms is the loop length.
  const TIMELINES = {
    behind: [[0, "a", "You only need"], [900, "b", "*one take.*"], [2700, "b", "Say it like"], [3500, "b", "you *mean* it."], [5400]],
    count: [[0, "a", "This month I posted"], [1100, "b", "*twelve* videos."], [3000, "b", "From my *sofa.*"], [5200]],
    list: [[0, "a", "Here's the whole edit."], [1100, "b", "Do *one* take."], [2100, "c", "Cut the *dead air.*"], [3100, "d", "Then *post* it."], [5000]],
    wipe: [[0, "a", "My *raw* take,"], [1400, "b", "and the *finished* one."], [4600]],
    shoulder: [[0, "a", "Here's what you get"], [1000, "b", "notes on *your* phone,"], [2600, "b", "with a *reply* each."], [4800]],
    cutaway: [[0, "a", "First I pick"], [1000, "b", "my *caption* style,"], [3400, "c", "then I *talk.*"], [5000]],
    screen: [[0, "a", "Let me show you"], [1000, "b", "*my* website,"], [2900, "c", "and *this* page."], [5200]],
  };

  const capHtml = (text) => text.split(" ").map((w) => {
    const on = /^\*.*\*[.,]?$/.test(w);
    return `<span class="w${on ? " on" : ""}">${esc(w.replace(/\*/g, ""))}</span>`;
  }).join(" ");

  function countUp(el) {
    const end = Number(el.dataset.count);
    if (reduced) { el.textContent = String(end); return; }
    const t0 = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - t0) / 1100);
      el.textContent = String(Math.round(end * (1 - Math.pow(1 - k, 3))));
      if (k < 1 && el.isConnected) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function demo(fx) {
    const steps = TIMELINES[fx.dataset.fx];
    const cap = fx.querySelector(".ph-cap");
    const num = fx.querySelector("[data-count]");
    wear(fx, byId[fx.dataset.preset]);
    let timers = [];
    let running = false;

    const show = ([, phase, text]) => {
      if (phase !== fx.dataset.phase && phase === "b" && num) countUp(num);
      fx.dataset.phase = phase;
      cap.innerHTML = capHtml(text);
    };
    function loop() {
      timers.forEach(clearTimeout);
      timers = [];
      if (num) num.textContent = "0";
      const length = steps[steps.length - 1][0];
      steps.slice(0, -1).forEach((s) => timers.push(setTimeout(() => show(s), s[0])));
      timers.push(setTimeout(() => running && loop(), length));
    }
    return {
      start() { if (running) return; running = true; loop(); },
      stop() { running = false; timers.forEach(clearTimeout); timers = []; },
      still() { show(steps[steps.length - 2]); if (num) num.textContent = num.dataset.count; },
    };
  }

  const demos = new Map();
  for (const fx of document.querySelectorAll(".fx[data-fx]")) demos.set(fx, demo(fx));

  if (reduced || !("IntersectionObserver" in window)) {
    demos.forEach((d) => d.still());
    return;
  }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const d = demos.get(e.target);
      if (e.isIntersecting) d.start(); else d.stop();
    }
  }, { threshold: 0.35 });
  demos.forEach((_, fx) => { fx.dataset.phase = "a"; io.observe(fx); });
})();
