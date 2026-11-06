// Yap Studio pilot 5: a sample transcript that cleans itself. Each um, restart and long pause gets a tag,
// then folds out of the sentence and out of the waveform below. One pause is kept, and says why.
(() => {
  "use strict";

  const script = document.getElementById("script");
  const wave = document.getElementById("wave");
  if (!script || !wave) return;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // kind: word, cut (removed, with a tag), pause (removed dead air), keep (a pause that stays).
  const TOKENS = [
    { k: "cut", t: "So, um,", tag: "um" },
    { k: "cut", t: "I do one, wait, let me start again.", tag: "restart" },
    { k: "word", t: "I do one take." },
    { k: "pause", t: "2.1s", tag: "dead air" },
    { k: "word", t: "Then I" },
    { k: "cut", t: "uh,", tag: "uh" },
    { k: "word", t: "cut all the dead air." },
    { k: "keep", t: "0.6s", tag: "kept: the beat before the punchline" },
    { k: "word", t: "And that's the edit." },
  ];

  // Bars per token, roughly in proportion to how long each piece runs.
  const barsFor = (tok) => (tok.k === "pause" ? 9 : tok.k === "keep" ? 3 : Math.max(3, Math.round(tok.t.length * 0.55)));
  let seed = 7;
  const rnd = () => ((seed = (seed * 9301 + 49297) % 233280) / 233280);

  function render() {
    script.innerHTML = TOKENS.map((tok, i) => {
      const tag = tok.tag ? `<span class="tag${tok.k === "keep" ? " keep-tag" : ""}">${esc(tok.tag)}</span>` : "";
      if (tok.k === "word") return `<span class="tok" data-i="${i}">${esc(tok.t)}</span>`;
      if (tok.k === "pause" || tok.k === "keep") return `<span class="tok ${tok.k === "pause" ? "cut" : ""}" data-i="${i}">${tag}<span class="pause">${esc(tok.t)}</span></span>`;
      return `<span class="tok cut" data-i="${i}">${tag}${esc(tok.t)}</span>`;
    }).join("");
    seed = 7;
    wave.innerHTML = TOKENS.map((tok, i) => Array.from({ length: barsFor(tok) }, () => {
      const quiet = tok.k === "pause" || tok.k === "keep";
      const h = quiet ? 6 + rnd() * 8 : 22 + rnd() * 70;
      const cls = tok.k === "cut" || tok.k === "pause" ? "cutbar" : tok.k === "keep" ? "keepbar" : "";
      return `<i class="${cls}" data-i="${i}" style="height:${h.toFixed(0)}%"></i>`;
    }).join("")).join("");
  }

  let timers = [];
  const later = (ms, fn) => timers.push(setTimeout(fn, reduced ? 0 : ms));
  function play() {
    timers.forEach(clearTimeout);
    timers = [];
    render();
    const removable = TOKENS.map((tok, i) => [tok, i]).filter(([tok]) => tok.k !== "word");
    removable.forEach(([, i], n) => later(500 + n * 650, () => script.querySelector(`.tok[data-i="${i}"]`).classList.add("flag")));
    const foldAt = 500 + removable.length * 650 + 700;
    removable.filter(([tok]) => tok.k !== "keep").forEach(([, i], n) => later(foldAt + n * 160, () => {
      script.querySelector(`.tok[data-i="${i}"]`).classList.add("gone");
      wave.querySelectorAll(`i[data-i="${i}"]`).forEach((b) => b.classList.add("gone"));
    }));
  }

  document.getElementById("replay").addEventListener("click", play);
  render();
  // Inside the "How it works" slider, the demo plays each time its step comes up.
  const slide = script.closest(".slide");
  if (slide) {
    document.addEventListener("yap:slide", (e) => { if (e.detail === slide.id) play(); });
    if (reduced) play();
    return;
  }
  if (reduced || !("IntersectionObserver" in window)) { play(); return; }
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) { io.disconnect(); play(); }
  }, { threshold: 0.5 });
  io.observe(script);
})();
