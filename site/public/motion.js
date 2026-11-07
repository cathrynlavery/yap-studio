// Yap Studio pilot 5 motion. The hero phone plays one authored moment (a raw take editing itself);
// the rest of the page settles in as it scrolls into view. Runs only when html.motion is set.
(() => {
  "use strict";

  const root = document.documentElement;
  window.__yapMotion = true;
  if (!root.classList.contains("motion")) return;
  const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));

  // ---------------------------------------------------------------- the hero edits itself
  const hero = document.querySelector(".hero");
  requestAnimationFrame(() => requestAnimationFrame(() => hero.classList.add("is-in")));
  // The hero reel (hero.js) runs its own sequence on the stage.

  // ---------------------------------------------------------------- the pasted notes arrive line by line
  const pre = document.querySelector(".paste pre");
  if (pre) {
    pre.innerHTML = pre.textContent.split("\n").map((line, i) => `<span class="ln" style="--i:${i}">${line.replace(/&/g, "&amp;").replace(/</g, "&lt;") || "&nbsp;"}</span>`).join("");
  }

  // ---------------------------------------------------------------- scroll reveals
  const SINGLE = ["main > section:not(.hero) h2", "main > section:not(.hero) .sub", ".local", ".more", ".hb-text", ".paste", ".closing .join", ".foot", ".clean", ".rules"];
  const FRAMES = [".shot .frame", ".pair .frame", ".rv", ".roll"];
  const LISTS = [".steps", ".fxs", ".start", ".library ul", ".pains", ".log", ".rv-notes"];
  for (const el of $$(SINGLE.join(","))) el.dataset.reveal = "";
  for (const el of $$(FRAMES.join(","))) el.dataset.reveal = "frame";
  for (const list of $$(LISTS.join(","))) {
    list.dataset.stagger = "";
    Array.from(list.children).forEach((child, i) => { if (!child.dataset.reveal) child.dataset.reveal = ""; child.style.setProperty("--i", String(Math.min(i, 8))); });
  }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const el = e.target;
      io.unobserve(el);
      if (el.hasAttribute("data-stagger")) Array.from(el.children).forEach((c) => c.classList.add("is-in"));
      else el.classList.add("is-in");
      if (el.matches(".shot .frame")) el.closest(".shot").classList.add("tags-in");
    }
  }, { threshold: 0.15, rootMargin: "0px 0px -8% 0px" });
  $$("[data-reveal]:not([data-stagger] > *), [data-stagger]").forEach((el) => io.observe(el));

  // ---------------------------------------------------------------- small rewards
  document.addEventListener("click", (e) => {
    const b = e.target.closest(".rule button[data-act=yes]");
    if (!b) return;
    const rule = b.closest(".rule");
    rule.classList.remove("pop"); void rule.offsetWidth; rule.classList.add("pop");
  });
  for (const form of $$("form.join")) {
    const status = form.querySelector(".form-status");
    new MutationObserver(() => {
      const done = status.textContent.startsWith("You're signed up") || status.textContent.startsWith("You're on the list");
      form.classList.toggle("is-done", done);
      if (done) form.querySelector("button[type=submit]").textContent = "You're in";
    }).observe(status, { childList: true, characterData: true, subtree: true });
  }
})();
