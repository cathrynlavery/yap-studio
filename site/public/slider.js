// Yap Studio pilot 5: "How it works" as three steps. Auto-advances while on screen, with the active tab
// filling as its step plays. Picking a step stops the auto-advance; hovering pauses it.
(() => {
  "use strict";

  const slider = document.getElementById("slider");
  if (!slider) return;
  const tabs = Array.from(slider.querySelectorAll('[role="tab"]'));
  const slides = Array.from(slider.querySelectorAll(".slide"));
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const STEP_MS = 9000;
  let index = 0;
  let auto = !reduced;
  let visible = false;
  let paused = false;
  let elapsed = 0;
  let last = 0;

  function show(i, fromUser) {
    index = (i + slides.length) % slides.length;
    slides.forEach((s, k) => { s.classList.toggle("on", k === index); s.setAttribute("aria-hidden", String(k !== index)); s.inert = k !== index; });
    tabs.forEach((t, k) => { t.setAttribute("aria-selected", String(k === index)); t.tabIndex = k === index ? 0 : -1; t.style.setProperty("--p", k < index && auto ? "1" : "0"); });
    elapsed = 0;
    if (fromUser) { auto = false; tabs.forEach((t) => t.style.setProperty("--p", "0")); }
    document.dispatchEvent(new CustomEvent("yap:slide", { detail: slides[index].id }));
  }

  function frame(now) {
    const dt = last ? now - last : 0;
    last = now;
    if (auto && visible && !paused && !document.hidden) {
      elapsed += dt;
      tabs[index].style.setProperty("--p", String(Math.min(1, elapsed / STEP_MS)));
      if (elapsed >= STEP_MS) show(index + 1, false);
    }
    if (auto) requestAnimationFrame(frame);
  }

  tabs.forEach((t, k) => {
    t.addEventListener("click", () => show(k, true));
    t.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      show(index + (e.key === "ArrowRight" ? 1 : -1), true);
      tabs[index].focus();
    });
  });
  slider.addEventListener("pointerenter", () => { paused = true; });
  slider.addEventListener("pointerleave", () => { paused = false; });
  slider.addEventListener("focusin", () => { paused = true; });
  slider.addEventListener("focusout", () => { paused = false; });
  new IntersectionObserver((entries) => { visible = entries.some((e) => e.isIntersecting); }, { threshold: 0.35 }).observe(slider);

  show(0, false);
  if (auto) requestAnimationFrame(frame);
})();
