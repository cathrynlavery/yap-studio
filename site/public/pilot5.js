// Yap Studio pilot 5: the hero phone plays captions over the sample presenter; taste rules can be approved.
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // ---------------------------------------------------------------- taste rules
  for (const rule of $$(".rule")) {
    rule.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-act]");
      if (!b) return;
      rule.classList.toggle("is-on", b.dataset.act === "yes");
      rule.classList.toggle("is-off", b.dataset.act === "no");
    });
  }

  // ---------------------------------------------------------------- effects rail
  const rail = document.getElementById("rail");
  if (rail) {
    const btns = $$(".rail-btn");
    const step = () => (rail.firstElementChild ? rail.firstElementChild.getBoundingClientRect().width + 18 : 300);
    const sync = () => {
      btns[0].disabled = rail.scrollLeft < 8;
      btns[1].disabled = rail.scrollLeft + rail.clientWidth > rail.scrollWidth - 8;
    };
    btns.forEach((b) => b.addEventListener("click", () => rail.scrollBy({ left: step() * Number(b.dataset.dir), behavior: "smooth" })));
    rail.addEventListener("scroll", sync, { passive: true });
    window.addEventListener("resize", sync);
    sync();
  }

  window.Yap.waitlist();
})();
