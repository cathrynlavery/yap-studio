(function () {
  "use strict";
  const config = window.__cardMotion;
  const tl = window.__cardMotionTimeline;
  const mount = document.getElementById("hf-mount");
  const entrance = config.enter ? document.getElementById("card-motion-in") : null;
  const host = config.enter ? "band" : "hf-host";
  if (config.enter) {
    tl.fromTo(entrance, { x: config.offset },
      { x: 0, duration: 0.4, ease: "expo.out" }, 0);
  }
  tl.to(mount, { x: config.offset, duration: 0.35, ease: "expo.in" }, config.duration - 0.35);
  function holdTime(inner) {
    const tweens = inner.getChildren(true, true, false)
      .filter(tween => tween.vars.data !== "card-motion-window");
    if (config.enter) {
      const ends = tweens.map(tween =>
        tween.globalTime(tween.totalDuration()) - inner.globalTime(0));
      const start = Math.max(0, ...ends.filter(end => end <= config.duration));
      return (start + config.duration) / 2;
    }
    const fades = tweens.filter(tween => (tween.vars.opacity === 0 || tween.vars.autoAlpha === 0)
      && tween.startTime() > config.duration / 2
      && tween.endTime() >= config.duration - 1 / 30);
    if (!fades.length) return config.settle;
    const end = Math.min(...fades.map(tween => tween.startTime()));
    const start = Math.max(0, ...tweens.filter(tween => !fades.includes(tween)
      && tween.endTime() <= end).map(tween => tween.endTime()));
    return (start + end) / 2;
  }
  function seekPose(inner, time) {
    inner.seek(time);
    // Quiet seeks still need counters painted from their settled properties.
    inner.getChildren(true, true, false).forEach(tween => {
      const update = tween.vars.onUpdate;
      if (update) update.apply(tween.vars.callbackScope || tween, tween.vars.onUpdateParams || []);
    });
  }
  function attachTarget(target, inner, time) {
    seekPose(inner, time);
    const before = new Set(tl.getChildren());
    attachEchoSmear(target, tl, { length: 50, threshold: 40 });
    const group = target.previousElementSibling;
    if (group?.hasAttribute("data-hf-echo-group")) {
      group.querySelectorAll("[data-composition-id],[data-composition-src],[data-hf-authored-id]")
        .forEach(element => {
          for (const name of ["data-composition-id", "data-composition-src", "data-hf-authored-id"]) {
            element.removeAttribute(name);
          }
        });
      const copies = Array.from(group.children).reverse();
      for (const copy of copies) {
        copy.querySelectorAll("*").forEach(element =>
          element.style.setProperty("visibility", "inherit", "important"));
      }
      group.querySelectorAll("*").forEach(element => {
        for (const name of ["box-shadow", "text-shadow", "filter"]) {
          element.style.setProperty(name, "none", "important");
        }
      });
      // One update owns both the helper and the fade, including on backward seeks.
      const update = tl.getChildren().find(child => !before.has(child));
      const render = update.eventCallback("onUpdate");
      update.eventCallback("onUpdate", null);
      function paint() {
        render();
        const shown = copies.filter(copy => copy.style.visibility !== "hidden");
        shown.forEach((copy, index) => {
          const rank = 1 - (index + 1) / (shown.length + 1);
          copy.style.opacity = String(Number(copy.style.opacity || 1) * rank * rank);
          copy.style.filter = "none";
        });
      }
      tl.to({ progress: 0 }, { progress: 1, duration: config.duration,
        ease: "none", onUpdate: paint }, 0);
      paint();
    }
  }
  function attach(inner) {
    // Snapshot the outer mount before the inner exit group exists.
    if (entrance) attachTarget(entrance, inner, 0);
    attachTarget(mount, inner, holdTime(inner));
    seekPose(inner, 0);
    tl.seek(0);
    window.__timelines = window.__timelines || {};
    window.__timelines[host] = tl;
    window.__hfForceTimelineRebind?.();
  }
  async function ready() {
    await document.fonts.ready;
    const inner = window.__timelines?.[config.inner];
    if (inner) {
      mount.getBoundingClientRect();
      await Promise.all([document.fonts.ready,
        ...Array.from(mount.querySelectorAll("img"), image => image.decode())]);
      attach(inner);
    } else setTimeout(ready, 10);
  }
  ready();
})();
