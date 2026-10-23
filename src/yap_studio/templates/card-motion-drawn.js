function cardMotionCue(timeline, enabled = true) {
  return {
    fromTo(target, from, to, at) {
      return timeline.fromTo(target, from, enabled ? { ...to, data: "cue" } : to, at);
    },
    set(target, to, at) {
      return timeline.set(target, { ...to, data: "cue" }, at);
    },
  };
}

function cardMotionDrawn(root, tl, duration) {
  if (root.dataset.cardMotion !== "streak") return;
  function finishEntrance(timeline, start = 0, rate = 1) {
    timeline.getChildren(false, true, true).forEach(child => {
      const at = start + child.startTime() / rate;
      const end = start + child.endTime() / rate;
      if ((child.vars.opacity === 0 || child.vars.autoAlpha === 0)
          && Math.abs(end - duration) < 0.000001) {
        timeline.remove(child);
      } else if (child.vars.data === "cue") {
        return;
      } else if (child.getChildren) {
        finishEntrance(child, at, rate * Math.abs(child.timeScale() || 1));
      } else {
        // Timed card actions carry cue tags; all other reveals belong to the entrance.
        // Pausing completed reveals prevents the timeline from shifting later cues past negative starts.
        child.paused(true).totalProgress(1, false);
        child.startTime(-start * rate - child.totalDuration());
      }
    });
  }
  finishEntrance(tl);
  tl.to({}, { duration, ease: "none", data: "card-motion-window" }, 0);
}
