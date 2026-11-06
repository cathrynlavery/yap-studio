// Shared caption engine for the pilot 3 versions: the ten real presets and a phone player.
window.Yap = (() => {
  "use strict";

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // The presets from docs/design/style-page. style: color, underline, block (a colour block follows each word), none.
  const PRESETS = [
    { id: "editorial", name: "Editorial", font: "Instrument Serif", text: "#FFFFFF", hl: "#E9C46A", hlText: "#1C1B19", style: "color", italic: true, words: 5, scale: 1.12, hook: { kind: "quiet", bg: "#F6F1E7", ink: "#1C1B19", accent: "#B7862A", font: "Instrument Serif" } },
    { id: "clean_minimal", name: "Clean Minimal", font: "Inter", text: "#FFFFFF", hl: null, style: "none", words: 6, scale: 0.86, weight: 600, hook: { kind: "quiet", bg: "#F4F3EE", ink: "#1F2A29", accent: "#365A58", font: "Gelasio" } },
    { id: "warm_editorial", name: "Warm Editorial", font: "Gelasio", text: "#FFF4DD", hl: "#9B462D", style: "underline", words: 5, scale: 0.92, weight: 500, hook: { kind: "quiet", bg: "#E9DDC8", ink: "#2A2018", accent: "#9B462D", font: "Gelasio" } },
    { id: "cobalt", name: "Cobalt", font: "Alata", text: "#FFFFFF", hl: "#2F3BFF", hlText: "#FFFFFF", style: "block", words: 5, scale: 0.92, hook: { kind: "quiet", bg: "#FFFFFF", ink: "#0E1230", accent: "#2F3BFF", font: "Alata" } },
    { id: "mono", name: "Mono", font: "Space Mono", text: "#F4F1EA", hl: "#7CFFB2", hlText: "#111111", style: "block", words: 5, scale: 0.78, hook: { kind: "paper", bg: "#111111", ink: "#F4F1EA", accent: "#7CFFB2", font: "Space Mono" } },
    { id: "electric", name: "Electric", font: "Archivo Black", text: "#FFFFFF", hl: "#D8FF2F", hlText: "#111111", style: "color", words: 4, scale: 0.8, hook: { kind: "band", bg: "#101522", ink: "#FFFFFF", accent: "#D8FF2F", font: "Archivo Black" } },
    { id: "headline", name: "Headline", font: "Anton SC", text: "#FFFFFF", hl: "#FFD93D", hlText: "#111111", style: "block", words: 3, scale: 1.04, hook: { kind: "band", bg: "#111111", ink: "#FFFFFF", accent: "#FFD93D", font: "Anton SC" } },
    { id: "one_word", name: "One Word", font: "Archivo Black", text: "#FFFFFF", hl: "#FF4A1C", style: "none", words: 1, scale: 0.8, hook: { kind: "band", bg: "#FF4A1C", ink: "#FFFFFF", accent: "#FFE14D", font: "Archivo Black" } },
    { id: "candy", name: "Candy", font: "Titan One", text: "#FFFFFF", hl: "#FF8AC6", hlText: "#2B0F2E", style: "block", words: 4, scale: 0.84, hook: { kind: "paper", bg: "#FFE3F1", ink: "#2B0F2E", accent: "#E0458F", font: "Titan One" } },
    { id: "marker", name: "Marker", font: "Permanent Marker", text: "#FFFFFF", hl: "#FFD23F", hlText: "#1D1D1D", style: "underline", words: 4, scale: 0.9, hook: { kind: "paper", bg: "#FFF6DC", ink: "#1D1D1D", accent: "#FF2E88", font: "Permanent Marker" } },
  ];
  const byId = Object.fromEntries(PRESETS.map((p) => [p.id, p]));
  // The style page's sample script and hook.
  const SCRIPT = "I do one take. Then I cut all the dead air. And that's the edit.".split(" ");
  const HOOK = ["You only need", "one take"];
  const CARD = { title: "The whole edit", items: ["Do one take", "Cut the dead air", "Post it"] };

  const inkOn = (hex) => {
    const n = parseInt(hex.slice(1), 16);
    const lum = (0.2126 * ((n >> 16) & 255) + 0.7152 * ((n >> 8) & 255) + 0.0722 * (n & 255)) / 255;
    return lum > 0.58 ? "#111111" : "#FFFFFF";
  };

  // Put a preset (and an optional highlight colour override) on any element as CSS variables.
  function wear(el, p, color = null) {
    const hl = color || p.hl;
    const style = color && p.style === "none" ? "color" : p.style;
    el.style.setProperty("--cap-font", `"${p.font}"`);
    el.style.setProperty("--cap-text", p.text);
    el.style.setProperty("--cap-hl", hl || "transparent");
    el.style.setProperty("--cap-hl-ink", hl ? (color ? inkOn(color) : p.hlText || inkOn(hl)) : p.text);
    el.style.setProperty("--cap-scale", String(p.scale));
    el.style.setProperty("--cap-weight", String(p.weight || 400));
    el.dataset.hl = style;
    el.dataset.italic = String(Boolean(p.italic));
    el.dataset.words = String(p.words);
  }

  function phrasesFor(p, words = SCRIPT) {
    const out = [];
    let cur = [];
    for (const w of words) {
      cur.push(w);
      if (cur.length >= p.words || /[.!?]$/.test(w)) { out.push(cur); cur = []; }
    }
    if (cur.length) out.push(cur);
    return out;
  }

  // A page can set window.YAP_PHOTO to show a photo of a presenter instead of the drawn one.
  const frameHtml = () => (window.YAP_PHOTO
    ? `<img class="ph-photo" src="${esc(window.YAP_PHOTO)}" alt="">`
    : `<img class="ph-bg" src="../pilot/assets/frame-room.svg" alt="">
      <img class="ph-person" src="../pilot/assets/frame-person.svg" alt="">`);
  const screenHtml = (extra = "") => `<div class="ph-screen ${extra}${window.YAP_PHOTO ? " has-photo" : ""}">
      ${frameHtml()}
      <div class="ph-title" hidden></div>
      <div class="ph-card" hidden></div>
      <p class="ph-cap" aria-hidden="true"></p>
      <span class="ph-badge" hidden>Ready to post</span>
      <span class="ph-synth">${window.YAP_PHOTO ? "Sample presenter, generated image" : "Sample presenter"}</span>
    </div>`;

  // A player drives one .ph-screen: captions word by word, with the hook title and the card on cue.
  function player(screen, { mode = "preview", words = SCRIPT, hook = HOOK, card = true, size = 1 } = {}) {
    const cap = screen.querySelector(".ph-cap");
    const titleEl = screen.querySelector(".ph-title");
    const cardEl = screen.querySelector(".ph-card");
    const badge = screen.querySelector(".ph-badge");
    let preset = PRESETS[0];
    let color = null;
    let timer = 0;
    let running = false;
    const state = { mode, card, size };

    function paintHook(on) {
      const h = preset.hook;
      titleEl.hidden = !on;
      if (!on) return;
      titleEl.className = `ph-title k-${h.kind}`;
      titleEl.style.setProperty("--h-bg", h.bg);
      titleEl.style.setProperty("--h-ink", h.ink);
      titleEl.style.setProperty("--h-accent", h.accent);
      titleEl.style.setProperty("--h-font", `"${h.font}"`);
      titleEl.innerHTML = `<span>${esc(hook[0])}</span> <span class="acc">${esc(hook[1])}</span>`;
    }
    function paintCard(on) {
      cardEl.hidden = !on;
      if (!on) return;
      cardEl.style.setProperty("--h-accent", preset.hook.accent);
      cardEl.style.setProperty("--h-font", `"${preset.hook.font}"`);
      cardEl.innerHTML = `<b>${esc(CARD.title)}</b>${CARD.items.map((x) => `<span>${esc(x)}</span>`).join("")}`;
    }
    function stop() { running = false; clearTimeout(timer); }
    function play() {
      stop();
      running = true;
      wear(screen, preset, color);
      screen.style.setProperty("--cap-adjust", String(state.size));
      const ph = phrasesFor(preset, words);
      let p = 0;
      let w = -1;
      if (state.mode === "preview") { paintHook(true); paintCard(false); }
      const tick = () => {
        if (!running) return;
        w++;
        if (w >= ph[p].length) { w = 0; p++; }
        if (p >= ph.length) { p = 0; w = 0; }
        const phrase = ph[p];
        cap.innerHTML = phrase.map((word, i) => `<span class="w${i === w ? " on" : ""}">${esc(word)}</span>`).join(" ");
        if (state.mode === "final") {
          paintHook(p === 0);
          paintCard(state.card && /\b(cut|dead|air)/i.test(phrase.join(" ")));
        }
        timer = setTimeout(tick, reduced ? 1e9 : w === phrase.length - 1 ? 620 : 300);
      };
      tick();
    }
    return {
      setPreset(p, c = null) { preset = p; color = c; if (running) play(); else wear(screen, preset, color); },
      setMode(m) { state.mode = m; if (running) play(); },
      set(o) { Object.assign(state, o); if (running) play(); },
      play,
      stop,
      clear() { stop(); cap.innerHTML = ""; paintHook(false); paintCard(false); badge.hidden = true; },
      badge(on) { badge.hidden = !on; },
      get preset() { return preset; },
    };
  }

  // Preset chips: buttons that show a sample in the preset's own look.
  const chipHtml = (p, cls = "chip") => `<button type="button" role="radio" class="${cls}" data-id="${p.id}" aria-checked="false"
      style="--cap-font:'${p.font}';--cap-hl:${p.hl || "transparent"};--cap-hl-ink:${p.hlText || p.text};--cap-text:${p.text}" data-hl="${p.style}">
      <span class="chip-sample"><span>Aa</span> <span class="on">one</span></span>
      <span class="chip-name">${esc(p.name)}</span>
    </button>`;

  // Waitlist forms post to Kit's public form endpoint when the page sets window.YAP_KIT_FORM_ID.
  function waitlist(root = document) {
    for (const form of root.querySelectorAll("[data-waitlist]")) {
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const status = form.querySelector(".form-status");
        const button = form.querySelector("button[type=submit]");
        const url = window.YAP_WAITLIST_URL;
        const id = window.YAP_KIT_FORM_ID;
        if (!url && !id) { status.textContent = "Not connected yet. This form needs a Kit form ID before it can save emails."; return; }
        const email = form.querySelector("input[type=email]").value.trim();
        const trap = form.querySelector("input[name=company]");
        button.disabled = true;
        status.textContent = "Adding you…";
        try {
          let res;
          if (url) {
            // The site's own endpoint adds the email to Kit with the waitlist tag.
            res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, company: trap ? trap.value : "" }) });
          } else {
            const body = new FormData();
            body.append("email_address", email);
            res = await fetch(`https://app.kit.com/forms/${encodeURIComponent(id)}/subscriptions`, { method: "POST", body, headers: { Accept: "application/json" } });
          }
          const data = await res.json().catch(() => ({}));
          if (!res.ok || data.status === "failed" || data.ok === false) throw new Error(data.error || data.errors?.messages?.join(" ") || `HTTP ${res.status}`);
          form.reset();
          form.dispatchEvent(new CustomEvent("yap:joined", { bubbles: true, detail: { email } }));
          status.textContent = url ? "You're signed up for Yap Studio updates." : "You're on the list. Check your inbox to confirm your email.";
        } catch (err) {
          status.textContent = "That didn't go through. Check the email address and try again.";
          console.error("waitlist", err);
        } finally {
          button.disabled = false;
        }
      });
    }
  }

  return { PRESETS, byId, SCRIPT, HOOK, CARD, wear, phrasesFor, screenHtml, player, chipHtml, waitlist, esc, reduced };
})();
