// Yap Studio pilot 5: after someone joins, swap the form for a short thank-you and a follow link,
// picked by where they came from (?utm_source= or the referring site).
(() => {
  "use strict";

  // Instagram stays off until the handle is confirmed.
  const SOCIAL = {
    x: { label: "Follow @cathrynlavery on X", url: "https://x.com/cathrynlavery" },
    instagram: null,
  };

  function source() {
    const saved = sessionStorage.getItem("yap-src");
    if (saved) return saved;
    const q = (new URLSearchParams(location.search).get("utm_source") || "").toLowerCase();
    const ref = (document.referrer || "").toLowerCase();
    let src = "other";
    if (/^(ig|instagram)/.test(q) || /instagram\.com/.test(ref)) src = "instagram";
    else if (/^(x|twitter)/.test(q) || /(^|\/\/)(www\.)?(x\.com|twitter\.com|t\.co)\//.test(ref)) src = "x";
    sessionStorage.setItem("yap-src", src);
    return src;
  }
  const src = source();

  function links() {
    const order = src === "instagram" ? ["instagram", "x"] : ["x", "instagram"];
    return order.map((k) => SOCIAL[k]).filter(Boolean)
      .map((s, i) => `<a class="follow${i ? " ghost" : ""}" href="${s.url}" target="_blank" rel="noopener">${s.label}</a>`).join("");
  }

  const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  document.addEventListener("yap:joined", (e) => {
    const form = e.target.closest("form.join");
    if (!form || form.nextElementSibling?.classList.contains("joined")) return;
    const email = e.detail?.email || "";
    const box = document.createElement("div");
    box.className = "joined";
    box.setAttribute("role", "status");
    box.innerHTML = `<p class="joined-h">You're signed up for updates.</p>
      <p class="joined-p">I'll send new releases and notes on using Yap Studio. <a href="#get">Get it now</a> if you haven't yet.</p>
      <div class="inbox" aria-hidden="true">
        <div class="ib-tabs"><span class="t-pri">Primary</span><span class="t-pro">Promotions</span></div>
        <div class="ib-rows">
          <div class="ib-row ghost"></div>
          <div class="ib-row mail"><b>Cathryn Lavery</b><span>Yap Studio updates</span></div>
          <div class="ib-row ghost short"></div>
        </div>
        <i class="ib-hand"></i>
      </div>
      <form class="why" novalidate>
        <label for="why-${form.id || "f"}">One question: why do you want this?</label>
        <textarea id="why-${form.id || "f"}" rows="2" maxlength="600" placeholder="The videos I keep not editing are…"></textarea>
        <div class="why-row"><button type="submit">Send</button><span class="why-status" role="status"></span></div>
      </form>
      <div class="joined-links">${links()}</div>`;
    form.hidden = true;
    form.after(box);

    const why = box.querySelector("form.why");
    why.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const text = why.querySelector("textarea").value.trim();
      const status = why.querySelector(".why-status");
      const btn = why.querySelector("button");
      if (!text) { status.textContent = "Write a few words first."; return; }
      if (!email) { why.outerHTML = `<p class="why-done">Thanks, that helps.</p><blockquote class="why-quote">${esc(text)}</blockquote>`; return; }
      btn.disabled = true;
      status.textContent = "Sending…";
      try {
        const res = await fetch("/api/why", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, why: text }) });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data.ok === false) throw new Error(data.error || `HTTP ${res.status}`);
        why.outerHTML = `<p class="why-done">Thanks, that helps.</p><blockquote class="why-quote">${esc(text)}</blockquote>`;
      } catch (err) {
        status.textContent = "That didn't send. Try again in a moment.";
        btn.disabled = false;
        console.error("why", err);
      }
    });
  });

  // ?preview=joined shows the after-signup screen without signing anyone up.
  if (new URLSearchParams(location.search).get("preview") === "joined") {
    const form = document.querySelector(".hero form.join");
    if (form) form.dispatchEvent(new CustomEvent("yap:joined", { bubbles: true, detail: { email: "" } }));
  }
})();
