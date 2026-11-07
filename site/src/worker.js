var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// worker.js
var EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
var json = /* @__PURE__ */ __name((body, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } }), "json");
async function waitlist(request, env) {
  if (request.method !== "POST") return json({ ok: false, error: "method" }, 405);
  let body = {};
  try {
    body = await request.json();
  } catch {
    return json({ ok: false, error: "bad_request" }, 400);
  }
  if (body.company) return json({ ok: true });
  const email = String(body.email || "").trim().toLowerCase();
  if (email.length > 254 || !EMAIL.test(email)) return json({ ok: false, error: "invalid_email" }, 400);
  const headers = { "Content-Type": "application/json", Accept: "application/json", "X-Kit-Api-Key": env.KIT_API_KEY };
  const sub = await fetch("https://api.kit.com/v4/subscribers", {
    method: "POST",
    headers,
    body: JSON.stringify({ email_address: email, state: "active" })
  });
  if (!sub.ok) return json({ ok: false, error: "kit_subscriber", status: sub.status }, 502);
  const tag = await fetch(`https://api.kit.com/v4/tags/${env.KIT_TAG_ID}/subscribers`, {
    method: "POST",
    headers,
    body: JSON.stringify({ email_address: email })
  });
  if (!tag.ok) return json({ ok: false, error: "kit_tag", status: tag.status }, 502);
  return json({ ok: true });
}
__name(waitlist, "waitlist");
async function why(request, env) {
  if (request.method !== "POST") return json({ ok: false, error: "method" }, 405);
  let body = {};
  try {
    body = await request.json();
  } catch {
    return json({ ok: false, error: "bad_request" }, 400);
  }
  if (body.company) return json({ ok: true });
  const email = String(body.email || "").trim().toLowerCase();
  const answer = String(body.why || "").trim().slice(0, 600);
  if (email.length > 254 || !EMAIL.test(email)) return json({ ok: false, error: "invalid_email" }, 400);
  if (!answer) return json({ ok: false, error: "empty" }, 400);
  const res = await fetch("https://api.kit.com/v4/subscribers", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json", "X-Kit-Api-Key": env.KIT_API_KEY },
    body: JSON.stringify({ email_address: email, fields: { yap_studio_why: answer } })
  });
  if (!res.ok) return json({ ok: false, error: "kit_field", status: res.status }, 502);
  return json({ ok: true });
}
__name(why, "why");
var EVENTS = /* @__PURE__ */ new Set(["signup", "why", "follow_x", "follow_instagram", "hero_pick", "slide_pick", "rail_next", "copy_feedback"]);
function record(env, request, name, detail = "") {
  if (!env.EVENTS) return;
  const ref = request.headers.get("Referer") || "";
  let refHost = "";
  try {
    refHost = ref ? new URL(ref).hostname : "";
  } catch {
  }
  env.EVENTS.writeDataPoint({
    indexes: [name],
    blobs: [name, new URL(request.url).pathname, refHost, request.cf?.country || "", String(detail).slice(0, 64)],
    doubles: [1]
  });
}
__name(record, "record");
async function event(request, env) {
  if (request.method !== "POST") return json({ ok: false }, 405);
  let body = {};
  try {
    body = await request.json();
  } catch {
    return json({ ok: false }, 400);
  }
  if (!EVENTS.has(body.name)) return json({ ok: false }, 400);
  const src = String(body.src || "").slice(0, 32);
  if (env.EVENTS) env.EVENTS.writeDataPoint({ indexes: [body.name], blobs: [body.name, String(body.path || "/").slice(0, 64), src, request.cf?.country || "", String(body.detail || "").slice(0, 64)], doubles: [1] });
  return json({ ok: true });
}
__name(event, "event");
var worker_default = {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.hostname === "www.yapstudio.app") return Response.redirect(`https://yapstudio.app${url.pathname}${url.search}`, 301);
    if (url.pathname === "/api/waitlist") return waitlist(request, env);
    if (url.pathname === "/api/why") return why(request, env);
    if (url.pathname === "/api/e") return event(request, env);
    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html") && !/bot|crawl|spider|preview/i.test(request.headers.get("User-Agent") || "")) record(env, request, "pageview", url.searchParams.get("utm_source") || "");
    return env.ASSETS.fetch(request);
  }
};
export {
  worker_default as default
};
//# sourceMappingURL=worker.js.map