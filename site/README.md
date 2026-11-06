# yapstudio.app

The Yap Studio website: one static page in `public/` served by the Worker in `src/worker.js`, which also
handles updates signup (`/api/waitlist`, saves to Kit with the original `yap-studio-waitlist` tag,
ID `24466506`), `/api/why` and `/api/e` (events).

Recovered 2026-11-06 from the deployed Worker and the live site; before that the site had no repo.
`src/worker.js` is the deployed bundle (esbuild output with its source map), not hand-written source.

Run commands from this directory (`cd site` from the repository root).

Test locally:

    npx wrangler dev

Local signup needs the Kit secret to save real addresses; validation and the honeypot work without it.
`KIT_API_KEY` stays a Worker secret in production. Never put it in source or `wrangler.jsonc`.

Deploy (LittleMight Cloudflare, token auth, never `wrangler login`):

    export CLOUDFLARE_API_TOKEN="$(op read 'op://Hermes/Cloudflare - Deploy/CLOUDFLARE_API_TOKEN')"
    export CLOUDFLARE_ACCOUNT_ID="$(op read 'op://Hermes/Cloudflare - Deploy/CLOUDFLARE_ACCOUNT_ID')"
    npx wrangler deploy

Website-only changes under `site/` skip product CI and do not select automatic render, speech or
helper release checks. Explicit release tags, manual full checks and nightly checks still run as usual.
The website is excluded from the Python source archive and wheel. Deploy it separately with Wrangler.

The product itself: https://github.com/cathrynlavery/yap-studio
