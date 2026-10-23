The engine page is `yap-studio style pick`. Its shipped page now lives in
`src/yap_studio/gallery/static/`, with one choice per screen and engine previews.

# Create your video style (design prototype)

A working prototype of the buyer portal page where someone builds a video style from four choices, watches it play on a sample video, and saves a style file.

```console
uv run python docs/design/style-page/serve.py
```

Run this from the repository root, then open the address it prints (it starts at port 8792 and moves up if that port is busy). Opening `index.html` straight from disk also works, except pulling colors from a website, which needs `serve.py`. Add `#preset=cobalt` (or any preset id) to the URL to start from that preset.

## How the page works

The big preview stays pinned on the left. On the right, the buyer can start from a preset, then change any of four choices:

| Choice | Options | What it sets in the style file |
| --- | --- | --- |
| Words on screen | One word, A few words, Short phrase, Full sentence | Words and characters per phrase, size, position |
| Highlight | Sliding block, Underline, Colored word, Pill, Word fill, Bold key word, No highlight, italics on or off, how each phrase appears | `highlight_style`, `italic_stress`, entry rise and time |
| Font | Same font for everything or Different font for graphics; ten bundled fonts including Geist; Instrument Serif captions with Geist titles and cards in one tap; Letter width: Normal, Condensed, Narrow | Captions font and graphics font (opening title and cards), caption letter spacing, `captions.width_scale` |
| Colors | Ten palettes, four hex fields, or colors pulled from a website | Every caption, title and graphics color |

Caption font options are drawn as live previews with the buyer's other choices already applied. A captions font option shows the current colors and highlight, and a palette option shows the current font. In Different font for graphics, a second grid shows sample text for the titles and cards font. Letter width applies to captions only. No highlight also turns italics off, since italics are a highlight too. Opening title, music volume and a fine-tune panel (exact size, letter width, word limit, position) sit below the four choices.

Pill puts the whole phrase inside a soft pill. Each spoken word lights up and stays lit, while upcoming words stay muted. Word fill sweeps color across each word from left to right as it is spoken. Bold key word keeps the stressed word bold for the whole phrase and colors it once spoken, so line breaks stay fixed. Italics still follow the separate choice for key words.

Letter width sits under Font: Normal is 1, Condensed is 0.85 and Narrow is 0.7. Fine-tune captions has an exact slider from 0.6 to 1 in steps of 0.01. Narrower letters fit more text per line. A custom width marks the closest tile and shows its actual percentage without changing the saved value.

## Colors

The four color roles are Captions, Highlight, Cards and Card text. The page derives the rest: text on a sliding block or pill picks whichever of card text, near black or white reads best, and the title strip on graphic cards is the card color mixed 8% toward the card text. Hex fields accept `#2F3BFF`, `2f3bff` or `#2BF`.

Pill warns if spoken text falls below 4.5:1 contrast or muted upcoming text falls below 3:1 against the pill color. The muted text uses the engine's mix of 45 percent highlight text color into the highlight color.

`serve.py` answers `GET /api/site-colors?url=...`. It combines two sources:

1. The context.dev brand lookup, when `context-dev-pp-cli` is installed and signed in. It returns named logo and site colors. Each lookup costs context.dev credits, so the server caches results while it runs.
2. The site's own HTML and up to six stylesheets. Colors on buttons and links count extra, the body background becomes the card color, and browser default link colors are ignored.

The highlight must be a saturated color. When a site is black and white, the page keeps the buyer's current highlight and says so. The server only listens on 127.0.0.1, only fetches public http and https addresses, and caps time and size on every request.

## Presets and source data

`style-data.js` is generated from the engine's `src/yap_studio/style/schema-v1.json` and preset files so the page works from disk. Run these commands from the repository root:

```console
python3 docs/design/style-page/sync_data.py
python3 docs/design/style-page/sync_data.py --check
```

No source argument is needed. `sync_data.py` only copies the engine presets on its `SHIPPED` list. The script never reads `creators/`, so no creator wording rules or spelling fixes reach the page.

| Preset | Status |
| --- | --- |
| Editorial Serif, Clean Minimal, Warm Editorial, Cobalt, Mono, Headline, Electric, Candy, Marker, One Word | Shipped in the engine |

The last five proposed files moved unchanged into the engine; `proposed-presets/order.json` is now empty. Every shipped preset passes the engine's style validation and uses bundled fonts. Editorial Serif uses Instrument Serif with a gold colored word, a quiet title and cream cards.

The preview and the engine share phrase boundaries and automatic emphasis selection, verified by the shared cases in `tests/fixtures/phrase-cases.json`.

`looks.js` builds the highlight choices and letter width limits from this schema. A new highlight value without page copy raises an error and fails the Node tests, so it cannot silently go missing from the page.

## Saving

**Background and placement** offers no ground, a fitted box, or a full-width band,
with color, opacity, padding and box corner controls. Choose fixed placement or
face avoidance. The preview uses the synthetic presenter's head bounds; the engine
uses measured faces from the video. Translucent grounds keep a contrasting text
outline. The saved file retains the chosen ordinary caption color, so turning off
the ground restores it.

Use **Open a learned style** beside the presets to choose the `measurements.json`
written by `yap-studio taste learn`. The page checks the file and opens its complete
proposed style. A panel lists the values measured from the videos and the choices
taken from the named base. The file input also works when opening the page from disk.
You can change any choice before saving. The source panel describes the style as opened.

Run `taste learn VIDEO... --out DIR --base preset:electric` to use a preset for
unmeasured choices, or pass a style file to `--base`. Without a base, learning uses
schema defaults. Fonts and words per phrase come from the base because learning does
not recognise text yet.

Save style checks the result with the same rules as the engine loader, adds it to a list of saved styles in `localStorage` (the stand-in for the account's style library), marks it as the default when "Use this style for my next videos" is ticked, and offers a copy as a download. Import the download with `yap-studio style add FILE --default` to save it in your own style library and select it for new videos. Each new video gets the style it names, else the library default, else Clean Minimal. The file holds `schema_version`, `name`, and the full `brand`, `captions`, `hook`, `graphics` and `music` sections. It leaves out `internal`, so the loader fills delivery and safe-zone values from the schema.

## Known gaps

- Schema v1 has no separate fields for all caps, outline stroke, colored glow or font weight. Pill, Word fill and Bold key word offer their phrase background, color sweep and bold stress through `highlight_style`. Anton SC gets its caps look from its letterforms.
- The page offers ten fonts that ship with Yap Studio: Instrument Serif, Gelasio, Inter, Geist, Alata, Archivo Black, Anton SC, Titan One, Permanent Marker and Space Mono. Same font for everything applies one choice to all roles. Different font for graphics keeps a captions font and a graphics font, with Instrument Serif captions and Geist titles and cards one tap away.
- `electric.json` sets no `titlebar_color`, so it inherits a light strip on a dark navy card. Choosing any palette replaces it with a derived strip.
- The sample frame (`frame-room.svg`, `frame-person.svg`) is drawn, not real footage.

## Screenshots

`screenshots/` holds desktop, tablet and mobile captures using a synthetic presenter:
the first view, a words-on-screen option playing on hover, No highlight, the font
choice, hex codes on mobile, the mobile saved state, the undo toast after switching
presets, and `preview-interaction.gif`, which shows the highlight moving word by word.
