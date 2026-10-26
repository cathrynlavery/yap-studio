# Styles and taste rules

A style is how your captions, titles and cards look. Taste rules are notes you approved in
review that apply to future videos.

## Pick your look

Run `yap-studio style pick` to try captions, titles and cards live over a still
from your newest take. Set your brand fonts and accent color, then choose how
words and cards look. **Save as my style** makes the pick your default and writes
a readable copy in the workspace. Run the command again to update it.
While the page is open it transcribes your whole newest take in the background, so
ingest can reuse the words. Closing the page stops that work, and ingest transcribes
the take itself.
With `--detach` the command prints the link and returns while the page keeps serving.
Running it again with the same options gives the same link. The page stops two minutes
after you save and close it, or when you run `yap-studio style pick --stop`.
[The style page](style-page.md) walks through each screen.

Giant captions is one of the caption looks: the line is drawn oversized in a solid
block, in the brand accent colour, else the caption ground colour.

## Style files

List the shipped looks and make a named style file:

The shipped presets are Candy, Clean Minimal, Cobalt, Editorial Serif, Electric, Headline,
Bold, Editorial, Neutral, Marker, Mono, One Word and Warm Editorial.

Set `captions.entrance` to `slide` (the default) or `rise` for a clipped entrance per word.
Set `captions.camera_3d` to `true` for the 3D camera caption look. It defaults to `false`.

```console
yap-studio style new --list
yap-studio style new "My Look" --preset electric --out my-look.json
yap-studio style validate my-look.json
```

Without `--preset`, `style new` starts with the neutral Clean Minimal preset. Without
`--out`, it writes a file named from the style name in the current directory. It refuses
to replace an existing file. `style add my-look.json` saves it in your style library;
`style default my-look` selects that saved style for new projects.

Run `yap-studio style still --inbox inbox --out preview` to pick a still from the newest
take and its first complete sentence of at least five words that starts without a filler
or greeting. You can pass a take path in place of `--inbox`.

## Picture colour

The `grade` section sets the picture colour. `look` is `none` by default or `film` for
matte blacks, teal shadows and warm highlights. `strength` runs from 0 to 1 and
defaults to 1. All shipped presets use `none`. Assembly grades footage after HDR
conversion to BT.709. Captions, cards and titles keep their own colours.
Use `yap-studio grade show DIR` to see the effective grade. A project override takes
precedence over the style; `yap-studio grade set DIR --clear` restores the style choice.

Eyedropper is a third grade choice. It pulls the picture toward `grade.color`, a
hex colour such as `#1E6BD6`, while protecting skin tones. Looks never stack.
Pick Eyedropper on the style page, then tap the video or enter a hex colour. A new
pick starts at Light (0.5). Its first colour is the average of the scene's top third.
An Eyedropper style without an explicit strength also uses 0.5.
If that area is too grey, pick a more colourful point or enter a colour. Captions
and cards keep their colours. A project override may store a sample point instead
of a colour; assembly resolves it from the current cut before rendering.

`grade.slider` defaults to false. Before/after slider on the style page previews
the raw picture on the left and the grade on the right. It asks the agent to add
one `grade_compare` window soon after the opening. It only matters with an active
grade. Captions, cards and titles keep their colours above the wipe.

## Brand fonts and colors

A style has a captions font and a graphics font. The captions font is
`captions.font`. The graphics font is `hook.font` for the opening title and
`graphics.font` for cards; the style page's Font step sets both together.

Set `brand.captions_font`, `brand.titles_font` and `brand.cards_font` in the style
file to use one font for each role across looks. `brand.accent_color` sets their
accent, and `brand.logo` stores an optional image path for later use. Empty brand
fields leave the look's own choices in place. Drop your own WOFF2, WOFF, TTF or
OTF font files into `brand/fonts/` in your workspace. The engine picks them up
for each new video. You can also add files or a folder directly with
`yap-studio brand fonts add FILE`; list them with `yap-studio brand fonts list`.
Put preferred spellings in `brand/words.txt`, one per line. Each new project adds them to
the stored list. Use `yap-studio brand words add WORD` to add one. Use
`yap-studio brand words remove WORD` to drop one.

Run `yap-studio style save my-look.json --workspace DIR` to save a pick and make it
the default in one step. Each changed save gets a new library revision; the workspace
gets a readable copy at `DIR/style/my-style.json`.
Set `graphics.card_kinds_off` to a list of card kinds the creator does not want.
Newer card kinds start off for every style; list them in `graphics.card_kinds_on`
(or switch them on in the style page) to use them.
`graphics.flow_look` sets how Flow cards look: `boxes` (the default) draws the steps
in a card, and `tags` stacks two or three bare labels with an arrow in the accent color.
`graphics.specs_rows` defaults to `together`; choose `on_cue` for spoken row entrances
or `on_cue_glow` for bare text with an accent glow on Specs list and Stat cards.
`graphics.card_motion` defaults to `none`. Choose `streak` for Motion streak, a
horizontal fly with a fading trail on supported cards. Registry cards keep their
own entrance and fly out only. Cards without a clear path enter as usual.
`broll.emphasis_on` defaults to `[]`; add `video_letters` to turn on mid-video letters inserts.
`broll.stock_clips` defaults to `"off"`; use `"ask"` to offer free stock clips.
`broll.generated_pictures` defaults to `"off"`; use `"ask"` to offer paid generated
pictures on a prompt and cost sheet before the creator approves a spend ceiling.
`broll.overlay_box` accepts `tall_behind` for long pages behind the speaker.
`broll.overlay_motion` accepts `scroll` for eased steps up a long page.
`broll.tall_framing` accepts `auto`, `keep` and `shrink` for framing behind the tall box.

`moves.zoom_mode` sets which zooms the agent uses: `none` (the default), `punches`, or
`punches_and_push`. `moves.punch_scale` sets how far a punch moves in, `1.5` or `2`. A 2x
punch on a take that is 1080 pixels wide enlarges the picture, so it looks softer than the
rest of the video; a 4K take stays sharp. See [Camera moves](editing.md#camera-moves).

`moves.jump_cuts` sets how a jump cut looks: `hard` (the default), `punch`, `smooth`, or `step_in`.
A punch is a zoom the agent adds on the join. Smooth makes the picture glide across a small
jump while the sound stays a hard cut. A big jump gets a punch, even with zooms off.
Step in changes the framing on cuts, then returns wide.
See [Jump cuts](editing.md#jump-cuts).

`moves.opening_push` starts off. It ends at the first sentence end or real cut
between 0.8 and 4 seconds after the first word. If the line runs past 4 seconds, it
uses the first clause end in that window. Longer lines push more slowly and reach
1.3x at their end. Shorter lines stop below 1.3x. If no boundary fits, the push is
skipped with a warning. It never cuts wide in the middle of a clause. It may sit
under an opening title.

`moves.transitions` lists the transition looks the agent may use. It defaults to
`["whip_pan", "cinematic_zoom", "light_leak"]`; use `[]` for none.
See [Transitions](editing.md#transitions).

## Taste rules

You can keep a `creator-rules.json` note next to `my-look.json`. The style loader reads
only the style JSON; it does not load a neighboring rules file automatically. The
supported rules format is a taste proposal tied to a saved review comment. For example:

```json
{
  "scope": "captions",
  "sentence": "Show no more than four words in a caption.",
  "machine": {
    "kind": "style_field",
    "field": "captions.max_words_per_phrase",
    "value": 4
  },
  "comments": [{"version": "v1", "id": "c1"}]
}
```

Replace `v1` and `c1` with a saved version and its review comment. Run
`yap-studio taste propose PROJECT creator-rules.json` to add the proposal to that
project's inbox. Open the project with `yap-studio review --versions PROJECT`; the
proposal waits under Rules for future videos, where you approve, edit or reject it.
Approved rules live in the private `taste.json` under the Yap Studio config folder,
outside the project, and `yap-studio taste show` lists them.

- **Yap Studio applies or checks:** Approved `style_field` rules set supported caption, card,
  B-roll and music style fields at render, unless a project instruction overrides them.
  Placement rules check that captions, titles or cards stay clear of the face through
  `faces.clearance`. Conditions with tags need all those tags to match. If style rules compete
  for one field, the rule with more matching tags wins, then rule id breaks ties. The render
  records applied and skipped rules, including conflicts. Verify uses that pinned taste
  record to check measurable fields and placement. It reports missing measurements as not
  measured, not passed. Setting `captions.placement` to `"fixed"` in this project's style
  or an approved taste rule accepts captions over your face. Both placement and final checks
  report those overlaps as warnings. Titles, cards and overlays still have to clear your face.
- **The agent follows (not enforced):** Pacing notes, including learned shot lengths, pauses
  and punch-ins, do not set the edit. Filler-word taste rules do not remove words automatically.
  Free text and conditions without tags need judgment. Free text is recorded as
  `followed_by_agent`; this does not verify that it was followed. The agent must also resolve
  conflicts between these notes.
