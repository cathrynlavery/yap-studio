# Captions, titles and cards

HyperFrames draws the captions, titles and cards. They render as separate layers, and
assembly places them over the cut. Your saved style decides how they look; see
[styles.md](styles.md).

## Captions

Fix a misheard word with `yap-studio words fix DIR --word t1:123 --to "captcha"`.
Use `--heard "blogging" --to "logging"` for every matching occurrence in this project.
Occurrence fixes win over heard-text fixes. Timing stays the same. The original trailing
punctuation stays unless the replacement supplies its own. Swear masking still applies.
Use `--list` to see both kinds and `--remove` with the same selector to undo a fix.
The next `run` or `assemble` rebuilds captions and titles that use the fixes.

Project captions write `captions/captions.mov`, `captions/captions.srt`,
`captions/phrases.json` and a receipt. A phrase overlapping a caption gap is hidden
whole. The last phrase keeps its normal tail and ends before trailing footage.
If every phrase is hidden, or there are no words, captions writes the SRT, phrases,
layout and receipt with `layer: null`. It skips rendering a layer, and assembly skips
the caption overlay.
The layer ends at the cut's last frame; assembly freezes that frame for
the silent end hold. With `captions.position: bottom`, `top_px` sets a one-line
caption's line-box top. Wrapped phrases grow upward, keeping their last line at
the same height. If one line plus entry rise and shadow padding cannot fit above
`internal.caption_safe_bottom_px`, the plan and receipt report the clamp in
`warnings`. With `position: middle`, the band stays centered on `top_px` with its
existing safe-boundary clamp.

Project captions use `captions.placement: "avoid_face"` by default. When placement
finds no clear position, captions checks the rendered phrase. It stops before
publishing if the phrase still fails face clearance, and lists the times. Earlier
captions with no clear position are checked again on the next run. Frame the
shot wider, use smaller captions, or set `captions.placement` to `"fixed"` in this
project's style to accept captions over your face. Face overlaps then warn and
allow the final to verify. Cards and overlays still have to clear your face.
Horizontal videos always keep captions in the bottom band; that alone does not accept face overlaps.
`run --allow-violations` reports confirmed overlaps as warnings and continues to assembly.

`captions.entrance` defaults to `slide`, which moves the whole phrase by
`entry_rise_px` over `entry_ms`. Choose `rise` to bring each word up from behind
its line box, using the same easing curve. Each word moves 0.6 of its line height
over 8 frames, with 2 frames between starts and opacity reaching full by frame 3.
At 30 fps the move takes 267 ms. Short phrases reduce the stagger, then the move
length to at least 1 frame. Rise also works with `max_words_per_phrase: 1`.
Choose `spoken` for Word by word. Each word fades in over three frames when it
is said and stays until the phrase ends. The full phrase sets the layout and
background size from the first frame. Unseen words keep their space. A word
said before the phrase starts appears on its first frame. There is no cursor.
With `max_words_per_phrase: 1`, the existing phrase entrance stays unchanged.

Giant captions is one of the caption looks: the line is drawn oversized in a solid
block, in the brand accent colour, else the caption ground colour.

### 3D camera captions

Choose 3D camera on the Captions screen or set `captions.camera_3d` to `true`.
It is off by default. Words sit at three depths around you. A key word sits large
behind you, above or beside your head. It uses your accent color and title font.
When too little of it would show, that sentence has no key word. The word stays
in its caption.
The virtual camera pushes in, moves sideways between sentences and pushes into
the last sentence. Your picture moves with it. Words appear at most 0.1 seconds
before you say them. Word fixes and brand words still apply.

The look writes `captions/captions.mov` for the front words,
`captions/camera3d.mp4` for the moving picture and the word behind you, and
`captions/camera3d.json` for its plan. It also writes the usual phrases, SRT,
layout and receipt files.

It uses your caption font, text color, shadow, `max_words_per_phrase` up to four
words per group, and `max_characters_per_phrase`. The big word uses your brand
accent color and title font. Box or band, highlight color, entrance and Keep
captions off my face do not apply. Depth sets sizes from 60 to 84 px.
When depth words have no room, they show as your normal captions at your size
and position, clamped to 60 to 84 px and moved only to keep clear.
Captions keep the same face clearance as other captions.
This includes your face plus room for hair. When the room above your head is
small, many captions land on your chest or low in the frame.
Words stay below the safe text top, as other text does.

It needs a vertical 1080 by 1920 video at 30 fps, a steady camera and you in
every shot with a usable person cutout. It refuses handheld or panning takes.
It also refuses zooms (punches, slow pushes and Push in on the first line),
moments, a background behind you, cards behind your head, Big number,
a tall box behind your head, or split B-roll. The refusal
names the reason and a fix. Remove the conflicting choices or pick another
caption look. Jump cut moves such as punch, smooth and step in become hard cuts.
When your head or a card fills the frame so a caption has no room, it stops with
the time. It asks you to frame the shot wider or pick another caption look.

Full screen B-roll, B-roll in a box, top cards, titles, transitions, the grade,
music and sound work with it. Words stay below a card or title while it is on
screen. Verify skips picture comparison for the moving picture as it does for
zooms. Caption, face, contrast and audio checks still run.

The single clip `captions CLIP` command refuses this look because it needs a
project. The backing option on `captions DIR` does not apply because this look
always draws its own halo.

## Captions that are hard to read

The final checks captions against the picture behind them. Phrases under 3:1 get a
dark halo automatically when the final is rendered. A phrase still under 3:1 shows a
warning with the time and a fix, and under 1.5:1 the warning says the word nearly
vanishes. Contrast never stops a final; it still publishes. When the picture or
caption style behind a phrase changes, the phrase is measured again without its halo.

Run `yap-studio captions DIR --back box` to put a box behind phrases still weak.
Use `yap-studio captions DIR --back none` to turn backing off for them, or
`yap-studio captions DIR --back shadow` to choose the halo. Choices are saved per
phrase in `project.json` and take precedence over automatic halos.
Run `yap-studio run DIR` to render the final again.

## Titles and cards

Titles and cards write `graphics/<element id>.mov` and a receipt with
each layer's box position, start frame and frame count. Layers render at their
own box sizes for assembly to place over the cut.
Text at the top has three choices: None, Just the opening, and The whole video.
There is no default for a new creator. For both kinds of text, the agent writes
three ideas, shows them in the Story cut note and review page, and waits for a
pick or the creator's own text. It never picks for them.

Opening title looks are Paper card, Quiet title, Bold band, Calm card, Lockup,
Handwritten write-on, Headline slam, Shard reveal, Textured, Shimmer, Stagger,
Morph, Typewriter, Video in letters and Kinetic hook.
An opening title uses the chosen look for the first few seconds. Its choice lives
in `plans/text-hook.json` with `kind: "opening"`, three `ideas`, `chosen` and
`source`. Put exactly the chosen text in the title's timeline `lines`.
Rendering refuses missing choices and titles with different text. Existing projects
keep their title layers until the creator answers Text at the top once.

Kinetic hook fills short lines with giant type and punches each word over the
picture as it is spoken. It centres above the head when face data is available
and needs the words to be spoken inside the title window.

### Whole-video label (Text hook)

Text hook uses plain Inter in joined rounded boxes near the top for the whole video.
Choose White, Black or Accent. Text fits one or two balanced lines from 72 to 60 px.
It waits for chosen text in `plans/text-hook.json`. Text that cannot fit or clear
the face is refused. Captions keep their place.

### Cards

Timeline `graphic` elements use the card library's template and typed slots, with
the saved style's graphics tokens. Local image-slot paths resolve from the project
folder. Unchanged layers are reused; changing a card slot or an image's bytes
renders that card again. Cards use the default front layer; behind-the-head
graphics need the person cutout and are refused until it is available.

Card kinds are `terminal`, `stat`, `checklist`, `text_tag`, `image_card`,
`transfer`, `image_compare`, `steps`, `flow`, `chart`, `quote`, `progress_stat`,
`specs_list`, `callout_circle`, `name_tag`, `success_check`, `edit_timeline`, `storyboard`,
`jiggle_delete`, `file_drop`, `video_card`, `prompt_bar`, `comment_cta`, `comment_drop`,
`icon_strike`, `gravity_drop`, opt-in `big_number`, `playing_grid`, `reel_feed` and opt-in `choices`. Use
`yap-studio templates show ID` for each kind's typed slots and limits.

Flow's opt-in `tags` look stacks 2 or 3 bare labels with accent arrows, using her
words shortened at word boundaries and one cue per tag. Text fits one line at
52 to 60 px, including padding and the pop reserve within 80% of the frame width.
The stack stays below the safe text top, centred above the heads first, then
beside them or below them with face and caption clearance. Each position refits
down to 52 px before trying the next. Shorten any tag that cannot fit at that size.
When the planner reports no room above or beside the head, use shorter tags or move
the card to another moment with more room above the head.
Keep at least 1.5 s for the card; each tag pops on its cue and the stack clears
quickly at the end. End the card on the first word after a cut.

The new kinds group as Words (`quote`, `specs_list`, `success_check`, `gravity_drop`, `choices`), Numbers
(`chart`, `progress_stat`, `big_number`), Media (`callout_circle`, `icon_strike`), Editor (`edit_timeline`,
`storyboard`, `jiggle_delete`, `file_drop`), Video (`video_card`, `playing_grid`, `reel_feed`), Chat
(`prompt_bar`, `comment_cta`, `comment_drop`) and People (`name_tag`).

`prompt_bar`, `comment_cta`, `comment_drop`, `icon_strike`, `playing_grid`, `reel_feed` and
`choices` start off until the creator turns them on in the style page or lists them in
`graphics.card_kinds_on`.

With Rows on cue enabled in the style, `specs_list` accepts optional `cues`, up to
five cue times in seconds from the card start, one per row, or word anchors in a cards plan.
`stat` accepts optional `rows`, up to two breakdown lines of 1 to 32 characters,
and optional `cues`, up to two values or word anchors on those lines.
Rows come from her words and numbers, keeping her exact figure. Each row needs
0.3 s to enter and 0.3 s of rest before the fade-out. Omitted cues spread the rows
from the title entrance to the latest allowed cue; partial cues fill forward.
Together refuses Specs list cues and Stat rows or cues until Rows on cue is on.

The glowing choice removes the panel and gives the title or figure an accent glow.
In the card look, the panel grows as each row arrives.
In the card, Specs titles use 52 to 60 px and rows use 44 to 52 px. Bare glowing
titles and figures use 72 to 90 px and rows use 48 to 56 px, with a dark core under pale text.
Stat breakdown rows use the same row sizes; its figure shrinks to leave room.
With the glowing rows look, a Stat with breakdown rows requires `reveal: "count"`.
Wheel and slot stay available without rows and in the card looks.
The panel uses 12 px inner padding and tighter title spacing, keeping the row dividers.
Rows stay on one line. These cards fit above the head first, then beside it, then
below it, with letters and panels below the safe line and clear of captions.
Effect padding may extend above the safe line and is included when fitting.
They refit to the size minimums before moving. When no position fits, use fewer
or shorter rows or move the card to another moment with more room above the head.
These cards may hold across cuts for up to 8 s when Rows on cue is on; a Stat needs
breakdown rows for the longer hold. End the card on the first word after the cut
that follows the last row's sentence.

Motion streak is an opt-in style choice (`graphics.card_motion: "streak"`). Supported
cards arrive fully drawn from the edge away from the face in 0.4 s and return to that
edge in 0.35 s. The fading trail copies the card's own colors. The card stays on its
row, below the 280 px line. Planning checks the whole swept box during each move,
tries the other edge if needed, and records a fallback reason when neither is clear.
Registry cards keep their own entrance and shared fade, with a streak on the exit only.
Number cards, full-screen cards, video windows and cards without a GSAP panel timeline
enter as usual. None preserves each card's current animation.


`prompt_bar` types a request and confirms it with a check. Its request and optional
placeholder share a fitted size. Word cues anchor typing; one confirm cue can
follow the last character. Use a hold of at least 2 seconds.

`comment_cta` types a reply keyword into a comment field with an optional label.
The keyword and placeholder share a fitted size of 30 to 44 px. Word cues anchor
typing; one send cue can follow the last character. Use a hold of at least 2 seconds.

`comment_drop` drops a comment bubble beside a circular avatar. Comments fit up to
two lines at 30 to 40 px. The avatar uses a local image, explicit initials, an
absolute `brand.logo` image path, or initials from the name, in that order. Local
avatar paths resolve from the project folder; the brand logo may live outside it.
Use `side` to align left or right and `drop_at` to set the arrival time. Use a hold
of at least 1.6 seconds, leaving 0.6 seconds to land and 0.6 seconds to read.
Use single spaces between words, with none at the start or end.

`icon_strike` crosses out a local icon image or one of six neutral glyphs: `app`,
`bell`, `chat`, `mail`, `cart` and `clock`. Its optional label fits one line at
30 to 56 px. The slash uses the accent when it contrasts with the tile, otherwise
a contrasting colour. Use `strike_at` after the 0.6 second entrance, leaving 0.28 seconds
for the strike and 0.3 seconds of rest. The minimum hold is 1.7 seconds, which
supports the default 0.8 second strike at 24 and 30 fps.

`choices` builds a bare text stack as she lists 2 to 4 options. Text settles at
56 to 64 px and lands at 1.6 times that size with an accent glow and a dark core.
It tries every single-line size before wrapping long options to two balanced,
tightly spaced lines at word boundaries, keeping the letter on the first line.
Rows leave room for each landing. The stack fits beside the head, in safe headroom
above it, or below it with caption clearance. It stays below the style's safe text top.
When the planner reports no room above or beside the head, use fewer or shorter
options or move the card to another moment with more room above the head. Use
`options` with 1 to 24 characters each, optional `cues` in seconds from the card
start, and `letters` to show or hide A), B), C) and D). In cards plans, cues may
be word anchors on each option. Cards with word cues keep their word timing when
beat snap is on. Keep her wording, shortened at word boundaries.
Each option finishes settling before the next cue. Cues need at least the landing
and settle time rounded up to whole frames, about 0.4 s apart. The minimum is
2.0 s; three or four options need a longer hold to keep the cue spacing.
It ends with a quick clear. End the card on the first word after the cut that
follows the last option, or on the word that finishes the point.

Editor cards are `edit_timeline`, `storyboard`, `jiggle_delete` and `file_drop`.
They stay off until the creator turns them on, on the style page or in
`graphics.card_kinds_on`.

- `edit_timeline` draws the video's own tracks: Video (the cut's clips), B-roll
  when there is any, and Cards. It shows the output timecode, and a playhead moves
  in real time while the card is on screen. `view: near` shows about 8 seconds
  around the card and `view: whole` shows the full video. `title` is optional.
- `storyboard` shows the edit's named segments as thumbnails. Each thumbnail is the
  picture at the middle of its segment, from the cut or from an insert on screen
  there. The segment on screen gets a ring. `count` is 3 to 7 and `numbers` is
  `show` or `hide`.
- `jiggle_delete` wiggles 2 to 6 chips, or image tiles when `images` is set, then
  removes the 1-based positions in `remove` at its `cues`.
- `file_drop` drops a file card for a `clip` in the project folder, with a still
  from the clip at `poster_at` percent, the file name or `name`, the clip's length
  and a `badge` (RAW by default). `playback: still` is the default. Set
  `playback: play` to play the clip muted from the poster frame after the card settles.

`edit_timeline` and `storyboard` read the project, so they render in the project
run. `templates preview` and the style page use sample data for them.

Video in cards uses transparent windows with ffmpeg placing each clip under the card
frame. `video_card` takes `clip: {"path":"broll/clip.mp4","sha256":"<64 lowercase hex>",
"in_s":0,"speed":1,"crop":[120,80,640,360],"focus":[0.5,0.5],"audio":"mute"}`, an optional `label`, and
`aspect` (`source`, `16:9`, `4:3`, `1:1`, `4:5` or `9:16`). Keep the file inside the
project and get its hash with `shasum -a 256`. Sound stays muted unless `audio` is
`{"gain_db":-6}`. The card needs at least 1.5 seconds and enough clip duration for its hold.
Optional `crop: [x, y, width, height]` uses displayed pixels after rotation, stays inside the
picture and is at least 32 by 32 pixels. Omit it for the whole picture. `self` takes no crop.
Cover fitting and focus apply inside that crop.
For screen recordings, crop to the part the speaker talks about so its text reads on a phone.
Aim for the smallest text to be about 24 px tall in the final video.
`playing_grid` plays two to six clips in fixed tiles around the speaker, with optional label chips.
`reel_feed` fills the frame on purpose with a generic phone and an inline post, using `post: "self"`
by default so the picture inside the phone stays in sync with the voice.
Its `backdrop: auto` chooses paper or ink for the strongest contrast with spoken captions;
explicit paper refuses caption contrast below 3:1, while ink and dim remain explicit choices.
The default `ending: handoff` needs `post: "self"` and 2.5 seconds, then swipes into the
same video at full frame without a clock or audio jump; use `ending: fade` for a media post.

## Big numbers

`big_number` shows a bare count-up above the speaker, tucked behind her hair.
It stays off until the creator adds it to `graphics.card_kinds_on`. Use only a number
she says near its anchor, and leave at least 2.1 seconds. Slots are `end`, optional
`start`, `prefix`, `suffix` and a short `label`. Values allow up to two decimal places.
Graphics checks every frame against the person matte. If the number cannot stay
90% visible, it sits beside her head in front and reports a warning. The final
checks the rendered ink and refuses more than three frames below 90% visibility.

## Video in letters

Choose `hook.style: video_letters` for an opening title whose letters open onto
the speaker's take. Write one or two spoken words in the title's `lines`.
Without `end` or `hold_s`, it lasts 48 frames. Other title windows must be 40 to 59 frames.

For an insert, add `letters: {"text":"DOG PARK","background":"color"}`.
`background` is `color` by default or `blur` for a dark blurred copy of the fill.
The opening title always uses color. Inserts starting before frame 30 use
`hook.font` and `hook.accent_color`. Later inserts use `graphics.font` and
`graphics.accent_color`. These accents follow `brand.accent_color`.
Opening inserts need `hook.style: video_letters` or the emphasis opt-in.
Add `video_letters` to `broll.emphasis_on` to allow inserts anywhere. Existing
styles keep this empty. More than two mid-video letters inserts gives a warning.

`word_hold_s` defaults to 1.2 and accepts 1.0 to 1.5 seconds. `reveal_frames`
defaults to 12 and accepts 10 to 14. At 30 fps the letters phase is
`round(word_hold_s * 30) + reveal_frames`, from 40 to 59 frames. A clip must last
at least that long and carries on full screen. A photo must last exactly that
long and cuts back to the speaker on the next frame. Photo requests without `end` or
`hold_s` get the exact hold automatically, 1.6 seconds with defaults.

The word uses upper case and weight 900. Text fits 960 px with 60 px side margins,
at the largest size up to 640 px. Two words can stack with a shared size and
0.9 em line height. The block is centered at y 860, above the caption band at
y 1503. The blur has sigma 30 px and brightness 0.45. The reveal grows from
inside a thick letter stroke and fades the plate over its last two frames.

`timeline check` refuses copy that needs less than 150 px and names the size and
about how many letters fit, so choose a shorter spoken word. Text must have one
or two words, at most 40 characters. Its words must occur consecutively in the
spoken line within the element window plus 90 frames on either side. Case and punctuation
do not matter. A refusal names nearby words to use. Disabled styles and wrong
lengths are also refused; the length fix gives the exact `hold_s` to write.

Gravity drop lands emphasis words in mixed creator fonts on their spoken words.
Each word falls for up to 0.26 s (shorter when the word is spoken in the card's
first quarter second).
Turn it on in the style page or add `gravity_drop` to `graphics.card_kinds_on`.
Each drop has 1 to 20 characters, with 1 to 4 drops per card and at most two rows.
Landing times follow reading order and leave 0.5 seconds before the exit fade.

```json
{"id":"agent-drop","kind":"graphic","template":"gravity_drop","start":{"word":"t1:410"},"hold_s":2.5,"slots":{"drops":[{"text":"Tokens","at":{"word":"t1:412"}}]}}
```

## Chapters

Choose Title card under Chapter title card on the style page. The default is None.
Add chapters in the agent lane with a short title:

```json
{"id":"agent-ch-tokens","kind":"chapter","start":{"word":"t1:412"},"title":"How tokens are counted"}
```

The title has 1 to 48 characters. Use `end` or `hold_s` to set its window, or omit
both to use `chapters.hold_s` (1.6 seconds). Cards must last 1 to 4 seconds and
cannot overlap. The engine numbers chapters in output order as 01, 02 and so on.
The timecode comes from the resolved start frame at 30 fps, with seconds floored.
It reads `MM:SS`, or `H:MM:SS` from one hour.

A chapter is a timeline element because it starts at a spoken word on the output
clock. Anchors follow re-cuts, check overlaps and feed receipts. The number, title
and timecode sit over the live picture. The speaker stays visible through a
45 percent colour wash. A soft band rises to 70 percent opacity behind the text
and fades back to the wash above and below. The whole layer fades in and out.
Captions stay on top and follow your face placement choice during chapters.
The wash uses the graphics paper or ink colour with higher contrast against the
caption text. The text colour is chosen to stay readable over dark and light
footage, with a readable fallback when needed. A soft shadow uses the wash colour.
The title uses the titles font and the rule uses the graphics accent colour.
A chapter does not cover a voiceover picture gap. Add an insert to supply picture
there.

## Backgrounds and overlays

An insert with `media.fit: "split"` puts the shot on top and the creator below,
with a light seam and cuts at both ends. Captions stay below the seam. Use split
when the style selects it or a review comment asks for it. Timeline warnings
suggest at most three insert starts per minute, at most 25% full-screen B-roll,
at least 65% face time, and standalone inserts of at least 1.2 seconds. Montage
groups count once for the rate. Voiceover-only videos skip the screen limits.
A shot showing a named thing should start on that word with `offset_s: -0.1`.

Timeline graphics include `title`, `chapter`, `graphic`, `overlay` and `background`.
`background` fills the frame with an image, page screenshot or silent clip while
the person matte keeps the speaker in front. It takes the same `media` fields as
`overlay`, plus `fit: cover|contain`, and takes `start` and `end` anchors or
`hold_s`. For example, `{"id":"broll-page","kind":"background","start":{"word":"t1:12"},
"end":{"word":"t1:18","edge":"end"},"media":{"path":"broll/page.png",
"sha256":"<64 hex>"}}` changes the room only for that span. A missing person
matte refuses assembly.

Takes shot on a green or blue screen use a colour key when at least half the side
and top edges share one flat screen colour across eight sample frames. The bottom
edge is left out because the body often fills it. Set `broll.cutout` in the project
style to `screen` to force a key, `person` to use person detection, or `auto` for
automatic detection. A thin edge band removes screen colour spill while keeping
clothing inside the person unchanged. `yap-studio status` shows which cutout the
project used.

Inserts and overlays match the colour of the nearby take by default. Set
`"broll": {"color_match": "off"}` in the style file to switch it off. An element's
`color_match: true` or `color_match: false` overrides the style for that shot.
Screenshots and screen recordings are planned with `false` to keep their own
colours. The speaker's own picture never changes; only the inserted or overlaid
shot does. Voiceover videos have no take to match, so their shots keep their colours.
Video cards playing your clips match too; cards showing you do not.

Person matte smoothing stops at every shot join and averages only where the picture is still.
The edge keeps the person mask's own detail with no blur.
When a person sits over a card or a new background, edge pixels lose the old background's colour.

Four optional fields add the green screen look. A background without them looks
as before.

- `layout` says where you sit. `full` (the default) keeps you full size. `bottom`
  scales you to 65% and sits you at the bottom centre, so the background shows
  above you. `corner_left` and `corner_right` scale you to 45% in a bottom corner.
  Where your frame ends inside the picture, its edge fades out over 24 px.
  Captions and cards avoid your face in its new place.
- `blend: wrap` blends you into the background. Light from the background, blurred,
  shows at 25% on a thin band at your edge (about 18 px wide). Your colours move 12%
  toward the background's average colour, at most 12 levels per channel, with
  brightness kept. Skin moves a third as much. The receipt records both.
- `enter: wipe` brings the background in from the top over 9 frames (0.3 s). The
  background needs at least 0.6 s for a wipe. It always ends on a cut.
- `motion: push` slowly pushes a photo or screenshot, using the style's
  `still_push` direction and `still_push_scale`. Clips cannot push, and they play
  without sound.

A style turns this on with `broll.photo_placement` or `broll.screenshot_placement`
set to `green_screen`, and `broll.green_screen_layout` for where you sit. The
B-roll plan then writes all four fields for those photos and screenshots. A
request's `layout` overrides the style.

Timeline lanes and their fragments are described in [commands.md](commands.md#timeline-lanes).

`grade_compare` is an agent-lane picture effect with `start` and either `end` or
`hold_s`. It compares the raw picture on the left with the chosen grade on the
right for 2 to 8 seconds. It needs an active grade. Use at most two windows, away
from overlays, backgrounds, transitions and other compare windows. Inserts and
zooms can be compared. Captions, titles and cards stay above the wipe.

The divider crosses the whole frame, including the face, by design. The knob sits
at 66 percent of the frame height. Both sides show the creator's own picture.
The handle is not a card and does not enter the layout record or card
face-clearance checks.

## Moments

A `moment` is a short picture beat on a punchline, key claim or reveal. Enable each
look under `moments.styles` in the project's style: `color_pop`, `background_blur`,
`spotlight`, `sticker_freeze`, `outline` or `clone_flash`.
Existing styles keep all of them off.
Use the agent lane, a required `style`, word `start`, and word `end` or `hold_s`.
Moments need the cut's person matte and last 0.8 to 3 seconds. Sticker freezes use
a 0.8 to 1.4 second window, including the return to live picture. Audio keeps playing
and the video stays the same length.

`clone_flash` flashes one or two moving copies of the speaker from later in the same take.
Clone flash is refused on green or blue screen takes; pick another Moment there.
Keep titles, cards and overlays out of a clone flash window.
Each copy is sized to the room on its side. Moments follow the active colour grade.
Use a 1 to 2 second window, a still camera and room beside the speaker's face clear of the captions.
It needs measured faces and enough of the same take left after the line. If assembly
refuses it, follow the fix or drop it.

Use at most two, with starts at least 10 seconds apart. Extras and close starts warn.
A moment cannot overlap an insert, background, zoom, transition, grade comparison,
behind graphic or another moment. Titles, front graphics and captions stay above it.

## Covers

`yap-studio cover DIR --headline "A *better* night"` makes a 1080×1920 vertical cover
or a 1280×720 horizontal YouTube thumbnail from the overlay-free cut at a frame of
the finished video. Asterisks mark one italic
accent phrase. The saved style supplies the fonts and colors. `--frame N` chooses
the frame; otherwise the command picks a sharp, open-eyed face.

The default `photo` look needs `--headline` and accepts `--kicker` and `--inset IMAGE`.
`number` needs `--number` and `--label`. Vertical-only `hook` needs `--headline` and a proof `--label`
and accepts `--kicker`. `before-after` needs `--before` and `--after` and accepts
`--label` and `--kicker`. Vertical covers check the centered 3:4 profile-grid crop;
horizontal thumbnails support `photo`, `number` and `before-after` and check every
visible head, safe margins, the duration-badge corner, and surviving strokes and
contrast at 168×94, including the marked accent phrase separately.
Wide copy sits directly on the photo with a soft fade,
on the opposite side from the face. Automatic wide picking samples up to 120 moments
across the edit, including the start, midpoint and last frame of speaker windows
within that budget, ranking eye openness, sharpness, lighting and room for text.
Existing Vision data does not measure mouth openness. The command
writes `covers/<name>-<look>.png` plus `covers/<name>-<look>.json`.

## Stock clips

Stock clips are off unless you allow them. Pick Stock clips when the agent asks where it may
get B-roll for a video. A project with no B-roll sources recorded follows `broll.stock_clips`
in its style, which allows stock when set to `"ask"`. When stock is allowed, the agent searches
Pexels for moments your own media does not cover, shows you one pick sheet
and asks which codes you want. Yap Studio downloads only the clips you approve, and
the agent adds them to the B-roll plan. They play muted and are cropped to fill the
frame, like other inserts. If Pexels is not set up or the Mac is offline, the agent
says so and carries on with your own footage. A stock clip used in a card needs the same
approval, whether a video card plays it or a file drop shows its poster.

Pick sheets live in `DIR/stock/sheets/sN/`. Your approvals and their credits live in
`DIR/stock/clips.json`. Downloads are kept in `assets/stock/pexels/ID.mp4` under the
Yap Studio config folder, named by provider and clip id, so approving the same clip
again reuses the file, even offline. `yap-studio stock credits DIR` prints credits for
the clips placed in the video; `--all` lists every approved clip.

The [Pexels License](https://www.pexels.com/license/) allows free use and changes,
with optional credit. People shown must not appear in a bad light or seem to
endorse a product. Do not resell clips unchanged or use a clip in a trademark or
logo.

## Generated pictures

Generated pictures are off by default. With `broll.generated_pictures` set to `"ask"`,
the agent first asks "Do you want me to create imagery for this video?" once per video.
Neither answer is preselected. The answer lives in `DIR/pictures/imagery.json`.
On no, there is no sheet and no price check. `pictures ask` and `pictures make` refuse
until the answer is yes. After a yes, the agent drafts one prompt per line your media
does not cover. It shows one sheet with each prompt, its estimated cost and the total.
Nothing is made until you say yes. It makes at most three pictures at a time under a
spend ceiling you approve. Each picture is paid on your own WaveSpeed account. The
ceiling is checked against WaveSpeed's quotes before each picture. WaveSpeed has no
price cap on a request. The run stops if a charge comes in above its quote.
The answer is checked before each picture, and revoking it stops the remaining pictures.

Sheets live in `DIR/pictures/sheets/sN/`. Files live in `DIR/pictures/files/`.
Approvals and ready B-roll requests live in `DIR/pictures/pictures.json`. The media
catalog records the provider, model, prompt, seed, cost and date, and marks each
picture as generated. A generated picture needs the same approval as stock before
it goes in a video, including in cards. If WaveSpeed is not set up, the agent says
so in one line and carries on with your own footage. The message ends with
"Sign up for WaveSpeed (affiliate link): https://littlemight.com/recommends/wavespeed".
Attempts live in `DIR/pictures/attempts.json`. A later make recovers an accepted
prediction for free if saving its picture failed. Use `--new` only when you approve
paying again for an uncertain attempt.
If a submission returns no prediction id, its quote counts as possible spend and
the batch stops.

[WaveSpeed's terms](https://wavespeed.ai/static/terms) say you own what you generate.
You are responsible for checking that it does not use someone else's rights before
you publish. The terms forbid deepfakes of a real person without that person's consent.
