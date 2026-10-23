# Edit from the command line

In the workspace, the agent runs these commands for you. This page is for running them
yourself, or for seeing what the agent does. Install first: [install.md](install.md).

## What each stage does

- Starts a project from raw takes with `init`, then transcribes them with `ingest`.
- Proposes filler, restart and pause cuts. `fillers --apply` applies verified removals.
- Renders the selected story with `cut`, then captions, graphics and a supplied music track
  with `captions`, `graphics`, `music` and `assemble`.
- Places word-anchored titles, cards and media through the timeline, and checks the result.
- Saves versions and review comments, and keeps approved creator taste rules in a private store.
- Verifies cuts, sound, sync, frames, loudness and delivery rules in a finished video.

For a voiceover over your own photos and clips, `yap-studio init DIR --voiceover MEMO.m4a`
starts the project from an audio memo. The [voiceover recipe](recipes/voiceover.md) walks
through it.

## Walk-through

Use your own short vertical talking-head take with audible speech. From a local
working folder, initialize and transcribe it:

```console
yap-studio init project --take take.mp4
yap-studio ingest project
yap-studio words project
yap-studio edit schema
```

Add `--video-type talking_head --broll folder` to `init` to choose this video's recipe and
B-roll sources; repeat `--broll` for more sources, or use `--broll none` for none.

If you used `new_project.py`, skip `init` and replace the `project` argument in
each command below with the directory it printed, quoted if it contains spaces.
The engine needs an `edit.json` that selects the story; it does not choose your
story automatically. This starter keeps the first take from its first recognized
word to its last. Run it once after ingest:

```sh
python3 - project <<'PY'
import hashlib
import json
import sys
from pathlib import Path

project = Path(sys.argv[1])
raw = (project / "ingest/t1/words.json").read_bytes()
words = json.loads(raw)["words"]
if not words:
    raise SystemExit("No words found; check the take's audio and ingest again.")
edit = {
    "schema": "yap-studio/edit@1",
    "words_sha256": {"t1": hashlib.sha256(raw).hexdigest()},
    "segments": [{
        "id": "story",
        "take": "t1",
        "from": {"word": words[0]["id"]},
        "to": {"word": words[-1]["id"]},
    }],
}
with (project / "edit.json").open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(edit, indent=2) + "\n")
PY
```

Read the word listing and edit `project/edit.json` to keep the spans you want.
Each segment names a take and its first and last word IDs; segments play in list
order. Give each segment a unique `id`. Keep `words_sha256` tied to the current
ingest file. For this first video, use the default style and no music or graphics.

Check the edit, apply agreed filler removals, and render a cut for story review.
Only removals applied before that review reach the final. `run` never applies new removals:

```console
yap-studio edit check project
yap-studio fillers project --apply
yap-studio cut project
yap-studio review project/cut/cut.mp4
```

`edit check` and `cut` warn about repeated phrases and points in the kept story, with times in
the cut. Keep the clearest telling, or keep both on purpose and say why in the version note.

Close the review server with Ctrl-C when finished. If you change `edit.json` or
removal decisions, repeat the check, fillers, cut and review commands above.
Once the story is approved, keep those inputs unchanged and run the remaining
stages to render captions, prepare the audio, assemble and verify the final, and
save a version:

```console
yap-studio run project
yap-studio status project
yap-studio review --versions project
```

The final is `project/final/project.mp4`; the checks are in
`project/final/receipt.json` and `project/final/verify-edl.json`. `run` stops at a
failed check and prints the reason and next action; fix it and rerun to resume.

## Captions for a clip you already edited

For captions on an already edited vertical clip, you can also run:

```console
yap-studio captions take.mp4 --style preset:clean_minimal --out captioned.mp4 --srt
```

To check a finished video on its own, see [verify.md](verify.md).

## Remove fillers and restarts

```console
yap-studio fillers project
yap-studio fillers project --apply
```

The first command proposes removals; `--apply` applies those that pass the join test.
A join that could not be verified stays proposed with a listen flag.
Every speech removal is checked by comparing a spliced clip with its uncut control,
preserving the surrounding words and checking for clicks.

Whole-take speech transcripts can hide false starts. Fillers checks the planned segment's
audio in 10 ms frames at the existing -45 dB silence floor. A word containing at least
0.15 s of quiet with speech on both sides, or a gap of at least 0.2 s containing at least
0.08 s of speech, triggers a local transcription. Clips include neighbouring words and
at least 1.2 s of context; overlapping clips merge and all refinement clips run in one
speech batch. These temporary words never rewrite `words.json` or `cut-words.json`.

The restart rules detect a two- or three-word phrase that comes back within two seconds,
and fragments followed within two seconds by a longer word with the same beginning and
matching preceding words. Inside a refined clip, a repeated phrase separated by one
abandoned non-filler word also proposes a restart. The earlier attempt is removed and the
later, complete copy is kept. Each proposal goes through the same join test. A single word
repeated for emphasis, such as "very, very" or "really really", is kept. The other emphasis
words kept by this rule are "no", "yeah", "bye", "ha", "na", and "so".

`fillers/report.json` records how many spots and clips were refined, the proposals from
each restart rule, and refined text with source times alongside existing source word IDs.
Without a speech engine, refinement and speech removals are skipped and the report says why.

## Run every stage

`yap-studio run DIR` does the rest of the edit once the takes are ingested and
`edit.json` is written. It keeps the cut's removals when the cut is current. Otherwise
it runs fillers for proposals only. It never applies new removals. It runs cut, music,
captions, graphics and assemble, checks the final against the project's `platform` rules,
and compares its words with the reviewed version. The words check reports every dropped
word and why. An unapproved or unexplained dropped word stops the run with `words_dropped`
before a version is saved. A passing run saves the final as the next version with its caption and
timeline windows, so review comments on a caption or card find it. It stops at the
first failed check and prints the reason and the fix. Filler notes and cut warnings
are printed as warnings. Run it again after the fix: finished
stages are skipped, so it carries on where it stopped. It exits 0 when the final is
saved, 1 when a check failed, 3 when it refused to overwrite or the project was busy,
and 2 for other errors. A final that is already the newest version is not saved twice.
`yap-studio run DIR --frames A-B` runs through graphics, renders that pilot range,
and skips final verification, the words check and version saving.

A long final can take more than 30 minutes. `yap-studio run DIR --detach` starts the run in
the background and returns at once, so a terminal or an agent's command time limit cannot stop
it. `yap-studio status DIR` then shows the stage it is on, its progress and, once it ends, the
same result `run --json` prints. `yap-studio status DIR --wait 540` waits up to 9 minutes and
returns as soon as the run ends. `yap-studio run DIR --stop` stops the run and the tools it
started; a rerun carries on from the last finished stage.

`status` reports the stages and the next command. Unchanged inputs skip rendering;
`--force` renders again while preserving receipt ownership checks. Inserts, zooms
and other graphics do not invalidate captions, and caption-style changes do not
invalidate titles. Standalone `captions CLIP --style STYLE --out OUT.mp4` keeps its
existing behavior, including optional `--title` and `--srt`.

## Timeline elements

Timeline element kinds include `title`, `chapter`, `graphic`, `insert`, `overlay`,
`background`, `zoom`, `transition`, `caption_gap`, `music_level`, `moment`,
`grade_compare` and `sound`. Use `sound` only when the creator asks for a sound at
a moment. It has an agent lane, a `start` anchor and a soft or extras kit `sound`
name, with optional `gain_db` from -12 to 6 and `align` set to `start` or `peak`.
It has no end anchor or hold. See [Sound effects](audio.md#sound-effects).

## Camera moves

A zoom is a timeline element on the agent lane. The style's `moves.zoom_mode` says which
zooms the agent uses: `none` (the default), `punches`, or `punches_and_push`. With `none`, add
a zoom only when a review comment asks for one. Punch in on a word that carries the point,
such as a key number or the line where a joke lands, and use a few per minute at most. A
punch moves in over two frames, holds for at least 0.6 seconds, and moves back out over
four frames, landing at the end of its `end` word. A push moves in slowly across a section
and holds, or ends on a cut. Save a fragment like this with word IDs from your cut, then
add it with `yap-studio timeline add FRAGMENT DIR --lane agent`:

```json
{
  "schema": "yap-studio/timeline@1",
  "elements": [
    {
      "id": "agent-punch", "kind": "zoom", "lane": "agent",
      "start": {"word": "t1:12"}, "end": {"word": "t1:16"},
      "move": "punch"
    },
    {
      "id": "agent-push", "kind": "zoom", "lane": "agent",
      "start": {"word": "t1:30"}, "end": {"word": "t1:48"},
      "move": "push", "scale": 1.2, "push_s": 4
    }
  ]
}
```

- `scale` is the framing the move holds or reaches, up to 2. A punch without `scale` uses
  the style's `moves.punch_scale`, 1.5 or 2.
- Scale changes by the same ratio each frame, so moving in and moving out at the same pace
  look equally fast.
- Leave `center` out and the move zooms around the speaker's head, from the top of the head
  to the chin, measured from the cut. The center then shifts just enough to keep the face
  and 24 pixels around it inside the frame, and the chin above the caption band. Captions
  and cards stay where they are. Set `[x, y]` in canvas fractions to aim somewhere else.
- `ease` is `snap` or `cut` for a punch, and `linear` or `ease-in-out` for a push.
  `push_s` sets how long a push moves before it holds; without it the push runs to `end`.
- `"move": "keys"` takes a list of `keys`, each the framing reached at its word: a word
  `at` anchor, a `scale`, an optional `center` and an `ease` (`cut`, `snap`, `linear` or
  `ease-in-out`, default `snap`). With `cut` or `snap` the framing holds until the next key.

`moves.opening_push` is separate from `zoom_mode`. When it is on, open with one
`opening` zoom starting on the first word of the cut, with no `end`. It cuts wide at
the first sentence end or real cut between 0.8 and 4 seconds after the first word,
whichever comes first. A cut in a word's pause is its end point when it fits that
window. If the line runs past 4 seconds, use the first clause end in that window,
marked by a comma, semicolon or colon. If none fits, the engine skips the push with
a warning. It never cuts wide in the middle of a clause.

The usual push reaches 1.3x in about one second. A shorter line stops below 1.3x.
A longer line moves more slowly and still reaches 1.3x at its end. It may sit under
the opening title, which moves up or shrinks if the zoomed face reaches it. Use one
per video. It cannot overlap a card, insert or another zoom.

```json
{
  "schema": "yap-studio/timeline@1",
  "elements": [{
    "id": "agent-opening", "kind": "zoom", "lane": "agent",
    "start": {"word": "t1:0"}, "move": "opening"
  }]
}
```

`timeline check` warns about a move that changes the framing by less than 15 percent, or
that moves slower than 0.03 scale per second, and names its time. Both read as the camera
drifting; use a punch or a push instead, or remove the zoom. Zooms written without `move`
are the older slow swell to at most 1.10, so they get the warning too, and they still
render as before. It also warns when the style does not allow a zoom. It refuses a move that
pushes the face out of frame or the chin onto the captions (use a 1.5x punch there or set
`center`), and a scale change from one frame to the next that no cut or snap explains. A
zoom other than the opening push cannot overlap a card, title or insert.

B-roll overlays are still by default. Set `"motion": "drift"` on an overlay to keep the
slow drift.
Use `"placement": "tall_behind"` for long pages only and `"motion": "scroll"` to move
them up in eased steps. The tall box lasts 2.5 to 8 s and starts and ends on picture
cuts. Start on the first word after a cut and end on the last word before one.
The person stays in front of the page. `broll.tall_framing` sets the framing.
A tall box cannot share its window with a card or title.
`keep` leaves your framing and room as they are. `shrink` makes only you smaller,
in front of a full-height page with paper colour in any side margins.
`auto` keeps your framing when
at least half the box is clear of you and there is enough room above your head in
every sampled frame. Otherwise it shrinks you.
Automatic also keeps your framing when shrinking would bring your face onto the captions.
Auto is measured during assembly;
captions are placed earlier for your normal framing. Normal framing returns on
the cut where the box leaves.
Use Bottom captions, or move captions lower to clear the box.

### Jump cuts

When a filler or a retake is cut out, the head jumps at the join. The style's
`moves.jump_cuts` says what happens there:

- `hard` (the default) keeps the cut as it is.
- `punch` covers each jump cut with a punch zoom. Add it as a zoom element with
  `"move": "punch"` and `"ease": "cut"`, starting on the first word after the join, so the
  framing changes on the same frame as the jump. With `punch`, the style allows punches even
  when `moves.zoom_mode` is `none`.
- `smooth` makes the picture glide across the join. Assemble replaces the 6 frames (0.2
  seconds) around the join with frames generated between the frame before them and the frame
  after them, using Apple's frame rate conversion on the Mac. The length does not change and
  the sound is still a hard cut, so the lips are approximate for those 6 frames. With
  `smooth`, the style allows punches even when `moves.zoom_mode` is `none`.
- `step_in` lets the engine change the framing on cuts. Add no punch on those joins.

Step in walks through 1.15x, 1.3x and 1.45x. It resets at a take change, a pause of at
least 0.5 seconds, a transition, or after three steps. The edit has no section marks,
so these stand in for a section break.
The framing changes only on cuts, with no motion between them. Faces stay inside the safe
area and clear of captions. Steps may sit under titles and front cards. They never sit under
an insert, overlay, background, transition, moment, grade compare, another zoom or a card
behind the head, including big numbers. Step in counts against no zoom cap.
A card placed by `cards plan` keeps its room; Step in stays at a wider framing under it.
`timeline check` reports how many jump cuts changed framing.

Smoothing works only when little moves across the join. The engine compares the frames on both
sides of the window and the faces measured on the cut:

- The face may move up to 0.25 face heights and change size up to 10 percent. A bigger move
  reads as a swoop.
- The picture may change up to 5.5, the average brightness change on a 0 to 255 scale. A hand
  or an object that enters or leaves the picture across the join comes out as a see-through
  ghost, and it shows up as a larger change.

A join that is over a limit, has no face detected next to it, or sits next to a piece shorter
than 6 frames falls back to a punch, even when `moves.zoom_mode` is `none`.
The engine never adds the punch; the agent does. The final receipt lists every join,
what happened to it and why. A join under a zoom, insert, transition, background, moment or
grade compare is left to that element.

`yap-studio joins DIR` prints every join with its time, the words on both sides, what happens
to it and why. `--json` adds the measurements. Smooth joins need macOS 15.4 or later and the
Xcode command line tools; on an older Mac, assemble stops and says to update macOS or to choose
`hard` or `punch`. Each generated window is saved in `.yap/cache/smooth-joins/` in the project,
so the same join renders the same frames every time.

## Transitions

A transition is a timeline element on the agent lane. Use one where the story turns at
a join between story segments, never at every join. Run `yap-studio cut DIR` first so the
word map and cut spans are current. A filler removal inside one segment is not a story join.
Save a fragment with word IDs from your cut, then add it with
`yap-studio timeline add FRAGMENT DIR --lane agent`:

```json
{
  "schema": "yap-studio/timeline@1",
  "elements": [
    {
      "id": "agent-transition", "kind": "transition", "lane": "agent",
      "start": {"word": "t1:30", "segment": "s2"},
      "look": "whip_pan", "frames": 12
    }
  ]
}
```

- `id` starts with `agent-`. `start` names the first word after the join, at its start
  with no offset. Add `segment` when that word occurs more than once in the cut.
- The last word before the join must end with `.`, `!` or `?`.
- `look` is `whip_pan`, `cinematic_zoom` or `light_leak`.
- `frames` is even, from 6 to 24, and defaults to 12. The window is centered on the join,
  with half its frames on each side. Leave at least one content frame before its start
  and after its end. It cannot enter the frozen end hold. Do not set `end` or `hold_s`.
- One join takes one transition. Its window cannot overlap a title, insert, overlay,
  background, zoom or another transition. Keep grade compares clear of it too.

The style's `moves.transitions` lists the looks the agent may use. It defaults to all
three looks, so existing styles work as before. An empty list turns transitions off;
add one only when a review comment asks for it. `timeline check` and `timeline add` warn
when the style does not allow a look. Replace it with an allowed look or remove it.
The warning still allows the lane to be saved and the transition to render.

## Assemble and pilot renders

With a grade picked, an agent-lane `grade_compare` timeline element shows the raw
picture beside the grade. Give it `start` and either `end` or `hold_s`, for 2 to
8 seconds. Keep it clear of overlays, backgrounds and transitions. The style's
Before/after slider choice asks the agent to add one soon after the opening.

After captions, graphics and music have current receipts, run `yap-studio assemble DIR`.
It writes `final/<name>.mp4`, `final/verify-edl.json` and `final/receipt.json`.
Use `yap-studio assemble DIR --frames A-B` to render `pilot/<name>.pilot-A-B.mp4`
with the same layers on those frames as the full render. A range ending inside a
caption or graphic is refused with a suggested complete end frame. Pilots are never
finals or saved versions; `pilot/receipt.json` records the phase timings.
The final ends with a frozen composited frame and a silent hold. Insert audio defaults
to mute; an insert's `{"gain_db": -18}` mixes its audio while voice continues.
`music DIR --file TRACK` saves the supplied track setting and prepares it.
Assemble decodes HEIC and HEIF photos once into `.yap/cache/stills/` in the project and reuses them in later renders.

For every command and its flags, see [commands.md](commands.md). Captions, titles and cards
are in [captions-and-cards.md](captions-and-cards.md); music, sound effects and loudness in
[audio.md](audio.md).
