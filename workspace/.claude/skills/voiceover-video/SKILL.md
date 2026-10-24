---
name: voiceover-video
description: Make a voiceover video from the creator's audio memos over their own Photos album or media folder.
---

# Make a voiceover video

A voiceover video is told only in the creator's recorded voice, over their own photos and clips,
often a Photos album of a project or a day. Nobody is on camera. FOLDER is the creator's media
folder. DIR is the project folder that `new_project.py` prints. Run commands from the workspace
root. Before DIR exists, keep your notes in `drafts/TOPIC/`; after, put the files you write in
`DIR/plans/`. When unsure what comes next, run `yap-studio status DIR`.

## Rules

- Follow the Rules in `.claude/skills/edit-video/SKILL.md`. They all apply here.
- For Text hook, follow Text hook there too. Wait for the creator's text before rendering it.
- Never change the creator's media. Work from FOLDER and from copies.
- The script says only what the footage shows or what the creator told you. Never invent a
  fact, number, name, or date. Ask instead.
- One shot per moment. A photo and its Live Photo motion, burst frames, and the same subject a
  few minutes apart are one moment. Use at most one of them.
- Ask before anything that costs money, such as ElevenLabs music.
- Ask early, before choosing B-roll, whether children, friends, or others may appear on screen.
- Never reuse a shot for the closing frames.
- When a graphic or diagram has an obvious home in the script, put it there and say so.

## Procedure

Reviews use `--detach` and the phone link exactly as in edit-video.

1. Media in.
   - Use only this video's B-roll sources: `yap-studio video-type show --json` before DIR exists,
     `yap-studio status DIR` after. If B-roll is not chosen yet, ask only the B-roll question as
     in What are you making? in `AGENTS.md`. `album` is a Photos album, `folder` a folder. With
     `stock`, fill moments their media cannot cover with Stock as in edit-video step 5. For a screen
     recording told in voice only, the screen recording is the picture: put it in FOLDER and
     cover every frame with it, a later `in_s` for each line.
   - From a Photos album, run `yap-studio media album "ALBUM" --out FOLDER --dry-run` first. It
     lists the album, counts edited items and items not downloaded to this Mac, and changes
     nothing. If it finds no items, it names the Photos library it searched; for another library
     add `--library PATH`. If osxphotos is missing, offer `uv tool install osxphotos`, or ask the
     creator to export the album by hand into FOLDER (in Photos: select the album's items, then
     File > Export > Export Items, which keeps their edits).
   - The export takes the edited version of each item the creator edited in Photos (a crop, a
     trim, an adjustment) and the original of the rest. It never writes to the Photos library.
   - If items are not downloaded, tell the creator before exporting: Photos will download them
     and macOS may ask this Mac for permission, so they should be at the Mac to click Allow. With
     `--library`, Photos must have that library open. Then run
     `yap-studio media album "ALBUM" --out FOLDER`. Add `--no-download` only when the creator
     wants just what is already on this Mac. If it stops with no progress, ask them to click
     Allow or export by hand, then run it again. Exported files are kept and skipped next time.
     If the summary says some items did not download, run it once more; ask the creator to export
     any that still fail by hand. `FOLDER/album-report.json` lists each file with its capture
     date and whether it is the edited version or the original.
   - A folder the creator points to works the same from here on.
   - Run `yap-studio media add FOLDER`. For every new item run `yap-studio media sheet ID`.
   - Read every sheet. After each batch of 5 to 8, write a note per item to
     `drafts/TOPIC/notes.jsonl` (id, capture date, what it shows, quality, speech) before reading
     more; older images drop out of your context. Capture dates come from `album-report.json` or
     `camera.captured_at` in `yap-studio media show ID --json`.
   - Assign phases (for a build: before, digging, plumbing, finished) and meaning only after you
     have seen everything. Early guesses are often wrong; rewrite notes that later footage
     corrects. Group each moment and pick its best shot.
   - For each item write `{"schema": "yap-studio/media-description@1", "description": "Phase:
     ... Captured ... Shows ... Quality ...", "tags": [...], "provenance": {"kind": "own",
     "source": "Photos album: ALBUM", "license": null, "url": null}}` and run
     `yap-studio media describe --file FILE ID`. Leave provenance out for media the creator did
     not make, and ask about it.
2. Script.
   - Write `drafts/TOPIC/footage-log.md`: each phase in capture-date order, its moments, the
     best shot of each, and gaps the footage cannot show.
   - Run `yap-studio taste show` and read `about` in `style/answers.json`. Write
     `drafts/TOPIC/script.md`: short, in the creator's own voice, hook first, built from what the
     footage shows. List anything you are unsure of as a question.
   - Show the script and stop for the creator's approval. Apply their edits.
3. Record. Ask the creator to record the approved script in Voice Memos or any recorder: phone 8
   to 10 inches away and off to one side, in a quiet room. To redo a line, say it again, or record
   a short pickup memo later. They drop the memos in `inbox/`. Wait until they say they are there.
4. Project and story.
   - Use the script pass and settle repeat warnings as in edit-video step 3.
   - Run `python3 new_project.py MEMO PICKUP...` with the memos in order (the first is the main
     take), or `python3 new_project.py` for the newest unused memo. An audio memo starts a
     voiceover project with a black stand-in picture. A pickup recorded later joins with
     `python3 new_project.py --add DIR PICKUP`.
   - Run `yap-studio ingest DIR`, then `yap-studio words DIR`. If ingest or status warns that
     speech has no words, listen to that part (often the ending) before you use or drop it. Name
     the project with `python3 new_project.py --name DIR TOPIC` and use the DIR it prints.
   - Proofread and fix misheard words as in edit-video step 2 before choosing the story.
   - Run `yap-studio edit schema` and write `DIR/edit.json`. A memo usually holds several reads of
     each line; keep the best one. Put back-to-back sentences in one segment. Start each segment
     on a word that follows a pause; if the word you want has none, start one word earlier and
     say so. Run `yap-studio edit check DIR`.
   - Run `yap-studio fillers DIR --max-pause 0.3`. It saves that limit as the project's
     `max_pause_s` and proposes filler cuts and trims of pauses longer than it, measured from the
     audio so a word stretched across a pause keeps its sound. Listen at the `listen:` times.
     Reject a cut that clips a word with
     `yap-studio fillers set DIR ID rejected`; add a missed one with
     `yap-studio fillers add DIR --manual TAKE START END`; never edit `removals.json` by hand.
     Then run `yap-studio fillers DIR --apply` and `yap-studio cut DIR`.
   - Plan B-roll only after the words are final. If a transcript changes later (a pickup or a
     new ingest), run `yap-studio edit rebase DIR --dry-run`, then `yap-studio edit rebase DIR`,
     instead of re-anchoring by hand.
5. Story review by ear. Run `yap-studio versions save DIR --note "Story cut"`, then
   `yap-studio review --versions DIR --phone --json --detach`. Say the picture is a black
   placeholder until B-roll goes in, and this review is for the words and pacing. Apply notes as
   in edit-video step 4. Repeat until the creator is happy with the story.
6. Picture. Plans read `DIR/cut/cut-words.json` and take `--frames N`, the cut's frame count
   (`plan.total_frames` in `DIR/cut/receipt.json`), and `--style` as in edit-video step 5.
   - Give each line of the cut a phase. Within a phase, shots follow capture date. Only the hook
     and a stated before and after may jump; mark those requests `"chronology": "hook"` or
     `"chronology": "before_after"`. Never show the finished result early.
   - `yap-studio broll suggest --cut-words DIR/cut/cut-words.json --style STYLE` ranks items
     against the words. Write `DIR/plans/broll.json` as in edit-video step 5:
     `{"schema": "yap-studio/broll-requests@1", "broll": [...]}`, every request `kind` `insert`.
   - Cover every frame. End each insert where the next one starts:
     `"end": {"word": "t1:40", "edge": "start"}` when the next starts at `t1:40`. Start the first
     at `{"time": 0}` and end the last at the cut end with `{"time": S}`, where S is N divided by
     30. After a new cut, update S; `yap-studio status DIR` reminds you.
   - For a long locked-off clip, use one muted timelapse (`"speed"` up to 20 with
     `"audio": "mute"`) and give each use a later `in_s`, so progress shows. Never repeat a window.
   - Crop with `"crop": [x, y, width, height]` in source pixels. Stills get a slow push. A still
     with text near its edge gets `"motion": "still"`, or pad a copy 8% in its own background
     color before the push; check its last frame.
   - To keep a clip's own sound (a splash, a short line), run `yap-studio media sound ID` for
     its moments and `in_s`, and set `"audio": {"gain_db": -18}`.
   - Run `yap-studio broll plan DIR/plans/broll.json --cut-words DIR/cut/cut-words.json --style
     STYLE --frames N --out DIR/plans/broll`, then `yap-studio timeline add
     DIR/plans/broll/broll-lane.json DIR --lane broll`. When you plan the lane again, add
     `--replace-lane` so old inserts do not stay.
   - Run `yap-studio timeline check DIR` until it passes and names no uncovered frames. Read each
     warning: dates going back, near duplicates, repeated sources, insert length. Fix it or keep
     it on purpose. The warning that media is outside the project is expected.
7. Music. Use a track the creator supplied with `yap-studio music DIR --file TRACK`. ElevenLabs
   can compose one: `yap-studio music DIR --yes --dry-run` shows the request without spending,
   and `yap-studio music DIR --yes` spends the creator's credits, so run it only after they
   approve the cost. If they have no ElevenLabs account, give them
   https://littlemight.com/recommend/elevenlabs and say it is an affiliate link. For voice only,
   run `yap-studio music DIR`. Then run `yap-studio timeline check DIR` again.
8. Run `yap-studio run DIR --note "Final" --json --detach`.
   It prints its JSON and returns while the render keeps going. Do not run it as a background task.
   A long final can take more than 30 minutes. Run `yap-studio status DIR --json --wait 540`
   until `run.state` is no longer `running`. Each call returns within 9 minutes.
   Read `run.result` as in edit-video step 6. `yap-studio run DIR --stop` stops it.
   Finish as in edit-video steps 7 and 8: the whole-video review,
   a `feedback reply` with `--version vM` for every note, the creator's
   approval, and taste proposals for lasting preferences. After any pause trim, read
   `yap-studio words DIR --cut` and check the captions show every word that is still spoken.

Do not call the video finished because the render passed checks. The creator decides.
