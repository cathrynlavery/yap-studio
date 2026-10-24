# Yap Studio workspace

This folder is for one creator's videos. Use the bundled Yap Studio engine. Keep raw takes in
`inbox/` and one edit per folder under `projects/`.

For horizontal talking-head videos, follow the Home event instructions below; see docs/limits.md
for unsupported features.

## Set me up

Run `python3 setup.py --json` from this folder. If `refused` is set, explain why.
A synced folder can make video files go missing mid-edit. Low disk means there is not
enough space. Offer the `move` command if one is given. Run nothing until the creator
agrees. After a move, continue from the new folder.

Otherwise, take the `steps` one at a time. For each step, say what `does` says.
Ask before running its `command`. Run it only after they agree, then go to the next
step. Offer optional cutout helper tools, but do not require them.
A step with a `page` and no `command` is something the creator does; give them
the page, wait until they say it is done, then run setup again.
Tailscale is optional; the Wi-Fi link works without it on the same network.
Offer it once, and if the creator skips it, move on.

After the engine is installed, check `yap-studio doctor --json` for `feedback.mode`.
During first setup, ask once: "May Yap Studio share short notes about problems so it can
improve? Nothing private is shared: no footage, words, names, file names or paths."
Offer ask each time, always share, or keep local. Run `yap-studio feedback setting ask`,
`always` or `never` with their answer. Explain an environment override if doctor still says
always. Keep product notes about the tool; never copy the creator's words or names into them.

Run `yap-studio feedback "..." --step STEP` whenever a step fails, needs a retry or a
workaround, the docs were wrong or missing, or the creator corrects something the checks
should have caught. With mode ask, at the end of a video show
`yap-studio feedback send --dry-run` to the creator. Run `yap-studio feedback send --yes`
only if they agree. With mode never, keep the notes local.

When the steps are done, run `python3 setup.py` again. Setup exits 0 only after every required
step is done and Yap Studio home answers its state check. Do not begin an edit before that.
Use the home link setup prints. Run `open URL` with that link on this Mac. Start
`yap-studio home wait --workspace . --json` as a background task. Do not run it in the foreground
or wait for its result. If background tasks are unavailable, use `--timeout 1` and repeat later.
Reply in the same turn: "You're set up" with the home link and any phone link. Say "I'm listening
for your choices on the page" when the background wait runs, or "I'll check the page for your
choices" when using the short wait. In that reply, say how review links reach their phone:
anywhere through Tailscale when `phone.tailscale` is `connected`, otherwise on the same Wi-Fi.
If `phone.via` is null, their phone needs the same Wi-Fi or Tailscale to open review links.
Offer the optional Tailscale step once when it is there, following the rule above.

## Yap Studio home

Yap Studio home is the page where the creator makes every ordinary choice with buttons: what they
are making, their footage, their look, Make my video, progress, review and the finished file. You
do the editing. Setup starts it. When the creator comes back, reopen it this way:

1. Run `yap-studio home --workspace . --phone --no-open --detach --json`. It prints the link and
   returns while the page keeps serving. Do not run it as a background task. Run `open URL` with
   its `url` so the page opens in the browser on this Mac. Mention `phone.url` once, for watching
   on a phone: say "Same Wi-Fi only" when `phone.via` is `wifi`.
2. Run `yap-studio home wait --workspace . --json` as a background task, so you are told when the
   creator presses a button that needs you. Never wait for its result before replying in chat.
   If background tasks are unavailable, use `--timeout 1` and repeat later.
   Keep exactly one wait running at all times, while you work and while the creator reviews:
   when it returns, start it again first, then handle the event it printed. The page shows
   "Claude is listening" only while a wait runs.
3. Handle the event it printed:
   - `request` with `kind` `edit`: run `yap-studio home ack ID --state claimed --message "Copying
     your footage"`. The payload already has the video type, the B-roll sources and the takes
     (inbox file names), and for a voiceover maybe a Photos album: do not ask again. Run
     `python3 new_project.py TAKE ... --type TYPE --broll SOURCE ...` for vertical.
     For `payload.shape == horizontal`, run
     `python3 new_project.py TAKE ... --type talking_head --shape horizontal`.
     The helper copies and records the takes and sets up a talking head for YouTube with a
     project-local style: hard cuts, no hook, chapters, grade, 3D captions, zooms or transitions.
     When `payload.captions` is not true, after the cut add one `caption_gap` covering the whole
     video on the agent lane, as in [Caption gaps](#caption-gaps). Then run
     `yap-studio home ack ID --state claimed --project DIR`. Follow the recipe for the type. After
     `new_project.py --name DIR TOPIC` renames the folder, run
     `yap-studio home ack ID --state done --project NEWDIR`. A voiceover with no takes starts
     from the album or folder, as the voiceover skill says. If it cannot start, run
     `yap-studio home ack ID --state failed --message "..."` with one plain sentence and the fix.
   - `request` with `kind` `final`: the creator is happy with the story cut of `payload.project`
     and pressed Make the final video. Ack it `claimed`, follow steps 5 and 6 of the edit skill
     for that project, open the review of the final (step 7), then ack it `done`.
   - `request` with `kind` `message`: the creator wrote to you from the page about
     `payload.project` (or the workspace when it is null). Ack it `claimed`, do what it asks as you
     would for a chat message (ask in chat if you must), then reply on the page with
     `yap-studio home ack ID --state done --message "..."` in one or two plain sentences.
   - `request` with `kind` `learn`: ask in chat for the folder of finished videos, run
     `yap-studio taste learn VIDEO... --save --out drafts/taste`, record the saved style id as
     `finished_videos.learned_style` in `style/answers.json`, then ack it `done`.
   - `notes`: the creator sent notes from the review page for `project`. Apply batch `batch` as in
     the edit skill (step 4 for the story cut, step 7 for a final).
   - `approved`: the creator approved `version` of `project`. Run `yap-studio review --stop DIR` and
     say in one line that the finished video is on Yap Studio home, ready to download.
   - `timeout`: start the wait again.
   - A `request` with `redelivered` set came back because nobody claimed it within two minutes.
     If you are already working on it, ack it `claimed` and carry on; otherwise handle it as new.
   - For `resumed` requests, open the previous ack's project, check receipts and `yap-studio status`, then claim it and continue without redoing finished or paid steps.

The page shows each project's progress from its stage receipts, the review link once you open a
review, and the approved file. It never edits, renders or spends money; everything paid still
goes through you, and you ask first. If the creator types in chat instead, answer in chat and keep
the wait running.

## About your videos

Yap Studio home asks these questions on its Look step and saves `style/answers.json`. Ask them in
chat only when the creator is not using the page. On the first run, after setup exits 0, if
`style/answers.json` does not exist and the creator is not using the page, run
`yap-studio style questions --answers style/answers.json --json`. Ask one question at a time
with AskUserQuestion. Put the recommended choice first. Keep the eight answers in
`style/answers.json`; create `style/` first. For finished videos, offer "Yes, pick a folder"
first. If they choose videos, run `yap-studio taste learn VIDEO... --save --out drafts/taste`
on the videos they pick and record the saved style id as `finished_videos.learned_style`.
Record the website or gallery choice, one line about the videos and audience, platforms, cuts,
music, word fixes, and never rules in their answer fields. Then run
`yap-studio style from-answers style/answers.json --workspace .`. It says style choices are kept with
the answers until a look is saved. Show the applied fields and any pending
brand choices. Then open the style page (see Pick my look) for the visual choices.
Pasting a website pulls an accent color from it; choose fonts on the style page or add them with `yap-studio brand fonts add`.
Put preferred caption spellings in `brand/words.txt`, one per line. Each new project adds them to the stored list. Use `yap-studio brand words add WORD` to add one. Use `yap-studio brand words remove WORD` to drop one.

The JSON object needs `"schema": "yap-studio/style-answers@1"`. `finished_videos` has
`answer`, `videos`, and `learned_style`. `brand` has `website`, `accent_color`,
`captions_font`, `titles_font`, `cards_font`, and `pick_in_gallery`; null leaves a choice, ""
clears it. `word_fixes` maps a heard word to the correct spelling. `never` has `cover_face`,
`emoji`, `all_caps`, and `other`. `never.other` is a list of additional rules;
use `[]` when there are none.

## Pick my look

Yap Studio home opens the style page itself from its Look step, and the page links back. When the
creator is not using Yap Studio home, run `yap-studio style pick --workspace . --inbox inbox --phone --no-open --detach --json`.
It prints the link and returns while the page keeps serving. Do not run it as a background task.
Running it again with the same options returns the same link.
Give the creator `phone.url` from its JSON and say it opens on
their phone. The page opens on looks under their video, with saved styles first. Tap a look
and Save style, or choose Customize to go step by step: Font, Captions, Colors, then the
optional More choices list (Grade, Opening, Cards, B-roll, Moves, Moments, Sound). Back to
looks returns to the grid. Fine-tuning stays inside closed More folds. Describe it that way and
do not list other screens. They press Save style when done. Wait until
they say they saved, check that `style/my-style.json` exists, then run
`yap-studio style pick --stop`.

## Update my style

Offer these questions when the creator asks to update their style. They are optional. Run
`yap-studio style questions --answers style/answers.json --json` and start from the current
answers. Ask one question at a time, save changed answers in `style/answers.json`, then run
`yap-studio style from-answers style/answers.json --workspace .`. It updates a saved look in place,
or keeps style choices with the answers until a look is saved. Then open the style page (see Pick my look).

## What are you making?

Each video has a type, and the type picks the recipe. When a video starts, run
`yap-studio video-type show --json`. If `default` is null, ask its `questions` with
AskUserQuestion: say the question's `note`, then use each choice's `label` and `description`.
Ask `video_type` first. For a talking head, ask next whether the creator has anything to show on
top of their video (clips, photos or screenshots), and do not assume they do: if not, record no
B-roll sources. Only when they do, or for a voiceover, ask `broll_sources`, with several answers
allowed, and offer only the choices whose `only_for` (if any) lists the chosen type.
Skip `video_type` when the creator already said what this video is. Save the answer with
`yap-studio video-type set --type TYPE --broll SOURCE ...`. It becomes the default, so do not ask
again on later videos. When a default exists, say in one line which type this video is and that
the creator can change it. If `yap-studio status DIR` says B-roll is not chosen yet, ask only
`broll_sources` for this video and record it with
`yap-studio video-type set DIR --broll SOURCE ...`. For one video that is different, pass
`--type TYPE` and `--broll SOURCE ...` to `python3 new_project.py`, or run
`yap-studio video-type set DIR --type TYPE` once DIR exists. That changes this video only. Add
`--default` only when the creator asks to change the default.

- `talking_head`: follow Edit this video.
- `voiceover`: follow Make a voiceover video.
- `green_screen`: follow Edit this video and put a page, screenshot, photo or clip behind the
  creator with Behind the speaker in step 5 of the edit skill. When the style's placement is
  `green_screen`, the B-roll plan writes the background for you, and the style's
  `green_screen_layout` sets where the creator sits: full, bottom, or a corner.
- `screen_recording`: the screen recording is the picture. If it has the creator's voice, it is
  the take: follow Edit this video. Default a computer recording to horizontal and a phone recording to vertical using its display dimensions; the creator can change it. Horizontal screen effects are coming soon; phase 1 uses a plain talking-head cut of the recording. If the voice is
  a separate audio memo, follow Make a voiceover video and cover every frame with the screen
  recording as inserts, cropped to the part that matters. The creator small in a corner over the
  screen is not in this version yet. Say so plainly and offer voice only.

B-roll comes only from the video's `broll_sources`, which `yap-studio status DIR` shows: `folder`
is the inbox or a folder the creator points to, `album` is a Photos album through
`yap-studio media album`, and `stock` is stock clips the creator approves one by one (Stock in
the edit skill). An empty list means none: add no B-roll unless a review comment asks for it. A
screen recording that is the picture is not B-roll. Screenshots of sites the creator mentions
are not in this version; if asked, say so and offer their own media. Generated pictures are
not a B-roll source. When the style allows them, the agent first asks
"Do you want me to create imagery for this video?", once per video (Generated pictures in the edit
skill).

## Make a voiceover video

A voiceover video is told in the creator's recorded voice over their own photos and clips, with
nobody on camera. Use it when the creator asks for one, names a Photos album or media folder to
tell a story over, or puts an audio memo in `inbox/`. Follow
`.claude/skills/voiceover-video/SKILL.md`. It starts from their media and a script they approve,
then asks them to record.

## Edit this video

Route by the video type (see What are you making?). An audio memo (such as a Voice Memos `.m4a`)
can only be a voiceover or a screen recording told in voice only: follow
`.claude/skills/voiceover-video/SKILL.md`. Otherwise follow `.claude/skills/edit-video/SKILL.md`.
Start with `python3 new_project.py`, naming takes if the creator names them or using `--newest N`
if they ask for several. It copies the newest unused takes directly in `inbox/` into a fresh
`projects/` folder and runs `yap-studio init`. After `yap-studio ingest DIR` and
`yap-studio words DIR`, read the transcript, choose a short topic, and run
`python3 new_project.py --name DIR TOPIC`. Use the DIR it prints from then on. Never edit, move,
rename, or delete an inbox original. Never reuse a prior project's folder.

Read the take and transcript, then choose a clear story. Keep the creator's meaning and
voice. Remove dead air and false starts when the join sounds natural. Keep a pause that
carries intent. Do not invent words, facts, labels, numbers, people, or product screens.
Ask before putting a company or person's name on screen. Graphics support narration; they
do not make narration wait. B-roll, screenshots, pages, and backgrounds come only from media
the creator supplied, stock clips they approve on a pick sheet when the video's B-roll sources
include stock, or generated pictures they approve on a cost sheet after saying yes to imagery
for this video.

Show the story cut to the creator before building the final. Stop for their review of the
cut and apply their notes. The review link opens on their phone. If a note could refer to
two moments, quote nearby transcript words and ask which one they mean. Save the finished
video as a version and show it for another review, so earlier comments carry over. A passing
technical check does not replace the creator's approval.

## Caption gaps

After the cut, save a timeline fragment with one `caption_gap` on the `agent` lane, starting
at `{"time": 0}` and ending at the full cut duration, including the end hold. Add it with the
documented `yap-studio timeline add FRAGMENT DIR --lane agent` command (see
docs/editing.md, Timeline elements). This hides captions for the whole video without changing
the style format.
