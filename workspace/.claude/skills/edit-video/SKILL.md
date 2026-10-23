---
name: edit-video
description: Edit inbox takes into a reviewed video with Yap Studio.
---

# Edit a video

DIR is the project folder that `new_project.py` prints. Run commands from the workspace root.
Put the files you write in `DIR/plans/`. When unsure what comes next, run
`yap-studio status DIR`; it names the next command and what waits on the creator.

## Rules

Still frames from pinned notes and contact sheets of the creator's clips go to Claude
(Anthropic), along with the transcript, notes and project files you read in this session.

- Never write to, move, or rename the original in `inbox/`. Work only in DIR.
- On-screen words come from the creator's words or facts they supplied. Ask before naming a
  person or company on screen.
- B-roll, screenshots, pages, and backgrounds come only from media the creator supplied, such
  as an `assets/` or media folder they point to, stock clips they approve (see Stock below),
  or generated pictures they approve (see Generated pictures below). Their own media and
  stock come only from the sources in this video's `broll_sources`; `yap-studio status DIR`
  shows them (What are you making? in `CLAUDE.md`). Never invent screens or images, and
  never download or generate anything they have not approved.
- Use a feature only where the story calls for it. Most videos need a few, not all.
- Run `yap-studio feedback "..." --step STEP` whenever a step fails, needs a retry or a
  workaround, the docs were wrong or missing, or the creator corrects something the checks
  should have caught. Describe the tool problem only, with no creator words or names.
  Check `feedback.mode` in `yap-studio doctor --json`. With mode ask, at the end of a video
  show `yap-studio feedback send --dry-run` and run `yap-studio feedback send --yes` only
  if the creator agrees. With mode never, keep the notes local.
- Stop for the creator's review of the cut and of the final. A passing check is not approval.
- Never run `assemble --allow-violations` or pass `--allow-violations` to `run`. Move or drop
  what covers the face instead.

## Text at the top

Before making the cut, follow Text at the top in `CLAUDE.md`. If `hook.answered`
is missing or false, ask once in plain words and wait. There is no default. A home's
`top_text` answer overrides the saved choice for this video only. None needs no text.

Both Just the opening and The whole video use `DIR/plans/text-hook.json`.
For an opening title add `"kind":"opening"` to the plan below; for a whole-video
label omit it. Use the creator's `text_hook` from the home request or their own text
from a note as `chosen`, with `source: typed` or `source: note`. Ideas may be empty
when they supplied the text. Otherwise write three ideas, show all three in the
Story cut note and on the review page, and wait for a pick or their own text.
Never choose for them. If final approval arrives without a text choice, ask once and wait.

For opening titles, use short runs of their spoken words in their title look.
After the pick, put exactly that chosen text in the title's `lines` in the agent
lane, keeping the existing timing and line layout rules below. Never render an
opening title without chosen text. The engine refuses a missing choice or different
title text. The review page saves either kind of choice with a note.

### Whole-video label

When the saved style uses `hook.style: text_hook`, its plain label stays on screen
for the whole video. Text is per video. Use the creator's `text_hook` from the home
request or text they typed in a note. Save it as `chosen` with `source: typed` or
`source: note` in `DIR/plans/text-hook.json`. Ideas may be empty for typed text.

Otherwise, after reading the transcript, write three ideas to that file:
`{"schema":"yap-studio/text-hook@1","ideas":["...","...","..."],"chosen":null,"source":"idea"}`.
Each is one line of their point in their plain voice, about 45 characters. Give each a
clear tension, claim or question that makes someone keep watching. Use no facts or
numbers they did not say, hashtags, emoji or all caps. Keep each to at most two
balanced lines when set, with at least two words on each line when it wraps.
Vary the three ideas: a short video label like "Security rant", "Is it just me, or ...?",
a claim that pushes back on an accepted belief, or a question that challenges an assumption.
Keep each short; a label of two or three words is fine.

Show all three ideas as plain text in the Story cut note. Wait for their choice or
their own text before rendering any Text hook. The review page offers those choices
and saves their choice with a note. If they choose in chat, save the chosen text and
use `source: idea` for an offered idea or `source: note` for their own words.
If they ask for or approve the final without choosing, ask once and wait. Never pick for them.

The engine fits the text from 72 down to 60 px in bundled Inter. It refuses text
that needs more than two balanced lines or covers the face at the 120 px top floor.
Ask for shorter text after a refusal. Move nothing else to make it fit.
The engine plans this title from the choice file. Do not add a title to the lane.
Opening title rules below apply after the creator has chosen their text.

## Procedure

Use a moment only when the style lists it or a review comment asks for one; then turn
it on in the project's style. Put it on a punchline, key claim or reveal. Keep starts
at least 10 seconds apart and use at most two moments in a video.
`clone_flash` needs a still camera and room beside the creator.
Keep titles, cards and overlays out of a clone flash window.
It uses moving copies from later in the same take, sized to the room on each side.
Moments follow the active colour grade.
If assembly refuses it, follow the fix or drop it.

Every version note is shown to the creator as Claude's notes on that version. Where a step below
gives a note such as "Story cut", "Notes from bN" or "Final", start with those words and add one to
three plain sentences on what this version changes and why, for example `--note "Story cut. Took
out the um after your opening line. Kept the doubled you at 0:07, because cutting it clipped the
sound."`. For the final, say what it adds and anything you left out and why.

Each review below uses `--detach`, so the command prints its JSON and returns while the page
keeps serving. Do not run it as a background task. An unchanged version review with the same
options reuses the live page. A changed review of DIR replaces the earlier page.
Read `phone.url` and `phone.via` from its JSON output. Say "Same Wi-Fi only" for `wifi` or
"Anywhere on your tailnet" for `tailscale`. End your reply with the `phone.url` link. When Yap
Studio home is open, it shows a Watch button for the review on its own, so say the review is ready
there. Keep `yap-studio home wait --workspace . --json` running as a background task while you
stop for the creator: their notes and their approval arrive through it (see Yap Studio home in
`CLAUDE.md`).

1. If setup is incomplete, follow Set me up in `CLAUDE.md`: one step at a time, asking first.
   When setup succeeds, reply "You're set up" with the home link in the same turn. Start the
   home wait as a background task and say you are listening. Do not wait for a page request
   before replying.
2. Run `python3 new_project.py` for the newest unused take, with named takes if the creator
   names them, or `--newest N` if they ask for several. Run `yap-studio ingest DIR`, then
   `yap-studio words DIR`.
   Read the transcript once as a proofreader. A word that makes no sense in its sentence is
   probably misheard. Look for near sounds, like "blogging" when they are logging into an
   account, cut-off words like "CAPTS", or a common word in place of a name or term they use
   elsewhere, like "capture" next to "riddle" in a video about captchas.
   Fix each one with `yap-studio words fix DIR --word ID --to TEXT`. Use `--heard` only when
   the same mishearing repeats. Never change what they actually said. Keep their slang, swears,
   grammar, fillers they kept and the way they phrase things. If unsure, listen to that word
   with `yap-studio media sound ID` or the take, or leave it and list it as a question.
   Name the fixes in the Story cut note so they can undo any, for example "Fixed 1 misheard
   word: capture -> captcha at 0:39." Fix review notes about a wrong caption word the same way.
   If ingest or status warns that speech has no words, listen to
   that part before using it in the story. Read the transcript and source. Choose a short
   topic of one to three words and run `python3 new_project.py --name DIR TOPIC`. Use the
   DIR it prints from then on.
   Run `yap-studio status DIR`; its first line names the video type and B-roll sources. If the
   type or B-roll is not chosen yet, ask as in What are you making? in `CLAUDE.md` and record it
   with `yap-studio video-type set DIR --type TYPE --broll SOURCE ...`. If only B-roll is not
   chosen yet, ask only `broll_sources` and record it with
   `yap-studio video-type set DIR --broll SOURCE ...`. For `green_screen` and
   `screen_recording`, follow that type's note there.
3. Story. Run `yap-studio taste show` for the creator's approved rules. Style rules apply at
   render on their own; follow pacing, filler, and placement rules yourself. Choose the story
   order; if `style/answers.json` has `about`, use it for titles, card moments and what to keep.
   Run `yap-studio edit schema` for the `edit.json` format and an example.
   Before writing `edit.json`, read the whole transcript as a script. List their points in order,
   one line each. If they make the same point more than once, keep the clearest and most complete
   telling. Leave the others out, even when each is a clean sentence. Their words stay theirs:
   choose among their sentences, never rewrite them. Leave out a line that points at something to
   show, such as "this is one I had the other day", unless the video will show it. If the video
   has no B-roll, ask whether they have it.
   Write `DIR/edit.json` with the engine's project schema. Keep
   the creator's words and meaning. Run `yap-studio fillers DIR`. Only removals applied before
   the cut review reach the final. `run` never applies new ones. To use a proposal, run
   `yap-studio fillers set DIR ID applied` before `yap-studio cut DIR` and the review, or when
   the creator asks in a note. Apply only joins that keep the words and sound natural.
   Listen first if a proposal is marked `listen`.
   On a long take fillers can take several minutes. Start
   `yap-studio fillers DIR` as a background task, then wait with
   `yap-studio status DIR --json --wait 540` until no stage is `running`. Never wait with
   `pgrep`. Run `yap-studio fillers DIR` again to print its saved result. It returns at once.
   Run `yap-studio edit check DIR`. Settle every `Repeated` and `Same point twice` warning:
   leave one telling out, or keep both on purpose and say why in the version note. Check again
   after any change, then run `yap-studio cut DIR`.
4. Review the cut. Run `yap-studio versions save DIR --note "Story cut. ..."`; when you left
   repeats out, name those lines in the note so they can ask for any back. Then run
   `yap-studio review --versions DIR --phone --json --detach`. Stop for the creator's review.
   Their notes arrive as `DIR/feedback/bN.json`. Write the edits as
   `yap-studio/feedback-operations@1` and run
   `yap-studio feedback apply DIR DIR/plans/ops-bN.json`.
   For example, this is the complete minimal `DIR/plans/ops-b1.json` for a Cut this note:

   ```json
   {
     "schema": "yap-studio/feedback-operations@1",
     "batch": "b1",
     "items": [
       {"comment": "v1:c1", "operations": [{"id": "cut-1", "kind": "cut"}]}
     ]
   }
   ```

   Replace the batch and comment with those in the notes. A Cut this note carries a cut
   request with the exact words; `feedback apply` removes them and join-tests each new join.
   If it declines a cut, reply `declined` and name its clean options for the creator to pick.
   A note with `pin` points at a spot on the picture. Open `pin.still` to see the frame with
   the pin drawn, and change the first element in `pin.under` (topmost first); `pin.canvas` is
   the spot in canvas pixels. When `pin.under` is empty and `pin.known` is all false, the spot
   is not mapped: look at the still and ask if the note could mean two things.
   If a note could mean two moments, ask with `yap-studio feedback ask` and quote nearby
   words. After applying notes to the story, reread the whole kept script, run
   `yap-studio edit check DIR` again, settle `Repeated` and `Same point twice` as in step 3,
   and name any repeat kept on purpose in the version note. A broad note such as
   "remove the repeats" means the whole video, not one spot.
   Then run `yap-studio cut DIR` and `yap-studio feedback check DIR`; exit 3 means a
   keep broke, so restore it. Run `yap-studio versions save DIR --note "Notes from bN"`; it prints
   the new version vM. Answer every note with `yap-studio feedback reply DIR --batch bN
   --comment vK:cN --version vM --status changed --text "..."` (or `kept`, `declined`,
   `needs_clarification`).
   If a note is completed in a later saved version, pass that version with `--version`;
   the reply measures that version without changing the batch's first resulting version.
   Then run `yap-studio review --versions DIR --phone --json --detach` again. Repeat until
   the creator is happy with the story. They say so with Make the final
   video on the review page or Yap Studio home, which arrives as a `final` request (see Yap
   Studio home in `CLAUDE.md`), or in chat.
5. Build the picture. Plans read `DIR/cut/cut-words.json` and take `--frames N`, the cut's
   frame count (`plan.total_frames` in `DIR/cut/receipt.json`). Pass `--style` the `style`
   value from `DIR/project.json`; a `styles/...` value is a path inside DIR.
   - The creator's style sets the grade. Assembly applies it. Change it for one video
     only when the creator asks in a note. Use `yap-studio grade set DIR --look film`
     or `yap-studio grade set DIR --look none`. When a note asks to pull the colours
     toward something in the video, read it with `yap-studio grade sample DIR --time T
     --x X --y Y`, then use `yap-studio grade set DIR --look eyedropper --time T --x X --y Y`.
     When `grade.slider` is on in the style, or a note asks, add one `grade_compare`
     with `yap-studio timeline add`. Use the agent lane and an `agent-` id. Pick
     2.5 to 4 seconds of talking head soon after the opening, away from overlays
     and backgrounds.
   - Media. If the creator points to a folder, run `yap-studio media add FOLDER`. For each new
     item run `yap-studio media sheet ID`, look at the sheet, write
     `{"schema": "yap-studio/media-description@1", "description": "...", "tags": [...]}`,
     and run `yap-studio media describe --file FILE ID`.
   - B-roll. `yap-studio broll suggest --cut-words DIR/cut/cut-words.json --style STYLE`
     ranks catalog items against the words. Choose, then write `DIR/plans/broll.json`
     (`yap-studio/broll-requests@1`: `id` starting `broll-`, `item`, a word `start`, an
     `end` or `hold_s`, and `kind` `insert` or `overlay`). Run `yap-studio broll plan
     DIR/plans/broll.json --cut-words DIR/cut/cut-words.json --style STYLE --frames N --out
     DIR/plans/broll`, then `yap-studio timeline add DIR/plans/broll/broll-lane.json DIR
     --lane broll`. Leave the creator's files where they are; the warning that media is
     outside the project is expected. Assembly matches inserts and overlays to the
     take unless the style sets `broll.color_match` off. Set `color_match` false on a
     request to keep one shot's own colour (screens are planned that way), or true
     to match one shot when the style is off. The speaker's picture never changes.
     Use placement `split` only when the style has it or a review comment asks for it.
     Start a shot showing a named thing on that word with `offset_s: -0.1`.
   - Tall box. Use `tall_behind` only for the creator's own long screenshots or pages exported
     as one tall image; ask before using anything else. Start it on the first word after a
     cut and end it on the last word before a cut. Use `scroll` for eased steps, with 2.5 to
     8 s on screen. The person stays in front and normal framing returns when the box leaves.
   - Stock. Only when `stock` is in this video's B-roll sources. A project with no B-roll sources
     recorded follows the style's `broll.stock_clips` (`ask`). After `broll suggest`, pick the
     phrases that need a picture their own media does not cover. Write `DIR/plans/stock.json`
     (`yap-studio/stock-requests@1`: `id` starting `stock-`, a word `start` and `end`, and `query`, a
     few plain words for the picture, never a name or a private detail; add `not_fitting` with the
     catalog ids you looked at and rejected). Run `yap-studio stock search DIR/plans/stock.json
     --cut-words DIR/cut/cut-words.json --style STYLE --json`. If `status` is `unavailable`, tell the
     creator its `message` in one line and carry on with their own footage. Otherwise run `open` on
     its `html` path and ask with AskUserQuestion, one question per phrase from `ask`
     (four at a time at most). Run `yap-studio stock get DIR --pick CODE ... --yes --json` only for the
     codes they chose, and only after they answer. Add each returned `request` to
     `DIR/plans/broll.json`. `yap-studio stock credits DIR` prints the credits for the video's
     description.
   - Generated pictures. Only when the style's `broll.generated_pictures` is `ask`, this video
     has B-roll (its sources are not none), and a line still needs a picture neither their media
     nor an approved stock clip covers. Ask first, once per video: run `yap-studio pictures
     question DIR --json`. If `answer` is null, ask its `question` with AskUserQuestion in its
     exact words, "Do you want me to create imagery for this video?", with Yes and No and
     neither marked recommended. Record the answer with `yap-studio pictures answer DIR yes` or
     `yap-studio pictures answer DIR no`. On no, write no picture sheet, run no price check, and
     do not ask again for this video unless the creator brings it up. On yes, write
     `DIR/plans/pictures.json` (`yap-studio/picture-requests@1`: `lines` with `pic-` ids, word
     `start` and `end`, a plain `prompt` that never names or depicts a real person, brand, logo
     or private detail, and `not_fitting` like stock). Run `yap-studio pictures ask
     DIR/plans/pictures.json --cut-words DIR/cut/cut-words.json --style STYLE --json`. If
     `unavailable`, tell the creator its `message` in one line and carry on. When WaveSpeed is
     not set up, the message ends with the WaveSpeed sign-up link
     https://littlemight.com/recommends/wavespeed; say it is an affiliate link. Otherwise `open`
     its `html` and ask with AskUserQuestion from `ask`, four at a time at most. Say each cost
     and the total. Only after they answer, run `yap-studio pictures make DIR --pick N ...
     --max-cost TOTAL --yes --json` for approved lines, at most three. TOTAL is the sum of those
     lines on the sheet. Show the pictures and what they cost before asking about more. Add each
     returned `request` to `DIR/plans/broll.json`. Use `--new` only after they approve paying
     again for an uncertain attempt.
   - Cards. `yap-studio templates list --project DIR` and `yap-studio templates show ID`
     give each card's slots. The list shows only the card kinds this project's style and
     approved taste have on; use no other kind (newer kinds stay off until the creator turns
     them on in the style page).
     Video clip objects use a path inside the project and sha256 from `shasum -a 256`.
     Use `big_number` only when the style has it on and only for a number they say.
     Never place two big numbers close together.
     For screen recordings, use `crop: [x, y, width, height]` in displayed pixels after rotation
     to show the part the speaker talks about, so its text reads on a phone. Aim for the smallest
     text to be about 24 px tall in the final video.
     Use `choices` only when they list 2 to 4 options, with their words shortened at word boundaries
     and numbers as they say them, and cues as word anchors on each option.
     With `graphics.flow_look` set to `tags`, use 2 or 3 labels from their words,
     shortened at word boundaries, one cue per tag and at least 1.5 s, ending
     the card on the first word after a cut for its quick clear.
     Use Specs list row cues or Stat breakdown rows only when the style has rows on cue,
     taking rows from their words and numbers, keeping their exact figure, with cues as word anchors.
     Motion streak follows `graphics.card_motion`; do not add a second card entrance.
     Keep the planner's rest box and motion evidence. Unsupported cards or unsafe paths enter as usual.
     Run `yap-studio faces detect DIR/cut/cut.mp4 --out DIR/plans/faces.json`. Write
     `DIR/plans/cards.json` (`yap-studio/card-requests@1`: `id` starting `agent-`,
     `template`, `slots`, a word `start`, an `end` or `hold_s`, `placement` `top` or
     `behind_head`). Word ids come from `DIR/cut/cut-words.json`. For example:
     ```json
     {"schema":"yap-studio/card-requests@1","cards":[{"id":"agent-terms","template":"text_tag","slots":{"text":"Technical terms","variant":"sign"},"start":{"word":"t1:12"},"hold_s":2.5,"placement":"top"}]}
     ```
     Run `yap-studio cards plan DIR/plans/cards.json --cut-words
     DIR/cut/cut-words.json --style STYLE --frames N --faces DIR/plans/faces.json --out
     DIR/plans/cards`. Rework any card it reports blocked (shorter text, or another
     moment with more room), then run `yap-studio timeline add
     DIR/plans/cards/agent-lane.json DIR --lane agent`.
     With `captions.camera_3d` on, add no zooms, jump cut punches, moments,
     backgrounds, `behind_head` cards, Big number, tall boxes or split B-roll. Full screen
     B-roll, B-roll in a box, top cards, titles and transitions are fine.
     `timeline check` names anything the look cannot share a video with.
   - Zooms. Read `moves.zoom_mode` in the style. With `none` (the default), add no zoom
     unless a review comment asks for one. With `punches`, punch in on a word that carries
     the point: a key number, a joke landing, or a strong claim. Use a few per minute at
     most, never every sentence. With `punches_and_push`, one slow push can also build a
     section; it holds or ends on a cut, best where the picture cuts. Write a
     `yap-studio/timeline@1` fragment with `zoom` elements: `"lane":"agent"`, `id` starting
     `agent-`, `start` on the punch word, `end` on the last word of the phrase, and `move`
     `punch` with no `scale` (it uses the style's `moves.punch_scale`, 1.5 or 2), or `move` `push` with
     `scale` 1.2 to 1.3. Leave `center` out to aim at the face. Add it with `yap-studio
     timeline add FRAGMENT DIR --lane agent`. Other zooms cannot overlap a card, title or insert.
     If `timeline check` says the chin reaches the captions, set `scale` 1.5 on that punch.
     It warns about a move under 15 percent or slower than 0.03 per second; replace it with
     a punch or a push, or remove it.
     With `moves.opening_push` on, open the video with one `zoom` with `move`
     `opening`, `start` on the first word of the cut and no `end`. The engine ends
     it at the first sentence end or real cut between 0.8 and 4 seconds after the
     first word. If the line runs longer, it uses the first clause end in that
     window. Longer lines push more slowly and reach 1.3x at their end. If no
     boundary fits, the push is skipped with a warning. Never cut wide in the
     middle of a clause. It may sit under the opening title.
   - Jump cuts. Read `moves.jump_cuts`. With `hard` (the default), do nothing new. With
     `smooth`, the engine smooths small jumps at assemble; run `yap-studio joins DIR` after
     the cut and add a punch on each join it lists as `punch`, even with zooms off.
     With `punch`, do the same.
     A punch here is a zoom with `move` `punch` and `ease` `cut`, `start` on the join's
     `after_word` and `end` on the last word of that phrase. Leave joins listed as
     `covered` alone.
     With `step_in`, the engine changes the framing on the cuts itself; add no punch
     on those joins.
   - Transitions. Read `moves.transitions` in the style and use only the looks it lists.
     With an empty list, add none unless a review comment asks for one. Use a transition
     only at a story join where the last word before the join ends a sentence and the
     story turns, never at every join. Write a `yap-studio/timeline@1` fragment with
     `"lane":"agent"`, `kind` `transition`, `id` starting `agent-`, `start` on the first word
     after the join (its start, no offset; add `segment` when that word occurs more than
     once in the cut),
     `look`, and optional `frames` (even, 6 to 24, default 12). Add it with
     `yap-studio timeline add FRAGMENT DIR --lane agent`. A transition cannot overlap a
     title, insert, overlay, background, zoom or another transition. Keep grade compares
     clear too. One join takes one transition. `timeline check` warns when the style does
     not allow the look; replace it with an allowed look or remove it.
   - When the creator or a review note asks for applause, a laugh, a cheer or a record scratch at a moment, add a `sound` element at that word; never add one unasked.
   - Behind the speaker. To show the creator's page, screenshot, or silent clip behind them,
     copy it into `DIR/broll/`, then add a `yap-studio/timeline@1` fragment with one
     `background` element (`id` starting `broll-`, word `start` and `end`, `media` with
     `path` `broll/NAME` and its `sha256` from `shasum -a 256`, `fit` `cover`)
     through `yap-studio timeline add FRAGMENT DIR --lane broll`. Assembly builds the person
     matte from the cut itself; it needs macOS Vision, and without it assembly refuses.
     A background cannot overlap an insert or zoom.
     When the style's placement is `green_screen`, the B-roll plan writes the background
     for you. Use `layout` and the look fields only when the style has green screen or a
     review comment asks for it.
   - Titles go in the agent lane as `title` elements with `lines`.
     Each element in any fragment needs `"lane"`. Use an `agent-` id, a word `start` from
     `DIR/cut/cut-words.json`, and one to three strings in `lines`. Optional `hold_s` or `end`
     sets the title's length; use one at most. Add it with
     `yap-studio timeline add FRAGMENT DIR --lane agent`.

     ```json
     {
       "schema": "yap-studio/timeline@1",
       "elements": [
         {
           "lane": "agent",
           "id": "agent-title-1",
           "kind": "title",
           "start": {"word": "t1:12"},
           "lines": ["Technical terms"]
         }
       ]
     }
     ```

     Use the speaker's own words from the cut, in order. Shorten only by dropping words at
     the start or end of their line. No rewording, new words, or turning a statement into a
     question. Do not add `?` or `!` they did not say. The engine applies brand spellings
     and numbers.
   - Use Video in letters only when the style allows it. Opening inserts need
     `hook.style: video_letters`. Key lines later need `video_letters` in
     `broll.emphasis_on`. Use at most two letters inserts mid-video. Use one or two words
     from the spoken line. Add `"letters":{"text":"DOG PARK","background":"color"}` to the
     insert request, or use `blur`. A clip carries on full screen. A photo cuts back to the
     speaker. `timeline check` names a shorter word when one does not fit.
   - Add `chapter` elements only when the style's chapter title card is on or the creator asked for chapters in a note.
     Anchor each at the first word of the new section and keep titles short.
   - Music: a track the creator supplied, with `yap-studio music DIR --file TRACK`. Or
     ElevenLabs composes one: `yap-studio music DIR --yes --dry-run` shows the request without
     spending, and `yap-studio music DIR --yes` spends the creator's credits, so run it only
     after they approve. If they have no ElevenLabs account, give them
     https://littlemight.com/recommend/elevenlabs and say it is an affiliate link. Otherwise
     `yap-studio music DIR` records none. Prepare it before the check below, because cards can
     snap to its beats.
   - Run `yap-studio timeline check DIR` and fix every violation it names.
6. Render. Run `yap-studio run DIR --note "Final" --json --detach`.
   It prints its JSON and returns while the render keeps going. Do not run it as a background task.
   A long final can take more than 30 minutes. Run `yap-studio status DIR --json --wait 540`
   until `run.state` is no longer `running`. Each call returns within 9 minutes.
   Then read `run.result`: `status`, `stopped` (its
   `step`, `message`, and `suggested_action`), `final`, `version`, and `warnings`.
   A finished detached run opens a review page and returns its links or a recovery command
   in `run.result.review`.
   Read `run.result.words`. Tell the creator every word in `words.dropped` with its time and why.
   Never say all their words are still there unless `words.status` is `pass` and
   `words.dropped` is empty. If the run stopped with `words_dropped`, follow its fix. It runs every
   remaining stage, verifies the final, and saves it with layer windows. If it stops, follow
   `suggested_action` and run it again; finished stages are skipped. Face clearance failures
   mean a caption or card covers the face; for detail run
   `yap-studio faces check DIR/final/NAME.mp4 --layout DIR/final/NAME.layout.json --faces
   DIR/final/NAME.faces.json`. Run `yap-studio taste check DIR/final/NAME.mp4`. Repair
   failures and run it again. `yap-studio run DIR --stop` stops the background render.
   Captions that are hard to read get a dark halo automatically; if `warnings` still name a
   caption, tell the creator the time and offer `yap-studio captions DIR --back box`.
   If `warnings` name a word heard with no caption, or a caption that is not heard, tell the
   creator the word and the time, and do not say every word is captioned.
7. Review the final on Yap Studio home, or get a phone link with
   `yap-studio review --versions DIR --phone --json --detach`. The final joins the cut's versions,
   so earlier comments carry over. Stop for the creator's review. Apply notes as in step 4,
   then run `yap-studio cut DIR`. Repeat step 5 and run
   `yap-studio run DIR --note "Notes from bN" --json --detach`.
   Wait with `yap-studio status DIR --json --wait 540` as in step 6.
   Read its new version vM from `run.result`. Run
   `yap-studio feedback check DIR --version vM` so card and caption keeps are measured on
   that final, and reply with
   `--version vM` before the next `yap-studio review --versions DIR --phone --json --detach`.
   After notes on the final, the next review is the new final (the whole video), with no
   separate cut review unless the creator asks to see the story again.
   `feedback apply` takes story and B-roll edits only; change cards, titles, and captions
   by planning them again and running `timeline add` after the apply.
8. Approval. The creator approves a version on the review page, which unlocks its download.
   Once the creator approves, run `yap-studio review --stop DIR`.
   If they approve in chat instead, run `yap-studio versions approve DIR vK`. Never approve
   for them. When a note states a lasting preference, write a proposal file (`scope`,
   `sentence`, `machine` or null, and `comments` as `[{"version": "vK", "id": "cN"}]`) and
   run `yap-studio taste propose DIR FILE`; the creator decides it under Rules for future
   videos on the review page.

Do not call an edit finished because the render passed checks. The creator decides when it
is ready.
