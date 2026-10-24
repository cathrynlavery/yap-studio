---
name: verify-video
description: Checks every cut in a finished video with Yap Studio (clicks, sync, frames, loudness, platform rules). Use after rendering or editing a video, before calling it done.
---

# Verify a finished video

## When to run

Run verification after every render and before calling a video done. Use the edit list when
one exists so Yap Studio can compare the finished video with the intended edit.

## How to run

Call the MCP tool `verify` with `media` set to the finished video. Set `edl` to the edit list
when there is one. Leave `asr` at its default unless the user asks for a specific speech
engine or asks to skip speech checks. Set `platform` when the video will be published to a
specific platform.

## Reading the result

Read `status` first:

* `pass` means no checks produced a warning or failure.
* `warn` means at least one finding needs review.
* `fail` means at least one check failed.

Report each actionable item in `findings` with its output time, message, and fix. Read
`checks` for each check's result and for the reason any check was skipped.

The JSON report always lists every finding and every check. Add `--detail full` to include the per-cut and per-segment measurements.

## Fixes by check id

| Check id | What it means | Usual fix |
| --- | --- | --- |
| `join.click` | A sharp sound change is present at a cut. | Add a short audio fade (5 to 30 ms) and render again. |
| `join.gap` | Audio was inserted or lost at a cut. | Inserted: encode the audio once for the whole video. Lost: re-render so each clip keeps its full audio. |
| `join.words` | At a cut, the render's audio differs from the edit list, so speech may be clipped, repeated or missing. | Render the cut again so its audio matches the edit list. |
| `join.frames` | The frames around a cut do not match the edit. Without an edit list, a frame at a found cut counts as repeated only when it changes almost nothing from its neighbour, or less than a quarter of the usual frame-to-frame change on that side of the cut. | Check the source range and render the cut again. |
| `join.listen` | A join needs a listening review, for example where the edit itself cuts through a word. | Listen around the named cut and repair any audible change. |
| `join.relisten` | The render's words at a cut differ from the edit's own audio, or a word is doubled across the cut that the source does not repeat there. | Listen to the join, move the cut if needed, and render again. |
| `join.second_model` | A second speech engine from another model family, for example faster-whisper next to Parakeet, hears different words at a cut. It runs in its own process. Unmeasured when no engine from another family is installed or the second engine stops before writing its result. | Listen to the join and decide which reading is right. |
| `sync.av_offset` | Picture and sound do not land together. | Correct the sound offset and render again. |
| `sync.drift` | The sound offset grows across the edit. | Encode the audio once for the whole video instead of per clip. |
| `sync.unmeasured` | Sync could not be measured for a part. | Supply the right source or use clearer source sound and picture. |
| `plan.picture` | Rendered picture does not match the source range. | Check the edit list and source mapping. |
| `plan.frames` | The render has the wrong frame count. | Check range ends, frame rate, and export bounds. |
| `plan.samples` | The render has the wrong audio sample count. | Check audio range ends and sample rate. |
| `plan.holds` | A hold declared in the edit list is not a still picture, or has sound where the edit list declares silence. | Render the hold as declared, or correct the edit list. |
| `edit_list.evidence` | An evidence file listed in the edit list is missing, or no longer matches the sha256 recorded for it, so the cuts it attests are unconfirmed. A matching file is reported as checked outside Yap Studio, never as a pass. | Point the edit list at the file that was checked, or review it again and record its current sha256. |
| `structure.av_length` | Picture and sound end at different times. | Render both streams to the same endpoint. |
| `structure.audio_timestamps` | The audio timestamps jump ahead or step back somewhere in the file. A player that follows them inserts silence or overlaps sound there. The finding lists each break with its time and size. | Encode the audio once for the whole video and mux it again. |
| `loudness.integrated` | Overall loudness misses the target. | Adjust the final mix toward the stated target. |
| `loudness.true_peak` | The true peak is above -1 dBTP. | Lower the limiter ceiling. |
| `loudness.parts` | Two neighbouring parts from different sources differ by more than 3 LU. Without an edit list, only parts at least 3 s long are compared. | Normalize the loudness of that part to match the rest. |
| `picture.black` | A black interval was found. | Replace unintended black frames. |
| `picture.freeze` | A long frozen interval was found. | Replace the held frames if they are not intentional. |
| `picture.flash` | One or two stray frames flash at a cut: the picture changes into them and out again within two frames of the join. With an edit list, frames that are in the source at that point are not counted. | Render the cut again so the outgoing segment's last frame meets the incoming segment's first frame. |
| `detect.cuts` | Lists the clear cuts found without an edit list. Jump cuts inside one take may be missed. | Nothing to fix. Add `--edl` for a full check, including sync and words. |
| `platform.shape` | The video shape does not fit the platform. | Render the required horizontal or vertical version. |
| `platform.duration` | The video is over the platform's length limit, or over a default limit an account can lift (YouTube past 15 minutes needs a verified account). | Follow the report's suggested action: verify the account when it says so, otherwise shorten it or choose another format. |
| `platform.resolution` | The frame size misses a platform rule. | Render at a supported size. |
| `platform.fps` | The frame rate misses a platform rule. | Render at a supported frame rate. |
| `platform.chapters` | The edit has chapters, but the YouTube chapter list breaks a YouTube rule: the first chapter must start at 0:00, there must be at least three, and each must last at least 10 seconds. No list is offered. | Move, merge or add chapters until the list meets those rules, then render again. |
| `taste.rules` | The finished picture breaks an approved taste rule applied to this render, or its receipt names a different style. Rules that could not be measured have a reason. | Render again with the approved style, or revoke the rule if it no longer holds. |

## Skipped checks

Never report a skipped check as passed. List every skipped check and its reason. If an
important check was skipped, explain what input or dependency is needed to run it.

## If the tool is missing

From a checkout, install the CLI with `uv tool install '.[asr,mcp]'`. Configure the MCP
server to run `yap-studio mcp`, then call `verify` again.
