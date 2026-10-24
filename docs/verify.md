# Verify a finished video

`yap-studio verify` checks a finished video on its own. It works on any render, with or
without the editor, and with or without an edit list. Verification does not need Node or the
renderer; install notes are in [install.md](install.md).

## What verify checks

Ranges JSON can declare `grade` with `look`, `strength`, `version`, `reference` and
`reference_sha256`. The reference is the ungraded picture on the output frame grid.
An Eyedropper grade also carries its resolved `color` as `#RRGGBB`. Assembly samples
the retained cut before rendering; verification uses that colour without sampling
the point again. A different target colour or an ungraded final fails `picture.grade`
when its colour error exceeds its limit.
Assembly publishes `../cut/cut.mp4` relative to `final/verify-edl.json` and records its
SHA-256. Grade math version 2 supports `none`, `film` and `eyedropper` at strengths
from 0 to 1.
An optional `skip_frames` list contains `[start, end]` pairs of non-negative integer
output frames. Each end is exclusive. Assembly puts transition windows here because
their final picture differs from grading the plain cut. These windows affect only
`picture.grade`; they do not mask sync, joins or other picture checks. Ungraded EDLs
do not add a grade record or transition overlays.
Assembly declares each `grade_compare` window as a full-frame picture region,
like a zoom. Source picture comparisons and `picture.grade` skip that window
because it deliberately mixes the raw picture and the grade.

For an active grade, sync and join checks convert HDR sources to BT.709 and apply the
same grade before comparing picture frames. `picture.grade` samples up to 24 visible
frames at 90 by 160 RGB pixels. It skips picture masks and holds and excludes covered
rectangles. For each frame it measures the mean absolute RGB difference between the
graded reference and the original reference over those same valid pixels. The median
of these frame measurements is the grade magnitude `M`, in RGB levels out of 255.
Both Film and Eyedropper fail when the median final colour error exceeds
`min(10, max(2.5, 0.5 * M))`. This ties the limit to the grade's effect on the picture,
with a 2.5 level floor for encode noise and a 10 level ceiling. A tiny grade whose
magnitude is below the noise floor cannot reliably be distinguished from encode noise.
Evidence records `grade_magnitude` and the effective `limit`; `threshold` keeps the
same value for existing report readers. The report says whether the final looks ungraded.
The fix is `yap-studio assemble`. Samples are spread over frames left visible after
picture masks, holds, transition skips, compare windows and fully covered regions.
With fewer than 6 usable frames or a missing or changed reference, `picture.grade`
is not measured: coverage records the reason and usable count, and a warning asks
you to check the final by eye or re-render with `yap-studio assemble`.
Timing and join checks still run. Without a declared grade, source decoding keeps
its existing behaviour.

- joins for clicks, gaps, broken words, and the wrong frames
- picture and sound sync, including drift across the edit
- expected frame and sample counts
- loudness across the whole video and within each part
- black or frozen picture
- the shape, length, resolution and frame rate rules published by YouTube, Shorts, TikTok
  and Reels

## Check a video

Check a finished video and let Yap Studio find its cuts:

```console
yap-studio verify final.mp4
```

An edit list lets Yap Studio compare the finished render with the intended source ranges:

```console
yap-studio verify final.mp4 --edl edits/cut.json
```

Check delivery rules for a platform:

```console
yap-studio verify final.mp4 --edl edits/cut.json --platform youtube
```

A summary looks like this:

```text
Yap Studio verify: final.mp4 (1920x1080 horizontal, 0:19, 7 cuts found in the picture)

WARN: 2 things to check

  Whole video       Overall loudness is -19.4 LUFS. Normalize the whole video to -14 LUFS.
  Whole video       Sound peaks at -0.8 dBTP. Limit the true peak to -1 dBTP.

  7 cuts OK: no click, right frames, words not checked
  Loudness -19.4 LUFS, target -14 (warn)
  No black or frozen picture

Not checked:
  Sync and words against the source: need the edit list
  Platform fit: no platform chosen
  Listening at the cuts: no speech engine installed; pip install 'yap-studio[asr]'
```

Warnings do not fail the command unless you add `--strict`.

The edit-list formats verify reads, and the optional fields for overlays, holds, declared
joins and evidence, are in [edit-lists.md](edit-lists.md).

## Reading the results

Each finding gives the time on the output, the problem and a fix.

| Check id | What it means | Usual fix |
| --- | --- | --- |
| `join.click` | A sharp sound change is present at a cut. | Add a short audio fade (5 to 30 ms) and render again. |
| `join.gap` | Audio was inserted or lost at a cut. | Inserted: encode the audio once for the whole video. Lost: re-render so each clip keeps its full audio. |
| `join.words` | The render's audio differs from the edit list, or a source edge excludes voice from a planned first or last word. | Match the render to the edit list, or move the named source edge to keep the word. |
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
| `plan.holds` | A hold declared in the edit list is not a still picture, or has sound where the edit list declares silence. A small encoder redraw at a video keyframe does not count as movement when spread across the frame, and is reported as a note. | Render the hold as declared, or correct the edit list. |
| `edit_list.evidence` | An evidence file listed in the edit list is missing, or no longer matches the sha256 recorded for it, so the cuts it attests are unconfirmed. A matching file is reported as checked outside Yap Studio, never as a pass. | Point the edit list at the file that was checked, or review it again and record its current sha256. |
| `structure.av_length` | Picture and sound end at different times. | Render both streams to the same endpoint. |
| `structure.audio_timestamps` | The audio timestamps jump ahead or step back somewhere in the file. A player that follows them inserts silence or overlaps sound there. The finding lists each break with its time and size. | Encode the audio once for the whole video and mux it again. |
| `loudness.integrated` | Overall loudness misses the target. | Adjust the final mix toward the stated target. |
| `loudness.true_peak` | The true peak is above -1 dBTP. | Lower the limiter ceiling. |
| `loudness.parts` | Two neighbouring parts from different sources differ by more than 3 LU. Without an edit list, only parts at least 3 s long are compared. | Normalize the loudness of that part to match the rest. |
| `picture.black` | A black interval was found. | Replace unintended black frames. |
| `picture.freeze` | A long frozen interval was found. | Replace the held frames if they are not intentional. |
| `picture.flash` | One or two stray frames flash at a cut: the picture changes into them and out again within two frames of the join. With an edit list, frames that are in the source at that point are not counted. | Render the cut again so the outgoing segment's last frame meets the incoming segment's first frame. |
| `captions.contrast` | Warns when a caption stays hard to read against the picture in the final. | Use `yap-studio captions DIR --back box` for phrases the automatic halo does not fix, then run again. |
| `captions.words` | A word heard in the final has no caption, a caption word is not heard, or several confidently matched words consistently start over 0.5 s before or after the speech (warning). The check transcribes the final's voice with the project's speech engine and caches the result for that audio. | Listen at the time. Restore missing spoken words, fix caption text, or align caption word starts with the speech. Then render and run again. |
| `detect.cuts` | Lists the clear cuts found without an edit list. Jump cuts inside one take may be missed. | Nothing to fix. Add `--edl` for a full check, including sync and words. |
| `platform.shape` | The video shape does not fit the platform. | Render the required horizontal or vertical version. |
| `platform.duration` | The video exceeds a duration limit or warning threshold for the selected format. | For YouTube uploads over 15 minutes, verify your YouTube account; trim only if you cannot verify. Above the hard limit, shorten it or choose another format. |
| `platform.resolution` | The frame size misses a platform rule. | Render at a supported size. |
| `platform.fps` | The frame rate misses a platform rule. | Render at a supported frame rate. |
| `taste.rules` | The finished picture breaks an approved taste rule applied to this render, or its receipt names a different style. Rules that could not be measured have a reason. | Render again with the approved style, or revoke the rule if it no longer holds. |

## Speech engines and first-run downloads

Speech checks use the Parakeet model. On Apple Silicon it runs through parakeet-mlx and
downloads about 2.5 GB the first time. Elsewhere it runs through onnx-asr and downloads about
670 MB the first time. You can select an engine with `--asr` or
turn word naming off with `--asr none`.

The word check compares the render's audio at each cut with the edit list, so it runs without
a speech engine. The engine names the word that changed and checks planned boundary words;
without one, the report says so in a note. A cut with no speech nearby is `skipped`.
A pass means the render's audio matches
the edit. A cut through a source word without planned boundary intent gets a `join.listen`
note; clipping an identified planned boundary word fails `join.words`.

## JSON report and exit codes

Print the `yap-studio/verify-report@1` JSON report on standard output:

```console
yap-studio verify final.mp4 --edl edits/cut.json --json
```

The JSON report always lists every finding and every check. Add `--detail full` to include the per-cut and per-segment measurements.

The exit codes are:

- `0`: no failures
- `1`: a failed check, or a warning with `--strict`
- `2`: invalid input, a missing dependency, or another command error
- `3`: the command refused (for example a timeline that breaks the rules)

Use `--report report.json` to save the report.

Use `--loudness note` for an intermediate render whose loudness is set when the video is
assembled. Overall loudness and true peak are reported as information; differences between
parts still warn. The default, `--loudness check`, checks overall loudness and true peak
against their limits.

## What it does not check

Yap Studio does not judge writing, pacing, or whether a subject is framed well.
It checks that captions agree with the speech it hears; it does not guarantee the transcript
is right. Its taste checks cover only supported, measurable approved style fields.

Agents can call verify as a tool through the Claude Code, Codex and Claude Desktop plugins;
see [install.md](install.md#use-it-from-claude-code-codex-or-claude-desktop).
