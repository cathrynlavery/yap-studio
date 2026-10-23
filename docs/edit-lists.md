# Edit lists

Verify compares a finished render with an edit list when you pass `--edl`. This page lists the
formats it reads and the optional fields that describe overlays, holds, declared joins and
cuts checked elsewhere. The checks themselves are in [verify.md](verify.md).

## Formats it reads

Yap Studio reads:

- Yap span JSON
- ranges JSON with named sources and source ranges
- FCPXML files
- Editor XML files

Yap span JSON can also name the first and last spoken word the editor intends to keep:

```json
{
  "source": "raw/take.mov",
  "segments": [
    {"start": 0.4, "end": 2.8, "first_word": "Bright", "last_word": "hills"},
    {"start": 3.8, "end": 6.2, "first_word": "Seven", "last_word": "garden"}
  ]
}
```

`first_word` and `last_word` are optional, non-empty strings. A speech engine identifies
the nearest matching word in the uncut source near the span's edge. Verify smooths its
10 ms energy envelope and locates the word's onset or offset relative to its own peak
and the valley beside the neighbouring word. Connected words can be separated by a
quiet dip without reaching silence. The search allows speech timestamps to run up to
0.4 seconds early or late; agreeing neighbouring word shapes can correct a shared clock.
If the first word's voice starts more than 20 ms before the span's start, or the last word's
voice ends more than 20 ms after its end, `join.words` fails and names the word, side and
excluded voice in milliseconds. This catches incorrect span times even when the render
matches the plan. The render's first and last segments use the same source check.
Source controls include context on both sides of each edge and are transcribed in the same
batch as the render comparison. If the word cannot be identified near the source edge, or
its energy boundary cannot be located, `join.listen` asks you to listen; recognition alone never
proves that a word was cut off. The render's audio is still compared with the plan separately.
For a segment with `runs`, intent applies only to the first run's start and last run's end.
The cut stage writes these fields for outer word anchors, using the first and last kept
words; time anchors and edges created by removals leave them out. Other edit-list formats
leave them unset. Without a speech engine, the report says the intent checks were skipped.

A ranges JSON file can point to one or more source files:

```json
{
  "sources": {"main": "raw/take1.mov"},
  "ranges": [
    {"source": "main", "start": 0.4, "end": 2.8},
    {"source": "main", "start": 3.8, "end": 6.2}
  ]
}
```

Range `start` and `end` are source times in seconds. The fields below are optional and use
the output's timeline instead.

### Overlays

An overlay marks a stretch of the output where something covers the picture, such as
burned-in captions. By default the whole picture is left out of the picture checks for that
time.
Add `region` (`[x, y, width, height]` in output pixels) when the overlay covers only part of
the frame:

```json
"overlays": [{"start_in_output": 5.967, "duration": 168.0, "region": [0, 1480, 1080, 200]}]
```

Sync and the frame checks then compare the rest of the picture instead of skipping it. When
an overlay without a region hides the middle of a part, sync measures the part at the
nearest moment the overlay does not cover.

### Holds

A hold is a still picture you meant to leave in the output, such as an end card over
silence. Frames count from the first output frame, and `end_frame` is not part of the hold:

```json
"holds": [{"start_frame": 5219, "end_frame": 5264, "audio": "silence"}]
```

Hold frames count toward the planned frame and sample totals. A range placed after a hold
starts once the hold ends. A frozen picture inside a declared hold becomes a note instead of
a warning, and so does missing sound that falls inside a silent hold. A declared hold warns
under `plan.holds` if its picture moves or its sound peaks above -90 dBFS; the first and last
50 ms of the hold are left out of the sound check. `audio` must be `"silence"`.
A single small step at a video keyframe, spread across the whole frame, is the encoder
redrawing the held picture and does not count as movement.

### Declared joins

When the edit list cannot show a cut, for example because its one range is a finished
master, you can declare the words a join should play:

```json
"joins": [
  {
    "out_frame": 274,
    "before": "going kind of viral",
    "after": "Now if you don't",
    "absent": ["mansplainers"],
    "once": ["kind of viral"]
  }
]
```

- `out_frame`: the first output frame after the join.
- `before` and `after`: the words heard just before and just after it. At least one is
  needed.
- `absent` (optional): phrases that must not be heard anywhere in the render, such as the
  line you cut.
- `once` (optional): phrases that must be heard exactly once.

Yap Studio transcribes the render around the join twice, starting 5 s and 9 s before it
and ending 3.5 s after. The join passes when both windows hear the `before` words and then
the `after` words in a row, running across the join: the `before` words end by the join
frame and the `after` words start from it, each within 1.5 s of it, with 0.3 s of slack for
word timing. Case, punctuation and fillers such as "um" are ignored.
`absent` and `once` are checked against a transcript of the whole render. Results are
reported under `join.words` with `"source": "declared"` in the evidence. This needs a speech
engine but not the source files.

### Finished masters and cuts checked elsewhere

Set `"composite": true` when the ranges point at a render that already contains its cuts,
so the edit list cannot show them. A single range that copies a source of the same length
as the render is treated the same way.

Cuts checked by other means can be listed as evidence:

```json
"evidence": [{"path": "checks/cut-review.json", "sha256": "<64 hex characters>"}]
```

Yap Studio hashes each file, reports whether it matches, and labels it "checked outside Yap
Studio". Evidence never counts as a pass, and the cut checks stay skipped. A missing file or
a hash that no longer matches is a warning under `edit_list.evidence`.

The report's `edit_list.cut_coverage` says how the cuts were covered, by the weakest
coverage of any cut, and the summary prints it unless every cut was checked:

- `checked`: every cut in the edit list was compared with its sources.
- `partial`: some cuts were checked, against their sources or by a declared join at their
  frame, and the others could not be, for example because a source file is missing.
  `planned_cuts`, `checked_cuts` and `declared_cuts` give the counts.
- `declared`: declared joins were checked by their words. When the edit list has cuts, every
  cut not compared with a source has a declared join within one frame of it.
- `attested`: listed evidence covers the cuts; `edit_list.attested` has each file's hash and
  whether it matches.
- `not_exposed`: the edit list describes a finished render, so its cuts cannot be checked
  from it.
- `unavailable`: the edit list has cuts, but none could be compared with a source.
- `none`: the edit list is one continuous range with no cuts.
