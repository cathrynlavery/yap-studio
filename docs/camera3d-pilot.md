# Vertical camera captions pilot

Camera captions are now the [3D camera caption look](captions-and-cards.md#3d-camera-captions).
The script below remains for experiments on a cut.

This pilot plans or renders a 1080 x 1920, 30 fps cut with captions at three
depths around the speaker. It reads the creator's resolved style and brand.

Run from the repository root after the project's cut stage:

```sh
uv run python3 scripts/camera3d_pilot.py PROJECT_DIR OUT.mp4 --plan-only --style STYLE.json
```

Use `--style SPEC` to choose a style file, saved style, or preset. Without it,
the pilot uses the saved default, then the project style. The plan records the
source, resolved look, fonts, and any conservative font metric fallback.

Use `--no-grain` to leave off the finish pass. Use `--keep DIR` to choose where
the prep assets and composition live. By default they stay beside `OUT.mp4` in
`OUT.camera3d/`. The script prints the composition path for preview.
It also writes `OUT.plan.json`. The rendered picture is opaque.
Bleep swear words follows the Sound choice and is off by default. When on, the
pilot uses the shared word list and the style's add and remove lists. Captions
keep the first letter and mask the rest. Listed words get a speech-level 1 kHz bleep.
The bleep has 10 ms fades. The rest of the decoded audio samples stay intact.
The edited audio is stored losslessly in the final MP4.
The default render uses one worker and screenshot capture, matching layer renders.
`--workers N` can render faster, but multiple workers may shift chunk frames and
are not frame-identical to the default.

Use `--plan-only` to prepare the cut and write `OUT.plan.json` without rendering.
Later runs reuse prep when the cut hash and matte settings match. A render run
rebuilds the composition folder from the current plan.

The plan breaks sentences at punctuation or gaps over 0.6 seconds. It groups up
to four words per clause, with comma and conjunction breaks. Each group arrives
0.2 seconds before speech on the 15 fps grid and fades over three steps after
replacement. Captions use three depths and stay within the portrait safe area.
The hero stays behind the speaker only when at least 90 percent of its projected
letter ink is visible on every frame. The planner reads matte frames in a stream.
It tries beside the head, above the head, and a smaller size before moving the
hero to the front. The plan records the chosen placement and worst visible share.
The hero
enters with its word, replaces that word in its caption group, lasts at most
three seconds, and fades before the next sentence. The plan samples the cut
picture under each caption. Every caption gets a contrast halo. Captions at rest
are opaque and sharp. Their depth blur is capped at 0.6 px. Ghosts appear only
while text moves, and fades last at most three steps.

After a render, the pilot writes `OUT.checks.json`. The `captions3d.behind_words`
check measures the transparent behind layer against the person matte. Any frame
with more than 10 percent of a word hidden fails. The `captions3d.contrast`
check measures each caption in the middle of its rest window and fails below
4.5:1. The script exits with an error if either check fails.

The camera begins with the measured opening whip, keeps a slow creep, trucks at
sentence changes, and pushes into the last sentence. The painter samples text at
15 fps with seven ghosts, while it moves the plate and person mask every frame.
Frame 0 uses the cut's own framing. The base plate scale is 1.08, and the plan
checks that normal frames show only the real picture. Brief punches stay under
1.25 scale. The plan records any padding visible during a whip and rejects more
than eight consecutive padding frames or 20 percent padding in one frame.
Captions start at 60 to 84 px; only placement fallbacks may use smaller text.
Masthead captions stay at 64 px or larger.
Heroes use the masthead when it has room and record their chosen zone.

For a plan review, run `scripts/camera3d_plan_sheet.py` with the plan JSON, the
prep plate, and an output PNG. It samples every five seconds, labels each tile,
and marks the real-picture edge in yellow.

The pilot omits the body wipe and word rings. It requires a locked portrait take.
When a phrase has no safe depth placement, the plan records a smaller size, a
split, a flat torso caption, or an unplaced group in its `fallbacks` list.
