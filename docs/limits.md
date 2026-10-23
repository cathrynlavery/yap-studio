# Limits

What version 1 does not do yet, and what it never judges.

## V1 limits

- Project editing and rendering target a vertical 1080 × 1920 final. Standalone
  `verify` also accepts other shapes and can check their platform rules.
- The word-based filler and restart rules are English only. For other project
  languages, `fillers` proposes pauses only.
- Person cutout and face detection use macOS Vision and the Xcode command line
  tools.
- Version 1 runs on macOS only. Linux is unverified; contributions are welcome.
- Music generation needs your own ElevenLabs account and your approval
  (`yap-studio music DIR --yes`). Without one, supply a track or use no music.
  Scoring music to the cut is deferred; see [audio.md](audio.md).
- [Taste rules](styles.md#taste-rules) lists what Yap Studio checks and what the agent follows.

## What it does not check

Yap Studio does not judge writing, pacing, or whether a subject is framed well.
It checks that captions agree with the speech it hears; it does not guarantee the transcript
is right. Its taste checks cover supported, measurable approved style fields and face clearance.
