# Limits

What version 1 does not do yet, and what it never judges.

## V1 limits

- Vertical 1080 × 1920 stays the default. New horizontal talking-head videos use
  `init DIR --take FILE --shape horizontal` (or `"shape": "horizontal"` in project.json).
  Horizontal phase 1 supports horizontal source takes, hard cuts, audio/music, caption gaps
  and fixed lower captions with wider lines and safe margins at 1920 × 1080, plus full-frame
  B-roll clips and stills and B-roll in explicit boxes, opening titles, chapter cards and
  name tags and floating cards; platform is `youtube` or `none`. Valid chapters include a
  copyable YouTube list. Floating paper cards use the creator's fonts and colours, a 660 px
  panel, 24 px corners and a soft shadow. They sit beside the speaker, inside title-safe and
  above captions. Face measurement is required; if neither side clears every face during the
  card, choose a moment with room beside the speaker. Only the card animates.
  Available built-ins: terminal, stat, checklist, text_tag, image_card, transfer,
  image_compare, steps, flow, chart, quote, progress_stat, specs_list, callout_circle,
  name_tag, success_check, edit_timeline, storyboard, jiggle_delete, file_drop, video_card,
  prompt_bar, comment_cta, comment_drop, icon_strike and choices. Long content may need
  shortening for the floating panel. Registry progress ring and slam headline also fit;
  install-only registry helpers cannot be cards. Separate video windows keep their native
  entrance and exit; the optional flying motion is disabled for those cards.
  Not available in horizontal: playing_grid (full-frame grid; use one video_card), reel_feed
  (phone frame; use video_card), gravity_drop (full-frame words; use checklist), big_number
  (behind-speaker layout; use stat). Side panels and full-frame blurred card layouts are later
  options. Explicit card boxes and behind-person floating cards are unavailable.
  Portrait and square inserts use blurred, darkened fill. Split B-roll, video letters,
  zooms, transitions, person effects, 3D captions,
  grades, applied taste preferences, covers and studio export fail with a fix line.
  Style/gallery samples remain vertical. Use the review page for final horizontal playback.
  Standalone `verify` also accepts other shapes and can check their platform rules.
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
