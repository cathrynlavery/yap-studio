# Music, sound effects and loudness

Music and sound effects are optional. Use your own track, or generate with your own
ElevenLabs account. Setup is in
[install.md](install.md#elevenlabs-for-music-and-sound-effects).

## Voice fixes

A voice fix first looks for clean matching words in your project's recordings, including
removed parts. It prefers the same take. Recorded words cost 0 credits and need no clone
consent or account. If none fit, it offers your instant clone as the fallback.
Unused takes without transcripts are skipped.
The rest of the take keeps its timing. Listen to Before and After, then choose
**Use the fix** or **Keep my original**. Only approved fixes go into the next version.
Both players use the version you selected. Saved choices stay visible when comments reload.
Removing a fix's whole phrase skips it in the cut and final without changing its approval.
Keeping only part of its slot still stops the cut.

Recorded words must fit with a small timing adjustment or the next quiet gap.
Proposals check this for words you choose yourself too.
Their audio stops at neighboring words.
Short recordings fade into nearby room tone. Its level comes from gaps outside the words.
Room-tone checks skip recorded and cloned speech and say when no quiet frames remain.
Quiet gaps inside cloned phrases and room-tone padding are checked before approval
and in the cut.
Loudness checks account for mono recordings in stereo cuts.
Audio-identical stand-ins can still supply recorded words without another ingest.

The cut keeps patched source audio in `cut/source-audio/` so checks using
`cut/spans.json` hear the approved fixes. The final does the same in
`final/source-audio/` for `final/verify-edl.json`.
Patched audio stays in WAV files, so verification does not need to remux the video.

The clone fallback needs a paid [ElevenLabs](https://littlemight.com/recommend/elevenlabs)
(affiliate link) account. Your key needs Text to Speech and Voices access.
With Models and User read access Yap Studio can read the credit rate and your balance,
and with History read it can show what each fix was charged. Without them the cost is
an estimate, and the cost note says so.

Read the statement with `yap-studio voice consent DIR`:

"This is my own voice. I agree to Yap Studio making an ElevenLabs voice from my
recordings and using it to re-say words I misspoke in my videos."

Run it with `--yes` only if you agree. Consent is saved in `voice/consent.json`, and you
can give it before this project has samples.

Your voice can come from this project's takes or from recordings across your videos.

- This project: run `yap-studio voice samples DIR`, then `yap-studio voice create DIR --yes`.
  This uses one voice slot. If the CLI cannot upload, it prints steps to upload
  the files it lists on the ElevenLabs site. Then run
  `yap-studio voice link DIR --name "NAME" --yes`.
- Across your videos: add `--samples PATH` to `create` or `link`. `PATH` is a samples manifest
  in the same format as `voice/samples.json` (see [commands.md](commands.md#voice-fixes)).
  You can use another project's exported `voice/samples.json` without changing its paths.
  Consent keeps the manifest's hash and file count. Yap Studio will not delete this shared
  voice. Delete it on the ElevenLabs site if you mean to.
  Relinking the same voice without `--samples` keeps this protection.

`link` checks that the voice was made from exactly these files: the same names, ignoring the
extension (ElevenLabs lists an uploaded WAV as an MP3), and the same byte sizes.
`create` checks the selected files' hashes and sizes again just before uploading.
Deletion stops if the linked voice changes while it is being checked.

Propose with `yap-studio voice propose DIR --from WORD --to WORD --say "cold plunge"`.
Use `--from-take-words FIRST LAST` to choose recorded words, or `--clone` to skip the search.
`voice list DIR` shows the source kind. `voice make DIR --fix f1` builds an own-recording fix
for free. For a clone fix it shows the cost; approve with `--yes --max-credits N`.
The estimate counts the text actually sent. Valid cached audio costs 0 credits in the
estimate, so you can reuse it with `--max-credits 0`. `--new` spends credits again.
A recipe change makes a new request. If the speech text changes after the estimate,
run `voice make` again without `--yes` to see the new cost.
Each fresh call's charge reduces the credit ceiling left for the batch. If the next fix
would exceed it, the command lists the fixes it did not make.

Your clone recipe lives at `voice/recipe.json` in your Yap Studio data folder, selected by
`YAP_STUDIO_DATA_DIR`. Without it, settings are neutral. This synthetic example lists the fields:

```json
{"schema":"yap-studio/voice-recipe@1","voice_name":"Test voice",
 "model_id":"eleven_multilingual_v2","voice_settings":{},"tag":"[test accent]",
 "strong_tag":"[strong test]","strong_tag_words":["house"],"context":true,
 "pace":"fit","pronunciations":[{"word":"example","type":"alias","value":"test-alias"}]}
```

`voice_name` is the default for consent and link. `model_id` and `voice_settings` control
the clone. `tag` prefixes its text; `strong_tag` replaces it for `strong_tag_words`.
Pronunciations match whole words, ignoring case: `ipa` wraps `value` in slashes and `alias`
uses it as written. A sentence's first word keeps its spelling. `context` sends neighbouring
words when true. `pace` is `fit` or `off`; off leaves speech unstretched and warns about a pause.
Tags and respellings stay out of review, captions and heard-word checks.
If a respelled fix warns, try `voice make DIR --fix f1 --plain --yes --max-credits N`.
`--plain` sends no tag or pronunciations.

For talking-head takes, full-frame B-roll must cover your face at the fix.
If it does not, the proposal explains why it was refused.
You can record the line again instead. `yap-studio voice consent DIR --withdraw`
stops clone fixes from applying to later cuts. Recorded-word fixes remain usable.
You can withdraw even when the original takes are offline. Consent is checked again
before uploading recordings or starting a paid request. It does not change saved videos.
Voice choices and consent cannot change while the cut or final video is being made.
You also cannot link, create or delete your voice during a run. Wait for the video to
finish, then try again. The review page shows the reason beside the fix.
If voice choices or consent change during assembly, it stops before publishing and asks
you to run assemble again.
See [commands.md](commands.md#voice-fixes) for all voice commands.

## Music

Run `yap-studio music DIR --file TRACK` to use your own track. `--file` reads the
path from the folder you run the command in and checks the track before saving it.
It saves a path relative to the project when the track is inside it, otherwise an
absolute path. Paths written by hand in `project.json` are relative to the project
or absolute: `{"mode": "file", "file": "track.wav"}`. The track must last at least
the cut; the end hold is silent.

`--loop` saves `music.loop` and repeats a shorter supplied track with a 0.5 s
equal-power crossfade until it covers the cut. The crossfade is a quarter of the
track for tracks under two seconds. Looping needs at least 1 s of track and applies
to supplied tracks only. `--no-loop` turns it off. A `music/bed.wav` from another
video already has its fade and silent end built in. Use the original track named
under `source.path` in its `music/receipt.json`.
A prepared bed is refused when the new cut runs past where its music ended.

`none` writes no bed. `auto` uses `music.file` when set, otherwise
none, even when `generate` is true. Set `music.mode` to `elevenlabs` or run
`yap-studio music DIR --yes` to approve generation. `--description` and `--tags`
shape the prompt. The free composition plan is saved at `music/plan.json` for
editing. `--new` approves another paid generation after a changed request;
recuts within the saved plan's length reuse cached audio without another generation.
`--confirm-retry` clears a failed attempt after you check your ElevenLabs
history. If a pending attempt came from another computer, use `--confirm-retry
--other-computer-stopped` only after checking that computer is stopped and your
ElevenLabs history. `--dry-run` previews the CLI request without writing project files.
The command records a pending request before spending credits, and an uncertain
failure needs confirmation before another attempt. Scoring music to the cut is
deferred. Signup and upgrade messages disclose the affiliate link. Whether
generated music can be used commercially depends on your ElevenLabs plan; check
their terms.

Assembly reads `music/receipt.json`: `mode: "none"` and `bed: null` mean voice
only. For `mode: "file"`, `bed` names a 48 kHz stereo float PCM WAV with exactly
`final_samples = (cut_frames + end_hold_frames) * 1600` samples per channel.
The content is trimmed and normalised to `internal.music_lufs` (default -16 LUFS),
with a raised-cosine fade over its last second (a quarter of the cut for cuts
under four seconds). The last content sample and the entire hold are zero.
The receipt records source path and SHA-256, trim, fade and normalisation.
A looped bed also records `loop` (passes and crossfade); its `trim` counts samples
of the looped track.
Assembly applies style level, dry open, ducking and timeline music windows;
changing those settings does not re-render the bed.

## Sound effects

The `soft` kit is a set of four sound recipes, not audio files. Each creator
generates it once with their own paid [ElevenLabs](https://littlemight.com/recommend/elevenlabs) account (affiliate link) using
`yap-studio sfx kit --yes`. `yap-studio sfx list` shows which sounds are ready
or stale; `yap-studio sfx add NAME --prompt TEXT --duration S --rule item --yes`
adds a personal sound. Generated audio lives in your user cache, never in the
repo or project. A recipe change makes its sound stale. A failed paid attempt
needs `sfx kit --confirm-retry --yes` after you check your account history.
Free-plan output is for non-commercial use only. Setup and upgrade messages
contain a sign-up link labelled as an affiliate link.

- A whoosh starts 0.2 seconds before an incoming section lands.
- A click follows each checklist row, step, specification row or chart bar reveal.
- A tick-roll starts a stat, progress stat or chart count; a chime marks its
  final value, a success check, and the end of the video before the silent hold.

Cues keep at least 2.5 seconds apart. When they crowd, the chime wins, then the
whoosh, the tick-roll and the click; the final receipt lists what was dropped.
Each cue peaks about 10 dB under your voice and never lowers it. The receipt
records each cue's peak against the voice and the hashes of the recipes and audio used.
The final's verify listens to each join without the sound effects, so a sound on a cut
is not taken for a bad join.

On a timeline element, `"sfx": false` mutes its cues, while
`"sfx": {"shift_s": 0.2}` moves them by 0.2 seconds (between -1 and 1).
Set the style's `music.sound_effects` to `"none"` to turn automatic effects off.

### More ElevenLabs sounds

Seven more sounds come from your ElevenLabs account. Run
`yap-studio sfx kit --kit extras --yes` once to make them, or add `--only typing` to
make one.

Three of them play on their own moment once you turn them on. Pick ElevenLabs under
More > Sound effects, then turn on Typing, Message ping or Camera shutter under More
sounds. All three start off. The style field is `music.sound_extras`, for example
`["typing"]`.

- Typing starts on the first typed character of a prompt bar, terminal or comment CTA.
  Short runs play for 0.35 seconds, or the whole sound if shorter. Longer runs
  stop where typing ends if it ends first. Short runs and trims use a 60 ms fade.
  The sound stays within the card.
- Message ping peaks when a comment drop or comment CTA lands.
- Camera shutter peaks on the first frame of an image card or image compare.

A sound you turn on replaces that card's own sound at the same moment. The receipt
records each trim and each replaced sound.

Applause, laugh, cheer and record scratch play only when you ask for them. Add a
`sound` element on the agent lane at the requested word:

```json
{"id": "agent-laugh", "kind": "sound", "lane": "agent", "start": {"word": "t1:12"}, "sound": "laugh"}
```

Use `applause`, `laugh`, `cheer` or `record-scratch`, or a name from the extras or
soft kit. Optional `gain_db` runs from -12 to 6. A requested sound starts at its
anchor. Add `"align": "peak"` to land its loudest point there instead. Requested
sounds keep their place when cues crowd. They follow the same voice level ceiling
and stop before the silent end hold. Assembly uses only ready cached sounds and
skips missing ones with a setup notice. It never generates paid sounds.

### Made by Yap Studio

This kit needs no account. Yap Studio makes its nine sounds on your computer from
small recipes of oscillators, noise, filters and envelopes, so they come out the same
every time. Pick Made by Yap Studio under Sound effects on the Sound screen, or set
`music.sound_effects` to `made`. `yap-studio sfx list --kit made` lists the sounds and
where each one peaks.

- A whoosh lands on the join to a new section.
- A swoosh-in lands on a card's first frame, and a swoosh-out where the card starts to leave.
- A click lands on each checklist row, step, specification row, chart bar or flow box.
- A pop lands on each flow tag or removed tile.
- A hit lands where a dropped word, file or comment lands, or where a strike completes.
- A riser builds into a chapter card and stops as the card appears.
- A tick-roll starts a count. A chime marks its final value, a success check, a confirmed
  prompt and the end of the video before the silent hold.

Each sound's loudest sample lands on its moment, so a whoosh starts before the cut and
peaks on it. Spacing, levels and ducking follow the rules above. Cues keep 2.5 seconds
apart, each peaks about 10 dB under your voice and never within 8 dB of it, and none
enters the silent end hold. When cues crowd, the riser wins, then the chime and the hit,
the whoosh, the tick-roll, the click and the pop, and last the swooshes. A card with its
own sounds keeps them and drops its swooshes. A card's swoosh-in and swoosh-out play as
a pair, and the swoosh-out plays only after its swoosh-in.

With ElevenLabs picked, the switch "Until ElevenLabs is set up, use Made by Yap Studio"
starts on (`music.sound_effects_fallback` is `made` by default). Assembly then uses this
kit while your ElevenLabs kit is missing or stale, and says so once per project. Turn the
switch off, or set the field to `none`, to leave the video without sound effects until
the ElevenLabs kit is ready. A style with sound effects off (`sound_effects: none`) stays
silent either way.

### Bleep swear words

Bleep swear words is off by default. Turn it on under Sound > More, or set
`music.bleep_swears` to `true`. Yap Studio then covers each listed word with a 1 kHz
tone made on your computer. The tone runs from just before the word to just after it,
and your voice is silent for that whole stretch, so no edge of the word slips through.
Captions, titles and cards show the word as its first letter and asterisks, like f***.

The built-in list holds common strong English swear words and their usual forms. Mild
words are not on it. Add words with `music.bleep_words_add`, or keep a built-in word
with `music.bleep_words_remove`. The final fails its check if a listed word can still
be heard or is shown in full. The join checks in verify listen to the edit without the
bleeps, so a bleep near a cut does not fail a correct join.

## Loudness

Projects default to `"audio": {"mode": "loudnorm", "integrated_lufs": -14,
"true_peak_dbtp": -1.7, "lra_lu": 11}`. Set `"audio": {"mode": "preserve"}` to
keep the retained voice PCM unchanged before AAC encoding. Music and enabled insert
audio still mix in preserve mode. Standalone captions keep the source loudness.
Loudnorm measures the final mix, applies one constant gain to the integrated target,
then uses an oversampled true-peak limiter with 0.5 dB of AAC headroom. It does not
fall back to time-varying loudness gain when the true-peak target needs limiting;
`lra_lu` is retained in the policy record and does not compress dialogue dynamics.
The receipt records the policy, target, input/output measurements, commands and delivery
asserts. Final and pilot audio must match their frame count at 48 kHz in the container.
Decoders may return fewer than 1024 extra terminal AAC samples; the receipt records
container duration and terminal padding too. `--allow-violations` permits editorial
timeline violations; media identity and end-hold bounds still apply.
Selected source audio streams get a portable copy-only reference under
`final/source-audio/` for edit-list verification.

A cut that joins more than one take, such as a voiceover memo and a pickup recorded
later, evens out their speech loudness. The cut measures each take over the words it
keeps, or over all of its kept audio when there are too few words, and changes each
take by one gain to match the first take in the cut. A gain stays within 12 dB. If
raising a take would clip it, every take comes down together, still within 12 dB, and
a take that would clip after that is raised less. This happens in both audio modes,
so preserve keeps the evened-out voice. `cut/receipt.json` records each take's level
and gain under `take_levels`. Set `"match_takes": false` in `"audio"` to keep the
recorded levels. If a jump between takes remains, its warning names the setting to
change or the take the jump is in.
