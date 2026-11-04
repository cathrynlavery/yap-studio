# Commands

`yap-studio --help` lists every command; `yap-studio COMMAND --help` shows its flags. Most
commands take `--json`. For the order you run them in, see [editing.md](editing.md).

## Every command

| Command | What it does |
| --- | --- |
| `init` | Initialize a project from raw takes (`--take`) or a voice memo (`--voiceover`). |
| `ingest` | Analyze takes and produce source words. |
| `words` | Print compact transcripts with word ids. |
| `words fix DIR --word ID --to TEXT [--json]` | Correct one source word in this video's captions and spoken titles. |
| `words fix DIR --heard TEXT --to TEXT [--json]` | Correct every matching occurrence in this project. |
| `words fix DIR --list [--json]` | List occurrence fixes and heard-text fixes. |
| `words fix DIR --word ID --remove [--json]` | Remove one occurrence fix. Use `--heard TEXT` to remove a heard-text fix. |
| `edit schema`, `edit check`, `edit rebase` | Describe and inspect the story (`edit.json`). |
| `fillers` | Test joins and propose removals; `--apply` applies verified cuts. An unverified join stays proposed. |
| `cut` | Render and verify the ordered story. |
| `joins` | List the joins in the cut and what happens to each one. |
| `timeline add`, `remove`, `show`, `check` | Manage word-anchored timeline lanes. |
| `captions` | Render project or clip captions. |
| `graphics` | Render project title layers. |
| `music` | Prepare a music bed or voice only. |
| `sfx list`, `kit`, `add` | List the made kit with `--kit made` or manage your ElevenLabs kit. |
| `voice` | Propose and review a fix for a misspoken word in your own voice. |
| `assemble` | Render and verify the final video. |
| `grade show`, `grade set`, `grade sample` | Inspect, sample or override the picture grade. |
| `video-type show [DIR] [--json]` | Show the saved default, this video's choices and the questions. |
| `video-type set [DIR] [--type TYPE] [--broll SOURCE ...] [--default] [--json]` | Record a video's type and B-roll sources, or save the default without DIR. |
| `run` | Run every stage, stop at the first failed check, resume on rerun. |
| `status` | Show stage states and the next action. |
| `verify` | Verify a finished video render. See [verify.md](verify.md). |
| `review` | Review a render and leave timed comments. See [review.md](review.md). |
| `home [--workspace DIR] [--phone] [--detach] [--stop] [--json]` | Open Yap Studio home, the guided page for a workspace. See [home.md](home.md). |
| `home wait [--workspace DIR] [--timeout SECONDS] [--json]` | Wait for the next button press, sent notes or approval on Yap Studio home. |
| `home ack ID --state claimed\|done\|failed [--project DIR] [--message TEXT] [--json]` | Tell Yap Studio home what happened to a request. |
| `versions save`, `list`, `rollback`, `approve`, `current` | Save and list immutable project versions. |
| `feedback build`, `apply`, `check`, `reply`, `replies`, `ask`, `answer` | Build feedback batches and check keeps. |
| `outline build`, `status`, `send`, `review` | Review the story before rendering. |
| `style` | Validate and manage video styles. See [styles.md](styles.md). |
| `brand fonts` | Manage creator font files. |
| `taste check`, `learn`, `show`, `approve`, `reject`, `revoke`, `add-proposal`, `propose` | Inspect and decide creator taste preferences. |
| `media add`, `list`, `show`, `sheet`, `sound`, `describe`, `pick`, `inventory` | Catalog your own clips and images. |
| `broll suggest`, `plan` | Suggest and plan supporting catalog media. |
| `stock search REQUESTS.json --cut-words FILE [--style SPEC] [--project DIR] [--per-phrase N] [--json]` | Show stock choices for gaps in your footage on a pick sheet. |
| `stock get DIR --pick CODE [--pick CODE ...] [--sheet sN] --yes [--json]` | Download the stock clips the creator approved. |
| `stock credits DIR [--all] [--json]` | Print credits for placed stock clips, or all approved picks. |
| `pictures question DIR [--json]` | Show the imagery question and this video's recorded answer. |
| `pictures answer DIR yes\|no [--json]` | Record whether the creator wants imagery for this video. |
| `pictures ask REQUESTS.json --cut-words FILE [--style SPEC] [--project DIR] [--json]` | Needs a yes for this video. Show prompts, estimated costs and the total before generation. |
| `pictures make DIR --pick N [--pick N ...] [--sheet sN] [--max-cost USD] [--yes] [--dry-run] [--new] [--json]` | Make up to three approved pictures. Check the ceiling against WaveSpeed's quotes before each picture and stop if a charge exceeds its quote. Use `--new` only after approving payment again for an uncertain attempt. |
| `cards plan` | Plan cards at the words they name. |
| `templates list`, `show`, `preview` | Inspect and preview card templates. |
| `faces detect`, `check` | Measure faces and check video clearance. |
| `matte make`, `composite`, `check` | Make a person matte for an upright video. |
| `doctor` | Check runtime dependencies and quietly show newer releases, checking GitHub at most once every 24 hours. See [install.md](install.md) and [release notices](feedback.md#release-notices). |
| `update-notice on\|off` | Enable or disable release notices in Home and doctor. `YAP_STUDIO_NO_UPDATE_NOTICE=1` also turns them off. |
| `renderer install` | Prepare the pinned renderer. |
| `speech download` | Prepare the speech model. |
| `mcp` | Run the MCP stdio server. |
| `studio export`, `open`, `stop`, `catch-up` | Agent and debugging inspection, not editing. Nothing is read back from it into the project. |
| `watch pack`, `note`, `list`, `scan` | Prepare offline render review evidence. |
| `fillers-score` | Score filler removals against gold labels. |

## Project commands

`yap-studio video-type show [DIR] [--json]` shows the questions, saved default and this video's
choices. `yap-studio video-type set --type TYPE --broll SOURCE` saves the default in
`video-defaults.json` in the application data folder (`YAP_STUDIO_DATA_DIR` overrides it).
With DIR, `video-type set DIR --type TYPE --broll SOURCE` records this video only, and saves
the first complete answer as the default. Add `--default` to change the default too. Repeat
`--broll` for folder, album or stock; `--broll none` chooses none. A voiceover needs at least
one source. A screen recording saves no sources unless you give them. `init --video-type TYPE --broll SOURCE` overrides
the default for a new project without changing it. `status` shows both choices.

`yap-studio grade show DIR [--json]` prints the effective look, strength and source.
`yap-studio grade set DIR --look film [--strength 0.7] [--json]` writes a project override.
Use `--look none` to leave this project's footage ungraded. Use
`yap-studio grade set DIR --clear [--json]` to follow the creator's style again.

`yap-studio grade sample DIR --time T --x X --y Y [--json]` reads a colour from
`cut/cut.mp4` without writing. Coordinates run from 0 to 1. It prints the hex colour,
Oklab chroma and whether the colour is colourful enough.
Use `yap-studio grade set DIR --look eyedropper --time T --x X --y Y` to save that
sample point, or `--color '#1E6BD6'` to save a colour directly. Eyedropper starts at
strength 0.5; `--strength S` changes it. `grade show` includes the resolved colour
and any saved sample point. Run `yap-studio cut DIR` before sampling.

| Command | What it does |
| --- | --- |
| `edit schema [--json]` | Prints the `edit.json` format and a short example with placeholder hashes and word ids; `--json` prints the JSON Schema. |
| `edit check [DIR] [--json]` | Resolves the story's pieces, frames and joins without rendering. |
| `joins DIR [--json]` | Lists every join in the cut, what happens to it under the style's `moves.jump_cuts` and why; `--json` adds the words on both sides and the measurements. Needs a fresh verified cut. Measures faces on the cut when they are not saved yet. |
| `edit rebase [DIR] [--dry-run] [--pin-unmatched] [--json]` | Re-anchors words after a transcript change and keeps the recorded cut's resolved edges when possible. |
| `run [DIR] [--json] [--note TEXT] [--allow-violations]` | Runs every stage after the edit in order, stops at the first failed check with the reason and the fix, and resumes there on the next run. |
| `music [DIR] [--json] [--force] [--yes] [--new] [--dry-run]` | Prepares a supplied track, an approved ElevenLabs bed, or no music after the cut. |
| `sfx list`, `sfx kit --yes`, `sfx add NAME` | Inspect a kit (`--kit made` needs no account) or generate your ElevenLabs kit. |
| `assemble [DIR] [--json] [--force] [--allow-violations]` | Mixes audio, composites picture and verifies the final delivery. |
| `captions [DIR] [--json] [--force] [--back shadow\|box\|none]` | Renders project captions. Weak phrases get a halo automatically in the final. `--back box` or `shadow` saves a choice for phrases still weak. `none` turns backing off. Choices stay in `project.json`. |
| `graphics [DIR] [--json] [--force]` | Renders timeline titles and cards using the saved style. |
| `cover DIR --headline TEXT [--look LOOK] [--frame N] [--json]` | Makes a checked PNG cover for a finished project. See [captions-and-cards.md](captions-and-cards.md#covers). |
| `timeline add FRAGMENT [DIR] --lane agent\|broll [--replace-lane]` | Checks every element and merges it into its writer's lane. `--replace-lane` clears the lane first. |
| `timeline remove [DIR] --lane agent\|broll [--id ID]` | Removes an id (repeat `--id` for several), or clears the named lane. |
| `timeline show [DIR] [--json]` | Shows merged elements with their output frames, seconds and payloads. |
| `timeline check [DIR] [--json]` | Checks anchors, media hashes, lengths and overlaps against the current cut and style. |
| `media sound ID [--json] [--top N]` | Finds loud moments and saved speech phrases in a cataloged clip. |

## Voice fixes

Each line starts with `yap-studio`. Add `--json` for a JSON result.
Use word ids from `words DIR` for `--from` and `--to`.

| Command | What it does |
| --- | --- |
| `voice samples DIR` | Exports clean speech from your takes. Free and local. |
| `voice consent DIR` | Shows the consent statement. |
| `voice consent DIR --yes [--by NAME] [--voice-name NAME]` | Records your agreement, even before this project has samples. |
| `voice consent DIR --withdraw` | Stops approved clone fixes from applying to later cuts. |
| `voice create DIR --yes [--dry-run] [--samples PATH]` | Makes your voice after consent. Uses one voice slot. |
| `voice link DIR [--name NAME] --yes [--samples PATH]` | Links an instant clone. The name can come from your recipe. |
| `voice propose DIR --from WORD --to WORD --say TEXT [--from-take-words FIRST LAST \| --clone] [--why TEXT]` | Finds clean recorded words first, then offers the clone fallback. |
| `voice make DIR --fix ID` | Builds recorded-word fixes for 0 credits. Shows the cost for clone fixes. |
| `voice make DIR --fix ID --yes --max-credits N [--plain] [--dry-run] [--new] [--confirm-retry]` | Makes clone clips within your credit limit. Repeat `--fix` for up to three fixes. |
| `voice approve DIR --fix ID` | Records approval given in chat. |
| `voice reject DIR --fix ID` | Records a choice to keep the original. |
| `voice list DIR` | Lists fixes, their source kind and status. |
| `voice delete-voice DIR --yes` | Deletes a project voice and withdraws consent. Refuses shared manifest voices. |

The consent statement is:

"This is my own voice. I agree to Yap Studio making an ElevenLabs voice from my
recordings and using it to re-say words I misspoke in my videos."

`--samples PATH` uses a samples manifest instead of this project's `voice/samples.json`.
It has the same format (`yap-studio/voice-samples@1`): a `samples` list whose rows hold
`path`, `take`, `start_s`, `end_s`, `duration_s`, `sha256`, `size_bytes` and optional
`ranges`. Paths are relative to the manifest's folder and stay inside it. Takes can come from
other projects. Both `link` and `create` accept 1 to 25 WAV files. File names must be
unique without their extensions, and every file must match its hash and size. Consent keeps
the manifest's hash and file count, not its path.

`link` compares the voice's samples with the files by name without the extension and by byte
size, because ElevenLabs lists an uploaded WAV as an MP3. If the CLI cannot upload, `create`
prints steps to upload the selected files on the ElevenLabs site; then run `link`. Listen on
the review page before choosing a fix. `--new` makes another paid generation. Use
`--confirm-retry` only after checking the earlier call and account history.
See [audio.md](audio.md#voice-fixes) for consent and cost.

Recorded-word fixes need no clone consent, account or paid approval. The mouth coverage rule
still applies. The review card says which recording supplied a fix, or that the clone re-said it.

`voice/recipe.json` in the creator's Yap Studio data folder (`YAP_STUDIO_DATA_DIR`) controls
the clone fallback. Schema: `yap-studio/voice-recipe@1`. Optional fields are `voice_name`,
`model_id`, `voice_settings`, `tag`, `strong_tag`, `strong_tag_words`, `context`, `pace`
(`fit` or `off`), and `pronunciations` with `word`, `type` (`ipa` or `alias`) and `value`.
For example: `{"schema":"yap-studio/voice-recipe@1","tag":"[test accent]","pace":"fit"}`.
Omitted fields keep neutral defaults. Review, captions and heard checks use plain words;
cost counts the text sent. A changed recipe makes a new request. If a respelling warns,
try `make --plain`, which sends no tag or pronunciations. The seed stays fixed per fix.

## Timeline lanes

Both lanes use `yap-studio/timeline@1` fragments. Agent ids start with `agent-`;
B-roll ids start with `broll-` and can contain inserts, overlays and backgrounds. Word anchors follow the
cut when words move; use `segment` when a word appears more than once. Titles,
`chapter` elements and graphics can omit `end` and `hold_s` to use the current
style's duration. Chapters belong to the agent lane and require a short `title`.
For a before/after wipe, add `{"id":"agent-grade","kind":"grade_compare",
"lane":"agent","start":{"time":2},"hold_s":3}` through `timeline add`.
It needs an active grade and a clear 2 to 8 second window. At most two are allowed.

Timeline commands accept `--json`. Add writes the whole fragment only when its
checks pass. It also takes the `agent-lane.json` that `cards plan` writes, and refuses
one planned against older cut words. Warnings exit 0; rule violations and stale cuts exit 3. After a
style change, run `timeline check` again; `status` also lists conflicts and a
command to fix the agent lane.
