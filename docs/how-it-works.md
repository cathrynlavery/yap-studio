# How it works

Yap Studio is a Python command-line engine. A project folder holds its source
references, edit decisions, generated media and verification receipts. The CLI
and MCP server use the same command registry. An agent or a person chooses the
story; the engine resolves the edit, renders it and checks the result.

## From takes to a final

1. `init` records the raw takes and selected style in `project.json`
   and creates empty timeline lanes.
2. `ingest` probes the takes and transcribes speech. Each take gets stable word
   IDs in `ingest/<take>/words.json`, with source timestamps and hashes.
   Yap Studio transcribes each take once on this computer and keeps the words in
   its cache. The style page starts this in the background, and ingest and the
   watch pack reuse it.
3. You write `edit.json` using word or time anchors. It lists the story segments
   in playback order and records the hashes of the source word files.
4. `fillers` proposes pauses, fillers and restarts. Applied removals must pass
   the join check. `cut` renders the selected story and maps source word IDs to
   their new positions in `cut/cut-words.json`.
5. `captions` and `graphics` render layers with the saved style, bundled fonts
   and pinned HyperFrames renderer. Timeline anchors follow words after a cut
   changes. `music` prepares a supplied track or records that there is no music.
6. `assemble` composites picture, mixes sound and writes `final/<name>.mp4` plus
   the edit list and receipt used to verify the final. `run` performs the stages,
   checks platform rules and saves a version; rerunning resumes from stale stages.
7. `review` serves the local browser page. Comments belong to an exact render.
   Versions preserve earlier renders and their review context.

Receipts record inputs, hashes, outputs, timings and checks. Changed inputs mark
dependent stages stale; unchanged inputs reuse existing work. Stale word anchors
and invalid joins stop the pipeline with a reason to fix.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/yap_studio/cli.py`, `registry.py`, `mcp_server.py` | Command entry points and shared tools |
| `src/yap_studio/project/` | Project contracts, schemas, anchors, receipts and status |
| `src/yap_studio/stages/` | Ingest, cut, layer generation and assembly |
| `src/yap_studio/checks/`, `edl/`, `transcript/` | Verification, edit-list readers and speech backends |
| `src/yap_studio/render/`, `templates/`, `fonts/` | Renderer installation and portable layer assets |
| `src/yap_studio/style/`, `taste/`, `feedback/` | Saved styles, approved preferences and review edits |
| `src/yap_studio/review/` | Browser review server, static UI and persistent comments |
| `workspace/`, `scripts/build_workspace.py` | Creator workspace and release zip builder |
| `tests/` | Unit checks and synthetic media integration tests |

Speech models, renderer packages and browser downloads are cached outside project
folders. Saved styles and approved taste rules live in the local configuration
store; review data lives in the local application-data store. Environment
variables can override these locations. There is no required hosted Yap Studio
account. Optional phone links use Tailscale or the local network.

Version 1 runs on macOS only. Linux is unverified; contributions are welcome.
Source downloads and bundled assets retain the licenses listed in
[NOTICE](../NOTICE) and [LICENSES](../LICENSES).

Start with the [README quick start](../README.md#three-steps-then-you-talk), or run each stage
yourself with [Edit from the command line](editing.md). Use
`yap-studio edit schema` for the edit format and `yap-studio status PROJECT` for
the next action.
