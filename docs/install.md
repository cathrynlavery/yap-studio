# Install

There are two ways in. The creator workspace is a folder you open in your agent app; the agent
sets it up and edits from there. A source install is for development, or for running the
commands yourself.

## Requirements

- A Mac with macOS 14 or newer.
- For the workspace, sign in to one of the four supported apps below. On a Mac, setup uses
  Homebrew for ffmpeg and Node and offers to install Homebrew when it is missing.
- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).
  The checkout uses Python 3.12 by default; uv can install it for you.
- [ffmpeg and ffprobe](https://ffmpeg.org/download.html) 6.1 or newer, on your PATH.
- [Node.js](https://nodejs.org/en/download) 22 or newer, including npm, for captions
  and graphics. Verification alone does not need Node or the renderer.
- At least 10 GB free on a local disk, plus room for your takes, intermediate layers
  and final videos. Longer projects need more space. Keep the working folder outside
  cloud-synced folders.
- Internet access for installation and the first renderer/browser and speech-model
  downloads. Editing uses local files after those dependencies are cached.

Your video files stay on your computer. Still frames from pinned notes and contact sheets of
your clips go to Anthropic in Claude Code and the Claude app's Code tab, or OpenAI in Codex,
along with the transcript, your notes, the project files your agent reads and command output,
as part of your agent session. On personal ChatGPT plans, OpenAI may train on it unless
"Improve the model for everyone" is off. In Cursor, the same data goes to Cursor (Anysphere)
and the company running your chosen model; Privacy Mode stops training.
ElevenLabs, WaveSpeed and Pexels get only what they need, and only if you turn them on.
Product feedback stays local until you agree to share it, with private values scrubbed.

On macOS, CI runs on macOS 15 with Apple silicon. The locked MLX speech packages
require macOS 14 or newer; earlier macOS versions are unverified. Apple silicon
uses Parakeet MLX by default. Intel Macs use Parakeet ONNX on the CPU and are not
covered by the macOS CI job. Xcode command line tools are optional for ordinary
editing and required for Vision person cutouts and face detection.

Linux is unverified in version 1. Contributions are welcome.

## From source

Clone this repository or download its source archive, open a terminal in the
repository root, and install the system requirements above. On a Mac with Homebrew,
`brew install uv ffmpeg node@22` installs the command-line dependencies. Yap Studio
finds Node 22 or newer on PATH or in Homebrew's node folders. Then run:

```console
uv tool install '.[asr,mcp]'
yap-studio renderer install
yap-studio speech download
yap-studio doctor --edit
```

If `yap-studio` is not found, add the directory printed by `uv tool dir --bin` to
your PATH and open a new terminal. `renderer install` prepares the pinned HyperFrames
renderer and its browser; `speech download` caches the selected Parakeet model.
Both downloads can take several minutes.

### ElevenLabs for music and sound effects

Generated music and the sound-effect kit use your own [ElevenLabs](https://littlemight.com/recommend/elevenlabs) account
(affiliate link).
For generated music (and the sound kit), install the optional ElevenLabs CLI:

```console
npx -y @mvanhorn/printing-press-library install elevenlabs --cli-only
```

Create your own ElevenLabs API key with Music, Sound Effects and User read access,
then save it with `elevenlabs-pp-cli auth set-token <your-key>` (replace the
placeholder with your key). Yap Studio never ships or reads a key. Both
`yap-studio doctor` and `yap-studio music` print the signup link when setup is
needed; it is an affiliate link.

### Pexels for stock clips

Stock clips use the optional Pexels CLI:

```console
npx -y @mvanhorn/printing-press-library install pexels --cli-only
```

Create a free Pexels account and request your key at https://www.pexels.com/api/.
It arrives at once. Save it with `pexels-pp-cli auth set-token <your-key>`
(replace the placeholder with your key). Yap Studio never ships or reads a key.
Turn on `broll.stock_clips` (`"ask"`) in your style to let the agent search and
ask you to choose clips. The free key allows 200 requests an hour and 20,000 a
month. `yap-studio doctor` shows whether it is set up.

### WaveSpeed for generated pictures

Install the optional WaveSpeed CLI:

```console
npx -y @mvanhorn/printing-press-library install wavespeed --cli-only
```

Create your own key at https://wavespeed.ai/accesskey and save it with
`wavespeed-pp-cli auth set-token <your-key>`.
[Sign up for WaveSpeed](https://littlemight.com/recommends/wavespeed) (affiliate link).
Yap Studio never ships or reads a key. Turn on `broll.generated_pictures` (`"ask"`)
in your style. Each picture is paid on your own account after you approve its cost.
The agent asks for each video before it drafts any picture.
`yap-studio doctor` shows whether it is set up.

### Check your setup with doctor

`doctor` checks ffmpeg, ffprobe, the filters verify needs, the workspace folder, free disk space,
the speech model, cutout helper tools and the phone link. It also lists the speech engines it
found and warns when a workspace version differs from the running engine. Keep workspaces out of
synced folders:
sync services can evict video files while an edit is running. Setup needs at least 10 GB free.
Yap Studio needs ffmpeg 6.1 or newer. `doctor` reports an older ffmpeg or ffprobe as a problem.
Plain `doctor` passes when verify can run. `doctor --edit` also fails when Node, the renderer
and its browser, a speech engine, a local folder or enough disk space is missing. A missing speech
model or cutout helper is a warning; the first ingest downloads the model, and cutouts need Xcode
command line tools on macOS. Run `yap-studio speech download` to fetch the model ahead of time.
Both doctor modes work with `--json`, which reports `ready_to_edit`, `edit_missing` and `phone`.
`yap-studio` is the canonical command.

## Downloadable workspace

From [Releases](https://github.com/cathrynlavery/yap-studio/releases), download
`yap-studio-workspace-<version>.zip` when a release provides it. Unzip it into a
local folder and open a terminal there. The zip contains setup scripts, agent edit
instructions, an inbox, a font folder and the matching engine wheel. The repository's
`workspace/` directory alone has no bundled wheel.
Install your chosen app and sign in before opening this folder in it.
On a Mac, setup uses Homebrew for ffmpeg and Node and offers to install Homebrew first.

```console
python3 setup.py --agent claude-code --json
python3 setup.py --agent claude-code
```

For Codex, use `--agent codex`; for Cursor, `--agent cursor`. The Claude app's Code tab
uses `--agent claude-code`. The agent passes its own identity during "set me up".
The choice is saved in `.yap-studio/setup.json`; older workspaces default to Claude Code.

The JSON mode lists steps without installing the engine. Read those steps first.
Text mode installs or replaces the bundled engine when needed, includes the speech
extra and downloads its model, then prints numbered steps for remaining dependencies.
Follow the required steps and rerun setup until it exits 0. Cutout tools and Tailscale
are optional; phone review works on the same Wi-Fi without Tailscale.
When setup finishes, it prints "Setup complete" and the home link after the page answers.
It prints a phone link too when one is available.
Setup's JSON includes `engine`, the full path of the installed command. If
`command -v yap-studio` prints nothing, use that path for commands in this session.
The required `engine_path` step runs `uv tool update-shell` by its absolute path.
After it runs, start a new session and rerun setup. In the Claude app, start a new
session; if the command is still missing, quit and reopen the app.

Put your own takes in `inbox/`. With an agent, open this folder in your chosen app and
ask it to edit the video; the workspace instructions guide story and final review.
For manual editing, run `python3 new_project.py --newest 1` to copy an unused take
into a fresh project, then use the project directory it prints with the commands in
[Edit from the command line](editing.md). The inbox originals stay intact.

- [Claude Code in Terminal](https://code.claude.com/docs/en/quickstart): change to the
  unzipped folder, run `claude`, and sign in once with `/login`.
- [The Claude app](https://claude.com/download): click Code, choose Local, then Open folder
  and pick the workspace. Sign in on a paid plan.
- [Codex in the ChatGPT app](https://chatgpt.com/download): choose Codex, add a project,
  then Use an existing folder. Trust the folder once. Or install the
  [Terminal CLI](https://developers.openai.com/codex/quickstart), run `codex` in this folder,
  and choose Trust and continue.
- [Cursor](https://cursor.com/download): File > Open Folder, then Cmd+I. Use Agent mode
  on this Mac, not a cloud agent. Privacy Mode is suggested.

In your chosen app, type `set me up`. Chat apps (Claude Chat, Cowork, ChatGPT chat)
can't edit these videos.

In Codex and Cursor, press the page button, then tell the agent in chat. It uses short
waits to check again each turn. Claude Code keeps its background listener.
Codex reads the shipped sandbox config and rules after you trust the folder once.
Cursor's project allowlist runs engine commands outside its sandbox; it asks for the
editable `setup.py` and `new_project.py` helpers because automatic approval would also
approve any code the agent added to them. The Claude Code allowlist applies after trust;
setup still asks in chat before installs and paid work.

The workspace instructions live in `AGENTS.md`, which Codex reads directly.
`CLAUDE.md` imports the same file with `@AGENTS.md` for Claude Code.
The zip contains regular copies of the two editing skills under `.claude/skills/` and
`.agents/skills/`, built from the single source in `workspace/.claude/skills/`.
Claude Code keeps `/edit`; Codex uses the `$edit` skill (or plain "edit this video").
Codex project skills replace custom prompts, which are deprecated and only load from
the user's Codex home. See [Codex skills](https://developers.openai.com/codex/skills)
and [custom prompts](https://developers.openai.com/codex/custom-prompts).

### Update

Download the new workspace zip from [Releases](https://github.com/cathrynlavery/yap-studio/releases)
and unzip it into a new local folder. Copy your `projects/`, `inbox/`, `brand/` and `style/`
folders from the old workspace into it. Open the new folder in your chosen app and say "set me up"
to install the matching engine. Keep the old folder until you have checked your projects in
the new one.

### Uninstall

Stop the home, style and review pages, then remove the tool:

```console
uv tool uninstall yap-studio
```

On macOS, delete `~/Library/Caches/yap-studio/` to remove the renderer and other Yap Studio
caches. Speech models use the Hugging Face cache at `~/.cache/huggingface/hub/`; remove only
the model folders downloaded for Yap Studio if other tools share that cache.
The default model folder is `models--mlx-community--parakeet-tdt-0.6b-v3/` on Apple silicon
or `models--istupakov--parakeet-tdt-0.6b-v3-onnx/` on Intel.
Delete `~/Library/Application Support/yap-studio/` to remove saved styles, taste rules and
review data. If you set `YAP_STUDIO_CACHE`, `YAP_STUDIO_CONFIG`, `YAP_STUDIO_DATA_DIR`,
`HF_HUB_CACHE` or `HF_HOME`, use those locations instead.

Your projects, footage, brand files and workspace style stay in the workspace until you
delete that folder. Keep anything you want before deleting it.

### Build the zip yourself

To build the zip yourself, run this from a Git checkout of the repository root:

```console
uv run python scripts/build_workspace.py
```

It writes `dist/yap-studio-workspace-<version>.zip`. The builder reads tracked files as
they are staged in the Git index, so unstaged edits stay out, builds a wheel from them, and
bundles it with the workspace files, `LICENSE`, `NOTICE` and `LICENSES/`. Untracked files
are not included.

## Versions and updates

Run `yap-studio --version` to see the installed version. A source install also shows its
commit. `doctor` warns when the installed tool is older than the checkout.

After updating your checkout, upgrade the installed tool from that directory:

```console
uv tool install --force '.[asr,mcp]'
```

Release changes are listed in [CHANGELOG.md](../CHANGELOG.md), newest first.

## Use it from Claude Code, Codex or Claude Desktop

These plugins add the `verify` check as a tool. The creator workspace above is what an agent
uses to edit.

### Claude Code

Install the plugin from this repository's marketplace:

```console
/plugin marketplace add cathrynlavery/yap-studio
/plugin install yap-studio@yap-studio
```

Or add only the MCP server after installing the CLI:

```console
claude mcp add yap-studio -- yap-studio mcp
```

### Codex

Add a local checkout or repository as a marketplace, then install the plugin:

```console
codex plugin marketplace add <path-or-repo>
codex plugin add yap-studio@yap-studio
```

### Claude Desktop

Build the bundle and open the resulting `.mcpb` file:

```console
uv run python scripts/build_mcpb.py
open dist/yap-studio-*.mcpb
```
