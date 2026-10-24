# Yap Studio home

Setup prints "Setup complete" and the home link after the page answers. It also prints a phone
link when one is available.

Yap Studio home opens in the browser on the Mac and walks through one video at a time:

1. What you're making: talking head or voiceover, then Vertical (Shorts, Reels, TikTok) or Horizontal (YouTube) for this video. Vertical starts selected; readable take dimensions suggest a matching shape on the footage step, and you can change it. For vertical, a talking head asks whether you have anything to show on top before asking where it comes from; a voiceover asks where your photos and clips are.
   Horizontal captions start off with one switch to add them; more looks, voiceover, screen effects, titles, cards and B-roll are coming soon.
2. Your footage: drag takes in from Finder or choose them from the file picker. Yap Studio
   copies them into the workspace `inbox/`; the originals stay where they were.
3. Your look: a few questions about your videos the first time, then the style page on a
   still from your own take.
4. Make it: Text at the top shows your saved answer. Override it for this video with
   None, Just the opening, or The whole video. Your saved look stays the same.
   For either kind, type your own text or leave it empty for three ideas in the
   Story cut note and review page. Your agent waits for your pick. The whole-video label
   must fit at most two balanced lines. The page sends `top_text` and any `text_hook`
   text with the request. Your look summary shows the saved answer too.
   The page shows each step as the work happens, read from the
   project's receipts, and says when it is your turn.
5. Watch and comment: the review page opens from here. Mark a stretch and say Cut this, Change this
   or Keep this, with a note. Notes you send go to your agent, and the next version shows up on the same
   page with your agent's notes on what changed. When the story is right, press Make the final video.
   Approve the final to open it, show it in Finder or download it.

Each video also has a Talk to Claude Code, Talk to Codex or Talk to Cursor box, named for the app you set up.
A message goes to your agent the same way a button does, and its reply shows under it.

Home and review use the app recorded during setup: Claude Code (including the Claude app's
Code tab), Codex or Cursor. Older workspaces use Claude Code.
Setup records `--agent claude-code|codex|cursor` in `.yap-studio/setup.json`;
rerunning setup without the option preserves that choice.

Your video files stay on your computer. Still frames from pinned notes and contact sheets of
your clips go to Anthropic in Claude Code and the Claude app's Code tab, or OpenAI in Codex,
along with the transcript, your notes, the project files your agent reads and command output,
as part of your agent session. On personal ChatGPT plans, OpenAI may train on it unless
"Improve the model for everyone" is off. In Cursor, the same data goes to Cursor (Anysphere)
and the company running your chosen model; Privacy Mode stops training.

The page never edits, renders, downloads stock clips or spends money. Those stay with the agent,
which asks before anything paid.

## Open it

In the workspace, the agent opens it after setup. To open it yourself:

```console
yap-studio home --workspace .
yap-studio home --workspace . --phone --detach --json
yap-studio home --stop
```

The link carries a private token, like the style and review pages. `--phone` adds a link for a
phone on the same Wi-Fi, or anywhere on your tailnet when Tailscale is connected. `--detach`
keeps the page serving after the command returns; running it again for the same workspace returns
the same link.

Phone links are labeled "Encrypted (Tailscale)" for HTTPS or "Not encrypted (same Wi-Fi)" for plain HTTP.

## How a button reaches the agent

A button that needs the agent, such as Make my video, writes a small request file under
`.yap-studio/homepage/` in the workspace. With background notifications, the agent keeps one listener running:

```console
yap-studio home wait --workspace . --json
```

It returns as soon as there is something to do and prints one JSON object:

| `event` | Meaning |
| --- | --- |
| `request` | The creator pressed a button. `request.kind` is `edit` (with the video type, B-roll sources, take names and an optional Photos album), `final` (the story cut is right; build the final), `message` (a note to your agent, answered with `home ack --message`) or `learn` (learn from finished videos). |
| `notes` | Notes were sent from the review page for `project`, as feedback batch `batch`. |
| `approved` | `version` of `project` was approved. |
| `timeout` | Nothing happened before `--timeout`; start it again. |

For an edit, `payload.shape` reaches the agent unchanged: horizontal starts with
`python3 new_project.py TAKE ... --type talking_head --shape horizontal`. The helper copies and
records the takes and sets up the plain horizontal style. When `payload.captions` is not true,
after the cut the agent adds one `caption_gap` covering the whole video on the agent lane using
`yap-studio timeline add FRAGMENT DIR --lane agent`; see
[Caption gaps](../workspace/AGENTS.md#caption-gaps). Vertical follows the existing recipe.

The agent reports back with `yap-studio home ack ID --state claimed|done|failed`, adding
`--project DIR` once the project folder exists and `--message` with one plain sentence. The page
says your app's name followed by "has it" only after that.
The video's link follows a project folder rename after the agent reports the new name.
If the agent's session stops for ten minutes in the middle of a request, the page offers Send
again and Cancel. A request sent again arrives with `resumed` set: the agent opens the project
from the earlier ack, checks receipts and `yap-studio status`, claims it and carries on without
redoing finished or paid steps.

While a listener is running, the page shows your app's name followed by "is listening".
In Codex and Cursor, tell the agent in chat after pressing a page button so it can check
again with `--timeout 1` on each turn. Claude Code keeps its background listener.
When none is, it says so and shows the one word to type in that app (`continue`), so the agent picks up where things are.
Requests wait until then; nothing is lost.

## What stays with the agent

- Installing (in chat, one step at a time, asking first).
- Picking the story, cutting, captions, cards, B-roll, music and rendering.
- Learning from finished videos and a voiceover's media folder, which need a folder on the Mac.
- Anything paid, such as generated music. The agent asks in chat first.
