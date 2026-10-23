# Yap Studio home

Setup prints "Setup complete" and the home link after the page answers. It also prints a phone
link when one is available.

Yap Studio home opens in the browser on the Mac and walks through one video at a time:

1. What you're making: talking head or voiceover. A talking head asks whether you have anything to
   show on top before asking where it comes from; a voiceover asks where your photos and clips are.
2. Your footage: drag takes in from Finder or choose them from the file picker. Yap Studio
   copies them into the workspace `inbox/`; the originals stay where they were.
3. Your look: a few questions about your videos the first time, then the style page on a
   still from your own take.
4. Make it: when your look uses Text hook, type an optional line here. Leave it empty
   and Claude will suggest three. Text must fit at most two balanced lines. The page
   sends your text with the video request. The page shows each step as the work happens, read from the
   project's receipts, and says when it is your turn.
5. Watch and comment: the review page opens from here. Mark a stretch and say Cut this, Change this
   or Keep this, with a note. Notes you send go to Claude, and the next version shows up on the same
   page with Claude's notes on what changed. When the story is right, press Make the final video.
   Approve the final to open it, show it in Finder or download it.

Each video also has a Talk to Claude box. A message goes to Claude the same way a button does, and
Claude's reply shows under it.

Your video files stay on your computer. Still frames from pinned notes and contact sheets of
your clips go to Claude (Anthropic), along with the transcript, your notes and the project files
Claude reads, as part of your Claude Code session.

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
`.yap-studio/homepage/` in the workspace. The agent keeps one listener running in the background:

```console
yap-studio home wait --workspace . --json
```

It returns as soon as there is something to do and prints one JSON object:

| `event` | Meaning |
| --- | --- |
| `request` | The creator pressed a button. `request.kind` is `edit` (with the video type, B-roll sources, take names and an optional Photos album), `final` (the story cut is right; build the final), `message` (a note to Claude, answered with `home ack --message`) or `learn` (learn from finished videos). |
| `notes` | Notes were sent from the review page for `project`, as feedback batch `batch`. |
| `approved` | `version` of `project` was approved. |
| `timeout` | Nothing happened before `--timeout`; start it again. |

The agent reports back with `yap-studio home ack ID --state claimed|done|failed`, adding
`--project DIR` once the project folder exists and `--message` with one plain sentence. The page
says "Claude has it" only after that.
The video's link follows a project folder rename after the agent reports the new name.
If the agent's session stops for ten minutes in the middle of a request, the page offers Send
again and Cancel. A request sent again arrives with `resumed` set: the agent opens the project
from the earlier ack, checks receipts and `yap-studio status`, claims it and carries on without
redoing finished or paid steps.

While a listener is running, the page shows "Claude is listening". When none is, it says so and
shows the one word to type in Claude Code (`continue`), so the agent picks up where things are.
Requests wait until then; nothing is lost.

## What stays with the agent

- Installing (in chat, one step at a time, asking first).
- Picking the story, cutting, captions, cards, B-roll, music and rendering.
- Learning from finished videos and a voiceover's media folder, which need a folder on the Mac.
- Anything paid, such as generated music. The agent asks in chat first.
