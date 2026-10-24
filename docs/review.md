# Review a render with timed comments

The review page is where you watch a render, on your computer or your phone, and leave
comments pinned to a frame or a range. In the workspace, the agent opens it for you and answers
each comment in the next version.

## Open a review

```console
yap-studio review final-v3.mp4
```

Your browser opens a page on 127.0.0.1 with the video, a scrubber, and a frame readout.
The link has a token for this run, and nothing leaves your machine. The command runs until
Ctrl-C. Your browser must be able to play the file; H.264 MP4 plays in every current browser.

The video map sits below the scrubber on the same time axis. Its lanes show the saved
version's story pieces and joins, caption phrases, titles and Text hook, cards, B-roll and
backgrounds, Moments, music and sounds. On a computer, empty lanes are hidden except
Story and Captions. A story cut shows only Story and Captions. Tap a block to jump to its
start and open a comment with its target selected and label quoted. An open draft stays
as it is. Blocks cannot be dragged or edited. On a phone,
tap the one-line Video map summary to open or close the lanes.

Uncertain joins appear under Flagged by the checks as “Listen to this join”; tap Listen to hear about 1.5 seconds on each side.
Joins whose source edges no longer meet in the reviewed version are left out.

## Several videos

Pass several files or a directory to review them on one server:

```console
yap-studio review first.mp4 second.mov
yap-studio review renders/ --no-open --json
```

A directory expands to `.mp4`, `.mov`, `.m4v`, `.mkv` and `.webm` files directly inside it,
in name order, ignoring case on the extension. Files and expanded directories keep their
argument order. Duplicate contents are refused, even under different names. With several
videos, the command opens and prints a private list link with names and durations, then
prints each video's link. Each video has its own token and comments; sharing one video's
link grants access to that video. The list link grants access to all videos in this run.
One video keeps its existing page and printed output.

## Versions

Use `--versions` to put successive renders under one link, oldest first, with a version
picker, a side by side comparison, and replies to earlier comments. In a project, save
each verified cut with `versions save`, then review the folder; saved word occurrences
match story moments even when their playback times change.

```console
yap-studio review --versions first.mp4 second.mp4
yap-studio versions save my-video --note "Ready for review"
yap-studio review --versions my-video
```

## Text hook

When three ideas are ready, a Text hook panel appears above the comments. Tap an
idea or write your own, then press Use this. The panel shows your chosen text and
adds a note. Send notes passes it to your agent. Text that needs more than two balanced
lines is refused. Nothing is rendered until you choose.

## Keep it running

Use `--detach` to keep the page serving after the command returns. A new detached review
of the same videos replaces the earlier one. Stop all background reviews with `--stop`,
or name a video or project folder to stop only its review:

```console
yap-studio review --versions my-video --phone --detach
yap-studio review --stop
yap-studio review --stop my-video
```

`--json` prints one JSON object on one line before serving, with `list` and `videos` keys.
`list` is the list URL, or `null` for one video; each video has `name`, `sha256` and `url`.

## On your phone

To review on your phone, add `--phone`. It prints one link:

```console
yap-studio review --versions my-video --phone
```

Phone links are labeled "Encrypted (Tailscale)" for HTTPS or "Not encrypted (same Wi-Fi)" for plain HTTP.

- When Tailscale is connected on this computer, the link is a tailnet-only HTTPS address.
  Yap Studio finds or starts a `tailscale serve` mapping to the review port. It opens on
  any device signed in to your tailnet, wherever it is. The mapping is removed when the
  review stops; one that was already there is left alone.
- Otherwise the server listens on this computer's Wi-Fi address and the link carries a
  private token. It opens only on the same network. Anyone on that network who has the link
  can open the review, so share it only with people you trust. The Wi-Fi link uses plain HTTP,
  so someone watching that network could see the link and the review; Tailscale links are encrypted.
- If the Tailscale mapping cannot be made, it says why and uses the Wi-Fi link instead.
- With no Wi-Fi and no Tailscale, it exits 2 and says how to fix it.

`--phone` cannot be combined with `--public-base`. With `--json`, a `phone` key holds `via`
(`tailscale` or `wifi`) and `url`.

## Behind your own proxy

For another setup, run a trusted reverse proxy on the same machine and forward it to
the review server's local port. Use `--public-base https://box.example.net:8443` to print and
open the token link on that origin and allow its Host header; the URL must have no path,
query or fragment. Add other trusted hosts with repeatable `--allow-host NAME[:PORT]`:
`NAME` allows that exact name with any port or no port, while `NAME:PORT` allows only that
port. Names match without regard to case. The server still binds only to 127.0.0.1, and
the token is still required for media and comments. All printed links, including the list
and `--json` links, use the `--public-base` origin.

## Keys and comments

| Key | Action |
| --- | --- |
| `C` | Pause and comment at the frame on screen, or on the marked range. |
| `I` / `O` | Mark the range start and end. |
| `X` | Clear the range. |
| Left / Right | Step one frame. |
| Space | Play or pause. |
| Enter / Shift-Enter | Save a comment or add a line in the comment box. |
| Option-L | Switch between Fix and Keep in the comment box. |
| Option-I / Option-O | Move the range edges in the comment box. |
| Esc | Cancel a comment. |

You can also Shift-drag on the scrubber to mark a range, then drag its edges to adjust it.
With a range marked, `C` comments on the whole range. Comments default to Fix. Pick Fix or
Keep, then optionally tap Story, Speech, Caption, Graphic or Whole video in About. Tap the
selected target again to clear it. A keep protects only its target over the comment's range.
A keep with no target protects the story: the same words in the same order.
Change the kind or target from the comment row. Click a comment to go to its frame; Play range
stops on the last frame. Edit or delete comments in the list.

Pause and tap a caption or choose Fix caption, then choose a word, type the right text and Save.
Use Every time for a repeated heard word, Undo caption fix to remove a draft change, and Copy notes or Send notes to pass it on.

Tap the playing picture to pause it. Tap elsewhere on the paused picture to drop a pin there and open
the comment box, ready to type. Comment opens a note without a pin. With the box open, tap
the picture or drag the pin to move it, or tap Remove pin. The comment row says where the pin
is, and selecting the note shows the pin on its saved frame.

On a phone or without a keyboard, tap Mark start and Mark end at the playhead, then drag
the range edges. The range shows its length and the spoken words inside it. Play range
plays just that part. Cut this saves one Fix note about the story with a cut request:
the range and its words. Text in the open comment box goes with it as a note. Cut this
needs a project review (`review --versions` on a project folder), because the words come
from the project.

## Copy and export feedback

Pinned notes add `(pinned ...)` to the text export and a `pin` field to JSON.

The header has one Copy notes button that copies all draft and sent notes for the current version in one click:

```
Feedback on final-v3.mp4 (0:58.6, 29.97 fps)
From Alex Reviewer, 2026-09-29
4 comments: 3 fixes, 1 keep

0:00.0-0:03.2 [Fix] Cut this. Open on "here's the thing".
0:14.9 [Fix: caption] Caption covers my face here.
       Move it up or drop it for this line.
0:32.2 [Keep: graphic] This zoom lands. Keep it.
0:44.1-0:51.3 [Fix] Music is louder than my voice through this bit.
```

Earlier versions appear as one summary line.

Times use the video's timeline and round down to a tenth. The JSON export API returns version 2
with each comment's kind and optional target. Old review files still load: `like` becomes
`keep` and `change` becomes `fix`, with no target.
Comments save as you go and match the file by content, so renaming or moving it keeps them.
If the file changes, the page shows how many comments belong to the earlier version and lets
you view or copy them. Nothing is written next to the video.

## Where reviews are stored

Reviews are stored in `~/Library/Application Support/yap-studio` on macOS,
`%APPDATA%\yap-studio` on Windows, and `~/.local/share/yap-studio` elsewhere
(or `$XDG_DATA_HOME/yap-studio`). `YAP_STUDIO_DATA_DIR` overrides this location.

## Other options

- `--port N` sets the local port.
- `--no-open` leaves the page for you to open yourself.
- `--reviewer "Name"` adds a From line to the export and remembers the name for later runs.
