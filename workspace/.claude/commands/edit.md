---
name: edit
description: Edit unused inbox takes in a fresh Yap Studio project
---

Use `$ARGUMENTS` as the creator's request in Claude Code; in Codex, use the message
that invokes `$edit`.

Read `AGENTS.md` and start with its What are you making? section. If the type is voiceover,
the take is an audio memo, or the creator asks for a voiceover over a Photos album or media
folder, follow `.claude/skills/voiceover-video/SKILL.md`; it passes memo names to `new_project.py`
the same way. Otherwise follow `.claude/skills/edit-video/SKILL.md`. Start with
`python3 new_project.py` from the workspace root; pass take names if the creator names them.
Use the creator's request as the edit direction. After reading the transcript, name the project
with `python3 new_project.py --name DIR TOPIC`. Use `yap-studio run` for the final. B-roll comes
only from the video's B-roll sources. If the creator names a media folder, use that folder for
B-roll and backgrounds. Keep the inbox takes unchanged and stop for cut and final reviews.
