# Feedback

When an agent struggles with Yap Studio, it records a short product note. Every note
is written locally first in `feedback.jsonl` in the Yap Studio data folder, with file
mode 600 so only your account can read or write it. On macOS that folder is
`~/Library/Application Support/yap-studio/`; `YAP_STUDIO_DATA_DIR` can change it.
Video review notes still use the existing `feedback build`, `apply`, `check` and reply commands.

## What is sent

A report contains the note (up to 500 characters), UTC time, Yap Studio version,
macOS version, CPU architecture, optional command or stage, optional error code,
kind (`bug`, `confusing`, `slow` or `missing`), agent ID from `AGENT_ID` when set,
and a fingerprint that groups repeats. Endpoint reports include each entry's local
sent flag and a count. GitHub issues include the latest entry and count; a repeated
issue gets a comment with the version and count.

The command scrubs paths, file names, emails, URLs, IP addresses, local hostnames
and common secret patterns on write and again before sharing. Links to github.com
and yapstudio.app retain only their domain. Automatic failure notes contain only
the command name and error code, never the error message.
No footage, transcripts, spoken words, names or project files are attached or read.
Write about the tool only: do not put names or the creator's words in a note.
The scrubber recognizes patterns; read free text before agreeing to share it.

```console
yap-studio feedback "The render needed a retry" --step run --kind bug
yap-studio feedback --stdin
yap-studio feedback list
yap-studio feedback list --pending --json
```

## Choose when to share

The setting lives in Yap Studio's `config.json`, under `feedback_mode`.
`YAP_STUDIO_CONFIG` can change the config folder. Default is `ask`.

- `ask`: nothing sends automatically. Show the person the dry run, then use `--yes`
  only if they agree.
- `always`: send after every local note, including automatic failure notes.
- `never`: keep notes local and refuse sends, even with `--yes`.

```console
yap-studio feedback setting ask
yap-studio feedback send --dry-run
yap-studio feedback send --yes
yap-studio feedback setting always
```

`YAP_STUDIO_FEEDBACK_AUTO_SEND=1`, `true` or `yes` forces `always`, including over
a saved `never`. Unset it to use the saved setting. `yap-studio doctor --json`
reports the effective mode and whether an endpoint or logged-in `gh` is available.

## Where reports go

With `YAP_STUDIO_FEEDBACK_ENDPOINT` set, pending entries are grouped by fingerprint
and POSTed as JSON to that address with a 15 second timeout. Otherwise a logged-in
`gh` creates issues labelled `feedback` on cathrynlavery/yap-studio, or comments on
an open issue with the same fingerprint. If `gh` is missing or logged out, the
command prints a prefilled issue link for you to open. Those notes stay pending.

`send --dry-run` prints the exact JSON, issue body or comment that would be posted
and changes nothing. `list` shows every entry after the same scrub used for sending.
Successful sends mark the group sent; those entries are not sent again. A failed
send leaves entries pending. An explicit note warns on stderr if its automatic send
fails; automatic failure capture preserves the original command's output and exit code.
`feedback "TEXT" --send --yes` records first, then requests a send under the same consent rules.

## Turn sharing off

```console
yap-studio feedback setting never
```

Unset `YAP_STUDIO_FEEDBACK_AUTO_SEND` too if you set it. Local notes continue to be
recorded. You can read them with `feedback list`; no sharing happens in `never` mode.
