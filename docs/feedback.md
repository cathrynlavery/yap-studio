# Feedback

When an agent struggles with Yap Studio, it records a short product note. When the
creator asks for something Yap Studio cannot do and the agent does it by hand, the
agent records a `missing` note that names the capability, not the video. Every note
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
and common secret patterns on write and again before sharing. Yap Studio's own fixed
file names, such as `edit.json`, and the abbreviations e.g. and i.e. stay readable.
Links to github.com and yapstudio.app retain only their domain. Automatic failure
notes contain the command name, registered subcommand when present (such as
`versions save`), the error code and where in Yap Studio's code it was raised
(module, function and line). Wrapped errors also name the original exception's class.
Other arguments and the error message are never included.
No footage, transcripts, spoken words, names or project files are attached or read.
Write about the tool only: do not put names or the creator's words in a note.
The scrubber recognizes patterns; read free text before agreeing to share it.

```console
yap-studio feedback "The render needed a retry" --step run --kind bug
yap-studio feedback "No way to blur part of the picture" --step assemble --kind missing
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
`gh` creates issues on cathrynlavery/yap-studio with the `feedback` label when
permissions allow, or comments on the issue with the same fingerprint, open or
closed, even if it has no label. An open issue comes first, locked issues are
skipped, and reports sent before code locations were added still match. If `gh` is
missing or logged out, the command prints a prefilled issue link for you to open.
Those notes stay pending.

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

## Release notices

Home and `yap-studio doctor` check GitHub for the newest Yap Studio release at most
once every 24 hours, sharing one local cache. Home checks in the background and never
waits for GitHub. A newer release shows `Yap Studio X is out. Say "update Yap Studio".`
with a link to its release notes. Failed or slow checks stay quiet and wait another
24 hours before trying again. The notice never changes doctor's exit code.

The check sends a GET request to
`https://api.github.com/repos/cathrynlavery/yap-studio/releases/latest` with the plain
User-Agent `Yap-Studio`, no auth token and nothing about the creator's work. GitHub
receives the normal connection information, including your IP address. No footage,
transcripts, project files, names or machine details are included in the request.
The latest tag and attempt time stay in `release-notice.json` in the Yap Studio
cache folder (`~/Library/Caches/yap-studio/` on macOS; `YAP_STUDIO_CACHE` can change it).

Run `yap-studio update-notice off` to save `"update_notice": false` in `config.json`;
`yap-studio update-notice on` enables it again. Setting `YAP_STUDIO_NO_UPDATE_NOTICE=1`
also turns it off, overriding the saved choice. Either switch stops requests and
hides any cached notice. Release notices are enabled by default and are independent
of the feedback sharing preference.
