# Contributing

Yap Studio is a Python project managed with uv. Use Python 3.11 or newer,
ffmpeg/ffprobe 6.1 or newer, and Node 22 or newer with npm. Install `espeak-ng`
for synthetic speech tests. See [Install](docs/install.md#requirements) for platform
requirements and [How it works](docs/how-it-works.md) for the project layout.

```console
uv sync --locked --extra asr --extra mcp
uv run yap-studio doctor
```

Scripts in `scripts/` need the project's environment. Run them with `uv run python scripts/<name>.py`.

PRs and main run lint, JavaScript tests, builds and the Python suite in ten macOS
shards. Release checks runs render, speech and helpers when matching paths changed
since each suite last passed on main; the nightly 07:17 UTC run checks all three.
To force all suites, dispatch Release checks with render, speech and helpers set to `full`.
Before merge, all required checks must pass and all review comments
must be resolved. A release tag needs green CI and Release checks on the tagged
commit, plus a clean workspace install that serves its home page.

Before opening a pull request, run the local gates:

```console
uv run ruff check src tests
uv run pytest -q -p no:cacheprovider -n 4 -m "not render"
node --test tests/js/*.mjs
```

If you change rendering, captions, graphics, timeline or assembly, also run:

```console
uv run yap-studio renderer install
YAP_STUDIO_RENDER_TESTS=1 uv run pytest -q -p no:cacheprovider -m render -n 4
```

For speech-engine changes, install the `asr` extra, cache the model with
`uv run yap-studio speech download`, and run the speech tests with
`YAP_STUDIO_ASR_TESTS=1 uv run pytest -q -p no:cacheprovider -m asr -n 4`.
Vision helper tests require macOS and Xcode command line tools. Tests that need
unavailable dependencies may skip; include those skips when reporting results.

Add a file named `changes/<short-slug>.<kind>.md`. Use lowercase letters, digits
and single hyphens for the slug. Choose feature, fix, performance, dependencies,
docs or tests for the kind. Write one or more one-line `- ` bullets. See
[change notes](changes/README.md) for an example. Don't edit CHANGELOG.md or the
version; they are set at release.

Releases are small and frequent, cut from a `release/vX.Y.Z` pull request whenever
main is green:

1. Branch `release/vX.Y.Z` from main once CI and Release checks have passed on
   main's latest commit.
2. Run `uv run python scripts/bump_version.py X.Y.Z`. It sets the version in
   pyproject.toml, uv.lock and the plugin and MCPB manifests. It moves the notes in
   `changes/` into the dated section under Features, Fixes, Performance,
   Dependencies, Docs and Tests and deletes those files. Notes still under
   Unreleased go first, as written.
3. In that section, add a one-line summary, sort any notes from Unreleased into
   their headings, and tidy. The Release workflow publishes this section as the
   release notes.
4. Merge the pull request once its checks pass and its review comments are resolved.
5. Run Release checks on the merge commit (`gh workflow run release-checks.yml
   --ref main`) or wait for the nightly run. When CI and Release checks pass on
   it, tag that commit and push the tag: `git tag vX.Y.Z <commit>` then
   `git push origin vX.Y.Z`.
6. The Release workflow builds the wheel, source archive, MCPB bundle and
   workspace zip, installs the zip in a clean folder and creates a draft release.
   Check its assets and notes, then publish it.

Use a patch version for fixes and small changes, a minor version when a release
adds features, and a major version when it breaks existing projects or styles.

Tests use synthetic speech and generated pictures. Keep fixtures synthetic and
do not commit real creator footage, transcripts or private taste files.

In the pull request, say what changed, how you checked it, and what remains.
Contributions use the project's Apache-2.0 license. Follow the
[code of conduct](CODE_OF_CONDUCT.md) and report vulnerabilities through the
[security policy](SECURITY.md).
