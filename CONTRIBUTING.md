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
shards. Render, speech and helper suites run on demand from the Release checks
workflow. Before merge, all required checks must pass and all review comments
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
yap-studio renderer install
YAP_STUDIO_RENDER_TESTS=1 uv run pytest -q -p no:cacheprovider -m render -n 4
```

For speech-engine changes, install the `asr` extra, cache the model with
`yap-studio speech download`, and run the speech tests with
`YAP_STUDIO_ASR_TESTS=1 uv run pytest -q -p no:cacheprovider -m asr -n 4`.
Vision helper tests require macOS and Xcode command line tools. Tests that need
unavailable dependencies may skip; include those skips when reporting results.

Add one bullet under `## Unreleased` in CHANGELOG.md. Don't change the version;
it is set once, at release.

Tests use synthetic speech and generated pictures. Keep fixtures synthetic and
do not commit real creator footage, transcripts or private taste files.

In the pull request, say what changed, how you checked it, and what remains.
Contributions use the project's Apache-2.0 license. Follow the
[code of conduct](CODE_OF_CONDUCT.md) and report vulnerabilities through the
[security policy](SECURITY.md).
