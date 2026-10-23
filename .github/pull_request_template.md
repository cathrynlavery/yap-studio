Related issue (if any):

## What this does

Describe the behavior and the files it changes.

## Acceptance

State how you checked the behavior, including a synthetic example when useful.

## Gates

- `uv run ruff check src tests`:
- `uv run pytest -q -p no:cacheprovider -n 4 -m "not render"`:
- `node --test tests/js/*.mjs`:
- Render suite, if applicable:

## What is left

List follow-up work or write "None."
