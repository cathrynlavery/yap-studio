# Change notes

Add one file per change so open pull requests don't conflict over CHANGELOG.md.
Name it `<short-slug>.<kind>.md`, for example `caption-pauses.fix.md`.
Use lowercase letters, digits and single hyphens for the slug. The kind must be
feature, fix, performance, dependencies, docs or tests.

Write one or more one-line `- ` bullets like CHANGELOG notes. Blank lines are allowed.
For example:

    - Keep captions visible during short pauses.

The release bump moves these bullets into the dated CHANGELOG section under their
kind and deletes the files. CI fails on a misnamed file or one with no bullet.

The HyperFrames bump workflow owns `hyperframes.dependencies.md` and rewrites it
for each new pin.
