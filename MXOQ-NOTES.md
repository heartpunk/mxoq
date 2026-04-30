# mxoq

prototype slug. holds the imark/instruction-LTS viewer + accompanying
gist-content (small-core .py syncs, LTS extraction artifacts, dot/svg
visualizations) for the symbolic-execution / spec-recovery research.

## what this repo serves

GitHub Pages deploy at `heartpunk.github.io/mxoq/`. The viewer is
imark.html — multi-granularity (block + instruction) LTS view with
source mapping, range selection, live compose_subs, and (with
coi-serviceworker) browser-side z3 simplify.

coi-serviceworker.min.js injects COOP/COEP headers via service
worker so SharedArrayBuffer is enabled — that lets z3-solver's
WASM threading work in-browser, no backend required.

## source of truth

Most files here are also in
`/Users/heartpunk/code/learnability-private/tech-update-gist/` and
mirrored to gist `ddbae59d842784f30eaa1f0202531657`. learnability-
private remains the canonical source for the small-core .py syncs
and the regen pipeline; mxoq mirrors the gist content + adds
coi-serviceworker for in-browser z3.
