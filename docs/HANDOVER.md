# Handover — TEMPORARY, delete when consumed

Written 2026-07-30, mid-session update after §16.28 landed. **This file is scaffolding, not a
record.** Delete it once #12/#13 (the F1 recording run) is fully done; anything in it worth keeping
permanently belongs in `PIPELINE_SPEC_V1.md`, `COOKBOOK.md`, or `RULINGS.md` instead.

## What changed since this file was last written

The prior version of this file said §16.27 item 1(g)'s frozen-literal authorization was
"deliberately withheld" and cited the supersession gate at `EXPECTED_PARSED_CLAIMS = 12`. Both are
now stale:

- **Step 0 is DONE.** The joint architect-plus-engineer pass ran, the two disagreed on one
  mechanism (how to gate a real recorded artifact silently dropping its `derivation` field —
  signed-per-pair vs. coverage-channel), and `fable-adjudicator` tie-broke it. The ruling is
  transcribed as `docs/PIPELINE_SPEC_V1.md` §16.28 (which **SUPERSEDES §16.27 item 1(g)** — read
  §16.28, not item 1(g), for the current authorized scope), with the full source record in
  `docs/RULINGS.md` under "§16.27 item 1(g) frozen-literal adaptation — Fable tie-break,
  2026-07-30". A fresh-reader step-1a review caught three defects in the first transcription
  attempt (all fixed before commit). Committed as `0c05882`.
- **The supersession gate is now `EXPECTED_PARSED_CLAIMS = 13`**, with
  `("16.28", "16.27 item 1(g)")` added to `RATIFIED_SUPERSESSIONS` in
  `crates/pc-testkit/tests/spec_supersession.rs`. Do not cite 12 anywhere going forward.
- **Steps 1 and 2 (below) are also DONE and committed.**

## Repo state

- Branch `claude/codex-plugin-install-jxirxa`, `HEAD` = `0c05882`, working tree clean at time of
  writing, **nothing pushed** (pushing needs asking, per standing convention).
- Verification bar for `0c05882`, actually run: `cargo test --workspace` (839 passed, 0 failed),
  the onnx tier (`cargo test --workspace --all-targets --features pc-cli/onnx`, green), `cargo
  clippy --workspace --all-targets --all-features -- -D warnings` clean, `cargo fmt --all --check`
  clean, `cargo test -p pc-testkit --test spec_supersession` (12/12).
- `export PATH="$HOME/.cargo/bin:$PATH"` or cargo is not found (`COOKBOOK.md` rule 11).
- Git has no configured identity; commits used `GIT_AUTHOR_NAME=Claude
  GIT_AUTHOR_EMAIL=noreply@anthropic.com` (+ `GIT_COMMITTER_*`) as env vars.

## Steps 1 and 2 — DONE, but re-derive the ephemeral parts if picking this up fresh

**Step 1 — the oracle checkout.** Rebuilt this session per `COOKBOOK.md` rule 3's recipe, and the
`comictextdetector.pt.onnx` weights were downloaded and sha256-verified
(`1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f`) — **but both live in the
session-scoped scratchpad**, not the repo, so a fresh session will find them gone again (same as
every prior session). Nothing to redo in the repo; just re-run the rule-3 recipe and re-download
the weights if a real upstream run is needed again (e.g. for the later recording phase).

**Step 2 — the oracle candidate pages.** Vendored and committed as `4a7951e`:
`tests/fixtures/upstream/oracle_pages/ja_Pepper-and-Carrot_by-David-Revoy_E01P0{1,2,3}.jpg`, with an
`ATTRIBUTION.md` entry. P01's sha256 matches §16.24 item 17's recorded hash exactly; P02/P03 match
item 17(a)'s recorded sizes. This is in the repo now — nothing to redo.

## Then the actual work — #12/#13, now split into phases

§16.28 authorizes Phase 1: the Rust-side schema (`OracleBlock`'s five new fields, `ExpectedPair`'s
signed `derivation`), the comparator logic (the leg-1 gate, its structural guard, the
signed-derivation-mismatch check with bidirectional equality, the `None`-disables-machinery rule),
three new `Divergence` variants (ratified total: 19), the anti-vacuity counter, and the exact
authorized edits to the frozen `crates/pc-detect/tests/f1_oracle_comparator.rs`. This does **not**
include the Python recording script, the real-page comparison run, or the atomic recording commit —
those are later phases with their own planning pass, once Phase 1 lands and is reviewed.

**In progress at the time of writing:** the joint architect-plus-engineer planning pass for Phase 1.
The architect's plan has landed (a full design: `Derivation` enum, exact field placements, the three
new `Divergence` variants, comparator emission order, enumerated frozen-file edit sites, six new
test designs, and a 5-task Codex sequencing plan). It flags two points it expects to disagree with
the engineer on — flat vs. grouped Rust encoding for the `eng_expanded`/`lines_pre_expand`/
`expand_size` trio, and whether the new leg-1 mismatch variant carries `rect_yolo` as `Option<Rect>`
to stay within the ratified count of 19 — and says explicitly these are Fable questions if the two
disagree, not something to split unilaterally. Waiting on the rust-engineer's independent plan to
compare. If they disagree on either flagged point (or anything else material), convene
`fable-adjudicator` per `CLAUDE.md` rather than picking a side. If they agree or the differences are
reconcilable, synthesize a final plan, run the CLAUDE.md step-2 pre-implementation test review
against §16.27/§16.28, and dispatch to Codex via `codex:rescue` following the architect's task
sequencing (batch simple/related tasks in one call; heavy tasks — the schema+frozen-file edit, and
the comparator logic — each get their own call).

Then #14, then §16.24 item 5's ratified order: GPU-1 → GPU-2 → CPU-EP investigation → INI + Lab
NLM → LaMa → PSD → DBNet lines → Annotation.

## Open risks worth carrying

- **Two agents editing the same file concurrently is a proven-survivable pattern** (`COOKBOOK.md`
  rule 11), but sequence write-agents on the same file where the task allows it.
- **Stop-time review findings have been reliable this session** — this very file was flagged stale
  by one (2026-07-30: citing the withdrawn "12" count and the resolved item 1(g) deferral), and the
  finding was correct. Keep trusting file-content claims from that channel.
- **The still-owed list in `COOKBOOK.md` rule 11** (`provenance_is_current`'s redundant condition,
  the R1–R23 doc-comment numbering, `UpstreamBoxOutsideFrame`'s missing register entry) is unrelated
  to #12/#13 and can be picked up whenever convenient — not blocking.
- **§16.28's own deferrals, not yet resolved**: whether an *unmatched* upstream entry's signed
  `DbnetScattered` mechanism is verified against that block's recorded `derivation` inside
  `compare()` (a residual gap in both original designs); `UpstreamBoxOutsideFrame` stays
  untranscribed; the serde spelling of `Derivation`'s enum values is Phase 1's call but must be
  pinned in the full-shape JSON literal in the same commit that chooses it.
