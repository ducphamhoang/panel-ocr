# Handover — TEMPORARY, delete when consumed

Written 2026-07-30, mid-session update after Phase 1 landed. **This file is scaffolding, not a
record.** Delete it once #12/#13 (the F1 recording run) is fully done; anything in it worth keeping
permanently belongs in `PIPELINE_SPEC_V1.md`, `COOKBOOK.md`, or `RULINGS.md` instead.

## What changed since this file was last written

- **§16.28 landed** (commit `0c05882`): resolves §16.27 item 1(g)'s deferral for the frozen-literal
  adaptation in `f1_oracle_comparator.rs`, via a Fable tie-break between the joint architect+engineer
  planning pass's two disagreeing designs. Full source record in `docs/RULINGS.md`; supersession
  gate is now `EXPECTED_PARSED_CLAIMS = 13`.
- **Phase 1 landed** (commit `f042749`): the actual Rust-side implementation §16.27/§16.28 authorize
  — the `Derivation` schema, the comparator's three new gating rows, the `DbnetScattered` mechanism,
  and the exact authorized edits to the frozen `f1_oracle_comparator.rs`, plus a new
  `f1_oracle_derivation.rs` acceptance-test file. A real bug (the structural guard not suppressing
  the pre-existing `GeometryIdentity`/`UnionMonotonicity` checks) was found and fixed mid-session,
  confirmed by mutation testing during an independent rust-engineer review. See that commit message
  for the full account.
- **Do not cite the old figures going forward**: not "12" for the supersession count, not "839
  passed" for the workspace test count (it's 847 now, 858 under the onnx tier), and item 1(g) is no
  longer the authoritative scope statement — §16.28 is.

## Repo state

- Branch `claude/codex-plugin-install-jxirxa`, `HEAD` = `f042749`, working tree clean at time of
  writing, **nothing pushed** (pushing needs asking, per standing convention).
- Verification bar for `f042749`, actually run: `cargo test --workspace` (847 passed, 0 failed), the
  onnx tier (`cargo test --workspace --all-targets --features pc-cli/onnx`, 858 passed, 0 failed),
  `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean, `cargo fmt --all
  --check` clean, `cargo test -p pc-testkit --test spec_supersession` (12/12).
- `export PATH="$HOME/.cargo/bin:$PATH"` or cargo is not found (`COOKBOOK.md` rule 11).
- Git has no configured identity; commits used `GIT_AUTHOR_NAME=Claude
  GIT_AUTHOR_EMAIL=noreply@anthropic.com` (+ `GIT_COMMITTER_*`) as env vars.
- **Codex CLI backend was hitting "Selected model is at capacity" errors repeatedly this session**,
  causing turns to fail mid-task and occasionally leaving a job in a genuinely stuck state (running
  status, no active process, stale log) rather than a clean failure. Check for a live `cargo`/`rustc`
  process and log-file mtime staleness before trusting a "running" status; cancel and re-dispatch
  fresh (not resumed) if it looks stuck — resuming a thread once also failed outright with a
  read-only-filesystem error, unrelated to capacity.

## Steps 1 and 2 (oracle checkout, oracle pages) — DONE, ephemeral parts need re-deriving

**Step 1 — the oracle checkout.** Rebuilt this session per `COOKBOOK.md` rule 3's recipe, and the
`comictextdetector.pt.onnx` weights were downloaded and sha256-verified
(`1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f`) — **both live in the
session-scoped scratchpad**, gone again in a fresh session. Re-run the recipe and re-download the
weights when Phase 2 needs a real upstream run (which it will).

**Step 2 — the oracle candidate pages.** Vendored and committed as `4a7951e`:
`tests/fixtures/upstream/oracle_pages/ja_Pepper-and-Carrot_by-David-Revoy_E01P0{1,2,3}.jpg`. This is
in the repo now — nothing to redo.

## Then the actual work — #12/#13 Phase 2

Phase 1 is done. What's left for the F1 recording run:

1. **The Python recording script** (`xtask/scripts/` has no oracle recorder yet — checked, `ls`
   returns `record_inter_area.py`, `record_nlm.py`, `verify_find_edges.py` only). It must instrument
   upstream's `group_output` (three module-global functions, per §16.27 item 1(f)'s method) to tag
   each block with `derivation`/`rect_yolo`/`eng_expanded`/`lines_pre_expand`/`expand_size`, run
   **pristine and instrumented** in parallel on deep-copied identical inputs with a hard-fail if they
   diverge (§16.27 item 1(b) — the binding condition on the recorder), and emit the serde spelling
   Phase 1 pinned (`yolo_unioned`, `yolo_synthesized_corners`, `yolo_split`, `dbnet_scattered`).
2. **The real-page comparison run** against P01 (the ratified oracle page) and, for the
   `DbnetScattered` census, P02/P03 too.
3. **The atomic recording commit** (§16.24 item 6's rule: the fixture, its provenance, the frozen
   `EXPECTED_*` edits, the real-page gate test, `docs/DETECTOR_ORACLE.md`, and the three §16.20 item
   3(f) signatures all land in one commit or none). This is where `leg1_rows_checked`'s real expected
   value gets hand-derived and authored (§16.28 Amendment 2 — no number is ratified yet; it must come
   from the actual signed pairing on the real page, never from the artifact or a re-run).
4. **§16.27 item 7's four additive reachable-shape controls** (synthesized-corners pair,
   dominating-lines pair, multi-line pair, split-shaped-unmatched structure) — separately authorized,
   may land alongside but don't gate Phase 2.

This needs its own joint architect+rust-engineer planning pass before implementation, same as
Phase 1 did — the Python recording script in particular is new-design work, not a mechanical
translation of an existing plan.

Then #14, then §16.24 item 5's ratified order: GPU-1 → GPU-2 → CPU-EP investigation → INI + Lab
NLM → LaMa → PSD → DBNet lines → Annotation.

## Open risks worth carrying

- **Two agents editing the same file concurrently is a proven-survivable pattern** (`COOKBOOK.md`
  rule 11), but sequence write-agents on the same file where the task allows it.
- **Stop-time review findings have been reliable this session, including catching a real
  implementation bug mid-flight** (the guard-suppression defect above) — keep trusting that channel,
  but note it sometimes fires on legitimate in-progress/mid-turn state, not just genuine defects;
  read the finding against the actual current code before reacting.
- **The still-owed list in `COOKBOOK.md` rule 11** (`provenance_is_current`'s redundant condition,
  the R1–R23 doc-comment numbering, `UpstreamBoxOutsideFrame`'s missing register entry) plus the
  five Phase-1-review follow-up notes now recorded in the same file — none blocking Phase 2, but
  cheaper to fix before the files they're in freeze further.
- **§16.28's own deferrals, not yet resolved**: whether an *unmatched* upstream entry's signed
  `DbnetScattered` mechanism is verified against that block's recorded `derivation` inside
  `compare()`; `UpstreamBoxOutsideFrame` stays untranscribed; one genuine spec ambiguity about
  whether §16.27 item 6's derivation-conditional message reverts to its unconditional form or goes
  silent on a `derivation == None` pair (flagged in `COOKBOOK.md`, needs the joint architects).
