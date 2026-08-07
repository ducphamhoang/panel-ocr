# Handover — TEMPORARY, delete when consumed

Written 2026-08-07. This is continuation scaffolding, not a new ratification; no
§16.x decision is added here. Permanent requirements remain in
`docs/PIPELINE_SPEC_V1.md` and process rules in `CLAUDE.md` / `docs/COOKBOOK.md`.

## Live worktree state

- **`mask-parity`** — `D:\Duc\panel-ocr-mask-parity`, branch `mask-parity`, HEAD
  `42073d0ffad257a3fd149e60d5a09c43c9893a41` (`Wire Annotation refinement into
  detection`), clean before this handover-only edit.
- **`lama-inpaint`** — `D:\Duc\panel-ocr-lama-inpaint`, branch `lama-inpaint`, HEAD
  `ea58f52cd5a009c4dd446ef1e058ff8374d6a92e` (`Update HANDOVER.md: current state,
  Codex sandbox blocker`), clean. Its implementation/ratification baseline remains
  `66cf7e4`; the newer commit is documentation only.

Use `export PATH="$HOME/.cargo/bin:$PATH"` in every Bash shell.

## Mask parity: A4-a complete; A4-b is next

A4-a is complete at `42073d0`. `pc_detect::run` now branches on
`MaskRefineMode`: `Annotation` calls `refine_mask` and then
`refine_undetected_mask`, while `Simple` retains `refine_simple`. Coverage is still
scored from `refined_mask` in both modes. The mode-dependent operand selection
ratified for A4-b is explicitly **not implemented** by this commit.

The frozen A4-a artifact is
`crates/pc-detect/tests/a4_run_annotation.rs` (6 tests). Corrections incorporated
before freezing matter to future review:

- the exact Simple quarantine test is
  `run_in_simple_mode_is_byte_identical_to_the_committed_recorded_mask`;
- recorded-page rectangle identity is order-independent, not a count assertion;
- non-zero counts and positional fingerprints are described only as aggregate
  checks that catch many wrong masks, not exact bitmap or lossless identities;
- the disk-artifact test asserts exact equality between the decoded destination
  image and the mask emitted by the output handle;
- the superseded frozen D7 refusal
  `run_rejects_annotation_refine_mode` was removed under §16.39 item 3(d), rather
  than retained with a misleading rejection claim.

Independent post-implementation review found one stale `run` rustdoc statement.
Codex corrected it; the independent reviewer re-checked the correction and
reported no remaining findings.

Live verification on `42073d0` passed:

- A4 integration file: **6 passed / 0 failed**;
- D7 integration file: **20 passed / 0 failed**;
- `cargo test --workspace`;
- `cargo test --workspace --all-targets --features pc-cli/onnx`;
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`;
- `cargo fmt --all --check`;
- `git diff --check`.

No aggregate workspace count is recorded here because none was captured reliably.
Re-run the complete bar before the next commit rather than treating this handover
as evidence for a changed tree.

### Exact next task: A4-b (heavy, isolated Codex call)

Implement only §16.39 item 1's mode-dependent coverage operand. No
`crates/pc-detect/tests/a4_coverage_operand.rs` file is written, frozen, or
committed yet; material from the prior transcript was draft only. Before
implementation, write and freeze an artifact containing exactly these three
§16.39 item 6(b) tests:

1. `annotation_coverage_scores_the_unrefined_mask_and_simple_scores_the_refined_one`;
2. `simple_mode_keeps_the_refined_operand_after_the_annotation_change`;
3. `the_recorded_page_keeps_the_same_block_set_under_both_operands`.

The historical four-test count is superseded for planning purposes. Do not add
the caller-threshold test; the existing A4-a Simple digest test remains the
separate quarantine.

Scope: This ruling applies only to the contents and status description of the
not-yet-written A4-b frozen artifact
`crates/pc-detect/tests/a4_coverage_operand.rs`: it must contain exactly the
three tests enumerated by §16.39 item 6(b), while the existing A4-a Simple digest
test remains a separate quarantine. It does not decide whether
`DetectInput.min_mask_coverage` should become configurable in a future
specification or whether a caller-threshold test belongs in a separately
ratified task.

`Annotation` coverage must use the original unrefined detector mask after
letterbox crop and resize. `Simple` must continue using `refined_mask`
byte-for-byte.

The committed E01P01 real page does **not** discriminate the operands: all four
blocks receive the same keep/drop verdict under both. The synthetic fixture must
force opposite verdicts and assert survivor rectangle identity in both directions;
a green real-page lock is only a no-regression check, not evidence for the operand
ruling.

## Benchmark and LaMa dependency

The three-way **Simple vs Annotation vs LaMa** real-page benchmark has **not run**.
`docs/MASK_QUALITY_CALIBRATION.md` is older replay-only evidence, not that
comparison: 1 page, 3 regions, 2 already accepted, 0 rescue-eligible, 1 failed even
with rescue, 0 dropped; the failed region's lowest deviation is `29.129848`.

LaMa cannot participate end-to-end until its L6-1 through L6-5 pipeline work
lands. The ordered path to the benchmark is:

1. Complete A4-b.
2. Land L6-1+L6-2 as a simple batch, then L6-3 (heavy export precedence),
   L6-4 (heavy pipeline wiring with eligibility-first model creation), and L6-5
   (simple CLI/provider handoff).
3. Reconcile the independently diverged branch and specification histories.
4. Run the pinned, non-gating Simple vs Annotation vs LaMa benchmark.

## Process cautions

- The Orchestrator drives the workflow and does not write implementation; Codex
  writes implementation against the pre-written tests.
- Tests freeze once written. A contradiction returns to both architects; do not
  edit a test merely to make implementation pass.
- Run the full workspace, ONNX all-target, clippy `-D warnings`, fmt, and diff
  checks before committing. Pushing still requires an explicit request.
- The Choujin Locke benchmark page is local-only and must never be committed.
- Inspect file diffs/mtimes and test output, not agent status claims. A completed
  job can have written nothing.
- A4-a succeeding in this worktree proves that job wrote here; it does **not** prove
  Codex ACL/onboarding is fixed globally or for other worktrees. Diagnose the live
  target rather than preserving or dismissing the old blocker by assumption.
- Read `docs/COOKBOOK.md` before trusting a green suite.
