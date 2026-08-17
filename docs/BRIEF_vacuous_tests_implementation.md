# BRIEF — implement Fable's T1→T2→gate-demo→T3 synthesis for the vacuous cross-crate tests

## Read first, in this order

1. `docs/BRIEF_vacuous_crosscrate_tests_plan.md` — original task framing, §16.42 item 10
   quoted verbatim.
2. `docs/BRIEF_vacuous_tests_rust_engineer_position.md` and
   `docs/BRIEF_vacuous_tests_architect_position.md` — both full positions, including the
   real mutation probes each side ran (re-derive, don't just trust, if anything here looks
   off).
3. `docs/BRIEF_fable_vacuous_tests_tiebreak.md` — the exact framing dispatched to Fable.
4. This file — Fable's ruling (as reported to the user and recorded in
   `docs/HANDOVER.md`) plus what to actually build.

**This is Spec-sensitive tier.** The user has explicitly chosen to proceed with the full
synthesis rather than the T1-only fallback.

## Fable's ruling (transcribed from the session where it was received; treat this section
as the ruling text to implement, not something to re-litigate)

- **Sole-detector tests** (`pc_mask_and_pc_denoise_dilate_agree_pixel_for_pixel_...`,
  `composite_rgb_agrees_across_pc_mask_pc_denoise_and_pc_inpaint`,
  `alpha_composite_over_agrees_across_pc_denoise_and_pc_inpaint_including_the_offset_clip`):
  rust-engineer wins on present-tense fact — **KEEP for now** (pending the gate
  demonstration below). Both sides' probes were each correct for a different mutation
  shape (per-call-path re-fork vs. whole-module re-fork); Fable re-ran both shapes
  independently and confirmed the reconciliation.
- **Group D** (`pc_export_blend_channel_agrees_with_the_hand_computed_lerp_and_pc_mask`,
  `pc_export_alpha_composite_over_agrees_with_pc_denoise_including_the_offset_clip`,
  `pc_export_resize_nearest_rgba_agrees_and_matches_the_hand_derived_floor_mapping`):
  architect wins — the "duplicates `composite_value_lock.rs`" premise is measured false
  (only 2/8 row overlap on blend; resize pins a different case, `(3,2)` vs `(3,1)`).
  **STRIP** the now-identity cross-crate loop/comparison line from each, **KEEP + RENAME**
  the hand-derived assertions so the name stops claiming a cross-crate comparison it no
  longer makes.
- **Q3 (replacement tripwire)**: architect wins — build the source-set gate in
  `pc-testkit`, with 2 corrections Fable specified: (a) use the **real, complete**
  re-export site set (re-derive it yourself by grepping — do not copy architect's draft
  verbatim, since Fable flagged it as needing a larger set than architect first assumed),
  (b) explicitly pin `PixelSet::dilate`'s exemption in the gate (it is an unrelated
  `pc-testkit` function that happens to share a name with `pc_imageops::morph::dilate`
  and must not be caught by the definition-site scan).
- **Fable's own synthesis (not proposed by either side)** — strict order:
  - **T1** (additive): a `composite_rgb` hand-derived value lock in a new
    `crates/pc-mask/tests/m5_alpha_composite_value_lock.rs`, plus fill the
    `pc_mask::combine::alpha_composite_over` coverage gap (also additive — rust-engineer's
    position doc describes the drafted test's shape at the end of its Q2 section: two
    tests, `pc_mask_alpha_composite_over_matches_the_hand_derived_real_source_over` with 4
    hand-derived cases and an anti-vacuity alpha-out-set check, and
    `pc_mask_alpha_composite_over_drops_a_layer_that_lands_outside_the_destination` for the
    clip rule, both traced to §16.45 item 4 and §16.9 item 15 — the actual test code was
    never written to disk, only described, so derive the concrete assertions yourself from
    those two ratified clauses and `pc_imageops::composite::composite_rgb`/
    `alpha_composite_over`'s real behavior, the same way the other hand-derived tables in
    this codebase are built. File name may combine both `composite_rgb` and
    `alpha_composite_over` coverage into one new test file, or split into two — your call,
    document which).
  - **T2** (the corrected source-set gate, per the 2 corrections above).
  - **A new gate-demonstration step**: re-run the 3 specific re-fork probes from the two
    positions (rust-engineer's per-call-path re-forks of `pc_denoise::morph::dilate`,
    `pc_inpaint::compose::{alpha_composite_over, composite_rgb}`; architect's whole-module
    wrapper re-forks of `pc-mask/src/combine.rs`, `pc-denoise/src/composite.rs`,
    `pc-inpaint/src/compose.rs`) against the T1+T2 tree, and **confirm the gate actually
    goes red on each** — this converts "the gate replaces the sole detectors" from
    argument to measurement, which neither side did. Record the actual pass/fail matrix,
    the same style both position docs used.
  - **T3** (only after the gate is demonstrated red-on-mutation): delete the 3
    sole-detector tests (now provably redundant against the gate) plus
    `pc_mask_and_pc_denoise_kernels_agree_cell_for_cell_for_thickness_0_through_7`
    (subsumed by the two per-path oracles, undisputed by both sides); strip-and-rename the
    3 Group D tests per the disposition above; leave
    `pc_denoise_kernel_matches_the_hand_derived_opencv_ellipse_oracle_for_thickness_0_through_7`
    and the three `both_crates_dilate_*` tests **untouched** (undisputed KEEP on both
    sides — both sides independently found these are not what the audit claimed: the
    kernel-oracle test is a distinct per-path value lock, not a byte-identical duplicate,
    and the `both_crates_dilate_*` tests contain no separable cross-crate "agree" half to
    begin with, each already asserting both paths against the same hard-coded literal).
- **Fallback condition** (not applicable — the user chose the full synthesis): if T2 is
  declined, rust-engineer's "keep everything, add T1 only" governs by default.

## What to actually build, task by task

### T1 — additive test(s) in `pc-mask`

New file(s) under `crates/pc-mask/tests/` (e.g. `m5_alpha_composite_value_lock.rs`):
- Hand-derived value lock for `composite_rgb` (currently has no crate-local or
  cross-crate hand-derived table anywhere in the workspace — verify this yourself before
  writing the test, the same way both position docs verified their claims).
- Hand-derived coverage for `pc_mask::combine::alpha_composite_over`, including the
  clip-rule case (a layer landing outside the destination is dropped, not clamped/wrapped)
  and the alpha-out formula, traced to §16.45 item 4 (the exact regression this must
  catch — the superseded `max(base_a, layer_a)` rule) and §16.9 item 15.
- Anti-vacuity requirement: each hand-derived case must not degenerate to a trivial
  input (e.g. don't only test alpha=0 or alpha=255; include an interior alpha value where
  the two candidate formulas disagree).

Do not touch `crates/pc-export/tests/composite_value_lock.rs` or any other frozen test
outside the two named files (`l1_morph_equivalence.rs`, `l4_composite_equivalence.rs`)
plus this new T1 file.

### T2 — source-set gate in `pc-testkit`

New test file under `crates/pc-testkit/tests/` (naming convention: match this project's
existing gate-test naming, e.g. `spec_supersession.rs`'s style). Walks every
`crates/*/src/**/*.rs` and:
1. Collects every `fn kernel|dilate|blend_channel|resize_nearest_rgba|
   alpha_composite_over|composite_rgb` **definition** site (not re-export). Asserts the
   set equals a pinned literal list. Re-derive the real, complete set yourself by
   grepping — expect the 6 definitions in `pc-imageops` (`morph.rs`, `composite.rs`) that
   both position docs already confirmed, but **do not stop there**: Fable's correction (a)
   says the real re-export site set is larger than architect's first draft assumed, so
   re-derive both lists independently rather than copying either position doc's numbers.
2. Collects every `pub use pc_imageops::{morph::..., composite::...}` re-export site
   across the workspace and asserts that set equals a second pinned literal list.
3. **Explicitly exempts `pc-testkit`'s `PixelSet::dilate`** (an unrelated function that
   happens to share a name with `pc_imageops::morph::dilate`) from the definition-site
   scan, by name or path, with a comment stating why the exemption exists (so a future
   reader doesn't mistake it for an oversight).
4. Hard-codes the expected row counts for both lists (the same anti-emptying-hole pattern
   `EXPECTED_PARSED_CLAIMS` uses for the supersession gate) so silently emptying either
   list fails loudly rather than passing vacuously.

### Gate-demonstration step (do this before T3, not after)

Re-run, against the T1+T2 tree:
1. rust-engineer's per-call-path re-forks: `pc_denoise::morph::dilate` (diverging at
   thickness ≥ 3), `pc_inpaint::compose::{alpha_composite_over, composite_rgb}`.
2. architect's whole-module wrapper re-forks: `pc-mask/src/combine.rs`,
   `pc-denoise/src/composite.rs`, `pc-inpaint/src/compose.rs`.

For each, confirm the new T2 gate does NOT go red (it shouldn't — re-forking behavior
inside an existing definition doesn't change the source-set gate's file-scan result) —
wait, re-read: the gate is a *replacement tripwire* for the now-deleted sole-detector
tests, so what actually needs to go red on these re-forks is either T1's new hand-derived
tests (for the composite_rgb/alpha_composite_over call-path re-forks) or the existing
crate-local tests the architect's probe already found catch the whole-module re-forks
(`m56_run`, `n4_run`, `l4_compose`). **State plainly in your report which mechanism
catches which probe** — the source-set gate (T2) itself is not expected to catch a
behavioral re-fork (it only catches a *structural* re-export change, i.e. someone
introducing a second independent implementation instead of a `pub use`). If T1's new
tests plus the existing crate-local tests together catch all 6 probes with the 3
sole-detector tests removed, that's the actual demonstration Fable asked for — write that
matrix out explicitly, the same style both position docs used, and revert every probe
afterward (`git diff --stat` must return empty for source files after revert, same probe
hygiene both position docs followed).

### T3 — deletions and strip-and-renames (only after the gate-demonstration step above
is complete and its matrix is recorded)

In `crates/pc-pipeline/tests/l1_morph_equivalence.rs`:
- DELETE `pc_mask_and_pc_denoise_kernels_agree_cell_for_cell_for_thickness_0_through_7`.
- DELETE `pc_mask_and_pc_denoise_dilate_agree_pixel_for_pixel_for_thickness_0_through_7`
  — but only once the gate-demonstration step confirms something else in the workspace
  now catches the `dilate` re-fork class this test was the sole detector for (per
  rust-engineer's probe) or that it's genuinely caught elsewhere (per architect's probe)
  — resolve which is actually true for the tree as it stands after T1, don't assume.
- Leave `pc_denoise_kernel_matches_the_hand_derived_opencv_ellipse_oracle_for_thickness_0_through_7`
  and the three `both_crates_dilate_*` tests untouched.

In `crates/pc-pipeline/tests/l4_composite_equivalence.rs`:
- DELETE `composite_rgb_agrees_across_pc_mask_pc_denoise_and_pc_inpaint` and
  `alpha_composite_over_agrees_across_pc_denoise_and_pc_inpaint_including_the_offset_clip`
  — only once T1's replacement coverage is confirmed green and the gate-demonstration
  step confirms the regression classes they caught are now caught elsewhere.
- STRIP-AND-RENAME `pc_export_blend_channel_agrees_with_the_hand_computed_lerp_and_pc_mask`,
  `pc_export_alpha_composite_over_agrees_with_pc_denoise_including_the_offset_clip`,
  `pc_export_resize_nearest_rgba_agrees_and_matches_the_hand_derived_floor_mapping`: keep
  the hand-derived assertion tables, remove the now-identity cross-crate comparison
  loop/line, rename so the new name does not claim a cross-crate comparison.

Also correct, as a small additive/prose fix (not in dispute, already agreed by both
sides): the brief's/architect's finding that §16.38 item 16(a) does NOT ratify
`l1_morph_equivalence.rs`'s transition from drift-check to value-lock (it's actually
about `crates/pc-denoise/tests/n3_noise_mask.rs`) — if `l1_morph_equivalence.rs`'s own
header prose currently claims that citation, correct it there. Do not add a new §16.x
supersession marker for this — it's a citation correction, not a ruling reversal.

## Constraints (carried from the original brief, still binding)

- Do not touch `crates/pc-export/tests/composite_value_lock.rs` or any other frozen test
  outside the files named above.
- Every deletion must be justified per cookbook rule 8's legitimate exits — state which
  exit applies (this is the "fourth exit in substance" architect proposed: a ratified
  structural change landed underneath the test and discharged its protective purpose,
  using exit 2's machinery — joint architects + ratification — without exit 2's literal
  "spec contradiction" wording). Do not just assert "Fable said so."
- Disclose every mutation-probe injection and its revert (probe hygiene): which files
  were temporarily modified, and confirm `git diff --stat` is empty afterward for all of
  them.
- Report `git diff --stat` for the real (non-probe) changes as part of your final report.
- Full verification bar before reporting done: `cargo test --workspace`, `cargo test
  --workspace --all-targets --features pc-cli/onnx`, `cargo clippy --workspace
  --all-targets --all-features -- -D warnings`, `cargo fmt --all --check`.
- This is Spec-sensitive tier — do NOT write the §16.x ratification section yourself.
  Report back to the Orchestrator with everything needed (final dispositions, the gate
  matrix, the exact test names added/deleted/renamed, cookbook-8 exit classification per
  deletion) so the Orchestrator can draft the transcription for a `fresh-reader` pass.
