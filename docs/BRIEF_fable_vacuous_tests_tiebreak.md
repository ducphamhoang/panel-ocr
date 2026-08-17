# Fable tie-break — vacuous cross-crate tests (2026-08-17)

## Context

`architect` and `rust-engineer` were dispatched jointly against
`docs/BRIEF_vacuous_crosscrate_tests_plan.md` (read it first) to decide what to do about
8-14 now-vacuous cross-crate "agree" tests in `crates/pc-pipeline/tests/
l1_morph_equivalence.rs` and `l4_composite_equivalence.rs`. Both ran real mutation/re-fork
probes rather than reasoning abstractly. Full positions:

- `docs/BRIEF_vacuous_tests_rust_engineer_position.md`
- `docs/BRIEF_vacuous_tests_architect_position.md`

**Read both in full before ruling.**

## What they agree on (not in dispute, no need to re-litigate)

- The audit's single-implementation premise is correct (verified independently by both).
- `pc_denoise_kernel_matches_the_hand_derived_opencv_ellipse_oracle_for_thickness_0_through_7`
  is NOT a byte-identical duplicate as the original audit claimed — it's the per-path
  value lock for `pc_denoise::morph::kernel` and should be KEPT. Both landed here
  independently.
- The brief's own proposed replacement tripwire (`const _: fn(u32) -> Kernel = path;`)
  is disproven — both independently compiled it against a real re-forked function and it
  passed clean while the real tests went red. It only proves signature compatibility,
  not function-item identity.
- `pc_mask::combine::alpha_composite_over` has a real, pre-existing coverage gap (no test
  anywhere in the workspace catches a regression in it) that the original audit did not
  find. Both agree this deserves new, additive test coverage.
- Neither believes §16.42 item 10 needs a `SUPERSEDES` marker under their own preferred
  disposition (though for different reasons — see below).

## The three real disagreements

### 1. Group A/C — are the "sole detector" tests actually the sole detectors?

This is not just a difference in judgment — it is a **direct empirical disagreement
about what currently catches what**, both sides having run real experiments:

- **rust-engineer's probe**: re-forking `pc_inpaint::compose::{alpha_composite_over,
  composite_rgb}` was caught **ONLY** by
  `alpha_composite_over_agrees_across_pc_denoise_and_pc_inpaint_...` and
  `composite_rgb_agrees_across_pc_mask_pc_denoise_and_pc_inpaint` — nothing else in the
  workspace failed. Conclusion: KEEP these two (and `pc_mask_and_pc_denoise_dilate_agree_
  pixel_for_pixel_...`, similarly found to be the sole detector for a `pc_denoise::morph::
  dilate` re-fork).
- **architect's probe**: re-forking `pc-inpaint/src/compose.rs` (a whole-module wrapper,
  not the same mutation shape as rust-engineer's) was caught by `l4_compose` (a
  *different* test file, not `l4_composite_equivalence.rs`'s agree tests at all).
  Re-forking `dilate` to no-op past `radius >= 3` left ALL of l1 green, but was caught by
  `m2_grow`/`n3_noise_mask`/several others. Conclusion: DELETE these tests (with
  `composite_rgb` needing a *new* replacement test first, since architect also found no
  existing hand-derived lock for it).

**Both measurements are real, but they used different mutation methodologies** (rust-
engineer perturbed the specific `composite_rgb`/`alpha_composite_over` call paths and a
`dilate` threshold; architect wrapped whole re-exported modules, and separately no-op'd
`dilate` past a radius threshold). It's plausible both are correct about their own
specific probe and the disagreement is really about *which mutation shape is the
representative one to test against* — but that needs someone to actually reconcile,
not be waved away. **This is the crux of the disagreement and where your ruling matters
most.**

### 2. Group D — do the three `pc_export_*` tests actually duplicate `composite_value_lock.rs`?

- rust-engineer accepted the original audit's framing that these are "genuinely
  redundant" with `composite_value_lock.rs`, then recommended KEEP anyway on cost
  grounds (0.01s runtime vs. the cost of narrowing a ratified clause).
- architect directly measured the row-level overlap between the two files' hand-derived
  tables and found only 2 of 8 rows overlap for the blend test, and the resize test pins
  a different downscale case entirely (`(3,2)` vs `(3,1)`) — i.e. the "redundant" premise
  itself is measured false, not just outweighed by cost. architect recommends STRIP the
  now-identity cross-crate loop but KEEP and rename the hand-derived assertions (net:
  neither full deletion nor full no-op).

### 3. Q3 — is a replacement tripwire worth building at all?

- rust-engineer: not warranted. The disproven `const fn` idea and an unsound runtime
  function-pointer-equality idea are the only two mechanisms considered; both rejected;
  conclusion is to build nothing.
- architect: proposes a third mechanism neither the brief nor rust-engineer considered —
  a source-set gate (`pc-testkit`-style runtime file-scanning test, walking every crate's
  `src/` for the six canonical function definitions plus every `pub use` re-export site,
  asserting both sets against a pinned literal). Argues its value doesn't depend on the
  retired tests' tripwire urgency (which architect's own probes found low) but on it
  being *broader* than what it replaces — reaching `pc-mask`'s currently-uncovered
  `alpha_composite_over`/`resize_nearest_rgba` and future crates, per §16.42 item 6's own
  disclosed gap.

## Secondary finding, not in dispute but worth preserving regardless of your ruling

architect found the original brief's own citation was wrong: §16.38 item 16(a) (quoted in
the brief as ratifying `l1_morph_equivalence.rs`'s transition from drift-check to value-
lock) is actually about a *different* file (`crates/pc-denoise/tests/n3_noise_mask.rs`),
confirmed by reading the spec's actual acceptance-gate text. **No ratified clause anywhere
currently disposes of `l1_morph_equivalence.rs`'s tests at all** — the only prior
disposition on record is the file's own header prose, not a ratified spec entry. This
changes how encumbered the situation actually is (less than the brief assumed) and should
be carried into whatever gets ratified, regardless of which side wins the three
disagreements above.

## What's asked of you

1. Rule on disagreements 1-3 above. For #1 especially, don't just pick a side — say
   whether the two probe methodologies are actually testing the same thing, and if not,
   which one (or what combination) should govern the final disposition per test.
2. State whether there's a synthesis neither reached (as happened in this project's
   prior tie-breaks — e.g. §16.51's channel-2 batching ruling, §16.52's three rulings).
3. Not asked: whether to fix the §16.38 item 16(a) mis-citation (not in dispute, already
   agreed should be corrected) or whether the coverage gap for
   `pc_mask::combine::alpha_composite_over` should be filled (not in dispute, already
   agreed it should be, only the exact test content might differ between the two
   drafts — pick either or synthesize, your call if you want to comment, not required).
