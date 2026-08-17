# BRIEF — joint plan: 8 now-vacuous cross-crate "agree" tests in `pc-pipeline`'s frozen tests

## Context

A test-suite audit (2026-08-16, solo `architect` dispatch) found that a set of
cross-crate "X and Y agree" tests in `crates/pc-pipeline/tests/l1_morph_equivalence.rs`
and `crates/pc-pipeline/tests/l4_composite_equivalence.rs` are now **vacuous** — the
"agree" half of each test compares two call sites that, after a prior hoist landed, both
resolve to the literal same function. Verified by the audit via direct grep (every
`kernel`/`dilate`/`blend_channel`/`resize_nearest_rgba`/`alpha_composite_over`/
`composite_rgb` call site outside `pc_imageops::morph`/`pc_imageops::composite` is a
`pub use` re-export, not an independent implementation):

```
crates/pc-mask/src/grow.rs:22      pub use pc_imageops::morph::{dilate, kernel, Kernel};
crates/pc-denoise/src/lib.rs:67    pub use pc_imageops::morph::{dilate, kernel, Kernel};
crates/pc-mask/src/combine.rs:28   pub use pc_imageops::composite::{alpha_composite_over, blend_channel, composite_rgb, resize_nearest_rgba};
crates/pc-denoise/src/composite.rs:20  (same four)
crates/pc-inpaint/src/compose.rs:41    (same four)
crates/pc-export/src/composite.rs:26   pub use pc_imageops::composite::{alpha_composite_over, blend_channel, resize_nearest_rgba};
```

The affected tests, both files, current names (re-confirmed present in the tree
2026-08-17):

`crates/pc-pipeline/tests/l1_morph_equivalence.rs`:
- `pc_mask_and_pc_denoise_kernels_agree_cell_for_cell_for_thickness_0_through_7`
- `pc_mask_and_pc_denoise_dilate_agree_pixel_for_pixel_for_thickness_0_through_7`
- `pc_denoise_kernel_matches_the_hand_derived_opencv_ellipse_oracle_for_thickness_0_through_7`
  (the audit noted this is now a byte-identical duplicate of a `pc_mask_kernel_...`-named
  test above it, post-hoist)
- `both_crates_dilate_a_single_interior_dot_to_the_hand_written_kernel_1_footprint`
- `both_crates_dilate_a_corner_dot_to_the_hand_written_clipped_kernel_2_footprint`
- `both_crates_dilate_with_the_thickness_0_kernel_as_the_identity`
  (for these three: the audit found the hand-written-footprint half is real, only the
  "both crates" half runs the same function twice)

`crates/pc-pipeline/tests/l4_composite_equivalence.rs`:
- `composite_rgb_agrees_across_pc_mask_pc_denoise_and_pc_inpaint`
  (only the trailing `assert_ne!(via_inpaint, canvas)` control still bites)
- `alpha_composite_over_agrees_across_pc_denoise_and_pc_inpaint_including_the_offset_clip`
  (its own doc comment concedes it doesn't pin the formula's value; a sibling test,
  `composite_value_lock.rs`, does)
- `pc_export_blend_channel_agrees_with_the_hand_computed_lerp_and_pc_mask`
- `pc_export_alpha_composite_over_agrees_with_pc_denoise_including_the_offset_clip`
- `pc_export_resize_nearest_rgba_agrees_and_matches_the_hand_derived_floor_mapping`
  (for these three: added *pre*-hoist by §16.42 item 9 to close a real coverage gap
  that's since closed by the hoist itself; the cross-crate half is now identity, the
  hand-derived half duplicates `crates/pc-export/tests/composite_value_lock.rs`)

## Why this is Spec-sensitive tier, not a unilateral cleanup

**§16.42 item 10 explicitly ruled on this exact file, before the hoist landed**, and its
ruling is quoted here verbatim so its scope travels with it (do not paraphrase further):

> "It does not authorize any edit to `crates/pc-pipeline/tests/l4_composite_equivalence.rs`'s
> **existing** assertions — but item 9 *does* authorize, and expects, an
> **additive-only** extension of that same file... that file legitimately shows a diff
> once item 9's work lands, and the diff being purely additive (no `-` lines against its
> current content) is precisely what this entry does and does not permit there."

That ruling predates the hoist landing and was written to protect the file *during* the
hoist. Whether it should still bind now that the hoist has landed and made part of what
it protected genuinely vacuous is exactly the kind of question two reasonable engineers
could disagree on — which is this project's own definition of Spec-sensitive tier. Do
not treat "the audit already said these are vacuous" as license to just delete them;
that determination, and what (if anything) replaces the tripwire value they still carry,
is what this joint plan is for.

**The audit's own framing, which the plan should engage with rather than re-derive**:
these tests were *deliberately* retained post-hoist as "re-fork tripwires" — `docs/
PIPELINE_SPEC_V1.md:8333` anticipated exactly this vacuity ("would pass vacuously — by
the time the hoist has happened, all four call sites resolve to the same function") and
§16.38 item 16(a) already ratified that `l1_morph_equivalence.rs`'s tests specifically
"stop being a drift check and become a value lock." The audit's finding is narrower than
"these tests are useless": it's that the tripwire value is now much smaller than the
maintenance/runtime cost of keeping N near-duplicate assertions per function, and that no
ratified clause actually disposes of the *specific* now-identity "agree" comparisons.

## What the joint plan needs to produce

1. **A position on whether §16.42 item 10's protection is superseded, narrowed, or still
   binding**, now that the hoist it was written to protect has landed. If superseded or
   narrowed, the transcription must quote item 10's own scope verbatim (already done
   above) and state precisely what changes.
2. **A disposition per test** (or per logical group, where several tests share the same
   disposition and reasoning): keep as-is (if the tripwire value is judged still worth
   the cost), delete the now-identity "agree" half while keeping the hand-derived-oracle
   half (the audit's suggested default), restructure into fewer, differently-shaped
   tests, or something else. Each disposition needs its own stated reasoning — do not
   apply one blanket rule to all 8 without checking whether it actually fits each case
   (e.g. `pc_denoise_kernel_matches_the_hand_derived_opencv_ellipse_oracle_...` is flagged
   as a full duplicate warranting removal, which is a different disposition than the
   three `pc_export_*` tests, which duplicate a *different* file's coverage, not a
   sibling test in the same file).
3. **Whether a replacement tripwire is warranted.** If the "agree" tests are cut, is
   there a cheaper mechanism that still catches a future re-fork (e.g. a single
   compile-time assertion like `const _: fn(...) = pc_mask::grow::dilate;` proving the
   re-export still points at the one real implementation, similar to how
   `l6_inpainter_provider_reexport.rs`'s existing pattern was itself flagged by the same
   audit as over-packaged for what it proves)? Or is losing the runtime tripwire an
   acceptable cost given the re-export sites are simple, rarely-touched `pub use`
   statements?
4. **Task classification**: is this "simple" (one Codex/cmdc call, one review) or does it
   need its own heavier treatment? Given it only touches test files (no production code,
   no fixtures), it is likely simple once the disposition is decided — but the *decision*
   itself is the Spec-sensitive part, not the mechanical edit.

## Constraints

- Do not touch `crates/pc-export/tests/composite_value_lock.rs` or any other frozen test
  outside the two named files without calling that out explicitly as a separate,
  additional finding.
- Any test deletion must be justified per cookbook rule 8's three legitimate exits from a
  frozen test (additive strengthening / spec contradiction routed to both architects /
  registered `DEVIATION(n)`) — state which exit applies to the disposition chosen, don't
  just assert "the audit said so."
- If the two of you disagree on any of the four numbered questions above, do not resolve
  it yourselves — report the disagreement plainly so it can be routed to
  `fable-adjudicator` per this project's standing rule, the same way §16.52's three
  disagreements were handled.
