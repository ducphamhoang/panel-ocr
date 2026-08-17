# rust-engineer's position (2026-08-17, joint planning pass vs. `docs/BRIEF_vacuous_crosscrate_tests_plan.md`)

## What I actually ran

| Check | Real output |
|---|---|
| `rg` for `fn kernel|dilate|blend_channel|alpha_composite_over|composite_rgb|resize_nearest_rgba` across `crates/**/src` | **Exactly one definition each**, all in `pc-imageops` (`morph.rs:77,121`; `composite.rs:39,49,93,137`). Every other site is `pub use`. Audit's premise **confirmed**. |
| `cargo test -p pc-pipeline --test l1_morph_equivalence --test l4_composite_equivalence` | **7 passed** and **7 passed** (0 failed), 0.01s / 0.00s |
| `diff` of the two kernel-oracle test bodies (l1 lines 167–202 vs 204–238) | **Not identical** — 8 hunks, different call path |
| 6 injected re-fork probes across 5 crates, each followed by `cargo test --workspace --no-fail-fast` | matrix below |
| `git diff --stat` over all 6 source files + both test dirs after revert | **empty** |
| `cargo test --workspace` post-revert | **1498 passing test lines, 0 FAILED, 0 errors** |

## The decisive experiment

The brief asks whether the tripwire value is real. That is measurable, so I measured it rather than arguing it. I injected six re-forks — each a genuinely wrong implementation, several of them the *exact* bugs this project already ratified fixes for (§16.45 item 4's superseded `max(base_a, layer_a)` rule; truncation standing in for `round`) — and recorded which tests in the whole workspace go red.

| Re-fork injected | Detected by |
|---|---|
| `pc_denoise::morph::dilate`, diverging only at thickness ≥ 3 | **ONLY** `pc_mask_and_pc_denoise_dilate_agree_pixel_for_pixel_...` |
| `pc_denoise::morph::kernel`, one cell at thickness 7 | `pc_denoise_kernel_matches_..._oracle` **and** `pc_mask_and_pc_denoise_kernels_agree_...` |
| `pc_inpaint::compose::{alpha_composite_over, composite_rgb}` | **ONLY** `alpha_composite_over_agrees_across_pc_denoise_and_pc_inpaint_...` and `composite_rgb_agrees_across_pc_mask_pc_denoise_and_pc_inpaint` |
| `pc_denoise::composite::alpha_composite_over` | `pc-denoise/tests/n5_transparent_destination.rs` ×2, plus 2 l4 agree tests |
| `pc_export::composite::alpha_composite_over` | `composite_value_lock.rs` ×3, plus `pc_export_alpha_composite_over_agrees_...` |
| `pc_mask::combine::{blend_channel, alpha_composite_over}` | `m56_run.rs` ×2 + 2 l4 tests **for `blend_channel` only**. For `alpha_composite_over`: **nothing in the workspace failed.** |

Three of the eight tests the audit recommends cutting are the **sole workspace detector** of their re-fork class. The audit's suggested default — "delete the now-identity agree half, keep the hand-derived-oracle half" — is undefined for exactly those three, because they have no hand-derived half for the crate they are the only cover for (`pc-inpaint` has no crate-local composite value-lock at all).

## Q1 — §16.42 item 10

**Position: spent in purpose, but not self-expiring in text — and under my Q2 disposition it does not need to be touched.**

Item 10's protective purpose was *pre-hoist sequencing*: keep the baseline still so the hoist could not be validated against a moved goalpost. That purpose is discharged. But its words carry no expiry — *"does not authorize any edit to ... **existing** assertions"* is unconditional, and the same sentence's authorization is scoped narrowly to *"as part of the pre-hoist test-writing step."* I will not claim it lapsed by itself; a claim that it did is precisely the kind of paraphrase-widening `CLAUDE.md` warns about.

So: **no supersession, no narrowing, no §16.x entry needed** — because I recommend deleting nothing. If the architect's position does call for deletion, then item 10 needs an explicit `NARROWED` entry quoting the passage verbatim (the brief already has the quote), plus the supersession cross-check gate's back-pointer at the item 10 site. That is cookbook rule 8 **exit 2** (spec contradiction → both architects jointly), not exit 1, and not a unilateral cleanup.

## Q2 — disposition per test

**Three factual corrections to the audit first**, since two of its dispositions rest on claims that are false:

1. **`pc_denoise_kernel_matches_the_hand_derived_opencv_ellipse_oracle_for_thickness_0_through_7` is not a "byte-identical duplicate."** `diff` of the two bodies returns 8 hunks; one calls `pc_mask::grow::kernel`, the other `pc_denoise::morph::kernel`. It is the per-path value lock for `pc-denoise`'s public name, and the probe shows it firing on a `pc-denoise` re-fork. Removing it — the audit's strongest recommendation — would delete one of the file's better tripwires.
2. **The three `both_crates_dilate_*` tests contain no cross-crate comparison at all.** Each asserts *each* path against the same hard-coded `EXPECTED` literal (l1:253–254, 280–281, 290–297). `from_mask` is never compared to `from_denoise`. There is no separable "agree half" to delete; both calls already *are* the oracle half.
3. **`pc_mask_and_pc_denoise_dilate_agree_...` is not fully vacuous.** Its tail carries three non-identity assertions — `dimensions() == (16,16)`, the hand-derived `total > 64` floor, and `identity dilation lights exactly 8` (l1:356–375).

**Dispositions:**

- **KEEP AS-IS, sole detectors** (measured): `pc_mask_and_pc_denoise_dilate_agree_pixel_for_pixel_...`, `composite_rgb_agrees_across_pc_mask_pc_denoise_and_pc_inpaint`, `alpha_composite_over_agrees_across_pc_denoise_and_pc_inpaint_...`. No cookbook-8 exit needed; nothing is edited.
- **KEEP AS-IS, misread by the audit**: the three `both_crates_dilate_*`, plus `pc_denoise_kernel_matches_..._oracle`. Per corrections 1 and 2.
- **Genuinely redundant, but still KEEP**: `pc_mask_and_pc_denoise_kernels_agree_cell_for_cell_...` (logically strictly subsumed by the two oracle tests — if both are green it cannot fail; the probe confirms it never fired alone), and the three `pc_export_*` tests (probes show `composite_value_lock.rs` / `m56_run.rs` catch every fork they catch, with better messages).

  My reason for keeping even these is a cost comparison, stated so it can be attacked: the claimed cost is *"maintenance/runtime cost of N near-duplicate assertions."* **Measured runtime: 0.01s and 0.00s for the two files.** Maintenance cost on a frozen file nobody edits is near zero. Against that, deletion costs a narrowing of a ratified clause, a fresh-reader pass, a supersession-gate update, and a permanent precedent that a frozen-test protection was overridden — for a hundredth of a second. The trade is inverted.

**Net: my position is "no change to either file."** The audit's finding is real but its remedy is not; it should close as a documentation correction plus the additive gap-fill below.

## The gap the audit did not find

`pc_mask::combine::alpha_composite_over` has **zero coverage anywhere in the workspace.** I re-forked it to the superseded `max(base_a, layer_a)` rule — the exact regression §16.45 exists to prevent — and `cargo test --workspace --no-fail-fast` reported **not one failure**. This is far more valuable than anything deletion would buy, it is cookbook rule 8 **exit 1** (additive), and it needs no ratification. Drafted test, `crates/pc-mask/tests/m5_alpha_composite_value_lock.rs`.

(Test code omitted here — full text was in the original dispatch output; two tests: `pc_mask_alpha_composite_over_matches_the_hand_derived_real_source_over` with 4 hand-derived cases and an anti-vacuity alpha-out-set check, and `pc_mask_alpha_composite_over_drops_a_layer_that_lands_outside_the_destination` for the clip rule, both traced to §16.45 item 4 and §16.9 item 15.)

## Q3 — replacement tripwire

**Not warranted, and the specific proposal in the brief is measurably weaker than what it would replace.**

`const _: fn(u32) -> Kernel = pc_mask::grow::kernel;` proves only that the path resolves with a compatible signature. **Every one of my six re-forks preserved the signature** — that is what a re-fork *is* — so such a const-assertion would have stayed green through all six. It does not detect the regression class it is proposed to detect.

Function-pointer equality (`pc_mask::grow::kernel as usize == pc_imageops::morph::kernel as usize`) *would* detect a re-fork, but Rust does not guarantee address-uniqueness for function items — identical-code folding can merge two genuinely distinct functions to one address, so the check can pass on a real re-fork. Unsound as a gate; I would reject it if proposed.

## Q4 — classification

The **decision** is Spec-sensitive (it engages a ratified clause), and this joint pass is the right venue for it. The **work** under my position is **simple**: one file added, one cmdc call, one review, no spec edit, no ratification, cookbook rule 8 exit 1.

## Additional findings

1. **The brief's own arithmetic does not close.** Title says *8* tests; the body enumerates *11* names; the two files contain *14* tests (7 + 7, observed).
2. **The flagged set is not complete on its own criterion.** `resize_nearest_rgba_agrees_and_matches_the_hand_derived_floor_mapping` (l4:188) and `the_blend_matches_the_hand_computed_lerp_in_all_three_crates` (l4:85) both contain now-identity cross-crate comparisons and are not in the list.
3. **No frozen test outside the two named files was touched.**
4. **Probe hygiene, disclosed:** six re-forks temporarily modified 6 source files, all reverted via `git checkout`/`rm`; `git diff --stat` empty. First probe was degenerate (discarded, disclosed) — clamping negative dilation targets is equivalent to dropping them for a convex kernel, producing a false "all green."

## Where I expect the architect may disagree

On **Q2 for the four genuinely-redundant tests**. My keep-them argument rests on a cost judgement (0.01s runtime vs. narrowing a ratified clause), not on a coverage fact — a reasonable engineer could weigh suite legibility higher and cut those four. **If we split there, do not let me and the architect settle it between us** — that goes to `fable-adjudicator`. What is *not* open to that disagreement is the empirical half: the three sole-detector tests and the `pc_denoise_kernel_...` oracle must stay regardless of how the cost question lands, because the probe matrix is a measurement, not a position.
