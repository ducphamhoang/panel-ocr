# architect's position (2026-08-17, joint planning pass vs. `docs/BRIEF_vacuous_crosscrate_tests_plan.md`)

Verified via mutation probes in a detached worktree at `6c809bb` (created, used, restored, removed — main worktree untouched). Confirmed the audit's single-implementation premise directly (grep: exactly six real definitions, all in `pc-imageops/src/{morph,composite}.rs`; every other site is `pub use`).

## Q1 — §16.42 item 10

**Position: item 10 never prohibited this — it's a negative/scoping clause ("this entry does not authorize X"), not an affirmative freeze — so no supersession marker is needed.** But cookbook rule 8's three exits don't literally fit a deletion motivated by "a ratified structural change landed underneath the test and destroyed its subject" either (exit 1 is additive-only; exit 2 is about spec-contradicting assertions; exit 3 is DEVIATION). Recommends treating this as a **fourth exit in substance**, using exit 2's machinery (joint architects → ratification) without exit 2's literal wording, with precedent in how §16.20's snapshot removal was handled. Recommends codifying this fourth exit in the cookbook.

**Two corrections to the brief's own citations, verified by reading the actual spec lines:**
- §16.38 item 16(a) (quoted in the brief) is NOT about `l1_morph_equivalence.rs` — its subject is explicitly `crates/pc-denoise/tests/n3_noise_mask.rs`'s cell-matrix test, and its acceptance gate names only `pc-mask/tests` and `pc-denoise/tests` (`l1_morph_equivalence.rs` is in neither, didn't exist yet). **No ratified clause anywhere actually disposes of `l1_morph_equivalence.rs`'s tests** — the only disposition on record is the file's own header prose, not a ratified artifact. This makes the situation less encumbered than the brief assumed.
- `docs/PIPELINE_SPEC_V1.md:8333` is §16.42 item 6, about `pc-export`'s *pre-hoist* value lock — predicts the vacuity correctly but disposes of nothing here.

## Q2 — disposition per test, from measurement

Ran real re-fork/mutation probes (whole-workspace `cargo test`, checking what actually goes red):

| mutation | result |
|---|---|
| `dilate` no-ops at `radius >= 3` | **l1 entirely green.** Caught elsewhere: `m2_grow`, `n3_noise_mask`, `l4_compose`, `l4_inpainter_trait`, `l6_inpaint_pipeline`, `q0_fit_report`, `q2_fallback_on_real_ladders`, `q4_fit_region_fallback`, `mask_sweep_cli` |
| `kernel` `.round()`→`.trunc()` | 3 red: both per-path oracles + `kernels_agree` (via a `total==496` literal, not the cross-crate comparison) |
| `pc_mask::grow::kernel` re-forked (off-by-one) | 6 of 7 l1 tests red, including both per-path oracles |
| `blend_channel` `.round()`→`.floor()` | 2 red (both hand-derived tables) |
| `resize_nearest_rgba` floor→ceil | 2 red (both hand-derived mappings) |
| `alpha_out` reverted to superseded `max(base_a, layer_a)` (§16.45's exact prior bug) | **0 red** in l4 or `l4_compose`. Caught only by `composite_value_lock.rs` |
| `composite_rgb`'s `a==0` branch returns layer, not canvas | **0 red** anywhere in l4 or `l4_compose` |
| re-fork `pc-mask/src/combine.rs`, `pc-denoise/src/composite.rs`, `pc-inpaint/src/compose.rs` (whole-module wrapper) | each caught by that crate's own test file (`m56_run`, `n4_run`, `l4_compose` respectively) |

**Dispositions:**
- **Group A — DELETE**: l1's `pc_mask_and_pc_denoise_kernels_agree_...` (subsumed by per-path oracles) and `pc_mask_and_pc_denoise_dilate_agree_...` (cross-crate half vacuous, `total>64` guard measured NOT to bite the `radius>=3` mutation, identity check redundant with another test). Discloses a real residual gap this creates: dilation at thicknesses 3,6,7 becomes covered nowhere (pre-existing gap, not created by deletion, but should be stated).
- **Group B — KEEP, disagrees with the audit**: `pc_denoise_kernel_matches_..._oracle` is NOT byte-identical (8 diff hunks, different call path) and is the per-path value lock for `pc_denoise::morph::kernel` — measured to catch what the agree test catches, plus more.
- **Group C — DELETE (with a blocking precondition)**: l4's `alpha_composite_over_agrees_across_pc_denoise_and_pc_inpaint_...` and `pc_export_alpha_composite_over_agrees_...` (measured 0-red under a full §16.45 revert; `composite_value_lock.rs` measured to catch it — delete outright). `composite_rgb_agrees_across_pc_mask_pc_denoise_and_pc_inpaint` measured 0-red under a real branch bug, and **there is no hand-derived value lock for `composite_rgb` anywhere in the workspace** — blocking condition: add one (T1) before deleting this test.
- **Group D — STRIP-AND-RENAME, not delete; disagrees with the audit's "duplicates `composite_value_lock.rs`" claim**: measured the two `pc_export_*` blend/resize tests' hand-derived tables against `composite_value_lock.rs`'s — only 2 of 8 rows overlap for blend; resize pins a different downscale case (`(3,2)` vs `(3,1)`). Deleting them loses real coverage (e.g. near-boundary rounding cases). Recommends stripping only the now-identity cross-crate loop/line, keeping the hand-derived assertions, and renaming so the name stops claiming a cross-crate comparison.

**Net: 3 deleted, 3 stripped-and-renamed, 1 kept unchanged (Group B), 1 replaced by a new stronger test (composite_rgb) before its old test is deleted.**

## Q3 — replacement tripwire

**Agrees with rust-engineer that the brief's `const _: fn(...) = path;` proposal is disproven** — architect actually compiled it against a live re-forked `pc_mask::grow::kernel` (off-by-one, wrong output) and it compiled clean while 6 of 7 l1 tests went red. Proves signature compatibility only, not function-item identity. Also rejects runtime function-pointer equality (unsound: identical-code-folding can merge distinct functions to one address; also likely trips `unpredictable_function_pointer_comparisons`, warn-by-default, unconfirmed against the pinned toolchain).

**Diverges from rust-engineer's "not warranted" conclusion**: proposes building a different mechanism — a **source-set gate** in `pc-testkit` (precedent: `spec_supersession.rs`-style runtime file parsing). Walks every `crates/*/src/**/*.rs`, collects every `fn kernel|dilate|blend_channel|resize_nearest_rgba|alpha_composite_over|composite_rgb` definition site, asserts the set equals a pinned literal (6 rows in `pc-imageops`, one named exemption for `pc-testkit`'s unrelated `PixelSet::dilate`) — and separately asserts every `pub use` re-export site set equals a pinned six. Argues this is worth building because it's *broader* than the 8 tests it replaces (also covers `pc-mask`'s currently-uncovered `alpha_composite_over`/`resize_nearest_rgba` per §16.42 item 6, plus `pc-export`, plus any future crate) — not because the runtime tripwire was urgent (measured: all three re-forks were already caught by the crates' own tests anyway).

## Q4 — sequencing

Decision is Spec-sensitive; edits are three simple, ORDER-DEPENDENT tasks: T1 (add the `composite_rgb` hand-derived value lock, additive, must land green first) + T2 (the source-set gate, if adopted, independent) together, then T3 (the deletions/strip-renames/header-prose fixes) only after T1/T2 are confirmed green against the pre-deletion tree.

## Where architect expects to disagree with rust-engineer

Stated explicitly by architect:
- **Group B**: not actually a disagreement — both independently landed on KEEP for `pc_denoise_kernel_matches_..._oracle`, for the same reason (it's not a duplicate, it's the per-path value lock).
- **Group A/C (the sole-detector tests)**: real disagreement. rust-engineer's probes found `pc_mask_and_pc_denoise_dilate_agree_...`, `composite_rgb_agrees_...`, and `alpha_composite_over_agrees_...` are each the SOLE workspace detector for their respective re-fork and recommends KEEP. architect's probes found these same regression classes ARE caught elsewhere (`m2_grow`/`n3_noise_mask` for dilate; `l4_compose` for the inpaint-side re-forks) and recommends DELETE (with composite_rgb needing new replacement coverage first, since no hand-derived lock exists for it anywhere). **This is not just a difference in cost-weighing — it's a direct empirical disagreement about which tests currently catch which regressions**, worth resolving by re-derivation rather than by picking a side.
- **Group D**: rust-engineer classified the three `pc_export_*` tests as "genuinely redundant, but keep anyway" (accepting the audit's redundancy claim, then overriding on cost grounds); architect measured the redundancy claim itself to be false (only 2/8 row overlap) and recommends strip-and-rename rather than full keep.
- **Q3**: rust-engineer says no replacement tripwire is warranted; architect proposes building the source-set gate, arguing its value is independent of the retired agree-tests' tripwire urgency.
