//! RGBA composition helpers (spec §11.3 steps 2 and 5, §16.42).
//!
//! **The arithmetic now lives in `pc_imageops::composite` (§16.42)** and is re-exported
//! here, so `pc_denoise::composite::<name>` still resolves for every existing caller and
//! for the frozen tests. `resize_nearest_rgba`, `blend_channel` and `composite_rgb` are
//! unchanged and still pinned by §16.9 items 13 and 15; `alpha_composite_over` is not —
//! §16.45 item 4 replaced its rule with real (Porter-Duff) source-over,
//! `out_a = sa + da*(1 - sa)`, because the old rule ignored the destination's alpha and
//! corrupted every partial-alpha rim pixel `noise_mask::build_noise_mask` composites onto
//! its transparent canvas (§16.45 item 1 — this crate is where the defect was found).
//!
//! §11.3 step 2 reproduces Stage 3's clean output at full resolution rather than trusting
//! `_clean.png`, so this crate's arithmetic must agree with `pc-mask`'s pixel-for-pixel.
//! §1 rule 2 forbids importing it from `pc-mask` -- so §16.10 item 3 pinned the
//! duplication instead. §16.42 overturns that pin: `pc-imageops` is not a stage crate, so
//! sharing through it satisfies rule 2 and makes the agreement structural rather than
//! maintained by hand -- the same move §16.38 item 16(a) made for `morph` and §16.38 item
//! 20 for `gaussian`.

pub use pc_imageops::composite::{
    alpha_composite_over, blend_channel, composite_rgb, resize_nearest_rgba,
};
