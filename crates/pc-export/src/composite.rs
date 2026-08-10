//! Nearest resampling + source-over composition (spec §12.3 step 4, §16.42).
//!
//! **The arithmetic now lives in `pc_imageops::composite` (§16.42)** and is re-exported
//! here, so `pc_export::composite::<name>` still resolves for every existing caller and
//! for the frozen tests. `resize_nearest_rgba` and `blend_channel` are unchanged and still
//! pinned by §16.9 items 13 and 15; `alpha_composite_over` is not — §16.45 item 4 replaced
//! its rule with real (Porter-Duff) source-over, `out_a = sa + da*(1 - sa)`. §16.45 item 6
//! records this crate's mask export as the second, independent site the old
//! destination-alpha-ignoring rule corrupted.
//!
//! The mask this stage upscales is the same artifact the denoiser saw, so its arithmetic
//! must agree with `pc-denoise`'s pixel-for-pixel. §1 rule 2 forbids importing it from a
//! sibling stage crate, and §16.11 item 10 pinned the duplication instead; §16.42
//! overturns that pin, because `pc-imageops` is not a stage crate and §1's crate
//! dependency graph already sanctions the `pc-imageops <- pc-export` edge.
//!
//! Only three functions are re-exported: this stage never had `composite_rgb` and does not
//! gain one here (§16.42 item 2).
//!
//! `DEVIATION(8)` used to be attached to `resize_nearest_rgba` in this module. It is about
//! **one call site** — the denoise-mask upscale in `lib.rs`'s `export_mask` — not about
//! nearest resampling in general, so per §16.42 item 5 it moved there rather than
//! travelling into the shared primitive, which serves four crates and cannot know which
//! caller it would be describing.

pub use pc_imageops::composite::{alpha_composite_over, blend_channel, resize_nearest_rgba};
