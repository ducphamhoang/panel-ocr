//! `pc-imageops` -- spec §1: pure image algorithms shared by stages, with no stage
//! logic and no config dependency.
//!
//! Current content, five modules:
//!   * `split` -- long-strip splitting (task D8, spec §8.5 / §8.7(A)7 / §8.7(B)8 /
//!     §16.6 item 8)
//!   * `mask`  -- the `BinaryMask` container + box-mask rasterisation (task M1,
//!     spec §10.2 / §10.3 steps 0-2 / §16.9 items 1-4)
//!   * `morph` -- growth kernel + binary dilation (task L1, spec §16.38 item 16(a);
//!     see below)
//!   * `gaussian` -- separable Gaussian blur, `sigma = radius` (task N2, hoisted here by
//!     task L4; the module header carries the ground and what it does not claim)
//!   * `composite` -- `blend_channel` / `resize_nearest_rgba` / `alpha_composite_over` /
//!     `composite_rgb` (§16.42, which overturns §16.10 item 3's and §16.11 item 10's
//!     duplication pins; the module header carries the ground and the scope limit)
//!
//! NLM denoise (task N1) is **not** here: §16.10 item 1 placed `nlm` in `pc-denoise`
//! instead, since `pc-imageops` was frozen at the end of Stage 3 and no other v1 stage
//! uses it. That item's ground was per-module and conditional, and for `gaussian` the
//! condition expired at v1.5 -- `pc-inpaint` needs the same blur for
//! `inpainting_fade_radius` (§16.38 item 3(h)) -- so task L4 took the *"v1.5 hoist is a
//! move plus a re-export"* §16.10 item 1 itself anticipated. `nlm` still has exactly one
//! consumer and stays where it is.
//!
//! §16.9 item 2 placed only M1 here, on the ground that the growth kernels, border
//! statistics, fitting policy and composition (M2-M5) are *masking policy* and belong in
//! `pc-mask`. **§16.38 item 16(a) supersedes that in part** (v1.5, task L1): the two
//! config-free primitives `kernel` and `dilate` are hoisted here as `morph`, because
//! `pc-mask`, `pc-denoise` and `pc-inpaint` all need them and §1 rule 2 forbids the
//! stage-to-stage edge that would let them share. Everything else §16.9 item 2 placed in
//! `pc-mask` stays there, including the `MaskerConfig`-shaped growth helpers -- which is
//! what keeps this crate free of a `pc-config` dependency, a property §16.38 item 16(a)
//! explicitly requires to survive the hoist.
//!
//! **§16.42 supersedes the "composition (M2-M5) is masking policy" part of that paragraph,
//! for four functions only.** `blend_channel`, `resize_nearest_rgba`,
//! `alpha_composite_over` and `composite_rgb` are config-free pixel math needed by
//! `pc-mask`, `pc-denoise`, `pc-export` and `pc-inpaint`, so they move here as `composite`.
//! §16.42 item 8 is explicit that this authorises nothing further: `cleaned_image`,
//! `text_layer`, `mask_overlay` and `build_combined_mask` remain masking policy and remain
//! in `pc-mask`. The `pc-config`-free property is unaffected -- see §16.42 item 3.

pub mod composite;
pub mod gaussian;
pub mod mask;
pub mod morph;
pub mod split;

pub use composite::{alpha_composite_over, blend_channel, composite_rgb, resize_nearest_rgba};
pub use gaussian::{blur, taps};
pub use mask::{rasterize_boxes, BinaryMask, PIL_BINARY_THRESHOLD};
pub use morph::{dilate, kernel, Kernel};
pub use split::{
    calculate_best_splits, row_scores, search_ranges, split_image, stitch_images, SplitParams,
};
