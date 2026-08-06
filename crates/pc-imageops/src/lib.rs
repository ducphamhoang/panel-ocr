//! `pc-imageops` -- spec §1: pure image algorithms shared by stages, with no stage
//! logic and no config dependency.
//!
//! Current content, three modules:
//!   * `split` -- long-strip splitting (task D8, spec §8.5 / §8.7(A)7 / §8.7(B)8 /
//!     §16.6 item 8)
//!   * `mask`  -- the `BinaryMask` container + box-mask rasterisation (task M1,
//!     spec §10.2 / §10.3 steps 0-2 / §16.9 items 1-4)
//!   * `morph` -- growth kernel + binary dilation (task L1, spec §16.38 item 16(a);
//!     see below)
//!
//! NLM denoise (task N1) is **not** here: §16.10 item 1 placed `nlm` in `pc-denoise`
//! instead, since `pc-imageops` was frozen at the end of Stage 3 and no other v1 stage
//! uses it.
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

pub mod mask;
pub mod morph;
pub mod split;

pub use mask::{rasterize_boxes, BinaryMask, PIL_BINARY_THRESHOLD};
pub use morph::{dilate, kernel, Kernel};
pub use split::{
    calculate_best_splits, row_scores, search_ranges, split_image, stitch_images, SplitParams,
};
