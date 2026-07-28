//! `pc-imageops` -- spec §1: pure image algorithms shared by stages, with no stage
//! logic and no config dependency.
//!
//! v1 content: long-strip splitting (task D8, spec §8.5 / §8.7(A)7 / §8.7(B)8 /
//! §16.6 item 8) and the `BinaryMask` container + box-mask rasterisation (task M1,
//! spec §10.2 / §10.3 steps 0-2 / §16.9 items 1-4). Task N1 (NLM denoise) adds a third
//! module alongside these.
//!
//! §16.9 item 2: only M1 lives here. The growth kernels, border statistics, fitting
//! policy and composition (M2-M5) are *masking policy* and live in `pc-mask`, which is
//! also what keeps this crate free of a `pc-config` dependency.

pub mod mask;
pub mod split;

pub use mask::{rasterize_boxes, BinaryMask, PIL_BINARY_THRESHOLD};
pub use split::{
    calculate_best_splits, row_scores, search_ranges, split_image, stitch_images, SplitParams,
};
