//! `pc-imageops` -- spec §1: pure image algorithms shared by stages, with no stage
//! logic and no config dependency.
//!
//! v1 content: long-strip splitting (task D8, spec §8.5 / §8.7(A)7 / §8.7(B)8 /
//! §16.6 item 8). Later tasks (M1 box-mask rasterisation, N1 NLM denoise) add modules
//! alongside `split`.

pub mod split;

pub use split::{
    calculate_best_splits, row_scores, search_ranges, split_image, stitch_images, SplitParams,
};
