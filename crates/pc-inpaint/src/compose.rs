//! RGBA composition helpers — spec §16.38 item 3(h), arithmetic pinned by §16.9 items 13 and 15.
//!
//! **The arithmetic now lives in `pc_imageops::composite` (§16.42)** and is re-exported here, so
//! `pc_inpaint::compose::<name>` still resolves for every existing caller and for the frozen tests.
//! It is unchanged.
//!
//! **This module used to hold a FOURTH verbatim copy, and recorded that as an open question rather
//! than resolving it.** `pc_mask::combine` held the original (§16.9 items 13, 15),
//! `pc_denoise::composite` a second, `crates/pc-export/src/composite.rs` a third and this module a
//! fourth: `blend_channel` / `alpha_composite_over` / `resize_nearest_rgba` existed in four crates,
//! `composite_rgb` in three (`pc-export` had none). §16.10 item 3 **pinned** that duplication, with
//! a pixel-for-pixel agreement requirement, because §11.3 step 2 reproduces stage 3's composite
//! rather than trusting `_clean.png`; §16.38 item 3(h) puts this stage in the same position, since
//! it rebuilds the cleaned page from the original plus the masks (`inpainting.py:149-157`), so
//! "`_clean_inpaint.png` is not derived from `_clean.png`".
//!
//! **§16.42 is the answer to the open question this module used to pose.** The question was whether
//! §16.38 item 16(a)'s "a third verbatim copy is not acceptable" reasoning also applied here, or
//! whether §16.10 item 3's pin stood and four copies were the ratified shape. §16.42 rules the
//! former: the four functions are config-free pixel math, `pc-imageops` is not a stage crate so §1
//! rule 2 is untouched, and it supersedes both §16.10 item 3 and §16.11 item 10. The agreement the
//! pin demanded is now structural rather than hand-maintained.
//!
//! **On coverage, stated as it stands rather than as the hoist makes it look.**
//! `crates/pc-pipeline/tests/l4_composite_equivalence.rs` pinned this module against
//! `pc_denoise::composite` and, for `blend_channel` and `composite_rgb`, against `pc_mask::combine`
//! — pre-hoist that was a real cross-implementation check; post-hoist those crates resolve to one
//! function, so those particular assertions no longer discriminate between implementations and are
//! kept as a regression guard on the re-export paths. The check that carries weight for the copy
//! that had none is `crates/pc-export/tests/composite_value_lock.rs`, whose expectations are
//! hand-derived from §16.9 items 13/15 and which was observed green against the **un-hoisted**
//! `pc-export` copy before this move (§16.42 item 6's binding sequencing requirement).

pub use pc_imageops::composite::{
    alpha_composite_over, blend_channel, composite_rgb, resize_nearest_rgba,
};
