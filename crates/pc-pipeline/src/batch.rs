//! spec §4.5 + §5 — the batch runner: per-image isolation, `catch_unwind`, rayon,
//! `--fail-fast`, and the deterministic summary.
//!
//! Task G2, **heavy** (§16.12 item 19). Both functions are `todo!()` for Codex; the
//! panic-message form they must produce is implemented in [`crate::outcome`].

use crate::ctx::PipelineCtx;
use crate::options::PipelineOptions;
use crate::outcome::{BatchSummary, ImageOutcome};
use std::path::{Path, PathBuf};

/// spec §5.2 — one image, with panics contained.
///
/// Contract Codex must satisfy:
///   * on success/normal failure, returns exactly what [`crate::single::process_image`]
///     returned;
///   * a panic anywhere inside becomes
///     `ImageOutcome::Failed { step, error: StageError::Inference(panic_message(..)) }`
///     using [`crate::outcome::panic_message`], and the panic does not propagate;
///   * `step` for a panic is the step the pipeline was on, or `Step::Detect` when that
///     is not knowable.
pub fn process_image_isolated(
    original: &Path,
    options: &PipelineOptions,
    ctx: &PipelineCtx<'_>,
) -> ImageOutcome {
    let _ = (original, options, ctx);
    todo!("task G2 (spec §5.2): catch_unwind boundary around process_image")
}

/// spec §4.5/§5 — the whole batch.
///
/// Contract Codex must satisfy:
///   * `summary.outcomes` is in **input order**, independent of `options.threads`
///     (§5.7 / §16.12 item 17);
///   * parallelism comes from a pipeline-owned `rayon::ThreadPoolBuilder` sized by
///     `options.threads`, never from mutating the global pool;
///   * one `Failed` image never stops the others, unless `options.fail_fast`, in which
///     case no image is *started* after the first failure is observed (images already
///     in flight are allowed to finish, and their outcomes are kept);
///   * every image is represented exactly once in `outcomes` — except under
///     `fail_fast`, where `outcomes` holds only the images actually attempted, still in
///     input order.
pub fn run_batch(
    images: &[PathBuf],
    options: &PipelineOptions,
    ctx: &PipelineCtx<'_>,
) -> BatchSummary {
    let _ = (images, options, ctx);
    todo!("task G2 (spec §4.5, §5): rayon batch runner with per-image isolation")
}
