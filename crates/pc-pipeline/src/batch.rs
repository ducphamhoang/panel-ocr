//! spec §4.5 + §5 — the batch runner: per-image isolation, `catch_unwind`, rayon,
//! `--fail-fast`, and the deterministic summary.
//!
//! Task G2, **heavy** (§16.12 item 19). Both functions are `todo!()` for Codex; the
//! panic-message form they must produce is implemented in [`crate::outcome`].

use crate::ctx::PipelineCtx;
use crate::options::PipelineOptions;
use crate::outcome::{panic_message, BatchSummary, ImageOutcome};
use pc_core::{StageError, Step};
use rayon::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

/// spec §5.2 — one image, with panics contained.
///
/// Contract:
///   * on success/normal failure, returns exactly what
///     [`crate::single::process_image_with_splitting`] returned — §16.14 item 1: the
///     split-aware entry point, so `general.split_long_strips` (default **true**) is
///     actually honoured by every batch;
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
    // DEVIATION(10): upstream lets an exception from a worker abort the whole process
    // pool; we isolate every image at this boundary so one bad page never kills a batch
    // (§5).
    match catch_unwind(AssertUnwindSafe(|| {
        crate::single::process_image_with_splitting(original, options, ctx)
    })) {
        Ok(outcome) => outcome,
        Err(payload) => ImageOutcome::Failed {
            original: original.to_path_buf(),
            // `process_image` does not expose its current stage to this boundary.  Detect
            // is the contract's specified fallback when the stage is not knowable.
            step: Step::Detect,
            error: StageError::Inference(panic_message(payload.as_ref())),
        },
    }
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
    // This lock makes the observation of a failure and the decision to start another
    // image one atomic operation.  Work which has passed this gate is in flight and is
    // deliberately allowed to finish.
    let failed = Arc::new(AtomicBool::new(false));
    let start_gate = Arc::new(Mutex::new(()));
    let fail_fast = options.fail_fast;
    let threads = options.threads.max(1);

    // DEVIATION(11): upstream parallelises *per stage* with Python process pools; we run
    // whole-pipeline, per-image parallelism on a pipeline-owned rayon pool instead (§4.5).
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a positive rayon thread count must build a thread pool");
    let outcomes = pool.install(|| {
        images
            .par_iter()
            .filter_map(|original| {
                if fail_fast {
                    let _gate = start_gate.lock().expect("fail-fast gate poisoned");
                    if failed.load(Ordering::Acquire) {
                        return None;
                    }
                }

                let outcome = process_image_isolated(original, options, ctx);
                if fail_fast && outcome.is_failed() {
                    // Synchronise with would-be starters before publishing the failure.
                    let _gate = start_gate.lock().expect("fail-fast gate poisoned");
                    failed.store(true, Ordering::Release);
                }
                Some(outcome)
            })
            .collect()
    });

    BatchSummary::new(outcomes)
}
