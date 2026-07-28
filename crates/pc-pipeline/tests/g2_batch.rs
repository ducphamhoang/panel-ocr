//! Task **G2** — the batch runner: per-image isolation, `catch_unwind`, ordering,
//! `--fail-fast`, exit codes (spec §4.5, §5.1–§5.5, §5.7, §16.12 items 17 and 18).
//!
//! The pure summary/exit-code parts pass today; everything that calls `run_batch` or
//! `process_image_isolated` fails until G2 is implemented.
//!
//! FROZEN (CLAUDE.md).

mod common;

use pc_core::{Rect, StageError, Step};
use pc_pipeline::outcome::{panic_message, BatchSummary, ImageAnalytics, ImageOutcome, SkipReason};
use pc_pipeline::{PipelineCtx, SharedDetector, EXIT_FATAL, EXIT_OK, EXIT_PARTIAL};
use std::path::PathBuf;

fn completed(name: &str) -> ImageOutcome {
    ImageOutcome::Completed {
        original: PathBuf::from(name),
        files_written: vec![PathBuf::from(format!("out/{name}_clean.png"))],
        analytics: Box::new(ImageAnalytics::default()),
    }
}

fn failed(name: &str) -> ImageOutcome {
    ImageOutcome::Failed {
        original: PathBuf::from(name),
        step: Step::Mask,
        error: StageError::Empty("nothing usable".into()),
    }
}

fn skipped(name: &str) -> ImageOutcome {
    ImageOutcome::Skipped {
        original: PathBuf::from(name),
        reason: SkipReason::NoTextDetected,
        files_written: vec![PathBuf::from(format!("out/{name}_clean.png"))],
    }
}

/// §5.5: `0` when everything completed (skips allowed), `2` when at least one failed.
#[test]
fn exit_codes_follow_the_outcome_mix() {
    assert_eq!(
        BatchSummary::new(vec![completed("a"), skipped("b")]).exit_code(),
        EXIT_OK
    );
    assert_eq!(
        BatchSummary::new(vec![completed("a"), failed("b")]).exit_code(),
        EXIT_PARTIAL
    );
    assert_eq!(BatchSummary::default().exit_code(), EXIT_OK);
}

/// §5.5: the summary always lists the failed and skipped images with their reasons.
#[test]
fn the_summary_names_failed_and_skipped_images() {
    let summary = BatchSummary::new(vec![completed("a"), skipped("b"), failed("c")]);

    let text = summary.render();

    assert_eq!(
        (summary.completed(), summary.skipped(), summary.failed()),
        (1, 1, 1)
    );
    assert!(
        text.contains('b') && text.contains("no text detected"),
        "{text}"
    );
    assert!(text.contains('c') && text.contains("Mask"), "{text}");
    assert_eq!(summary.files_written().len(), 2);
}

/// §16.19 item 5: provider-declared fatality, not the `StageError` variant, controls the
/// batch outcome. This deliberately uses the same `StageError::Model` for both arms.
#[test]
fn provider_declared_fatality_controls_batch_outcome() {
    let dir = tempfile::tempdir().unwrap();
    let images = ["a.png", "b.png", "c.png"]
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let mut options = common::options(&dir.path().join("cache"), &dir.path().join("out"));
    options.threads = 1;

    let per_image = common::FatalityProvider::new(false);
    let summary = pc_pipeline::run_batch(&images, &options, &PipelineCtx::new(&per_image));
    assert_eq!(
        per_image.attempts.load(std::sync::atomic::Ordering::SeqCst),
        3
    );
    assert_eq!(summary.outcomes.len(), 3);
    assert!(summary
        .outcomes
        .iter()
        .all(|outcome| matches!(outcome, ImageOutcome::Failed { .. })));
    assert_eq!(summary.exit_code(), EXIT_PARTIAL);

    let run_fatal = common::FatalityProvider::new(true);
    let summary = pc_pipeline::run_batch(&images, &options, &PipelineCtx::new(&run_fatal));
    assert_eq!(summary.exit_code(), EXIT_FATAL);
    assert!(!summary.outcomes.is_empty());
    assert!(summary
        .outcomes
        .iter()
        .all(|outcome| matches!(outcome, ImageOutcome::RunFatal { .. })));
}

/// §16.19 item 5(c): a provider which does not override the method gets the safe default.
#[test]
fn detector_provider_fatality_defaults_to_false() {
    let provider = common::RefusingProvider;
    assert!(!pc_pipeline::DetectorProvider::failures_are_run_fatal(
        &provider
    ));
}

/// §5.2 / §16.12 item 18 — the exact panic-message form.
#[test]
fn panic_messages_are_prefixed_and_never_lose_the_payload() {
    let string_payload: Box<dyn std::any::Any + Send> = Box::new("boom".to_string());
    let str_payload: Box<dyn std::any::Any + Send> = Box::new("bang");
    let other_payload: Box<dyn std::any::Any + Send> = Box::new(7_u32);

    assert_eq!(panic_message(string_payload.as_ref()), "panicked: boom");
    assert_eq!(panic_message(str_payload.as_ref()), "panicked: bang");
    assert_eq!(
        panic_message(other_payload.as_ref()),
        "panicked: <non-string panic payload>"
    );
}

/// §5.1: one bad image never kills a batch; the others still complete and the run
/// reports exit code 2.
#[test]
fn a_failing_image_does_not_stop_the_batch() {
    let dir = tempfile::tempdir().unwrap();
    let images: Vec<PathBuf> = ["a.png", "b.png", "c.png"]
        .iter()
        .map(|name| common::write_page(dir.path(), name, (48, 48)))
        .collect();
    let mut options = common::options(&dir.path().join("cache"), &dir.path().join("out"));
    options.profile.general.preferred_file_type = ".png".to_string();
    options.threads = 2;

    let provider = common::SelectiveProvider::new(&["b"], common::failing_detector())
        .with_fallback(common::detector_with_block(Rect::new(6, 6, 40, 40)));
    let summary = pc_pipeline::run_batch(&images, &options, &PipelineCtx::new(&provider));

    assert_eq!(summary.outcomes.len(), 3);
    assert_eq!(summary.failed(), 1);
    assert_eq!(summary.exit_code(), EXIT_PARTIAL);
    assert!(summary.outcomes[1].is_failed());
    assert!(!summary.outcomes[0].is_failed() && !summary.outcomes[2].is_failed());
}

/// §5.2: a panic inside a stage becomes a `Failed`, never an aborted batch.
#[test]
fn a_panicking_stage_is_contained() {
    let dir = tempfile::tempdir().unwrap();
    let images: Vec<PathBuf> = ["a.png", "b.png"]
        .iter()
        .map(|name| common::write_page(dir.path(), name, (48, 48)))
        .collect();
    let mut options = common::options(&dir.path().join("cache"), &dir.path().join("out"));
    options.profile.general.preferred_file_type = ".png".to_string();

    let provider =
        common::SelectiveProvider::new(&["a"], std::sync::Arc::new(common::PanickingDetector))
            .with_fallback(common::detector_with_block(Rect::new(6, 6, 40, 40)));
    let summary = pc_pipeline::run_batch(&images, &options, &PipelineCtx::new(&provider));

    assert_eq!(summary.outcomes.len(), 2);
    match &summary.outcomes[0] {
        ImageOutcome::Failed { error, .. } => assert!(
            error.to_string().contains("panicked"),
            "panic must be reported as such, got {error}"
        ),
        other => panic!("expected Failed, got {other:?}"),
    }
    assert!(!summary.outcomes[1].is_failed());
}

/// §5.7 / §16.12 item 17 — outcomes are in input order regardless of thread count, and
/// the exported bytes do not depend on parallelism.
#[test]
fn results_are_deterministic_across_thread_counts() {
    let dir = tempfile::tempdir().unwrap();
    let images: Vec<PathBuf> = ["p1.png", "p2.png", "p3.png", "p4.png"]
        .iter()
        .map(|name| common::write_page(dir.path(), name, (48, 48)))
        .collect();
    let provider = SharedDetector::new(common::detector_with_block(Rect::new(6, 6, 40, 40)));

    let mut runs = Vec::new();
    for (index, threads) in [1_usize, 4].into_iter().enumerate() {
        let mut options = common::options(
            &dir.path().join(format!("cache{index}")),
            &dir.path().join(format!("out{index}")),
        );
        options.profile.general.preferred_file_type = ".png".to_string();
        options.threads = threads;
        let summary = pc_pipeline::run_batch(&images, &options, &PipelineCtx::new(&provider));
        let order: Vec<PathBuf> = summary
            .outcomes
            .iter()
            .map(|outcome| outcome.original().clone())
            .collect();
        assert_eq!(order, images, "outcomes must be in input order");
        runs.push(
            summary
                .files_written()
                .iter()
                .map(|path| std::fs::read(path).unwrap())
                .collect::<Vec<_>>(),
        );
    }

    assert_eq!(
        runs[0], runs[1],
        "§5.7: thread count must not change output"
    );
}

/// §5.4: `--fail-fast` stops the batch; only attempted images appear, and at least one
/// of them failed.
#[test]
fn fail_fast_stops_after_the_first_failure() {
    let dir = tempfile::tempdir().unwrap();
    let images: Vec<PathBuf> = ["a.png", "b.png", "c.png", "d.png"]
        .iter()
        .map(|name| common::write_page(dir.path(), name, (48, 48)))
        .collect();
    let mut options = common::options(&dir.path().join("cache"), &dir.path().join("out"));
    options.profile.general.preferred_file_type = ".png".to_string();
    options.threads = 1;
    options.fail_fast = true;

    let provider = common::SelectiveProvider::new(&["a"], common::failing_detector())
        .with_fallback(common::detector_with_block(Rect::new(6, 6, 40, 40)));
    let summary = pc_pipeline::run_batch(&images, &options, &PipelineCtx::new(&provider));

    assert!(summary.failed() >= 1);
    assert!(
        summary.outcomes.len() < images.len(),
        "fail-fast must not attempt every image, got {} outcomes",
        summary.outcomes.len()
    );
    assert_eq!(summary.exit_code(), EXIT_PARTIAL);
}

/// §5.2: the isolation boundary lives in `process_image_isolated`, so a panic is
/// contained even for a single image outside `run_batch`.
#[test]
fn process_image_isolated_contains_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "a.png", (32, 32));
    let options = common::options(&dir.path().join("cache"), &dir.path().join("out"));

    let provider = SharedDetector::new(std::sync::Arc::new(common::PanickingDetector));
    let outcome =
        pc_pipeline::process_image_isolated(&page, &options, &PipelineCtx::new(&provider));

    match outcome {
        ImageOutcome::Failed { error, .. } => {
            assert!(error.to_string().contains("panicked"), "{error}")
        }
        other => panic!("expected Failed, got {other:?}"),
    }
}
