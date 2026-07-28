//! Task **G1**, v1 review finding 4 — an unsupported input format is a graceful skip.
//!
//! §5.1 lists `Skipped { reason }` with "e.g. unsupported format" and §5.5 makes exit code
//! `0` mean "all images completed (skips allowed)"; §5.6 keeps genuine non-errors out of
//! the `Failed` bucket. `discovery::expand_inputs` deliberately passes an explicitly named
//! file through even when its suffix is unsupported, "so a typo'd file is visible in the
//! summary rather than silently dropped" — which only holds if the per-image boundary then
//! reports it as `Skipped { UnsupportedFormat }`. Before this fix it reached the detector
//! and came back `Failed at Detect: failed to decode image`, so one stray `notes.xyz`
//! turned an otherwise-successful batch into exit code 2.
//!
//! FROZEN (CLAUDE.md).

mod common;

use common::{ctx, detector_with_block, options, write_page};
use pc_core::Rect;
use pc_pipeline::{ImageOutcome, SharedDetector, SkipReason, EXIT_OK};

/// §5.1/§5.5/§5.6: the stray file is `Skipped { UnsupportedFormat }` naming its suffix, the
/// real page still completes and exports, and the batch exit code is success.
#[test]
fn an_unsupported_file_is_skipped_and_the_batch_still_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let page = write_page(dir.path(), "page01.png", (64, 64));
    let stray = dir.path().join("notes.xyz");
    std::fs::write(&stray, b"this is not an image").unwrap();

    let options = options(&dir.path().join("cache"), &dir.path().join("out"));
    let provider = SharedDetector::new(detector_with_block(Rect::new(8, 8, 48, 48)));
    let ctx = ctx(&provider);

    let summary = pc_pipeline::run_batch(&[page.clone(), stray.clone()], &options, &ctx);

    assert_eq!(summary.outcomes.len(), 2, "one outcome per input (§5.7)");

    match &summary.outcomes[1] {
        ImageOutcome::Skipped {
            original,
            reason,
            files_written,
        } => {
            assert_eq!(original, &stray);
            assert_eq!(
                reason,
                &SkipReason::UnsupportedFormat {
                    suffix: ".xyz".to_string()
                },
                "the skip must name the offending suffix"
            );
            assert!(
                files_written.is_empty(),
                "an unreadable input exports nothing: {files_written:?}"
            );
        }
        other => panic!("expected Skipped {{ UnsupportedFormat }}, got {other:?}"),
    }

    assert!(
        summary.outcomes[0].is_completed(),
        "the valid page must still process: {:?}",
        summary.outcomes[0]
    );
    assert!(
        !summary.outcomes[0].files_written().is_empty(),
        "the valid page must still export"
    );

    assert_eq!(
        summary.failed(),
        0,
        "an unsupported format is not a failure"
    );
    assert_eq!(summary.skipped(), 1);
    assert_eq!(summary.completed(), 1);
    assert_eq!(
        summary.exit_code(),
        EXIT_OK,
        "nothing failed, so §5.5 requires exit code 0; got:\n{}",
        summary.render()
    );
    assert!(
        summary.render().contains("notes.xyz"),
        "the summary must name the skipped file (§5.5):\n{}",
        summary.render()
    );
}

/// The skip happens *before* the detector is consulted, so it does not depend on how the
/// backend happens to fail: a suffix-less file is skipped too, and the detector-refusing
/// path is never reached.
#[test]
fn the_check_precedes_detection_and_covers_a_suffixless_file() {
    let dir = tempfile::tempdir().unwrap();
    let stray = dir.path().join("README");
    std::fs::write(&stray, b"no suffix at all").unwrap();

    let options = options(&dir.path().join("cache"), &dir.path().join("out"));
    let provider = common::RefusingProvider;
    let ctx = ctx(&provider);

    let outcome = pc_pipeline::process_image_with_splitting(&stray, &options, &ctx);
    match outcome {
        ImageOutcome::Skipped { reason, .. } => assert_eq!(
            reason,
            SkipReason::UnsupportedFormat {
                suffix: String::new()
            },
            "no extension => an empty suffix, still an UnsupportedFormat skip"
        ),
        other => panic!("expected Skipped {{ UnsupportedFormat }}, got {other:?}"),
    }
}

/// The counterexample that keeps the skip narrow: a `.png` whose *bytes* are not an image
/// is still a real `Failed` at `Detect` (§5.6's closed skip set — everything else fails).
#[test]
fn a_corrupt_supported_file_still_fails() {
    let dir = tempfile::tempdir().unwrap();
    let broken = dir.path().join("broken.png");
    std::fs::write(&broken, b"not an image at all").unwrap();

    let options = options(&dir.path().join("cache"), &dir.path().join("out"));
    let provider = SharedDetector::new(common::empty_detector());
    let ctx = ctx(&provider);

    let summary = pc_pipeline::run_batch(&[broken], &options, &ctx);
    assert_eq!(summary.failed(), 1, "{}", summary.render());
    assert_ne!(summary.exit_code(), EXIT_OK);
}
