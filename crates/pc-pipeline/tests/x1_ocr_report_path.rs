//! Task **X1-ocr-outcome** — `panel-ocr ocr`'s report path must not be misread as
//! "no text detected" (spec §13.1, §5 item 6, §16.8 item 9, §16.11 item 11, §15 item 5).
//!
//! §15 item 5 / §16.11 item 11 make the report path's emptied `text_boxes` a *design*:
//! `run_ocr` forces `ocr_blacklist_pattern = ".*"` and `ocr_max_size = 10**10`, so every
//! detected box is OCR'd and moved into `OcrAnalytic.removed` — "in the report code path
//! *every* box is 'removed' and `removed` is the complete, ordered box list."
//!
//! §5 item 6's skip trigger is a different population: *"zero boxes on a page →
//! `Skipped { NoTextDetected }`"*. These tests pin the distinction — boxes existed and were
//! consumed by the report pass (Completed, analytic carried) versus no boxes at all
//! (Skipped) — and pin that the `clean` path's reading of an empty `text_boxes` is unchanged.
//!
//! FROZEN (CLAUDE.md).

mod common;

use pc_core::{Rect, RemovedBox};
use pc_detect::{MockDetector, RawBlock};
use pc_ocr::{MockOcrEngine, MockOcrFactory};
use pc_pipeline::{ImageOutcome, PipelineCtx, PipelineOptions, SharedDetector, SkipReason};
use std::path::Path;
use std::sync::Arc;

/// The two detector blocks both tests with text use, in detector order.
///
/// `class_index: 1` is the model's `ja` class (`pc_detect::yolo::class_to_language`), so
/// neither box is language-unknown and §9.3 step 2's `suspicious_box_min_size` rule does
/// not apply; each block's area is 60 x 40 = 2400, above `box_min_size` (400).
fn two_blocks() -> Vec<RawBlock> {
    vec![
        RawBlock {
            rect: Rect::new(20, 20, 80, 60),
            class_index: 1,
            confidence: 0.9,
        },
        RawBlock {
            rect: Rect::new(20, 100, 80, 140),
            class_index: 1,
            confidence: 0.9,
        },
    ]
}

/// The two boxes as they reach `OcrAnalytic.removed`, hand-derived from the spec, not read
/// off a run:
///
/// * the page is 200 x 200 and `input_height_upper_target` is 4000, so §8.3 step 2's
///   `height <= target_upper` branch gives `scale == 1.0` and `image_size == (200, 200)`:
///   the detector rects are already original-image coordinates and §9.3 step 7's
///   `rect.scale(1.0 / scale)` is the identity;
/// * §9.3 step 5's tight tier is `pad(box_padding_initial = 2)` then
///   `right_pad(box_right_padding_initial = 3)`, i.e. `(20,20,80,60)` → `(18,18,82,62)` →
///   `(18,18,85,62)`, and likewise `(20,100,80,140)` → `(18,98,85,142)`. Nothing clamps:
///   the canvas is 200 x 200;
/// * §9.3 step 6's key is `-0.4*x1 + y1` for a Japanese (right-to-left) page, so the sort
///   order is 10.8 then 90.8 — the upper box first, which is also the OCR call order the
///   scripted engine below answers in.
fn expected_removed() -> Vec<RemovedBox> {
    vec![
        RemovedBox {
            text: "ハロー".to_string(),
            rect: Rect::new(18, 18, 85, 62),
        },
        RemovedBox {
            text: "ワールド".to_string(),
            rect: Rect::new(18, 98, 85, 142),
        },
    ]
}

/// `common::options` with the OCR pass live and §15 item 5's report-path overrides applied by
/// hand (the `pc-cli` half is gated separately, in `pc-cli/tests/x1_ocr_report_rows.rs`).
fn ocr_pass_options(cache: &Path, out: &Path) -> PipelineOptions {
    let mut options = common::options(cache, out);
    options.profile.general.preferred_file_type = ".png".to_string();
    options.profile.preprocessor.ocr_enabled = true;
    options.profile.preprocessor.ocr_blacklist_pattern = ".*".to_string();
    options.profile.preprocessor.ocr_max_size = 10_000_000_000;
    options
}

fn scripted_factory() -> MockOcrFactory {
    MockOcrFactory::new(MockOcrEngine::new().with_script(["ハロー", "ワールド"]))
}

/// §13.1 + §16.11 item 11: an `ocr` run over a page whose boxes were all consumed by the
/// report pass is `Completed`, and carries the analytic the report is rendered from.
///
/// This is the assertion the reproduced defect fails: `single.rs` derived `no_text` from
/// `page.text_boxes.is_empty()`, which the report path empties by design, so the image was
/// reported as `Skipped { NoTextDetected }` and its OCR analytic was dropped with the
/// outcome variant.
#[test]
fn an_ocr_run_over_two_detected_boxes_completes_and_carries_both_boxes_in_its_analytic() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (200, 200));
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let mut options = ocr_pass_options(&cache, &out);
    options.performing_ocr = true;

    let provider = SharedDetector::new(Arc::new(
        MockDetector::new()
            .with_blocks(two_blocks())
            .with_block_fill(255),
    ));
    let factory = scripted_factory();
    let ctx = PipelineCtx::new(&provider).with_ocr(&factory);

    let outcome = pc_pipeline::process_image(&page, &options, &ctx);

    match &outcome {
        ImageOutcome::Completed {
            files_written,
            analytics,
            ..
        } => {
            // §13.1: `ocr` has no `--output-dir` and exports nothing.
            assert!(
                files_written.is_empty(),
                "an `ocr` run exports nothing, got {files_written:?}"
            );
            let ocr = analytics
                .ocr
                .as_ref()
                .expect("§16.8 item 3: the OCR pass ran, so the analytic is Some");
            // Boxes considered before removal — the population §5 item 6 counts.
            assert_eq!(ocr.num_boxes, 2, "two blocks were detected and kept");
            // The whole vector, not its length: a count cannot tell a swapped or
            // truncated box list from the right one.
            assert_eq!(ocr.removed, expected_removed());
        }
        other => {
            panic!("§13.1: boxes were detected and OCR'd, so the image completed; got {other:?}")
        }
    }
}

/// §5 item 6, unchanged for the report path: *zero boxes on a page* is still
/// `Skipped { NoTextDetected }`, and an `ocr` run still exports nothing.
///
/// The `run_stages` call pins the premise this test depends on — that the OCR pass saw
/// nothing, rather than seeing boxes and discarding them — so the assertion below cannot
/// pass for the wrong reason.
#[test]
fn an_ocr_run_over_a_page_with_no_detected_boxes_is_skipped_as_no_text_detected() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (200, 200));
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let mut options = ocr_pass_options(&cache, &out);
    options.performing_ocr = true;

    let provider = SharedDetector::new(common::empty_detector());
    let factory = scripted_factory();
    let ctx = PipelineCtx::new(&provider).with_ocr(&factory);

    std::fs::create_dir_all(&cache).unwrap();
    let chain = pc_pipeline::run_stages(&page, None, &options, &ctx).expect("stages 1-2 succeed");
    let ocr = chain
        .analytics
        .ocr
        .as_ref()
        .expect("§16.8 item 3: the pass ran on an empty box list");
    assert_eq!(
        ocr.num_boxes, 0,
        "premise: the OCR pass had no boxes to consume"
    );
    assert!(ocr.removed.is_empty(), "premise: nothing was OCR'd");

    let outcome = pc_pipeline::process_image(&page, &options, &ctx);

    match &outcome {
        ImageOutcome::Skipped {
            reason,
            files_written,
            ..
        } => {
            assert_eq!(*reason, SkipReason::NoTextDetected);
            assert!(
                files_written.is_empty(),
                "an `ocr` run exports nothing, got {files_written:?}"
            );
        }
        other => panic!("§5 item 6: zero boxes is a skip; got {other:?}"),
    }
}

/// §5 item 6 for the `clean` path, which must be untouched: with `performing_ocr =
/// false` a page whose every box was discarded by the OCR filter has nothing left to
/// mask, so it is `Skipped { NoTextDetected }` **and still exported**.
///
/// This is the control that stops the fix from being widened past §13.1's report path: a
/// rule keyed on "the OCR analytic removed some boxes" rather than on
/// `options.performing_ocr` would turn this page into `Completed` and contradict
/// `g1_chain.rs`'s `a_page_without_text_is_skipped_but_still_exported`.
#[test]
fn a_clean_run_whose_ocr_filter_discarded_every_box_is_still_skipped_and_exported() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (200, 200));
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let options = ocr_pass_options(&cache, &out);
    assert!(
        !options.performing_ocr,
        "premise: this is the `clean` path (§13.1's `clean`, not `ocr`)"
    );

    let provider = SharedDetector::new(Arc::new(
        MockDetector::new()
            .with_blocks(two_blocks())
            .with_block_fill(255),
    ));
    let factory = scripted_factory();
    let ctx = PipelineCtx::new(&provider).with_ocr(&factory);

    let outcome = pc_pipeline::process_image(&page, &options, &ctx);

    match &outcome {
        ImageOutcome::Skipped {
            reason,
            files_written,
            ..
        } => {
            assert_eq!(*reason, SkipReason::NoTextDetected);
            assert!(
                files_written.contains(&out.join("page01_clean.png")),
                "§5 item 6: the untouched page is still exported, got {files_written:?}"
            );
        }
        other => {
            panic!("§5 item 6: a `clean` page with no surviving boxes is skipped; got {other:?}")
        }
    }
}

/// §5.5: the summary a user reads must not call a successful `ocr` run a skip.
///
/// The literal line is the reproduced defect's user-visible symptom ("0 completed, 1
/// skipped … no text detected") stated in its correct form; it is hard-coded, not
/// computed from the outcome list.
#[test]
fn an_ocr_batch_over_one_page_with_text_summarises_as_one_completed_zero_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let page = common::write_page(dir.path(), "page01.png", (200, 200));
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let mut options = ocr_pass_options(&cache, &out);
    options.performing_ocr = true;

    let provider = SharedDetector::new(Arc::new(
        MockDetector::new()
            .with_blocks(two_blocks())
            .with_block_fill(255),
    ));
    let factory = scripted_factory();
    let ctx = PipelineCtx::new(&provider).with_ocr(&factory);

    let summary = pc_pipeline::run_batch(std::slice::from_ref(&page), &options, &ctx);

    assert_eq!(
        summary.render().lines().next(),
        Some("1 completed, 0 skipped, 0 failed")
    );
    assert_eq!(summary.exit_code(), pc_pipeline::EXIT_OK);
}
