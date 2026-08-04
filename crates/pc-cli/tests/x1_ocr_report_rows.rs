//! Task **X1-ocr-outcome** — the rows `panel-ocr ocr` actually writes (spec §13.1, §12.6,
//! §16.11 items 11 and 12, §15.5 item 5).
//!
//! `pc-cli::ocr_report` is the seam under test, reached through the real
//! `pc_cli::ocr::apply_report_overrides` and the real `pc_pipeline::run_batch`, because the
//! reproduced defect lived in the *join* of those three: the overrides emptied
//! `page.text_boxes` by design, the pipeline read that as `NoTextDetected`, and
//! `ocr_report` collects analytics from `ImageOutcome::Completed` only — so the report came
//! out empty for every page that had text. A test of any one of the three passes while the
//! composition is broken, which is what
//! `pc-cli/src/ocr.rs::report_profile_overrides_disable_report_filtering` demonstrated: it
//! asserts the two override values and nothing about the rows they produce.
//!
//! The `onnx` feature is not needed and not used: the OCR engine is `pc_ocr::MockOcrFactory`
//! and the detector is `pc_detect::MockDetector`, per §7.2's mocked-boundary pattern.
//!
//! FROZEN (CLAUDE.md).

use image::{Rgb, RgbImage};
use pc_config::Profile;
use pc_core::Rect;
use pc_detect::{MockDetector, RawBlock};
use pc_ocr::{MockOcrEngine, MockOcrFactory};
use pc_pipeline::{ImageOutcome, PipelineCtx, PipelineOptions, SharedDetector};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn write_page(dir: &Path, name: &str) -> PathBuf {
    let mut image = RgbImage::from_pixel(200, 200, Rgb([255, 255, 255]));
    for y in 20..60 {
        for x in 20..80 {
            image.put_pixel(x, y, Rgb([16, 16, 16]));
        }
    }
    let path = dir.join(name);
    image
        .save_with_format(&path, image::ImageFormat::Png)
        .unwrap();
    path
}

/// Two `ja`-class blocks (`class_index: 1`), each 60 x 40 = 2400 px² — above
/// `box_min_size` (400) and, once the §15.5 overrides are applied, below
/// `ocr_max_size` (10^10), so both are OCR candidates.
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

/// `run_ocr`'s own option set, minus the parts that need a model: the profile goes through
/// the real `apply_report_overrides`, and `performing_ocr` is `true` as `run_ocr` sets it.
fn ocr_options(cache_dir: &Path) -> PipelineOptions {
    let mut profile = Profile::default();
    pc_cli::ocr::apply_report_overrides(&mut profile);
    PipelineOptions {
        threads: 1,
        profile,
        cache_dir: cache_dir.to_path_buf(),
        performing_ocr: true,
        ..PipelineOptions::default()
    }
}

/// §12.6 / §16.11 item 12, hand-derived — never read off a run.
///
/// Header verbatim from §12.6. `filename` is `analytic.path.file_name()` (§16.11 item 11).
/// The page is 200 x 200 against `input_height_upper_target = 4000`, so §8.3 step 2 gives
/// `scale == 1.0` and §9.3 step 7's `rect.scale(1.0 / scale)` is the identity; §9.3 step
/// 5's tight tier is `pad(2)` then `right_pad(3)`, so `(20,20,80,60)` → `(18,18,85,62)` and
/// `(20,100,80,140)` → `(18,98,85,142)`. §9.3 step 6's right-to-left key (`-0.4*x1 + y1`,
/// the page language is Japanese) orders them 10.8 before 90.8, and that is also the order
/// the scripted engine answers in. Neither text needs RFC 4180 quoting.
const EXPECTED_CSV: &str = "filename,startx,starty,endx,endy,text\n\
page01.png,18,18,85,62,ハロー\n\
page01.png,18,98,85,142,ワールド\n";

/// The defect, stated as the artifact: a page with detected text must produce report rows.
#[test]
fn the_csv_report_carries_one_row_per_ocred_box_for_a_page_with_text() {
    let dir = tempfile::tempdir().unwrap();
    let page = write_page(dir.path(), "page01.png");
    let options = ocr_options(&dir.path().join("cache"));

    let provider = SharedDetector::new(Arc::new(
        MockDetector::new()
            .with_blocks(two_blocks())
            .with_block_fill(255),
    ));
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_script(["ハロー", "ワールド"]));
    let ctx = PipelineCtx::new(&provider).with_ocr(&factory);

    let summary = pc_pipeline::run_batch(std::slice::from_ref(&page), &options, &ctx);
    let csv = pc_cli::ocr_report(&summary, pc_export::ReportFormat::Csv);

    assert_eq!(csv, EXPECTED_CSV);
}

/// §12.6's TXT form of the same run: the two recognised strings reach the report, in
/// reading order, under the bare file name.
#[test]
fn the_txt_report_carries_both_recognised_strings_for_a_page_with_text() {
    let dir = tempfile::tempdir().unwrap();
    let page = write_page(dir.path(), "page01.png");
    let options = ocr_options(&dir.path().join("cache"));

    let provider = SharedDetector::new(Arc::new(
        MockDetector::new()
            .with_blocks(two_blocks())
            .with_block_fill(255),
    ));
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_script(["ハロー", "ワールド"]));
    let ctx = PipelineCtx::new(&provider).with_ocr(&factory);

    let summary = pc_pipeline::run_batch(std::slice::from_ref(&page), &options, &ctx);
    let txt = pc_cli::ocr_report(&summary, pc_export::ReportFormat::Txt);

    assert_eq!(txt, "page01.png: \nハロー\nワールド\n");
}

/// §16.11 item 12's rule, preserved: with no boxes at all the report is empty —
/// *"no OCR was run" must not look like "OCR found nothing"* — and the page is a skip, so
/// no analytic exists to render.
#[test]
fn a_page_with_no_detected_boxes_renders_an_empty_report() {
    let dir = tempfile::tempdir().unwrap();
    let page = write_page(dir.path(), "page01.png");
    let options = ocr_options(&dir.path().join("cache"));

    let provider = SharedDetector::new(Arc::new(MockDetector::new()));
    let factory = MockOcrFactory::new(MockOcrEngine::new().with_script(["ハロー"]));
    let ctx = PipelineCtx::new(&provider).with_ocr(&factory);

    let summary = pc_pipeline::run_batch(std::slice::from_ref(&page), &options, &ctx);

    assert_eq!(summary.skipped(), 1, "premise: the page had no boxes");
    assert_eq!(
        pc_cli::ocr_report(&summary, pc_export::ReportFormat::Csv),
        ""
    );
    assert_eq!(
        pc_cli::ocr_report(&summary, pc_export::ReportFormat::Txt),
        ""
    );
}

/// §16.34 item 5's decided behaviour: `pc-cli::ocr_report` skips an `OcrAnalytic` whose
/// `removed` is empty, so a page whose every box failed OCR renders `""` — not
/// `write_csv`'s bare header row and not `write_txt`'s `"page01.png: \n"` phantom
/// path-header with no lines. Grounded in upstream's per-box (not per-page) report shape:
/// `pcleaner/ocr/ocr.py`'s `format_output_plain` writes the path header only *inside* the
/// per-box loop, so a page contributing zero boxes contributes no header.
///
/// `MockOcrEngine::failing()` returns `StageError::Inference` for every crop, which
/// DEVIATION(9) / §9.3 step 7 handles fail-open: the box is kept, so nothing lands in
/// `removed` while `num_boxes` still counts both boxes. The three premise assertions pin
/// that shape — this is the one case in §16.34 that changes previously-observable output,
/// so the test must prove it started from the reachable state item 5 describes, not from a
/// hand-built analytic.
///
/// Measured at `HEAD` with a standalone `pc-pipeline` probe (2 detected boxes,
/// `performing_ocr = true`, failing engine): `num_boxes=2`, `removed=[]`,
/// `ImageOutcome::Completed`, CSV `"filename,startx,starty,endx,endy,text\n"`, TXT
/// `"page01.png: \n"`. Both expected strings below are `""`, hard-coded, so the pre-fix
/// output above is what turns this red.
///
/// §16.34 item 2's predicate does not move this page: `num_boxes == 0` is false here, so it
/// stays `Completed` exactly as it is today — the fix is `ocr_report`'s alone.
#[test]
fn a_page_whose_every_box_failed_ocr_renders_an_empty_report_not_a_bare_header() {
    let dir = tempfile::tempdir().unwrap();
    let page = write_page(dir.path(), "page01.png");
    let options = ocr_options(&dir.path().join("cache"));

    let provider = SharedDetector::new(Arc::new(
        MockDetector::new()
            .with_blocks(two_blocks())
            .with_block_fill(255),
    ));
    let factory = MockOcrFactory::new(MockOcrEngine::new().failing());
    let ctx = PipelineCtx::new(&provider).with_ocr(&factory);

    let summary = pc_pipeline::run_batch(std::slice::from_ref(&page), &options, &ctx);

    // Premise, all three parts: the page COMPLETED (so `ocr_report` reaches its analytic
    // at all), two boxes entered step 7, and every one of them survived OCR failure.
    let ocr = match summary.outcomes.as_slice() {
        [ImageOutcome::Completed { analytics, .. }] => analytics
            .ocr
            .as_ref()
            .expect("premise: the OCR pass ran, so the analytic is Some"),
        other => panic!("premise: one completed page; got {other:?}"),
    };
    assert_eq!(
        ocr.num_boxes, 2,
        "premise: both boxes entered step 7 (§16.8 item 11)"
    );
    assert!(
        ocr.removed.is_empty(),
        "premise: every recognize() failed and the box was kept (fail-open), got {:?}",
        ocr.removed
    );

    assert_eq!(
        pc_cli::ocr_report(&summary, pc_export::ReportFormat::Csv),
        ""
    );
    assert_eq!(
        pc_cli::ocr_report(&summary, pc_export::ReportFormat::Txt),
        ""
    );
}
