//! Task A4-b -- spec §16.39 items 1 and 6(b): select the coverage operand by
//! `MaskRefineMode`. FROZEN once written.
//!
//! The synthetic expectations below are hand-derived from the literal fixture:
//! block A contains 256 pixels at value 40 in the detector mask, so its unrefined
//! coverage is `40 / 255`; block B contains no detector-mask pixels, while Simple's
//! radius-3 dilation moves three 16-pixel columns into it, so its refined coverage is
//! `48 / 256 = 0.1875`. No expected value is derived from `run`'s output.

use image::{GrayImage, Luma, Rgb, RgbImage};
use pc_config::{MaskRefineMode, TextDetectorConfig};
use pc_core::{ImageHandle, Rect};
use pc_detect::{DetectInput, MockDetector, RawBlock, ReplayDetector};
use std::path::PathBuf;

const SYNTHETIC_A: Rect = Rect {
    x1: 8,
    y1: 8,
    x2: 24,
    y2: 24,
};
const SYNTHETIC_B: Rect = Rect {
    x1: 40,
    y1: 40,
    x2: 56,
    y2: 56,
};
const RECORDED_STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";
const RECORDED_ORIGINAL_PATH: &str =
    "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg";
const RECORDED_EXPECTED_RECTS: [Rect; 3] = [
    Rect {
        x1: 674,
        y1: 1397,
        x2: 740,
        y2: 1438,
    },
    Rect {
        x1: 567,
        y1: 74,
        x2: 663,
        y2: 123,
    },
    Rect {
        x1: 607,
        y1: 631,
        x2: 724,
        y2: 703,
    },
];

fn synthetic_fixture() -> (RgbImage, GrayImage, Vec<RawBlock>) {
    let page = RgbImage::from_pixel(64, 64, Rgb([230, 230, 230]));
    let mut detector_mask = GrayImage::new(64, 64);

    // A's 256 values of 40 exceed the 0.1 coverage threshold before refinement,
    // but remain below Simple's strict refinement threshold of 60.
    fill_gray(&mut detector_mask, 8, 8, 24, 24, 40);
    // This 3x16 stripe is wholly outside B. Simple's radius-3 dilation moves exactly
    // three columns (48 pixels) into B; the unrefined coverage of B remains zero.
    fill_gray(&mut detector_mask, 37, 40, 40, 56, 255);

    let blocks = vec![
        RawBlock {
            rect: SYNTHETIC_A,
            class_index: 0,
            confidence: 0.9,
        },
        RawBlock {
            rect: SYNTHETIC_B,
            class_index: 1,
            confidence: 0.8,
        },
    ];
    (page, detector_mask, blocks)
}

fn synthetic_input(page: RgbImage, mode: MaskRefineMode) -> DetectInput {
    DetectInput {
        schema_version: pc_core::SCHEMA_VERSION,
        source: ImageHandle::from_memory(image::DynamicImage::ImageRgb8(page)),
        original_path: PathBuf::from("/synthetic/a4-coverage.png"),
        target_height_lower: 1000,
        target_height_upper: 4000,
        base_image_dest: None,
        raw_mask_dest: None,
        min_mask_coverage: pc_detect::DEFAULT_MIN_MASK_COVERAGE,
        config: TextDetectorConfig {
            mask_refine_mode: mode,
            ..TextDetectorConfig::default()
        },
    }
}

fn recorded_input(mode: MaskRefineMode) -> DetectInput {
    DetectInput {
        schema_version: pc_core::SCHEMA_VERSION,
        source: ImageHandle::from_path(pc_testkit::paths::recorded(format!(
            "detector/{RECORDED_STEM}.jpg"
        ))),
        original_path: PathBuf::from(RECORDED_ORIGINAL_PATH),
        target_height_lower: 1000,
        target_height_upper: 4000,
        base_image_dest: None,
        raw_mask_dest: None,
        min_mask_coverage: pc_detect::DEFAULT_MIN_MASK_COVERAGE,
        config: TextDetectorConfig {
            mask_refine_mode: mode,
            ..TextDetectorConfig::default()
        },
    }
}

fn recorded_detector() -> ReplayDetector {
    ReplayDetector::new(
        &pc_testkit::paths::recorded_root().join("detector"),
        RECORDED_STEM,
    )
}

fn rects(output: &pc_detect::DetectOutput) -> Vec<(i32, i32, i32, i32)> {
    let mut rects = output
        .page
        .blocks
        .iter()
        .map(|block| (block.rect.x1, block.rect.y1, block.rect.x2, block.rect.y2))
        .collect::<Vec<_>>();
    rects.sort_unstable();
    rects
}

fn expected_recorded_rects() -> Vec<(i32, i32, i32, i32)> {
    let mut rects = RECORDED_EXPECTED_RECTS
        .iter()
        .map(|rect| (rect.x1, rect.y1, rect.x2, rect.y2))
        .collect::<Vec<_>>();
    rects.sort_unstable();
    rects
}

fn fill_gray(image: &mut GrayImage, x1: u32, y1: u32, x2: u32, y2: u32, value: u8) {
    for y in y1..y2 {
        for x in x1..x2 {
            image.put_pixel(x, y, Luma([value]));
        }
    }
}

/// Requirement: §16.39 item 1 and item 6(b)'s discriminating synthetic gate.
/// Annotation scores the unrefined mask and therefore keeps A at `40 / 255` while
/// dropping B at zero; Simple scores its refined mask and therefore drops A while
/// keeping B at the hand-derived `48 / 256`. Rectangle identity is asserted in both
/// directions, so equal cardinality or a survivor swap cannot satisfy the test.
///
/// Turns red if Annotation still scores its refined mask, if the operand change is
/// applied to Simple too, or if either mode returns the other rectangle.
#[test]
fn annotation_coverage_scores_the_unrefined_mask_and_simple_scores_the_refined_one() {
    let (page, detector_mask, blocks) = synthetic_fixture();
    let annotation_detector = MockDetector::new()
        .with_mask(detector_mask.clone())
        .with_blocks(blocks.clone());
    let simple_detector = MockDetector::new()
        .with_mask(detector_mask)
        .with_blocks(blocks);

    let annotation = pc_detect::run(
        synthetic_input(page.clone(), MaskRefineMode::Annotation),
        &annotation_detector,
    )
    .expect("Annotation run succeeds");
    let simple = pc_detect::run(
        synthetic_input(page, MaskRefineMode::Simple),
        &simple_detector,
    )
    .expect("Simple run succeeds");

    assert_eq!(rects(&annotation), vec![(8, 8, 24, 24)]);
    assert_eq!(annotation.page.blocks[0].mask_coverage, 40.0 / 255.0);
    assert_eq!(rects(&simple), vec![(40, 40, 56, 56)]);
    assert_eq!(simple.page.blocks[0].mask_coverage, 0.1875);
}

/// Requirement: §16.39 item 1(b), as named by item 6(b): the Annotation-only
/// operand ruling must not alter Simple. On the discriminating fixture, Simple's
/// refined operand has the literal survivor B and coverage `0.1875`; using the
/// unrefined operand in Simple would instead keep A at `40 / 255`.
///
/// Turns red specifically if the operand lift changes Simple to score the unrefined mask.
#[test]
fn simple_mode_keeps_the_refined_operand_after_the_annotation_change() {
    let (page, detector_mask, blocks) = synthetic_fixture();
    let detector = MockDetector::new()
        .with_mask(detector_mask)
        .with_blocks(blocks);

    let output = pc_detect::run(synthetic_input(page, MaskRefineMode::Simple), &detector)
        .expect("Simple run succeeds");

    assert_eq!(rects(&output), vec![(40, 40, 56, 56)]);
    assert_eq!(output.page.blocks[0].mask_coverage, 0.1875);
}

/// Requirement: §16.39 item 6(b), recorded-page no-regression lock only. The
/// committed E01P01 page is explicitly non-discriminating: both operands keep the
/// same three literal rectangles. This test is not evidence for item 1's operand
/// ruling; the synthetic test above carries that obligation.
///
/// Turns red if either mode changes the recorded survivor rectangle set.
#[test]
fn the_recorded_page_keeps_the_same_block_set_under_both_operands() {
    let annotation_detector = recorded_detector();
    let annotation = pc_detect::run(
        recorded_input(MaskRefineMode::Annotation),
        &annotation_detector,
    )
    .expect("recorded Annotation run succeeds");
    let simple_detector = recorded_detector();
    let simple = pc_detect::run(recorded_input(MaskRefineMode::Simple), &simple_detector)
        .expect("recorded Simple run succeeds");
    let expected = expected_recorded_rects();

    assert_eq!(rects(&annotation), expected);
    assert_eq!(rects(&simple), expected);
}
