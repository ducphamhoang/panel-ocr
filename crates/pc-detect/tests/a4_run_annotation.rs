//! Task A4-a -- spec §16.39 item 6(a): wire `MaskRefineMode::Annotation`
//! through `pc_detect::run`. FROZEN once written.
//!
//! Expected mask values in this file are not derived from `run`. The synthetic
//! Annotation expectations are upstream PanelCleaner oracle values already
//! recorded and frozen by `a3b_annotate_refine.rs`. The Simple regression digest
//! is the literal digest of the committed recorded mask from
//! `tests/fixtures/recorded/detector/PROVENANCE.json`.
//!
//! A4-b owns the mode-dependent coverage operand. The recorded page used here has
//! the same survivor set under the relevant operands (§16.39 item 6(b)), so its
//! literal rectangle assertion remains valid before and after A4-b.

use image::{GrayImage, Luma, Rgb, RgbImage};
use pc_config::{MaskRefineMode, TextDetectorConfig};
use pc_core::{ImageHandle, Rect, StageError};
use pc_detect::{DetectInput, MockDetector, RawBlock, ReplayDetector};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const RECORDED_STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";
const RECORDED_ORIGINAL_PATH: &str =
    "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg";
const RECORDED_HEIGHT_LOWER: u32 = 1000;
const RECORDED_HEIGHT_UPPER: u32 = 4000;

/// §16.37 item 9, retained by §16.39 items 2 and 6(a): the shipped default remains
/// `Simple`, and its recorded raw-mask PNG stays byte-identical. This literal comes
/// from committed provenance, not from the artifact under test.
const RECORDED_SIMPLE_RAW_MASK_SHA256: &str =
    "d33e5359541962c563cf89c42d9ab99c52b690a7f425f40d7a9e4535911c78d7";

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

fn memory_input(image: RgbImage, mode: MaskRefineMode) -> DetectInput {
    DetectInput {
        schema_version: pc_core::SCHEMA_VERSION,
        source: ImageHandle::from_memory(image::DynamicImage::ImageRgb8(image)),
        original_path: PathBuf::from("/synthetic/a4-page.png"),
        // No resize for these synthetic fixtures.
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

fn disk_input(
    image: RgbImage,
    mode: MaskRefineMode,
    base_dest: PathBuf,
    raw_mask_dest: PathBuf,
) -> DetectInput {
    let mut input = memory_input(image, mode);
    input.base_image_dest = Some(base_dest);
    input.raw_mask_dest = Some(raw_mask_dest);
    input
}

fn recorded_input(mode: MaskRefineMode, destinations: Option<&Path>) -> DetectInput {
    let page = pc_testkit::paths::recorded(format!("detector/{RECORDED_STEM}.jpg"));
    DetectInput {
        schema_version: pc_core::SCHEMA_VERSION,
        source: ImageHandle::from_path(page),
        original_path: PathBuf::from(RECORDED_ORIGINAL_PATH),
        target_height_lower: RECORDED_HEIGHT_LOWER,
        target_height_upper: RECORDED_HEIGHT_UPPER,
        base_image_dest: destinations.map(|dir| dir.join(format!("{RECORDED_STEM}_base.png"))),
        raw_mask_dest: destinations.map(|dir| dir.join(format!("{RECORDED_STEM}_raw_mask.png"))),
        min_mask_coverage: pc_detect::DEFAULT_MIN_MASK_COVERAGE,
        config: TextDetectorConfig {
            mask_refine_mode: mode,
            ..TextDetectorConfig::default()
        },
    }
}

fn recorded_replay_detector() -> ReplayDetector {
    ReplayDetector::new(
        &pc_testkit::paths::recorded_root().join("detector"),
        RECORDED_STEM,
    )
}

/// The upstream-oracle A3b fixture: a 30x30 page, paper value 230, a dark T at
/// `[9,9,15,11)` and `[11,11,13,19)`, detector mask 255 over `[8,8,16,20)`, and
/// detector block `[8,8,16,20]`. Upstream Annotation refinement produces exactly
/// 28 non-zero pixels with positional fingerprint 10_730; both values are already
/// independently established in `a3b_annotate_refine.rs`.
fn detected_annotation_fixture() -> (RgbImage, GrayImage, Vec<RawBlock>) {
    let mut page = RgbImage::from_pixel(30, 30, Rgb([230, 230, 230]));
    fill_rgb(&mut page, 9, 9, 15, 11, 20);
    fill_rgb(&mut page, 11, 11, 13, 19, 20);

    let mut mask = GrayImage::new(30, 30);
    fill_gray(&mut mask, 8, 8, 16, 20, 255);

    let blocks = vec![RawBlock {
        rect: Rect::new(8, 8, 16, 20),
        class_index: 0,
        confidence: 0.875,
    }];
    (page, mask, blocks)
}

/// Additive obligation grounded in §16.37 item 11(d)'s complete-algorithm
/// contract. With zero detected blocks, Part 1 is necessarily empty; the literal
/// non-empty result can only be reached through `refine_undetected_mask`.
/// Upstream's final mask has 36 non-zero pixels and fingerprint 15_192.
fn undetected_annotation_fixture() -> (RgbImage, GrayImage) {
    let mut page = RgbImage::from_pixel(40, 20, Rgb([230, 230, 230]));
    fill_rgb(&mut page, 3, 3, 6, 11, 20);
    fill_rgb(&mut page, 21, 3, 22, 18, 20);

    let mut mask = GrayImage::new(40, 20);
    fill_gray(&mut mask, 2, 2, 7, 12, 200); // 50 px: not invented.
    fill_gray(&mut mask, 20, 2, 23, 19, 200); // 51 px: invented.
    (page, mask)
}

fn output_mask(output: &pc_detect::DetectOutput) -> GrayImage {
    output
        .page
        .raw_mask
        .load()
        .expect("run output contains a loadable raw mask")
        .to_luma8()
}

fn nonzero_count(mask: &GrayImage) -> usize {
    mask.pixels().filter(|pixel| pixel.0[0] != 0).count()
}

fn positional_fingerprint(mask: &GrayImage) -> u64 {
    mask.pixels()
        .enumerate()
        .filter(|(_, pixel)| pixel.0[0] != 0)
        .map(|(index, _)| index as u64 + 1)
        .sum()
}

fn fill_rgb(image: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, value: u8) {
    for y in y1..y2 {
        for x in x1..x2 {
            image.put_pixel(x, y, Rgb([value, value, value]));
        }
    }
}

fn fill_gray(image: &mut GrayImage, x1: u32, y1: u32, x2: u32, y2: u32, value: u8) {
    for y in y1..y2 {
        for x in x1..x2 {
            image.put_pixel(x, y, Luma([value]));
        }
    }
}

/// Requirement: §16.39 item 6(a), revising §16.5 item 3 and §8.3 step 5.
/// `mask_refine_mode = Annotation` is the opt-in itself; `run` must invoke the
/// Annotation algorithm rather than refuse the mode or silently execute Simple.
/// This replaces `d7_run.rs::run_rejects_annotation_refine_mode` under §16.39
/// item 3(d)'s joint-architect cookbook-rule-8 ruling.
///
/// Turns red if the refusal remains, Annotation routes through Simple on this
/// fixture, or the Annotation mask has the wrong dimensions, population, or
/// positional fingerprint.
#[test]
fn run_in_annotation_mode_refines_with_the_annotation_path_instead_of_refusing() {
    let (page, pred_mask, blocks) = detected_annotation_fixture();

    let annotation_detector = MockDetector::new()
        .with_blocks(blocks.clone())
        .with_mask(pred_mask.clone());
    let annotation = pc_detect::run(
        memory_input(page.clone(), MaskRefineMode::Annotation),
        &annotation_detector,
    )
    .expect("Annotation is an accepted opt-in mode");

    let simple_detector = MockDetector::new().with_blocks(blocks).with_mask(pred_mask);
    let simple = pc_detect::run(memory_input(page, MaskRefineMode::Simple), &simple_detector)
        .expect("Simple remains supported");

    let annotation_mask = output_mask(&annotation);
    let simple_mask = output_mask(&simple);
    assert_eq!(annotation_detector.calls(), 1);
    assert_eq!(annotation_mask.dimensions(), (30, 30));
    assert_eq!(nonzero_count(&annotation_mask), 28);
    assert_eq!(positional_fingerprint(&annotation_mask), 10_730);
    assert_ne!(
        annotation_mask.as_raw(),
        simple_mask.as_raw(),
        "accepting Annotation by silently running Simple is not sufficient"
    );
}

/// Requirement: §16.37 item 9, retained by §16.39 items 2 and 6(a). `Simple`
/// stays the shipped default and every value frozen against it stays unchanged.
/// The expected digest is the hard-coded provenance digest, not a value derived
/// from `run` or from the committed mask at test time.
///
/// Turns red if the default changes, Simple routes through Annotation, or A4
/// changes the recorded Simple mask's encoded bytes.
#[test]
fn run_in_simple_mode_is_byte_identical_to_the_committed_recorded_mask() {
    assert_eq!(
        TextDetectorConfig::default().mask_refine_mode,
        MaskRefineMode::Simple,
        "the shipped default itself is part of the requirement"
    );

    let detector = recorded_replay_detector();
    let scratch = tempfile::TempDir::new().expect("temp dir");
    let output = pc_detect::run(
        recorded_input(MaskRefineMode::Simple, Some(scratch.path())),
        &detector,
    )
    .expect("default recorded-page detection succeeds");

    let produced_path = output
        .page
        .raw_mask
        .path
        .as_ref()
        .expect("disk-mode raw mask has a path");
    let produced_bytes = std::fs::read(produced_path).expect("produced mask is readable");
    let actual = format!("{:x}", Sha256::digest(&produced_bytes));
    assert_eq!(actual, RECORDED_SIMPLE_RAW_MASK_SHA256);
}

/// Additive obligation grounded in §16.37 item 11(d)'s complete-algorithm
/// contract and §16.39 item 6(a)'s required second call. There are zero detector
/// blocks, so the first pass necessarily returns an empty mask. The non-empty
/// literal result can only be reached through the undetected-mask pass.
///
/// Turns red if that pass is omitted or guarded on a non-empty detected list.
/// The aggregate oracle values additionally catch many, but not all, wrong
/// implementations of the second pass; they are not an exact bitmap oracle.
#[test]
fn run_in_annotation_mode_reaches_the_undetected_pass_with_zero_detected_blocks() {
    let (page, pred_mask) = undetected_annotation_fixture();
    let detector = MockDetector::new().with_mask(pred_mask);

    let output = pc_detect::run(memory_input(page, MaskRefineMode::Annotation), &detector)
        .expect("zero detected blocks do not skip Annotation's undetected pass");

    let mask = output_mask(&output);
    assert_eq!(detector.calls(), 1);
    assert_eq!(output.analytics.blocks_detected, 0);
    assert_eq!(output.analytics.blocks_kept, 0);
    assert!(output.page.blocks.is_empty());
    assert_eq!(mask.dimensions(), (40, 20));
    assert_eq!(nonzero_count(&mask), 36);
    assert_eq!(positional_fingerprint(&mask), 15_192);
}

/// Requirement: §16.39 items 4 and 6(a). An empty expanded Annotation window is
/// per-image `StageError::InvalidInput`, and `run` must propagate it rather than
/// dropping the malformed block.
///
/// Turns red if the block is skipped, the error is swallowed, or its class is
/// changed.
#[test]
fn run_in_annotation_mode_propagates_an_empty_expanded_window_as_invalid_input() {
    let page = RgbImage::from_pixel(20, 20, Rgb([230, 230, 230]));
    let pred_mask = GrayImage::new(20, 20);
    let detector = MockDetector::new()
        .with_blocks(vec![RawBlock {
            rect: Rect::new(5, 5, 5, 5),
            class_index: 0,
            confidence: 0.9,
        }])
        .with_mask(pred_mask);

    let error = pc_detect::run(memory_input(page, MaskRefineMode::Annotation), &detector)
        .expect_err("the malformed block must fail the image");

    assert_eq!(
        detector.calls(),
        1,
        "the error comes from Annotation refinement after detection"
    );
    match error {
        StageError::InvalidInput(message) => assert!(
            message.contains("expands to the empty window"),
            "run must propagate the specific refinement failure: {message}"
        ),
        other => panic!("expected per-image InvalidInput, got {other:?}"),
    }
}

/// Additive obligation grounded in §4.1's disk artifact contract, applied to the
/// new §16.39 item 6(a) branch. This checks the artifact at its destination, not
/// merely the in-memory handle.
///
/// Turns red if Annotation is not written, the output handle names another path,
/// or the destination decodes to pixels different from the emitted handle. The
/// aggregate oracle checks are supplementary diagnostics, not exact identity.
#[test]
fn run_in_annotation_mode_writes_the_refined_raw_mask_to_its_destination() {
    let (page, pred_mask, blocks) = detected_annotation_fixture();
    let detector = MockDetector::new().with_blocks(blocks).with_mask(pred_mask);

    let scratch = tempfile::TempDir::new().expect("temp dir");
    let base_dest = scratch.path().join("annotation_base.png");
    let raw_mask_dest = scratch.path().join("annotation_raw_mask.png");
    let output = pc_detect::run(
        disk_input(
            page,
            MaskRefineMode::Annotation,
            base_dest.clone(),
            raw_mask_dest.clone(),
        ),
        &detector,
    )
    .expect("Annotation disk-mode run succeeds");

    assert!(base_dest.is_file());
    assert!(raw_mask_dest.is_file());
    assert_eq!(
        output.page.raw_mask.path.as_deref(),
        Some(raw_mask_dest.as_path())
    );

    let written = image::open(&raw_mask_dest)
        .expect("written Annotation mask decodes")
        .to_luma8();
    let emitted = output_mask(&output);
    assert_eq!(
        written, emitted,
        "the destination must decode to the mask emitted by the output handle"
    );
    assert_eq!(written.dimensions(), (30, 30));
    assert_eq!(nonzero_count(&written), 28);
    assert_eq!(positional_fingerprint(&written), 10_730);
}

/// Requirement: §16.39 item 6(a), under §16.37 item 3's ours-vs-ours
/// determinism framing. Expected rectangles are literals, not loaded from the
/// detector fixture or `#raw.json`; identity is asserted as an order-independent
/// coordinate collection, not merely cardinality. The aggregate mask literals
/// assert dimensions, non-zero population, and positional sum; they are not an
/// exact bitmap oracle.
///
/// Turns red if Annotation remains refused, survivor identity changes, or any of
/// those recorded-page aggregate mask values changes.
#[test]
fn run_in_annotation_mode_on_the_recorded_page_produces_the_expected_block_rects() {
    let detector = recorded_replay_detector();
    let output = pc_detect::run(recorded_input(MaskRefineMode::Annotation, None), &detector)
        .expect("recorded page runs in Annotation mode");

    assert_eq!(detector.calls(), 1);
    assert_eq!(output.analytics.blocks_detected, 4);
    assert_eq!(output.analytics.blocks_kept, 3);
    let mut actual_rects = output
        .page
        .blocks
        .iter()
        .map(|block| (block.rect.x1, block.rect.y1, block.rect.x2, block.rect.y2))
        .collect::<Vec<_>>();
    let mut expected_rects = RECORDED_EXPECTED_RECTS
        .iter()
        .map(|rect| (rect.x1, rect.y1, rect.x2, rect.y2))
        .collect::<Vec<_>>();
    actual_rects.sort_unstable();
    expected_rects.sort_unstable();
    assert_eq!(actual_rects, expected_rects);

    let mask = output_mask(&output);
    assert_eq!(mask.dimensions(), (1200, 1660));
    assert_eq!(nonzero_count(&mask), 2_942);
    assert_eq!(positional_fingerprint(&mask), 2_790_551_277);
}
