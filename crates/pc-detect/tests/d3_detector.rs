//! Task D3 -- spec §8.2 (`TextDetector` / `RawDetection`) and §7.2 / §7.2.1 (the
//! mocked ML boundary). FROZEN per CLAUDE.md.

mod common;

use common::{synthetic_blocks, synthetic_page, synthetic_raw_mask, synthetic_replay_fixture};
use image::{GrayImage, Luma};
use pc_core::{Language, Rect, StageError};
use pc_detect::{MockDetector, RawBlock, ReplayDetector, TextDetector};

// ------------------------------------------------------------ trait shape

/// Compiles only if `TextDetector` is object safe -- the whole `Ctx<'a> = &'a dyn
/// TextDetector` contract (spec §3) depends on it.
fn assert_object_safe(_detector: &dyn TextDetector) {}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn text_detector_is_object_safe_and_shareable() {
    // §4.5 shares one detector across rayon threads, so `Send + Sync` is load-bearing.
    assert_send_sync::<MockDetector>();
    assert_send_sync::<ReplayDetector>();
    assert_object_safe(&MockDetector::new());

    let boxed: Box<dyn TextDetector> = Box::new(MockDetector::new());
    assert_object_safe(boxed.as_ref());
}

// ------------------------------------------------------------ MockDetector

#[test]
fn mock_detector_counts_calls() {
    let detector = MockDetector::new();
    let image = synthetic_page(16, 16);

    assert_eq!(detector.calls(), 0);
    for expected in 1..=3 {
        detector.detect(&image).expect("mock succeeds");
        assert_eq!(detector.calls(), expected);
    }
}

#[test]
fn mock_detector_returns_the_configured_blocks() {
    let blocks = synthetic_blocks();
    let detector = MockDetector::new().with_blocks(blocks.clone());

    let detection = detector
        .detect(&synthetic_page(64, 64))
        .expect("mock succeeds");

    assert_eq!(detection.blocks, blocks);
}

#[test]
fn mock_detector_mask_defaults_to_the_image_size() {
    let detector = MockDetector::new();

    let detection = detector
        .detect(&synthetic_page(23, 41))
        .expect("mock succeeds");

    assert_eq!(detection.mask.dimensions(), (23, 41));
    assert!(detection.mask.pixels().all(|pixel| pixel.0[0] == 0));
}

#[test]
fn mock_detector_mask_fill_sets_the_background() {
    let detector = MockDetector::new().with_mask_fill(77);

    let detection = detector
        .detect(&synthetic_page(8, 8))
        .expect("mock succeeds");

    assert!(detection.mask.pixels().all(|pixel| pixel.0[0] == 77));
}

#[test]
fn mock_detector_block_fill_paints_exclusive_rects() {
    // spec §16.6 item 5: x2/y2 are exclusive, matching `Rect::to_crop`.
    let block = RawBlock {
        rect: Rect::new(1, 1, 3, 3),
        class_index: 0,
        confidence: 1.0,
    };
    let detector = MockDetector::new()
        .with_blocks(vec![block])
        .with_mask_fill(0)
        .with_block_fill(255);

    let detection = detector
        .detect(&synthetic_page(4, 4))
        .expect("mock succeeds");

    assert_eq!(detection.mask.get_pixel(1, 1).0[0], 255);
    assert_eq!(detection.mask.get_pixel(2, 2).0[0], 255);
    assert_eq!(
        detection.mask.get_pixel(3, 3).0[0],
        0,
        "x2/y2 are exclusive"
    );
    assert_eq!(detection.mask.get_pixel(0, 0).0[0], 0);
    assert_eq!(common::nonzero_count(&detection.mask), 4);
}

#[test]
fn mock_detector_explicit_mask_wins_over_the_fills() {
    let mask = GrayImage::from_pixel(5, 6, Luma([9]));
    let detector = MockDetector::new()
        .with_mask_fill(200)
        .with_mask(mask.clone());

    let detection = detector
        .detect(&synthetic_page(64, 64))
        .expect("mock succeeds");

    assert_eq!(detection.mask.dimensions(), (5, 6));
    assert_eq!(detection.mask.as_raw(), mask.as_raw());
}

#[test]
fn mock_detector_failing_returns_an_error_and_still_counts_the_call() {
    let detector = MockDetector::new().failing();

    let error = detector
        .detect(&synthetic_page(8, 8))
        .expect_err("configured to fail");

    assert!(matches!(error, StageError::Inference(_)));
    assert_eq!(detector.calls(), 1);
}

// ------------------------------------------------------------ ReplayDetector

#[test]
fn replay_detector_returns_the_recorded_pair_byte_for_byte() {
    let (dir, stem) = synthetic_replay_fixture();
    let detector = ReplayDetector::new(dir.path(), stem);

    let detection = detector
        .detect(&synthetic_page(1, 1))
        .expect("replay succeeds");

    assert_eq!(detection.blocks, synthetic_blocks());
    assert_eq!(detection.mask.dimensions(), common::REPLAY_SIZE);
    assert_eq!(detection.mask.as_raw(), synthetic_raw_mask().as_raw());
}

#[test]
fn replay_detector_ignores_the_input_image() {
    // spec §7.2.1: the trait boundary is stem-less, so replay is bound at construction
    // and the passed image's content and size are irrelevant.
    let (dir, stem) = synthetic_replay_fixture();
    let detector = ReplayDetector::new(dir.path(), stem);

    let small = detector
        .detect(&synthetic_page(3, 3))
        .expect("replay succeeds");
    let large = detector
        .detect(&synthetic_page(400, 7))
        .expect("replay succeeds");

    assert_eq!(small.mask.as_raw(), large.mask.as_raw());
    assert_eq!(small.blocks, large.blocks);
    assert_eq!(detector.calls(), 2);
}

#[test]
fn replay_detector_exposes_the_section_7_2_1_artifact_names() {
    let (dir, stem) = synthetic_replay_fixture();
    let detector = ReplayDetector::new(dir.path(), stem);

    assert_eq!(
        detector.mask_path(),
        dir.path().join(format!("{stem}_detector_mask.png"))
    );
    assert_eq!(
        detector.blocks_path(),
        dir.path().join(format!("{stem}_detector_blocks.json"))
    );
    assert!(detector.mask_path().is_file());
    assert!(detector.blocks_path().is_file());
}

#[test]
fn replay_blocks_json_round_trips_raw_block() {
    let (dir, stem) = synthetic_replay_fixture();
    let bytes = std::fs::read(dir.path().join(format!("{stem}_detector_blocks.json")))
        .expect("fixture written");

    let blocks: Vec<RawBlock> = serde_json::from_slice(&bytes).expect("RawBlock is Deserialize");

    assert_eq!(blocks, synthetic_blocks());
}

// ------------------------------------------------------------ type invariants

#[test]
fn class_index_is_not_recoverable_from_language() {
    // Why `RawBlock` keeps `class_index` rather than storing a `Language`: the mapping
    // (§8.3 step 4) is many-to-one -- class 2 and any unknown class both map to `None`
    // -- so the replay artifact must record the raw index to be faithful.
    let unknown = RawBlock {
        rect: Rect::new(0, 0, 1, 1),
        class_index: 2,
        confidence: 0.5,
    };
    let out_of_range = RawBlock {
        class_index: 7,
        ..unknown
    };

    let language_of = |_block: &RawBlock| -> Option<Language> { None };
    assert_eq!(language_of(&unknown), language_of(&out_of_range));
    assert_ne!(unknown.class_index, out_of_range.class_index);
    assert_eq!(pc_detect::yolo::class_to_language(2), None);
    assert_eq!(pc_detect::yolo::class_to_language(7), None);
    assert_eq!(
        pc_detect::yolo::class_to_language(2),
        pc_detect::yolo::class_to_language(7)
    );
}
