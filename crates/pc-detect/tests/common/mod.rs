//! Shared fixtures for the frozen `pc-detect` test suites. FROZEN with the tests.
#![allow(dead_code)]

use image::{GrayImage, Luma, Rgb, RgbImage};
use pc_config::{MaskRefineMode, TextDetectorConfig};
use pc_core::{ImageHandle, Rect};
use pc_detect::{DetectInput, RawBlock};
use std::path::PathBuf;
use tempfile::TempDir;

/// The stem the §7.2.1 replay pair is written under in these tests.
pub const REPLAY_STEM: &str = "synthetic_page";
/// Every synthetic fixture in this suite is 64x64 -- small enough to hand-trace,
/// large enough for the 16px expand + 3px dilate of §8.3 step 5 to be non-degenerate.
pub const REPLAY_SIZE: (u32, u32) = (64, 64);

/// The block that survives the coverage filter: the synthetic raw mask is saturated
/// exactly over this rect.
pub const COVERED_RECT: Rect = Rect {
    x1: 8,
    y1: 8,
    x2: 24,
    y2: 24,
};
/// The block that is dropped: the synthetic raw mask is zero everywhere inside it.
pub const UNCOVERED_RECT: Rect = Rect {
    x1: 40,
    y1: 40,
    x2: 56,
    y2: 56,
};

/// A deterministic RGB page. Content is irrelevant to the mocked detectors; it only
/// has to be stable so byte-identity assertions mean something.
pub fn synthetic_page(width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    })
}

/// Two blocks: index 0 gets full coverage under [`synthetic_raw_mask`] and is kept by
/// the §8.3 step 6 filter; index 1 gets zero coverage and is dropped.
///
/// Class indices differ so §8.3 step 4's class -> language mapping is observable
/// per block (0 => English, 1 => Japanese).
pub fn synthetic_blocks() -> Vec<RawBlock> {
    vec![
        RawBlock {
            rect: COVERED_RECT,
            class_index: 0,
            confidence: 0.875,
        },
        RawBlock {
            rect: UNCOVERED_RECT,
            class_index: 1,
            confidence: 0.75,
        },
    ]
}

/// Saturated (255) over [`COVERED_RECT`], zero elsewhere. Both blocks' 16px-expanded
/// windows are in bounds, so the refinement keeps the saturated area and nothing else.
pub fn synthetic_raw_mask() -> GrayImage {
    let (width, height) = REPLAY_SIZE;
    let mut mask = GrayImage::from_pixel(width, height, Luma([0]));
    for y in COVERED_RECT.y1 as u32..COVERED_RECT.y2 as u32 {
        for x in COVERED_RECT.x1 as u32..COVERED_RECT.x2 as u32 {
            mask.put_pixel(x, y, Luma([255]));
        }
    }
    mask
}

/// Writes the §7.2.1 replay pair into a fresh temp dir. The `TempDir` must be kept
/// alive by the caller for as long as the fixture is read.
pub fn synthetic_replay_fixture() -> (TempDir, &'static str) {
    let dir = TempDir::new().expect("temp dir");
    pc_detect::write_replay_fixture(
        dir.path(),
        REPLAY_STEM,
        &synthetic_raw_mask(),
        &synthetic_blocks(),
    );
    (dir, REPLAY_STEM)
}

/// A `DetectInput` in Memory mode (no destinations) over an in-memory source image.
pub fn memory_input(image: RgbImage) -> DetectInput {
    DetectInput {
        schema_version: pc_core::SCHEMA_VERSION,
        source: ImageHandle::from_memory(image::DynamicImage::ImageRgb8(image)),
        original_path: PathBuf::from("/synthetic/page.png"),
        // Wide enough that no resize happens for a 64x64 page (§8.3 step 2, `h <= upper`).
        target_height_lower: 1000,
        target_height_upper: 4000,
        base_image_dest: None,
        raw_mask_dest: None,
        min_mask_coverage: pc_detect::DEFAULT_MIN_MASK_COVERAGE,
        // §16.46 item 13(b) INPUT PIN -- no assertion, name or expected value changes with
        // it. This suite's subject is `Simple`'s behaviour, and item 1(a) moves the shipped
        // default to `Annotation`; inheriting the default would silently re-point every
        // expectation below at a different algorithm while the suite stayed green.
        config: TextDetectorConfig {
            mask_refine_mode: MaskRefineMode::Simple,
            ..TextDetectorConfig::default()
        },
    }
}

pub fn nonzero_count(mask: &GrayImage) -> usize {
    mask.pixels().filter(|pixel| pixel.0[0] != 0).count()
}

#[track_caller]
pub fn assert_binary(mask: &GrayImage) {
    assert!(
        mask.pixels().all(|pixel| matches!(pixel.0[0], 0 | 255)),
        "refined masks must be strictly binary (0 or 255)"
    );
}
