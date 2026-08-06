//! `pc-detect` -- STAGE 1, text detection (spec §8).
//!
//! Everything model-shaped goes through [`TextDetector`] (§1 rule 4), so the whole
//! stage is testable with mocks and replay fixtures and no ONNX runtime present. The
//! `onnx` feature (task D4, not implemented in this pass) adds the real backend.
//!
//! [`DetectStage::run`]'s wiring is implemented here (task D7); the signatures are frozen
//! with the tests.

/// Task A1 (spec §16.37 item 8) -- pure primitives for `MaskRefineMode::Annotation`.
/// **Not wired into [`run`]**, which still rejects that mode; wiring is task A4.
pub mod annotate;
/// Task A3 (spec §16.37 item 8) -- connected components, the XOR merge loop and hole filling
/// for `MaskRefineMode::Annotation`. **Not wired into [`run`]**, which still rejects that mode;
/// wiring is task A4.
pub mod annotate_merge;
/// Task A3b (spec §16.37 item 11) -- `refine_mask`'s page-level driver and
/// `refine_undetected_mask` for `MaskRefineMode::Annotation`. **Not wired into [`run`]**, which
/// still rejects that mode; wiring is task A4.
pub mod annotate_refine;
pub mod detector;
pub mod mask;
pub mod onnx;
pub mod resize;
pub mod yolo;

#[cfg(any(test, feature = "testkit"))]
pub mod mock;

#[cfg(any(test, feature = "testkit"))]
pub mod oracle;

pub use annotate::{
    candidate_grey_values, erode_rect3x3, expand_text_window, histogram_255, rgb_to_gray,
    top_k_colors, top_k_colors_default, Histogram255, ANNOTATION_EXPAND_R,
};
pub use detector::{DetectInput, DetectOutput, RawBlock, RawDetection, TextDetector};
pub use mask::{refine_simple, REFINE_DILATE_RADIUS, REFINE_EXPAND, REFINE_THRESHOLD};
pub use resize::{calculate_new_size_and_scale, resize_area, round_half_away};
pub use yolo::{Candidate, LetterboxGeometry};

#[cfg(any(test, feature = "testkit"))]
pub use mock::{write_replay_fixture, MockDetector, ReplayDetector};

use image::GrayImage;
use pc_config::MaskRefineMode;
use pc_core::{DetectAnalytic, DetectedBlock, ImageHandle, PageDataRaw, Rect, StageError, Step};
use std::path::Path;

/// spec §8.3 step 6 / §8.2: fixed at 0.1 in v1.
pub const DEFAULT_MIN_MASK_COVERAGE: f32 = 0.1;

/// `mean(mask over rect) / 255` (spec §8.3 step 6). `rect` uses the codebase-wide
/// exclusive `x2`/`y2` convention (§16.6 item 5); an empty or fully out-of-bounds rect
/// has coverage `0.0`.
pub fn mask_coverage(mask: &GrayImage, rect: Rect) -> f32 {
    let Some((x, y, width, height)) = rect.to_crop(mask.dimensions()) else {
        return 0.0;
    };

    let mut sum = 0_u64;
    for pixel_y in y..y + height {
        for pixel_x in x..x + width {
            sum += u64::from(mask.get_pixel(pixel_x, pixel_y).0[0]);
        }
    }
    sum as f32 / (width as f32 * height as f32 * 255.0)
}

/// spec §3 -- the stage contract. `Ctx` is the injected detector.
pub struct DetectStage;

impl pc_core::Stage for DetectStage {
    type Input = DetectInput;
    type Output = DetectOutput;
    type Ctx<'a> = &'a dyn TextDetector;
    const STEP: Step = Step::Detect;

    fn run(input: Self::Input, ctx: Self::Ctx<'_>) -> Result<Self::Output, StageError> {
        run(input, ctx)
    }
}

/// spec §8.3, steps 1-7.
///
/// Rejects `MaskRefineMode::Annotation` with `StageError::InvalidInput` **before** any
/// detection work (§8.3 step 5, §15.2, §16.5 item 3) -- failing after paying for
/// inference would be gratuitous.
pub fn run(input: DetectInput, detector: &dyn TextDetector) -> Result<DetectOutput, StageError> {
    if input.config.mask_refine_mode == MaskRefineMode::Annotation {
        return Err(StageError::InvalidInput(
            "mask_refine_mode `annotation` is not implemented in v1".into(),
        ));
    }

    let original = input.source.load()?;
    let original = original.to_rgb8();
    let (new_width, new_height, scale) = calculate_new_size_and_scale(
        original.width(),
        original.height(),
        input.target_height_lower,
        input.target_height_upper,
    );
    let base_image = resize_area(&original, new_width, new_height);

    if let Some(path) = &input.base_image_dest {
        write_png(&image::DynamicImage::ImageRgb8(base_image.clone()), path)?;
    }

    let detection = detector.detect(&base_image)?;
    let blocks_detected = detection.blocks.len();
    let geometry = LetterboxGeometry {
        net_size: new_width.max(new_height),
        dw: 0.0,
        dh: 0.0,
        image_size: (new_width, new_height),
    };
    let refined_mask = refine_simple(&detection.mask, &geometry, &detection.blocks)?;

    if let Some(path) = &input.raw_mask_dest {
        write_png(&image::DynamicImage::ImageLuma8(refined_mask.clone()), path)?;
    }

    let blocks = detection
        .blocks
        .into_iter()
        .filter_map(|block| {
            let coverage = mask_coverage(&refined_mask, block.rect);
            (coverage >= input.min_mask_coverage).then(|| DetectedBlock {
                rect: block.rect,
                language: yolo::class_to_language(block.class_index),
                confidence: block.confidence,
                mask_coverage: coverage,
            })
        })
        .collect::<Vec<_>>();
    let blocks_kept = blocks.len();

    let base_handle = match &input.base_image_dest {
        Some(path) => ImageHandle::with_both(path, image::DynamicImage::ImageRgb8(base_image)),
        None => ImageHandle::from_memory(image::DynamicImage::ImageRgb8(base_image)),
    };
    let mask_handle = match &input.raw_mask_dest {
        Some(path) => ImageHandle::with_both(path, image::DynamicImage::ImageLuma8(refined_mask)),
        None => ImageHandle::from_memory(image::DynamicImage::ImageLuma8(refined_mask)),
    };
    let page = assemble_page(
        &input,
        base_handle,
        mask_handle,
        scale,
        (new_width, new_height),
        blocks,
    );
    let analytics = build_analytic(&input.original_path, blocks_detected, blocks_kept);

    Ok(DetectOutput { page, analytics })
}

fn write_png(image: &image::DynamicImage, path: &Path) -> Result<(), StageError> {
    image
        .save_with_format(path, image::ImageFormat::Png)
        .map_err(|error| StageError::Io {
            path: path.to_path_buf(),
            source: std::io::Error::other(error),
        })
}

/// spec §8.3 step 7: `PageDataRaw` from the surviving blocks, in NMS order.
pub fn assemble_page(
    input: &DetectInput,
    base_image: ImageHandle,
    raw_mask: ImageHandle,
    scale: f64,
    image_size: (u32, u32),
    blocks: Vec<DetectedBlock>,
) -> PageDataRaw {
    PageDataRaw {
        schema_version: input.schema_version,
        original_path: input.original_path.clone(),
        base_image,
        raw_mask,
        scale,
        image_size,
        blocks,
    }
}

/// spec §2.7 / §8.3 step 7.
pub fn build_analytic(path: &Path, blocks_detected: usize, blocks_kept: usize) -> DetectAnalytic {
    DetectAnalytic {
        path: path.to_path_buf(),
        blocks_detected,
        blocks_kept,
    }
}
