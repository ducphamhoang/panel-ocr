//! L6-4 / §16.40 frozen integration tests for ordinary and split pipeline execution.
//!
//! FROZEN (CLAUDE.md).

mod common;

use image::{Rgb, RgbImage, Rgba};
use pc_core::{Rect, StageError, Step};
use pc_detect::{MockDetector, RawBlock, TextDetector};
use pc_inpaint::{Inpainter, TILE};
use pc_pipeline::cache::{CLEAN_INPAINT_SUFFIX, INPAINTING_SUFFIX};
use pc_pipeline::{
    CachePaths, DetectorProvider, ImageOutcome, InpainterProvider, PipelineCtx, SaveOnly,
    SharedDetector,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const FALLBACK_TOP: Rgb<u8> = Rgb([21, 31, 41]);
const FALLBACK_MIDDLE: Rgb<u8> = Rgb([61, 71, 81]);
const FALLBACK_BOTTOM: Rgb<u8> = Rgb([101, 111, 121]);
const SENTINEL: Rgb<u8> = Rgb([231, 17, 149]);

struct FlatInpainter {
    calls: AtomicUsize,
}

impl FlatInpainter {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
}

impl Inpainter for FlatInpainter {
    fn inpaint_tile(
        &self,
        _tile: &RgbImage,
        _mask: &pc_imageops::BinaryMask,
    ) -> Result<RgbImage, StageError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(RgbImage::from_pixel(TILE, TILE, SENTINEL))
    }
}

struct CountingProvider {
    requests: AtomicUsize,
    constructions: AtomicUsize,
    inpainter: Arc<FlatInpainter>,
}

impl CountingProvider {
    fn new() -> Self {
        Self {
            requests: AtomicUsize::new(0),
            constructions: AtomicUsize::new(0),
            inpainter: Arc::new(FlatInpainter::new()),
        }
    }
}

impl InpainterProvider for CountingProvider {
    fn inpainter(&self) -> Result<Arc<dyn Inpainter>, StageError> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        // This double deliberately records "construction" independently of requests. A real
        // L5 provider latches this operation; §16.40's assertions count requests, not sessions.
        self.constructions.fetch_add(1, Ordering::SeqCst);
        Ok(self.inpainter.clone())
    }

    fn failures_are_run_fatal(&self) -> bool {
        true
    }
}

struct SegmentDetectorProvider {
    eligible_segment_token: Option<String>,
}

impl DetectorProvider for SegmentDetectorProvider {
    fn detector_for(&self, original: &Path) -> Result<Arc<dyn TextDetector>, StageError> {
        let name = original.file_name().unwrap().to_string_lossy();
        if self
            .eligible_segment_token
            .as_ref()
            .is_some_and(|token| name.contains(token))
        {
            Ok(Arc::new(
                MockDetector::new()
                    .with_blocks(vec![RawBlock {
                        rect: Rect::new(8, 8, 24, 23),
                        class_index: 1,
                        confidence: 0.9,
                    }])
                    .with_block_fill(255),
            ))
        } else {
            Ok(common::empty_detector())
        }
    }
}

fn write_three_band_strip(path: &Path) {
    let image = RgbImage::from_fn(32, 96, |_x, y| match y {
        0..=30 => FALLBACK_TOP,
        31..=62 => FALLBACK_MIDDLE,
        _ => FALLBACK_BOTTOM,
    });
    image.save(path).unwrap();
}

fn strip_options(cache: &Path, out: &Path) -> pc_pipeline::PipelineOptions {
    let mut options = common::options(cache, out);
    options.profile.preprocessor.box_min_size = 1;
    options.profile.preprocessor.suspicious_box_min_size = 1;
    options.profile.general.preferred_file_type = ".png".to_string();
    options.profile.general.preferred_split_height = 32;
    options.profile.general.split_tolerance_margin = 1;
    options.profile.general.long_strip_aspect_ratio = 0.34;
    options.profile.general.merge_after_split = true;
    options.profile.inpainter.inpainting_enabled = true;
    options.profile.inpainter.inpainting_fade_radius = 0;
    options.profile.inpainter.inpainting_isolation_radius = 0;
    options.profile.inpainter.min_inpainting_radius = 0;
    options.profile.inpainter.max_inpainting_radius = 0;
    options.profile.inpainter.inpainting_radius_multiplier = 0.0;
    // Any attempted mask fit has non-negative deviation and is therefore failed/eligible.
    // Empty-detector segments have no attempted region and remain ineligible.
    options.profile.masker.mask_max_standard_deviation = -1.0;
    options.skips.denoise = true;
    options
}

fn rgba(path: &Path) -> image::RgbaImage {
    image::open(path).unwrap().to_rgba8()
}

fn discover_segment_caches(cache_dir: &Path, segments: &[PathBuf]) -> Vec<CachePaths> {
    segments
        .iter()
        .map(|segment| {
            CachePaths::discover(cache_dir, segment)
                .unwrap()
                .expect("segment has a cache entry")
        })
        .collect()
}

/// §16.40 item 6(a), all nine required facts, on the verbatim scope from item 1:
/// fresh Disk execution entering the qualifying merged split branch. The authored eligible
/// segment count is the literal 1; provider requests, artifacts, pixels, alpha spans and the
/// original-sized exported sources are all observed independently.
#[test]
fn mixed_three_segment_strip_preserves_ineligible_spans_and_uses_one_eligible_segment() {
    const ELIGIBLE_SEGMENTS: usize = 1;
    const TOTAL_SEGMENTS: usize = 3;
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("cache");
    let out = dir.path().join("out");
    let strip = dir.path().join("three-band.png");
    write_three_band_strip(&strip);
    let options = strip_options(&cache_dir, &out);
    let detectors = SegmentDetectorProvider {
        eligible_segment_token: Some("_seg001.png".into()),
    };
    let inpainters = CountingProvider::new();
    let ctx = PipelineCtx::new(&detectors).with_inpainter(&inpainters);

    let outcome = pc_pipeline::process_image_with_splitting(&strip, &options, &ctx);

    assert!(
        matches!(outcome, ImageOutcome::Completed { .. }),
        "{outcome:?}"
    );
    assert_eq!(
        inpainters.requests.load(Ordering::SeqCst),
        ELIGIBLE_SEGMENTS
    );
    assert!(inpainters.requests.load(Ordering::SeqCst) < TOTAL_SEGMENTS);
    assert_eq!(
        outcome.files_written().len(),
        2,
        "one merged export operation writes cleaned + mask"
    );
    for written in outcome.files_written() {
        assert_eq!(image::image_dimensions(written).unwrap(), (32, 96));
    }

    let original_cache = CachePaths::discover(&cache_dir, &strip).unwrap().unwrap();
    let manifest = pc_pipeline::strip::read_manifest(&original_cache.splits_manifest()).unwrap();
    assert_eq!(manifest.segments.len(), TOTAL_SEGMENTS);
    assert_eq!(manifest.split_rows, vec![31, 63]);
    let segment_caches = discover_segment_caches(&cache_dir, &manifest.segments);

    let merged_clean_path = original_cache.for_suffix(CLEAN_INPAINT_SUFFIX);
    let merged_inpainting_path = original_cache.for_suffix(INPAINTING_SUFFIX);
    let merged_clean = rgba(&merged_clean_path);
    let merged_inpainting = rgba(&merged_inpainting_path);

    assert_eq!(*merged_clean.get_pixel(2, 10), Rgba([21, 31, 41, 255]));
    assert_eq!(*merged_clean.get_pixel(10, 42), Rgba([231, 17, 149, 255]));
    assert_eq!(*merged_clean.get_pixel(2, 82), Rgba([101, 111, 121, 255]));
    assert_eq!(merged_inpainting.get_pixel(2, 10).0[3], 0);
    assert!(merged_inpainting.get_pixel(10, 42).0[3] > 0);
    assert_eq!(merged_inpainting.get_pixel(2, 82).0[3], 0);

    assert!(segment_caches[1].for_suffix(CLEAN_INPAINT_SUFFIX).exists());
    assert!(segment_caches[1].for_suffix(INPAINTING_SUFFIX).exists());
    for index in [0, 2] {
        assert!(!segment_caches[index]
            .for_suffix(CLEAN_INPAINT_SUFFIX)
            .exists());
        assert!(!segment_caches[index].for_suffix(INPAINTING_SUFFIX).exists());
    }

    let cleaned_export = outcome
        .files_written()
        .iter()
        .find(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .contains("_clean")
        })
        .unwrap();
    let mask_export = outcome
        .files_written()
        .iter()
        .find(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .contains("_mask")
        })
        .unwrap();
    let exported_clean = rgba(cleaned_export);
    let exported_mask = rgba(mask_export);
    assert_eq!(exported_clean.into_raw(), merged_clean.into_raw());
    let mut expected_exported_mask = image::RgbaImage::from_pixel(32, 96, Rgba([0, 0, 0, 0]));
    // Independent hand trace of the export composite: final_mask is the eligible detector
    // rectangle in segment 1 (global y=33..59), the optional denoise layer is disabled
    // and transparent in this fixture, and inpainting_mask is composited last.
    let mut expected_final_mask = image::RgbaImage::from_pixel(32, 96, Rgba([0, 0, 0, 0]));
    let expected_denoise_layer = image::RgbaImage::from_pixel(32, 96, Rgba([0, 0, 0, 0]));
    let mut expected_inpainting_mask = image::RgbaImage::from_pixel(32, 96, Rgba([0, 0, 0, 0]));
    for (row, x_start, x_end) in [
        (32, 8, 24),
        (33, 5, 27),
        (34, 4, 28),
        (35, 3, 29),
        (36, 2, 30),
        (37, 2, 30),
        (38, 1, 31),
        (39, 1, 31),
        (40, 1, 31),
        (41, 1, 31),
        (42, 1, 31),
        (43, 1, 31),
        (44, 1, 31),
        (45, 1, 31),
        (46, 1, 31),
        (47, 1, 31),
        (48, 1, 31),
        (49, 1, 31),
        (50, 1, 31),
        (51, 1, 31),
        (52, 1, 31),
        (53, 1, 31),
        (54, 1, 31),
        (55, 2, 30),
        (56, 2, 30),
        (57, 3, 29),
        (58, 4, 28),
        (59, 5, 27),
        (60, 8, 24),
    ] {
        for x in x_start..x_end {
            expected_final_mask.put_pixel(x, row, Rgba([0, 0, 0, 255]));
            expected_inpainting_mask.put_pixel(x, row, Rgba([231, 17, 149, 255]));
        }
    }
    for (expected, final_mask) in expected_exported_mask
        .pixels_mut()
        .zip(expected_final_mask.pixels())
    {
        if final_mask.0[3] != 0 {
            *expected = *final_mask;
        }
    }
    for (expected, denoise) in expected_exported_mask
        .pixels_mut()
        .zip(expected_denoise_layer.pixels())
    {
        if denoise.0[3] != 0 {
            *expected = *denoise;
        }
    }
    for (expected, inpainting) in expected_exported_mask
        .pixels_mut()
        .zip(expected_inpainting_mask.pixels())
    {
        if inpainting.0[3] != 0 {
            *expected = *inpainting;
        }
    }
    assert_eq!(exported_mask.into_raw(), expected_exported_mask.into_raw());
}

/// §16.40 items 3 and 6(b): on the same fresh Disk merged-split scope, an all-ineligible
/// strip requests no provider, exports the three authored fallback bands in manifest order,
/// and creates no segment or original-strip L6 PNG.
#[test]
fn all_ineligible_three_segment_strip_uses_fallback_pixels_and_no_l6_pngs() {
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("cache");
    let out = dir.path().join("out");
    let strip = dir.path().join("three-band.png");
    write_three_band_strip(&strip);
    let mut options = strip_options(&cache_dir, &out);
    options.save_only = Some(SaveOnly::Cleaned);
    let detectors = SegmentDetectorProvider {
        eligible_segment_token: None,
    };
    let inpainters = CountingProvider::new();
    let ctx = PipelineCtx::new(&detectors).with_inpainter(&inpainters);

    let outcome = pc_pipeline::process_image_with_splitting(&strip, &options, &ctx);

    assert_eq!(inpainters.requests.load(Ordering::SeqCst), 0);
    assert_eq!(inpainters.constructions.load(Ordering::SeqCst), 0);
    assert_eq!(outcome.files_written().len(), 1, "{outcome:?}");
    let exported = rgba(&outcome.files_written()[0]);
    assert_eq!(*exported.get_pixel(2, 10), Rgba([21, 31, 41, 255]));
    assert_eq!(*exported.get_pixel(2, 42), Rgba([61, 71, 81, 255]));
    assert_eq!(*exported.get_pixel(2, 82), Rgba([101, 111, 121, 255]));

    let original_cache = CachePaths::discover(&cache_dir, &strip).unwrap().unwrap();
    let manifest = pc_pipeline::strip::read_manifest(&original_cache.splits_manifest()).unwrap();
    for segment_cache in discover_segment_caches(&cache_dir, &manifest.segments) {
        assert!(!segment_cache.for_suffix(CLEAN_INPAINT_SUFFIX).exists());
        assert!(!segment_cache.for_suffix(INPAINTING_SUFFIX).exists());
    }
    assert!(!original_cache.for_suffix(CLEAN_INPAINT_SUFFIX).exists());
    assert!(!original_cache.for_suffix(INPAINTING_SUFFIX).exists());
}

/// §16.38 item 23(e), ordinary-page half: enabled but ineligible execution does not need
/// an injected provider and still exports the exact non-inpainted cleaned bytes.
#[test]
fn ordinary_ineligible_page_without_provider_exports_non_inpainted_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let page = dir.path().join("ordinary.png");
    RgbImage::from_pixel(24, 20, FALLBACK_TOP)
        .save(&page)
        .unwrap();
    let mut options = common::options(&cache, &out);
    options.profile.general.preferred_file_type = ".png".into();
    options.profile.inpainter.inpainting_enabled = true;
    options.save_only = Some(SaveOnly::Cleaned);
    let detectors = SharedDetector::new(common::empty_detector());

    let outcome = pc_pipeline::process_image(&page, &options, &PipelineCtx::new(&detectors));

    assert!(!outcome.is_failed(), "{outcome:?}");
    assert_eq!(outcome.files_written().len(), 1);
    assert_eq!(
        image::open(&outcome.files_written()[0])
            .unwrap()
            .to_rgb8()
            .into_raw(),
        RgbImage::from_pixel(24, 20, FALLBACK_TOP).into_raw()
    );
}

/// §16.38 item 24(d): disabling the stage leaves the optional provider unreachable and
/// ordinary export unchanged. This pins pipeline control flow, not `run_inpaint` behavior;
/// the joint interface explicitly says the disabled adapter is not called.
#[test]
fn disabled_ordinary_stage_does_not_request_provider() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let page = dir.path().join("ordinary.png");
    RgbImage::from_pixel(24, 20, FALLBACK_TOP)
        .save(&page)
        .unwrap();
    let mut options = common::options(&cache, &out);
    options.profile.inpainter.inpainting_enabled = false;
    options.save_only = Some(SaveOnly::Cleaned);
    let detectors = SharedDetector::new(common::empty_detector());
    let inpainters = CountingProvider::new();
    let ctx = PipelineCtx::new(&detectors).with_inpainter(&inpainters);

    let outcome = pc_pipeline::process_image(&page, &options, &ctx);

    assert!(!outcome.is_failed(), "{outcome:?}");
    assert_eq!(inpainters.requests.load(Ordering::SeqCst), 0);
}

/// §16.40 item 5 and §16.38 item 25(c): a second ordinary run beginning from the existing
/// mask predecessor still executes `Step::Inpaint` and writes the PNG pair. No test or
/// fixture invents an inpaint JSON checkpoint or resume-at-Export level.
#[test]
fn resumed_from_mask_predecessor_executes_inpaint_and_writes_png_pair() {
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("cache");
    let out = dir.path().join("out");
    let page = dir.path().join("ordinary.png");
    let mut image = RgbImage::from_fn(32, 32, |x, y| {
        if (x + y) % 2 == 0 {
            FALLBACK_TOP
        } else {
            Rgb([211, 221, 231])
        }
    });
    for y in 8..24 {
        for x in 8..24 {
            let value = if (x + y) % 2 == 0 { 0 } else { 255 };
            image.put_pixel(x, y, Rgb([value, value, value]));
        }
    }
    image.save(&page).unwrap();
    let detector = SharedDetector::new(common::detector_with_block(Rect::new(8, 8, 24, 24)));
    let mut first = common::options(&cache_dir, &out);
    first.profile.inpainter.inpainting_enabled = false;
    first.profile.masker.mask_max_standard_deviation = 100.0;
    first.profile.preprocessor.box_min_size = 1;
    first.profile.preprocessor.suspicious_box_min_size = 1;
    first.skips.denoise = true;
    let first_outcome = pc_pipeline::process_image(&page, &first, &PipelineCtx::new(&detector));
    assert!(!first_outcome.is_failed(), "{first_outcome:?}");

    let cache = CachePaths::discover(&cache_dir, &page).unwrap().unwrap();
    let mask_data = pc_pipeline::checkpoint::read_mask_data(
        &cache.for_output(pc_core::Output::MaskDataJson),
    )
    .unwrap();
    assert_eq!(mask_data.regions.len(), 1);
    let region = &mask_data.regions[0];
    assert_eq!(region.rect, Rect::new(1, 1, 32, 31));
    assert!(region.std_deviation > 0.0);
    assert!(!region.failed);
    assert_eq!(region.thickness, Some(4));
    let combined_mask = mask_data.combined_mask.load().unwrap().to_rgba8();
    assert!(combined_mask.pixels().any(|pixel| pixel.0[3] > 0));

    let mut resumed = first.clone();
    resumed.profile.inpainter.inpainting_enabled = true;
    resumed.profile.inpainter.inpainting_fade_radius = 0;
    resumed.profile.inpainter.inpainting_isolation_radius = 0;
    resumed.profile.inpainter.inpainting_min_std_dev = 0.0;
    resumed.profile.inpainter.min_inpainting_radius = 4;
    resumed.profile.inpainter.max_inpainting_radius = 0;
    resumed.skips.mask = true;
    let inpainters = CountingProvider::new();
    let ctx = PipelineCtx::new(&detector).with_inpainter(&inpainters);

    let outcome = pc_pipeline::process_image(&page, &resumed, &ctx);

    assert!(!outcome.is_failed(), "{outcome:?}");
    assert_eq!(inpainters.requests.load(Ordering::SeqCst), 1);
    let cache = CachePaths::discover(&cache_dir, &page).unwrap().unwrap();
    assert!(cache.for_suffix(CLEAN_INPAINT_SUFFIX).exists());
    assert!(cache.for_suffix(INPAINTING_SUFFIX).exists());
    let json_names = std::fs::read_dir(&cache_dir)
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains("inpaint") && name.ends_with(".json"))
        .collect::<Vec<_>>();
    assert!(
        json_names.is_empty(),
        "no inpaint JSON checkpoint: {json_names:?}"
    );
}


/// §16.38 item 3(d): Memory-mode failed-region eligibility passes the detector's raw mask to
/// the tile loop, not `MaskData::base_image`; authored values make that source choice observable.
#[test]
fn memory_failed_region_uses_raw_mask_for_tile_fill_bit() {
    struct CapturingInpainter {
        calls: AtomicUsize,
        pixel_zero_mask_bit: std::sync::Mutex<Option<u8>>,
    }

    impl Inpainter for CapturingInpainter {
        fn inpaint_tile(
            &self,
            _tile: &RgbImage,
            mask: &pc_imageops::BinaryMask,
        ) -> Result<RgbImage, StageError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.pixel_zero_mask_bit.lock().unwrap() = Some(mask.as_bits()[8 * 512 + 8]);
            Ok(RgbImage::from_pixel(TILE, TILE, SENTINEL))
        }
    }

    struct CapturingProvider {
        requests: AtomicUsize,
        inpainter: Arc<CapturingInpainter>,
    }

    impl InpainterProvider for CapturingProvider {
        fn inpainter(&self) -> Result<Arc<dyn Inpainter>, StageError> {
            self.requests.fetch_add(1, Ordering::SeqCst);
            Ok(self.inpainter.clone())
        }

        fn failures_are_run_fatal(&self) -> bool {
            true
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    let out = dir.path().join("out");
    let page = dir.path().join("memory-raw-mask.png");
    let mut authored = RgbImage::from_pixel(32, 32, Rgb([17, 19, 23]));
    authored.put_pixel(8, 8, Rgb([0, 0, 0]));
    authored.save(&page).unwrap();

    let mut raw_mask = image::GrayImage::from_pixel(32, 32, image::Luma([0]));
    for y in 8..24 {
        for x in 8..24 {
            raw_mask.put_pixel(x, y, image::Luma([255]));
        }
    }
    let detector = SharedDetector::new(Arc::new(
        MockDetector::new()
            .with_blocks(vec![RawBlock {
                rect: Rect::new(8, 8, 24, 24),
                class_index: 1,
                confidence: 0.9,
            }])
            .with_mask(raw_mask),
    ));
    let inpainter = Arc::new(CapturingInpainter {
        calls: AtomicUsize::new(0),
        pixel_zero_mask_bit: std::sync::Mutex::new(None),
    });
    let provider = CapturingProvider {
        requests: AtomicUsize::new(0),
        inpainter: inpainter.clone(),
    };
    let mut options = common::options(&cache, &out);
    options.checkpointing = pc_pipeline::Checkpointing::Memory;
    options.profile.preprocessor.box_min_size = 1;
    options.profile.preprocessor.suspicious_box_min_size = 1;
    options.profile.masker.mask_max_standard_deviation = -1.0;
    options.profile.inpainter.inpainting_enabled = true;
    options.profile.inpainter.inpainting_fade_radius = 0;
    options.profile.inpainter.inpainting_isolation_radius = 0;
    options.profile.inpainter.min_inpainting_radius = 0;
    options.profile.inpainter.max_inpainting_radius = 0;
    options.profile.inpainter.inpainting_radius_multiplier = 0.0;
    options.skips.denoise = true;
    options.profile.general.split_long_strips = false;

    let outcome = pc_pipeline::process_image(
        &page,
        &options,
        &PipelineCtx::new(&detector).with_inpainter(&provider),
    );

    assert!(matches!(outcome, ImageOutcome::Completed { .. }), "{outcome:?}");
    assert_eq!(provider.requests.load(Ordering::SeqCst), 1);
    assert_eq!(inpainter.calls.load(Ordering::SeqCst), 1);
    assert_eq!(*inpainter.pixel_zero_mask_bit.lock().unwrap(), Some(1));
}

/// ordinary image and is not promoted to run-fatal by the provider's acquisition policy.
#[test]
fn ordinary_acquired_inference_failure_names_inpaint_step_and_is_not_run_fatal() {
    struct Fail;
    impl Inpainter for Fail {
        fn inpaint_tile(
            &self,
            _tile: &RgbImage,
            _mask: &pc_imageops::BinaryMask,
        ) -> Result<RgbImage, StageError> {
            Err(StageError::Inference("ordinary literal failure".into()))
        }
    }
    struct Provider;
    impl InpainterProvider for Provider {
        fn inpainter(&self) -> Result<Arc<dyn Inpainter>, StageError> {
            Ok(Arc::new(Fail))
        }
        fn failures_are_run_fatal(&self) -> bool {
            true
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let page = dir.path().join("ordinary.png");
    RgbImage::from_pixel(32, 32, FALLBACK_TOP)
        .save(&page)
        .unwrap();
    let detector = SharedDetector::new(common::detector_with_block(Rect::new(8, 8, 24, 24)));
    let mut options = common::options(&dir.path().join("cache"), &dir.path().join("out"));
    options.profile.masker.mask_max_standard_deviation = -1.0;
    options.profile.preprocessor.box_min_size = 1;
    options.profile.preprocessor.suspicious_box_min_size = 1;
    options.profile.inpainter.inpainting_enabled = true;
    options.profile.inpainter.inpainting_fade_radius = 0;
    options.profile.inpainter.inpainting_isolation_radius = 0;
    options.profile.inpainter.min_inpainting_radius = 0;
    options.profile.inpainter.max_inpainting_radius = 0;
    options.skips.denoise = true;
    let provider = Provider;
    let ctx = PipelineCtx::new(&detector).with_inpainter(&provider);

    let outcome = pc_pipeline::process_image(&page, &options, &ctx);

    assert!(matches!(
        outcome,
        ImageOutcome::Failed {
            step: Step::Inpaint,
            error: StageError::Inference(ref message),
            ..
        } if message == "ordinary literal failure"
    ));
}
