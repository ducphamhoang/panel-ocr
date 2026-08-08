//! L6-4 frozen tests — the pipeline adapter for §16.38 items 23, 24(d), 25(c)-(d)
//! and §16.40 items 2(d), 5.
//!
//! FROZEN (CLAUDE.md). Production may be changed to satisfy these tests; these tests may
//! not be weakened or amended outside cookbook rule 8.

use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use pc_config::InpainterConfig;
use pc_core::{ImageHandle, MaskData, MaskRegionStats, Rect, StageError, SCHEMA_VERSION};
use pc_denoise::DenoiseOutput;
use pc_inpaint::{Inpainter, TILE};
use pc_mask::MaskOutput;
use pc_pipeline::cache::{CLEAN_INPAINT_SUFFIX, INPAINTING_SUFFIX};
use pc_pipeline::single::{
    export_sources, inpaint_dests, run_inpaint, InpaintDests, InpaintOutput,
};
use pc_pipeline::{CachePaths, InpainterProvider, PipelineError};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

const ORIGINAL: Rgb<u8> = Rgb([11, 22, 33]);
const MODEL: Rgb<u8> = Rgb([201, 17, 93]);
const NOISE: Rgba<u8> = Rgba([7, 181, 29, 255]);

struct RecordingInpainter {
    calls: AtomicUsize,
    fail: bool,
    saw_first_tile_pixel: Mutex<Option<Rgb<u8>>>,
    saw_nonzero_mask: Mutex<Option<bool>>,
}

impl RecordingInpainter {
    fn succeeding() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            fail: false,
            saw_first_tile_pixel: Mutex::new(None),
            saw_nonzero_mask: Mutex::new(None),
        }
    }

    fn failing() -> Self {
        Self {
            fail: true,
            ..Self::succeeding()
        }
    }
}

impl Inpainter for RecordingInpainter {
    fn inpaint_tile(
        &self,
        tile: &RgbImage,
        mask: &pc_imageops::BinaryMask,
    ) -> Result<RgbImage, StageError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.saw_first_tile_pixel.lock().unwrap() = Some(*tile.get_pixel(0, 0));
        *self.saw_nonzero_mask.lock().unwrap() =
            Some(mask.as_bits().iter().copied().any(|value| value != 0));
        if self.fail {
            return Err(StageError::Inference("literal tile failure".into()));
        }
        Ok(RgbImage::from_pixel(TILE, TILE, MODEL))
    }
}

struct SpyProvider {
    requests: AtomicUsize,
    fatal: bool,
    result: Mutex<Result<Arc<dyn Inpainter>, String>>,
}

impl SpyProvider {
    fn succeeds(inpainter: Arc<dyn Inpainter>) -> Self {
        Self {
            requests: AtomicUsize::new(0),
            fatal: true,
            result: Mutex::new(Ok(inpainter)),
        }
    }

    fn refuses(fatal: bool, message: &str) -> Self {
        Self {
            requests: AtomicUsize::new(0),
            fatal,
            result: Mutex::new(Err(message.to_string())),
        }
    }

    fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

impl InpainterProvider for SpyProvider {
    fn inpainter(&self) -> Result<Arc<dyn Inpainter>, StageError> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        match &*self.result.lock().unwrap() {
            Ok(inpainter) => Ok(Arc::clone(inpainter)),
            Err(message) => Err(StageError::Model(message.clone())),
        }
    }

    fn failures_are_run_fatal(&self) -> bool {
        self.fatal
    }
}

struct Inputs {
    original: ImageHandle,
    raw_mask: ImageHandle,
    mask_data: MaskData,
    noise_mask: Option<ImageHandle>,
}

fn memory_inputs(regions: Vec<MaskRegionStats>, noise: bool) -> Inputs {
    let size = (32, 24);
    let original = ImageHandle::from_memory(DynamicImage::ImageRgb8(RgbImage::from_pixel(
        size.0, size.1, ORIGINAL,
    )));
    let mut raw = GrayImage::from_pixel(size.0, size.1, Luma([0]));
    for y in 7..15 {
        for x in 9..20 {
            raw.put_pixel(x, y, Luma([255]));
        }
    }
    let raw_mask = ImageHandle::from_memory(DynamicImage::ImageLuma8(raw));
    let combined = ImageHandle::from_memory(DynamicImage::ImageRgba8(RgbaImage::from_pixel(
        size.0,
        size.1,
        Rgba([0, 0, 0, 0]),
    )));
    let noise_mask = noise.then(|| {
        ImageHandle::from_memory(DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            size.0, size.1, NOISE,
        )))
    });
    Inputs {
        original,
        raw_mask,
        mask_data: MaskData {
            schema_version: SCHEMA_VERSION,
            original_path: PathBuf::from("literal-page.png"),
            base_image: ImageHandle::from_memory(DynamicImage::ImageRgb8(RgbImage::from_pixel(
                size.0, size.1, ORIGINAL,
            ))),
            combined_mask: combined,
            scale: 1.0,
            regions,
        },
        noise_mask,
    }
}

fn eligible_region() -> MaskRegionStats {
    MaskRegionStats {
        rect: Rect::new(9, 7, 20, 15),
        std_deviation: 91.0,
        failed: true,
        thickness: None,
    }
}

fn ineligible_region() -> MaskRegionStats {
    MaskRegionStats {
        rect: Rect::new(9, 7, 20, 15),
        std_deviation: 3.0,
        failed: false,
        thickness: None,
    }
}

fn enabled_config() -> InpainterConfig {
    InpainterConfig {
        inpainting_enabled: true,
        inpainting_fade_radius: 0,
        inpainting_isolation_radius: 0,
        min_inpainting_radius: 0,
        max_inpainting_radius: 0,
        inpainting_radius_multiplier: 0.0,
        ..InpainterConfig::default()
    }
}

fn call_adapter(
    inputs: &Inputs,
    config: &InpainterConfig,
    dests: InpaintDests,
    provider: Option<&dyn InpainterProvider>,
) -> Result<Option<InpaintOutput>, PipelineError> {
    run_inpaint(
        &inputs.original,
        &inputs.raw_mask,
        &inputs.mask_data,
        inputs.noise_mask.as_ref(),
        0,
        config,
        dests,
        provider,
    )
}

fn rgba_bytes(handle: &ImageHandle) -> Vec<u8> {
    handle.load().unwrap().to_rgba8().into_raw()
}

fn pixel(handle: &ImageHandle, x: u32, y: u32) -> Rgba<u8> {
    *handle.load().unwrap().to_rgba8().get_pixel(x, y)
}

/// §16.38 item 23(a), item 23(e) case 1, and §16.40 item 2(b): eligibility is
/// decided before image decoding and before provider acquisition. The deliberately broken
/// handles prove that an all-ineligible page returns `None` without touching either input.
///
/// What turns this red: decoding before `select_regions`, asking the provider before the
/// empty-set return, or fabricating an output/artifact for the empty set.
#[test]
fn ineligible_page_returns_none_before_decode_or_provider_request() {
    let mut inputs = memory_inputs(vec![ineligible_region()], false);
    inputs.original = ImageHandle::from_path("missing-original.png");
    inputs.raw_mask = ImageHandle::from_path("missing-raw-mask.png");
    inputs.mask_data.combined_mask = ImageHandle::from_path("missing-combined-mask.png");
    let provider = SpyProvider::refuses(true, "must remain unreachable");
    let dir = tempfile::tempdir().unwrap();
    let dests = InpaintDests {
        inpainting: Some(dir.path().join("should-not-exist-inpainting.png")),
        clean_inpaint: Some(dir.path().join("should-not-exist-clean.png")),
    };

    let output = call_adapter(&inputs, &enabled_config(), dests.clone(), Some(&provider)).unwrap();

    assert!(output.is_none());
    assert_eq!(provider.requests(), 0);
    assert!(!dests.inpainting.unwrap().exists());
    assert!(!dests.clean_inpaint.unwrap().exists());
}

/// §16.38 item 23(e) case 2: the companion to the zero-call case. One eligible page
/// requests the provider exactly once, and the returned model sees the authored original
/// pixel and a non-empty fill mask. The pair prevents an entirely unwired provider from
/// passing the eligibility-first gate vacuously.
#[test]
fn eligible_page_requests_provider_once_and_passes_original_and_fill_mask() {
    let inputs = memory_inputs(vec![eligible_region()], false);
    let model = Arc::new(RecordingInpainter::succeeding());
    let provider = SpyProvider::succeeds(model.clone());

    let output = call_adapter(
        &inputs,
        &enabled_config(),
        InpaintDests::default(),
        Some(&provider),
    )
    .unwrap()
    .expect("eligible page produces an output");

    assert_eq!(provider.requests(), 1);
    assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    assert_eq!(*model.saw_first_tile_pixel.lock().unwrap(), Some(ORIGINAL));
    assert_eq!(*model.saw_nonzero_mask.lock().unwrap(), Some(true));
    assert_eq!(output.tiles_inferred, 1);
    assert_eq!(output.growths, vec![0]);
    assert_eq!(
        pixel(&output.clean_inpaint, 10, 8),
        Rgba([201, 17, 93, 255])
    );
    assert!(pixel(&output.inpainting, 10, 8).0[3] > 0);
}

/// §16.40 item 2(d): enabled + eligible without a provider is a named stage-wiring
/// error, not ineligibility, fallback, or a model error inferred by the adapter.
#[test]
fn eligible_page_without_provider_is_an_explicit_stage_wiring_error() {
    let inputs = memory_inputs(vec![eligible_region()], false);

    let error =
        call_adapter(&inputs, &enabled_config(), InpaintDests::default(), None).unwrap_err();

    assert!(matches!(
        error,
        PipelineError::Stage(StageError::InvalidInput(ref message))
            if message.contains("inpainter") && message.contains("provider")
    ));
}

/// §16.38 item 23(f), §16.38 item 9(g), and cookbook rule 4: acquisition fatality
/// is declared by the provider. The same `StageError::Model` text is classified both ways;
/// changing the variant-based inference cannot satisfy both literal rows.
#[test]
fn acquisition_error_classification_follows_provider_predicate() {
    for (fatal, expected_run_fatal) in [(false, false), (true, true)] {
        let inputs = memory_inputs(vec![eligible_region()], false);
        let provider = SpyProvider::refuses(fatal, "same literal acquisition refusal");
        let error = call_adapter(
            &inputs,
            &enabled_config(),
            InpaintDests::default(),
            Some(&provider),
        )
        .unwrap_err();

        match (expected_run_fatal, error) {
            (false, PipelineError::Stage(StageError::Model(message)))
            | (true, PipelineError::RunFatal(StageError::Model(message))) => {
                assert_eq!(message, "same literal acquisition refusal");
            }
            (_, other) => panic!("wrong classification for fatal={fatal}: {other:?}"),
        }
        assert_eq!(provider.requests(), 1);
    }
}

/// §16.38 item 9(g) and §16.40 item 2(d): after acquisition, an inference failure
/// receives image data and remains per-image even though this provider declares acquisition
/// failures fatal.
#[test]
fn post_acquisition_inference_failure_is_per_image_stage_error() {
    let inputs = memory_inputs(vec![eligible_region()], false);
    let model = Arc::new(RecordingInpainter::failing());
    let provider = SpyProvider::succeeds(model);

    let error = call_adapter(
        &inputs,
        &enabled_config(),
        InpaintDests::default(),
        Some(&provider),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        PipelineError::Stage(StageError::Inference(ref message))
            if message == "literal tile failure"
    ));
    assert_eq!(provider.requests(), 1);
}

/// §16.38 item 24(d) and §16.40 item 5: the Inpaint checkpoint is exactly the two
/// PNGs. Their paths and decoded pixels are asserted independently; path-only
/// `ImageHandle::PartialEq` is deliberately not used.
#[test]
fn disk_success_materializes_both_png_artifacts_at_the_authored_paths() {
    let dir = tempfile::tempdir().unwrap();
    let cache = CachePaths::from_parts(dir.path(), "literal-page", Uuid::nil());
    let dests = inpaint_dests(Some(&cache));
    assert_eq!(dests.inpainting, Some(cache.for_suffix(INPAINTING_SUFFIX)));
    assert_eq!(
        dests.clean_inpaint,
        Some(cache.for_suffix(CLEAN_INPAINT_SUFFIX))
    );
    let inputs = memory_inputs(vec![eligible_region()], false);
    let provider = SpyProvider::succeeds(Arc::new(RecordingInpainter::succeeding()));

    let output = call_adapter(&inputs, &enabled_config(), dests.clone(), Some(&provider))
        .unwrap()
        .unwrap();

    assert_eq!(output.inpainting.path, dests.inpainting);
    assert_eq!(output.clean_inpaint.path, dests.clean_inpaint);
    assert!(output.inpainting.path.as_ref().unwrap().exists());
    assert!(output.clean_inpaint.path.as_ref().unwrap().exists());
    assert_eq!(
        image::open(output.inpainting.path.as_ref().unwrap())
            .unwrap()
            .to_rgba8()
            .into_raw(),
        rgba_bytes(&output.inpainting)
    );
    assert_eq!(
        image::open(output.clean_inpaint.path.as_ref().unwrap())
            .unwrap()
            .to_rgba8()
            .into_raw(),
        rgba_bytes(&output.clean_inpaint)
    );
    assert_eq!(output.inpainting.dimensions().unwrap(), (32, 24));
    assert_eq!(output.clean_inpaint.dimensions().unwrap(), (32, 24));
    assert_eq!(
        pixel(&output.clean_inpaint, 10, 8),
        Rgba([201, 17, 93, 255])
    );
    assert!(pixel(&output.inpainting, 10, 8).0[3] > 0);
    assert_eq!(pixel(&output.clean_inpaint, 0, 0), Rgba([11, 22, 33, 255]));
    let names = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(names.len(), 2, "no JSON inpaint checkpoint: {names:?}");
}

/// §4.1 plus §16.40 item 5: Memory mode returns two path-less handled images with
/// the same literal result pixels and writes no cache artifact.
#[test]
fn memory_success_returns_pathless_images_and_writes_no_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let inputs = memory_inputs(vec![eligible_region()], false);
    let provider = SpyProvider::succeeds(Arc::new(RecordingInpainter::succeeding()));

    let output = call_adapter(
        &inputs,
        &enabled_config(),
        inpaint_dests(None),
        Some(&provider),
    )
    .unwrap()
    .unwrap();

    assert!(output.inpainting.path.is_none());
    assert!(output.clean_inpaint.path.is_none());
    assert_eq!(
        pixel(&output.clean_inpaint, 10, 8),
        Rgba([201, 17, 93, 255])
    );
    assert!(pixel(&output.inpainting, 10, 8).0[3] > 0);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

fn handled_rgba(path: &str, color: Rgba<u8>) -> ImageHandle {
    ImageHandle::with_both(
        path,
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 1, color)),
    )
}

/// §16.38 item 24(d): `export_sources` maps all seven identities. Every field has a
/// different hard-coded pixel, so a swap fails even if paths happen to compare equal.
#[test]
fn export_sources_maps_all_seven_handles_by_identity_and_pixels() {
    let mask = MaskOutput {
        mask_data: memory_inputs(Vec::new(), false).mask_data,
        combined_mask: handled_rgba("final.png", Rgba([41, 0, 0, 255])),
        cleaned: handled_rgba("masked.png", Rgba([42, 0, 0, 255])),
        text_layer: Some(handled_rgba("text.png", Rgba([43, 0, 0, 255]))),
        analytics: Vec::new(),
    };
    let denoise = DenoiseOutput {
        denoised: handled_rgba("denoised.png", Rgba([44, 0, 0, 255])),
        noise_mask: handled_rgba("noise.png", Rgba([45, 0, 0, 255])),
        analytics: pc_core::DenoiseAnalytic {
            path: PathBuf::from("literal-page.png"),
            std_deviations: Vec::new(),
            boxes_denoised: 0,
        },
    };
    let inpaint = InpaintOutput {
        inpainting: handled_rgba("inpainting.png", Rgba([46, 0, 0, 255])),
        clean_inpaint: handled_rgba("clean-inpaint.png", Rgba([47, 0, 0, 255])),
        growths: vec![0],
        tiles_inferred: 1,
    };

    let sources = export_sources(Some(&mask), Some(&denoise), Some(&inpaint));

    let rows = [
        (sources.masked.as_ref(), "masked.png", 42),
        (sources.denoised.as_ref(), "denoised.png", 44),
        (sources.inpainted.as_ref(), "clean-inpaint.png", 47),
        (sources.final_mask.as_ref(), "final.png", 41),
        (sources.denoise_mask.as_ref(), "noise.png", 45),
        (sources.inpainted_mask.as_ref(), "inpainting.png", 46),
        (sources.isolated_text.as_ref(), "text.png", 43),
    ];
    assert_eq!(rows.len(), 7);
    for (handle, expected_path, expected_red) in rows {
        let handle = handle.expect("all seven sources are populated");
        assert_eq!(handle.path.as_deref(), Some(Path::new(expected_path)));
        assert_eq!(pixel(handle, 0, 0), Rgba([expected_red, 0, 0, 255]));
    }
}

/// §16.38 item 25(c): `noise_mask` is handed to `pc_inpaint` after denoising. An
/// opaque literal noise layer changes a pixel outside the inpainting write region; omitting
/// it leaves the authored original pixel there.
#[test]
fn denoise_mask_some_and_none_are_distinguishable_in_clean_inpaint() {
    let without = memory_inputs(vec![eligible_region()], false);
    let with = memory_inputs(vec![eligible_region()], true);
    let provider_without = SpyProvider::succeeds(Arc::new(RecordingInpainter::succeeding()));
    let provider_with = SpyProvider::succeeds(Arc::new(RecordingInpainter::succeeding()));

    let clean_without = call_adapter(
        &without,
        &enabled_config(),
        InpaintDests::default(),
        Some(&provider_without),
    )
    .unwrap()
    .unwrap()
    .clean_inpaint;
    let clean_with = call_adapter(
        &with,
        &enabled_config(),
        InpaintDests::default(),
        Some(&provider_with),
    )
    .unwrap()
    .unwrap()
    .clean_inpaint;

    assert_eq!(pixel(&clean_without, 0, 0), Rgba([11, 22, 33, 255]));
    assert_eq!(pixel(&clean_with, 0, 0), NOISE);
}
