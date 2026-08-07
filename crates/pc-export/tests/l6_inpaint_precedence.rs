//! Task **L6-3** — inpainting export precedence and mask composition.
//! Requirements: §16.38 items 12(a)-(b), 14(c), 22(c)-(h), and 24(c). FROZEN.

use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use pc_core::{ImageHandle, Output, SCHEMA_VERSION};
use pc_export::discover::{resolve, MaskChoice};
use pc_export::{run, ColorMode, ExportInput, ExportSources};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Record};
use tracing::{Event, Id, Level, Metadata, Subscriber};

fn handle(path: &str, image: DynamicImage) -> ImageHandle {
    ImageHandle::with_both(PathBuf::from(path), image)
}

fn handle_path(handle: &ImageHandle) -> &Path {
    handle.path.as_deref().expect("test handles carry paths")
}

fn mask_outputs() -> Vec<Output> {
    vec![Output::FinalMask, Output::DenoiseMask]
}

fn cleaned_outputs() -> Vec<Output> {
    vec![Output::MaskedOutput, Output::DenoisedOutput]
}

fn solid_rgba(pixel: [u8; 4]) -> DynamicImage {
    DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, Rgba(pixel)))
}

fn write_grayscale_original(dir: &Path) -> PathBuf {
    let path = dir.join("page.png");
    DynamicImage::ImageLuma8(GrayImage::from_pixel(1, 1, Luma([33])))
        .save(&path)
        .expect("write grayscale original");
    path
}

fn input(
    original_path: &Path,
    output_dir: &Path,
    outputs: Vec<Output>,
    sources: ExportSources,
    denoising_enabled: bool,
    inpainting_enabled: bool,
) -> ExportInput {
    ExportInput {
        schema_version: SCHEMA_VERSION,
        original_path: original_path.to_path_buf(),
        export_path: original_path.to_path_buf(),
        output_dir: output_dir.to_path_buf(),
        outputs,
        sources,
        preferred_file_type: None,
        preferred_mask_file_type: ".png".into(),
        denoising_enabled,
        inpainting_enabled,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedEvent {
    level: Level,
    fields: BTreeMap<String, String>,
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<CapturedEvent>>>);

impl Capture {
    fn events(&self) -> Vec<CapturedEvent> {
        self.0.lock().expect("capture lock").clone()
    }
}

#[derive(Default)]
struct FieldVisitor(BTreeMap<String, String>);

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().into(), format!("{value:?}"));
    }
}

impl Subscriber for Capture {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        self.0.lock().expect("capture lock").push(CapturedEvent {
            level: *event.metadata().level(),
            fields: visitor.0,
        });
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}

    fn max_level_hint(&self) -> Option<tracing::metadata::LevelFilter> {
        Some(tracing::metadata::LevelFilter::TRACE)
    }
}

#[test]
fn inpainted_cleaned_source_precedes_denoised_and_masked_when_inpainting_is_enabled() {
    // §16.38 item 12(a), cleaned half: inpainted > the existing denoised > masked
    // precedence. Paths are asserted explicitly because ImageHandle equality is path-only.
    let sources = ExportSources {
        masked: Some(handle("/cache/masked.png", solid_rgba([11, 11, 11, 255]))),
        denoised: Some(handle("/cache/denoised.png", solid_rgba([22, 22, 22, 255]))),
        inpainted: Some(handle(
            "/cache/inpainted.png",
            solid_rgba([33, 33, 33, 255]),
        )),
        ..ExportSources::default()
    };

    let selection = resolve(&sources, &cleaned_outputs(), true, true);
    assert_eq!(
        handle_path(selection.cleaned.as_ref().expect("cleaned selected")),
        Path::new("/cache/inpainted.png")
    );
}

#[test]
fn disabling_inpainting_drops_stale_cleaned_and_mask_artifacts() {
    // §16.38 item 12(b): populated inpainting handles are stale when inpainting is
    // disabled. The independently named fallback paths pin identity, not cardinality.
    let sources = ExportSources {
        masked: Some(handle("/cache/masked.png", solid_rgba([11, 11, 11, 255]))),
        denoised: Some(handle("/cache/denoised.png", solid_rgba([22, 22, 22, 255]))),
        inpainted: Some(handle(
            "/cache/stale_clean_inpaint.png",
            solid_rgba([33, 33, 33, 255]),
        )),
        final_mask: Some(handle(
            "/cache/final_mask.png",
            solid_rgba([44, 44, 44, 255]),
        )),
        denoise_mask: Some(handle(
            "/cache/noise_mask.png",
            solid_rgba([55, 55, 55, 255]),
        )),
        inpainted_mask: Some(handle(
            "/cache/stale_inpainting.png",
            solid_rgba([66, 66, 66, 255]),
        )),
        ..ExportSources::default()
    };

    let selection = resolve(
        &sources,
        &[Output::MaskedOutput, Output::FinalMask],
        true,
        false,
    );
    assert_eq!(
        handle_path(selection.cleaned.as_ref().expect("fallback cleaned")),
        Path::new("/cache/denoised.png")
    );
    match selection.mask.expect("fallback mask") {
        MaskChoice::WithDenoise {
            final_mask,
            denoise_mask,
        } => {
            assert_eq!(handle_path(&final_mask), Path::new("/cache/final_mask.png"));
            assert_eq!(
                handle_path(&denoise_mask),
                Path::new("/cache/noise_mask.png")
            );
        }
        other => panic!("expected existing denoise composite, got {other:?}"),
    }
}

#[test]
fn with_inpaint_resolution_keeps_noise_only_when_denoising_is_enabled() {
    // §16.38 items 22(c)-(d): one WithInpaint variant carries an optional noise layer.
    // The two cases assert each handle's identity and the denoising guard explicitly.
    let sources = ExportSources {
        final_mask: Some(handle(
            "/cache/final_mask.png",
            solid_rgba([10, 20, 30, 255]),
        )),
        denoise_mask: Some(handle(
            "/cache/noise_mask.png",
            solid_rgba([40, 50, 60, 255]),
        )),
        inpainted_mask: Some(handle(
            "/cache/inpainting.png",
            solid_rgba([70, 80, 90, 255]),
        )),
        ..ExportSources::default()
    };

    for (denoising_enabled, expected_noise_path) in [
        (true, Some(Path::new("/cache/noise_mask.png"))),
        (false, None),
    ] {
        let selection = resolve(&sources, &mask_outputs(), denoising_enabled, true);
        match selection.mask.expect("inpainting mask selected") {
            MaskChoice::WithInpaint {
                final_mask,
                denoise_mask,
                inpainted_mask,
            } => {
                assert_eq!(handle_path(&final_mask), Path::new("/cache/final_mask.png"));
                assert_eq!(denoise_mask.as_ref().map(handle_path), expected_noise_path);
                assert_eq!(
                    handle_path(&inpainted_mask),
                    Path::new("/cache/inpainting.png")
                );
            }
            other => panic!("expected WithInpaint, got {other:?}"),
        }
    }
}

#[test]
fn an_inpainted_mask_without_a_final_mask_selects_no_mask() {
    // §16.38 item 22(e): the inpainting artifact cannot replace the absent base; this
    // takes the existing warn-and-export-no-mask path.
    let sources = ExportSources {
        inpainted_mask: Some(handle(
            "/cache/orphan_inpainting.png",
            solid_rgba([70, 80, 90, 255]),
        )),
        ..ExportSources::default()
    };

    let capture = Capture::default();
    let selection = tracing::subscriber::with_default(capture.clone(), || {
        resolve(&sources, &mask_outputs(), false, true)
    });
    assert!(
        selection.mask.is_none(),
        "an orphan inpainting layer must not become an exported mask"
    );

    let warnings = capture
        .events()
        .into_iter()
        .filter(|event| event.level == Level::WARN)
        .collect::<Vec<_>>();
    assert_eq!(warnings.len(), 1, "the orphan emits exactly one WARN");
    let warning_fields = warnings[0]
        .fields
        .values()
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        warning_fields.contains("inpainting"),
        "WARN must identify the inpainting artifact: {warning_fields}"
    );
    assert_ne!(
        warning_fields,
        "\"a denoise mask was available without a combined mask; no mask will be exported\"",
        "the existing denoise-only warning must not be reused unchanged"
    );
}

#[test]
fn exported_inpaint_mask_pixel_is_combined_then_noise_then_inpainting() {
    // §16.38 items 22(f), 22(h), and 24(c). Hand trace using
    // round(base * (1-alpha) + layer * alpha):
    // [20,40,60,180] overlaid by [220,20,100,128] => [120,30,80,180], then
    // [10,210,30,64] => [92,75,67,180]. Every permutation of these three source
    // colours yields a different pixel; the literal therefore pins identity and order.
    let dir = tempfile::tempdir().expect("temp dir");
    let original = write_grayscale_original(dir.path());
    let output_dir = dir.path().join("out");
    let sources = ExportSources {
        final_mask: Some(handle(
            "/cache/final_mask.png",
            solid_rgba([20, 40, 60, 180]),
        )),
        denoise_mask: Some(handle(
            "/cache/noise_mask.png",
            solid_rgba([220, 20, 100, 128]),
        )),
        inpainted_mask: Some(handle(
            "/cache/inpainting.png",
            solid_rgba([10, 210, 30, 64]),
        )),
        ..ExportSources::default()
    };

    run(input(
        &original,
        &output_dir,
        mask_outputs(),
        sources,
        true,
        true,
    ))
    .expect("mask export succeeds");

    let exported = pc_testkit::images::load_rgba8(output_dir.join("page_mask.png"));
    assert_eq!(exported.get_pixel(0, 0).0, [92, 75, 67, 180]);
}

#[test]
fn exported_inpaint_mask_omits_noise_when_denoising_is_disabled() {
    // §16.38 items 22(b)(ii), 22(f), and 22(h): disabled denoising gives the two-layer
    // base -> inpainting result. The stale noise handle is deliberately still populated.
    // Hand trace: [20,40,60,180] overlaid by [10,210,30,64] => [17,83,52,180].
    let dir = tempfile::tempdir().expect("temp dir");
    let original = write_grayscale_original(dir.path());
    let output_dir = dir.path().join("out");
    let sources = ExportSources {
        final_mask: Some(handle(
            "/cache/final_mask.png",
            solid_rgba([20, 40, 60, 180]),
        )),
        denoise_mask: Some(handle(
            "/cache/stale_noise_mask.png",
            solid_rgba([220, 20, 100, 128]),
        )),
        inpainted_mask: Some(handle(
            "/cache/inpainting.png",
            solid_rgba([10, 210, 30, 64]),
        )),
        ..ExportSources::default()
    };

    run(input(
        &original,
        &output_dir,
        mask_outputs(),
        sources,
        false,
        true,
    ))
    .expect("mask export succeeds");

    let exported = pc_testkit::images::load_rgba8(output_dir.join("page_mask.png"));
    assert_eq!(exported.get_pixel(0, 0).0, [17, 83, 52, 180]);
}

#[test]
fn inpainted_cleaned_source_exports_in_the_grayscale_originals_colour_mode() {
    // §16.38 items 14(c) and 24(c): an RGB `_clean_inpaint.png` must pass through the
    // existing export_cleaned seam with the grayscale original's ColorMode::L.
    let dir = tempfile::tempdir().expect("temp dir");
    let original = write_grayscale_original(dir.path());
    let output_dir = dir.path().join("out");
    let sources = ExportSources {
        masked: Some(handle(
            "/cache/masked.png",
            DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([10, 10, 10]))),
        )),
        inpainted: Some(handle(
            "/cache/clean_inpaint.png",
            DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([90, 120, 150]))),
        )),
        ..ExportSources::default()
    };

    run(input(
        &original,
        &output_dir,
        cleaned_outputs(),
        sources,
        false,
        true,
    ))
    .expect("cleaned export succeeds");

    assert_eq!(
        pc_export::formats::read_color_mode(&output_dir.join("page_clean.png"))
            .expect("read exported header"),
        ColorMode::L
    );
}
