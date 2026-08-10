//! §16.45 item 6 — the **second, independent** instance of the transparent-destination
//! compositing defect, gated at `pc-export`'s own mask-export call site.
//!
//! §16.45 item 6 is explicit that the `pc-denoise` gate does not cover this one: the
//! destination here is a *different* artifact (the combined mask resized to the original's
//! size, which is transparent everywhere outside the mask footprint) and the product is a
//! *different* file (`_mask.png`). The padded, faded noise rim reaches beyond the mask
//! footprint, so it lands on transparent destination pixels — exactly the regime
//! `alpha_composite_over` got wrong. Treating this as structurally covered by the
//! `pc-denoise` test would be `docs/COOKBOOK.md`'s "gate a copy of the risk, not the risk".
//!
//! The expected values are not read back from any run: the correct answer for a
//! `dst_alpha == 0` destination is the layer pixel itself, by §16.45 item 4's `da == 0`
//! reduction. §16.45 item 6 records the *unfixed* output for this same shape, measured on
//! the live code before the fix: a `[200,100,50,64]` noise pixel exported as
//! `[50,25,13,64]`, i.e. the layer's colour premultiplied once by `64/255`.

use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage};
use pc_core::{ImageHandle, Output, SCHEMA_VERSION};
use pc_export::{run, ExportInput, ExportSources};
use std::path::{Path, PathBuf};

/// The noise-mask rim pixel under test: partially transparent, and with three distinct,
/// non-zero channels so a premultiplication shows a different error on each.
const NOISE_RIM: [u8; 4] = [200, 100, 50, 64];

/// The combined mask's own footprint pixel: fully opaque, so it exercises the untouched
/// `dst_alpha == 255` regime in the same export.
const MASK_FOOTPRINT: [u8; 4] = [255, 255, 255, 255];

fn handle(path: &str, image: DynamicImage) -> ImageHandle {
    ImageHandle::with_both(PathBuf::from(path), image)
}

fn write_original(dir: &Path) -> PathBuf {
    let path = dir.join("page.png");
    DynamicImage::ImageLuma8(GrayImage::from_pixel(2, 2, Luma([33])))
        .save(&path)
        .expect("write the 2x2 original");
    path
}

/// A 2x2 RGBA image: `pixel` at `(1, 1)`, fully transparent everywhere else.
fn only_bottom_right(pixel: [u8; 4]) -> DynamicImage {
    let mut image = RgbaImage::from_pixel(2, 2, Rgba([0, 0, 0, 0]));
    image.put_pixel(1, 1, Rgba(pixel));
    DynamicImage::ImageRgba8(image)
}

/// A 2x2 RGBA image: `pixel` at `(0, 0)`, fully transparent everywhere else — the shape
/// `pc_mask::build_combined_mask` produces (opaque inside the footprint, `(0,0,0,0)`
/// outside it).
fn only_top_left(pixel: [u8; 4]) -> DynamicImage {
    let mut image = RgbaImage::from_pixel(2, 2, Rgba([0, 0, 0, 0]));
    image.put_pixel(0, 0, Rgba(pixel));
    DynamicImage::ImageRgba8(image)
}

/// §16.45 item 6: a faded noise-mask pixel that lands **outside** the combined mask's
/// footprint must be exported with the noise layer's own colour and alpha, unchanged.
///
/// What turns this red: `alpha_composite_over` ignoring the destination's alpha. Under
/// that formula this exact input exports `[50, 25, 13, 64]` — `round(200·64/255) = 50`,
/// `round(100·64/255) = 25`, `round(50·64/255) = 13`, `alpha_out = max(0, 64) = 64` —
/// which is the value §16.45 item 6 records as measured on the unfixed code.
#[test]
fn an_exported_noise_pixel_outside_the_mask_footprint_keeps_the_noise_layers_own_colour() {
    let dir = tempfile::tempdir().expect("temp dir");
    let original = write_original(dir.path());
    let output_dir = dir.path().join("out");

    let sources = ExportSources {
        final_mask: Some(handle(
            "/cache/final_mask.png",
            only_top_left(MASK_FOOTPRINT),
        )),
        denoise_mask: Some(handle(
            "/cache/noise_mask.png",
            only_bottom_right(NOISE_RIM),
        )),
        ..ExportSources::default()
    };

    run(ExportInput {
        schema_version: SCHEMA_VERSION,
        original_path: original.clone(),
        export_path: original,
        output_dir: output_dir.clone(),
        outputs: vec![Output::FinalMask, Output::DenoiseMask],
        sources,
        preferred_file_type: None,
        preferred_mask_file_type: ".png".into(),
        denoising_enabled: true,
        inpainting_enabled: false,
    })
    .expect("mask export succeeds");

    let exported = pc_testkit::images::load_rgba8(output_dir.join("page_mask.png"));
    assert_eq!(exported.dimensions(), (2, 2));

    assert_eq!(
        exported.get_pixel(1, 1).0,
        NOISE_RIM,
        "a noise pixel over a fully transparent destination must export as the noise \
         layer itself, not premultiplied against the empty canvas (§16.45 item 6)"
    );

    // Anti-vacuity in the same artifact: the combined mask's own footprint must still be
    // present and untouched, so the assertion above cannot be satisfied by an export that
    // simply dropped the combined mask and wrote the noise layer alone.
    assert_eq!(
        exported.get_pixel(0, 0).0,
        MASK_FOOTPRINT,
        "the combined mask's opaque footprint must survive the noise composite unchanged"
    );

    // ... and the two pixels the neither layer covers stay fully transparent, so the
    // export is not simply flooding the canvas.
    assert_eq!(exported.get_pixel(1, 0).0, [0, 0, 0, 0]);
    assert_eq!(exported.get_pixel(0, 1).0, [0, 0, 0, 0]);
}

/// The same call site under the `WithInpaint` branch (`export_mask`'s third arm composites
/// the noise layer and then the inpainting layer, each with its own
/// `alpha_composite_over`): a faded **inpainting** pixel landing outside every other
/// layer's footprint must likewise export unchanged.
///
/// Hand-derived rather than run: the destination at `(1, 1)` is `(0,0,0,0)` after the
/// combined mask (footprint at `(0,0)` only) and after the absent-here noise layer, so
/// §16.45 item 4's `da == 0` reduction gives the inpainting pixel itself.
///
/// What turns this red: the destination-alpha-ignoring blend on the `WithInpaint` arm;
/// it exports `[round(10·64/255), round(210·64/255), round(30·64/255), 64]`
/// = `[3, 53, 8, 64]`.
#[test]
fn an_exported_inpainting_pixel_outside_every_footprint_keeps_its_own_colour() {
    let dir = tempfile::tempdir().expect("temp dir");
    let original = write_original(dir.path());
    let output_dir = dir.path().join("out");

    let inpainting = [10_u8, 210, 30, 64];
    let sources = ExportSources {
        final_mask: Some(handle(
            "/cache/final_mask.png",
            only_top_left(MASK_FOOTPRINT),
        )),
        inpainted_mask: Some(handle(
            "/cache/inpainting.png",
            only_bottom_right(inpainting),
        )),
        ..ExportSources::default()
    };

    run(ExportInput {
        schema_version: SCHEMA_VERSION,
        original_path: original.clone(),
        export_path: original,
        output_dir: output_dir.clone(),
        outputs: vec![Output::FinalMask, Output::DenoiseMask],
        sources,
        preferred_file_type: None,
        preferred_mask_file_type: ".png".into(),
        denoising_enabled: false,
        inpainting_enabled: true,
    })
    .expect("mask export succeeds");

    let exported = pc_testkit::images::load_rgba8(output_dir.join("page_mask.png"));
    assert_eq!(
        exported.get_pixel(1, 1).0,
        inpainting,
        "an inpainting pixel over a fully transparent destination must export as the \
         inpainting layer itself (§16.45 items 4 and 6)"
    );
    assert_eq!(
        exported.get_pixel(0, 0).0,
        MASK_FOOTPRINT,
        "the combined mask's opaque footprint must survive the inpainting composite"
    );
}
