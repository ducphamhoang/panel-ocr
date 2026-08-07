//! Shared fixtures for the frozen `pc-export` test suites. FROZEN with the tests.
#![allow(dead_code)]

use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage};
use pc_core::{ImageHandle, OcrAnalytic, Output, Rect, RemovedBox, SCHEMA_VERSION};
use pc_export::{ExportInput, ExportSources};
use std::path::{Path, PathBuf};

/// Small enough to keep the wiring tests instant, large enough that a 2x mask upscale
/// has something to interpolate wrongly if someone swaps nearest for bilinear.
pub const PAGE_SIZE: (u32, u32) = (40, 24);

/// A grayscale page: `background` everywhere, `value` inside each rect.
pub fn gray_page(size: (u32, u32), background: u8, marks: &[(Rect, u8)]) -> DynamicImage {
    let mut image = GrayImage::from_pixel(size.0, size.1, Luma([background]));
    for (rect, value) in marks {
        if let Some((x, y, w, h)) = rect.to_crop(size) {
            for py in y..y + h {
                for px in x..x + w {
                    image.put_pixel(px, py, Luma([*value]));
                }
            }
        }
    }
    DynamicImage::ImageLuma8(image)
}

/// A combined-mask-shaped RGBA image: transparent everywhere, `fill` inside each rect.
pub fn rgba_mask(size: (u32, u32), fills: &[(Rect, [u8; 4])]) -> RgbaImage {
    let mut mask = RgbaImage::from_pixel(size.0, size.1, Rgba([0, 0, 0, 0]));
    for (rect, fill) in fills {
        if let Some((x, y, w, h)) = rect.to_crop(size) {
            for py in y..y + h {
                for px in x..x + w {
                    mask.put_pixel(px, py, Rgba(*fill));
                }
            }
        }
    }
    mask
}

/// An in-memory handle that still carries a path, so the value stays serializable
/// (§2.3) while `load()` never touches the filesystem.
pub fn handle(path: impl AsRef<Path>, image: DynamicImage) -> ImageHandle {
    ImageHandle::with_both(path.as_ref().to_path_buf(), image)
}

/// Every category requested (`--save-only-*` absent) — §16.11 item 2.
pub fn all_outputs() -> Vec<Output> {
    pc_export::discover::all_exportable_outputs()
}

/// `--save-only-cleaned` / `--save-only-mask` / `--save-only-text`, as the CLI would
/// populate `ExportInput.outputs`.
pub fn only_cleaned() -> Vec<Output> {
    vec![Output::MaskedOutput, Output::DenoisedOutput]
}

pub fn only_mask() -> Vec<Output> {
    vec![Output::FinalMask, Output::DenoiseMask]
}

pub fn only_text() -> Vec<Output> {
    vec![Output::IsolatedText]
}

/// A default input: a grayscale original on disk, a masked and a denoised cleaned
/// artifact in memory, a combined mask, a noise mask and a text layer. `output_dir` is
/// absolute (`base`), so destinations land exactly there.
pub fn input_with(
    original_path: &Path,
    base: &Path,
    sources: ExportSources,
    outputs: Vec<Output>,
    denoising_enabled: bool,
) -> ExportInput {
    ExportInput {
        schema_version: SCHEMA_VERSION,
        original_path: original_path.to_path_buf(),
        export_path: original_path.to_path_buf(),
        output_dir: base.to_path_buf(),
        outputs,
        sources,
        preferred_file_type: None,
        preferred_mask_file_type: ".png".into(),
        denoising_enabled,
        inpainting_enabled: false,
    }
}

/// The full set of source artifacts, all in memory, all distinguishable by pixel value.
pub fn full_sources() -> ExportSources {
    let masked = gray_page(PAGE_SIZE, 200, &[(Rect::new(4, 4, 12, 12), 10)]);
    let denoised = gray_page(PAGE_SIZE, 200, &[(Rect::new(4, 4, 12, 12), 20)]);
    let final_mask = rgba_mask(
        PAGE_SIZE,
        &[(Rect::new(4, 4, 12, 12), [255, 255, 255, 255])],
    );
    let denoise_mask = rgba_mask(PAGE_SIZE, &[(Rect::new(6, 6, 10, 10), [0, 0, 0, 255])]);
    let text = rgba_mask(PAGE_SIZE, &[(Rect::new(5, 5, 9, 9), [0, 0, 0, 255])]);
    ExportSources {
        masked: Some(handle("/cache/page_clean.png", masked)),
        denoised: Some(handle("/cache/page_clean_denoised.png", denoised)),
        inpainted: None,
        final_mask: Some(handle(
            "/cache/page_combined_mask.png",
            DynamicImage::ImageRgba8(final_mask),
        )),
        denoise_mask: Some(handle(
            "/cache/page_noise_mask.png",
            DynamicImage::ImageRgba8(denoise_mask),
        )),
        inpainted_mask: None,
        isolated_text: Some(handle(
            "/cache/page_text.png",
            DynamicImage::ImageRgba8(text),
        )),
    }
}

/// Write a grayscale original of `PAGE_SIZE` at `dir/page.png` and return its path.
pub fn write_original_png(dir: &Path) -> PathBuf {
    let path = dir.join("page.png");
    gray_page(PAGE_SIZE, 200, &[(Rect::new(4, 4, 12, 12), 30)])
        .save_with_format(&path, image::ImageFormat::Png)
        .expect("write the synthetic original");
    path
}

/// §12.6's CSV fixture, as `OcrAnalytic` values: `img1.jpg` with two boxes.
pub fn csv_fixture_analytics() -> Vec<OcrAnalytic> {
    vec![analytic(
        "img1.jpg",
        &[
            (Rect::new(100, 100, 300, 200), "some text perhaps"),
            (Rect::new(534, 275, 592, 414), "or nothing at all"),
        ],
    )]
}

/// §12.6's TXT fixture, as `OcrAnalytic` values: `page1.jpg` (2 lines) then
/// `page2.jpg` (3 lines). The rects are irrelevant to the TXT format.
pub fn txt_fixture_analytics() -> Vec<OcrAnalytic> {
    vec![
        analytic(
            "page1.jpg",
            &[
                (Rect::new(0, 0, 1, 1), "hello everyone."),
                (Rect::new(0, 0, 1, 1), "I wish I were a bird."),
            ],
        ),
        analytic(
            "page2.jpg",
            &[
                (Rect::new(0, 0, 1, 1), "Such a thing is impossible."),
                (Rect::new(0, 0, 1, 1), "Truly a shame."),
                (Rect::new(0, 0, 1, 1), "A different wish will suffice."),
            ],
        ),
    ]
}

/// §16.11 item 11: the report reads `removed`, which in the `run_ocr` code path is the
/// complete box list (§15.5 — the blacklist is `.*` there).
pub fn analytic(path: &str, boxes: &[(Rect, &str)]) -> OcrAnalytic {
    OcrAnalytic {
        path: PathBuf::from(path),
        num_boxes: boxes.len(),
        box_areas_ocred: Vec::new(),
        box_areas_removed: boxes.iter().map(|(rect, _)| rect.area()).collect(),
        removed: boxes
            .iter()
            .map(|(rect, text)| RemovedBox {
                text: (*text).to_string(),
                rect: *rect,
            })
            .collect(),
    }
}

/// Fixture text with its trailing newline(s) removed — §12.6 mandates comparing
/// trailing-newline-normalized, because the vendored fixtures lack one.
pub fn read_fixture_trimmed(path: &Path) -> String {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read `{}`: {error}", path.display()));
    text.trim_end_matches('\n').to_string()
}
