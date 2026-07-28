//! Shared fixtures for the frozen `pc-mask` test suites. FROZEN with the tests.
#![allow(dead_code)]

use image::{DynamicImage, GrayImage, Luma, RgbImage};
use pc_config::MaskerConfig;
use pc_core::{ImageHandle, MaskingRegion, PageData, Rect, TextBox};
use pc_imageops::BinaryMask;
use pc_mask::{MaskDests, MaskInput};
use std::path::PathBuf;

/// Big enough that the default growth (up to 24 px) never clips against the canvas by
/// accident, so a clip that shows up in an assertion is the clip the test is about.
pub const PAGE_SIZE: (u32, u32) = (200, 160);

pub const ORIGINAL_PATH: &str = "/synthetic/page.png";
pub const BASE_IMAGE_PATH: &str = "/synthetic/page_base.png";
pub const RAW_MASK_PATH: &str = "/synthetic/page_raw_mask.png";
pub const ORIGINAL_IMAGE_PATH: &str = "/synthetic/page_original.png";

/// Every set pixel, row-major.
pub fn set_pixels(mask: &BinaryMask) -> Vec<(u32, u32)> {
    let (width, height) = mask.dimensions();
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if mask.get(x, y) {
                pixels.push((x, y));
            }
        }
    }
    pixels
}

/// A grayscale page: `background` everywhere, `0` inside each text rect.
pub fn uniform_base(size: (u32, u32), background: u8, text: &[Rect]) -> DynamicImage {
    let mut image = GrayImage::from_pixel(size.0, size.1, Luma([background]));
    for rect in text {
        if let Some((x, y, width, height)) = rect.to_crop(size) {
            for pixel_y in y..y + height {
                for pixel_x in x..x + width {
                    image.put_pixel(pixel_x, pixel_y, Luma([0]));
                }
            }
        }
    }
    DynamicImage::ImageLuma8(image)
}

/// The matching detector mask: 255 inside each text rect, 0 elsewhere.
pub fn text_raw_mask(size: (u32, u32), text: &[Rect]) -> GrayImage {
    let mut mask = GrayImage::new(size.0, size.1);
    for rect in text {
        if let Some((x, y, width, height)) = rect.to_crop(size) {
            for pixel_y in y..y + height {
                for pixel_x in x..x + width {
                    mask.put_pixel(pixel_x, pixel_y, Luma([255]));
                }
            }
        }
    }
    mask
}

/// An RGB page with a solid `background`, `0,0,0` inside each text rect.
pub fn uniform_base_rgb(size: (u32, u32), background: [u8; 3], text: &[Rect]) -> DynamicImage {
    let mut image = RgbImage::from_pixel(size.0, size.1, image::Rgb(background));
    for rect in text {
        if let Some((x, y, width, height)) = rect.to_crop(size) {
            for pixel_y in y..y + height {
                for pixel_x in x..x + width {
                    image.put_pixel(pixel_x, pixel_y, image::Rgb([0, 0, 0]));
                }
            }
        }
    }
    DynamicImage::ImageRgb8(image)
}

/// A `PageData` whose handles carry decoded images in cache (so `load()` needs no file)
/// while keeping a `path`, so the page stays serializable (§2.3, §16.8 item 13).
///
/// One masking region per `boxes` entry: `masking` is the box itself, `reference` is it
/// padded by `reference_pad` -- the §2.5 invariant `reference` contains `masking` holds
/// by construction.
pub fn page(
    base: DynamicImage,
    raw_mask: GrayImage,
    boxes: &[Rect],
    reference_pad: i32,
    scale: f64,
) -> PageData {
    let image_size = (base.width(), base.height());
    PageData {
        schema_version: pc_core::SCHEMA_VERSION,
        original_path: PathBuf::from(ORIGINAL_PATH),
        base_image: ImageHandle::with_both(BASE_IMAGE_PATH, base),
        raw_mask: ImageHandle::with_both(RAW_MASK_PATH, DynamicImage::ImageLuma8(raw_mask)),
        scale,
        image_size,
        page_language: None,
        text_boxes: boxes
            .iter()
            .map(|rect| TextBox {
                rect: *rect,
                language: None,
            })
            .collect(),
        extended_boxes: boxes.to_vec(),
        masking_regions: boxes
            .iter()
            .map(|rect| MaskingRegion {
                masking: *rect,
                reference: rect.pad(reference_pad, image_size),
            })
            .collect(),
    }
}

/// The canonical synthetic page: a uniform light background with one dark text block,
/// one masking region around it. Every candidate's border sits on the uniform
/// background, so the border deviation is exactly `0.0` -- which is what makes the
/// composition and determinism gates hand-checkable.
pub fn simple_page() -> PageData {
    let text = [Rect::new(80, 60, 120, 100)];
    page(
        uniform_base(PAGE_SIZE, 200, &text),
        text_raw_mask(PAGE_SIZE, &text),
        &[Rect::new(70, 50, 130, 110)],
        20,
        1.0,
    )
}

/// spec §10.3 step 0: the precise mask, thresholded at PIL's `> 127`.
pub fn precise_of(page: &PageData) -> BinaryMask {
    let raw = page.raw_mask.load().expect("synthetic raw mask is cached");
    BinaryMask::from_gray_threshold(&raw.to_luma8(), pc_imageops::PIL_BINARY_THRESHOLD)
}

/// spec §10.3 step 1: the box mask over `extended_boxes` (exclusive rects, §16.9 item 1).
pub fn box_mask_of(page: &PageData) -> BinaryMask {
    pc_imageops::rasterize_boxes(&page.extended_boxes, page.image_size)
}

/// spec §10.3 step 2: `cut = precise AND box_mask`.
pub fn cut_of(page: &PageData) -> BinaryMask {
    precise_of(page).and(&box_mask_of(page))
}

/// spec §10.3 step 0: the analysis canvas.
pub fn canvas_of(page: &PageData) -> pc_mask::BaseCanvas {
    let base = page.base_image.load().expect("synthetic base is cached");
    pc_mask::BaseCanvas::from_dynamic(&base)
}

/// A `MaskInput` in memory mode (every destination `None`, §4.1).
pub fn memory_input(page: PageData, config: MaskerConfig) -> MaskInput {
    let original = page
        .base_image
        .load()
        .expect("synthetic base image is cached")
        .as_ref()
        .clone();
    MaskInput {
        schema_version: pc_core::SCHEMA_VERSION,
        page,
        original_image: ImageHandle::with_both(ORIGINAL_IMAGE_PATH, original),
        config,
        extract_text: false,
        debug_outputs: false,
        dests: MaskDests::default(),
    }
}

/// Every destination pointed into `dir`, with upstream's suffixes (§2.8).
pub fn disk_dests(dir: &std::path::Path) -> MaskDests {
    MaskDests {
        combined_mask: Some(dir.join("page_combined_mask.png")),
        cleaned: Some(dir.join("page_clean.png")),
        text_layer: Some(dir.join("page_text.png")),
        box_mask: Some(dir.join("page_box_mask.png")),
        cut_mask: Some(dir.join("page_cut_mask.png")),
        mask_overlay: Some(dir.join("page_with_masks.png")),
    }
}
