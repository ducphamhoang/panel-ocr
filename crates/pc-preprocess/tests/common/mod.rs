//! Shared fixtures for the frozen `pc-preprocess` test suites. FROZEN with the tests.
#![allow(dead_code)]

use image::DynamicImage;
use pc_config::PreprocessorConfig;
use pc_core::{DetectedBlock, ImageHandle, Language, PageDataRaw, Rect, TextBox};
use pc_ocr::OcrEngineFactory;
use std::path::PathBuf;

/// Every synthetic page in these suites is 1000x1000 unless the test says otherwise —
/// big enough that the default padding tiers never clamp by accident, so a clamp that
/// shows up in an assertion is the clamp the test is actually about.
pub const PAGE_SIZE: (u32, u32) = (1000, 1000);

pub const ORIGINAL_PATH: &str = "/synthetic/page.png";
pub const BASE_IMAGE_PATH: &str = "/synthetic/page_base.png";
pub const RAW_MASK_PATH: &str = "/synthetic/page_raw_mask.png";

/// spec §16.8 item 13: the preprocess stage never *creates* an `ImageHandle`, it passes
/// the input page's through. Path-bearing handles are therefore the realistic case and
/// the one the §9.7(A)10 JSON determinism gate needs.
pub fn base_handle() -> ImageHandle {
    ImageHandle::from_path(BASE_IMAGE_PATH)
}

pub fn mask_handle() -> ImageHandle {
    ImageHandle::from_path(RAW_MASK_PATH)
}

pub fn block(rect: Rect, language: Option<Language>) -> DetectedBlock {
    DetectedBlock {
        rect,
        language,
        // Dyadic rationals: exactly representable, so no test ever trips on float
        // formatting when a page is serialized.
        confidence: 0.875,
        mask_coverage: 0.5,
    }
}

pub fn text_box(rect: Rect, language: Option<Language>) -> TextBox {
    TextBox { rect, language }
}

/// A `PageDataRaw` at `scale == 1.0` over [`PAGE_SIZE`], with path-only handles (never
/// loaded, because these fixtures are used by the no-OCR suites).
pub fn raw_page(blocks: Vec<DetectedBlock>) -> PageDataRaw {
    PageDataRaw {
        schema_version: pc_core::SCHEMA_VERSION,
        original_path: PathBuf::from(ORIGINAL_PATH),
        base_image: base_handle(),
        raw_mask: mask_handle(),
        scale: 1.0,
        image_size: PAGE_SIZE,
        blocks,
    }
}

/// A `PageDataRaw` whose `base_image` carries a real decoded image in its cache, so the
/// OCR pass can crop it without any file existing on disk (`ImageHandle::with_both`
/// serves `load()` from the cache — §2.3), while `path` stays `Some` so the page is
/// still serializable.
pub fn raw_page_with_image(
    image: DynamicImage,
    scale: f64,
    blocks: Vec<DetectedBlock>,
) -> PageDataRaw {
    let image_size = (image.width(), image.height());
    PageDataRaw {
        schema_version: pc_core::SCHEMA_VERSION,
        original_path: PathBuf::from(ORIGINAL_PATH),
        base_image: ImageHandle::with_both(BASE_IMAGE_PATH, image),
        raw_mask: mask_handle(),
        scale,
        image_size,
        blocks,
    }
}

/// A deterministic grayscale page. Content is irrelevant to the mock engine; it only
/// has to be stable.
pub fn synthetic_base(width: u32, height: u32) -> DynamicImage {
    DynamicImage::ImageLuma8(image::GrayImage::from_fn(width, height, |x, y| {
        image::Luma([((x * 3 + y * 5) % 256) as u8])
    }))
}

pub fn input(page: PageDataRaw, config: PreprocessorConfig) -> pc_preprocess::PreprocessInput {
    pc_preprocess::PreprocessInput {
        schema_version: pc_core::SCHEMA_VERSION,
        page,
        config,
        performing_ocr: false,
    }
}

/// A typed `None` for the stage's `Ctx` — `run(input, None)` alone cannot infer the
/// trait object type.
pub fn no_ocr() -> Option<&'static dyn OcrEngineFactory> {
    None
}

/// A config with every padding tier at zero, so a test that is about *box selection*
/// can assert on the detector rects unchanged.
pub fn unpadded_config() -> PreprocessorConfig {
    PreprocessorConfig {
        box_padding_initial: 0,
        box_right_padding_initial: 0,
        box_padding_extended: 0,
        box_right_padding_extended: 0,
        box_reference_padding: 0,
        ..PreprocessorConfig::default()
    }
}

/// spec §16.8 item 13: `PageData` has no `PartialEq` (it holds `ImageHandle`s), so the
/// determinism and equivalence gates compare canonical JSON. Legal here precisely
/// because the handles are path-bearing.
pub fn page_json(page: &pc_core::PageData) -> String {
    serde_json::to_string(page).expect("PageData with materialized handles is serializable")
}

pub fn rects(boxes: &[TextBox]) -> Vec<Rect> {
    boxes.iter().map(|text_box| text_box.rect).collect()
}

pub fn languages(boxes: &[TextBox]) -> Vec<Option<Language>> {
    boxes.iter().map(|text_box| text_box.language).collect()
}
