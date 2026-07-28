//! Shared fixtures for the frozen `pc-denoise` test suites. FROZEN with the tests.
#![allow(dead_code)]

use image::{DynamicImage, GenericImageView, GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use pc_config::DenoiserConfig;
use pc_core::{ImageHandle, MaskData, MaskRegionStats, Rect, SCHEMA_VERSION};
use pc_denoise::{DenoiseDests, DenoiseInput};
use std::path::PathBuf;

/// Big enough that a default grow (5 px) plus fade (3 * 1 px) never reaches the canvas
/// edge from the regions used below, so a clip that shows up in an assertion is the
/// clip the test is about.
pub const PAGE_SIZE: (u32, u32) = (120, 90);

pub const ORIGINAL_PATH: &str = "/synthetic/page.png";

/// The one region every wiring test uses: comfortably interior to `PAGE_SIZE`.
pub const REGION: Rect = Rect {
    x1: 40,
    y1: 30,
    x2: 70,
    y2: 60,
};

/// `MaskRegionStats` with only the two fields §11.3 step 3 filters on varying.
pub fn region(rect: Rect, std_deviation: f64, failed: bool) -> MaskRegionStats {
    MaskRegionStats {
        rect,
        std_deviation,
        failed,
        thickness: Some(4),
    }
}

/// A grayscale page: `background` everywhere, `text` value inside each rect.
pub fn gray_page(size: (u32, u32), background: u8, text: &[(Rect, u8)]) -> DynamicImage {
    let mut image = GrayImage::from_pixel(size.0, size.1, Luma([background]));
    for (rect, value) in text {
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

/// An RGB page with a solid background and one differently-coloured rect.
pub fn rgb_page(size: (u32, u32), background: [u8; 3], text: &[(Rect, [u8; 3])]) -> DynamicImage {
    let mut image = RgbImage::from_pixel(size.0, size.1, Rgb(background));
    for (rect, value) in text {
        if let Some((x, y, w, h)) = rect.to_crop(size) {
            for py in y..y + h {
                for px in x..x + w {
                    image.put_pixel(px, py, Rgb(*value));
                }
            }
        }
    }
    DynamicImage::ImageRgb8(image)
}

/// A combined mask: transparent everywhere, `fill` (opaque) inside each rect. This is
/// exactly the shape `pc_mask::build_combined_mask` produces.
pub fn combined_mask(size: (u32, u32), fills: &[(Rect, [u8; 4])]) -> RgbaImage {
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

/// A `MaskData` whose handles carry decoded images in cache (so `load()` needs no file)
/// while keeping a `path`, so the value stays serializable (§2.3, §16.8 item 13).
pub fn mask_data(
    base: DynamicImage,
    mask: RgbaImage,
    scale: f64,
    regions: Vec<MaskRegionStats>,
) -> MaskData {
    MaskData {
        schema_version: SCHEMA_VERSION,
        original_path: PathBuf::from(ORIGINAL_PATH),
        base_image: ImageHandle::with_both("/synthetic/page_base.png", base),
        combined_mask: ImageHandle::with_both(
            "/synthetic/page_combined_mask.png",
            DynamicImage::ImageRgba8(mask),
        ),
        scale,
        regions,
    }
}

/// A memory-mode `DenoiseInput` (no destinations): every handle carries its decoded
/// image, nothing touches the filesystem.
pub fn memory_input(
    original: DynamicImage,
    mask: RgbaImage,
    regions: Vec<MaskRegionStats>,
    config: DenoiserConfig,
) -> DenoiseInput {
    let masked = DynamicImage::ImageRgb8(pc_denoise::composite_rgb(
        &original.to_rgb8(),
        &pc_denoise::resize_nearest_rgba(&mask, original.dimensions()),
    ));
    DenoiseInput {
        schema_version: SCHEMA_VERSION,
        mask_data: mask_data(original.clone(), mask, 1.0, regions),
        original_image: ImageHandle::from_memory(original),
        masked_image: ImageHandle::from_memory(masked),
        config,
        dests: DenoiseDests::default(),
    }
}

/// The default profile's denoiser section (`h = 10`, `t = 7`, `s = 21`,
/// `noise_min_standard_deviation = 0.25`, `noise_outline_size = 5`,
/// `noise_fade_radius = 1`).
pub fn default_config() -> DenoiserConfig {
    DenoiserConfig::default()
}

/// A config with the NLM windows shrunk so wiring tests stay fast; the *numeric* NLM
/// gates in `n1_nlm.rs` always use the real defaults.
pub fn fast_config() -> DenoiserConfig {
    DenoiserConfig {
        template_window_size: 3,
        search_window_size: 5,
        ..DenoiserConfig::default()
    }
}

/// Population standard deviation of a grayscale image, in `f64`.
pub fn std_deviation(image: &GrayImage) -> f64 {
    let n = (image.width() as f64) * (image.height() as f64);
    let mean = mean(image);
    let variance = image
        .pixels()
        .map(|pixel| {
            let delta = f64::from(pixel.0[0]) - mean;
            delta * delta
        })
        .sum::<f64>()
        / n;
    variance.sqrt()
}

pub fn mean(image: &GrayImage) -> f64 {
    let n = (image.width() as f64) * (image.height() as f64);
    image
        .pixels()
        .map(|pixel| f64::from(pixel.0[0]))
        .sum::<f64>()
        / n
}

/// §16.10 item 11's counter is process-global, and cargo runs a test binary's tests on
/// several threads. Discipline, binding on every test in this crate: a test that merely
/// *calls* NLM takes [`shared_nlm`]; a test that *observes* the counter takes
/// [`exclusive_nlm`]. That makes the counter assertions deterministic without making
/// the counter itself `cfg`-gated (which §16.10 item 11 rules out, since these are
/// integration tests).
static NLM_COUNTER: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Take while calling `nlm::denoise` (directly or through `run`) without observing the
/// counter.
pub fn shared_nlm() -> std::sync::RwLockReadGuard<'static, ()> {
    NLM_COUNTER
        .read()
        .unwrap_or_else(|error| error.into_inner())
}

/// Take while asserting on `nlm::denoise_call_count()`.
pub fn exclusive_nlm() -> std::sync::RwLockWriteGuard<'static, ()> {
    NLM_COUNTER
        .write()
        .unwrap_or_else(|error| error.into_inner())
}

/// Run `body` inside a rayon pool of exactly `threads` threads — the mechanism behind
/// §11.7(A)5's thread-invariance gate.
pub fn with_threads<T: Send>(threads: usize, body: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a rayon pool of a fixed size")
        .install(body)
}
