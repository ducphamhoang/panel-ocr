//! Task N3 -- region selection, scale-up, crop, alpha-mask extraction, grow, fade,
//! alpha attach and composition (spec §11.3 steps 3-5, §11.7(A)6/8/9/10, §16.10 items
//! 14-16).
//!
//! The small pure helpers are implemented (their arithmetic is fully pinned by §11.3
//! and §16.10); `build_noise_mask` -- the per-region loop plus composition -- is a
//! `todo!()` skeleton for task N3, with its signature frozen by the tests.
#![allow(unused_variables)]

use crate::{composite, gaussian, morph, nlm::NlmParams};
use image::{GenericImageView, GrayImage, Luma, RgbImage, Rgba, RgbaImage};
use pc_config::DenoiserConfig;
use pc_core::{MaskRegionStats, Rect};
use pc_imageops::BinaryMask;

/// spec §11.3 step 3: `!failed && std_deviation > noise_min_standard_deviation`,
/// **strictly** greater, order preserved (§16.10 item 16).
///
/// Intuition (§11.3): a *perfect* fit (sigma ~ 0) means the surrounding area was
/// uniform, so there is no JPEG noise ring to hide; a *failed* fit means nothing was
/// painted, so there is nothing to blend.
pub fn select_regions(
    regions: &[MaskRegionStats],
    noise_min_standard_deviation: f64,
) -> Vec<&MaskRegionStats> {
    regions
        .iter()
        .filter(|region| !region.failed && region.std_deviation > noise_min_standard_deviation)
        .collect()
}

/// spec §11.3 step 4.3 -- binary mask extraction from the RGBA cutout, from the
/// **alpha** channel (`alpha > 0`).
///
/// DEVIATION(5): upstream converts the RGBA cutout with `.convert("L")`, which computes
/// the luma of the RGB channels and *discards alpha* -- so for a black-filled mask
/// (`black_bubble`, fill `(0,0,0,255)`) the luma is 0 and the noise mask silently
/// vanishes. Verified upstream bug, not design intent (§14.5, decided in §15.6): a
/// mask's "is this pixel covered" signal must not depend on the brightness of its fill
/// colour. v1 uses alpha; §11.7(A)9 is the regression test.
pub fn alpha_binary(cutout: &RgbaImage) -> BinaryMask {
    BinaryMask::from_fn(cutout.width(), cutout.height(), |x, y| {
        cutout.get_pixel(x, y).0[3] > 0
    })
}

/// spec §11.3 step 4.4-4.5 -- "grow the mask and fade its edges": dilate the binary
/// cutout with `kernel(noise_outline_size)` (`noise_outline_size = 5` -> an 11x11
/// ellipse; `0` -> identity, §16.9 item 5), lift to 8-bit `L` (`1 -> 255`), then blur
/// with the true separable Gaussian of `sigma = noise_fade_radius` (§14.6).
pub fn fade_mask(
    binary: &BinaryMask,
    noise_outline_size: u32,
    noise_fade_radius: u32,
) -> GrayImage {
    let grown = morph::dilate(binary, &morph::kernel(noise_outline_size));
    gaussian::blur(&grown.to_gray(), noise_fade_radius)
}

/// spec §11.3 step 4.7 -- attach `alpha` as the alpha channel of `rgb`. Panics on a
/// dimension mismatch (a programming error: both come from the same crop).
pub fn attach_alpha(rgb: &RgbImage, alpha: &GrayImage) -> RgbaImage {
    assert_eq!(
        rgb.dimensions(),
        alpha.dimensions(),
        "alpha attach needs matching sizes: {:?} vs {:?}",
        rgb.dimensions(),
        alpha.dimensions()
    );
    RgbaImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        let image::Rgb([r, g, b]) = *rgb.get_pixel(x, y);
        Rgba([r, g, b, alpha.get_pixel(x, y).0[0]])
    })
}

/// `NlmParams` from the profile (§11.3; §16.10 item 8: `colored_images` does not
/// change the v1 code path, the cutout is always the 3-channel canvas crop).
pub fn nlm_params(config: &DenoiserConfig) -> NlmParams {
    NlmParams {
        h: config.filter_strength as f32,
        template_window: config.template_window_size,
        search_window: config.search_window_size,
    }
}

/// spec §11.3 step 4.1 -- the region's rect in `cleaned` coordinates, and the crop
/// window it maps to. `None` when the scaled rect is degenerate or fully out of the
/// canvas: §16.10 item 14 makes that a skip (with a `WARN` at the call site), never a
/// `StageError`.
pub fn region_crop(
    rect: Rect,
    scale_up: f64,
    canvas: (u32, u32),
) -> Option<(Rect, (u32, u32, u32, u32))> {
    let scaled = rect.scale(scale_up);
    scaled.to_crop(canvas).map(|crop| (scaled, crop))
}

/// spec §11.3 steps 4-5 -- build the noise mask.
///
/// Contract (frozen with the tests; §16.10 items 14-16):
///   * `cleaned` is the full-resolution stage-3 composite (RGB) and `mask` is the
///     combined mask already resized to `cleaned`'s dimensions (RGBA).
///   * `selected` are the regions §11.3 step 3 kept, **in `mask_data.regions` order**;
///     each `rect` is still in `combined_mask` coordinates and must be scaled by
///     `scale_up` (§11.3 step 4.1).
///   * For each region, in order: crop `cleaned` and `mask`; take the binary mask from
///     the cutout's **alpha** (`alpha_binary`); `fade_mask` it; denoise the image cutout
///     with `nlm::denoise`; `attach_alpha` the faded mask to the denoised cutout; and
///     `composite::alpha_composite_over` the resulting RGBA layer onto a transparent
///     canvas of `cleaned`'s size at `(scaled.x1, scaled.y1)` -- clamped to the crop
///     origin, since `to_crop` may have clipped a partially out-of-canvas rect.
///   * A region whose `region_crop` is `None` is skipped with `tracing::warn!` and does
///     **not** count (§16.10 item 14).
///   * Returns the RGBA noise mask (fully transparent when nothing was denoised) and
///     `boxes_denoised` = the number of layers actually produced (§16.10 item 15).
pub fn build_noise_mask(
    cleaned: &RgbImage,
    mask: &RgbaImage,
    selected: &[&MaskRegionStats],
    scale_up: f64,
    config: &DenoiserConfig,
) -> (RgbaImage, usize) {
    let mut noise_mask = blank_noise_mask(cleaned.dimensions());
    let mut boxes_denoised = 0;
    // DEVIATION(§16.14 item 3): the working window is the region rect **padded** by
    // `noise_outline_size + 3 * noise_fade_radius`, not §11.3 step 4.2's literal
    // `cleaned.crop(rect)`. Dilation reaches `noise_outline_size` px beyond the fill and
    // the Gaussian has non-zero support through `3 * noise_fade_radius`; cropping at the
    // bare rect clips the alpha fade into a hard step at an arbitrary bounding-box edge —
    // a visible seam in exactly the case the fade exists to prevent — and leaves NLM with
    // only the already-filled (usually uniform) region as context. The padded reach is
    // exactly the bound §11.7(A)8 already permits, so the containment guarantee is
    // unchanged. Ratified in §16.14 item 3.
    let reach = config
        .noise_outline_size
        .saturating_add(config.noise_fade_radius.saturating_mul(3))
        .min(i32::MAX as u32) as i32;

    for region in selected {
        // Validate the scaled rect itself before padding it: a degenerate original
        // rect remains a skipped region, rather than becoming a valid padded window.
        let Some((scaled, _)) = region_crop(region.rect, scale_up, cleaned.dimensions()) else {
            tracing::warn!(rect = ?region.rect, scale_up, "skipping denoise region with no usable crop");
            continue;
        };
        let crop = scaled
            .pad(reach, cleaned.dimensions())
            .to_crop(cleaned.dimensions())
            .expect("padding a usable in-canvas rect produces a usable crop");

        let image_cutout = crop_rgb(cleaned, crop);
        let mask_cutout = crop_rgba(mask, crop);
        let alpha = fade_mask(
            &alpha_binary(&mask_cutout),
            config.noise_outline_size,
            config.noise_fade_radius,
        );
        let denoised = crate::nlm::denoise(
            &image::DynamicImage::ImageRgb8(image_cutout),
            nlm_params(config),
        )
        .to_rgb8();
        let layer = attach_alpha(&denoised, &alpha);

        // `crop` is the clamped placement origin for partially out-of-canvas regions.
        composite::alpha_composite_over(&mut noise_mask, &layer, (crop.0 as i32, crop.1 as i32));
        boxes_denoised += 1;
    }

    (noise_mask, boxes_denoised)
}

/// Crop helper shared by the per-region loop and the tests: the `(x, y, w, h)` window
/// `Rect::to_crop` returns, applied to an RGB image.
pub fn crop_rgb(image: &RgbImage, crop: (u32, u32, u32, u32)) -> RgbImage {
    image.view(crop.0, crop.1, crop.2, crop.3).to_image()
}

/// `crop_rgb`'s RGBA twin.
pub fn crop_rgba(image: &RgbaImage, crop: (u32, u32, u32, u32)) -> RgbaImage {
    image.view(crop.0, crop.1, crop.2, crop.3).to_image()
}

/// A fully transparent RGBA canvas -- §11.3 step 5's "no regions" noise mask and
/// §11.3 step 1's 1-bit-shortcut noise mask (§11.7(A)10, §11.7(A)7).
pub fn blank_noise_mask(size: (u32, u32)) -> RgbaImage {
    RgbaImage::from_pixel(size.0, size.1, Rgba([0, 0, 0, 0]))
}

/// A grayscale image's histogram, 256 buckets -- §11.7(B)13 asserts on the noise
/// mask's alpha histogram, and this keeps that assertion out of the test file's way.
pub fn alpha_histogram(mask: &RgbaImage) -> [u64; 256] {
    let mut histogram = [0_u64; 256];
    for pixel in mask.pixels() {
        histogram[pixel.0[3] as usize] += 1;
    }
    histogram
}

/// The alpha channel as a standalone `L` image, for metric helpers that take
/// `GrayImage` (`pc_testkit::metrics::ssim_gray`).
pub fn alpha_channel(mask: &RgbaImage) -> GrayImage {
    GrayImage::from_fn(mask.width(), mask.height(), |x, y| {
        Luma([mask.get_pixel(x, y).0[3]])
    })
}

/// Re-exported for the frozen tests, which assert that Stage 4's composition is the
/// same arithmetic Stage 3 used (§16.10 item 3).
pub use composite::{alpha_composite_over, composite_rgb, resize_nearest_rgba};
