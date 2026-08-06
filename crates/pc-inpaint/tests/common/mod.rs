//! Shared fixture builders for the L4 test suite. No assertions live here — a helper that
//! asserts hides which test failed.

#![allow(dead_code)]

use image::{GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use pc_config::InpainterConfig;
use pc_core::{MaskRegionStats, Rect};
use pc_imageops::BinaryMask;
use std::collections::BTreeSet;

pub fn region(
    rect: Rect,
    std_deviation: f64,
    failed: bool,
    thickness: Option<u32>,
) -> MaskRegionStats {
    MaskRegionStats {
        rect,
        std_deviation,
        failed,
        thickness,
    }
}

/// An `InpainterConfig` with the radii a test cares about and everything else at its ratified
/// default (§16.38 item 13(a)).
pub fn radii(
    min_inpainting_radius: u32,
    max_inpainting_radius: u32,
    inpainting_radius_multiplier: f64,
    inpainting_isolation_radius: u32,
    inpainting_fade_radius: u32,
) -> InpainterConfig {
    InpainterConfig {
        min_inpainting_radius,
        max_inpainting_radius,
        inpainting_radius_multiplier,
        inpainting_isolation_radius,
        inpainting_fade_radius,
        ..InpainterConfig::default()
    }
}

/// A `GrayImage` that is `255` at `set` and `0` elsewhere — a `_raw_mask.png` stand-in.
pub fn gray_with(size: (u32, u32), set: &[(u32, u32)]) -> GrayImage {
    let wanted: BTreeSet<(u32, u32)> = set.iter().copied().collect();
    GrayImage::from_fn(size.0, size.1, |x, y| {
        Luma([if wanted.contains(&(x, y)) { 255 } else { 0 }])
    })
}

/// A `GrayImage` that is `255` inside `rect` and `0` elsewhere.
pub fn gray_rect(size: (u32, u32), rect: Rect) -> GrayImage {
    GrayImage::from_fn(size.0, size.1, |x, y| {
        let inside = (x as i32) >= rect.x1
            && (x as i32) < rect.x2
            && (y as i32) >= rect.y1
            && (y as i32) < rect.y2;
        Luma([if inside { 255 } else { 0 }])
    })
}

/// A fully transparent RGBA canvas — a combined mask that covers nothing.
pub fn transparent(size: (u32, u32)) -> RgbaImage {
    RgbaImage::from_pixel(size.0, size.1, Rgba([0, 0, 0, 0]))
}

/// A transparent RGBA canvas with `color` at full alpha inside `rect`.
pub fn rgba_rect(size: (u32, u32), rect: Rect, color: [u8; 3]) -> RgbaImage {
    RgbaImage::from_fn(size.0, size.1, |x, y| {
        let inside = (x as i32) >= rect.x1
            && (x as i32) < rect.x2
            && (y as i32) >= rect.y1
            && (y as i32) < rect.y2;
        if inside {
            Rgba([color[0], color[1], color[2], 255])
        } else {
            Rgba([0, 0, 0, 0])
        }
    })
}

pub fn flat_rgb(size: (u32, u32), color: [u8; 3]) -> RgbImage {
    RgbImage::from_pixel(size.0, size.1, Rgb(color))
}

/// Every set pixel of a mask, as a set — so an assertion can compare *identity*, not a count.
/// §16.38's own discipline: "cardinality is not identity".
pub fn set_pixels(mask: &BinaryMask) -> BTreeSet<(u32, u32)> {
    let mut out = BTreeSet::new();
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            if mask.get(x, y) {
                out.insert((x, y));
            }
        }
    }
    out
}

pub fn pixel_set(pixels: &[(u32, u32)]) -> BTreeSet<(u32, u32)> {
    pixels.iter().copied().collect()
}
