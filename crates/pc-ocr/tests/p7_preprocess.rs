//! Task P7b — manga-ocr's preprocessing, ported exactly (spec §9.5 row P7).
//! FROZEN per CLAUDE.md.
//!
//! Upstream chain: `img.convert("L").convert("RGB")`, then `ViTImageProcessor` with
//! `size {224, 224}`, `resample: 2` (PIL BILINEAR), `rescale_factor: 1/255`,
//! `image_mean/std = 0.5`.
//!
//! Every expected value below came out of Pillow 12.3.0, executed, independently
//! re-verified during dispatch of this task (not merely carried from planning notes).
//!
//! WHY A HAND-PORTED RESAMPLER RATHER THAN THE `image` crate's `Triangle`: measured,
//! `Triangle` reproduces 11 of manga-ocr's 12 published labels where Pillow reproduces
//! 12. This is NOT the same function as `pc_detect::onnx::resize_bilinear_rgb` — do not
//! reuse or "consolidate" with that one; it would silently change the detector's pinned
//! §8.3 preprocessing.

use image::{DynamicImage, GrayImage, Rgb, RgbImage};
use pc_ocr::preprocess::{pil_luma, pillow_resize_bilinear, pixel_values, INPUT_SIZE};
use sha2::{Digest, Sha256};

fn gray(width: u32, height: u32, bytes: &[u8]) -> GrayImage {
    GrayImage::from_raw(width, height, bytes.to_vec()).expect("dimensions match the buffer")
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn pil_luma_uses_pillows_l24_fixed_point_rule() {
    assert_eq!(pil_luma(0, 242, 39), 147);
    assert_eq!(pil_luma(21, 121, 221), 103);
    assert_eq!(pil_luma(49, 253, 117), 177);

    for (r, g, b) in [
        (0u8, 0u8, 0u8),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ] {
        let expected =
            ((u32::from(r) * 19595 + u32::from(g) * 38470 + u32::from(b) * 7471 + 0x8000) >> 16)
                as u8;
        assert_eq!(pil_luma(r, g, b), expected);
    }
}

#[test]
fn pil_luma_is_exact_on_achromatic_pixels() {
    for value in 0..=255_u8 {
        assert_eq!(pil_luma(value, value, value), value);
    }
}

#[test]
fn pillow_bilinear_downscale_matches_pillow_on_the_four_by_four_ramp() {
    let ramp = gray(
        4,
        4,
        &[
            0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240,
        ],
    );

    assert_eq!(
        pillow_resize_bilinear(&ramp, 2, 2).into_raw(),
        vec![57, 83, 157, 183]
    );
    assert_eq!(
        pillow_resize_bilinear(&ramp, 3, 3).into_raw(),
        vec![24, 43, 62, 101, 120, 139, 178, 197, 216]
    );
}

#[test]
fn pillow_bilinear_upscale_matches_pillow_on_the_two_by_two_checker() {
    let checker = gray(2, 2, &[0, 255, 255, 0]);

    assert_eq!(
        pillow_resize_bilinear(&checker, 4, 4).into_raw(),
        vec![0, 64, 191, 255, 64, 96, 159, 191, 191, 159, 96, 64, 255, 191, 64, 0]
    );
    assert_eq!(
        pillow_resize_bilinear(&checker, 5, 5).into_raw(),
        vec![
            0, 25, 128, 230, 255, 25, 45, 128, 210, 230, 128, 128, 128, 128, 128, 230, 210, 128,
            45, 25, 255, 230, 128, 25, 0,
        ]
    );
}

#[test]
fn pillow_bilinear_matches_pillow_when_the_two_axes_scale_differently() {
    let ramp = gray(
        4,
        4,
        &[
            0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240,
        ],
    );

    assert_eq!(
        pillow_resize_bilinear(&ramp, 5, 3).into_raw(),
        vec![19, 30, 43, 56, 67, 96, 107, 120, 133, 144, 173, 184, 197, 210, 221]
    );
}

#[test]
fn pillow_bilinear_resize_of_the_handwritten_bubble_reproduces_pillows_bytes() {
    let source = image::open(pc_testkit::paths::upstream(
        "demo_bubbles/handwritten_bubble_raw.png",
    ))
    .expect("committed fixture")
    .to_luma8();
    assert_eq!(source.dimensions(), (72, 132));

    let resized = pillow_resize_bilinear(&source, INPUT_SIZE, INPUT_SIZE);

    assert_eq!(resized.dimensions(), (224, 224));
    assert_eq!(
        hex(resized.as_raw()),
        "3faf9f21ae6f55609212d3e06895bdae38a16e226f97d272b68ab3e4f0a8a019"
    );
}

#[test]
fn pillow_bilinear_resize_of_the_square_bubble_reproduces_pillows_bytes() {
    let source = image::open(pc_testkit::paths::upstream(
        "demo_bubbles/square_bubble_raw.png",
    ))
    .expect("committed fixture")
    .to_luma8();
    assert_eq!(source.dimensions(), (144, 270));

    let resized = pillow_resize_bilinear(&source, INPUT_SIZE, INPUT_SIZE);

    assert_eq!(
        hex(resized.as_raw()),
        "644c04e7cbfbae3327580125e22ea6e92153ddaa7e25c6e6ceb4ad4e7919c0b8"
    );
}

#[test]
fn pixel_values_are_nchw_with_three_identical_planes() {
    let mut colour = RgbImage::new(8, 5);
    for (x, y, pixel) in colour.enumerate_pixels_mut() {
        *pixel = Rgb([(x * 31) as u8, (y * 51) as u8, 200]);
    }
    let values = pixel_values(&DynamicImage::ImageRgb8(colour));

    let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
    assert_eq!(values.len(), 3 * plane);
    assert_eq!(&values[..plane], &values[plane..2 * plane]);
    assert_eq!(&values[..plane], &values[2 * plane..]);
}

#[test]
fn pixel_values_normalize_black_to_minus_one_and_white_to_plus_one() {
    let plane = (INPUT_SIZE * INPUT_SIZE) as usize;

    let black = pixel_values(&DynamicImage::ImageLuma8(gray(2, 2, &[0, 0, 0, 0])));
    let white = pixel_values(&DynamicImage::ImageLuma8(gray(2, 2, &[255, 255, 255, 255])));

    assert!(
        black.iter().all(|value| *value == -1.0),
        "black must map to -1.0"
    );
    assert!(
        white.iter().all(|value| *value == 1.0),
        "white must map to +1.0"
    );
    assert_eq!(black.len(), 3 * plane);
}
