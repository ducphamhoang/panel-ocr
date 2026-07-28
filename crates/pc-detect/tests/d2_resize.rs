//! Task D2 -- spec §8.3 step 2, §8.7(A)1-2, §14.1. FROZEN per CLAUDE.md.

use image::{Rgb, RgbImage};
use pc_detect::resize::{calculate_new_size_and_scale, resize_area, round_half_away};
use pc_testkit::assert_close;

/// A `w x h` RGB image whose three channels all carry `values` (given row-major), so a
/// hand-computed grayscale expectation applies to every channel.
fn rgb_from_gray_rows(rows: &[&[u8]]) -> RgbImage {
    let width = rows[0].len() as u32;
    let height = rows.len() as u32;
    RgbImage::from_fn(width, height, |x, y| {
        let value = rows[y as usize][x as usize];
        Rgb([value, value, value])
    })
}

// ------------------------------------------------------------ §8.7(A)1 table

#[test]
fn a1_calculate_new_size_and_scale_matches_the_spec_table() {
    // spec §8.7(A)1, transcribed literally; w = 1000 throughout.
    let cases: [(u32, u32, u32, (u32, u32, f64)); 7] = [
        // h,    lower, upper, (new_w, new_h, scale)
        (3000, 1000, 4000, (1000, 3000, 1.0)), // no-op (h <= upper)
        (8000, 1000, 4000, (500, 4000, 0.5)),  // integer inverse scale n=2
        (5000, 1000, 4000, (500, 2500, 0.5)),  // integer inverse scale n=2
        (4300, 2000, 2100, (488, 2100, 2100.0 / 4300.0)), // multiple-of-4 branch
        (4500, 4000, 4000, (889, 4000, 4000.0 / 4500.0)), // exact-size branch
        (8000, 0, 4000, (1000, 8000, 1.0)),    // disabled
        (8000, 1000, 0, (1000, 8000, 1.0)),    // disabled
    ];

    for (height, lower, upper, (expected_w, expected_h, expected_scale)) in cases {
        let (width, new_height, scale) = calculate_new_size_and_scale(1000, height, lower, upper);
        assert_eq!(
            (width, new_height),
            (expected_w, expected_h),
            "h={height} lower={lower} upper={upper}"
        );
        assert_close(scale, expected_scale, 1e-12);
    }
}

#[test]
fn no_resize_leaves_scale_exactly_one() {
    // §2.4: `scale == 1.0` exactly (not approximately) when nothing was resized.
    let (_, _, scale) = calculate_new_size_and_scale(1000, 3000, 1000, 4000);
    assert_eq!(scale, 1.0);
}

#[test]
fn deviation_1_lower_greater_than_upper_uses_upper_for_both() {
    // DEVIATION(1) / spec §14.1: upstream would set new_height = lower (4000) while
    // computing scale from upper; we use upper for both.
    let (width, height, scale) = calculate_new_size_and_scale(1000, 4500, 4000, 2000);

    assert_eq!(
        height, 2000,
        "new_height must come from `upper`, not `lower`"
    );
    assert_close(scale, 2000.0 / 4500.0, 1e-12);
    assert_eq!(width, 444, "round(1000 * 2000/4500) = round(444.44)");
}

// ------------------------------------------------------------ rounding boundary

#[test]
fn round_half_away_from_zero_at_the_exact_boundary() {
    // spec §8.3 step 2 pins away-from-zero (Python's round() is banker's rounding),
    // and requires the .5 boundary to be tested explicitly.
    assert_eq!(round_half_away(0.5), 1);
    assert_eq!(round_half_away(-0.5), -1);
    assert_eq!(round_half_away(1.5), 2);
    assert_eq!(round_half_away(2.5), 3); // banker's rounding would give 2
    assert_eq!(round_half_away(-2.5), -3); // banker's rounding would give -2
    assert_eq!(round_half_away(3.5), 4);
    assert_eq!(round_half_away(0.0), 0);
    assert_eq!(round_half_away(0.4999999999), 0);
    assert_eq!(round_half_away(-0.4999999999), 0);
    assert_eq!(round_half_away(2.4), 2);
    assert_eq!(round_half_away(2.6), 3);
}

// ------------------------------------------------------------ §8.7(A)2 INTER_AREA

#[test]
fn a2_inter_area_4x4_to_2x2_is_the_exact_block_mean() {
    // spec §8.7(A)2, hand-computed: each output pixel is the mean of its 2x2 block.
    let image = rgb_from_gray_rows(&[
        &[0, 10, 20, 30],
        &[40, 50, 60, 70],
        &[80, 90, 100, 110],
        &[120, 130, 140, 150],
    ]);

    let resized = resize_area(&image, 2, 2);

    assert_eq!(resized.dimensions(), (2, 2));
    assert_eq!(resized.get_pixel(0, 0).0, [25, 25, 25]); // (0+10+40+50)/4
    assert_eq!(resized.get_pixel(1, 0).0, [45, 45, 45]); // (20+30+60+70)/4
    assert_eq!(resized.get_pixel(0, 1).0, [105, 105, 105]); // (80+90+120+130)/4
    assert_eq!(resized.get_pixel(1, 1).0, [125, 125, 125]); // (100+110+140+150)/4
}

#[test]
fn inter_area_uses_fractional_edge_weights() {
    // 3 -> 2 columns: sx = 1.5, so output 0 = (1.0*p0 + 0.5*p1)/1.5 and
    // output 1 = (0.5*p1 + 1.0*p2)/1.5. A whole-pixel (nearest/skip) implementation
    // would give 0 and 60 instead.
    let image = rgb_from_gray_rows(&[&[0, 60, 120]]);

    let resized = resize_area(&image, 2, 1);

    assert_eq!(resized.get_pixel(0, 0).0, [20, 20, 20]);
    assert_eq!(resized.get_pixel(1, 0).0, [100, 100, 100]);
}

#[test]
fn inter_area_to_the_same_size_is_the_identity() {
    let image = rgb_from_gray_rows(&[&[1, 2, 3], &[4, 5, 6]]);

    let resized = resize_area(&image, 3, 2);

    assert_eq!(resized.as_raw(), image.as_raw());
}

#[test]
fn inter_area_of_a_uniform_image_is_uniform() {
    let image = RgbImage::from_pixel(37, 91, Rgb([7, 128, 255]));

    let resized = resize_area(&image, 5, 11);

    assert!(resized.pixels().all(|pixel| pixel.0 == [7, 128, 255]));
}

// ------------------------------------------------------------ fixture-driven

#[test]
fn demo_bubbles_take_the_no_resize_path() {
    // spec §8.6: every demo bubble is <= 4000 tall, so scale == 1.0 and the dimensions
    // are unchanged.
    for bubble in pc_testkit::paths::DEMO_BUBBLES {
        let (width, height, scale) =
            calculate_new_size_and_scale(bubble.width, bubble.height, 1000, 4000);
        assert_eq!((width, height), bubble.size(), "{}", bubble.name);
        assert_eq!(scale, 1.0, "{}", bubble.name);
    }
}

#[test]
fn demo_bubble_pixels_are_untouched_on_the_no_resize_path() {
    let bubble = pc_testkit::paths::demo_bubble("handwritten");
    let image = pc_testkit::images::load_rgb8(bubble.path(pc_testkit::paths::BubbleKind::Raw));

    let resized = resize_area(&image, bubble.width, bubble.height);

    assert_eq!(resized.as_raw(), image.as_raw());
}

#[test]
fn long_strip_halves_to_500x4000() {
    // spec §8.6: long_strip.jpg is 1000x8000; 8000 -> 4000 at scale 0.5.
    let (width, height) = pc_testkit::paths::LONG_STRIP_SIZE;
    let (new_width, new_height, scale) = calculate_new_size_and_scale(width, height, 1000, 4000);

    assert_eq!((new_width, new_height), (500, 4000));
    assert_eq!(scale, 0.5);
}

#[test]
fn long_strip_resize_produces_the_declared_dimensions() {
    let image = pc_testkit::images::load_rgb8(pc_testkit::paths::long_strip());
    let (new_width, new_height, _) =
        calculate_new_size_and_scale(image.width(), image.height(), 1000, 4000);

    let resized = resize_area(&image, new_width, new_height);

    assert_eq!(resized.dimensions(), (500, 4000));
}

#[test]
#[ignore = "pending task F1: needs the cv2.INTER_AREA reference recorded by `cargo xtask record-fixtures`"]
fn a2_pending_inter_area_matches_the_recorded_opencv_reference() {
    // spec §8.7(A)2, second half: mean absolute difference <= 1.0 and max per-channel
    // delta <= 2 against a recorded cv2.INTER_AREA downscale of long_strip.jpg.
    // Unignore once F1 lands `long_strip_inter_area_500x4000.png` under
    // tests/fixtures/recorded/.
    unimplemented!("blocked on F1");
}
