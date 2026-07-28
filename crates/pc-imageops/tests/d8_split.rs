//! Task D8 -- `pc-imageops::split`. FROZEN per CLAUDE.md: these tests are the contract
//! for spec §8.7(A)7, §8.7(B)8 and §16.6 item 8. Implementation may change; these may
//! not, except by joint-architect decision.

use approx::assert_relative_eq;
use image::{GrayImage, Luma, Rgb, RgbImage};
use pc_imageops::{
    calculate_best_splits, row_scores, search_ranges, split_image, stitch_images, SplitParams,
};
use std::sync::OnceLock;

// ---------------------------------------------------------------- helpers

/// The real 1000x8000 strip fixture (§7.1), decoded once for the whole test binary --
/// it is a progressive JPEG and decoding it four times is pure wall-clock waste.
fn long_strip() -> &'static RgbImage {
    static IMAGE: OnceLock<RgbImage> = OnceLock::new();
    IMAGE.get_or_init(|| pc_testkit::images::load_rgb8(pc_testkit::paths::long_strip()))
}

/// A synthetic strip whose only flat (zero-score) rows are `flat_rows`; every other row
/// alternates 0/255 horizontally, giving it a large squared-difference score.
fn strip_with_gutters(width: u32, height: u32, flat_rows: &[u32]) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        if flat_rows.contains(&y) {
            Rgb([128, 128, 128])
        } else if x % 2 == 0 {
            Rgb([0, 0, 0])
        } else {
            Rgb([255, 255, 255])
        }
    })
}

fn gradient_rgb(width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        Rgb([(x % 251) as u8, (y % 253) as u8, ((x * y) % 249) as u8])
    })
}

fn gradient_gray(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| Luma([((x * 7 + y * 13) % 256) as u8]))
}

fn strip_params(preferred: u32, tolerance: u32) -> SplitParams {
    SplitParams {
        preferred_height: preferred,
        tolerance_margin: tolerance,
        split_long_strips: true,
        max_aspect_ratio: 0.33,
    }
}

// ---------------------------------------------------------------- params

#[test]
fn defaults_match_spec_section_6_general_keys() {
    let params = SplitParams::defaults();
    assert_eq!(params.preferred_height, 2000);
    assert_eq!(params.tolerance_margin, 500);
    assert!(params.split_long_strips);
    assert_relative_eq!(params.max_aspect_ratio, 0.33);
}

// ---------------------------------------------------------------- row scores

#[test]
fn row_zero_is_never_a_candidate() {
    // spec §16.6 item 8: score[0] == +infinity, so row 0 can never win a window.
    let image = strip_with_gutters(8, 40, &[0, 5]);
    let scores = row_scores(&image);

    assert!(scores[0].is_infinite() && scores[0].is_sign_positive());
    assert_relative_eq!(scores[5], 0.0);

    // A window that would otherwise reach row 0 is clamped to start at 1.
    let ranges = search_ranges(image.dimensions(), &strip_params(10, 20));
    assert!(ranges.iter().all(|range| range.start >= 1));
}

#[test]
fn row_score_is_the_sum_of_squared_horizontal_differences() {
    // Row 1 is 0,255,0,255 -> three transitions of 255 each.
    let image = strip_with_gutters(4, 3, &[2]);
    let scores = row_scores(&image);

    assert_relative_eq!(scores[1], 3.0 * 255.0 * 255.0);
    assert_relative_eq!(scores[2], 0.0);
}

#[test]
fn a_flat_row_scores_zero_and_wins_its_window() {
    let image = strip_with_gutters(40, 400, &[95, 195, 295]);
    let splits = calculate_best_splits(&image, &strip_params(100, 20));

    assert_eq!(splits, vec![95, 195, 295]);
}

#[test]
fn ties_break_toward_the_smallest_row_index() {
    // Two equally flat rows inside the same (single) window: the earlier one must win.
    let image = strip_with_gutters(40, 200, &[95, 105]);
    let splits = calculate_best_splits(&image, &strip_params(100, 20));

    assert_eq!(splits, vec![95]);
}

// ---------------------------------------------------------------- search ranges / gates

#[test]
fn search_ranges_are_half_open_windows_around_multiples_of_preferred_height() {
    // spec §16.6 item 8: n = round(8000/2000) = 4 -> 3 splits.
    let ranges = search_ranges((1000, 8000), &SplitParams::defaults());

    assert_eq!(ranges, vec![1500..2500, 3500..4500, 5500..6500]);
}

#[test]
fn b8_long_strip_yields_exactly_three_splits() {
    let splits = calculate_best_splits(long_strip(), &SplitParams::defaults());

    assert_eq!(splits.len(), 3, "spec §8.7(B)8: 8000/2000 -> 3 splits");
}

#[test]
fn b8_each_split_lands_in_its_half_open_tolerance_window() {
    let splits = calculate_best_splits(long_strip(), &SplitParams::defaults());

    for (index, split) in splits.iter().enumerate() {
        let i = (index + 1) as u32;
        let lower = 2000 * i - 500;
        let upper = 2000 * i + 500;
        assert!(
            (lower..upper).contains(split),
            "split {split} is outside [{lower}, {upper})"
        );
    }
}

#[test]
fn b8_each_split_lands_in_a_locally_low_score_band() {
    // spec §8.7(B)8: the contract is "lands in the low-change band", stated as being at
    // or below the 10th percentile of the scores in its own search range.
    let image = long_strip();
    let scores = row_scores(image);
    let ranges = search_ranges(image.dimensions(), &SplitParams::defaults());
    let splits = calculate_best_splits(image, &SplitParams::defaults());

    assert_eq!(splits.len(), ranges.len());

    for (split, range) in splits.iter().zip(ranges) {
        let mut window: Vec<f64> = range.map(|row| scores[row as usize]).collect();
        window.sort_by(f64::total_cmp);
        let percentile_10 = window[window.len() / 10];
        assert!(
            scores[*split as usize] <= percentile_10,
            "split {split} scored {} > 10th percentile {percentile_10}",
            scores[*split as usize]
        );
    }
}

#[test]
fn b8_aspect_ratio_gate_rejects_when_ratio_exceeds_max() {
    // 1000/8000 = 0.125 > 0.1, so the strip no longer qualifies (spec §8.7(B)8).
    let params = SplitParams {
        max_aspect_ratio: 0.1,
        ..SplitParams::defaults()
    };

    assert!(search_ranges((1000, 8000), &params).is_empty());
    assert_eq!(
        calculate_best_splits(long_strip(), &params),
        Vec::<u32>::new()
    );
}

#[test]
fn square_and_wide_images_never_split() {
    let params = SplitParams::defaults();

    let square = gradient_rgb(64, 64);
    assert!(calculate_best_splits(&square, &params).is_empty());

    let wide = gradient_rgb(8000, 1000);
    assert!(search_ranges(wide.dimensions(), &params).is_empty());
}

#[test]
fn disabling_split_long_strips_disables_splitting() {
    let params = SplitParams {
        split_long_strips: false,
        ..strip_params(100, 20)
    };
    let image = strip_with_gutters(40, 400, &[95, 195, 295]);

    assert!(search_ranges(image.dimensions(), &params).is_empty());
    assert!(calculate_best_splits(&image, &params).is_empty());
}

#[test]
fn a_short_image_produces_no_splits() {
    // round(400/2000) = 0 -> clamped to 1 segment -> 0 splits.
    let image = strip_with_gutters(40, 400, &[95]);

    assert!(calculate_best_splits(&image, &SplitParams::defaults()).is_empty());
}

#[test]
fn calculate_best_splits_is_deterministic() {
    let image = strip_with_gutters(40, 400, &[95, 195, 295]);
    let params = strip_params(100, 20);
    let first = calculate_best_splits(&image, &params);

    for _ in 0..5 {
        assert_eq!(calculate_best_splits(&image, &params), first);
    }
}

// ---------------------------------------------------------------- split / stitch

#[test]
fn a7_split_then_stitch_long_strip_is_bit_exact() {
    // spec §8.7(A)7.
    let image = long_strip();
    let splits = calculate_best_splits(image, &SplitParams::defaults());
    let segments = split_image(image, &splits).expect("legal split rows");
    let stitched = stitch_images(&segments).expect("uniform-width segments");

    assert_eq!(stitched.dimensions(), image.dimensions());
    assert_eq!(stitched.as_raw(), image.as_raw());
}

#[test]
fn split_then_stitch_arbitrary_rows_is_bit_exact() {
    let image = gradient_rgb(37, 91);

    for splits in [
        vec![],
        vec![1],
        vec![90],
        vec![5, 6, 7],
        vec![10, 40, 41, 80],
    ] {
        let segments = split_image(&image, &splits).expect("legal split rows");
        assert_eq!(segments.len(), splits.len() + 1);
        let stitched = stitch_images(&segments).expect("uniform-width segments");
        assert_eq!(stitched.as_raw(), image.as_raw(), "splits {splits:?}");
    }
}

#[test]
fn segments_are_contiguous_half_open_ranges() {
    let image = gradient_rgb(9, 20);
    let splits = vec![4, 11];
    let segments = split_image(&image, &splits).expect("legal split rows");

    assert_eq!(
        segments.iter().map(|s| s.height()).collect::<Vec<_>>(),
        vec![4, 7, 9]
    );

    // Segment i row r must equal source row (offset + r): boundaries are half-open, so
    // the split row itself belongs to the *following* segment.
    let mut offset = 0;
    for segment in &segments {
        for row in 0..segment.height() {
            for x in 0..image.width() {
                assert_eq!(
                    segment.get_pixel(x, row),
                    image.get_pixel(x, offset + row),
                    "mismatch at ({x}, {row}) of segment starting at {offset}"
                );
            }
        }
        offset += segment.height();
    }
    assert_eq!(offset, image.height());
}

#[test]
fn split_and_stitch_are_generic_over_pixel_type() {
    let gray = gradient_gray(11, 30);
    let gray_segments = split_image(&gray, &[7, 22]).expect("legal split rows");
    assert_eq!(gray_segments.len(), 3);
    assert_eq!(
        stitch_images(&gray_segments)
            .expect("uniform width")
            .as_raw(),
        gray.as_raw()
    );

    let rgb = gradient_rgb(11, 30);
    let rgb_segments = split_image(&rgb, &[7, 22]).expect("legal split rows");
    assert_eq!(
        stitch_images(&rgb_segments)
            .expect("uniform width")
            .as_raw(),
        rgb.as_raw()
    );
}

#[test]
fn empty_split_list_yields_the_whole_image_as_one_segment() {
    let image = gradient_rgb(5, 6);
    let segments = split_image(&image, &[]).expect("empty splits are legal");

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].as_raw(), image.as_raw());
}

#[test]
fn split_rejects_illegal_row_lists() {
    let image = gradient_rgb(5, 10);

    for splits in [
        vec![0],    // row 0 would produce an empty leading segment
        vec![10],   // == height: empty trailing segment
        vec![11],   // beyond height
        vec![5, 5], // duplicate
        vec![6, 3], // not increasing
        vec![3, 0], // not increasing, and row 0
    ] {
        assert!(
            split_image(&image, &splits).is_err(),
            "expected rejection of {splits:?}"
        );
    }
}

#[test]
fn stitch_rejects_empty_input() {
    let segments: Vec<RgbImage> = Vec::new();
    assert!(stitch_images(&segments).is_err());
}

#[test]
fn stitch_rejects_mismatched_widths() {
    let segments = vec![gradient_rgb(5, 3), gradient_rgb(6, 3)];
    assert!(stitch_images(&segments).is_err());
}
