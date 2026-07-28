//! Task M3 -- spec §10.3's `border_std_deviation`, §10.7(A)4-7, §14.4, §16.9 items 7, 8.
//! FROZEN.

mod common;

use image::{DynamicImage, Luma, Rgb, RgbImage};
use pc_imageops::BinaryMask;
use pc_mask::border::{
    border_std_deviation, color_stats, edge_pixels, geometric_median, gray_stats, heuristic_median,
    numpy_median_u8, pil_luma, population_std, sample_std, snap_off_white, BaseCanvas, BlankMask,
};
use pc_testkit::assert_close;

fn gray_canvas(rows: &[&[u8]]) -> BaseCanvas {
    BaseCanvas::Gray(pc_testkit::images::gray_from_rows(rows))
}

/// A mask with a `w x h` block of set pixels at `(x, y)`.
fn block_mask(size: (u32, u32), x: u32, y: u32, w: u32, h: u32) -> BinaryMask {
    BinaryMask::from_fn(size.0, size.1, |px, py| {
        (x..x + w).contains(&px) && (y..y + h).contains(&py)
    })
}

// ------------------------------------------------------------ edge extraction (§10.7(A)4)

#[test]
fn a4_edges_of_a_3x3_block_are_its_eight_boundary_pixels() {
    // spec §10.7(A)4: for a 5x5 mask with a 3x3 set block at the centre, exactly the 8
    // boundary pixels of the block are edges -- the block's centre is not (all 8 of its
    // neighbours are set, so PIL's FIND_EDGES response there is 0).
    let mask = block_mask((5, 5), 1, 1, 3, 3);

    assert_eq!(
        edge_pixels(&mask),
        vec![
            (1, 1),
            (2, 1),
            (3, 1),
            (1, 2),
            (3, 2),
            (1, 3),
            (2, 3),
            (3, 3)
        ]
    );
}

#[test]
fn a4_a_fully_set_3x3_mask_has_eight_edges_its_ring() {
    // spec §10.7(A)4 says "all 9"; §16.9 item 21 corrects that to **8**. PIL leaves the
    // outermost 1-pixel ring unfiltered (8 pixels, copied as 255 -> truthy -> edges) and
    // filters only the centre, whose response is 255*8 - 8*255 = 0. The centre of a 3x3
    // is not on the 1-pixel border ring, so the parenthetical "all on the 1-px image
    // border" does not hold for it. §10.3 step 2's own formula already gives 8.
    let mask = BinaryMask::from_fn(3, 3, |_, _| true);

    let edges = edge_pixels(&mask);

    assert_eq!(edges.len(), 8);
    assert!(!edges.contains(&(1, 1)));
}

#[test]
fn a4_an_interior_pixel_of_a_large_block_is_not_an_edge() {
    let mask = block_mask((9, 9), 2, 2, 5, 5);

    let edges = edge_pixels(&mask);

    assert_eq!(
        edges.len(),
        16,
        "the 5x5 block's ring, not its 3x3 interior"
    );
    assert!(!edges.contains(&(4, 4)));
}

#[test]
fn a4_an_empty_mask_yields_blank_mask() {
    // spec §10.7(A)4 / §10.3 step 4: zero edges -> BlankMask, which abandons the region.
    let canvas = BaseCanvas::Gray(pc_testkit::images::solid_gray(5, 5, 128));
    let mask = BinaryMask::new(5, 5);

    assert_eq!(edge_pixels(&mask).len(), 0);
    assert_eq!(
        border_std_deviation(&canvas, &mask, 240, true),
        Err(BlankMask)
    );
}

// ------------------------------------------------------------ grayscale stats (§10.7(A)5)

#[test]
fn a5_grayscale_border_std_and_median_are_hand_computed() {
    // spec §10.7(A)5. A 5x5 base and a 3x3 centred mask: the 8 border-adjacent base
    // values, row-major, are [10, 20, 30, 40, 60, 70, 80, 90].
    //   mean = 400/8 = 50
    //   sum of squared deviations = 2*(1600+900+400+100) = 6000
    //   population variance = 6000/8 = 750  ->  std = sqrt(750)
    //   sorted middles are 40 and 60 -> median = 50.0 -> int() -> 50
    let canvas = gray_canvas(&[
        &[0, 0, 0, 0, 0],
        &[0, 10, 20, 30, 0],
        &[0, 40, 50, 60, 0],
        &[0, 70, 80, 90, 0],
        &[0, 0, 0, 0, 0],
    ]);
    let mask = block_mask((5, 5), 1, 1, 3, 3);

    let stats = border_std_deviation(&canvas, &mask, 240, true).expect("non-blank mask");

    assert_close(stats.std_deviation, 750.0_f64.sqrt(), 1e-12);
    assert_eq!(stats.median_color, [50, 50, 50]);
}

#[test]
fn a5_even_count_median_truncates_after_averaging_the_two_middles() {
    // spec §10.7(A)5: an even sample whose two middles are 100 and 101 gives
    // (100+101)/2 = 100.5 -> Python int() -> **100**, not 101.
    let values = [85_u8, 90, 95, 100, 101, 105, 110, 115];

    assert_eq!(numpy_median_u8(&values), 100);
    assert_eq!(gray_stats(&values).1, [100, 100, 100]);
}

#[test]
fn a5_odd_count_median_is_the_middle_element() {
    assert_eq!(numpy_median_u8(&[7, 1, 9]), 7);
}

#[test]
fn a5_population_std_divides_by_n_and_sample_std_by_n_minus_1() {
    // The two estimators are used on different paths (§10.3 steps 5 and 6); mixing them
    // up is exactly the kind of silent numeric bug §15.9 exists to prevent.
    let values = [1.0_f64, 3.0];

    assert_close(population_std(&values), 1.0, 1e-12);
    assert_close(sample_std(&values), 2.0_f64.sqrt(), 1e-12);
}

#[test]
fn a5_sample_std_of_one_value_is_zero_not_nan() {
    // DEVIATION §14.4: upstream's np.std(..., ddof=1) yields NaN for a single sample;
    // a single border pixel is trivially uniform, so v1 yields 0.0.
    assert_eq!(sample_std(&[42.0]), 0.0);
}

// ------------------------------------------------------------ off-white snap (§10.7(A)6)

#[test]
fn a6_off_white_snap_fires_on_the_minimum_channel() {
    // spec §10.7(A)6: (241,241,241) with a threshold of 240 snaps to white;
    // (240,255,255) does not, because the rule tests the *minimum* channel and is
    // strictly greater.
    assert_eq!(snap_off_white([241, 241, 241], 240), [255, 255, 255]);
    assert_eq!(snap_off_white([240, 255, 255], 240), [240, 255, 255]);
    assert_eq!(snap_off_white([0, 0, 0], 240), [0, 0, 0]);
}

#[test]
fn a6_off_white_snap_applies_to_the_grayscale_path_too() {
    // A near-white bubble interior: every border pixel is 250, so the median is 250,
    // which snaps to pure white before it is returned.
    let canvas = BaseCanvas::Gray(pc_testkit::images::solid_gray(5, 5, 250));
    let mask = block_mask((5, 5), 1, 1, 3, 3);

    let stats = border_std_deviation(&canvas, &mask, 240, true).expect("non-blank mask");

    assert_eq!(stats.median_color, [255, 255, 255]);
    assert_eq!(stats.std_deviation, 0.0);
}

// ------------------------------------------------------------ colour path (§10.7(A)7)

#[test]
fn a7_colour_heuristic_median_fires_on_a_strict_majority() {
    // spec §10.7(A)7: border pixels [(0,0,0) x3, (255,255,255)] -- 3 > 4/2, so the
    // heuristic median wins outright and Weiszfeld is never consulted.
    let colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255]];

    assert_eq!(heuristic_median(&colors), Some([0, 0, 0]));

    // Distances to the mean (63.75, 63.75, 63.75) are a = 63.75*sqrt(3) three times and
    // 3a once; the sample (ddof=1) std of [a,a,a,3a] is exactly a.
    let (std_deviation, median) = color_stats(&colors);
    assert_close(std_deviation, 63.75 * 3.0_f64.sqrt(), 1e-9);
    assert_eq!(median, [0, 0, 0]);
}

#[test]
fn a7_no_majority_falls_back_to_the_geometric_median() {
    // spec §10.7(A)7: two samples, no majority -> Weiszfeld runs and lands within 1 of
    // the true geometric median of {(0,0,0), (255,255,255)}, whose midpoint is 127.5.
    let colors = [[0, 0, 0], [255, 255, 255]];

    assert_eq!(heuristic_median(&colors), None);

    let median = geometric_median(&colors);
    for channel in median {
        assert!(
            channel.abs_diff(127) <= 1,
            "geometric median channel {channel} is not within 1 of 127"
        );
    }
    assert_eq!(color_stats(&colors).1, median);
}

#[test]
fn a7_geometric_median_of_coincident_points_is_that_point() {
    // The "all points coincide with the current estimate" termination branch.
    assert_eq!(geometric_median(&[[7, 8, 9], [7, 8, 9]]), [7, 8, 9]);
}

#[test]
fn a7_colour_path_is_used_for_an_rgb_canvas_when_colours_are_allowed() {
    // A 3x3 fully-set mask has 8 edges -- its ring, centre excluded (§16.9 item 21).
    // 5 of those 8 are exactly (10,20,30), a strict majority (10 > 8), so that triple is
    // the median -- a value no grayscale path could ever produce.
    let majority = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1)];
    let canvas = BaseCanvas::Rgb(RgbImage::from_fn(3, 3, |x, y| {
        if majority.contains(&(x, y)) {
            Rgb([10, 20, 30])
        } else {
            Rgb([200, 100, 50])
        }
    }));
    let mask = BinaryMask::from_fn(3, 3, |_, _| true);

    let stats = border_std_deviation(&canvas, &mask, 240, true).expect("non-blank mask");

    assert_eq!(stats.median_color, [10, 20, 30]);
    assert!(stats.std_deviation > 0.0);
}

// ------------------------------------------------------- canvas selection (§16.9 item 7)

#[test]
fn pil_luma_uses_the_itu_r_601_2_coefficients_with_truncation() {
    // spec §10.3 step 1 of border_std_deviation: L = (R*299 + G*587 + B*114) / 1000,
    // truncating. The `image` crate's to_luma8 uses different coefficients and must not
    // be used here.
    assert_eq!(pil_luma(255, 0, 0), 76);
    assert_eq!(pil_luma(0, 255, 0), 149);
    assert_eq!(pil_luma(0, 0, 255), 29);
    assert_eq!(pil_luma(100, 100, 100), 100);
}

#[test]
fn disallowing_coloured_masks_forces_the_grayscale_path_on_an_rgb_canvas() {
    // spec §16.9 item 7: the grayscale path is taken when the canvas is Gray OR
    // !allow_colored_masks; in the latter case the reduction is pil_luma, so a uniform
    // pure-red canvas yields the achromatic median (76,76,76), not (255,0,0).
    let canvas = BaseCanvas::Rgb(RgbImage::from_pixel(3, 3, Rgb([255, 0, 0])));
    let mask = BinaryMask::from_fn(3, 3, |_, _| true);

    let stats = border_std_deviation(&canvas, &mask, 240, false).expect("non-blank mask");

    assert_eq!(stats.median_color, [76, 76, 76]);
    assert_eq!(stats.std_deviation, 0.0);

    let coloured = border_std_deviation(&canvas, &mask, 240, true).expect("non-blank mask");
    assert_eq!(coloured.median_color, [255, 0, 0]);
}

#[test]
fn base_canvas_from_dynamic_maps_luma_sources_to_gray() {
    // §16.9 item 7: L/LA -> Gray, everything else -> Rgb. This is also what §15.3's
    // output-mode rule keys off.
    let gray = DynamicImage::ImageLuma8(pc_testkit::images::solid_gray(2, 2, 5));
    let rgb = DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 2, Rgb([5, 5, 5])));

    assert!(BaseCanvas::from_dynamic(&gray).is_gray());
    assert!(!BaseCanvas::from_dynamic(&rgb).is_gray());
    assert_eq!(BaseCanvas::from_dynamic(&gray).color_at(0, 0), [5, 5, 5]);
}

#[test]
fn base_canvas_crop_is_none_for_a_degenerate_rect() {
    // spec §16.9 item 11: the caller turns this into a skipped region, never an error.
    let canvas = BaseCanvas::Gray(pc_testkit::images::solid_gray(10, 10, 1));

    assert!(canvas.crop(pc_core::Rect::new(2, 2, 2, 8)).is_none());
    assert!(canvas.crop(pc_core::Rect::new(50, 50, 60, 60)).is_none());
    let cropped = canvas
        .crop(pc_core::Rect::new(2, 3, 6, 9))
        .expect("in-bounds rect");
    assert_eq!(cropped.dimensions(), (4, 6));
}

#[test]
fn gray_canvas_luma_at_reads_the_stored_value() {
    let canvas = gray_canvas(&[&[0, 255], &[128, 64]]);

    assert_eq!(canvas.luma_at(1, 0), 255);
    assert_eq!(canvas.luma_at(0, 1), 128);
    assert_eq!(
        *pc_testkit::images::gray_from_rows(&[&[0, 255], &[128, 64]]).get_pixel(1, 1),
        Luma([64])
    );
}
