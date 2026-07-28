//! Task D6 -- spec §8.3 step 5, §8.7(A)4, §16.6 items 5/6/7. FROZEN.

mod common;

use common::{assert_binary, nonzero_count};
use image::{GrayImage, Luma};
use pc_core::Rect;
use pc_detect::mask::{
    crop_letterbox, dilate_l1, postprocess_mask, rasterize_union, refine_simple, resize_bilinear,
    REFINE_DILATE_RADIUS, REFINE_EXPAND, REFINE_THRESHOLD,
};
use pc_detect::yolo::LetterboxGeometry;
use pc_detect::RawBlock;

fn block(rect: Rect) -> RawBlock {
    RawBlock {
        rect,
        class_index: 0,
        confidence: 1.0,
    }
}

/// No letterbox padding, mask already at base-image size.
fn identity_geometry(size: (u32, u32)) -> LetterboxGeometry {
    LetterboxGeometry {
        net_size: size.0.max(size.1),
        dw: 0.0,
        dh: 0.0,
        image_size: size,
    }
}

// ------------------------------------------------------------ constants

#[test]
fn constants_match_the_spec() {
    assert_eq!(REFINE_THRESHOLD, 60);
    assert_eq!(REFINE_EXPAND, 16);
    assert_eq!(REFINE_DILATE_RADIUS, 3);
}

// ------------------------------------------------------------ postprocess_mask

#[test]
fn postprocess_mask_clamps_then_truncates() {
    // spec §16.6 item 7: clamp to [0, 255] THEN truncate -- upstream's
    // `(img * 255).astype(np.uint8)` truncates; the clamp is the one correction
    // (astype wraps out-of-range floats).
    let values = [-0.5_f32, 0.0, 0.25, 0.5, 0.99999, 1.0, 2.0];

    let mask = postprocess_mask(&values, 7, 1).expect("well-sized input");

    let actual: Vec<u8> = mask.pixels().map(|pixel| pixel.0[0]).collect();
    assert_eq!(actual, vec![0, 0, 63, 127, 254, 255, 255]);
}

#[test]
fn postprocess_mask_rejects_a_length_mismatch() {
    assert!(postprocess_mask(&[0.0, 1.0, 0.5], 2, 2).is_err());
}

// ------------------------------------------------------------ crop_letterbox

#[test]
fn crop_letterbox_keeps_the_top_left_region() {
    // Padding is added right/bottom only (spec §8.3 step 3), so cropping it is a
    // top-left crop of `W - dw` by `H - dh`.
    let source = GrayImage::from_fn(8, 8, |x, y| Luma([(y * 8 + x) as u8]));

    let cropped = crop_letterbox(&source, 2, 3).expect("padding fits");

    assert_eq!(cropped.dimensions(), (6, 5));
    for y in 0..5 {
        for x in 0..6 {
            assert_eq!(cropped.get_pixel(x, y), source.get_pixel(x, y));
        }
    }
}

#[test]
fn crop_letterbox_with_no_padding_is_the_identity() {
    let source = GrayImage::from_fn(4, 3, |x, y| Luma([(x + y) as u8]));

    let cropped = crop_letterbox(&source, 0, 0).expect("no padding");

    assert_eq!(cropped.as_raw(), source.as_raw());
}

#[test]
fn crop_letterbox_rejects_padding_larger_than_the_image() {
    let source = GrayImage::from_pixel(4, 4, Luma([1]));

    assert!(crop_letterbox(&source, 4, 0).is_err());
    assert!(crop_letterbox(&source, 0, 9).is_err());
}

// ------------------------------------------------------------ resize_bilinear

#[test]
fn resize_bilinear_uses_half_pixel_centres_with_border_clamping() {
    // spec §16.6 item 6: src = (dst + 0.5) * (src_len / dst_len) - 0.5, clamped.
    // 2 -> 4 columns: src centres are -0.25, 0.25, 0.75, 1.25 -> clamp, clamp-free,
    // clamp-free, clamp.
    let source = pc_testkit::images::gray_from_rows(&[&[0, 100]]);

    let resized = resize_bilinear(&source, 4, 1);

    assert_eq!(resized.dimensions(), (4, 1));
    let actual: Vec<u8> = resized.pixels().map(|pixel| pixel.0[0]).collect();
    assert_eq!(actual, vec![0, 25, 75, 100]);
}

#[test]
fn resize_bilinear_downscale_averages_neighbouring_pairs() {
    // 4 -> 2 columns: src centres are 0.5 and 2.5, i.e. exact midpoints.
    let source = pc_testkit::images::gray_from_rows(&[&[0, 40, 80, 120]]);

    let resized = resize_bilinear(&source, 2, 1);

    let actual: Vec<u8> = resized.pixels().map(|pixel| pixel.0[0]).collect();
    assert_eq!(actual, vec![20, 100]);
}

#[test]
fn resize_bilinear_scales_the_two_axes_independently() {
    let source = pc_testkit::images::gray_from_rows(&[&[0, 100], &[0, 100]]);

    let resized = resize_bilinear(&source, 4, 2);

    assert_eq!(resized.dimensions(), (4, 2));
    for y in 0..2 {
        assert_eq!(resized.get_pixel(1, y).0[0], 25);
        assert_eq!(resized.get_pixel(2, y).0[0], 75);
    }
}

#[test]
fn resize_bilinear_to_the_same_size_is_the_identity() {
    let source = pc_testkit::images::gray_from_rows(&[&[3, 9, 27], &[81, 243, 5]]);

    let resized = resize_bilinear(&source, 3, 2);

    assert_eq!(resized.as_raw(), source.as_raw());
}

#[test]
fn resize_bilinear_of_a_uniform_image_is_uniform() {
    let source = GrayImage::from_pixel(7, 3, Luma([137]));

    let resized = resize_bilinear(&source, 19, 11);

    assert!(resized.pixels().all(|pixel| pixel.0[0] == 137));
}

// ------------------------------------------------------------ dilate_l1

#[test]
fn dilate_l1_radius_3_is_a_25_cell_diamond() {
    // |dx| + |dy| <= 3 has 1 + 4 + 8 + 12 = 25 cells.
    let mut source = GrayImage::from_pixel(15, 15, Luma([0]));
    source.put_pixel(7, 7, Luma([255]));

    let dilated = dilate_l1(&source, 3);

    assert_eq!(nonzero_count(&dilated), 25);
    for y in 0..15_i32 {
        for x in 0..15_i32 {
            let inside = (x - 7).abs() + (y - 7).abs() <= 3;
            let value = dilated.get_pixel(x as u32, y as u32).0[0];
            assert_eq!(value != 0, inside, "at ({x}, {y})");
        }
    }
    // Spot checks on the diamond's actual shape: the tips reach 3, the diagonals do not.
    assert_eq!(dilated.get_pixel(7, 4).0[0], 255);
    assert_eq!(dilated.get_pixel(7, 3).0[0], 0);
    assert_eq!(dilated.get_pixel(6, 5).0[0], 255, "|1| + |2| = 3");
    assert_eq!(dilated.get_pixel(5, 5).0[0], 0, "|2| + |2| = 4");
}

#[test]
fn dilate_l1_radius_zero_is_the_identity() {
    let source = pc_testkit::images::gray_from_rows(&[&[0, 255], &[255, 0]]);

    assert_eq!(dilate_l1(&source, 0).as_raw(), source.as_raw());
}

#[test]
fn dilate_l1_clamps_at_the_image_border() {
    let mut source = GrayImage::from_pixel(4, 4, Luma([0]));
    source.put_pixel(0, 0, Luma([255]));

    let dilated = dilate_l1(&source, 1);

    // Only the in-bounds part of the diamond exists: (0,0), (1,0), (0,1).
    assert_eq!(nonzero_count(&dilated), 3);
    assert_eq!(dilated.dimensions(), (4, 4));
}

#[test]
fn dilate_l1_takes_the_maximum_not_a_binarisation() {
    let source = pc_testkit::images::gray_from_rows(&[&[10, 200, 10]]);

    let dilated = dilate_l1(&source, 1);

    let actual: Vec<u8> = dilated.pixels().map(|pixel| pixel.0[0]).collect();
    assert_eq!(actual, vec![200, 200, 200]);
}

// ------------------------------------------------------------ rasterize_union

#[test]
fn expanded_box_right_edge_is_exclusive() {
    // spec §16.6 item 5 (settled): x2/y2 are EXCLUSIVE everywhere, matching
    // `pc_core::Rect::to_crop`. A 1..3 x 1..3 rect covers exactly 4 pixels.
    let mask = rasterize_union(&[Rect::new(1, 1, 3, 3)], (4, 4));

    assert_eq!(nonzero_count(&mask), 4);
    assert_eq!(mask.get_pixel(1, 1).0[0], 255);
    assert_eq!(mask.get_pixel(2, 2).0[0], 255);
    assert_eq!(mask.get_pixel(3, 3).0[0], 0, "x2/y2 are exclusive");
    assert_eq!(mask.get_pixel(0, 1).0[0], 0);
    assert_binary(&mask);
}

#[test]
fn rasterize_union_unions_overlapping_rects_and_clips_to_the_canvas() {
    let mask = rasterize_union(&[Rect::new(0, 0, 2, 2), Rect::new(1, 1, 8, 8)], (4, 4));

    assert_eq!(mask.dimensions(), (4, 4));
    // (0,0)..(2,2) is 4 px; (1,1)..(4,4) clipped is 9 px; they share (1,1).
    assert_eq!(nonzero_count(&mask), 12);
    assert_eq!(mask.get_pixel(3, 3).0[0], 255);
}

#[test]
fn rasterize_union_of_nothing_is_all_zero() {
    let mask = rasterize_union(&[], (5, 5));

    assert_eq!(mask.dimensions(), (5, 5));
    assert_eq!(nonzero_count(&mask), 0);
}

#[test]
fn rasterize_union_ignores_degenerate_and_out_of_bounds_rects() {
    let mask = rasterize_union(&[Rect::new(2, 2, 2, 2), Rect::new(-8, -8, -1, -1)], (4, 4));

    assert_eq!(nonzero_count(&mask), 0);
}

// ------------------------------------------------------------ refine_simple (§8.7(A)4)

#[test]
fn a4_threshold_is_strict_at_60() {
    // spec §8.7(A)4: value 61 survives, value 60 does not. The two seeds are 20px apart
    // so the radius-3 dilation of the survivor cannot reach the other one.
    let mut mask = GrayImage::from_pixel(64, 64, Luma([0]));
    mask.put_pixel(20, 20, Luma([61]));
    mask.put_pixel(40, 40, Luma([60]));

    let refined = refine_simple(
        &mask,
        &identity_geometry((64, 64)),
        &[block(Rect::new(16, 16, 32, 32))],
    )
    .expect("well-formed input");

    assert_eq!(refined.get_pixel(20, 20).0[0], 255);
    assert_eq!(refined.get_pixel(40, 40).0[0], 0);
    assert_binary(&refined);
    assert_eq!(
        nonzero_count(&refined),
        25,
        "one seed, dilated to a 25-cell diamond"
    );
}

#[test]
fn a4_nothing_survives_outside_the_expanded_boxes() {
    // spec §8.7(A)4: a lone pixel at 200 outside every expanded box is zeroed, and no
    // output pixel lies outside `rect.pad(16)`.
    let mut mask = GrayImage::from_pixel(64, 64, Luma([0]));
    mask.put_pixel(20, 20, Luma([200]));
    mask.put_pixel(60, 60, Luma([200]));

    let rect = Rect::new(16, 16, 32, 32);
    let refined = refine_simple(&mask, &identity_geometry((64, 64)), &[block(rect)])
        .expect("well-formed input");

    assert_eq!(refined.get_pixel(60, 60).0[0], 0);

    let expanded = rect.pad(REFINE_EXPAND, (64, 64));
    for (x, y, pixel) in refined.enumerate_pixels() {
        if pixel.0[0] != 0 {
            assert!(
                expanded.to_crop((64, 64)).is_some_and(|(cx, cy, cw, ch)| {
                    x >= cx && x < cx + cw && y >= cy && y < cy + ch
                }),
                "pixel ({x}, {y}) survived outside the expanded box {expanded:?}"
            );
        }
    }
}

#[test]
fn refine_simple_with_no_blocks_is_all_zero() {
    // spec §8.3 step 5: "If `blocks` is empty, the refined mask is all-zero."
    let mask = GrayImage::from_pixel(32, 32, Luma([255]));

    let refined = refine_simple(&mask, &identity_geometry((32, 32)), &[]).expect("no blocks");

    assert_eq!(refined.dimensions(), (32, 32));
    assert_eq!(nonzero_count(&refined), 0);
}

#[test]
fn refine_simple_output_is_always_the_base_image_size() {
    // The mask arrives at network resolution with letterbox padding; the refined mask
    // must come back at base-image resolution (spec §8.3 step 5).
    let mask = GrayImage::from_pixel(16, 16, Luma([255]));
    let geometry = LetterboxGeometry {
        net_size: 16,
        dw: 0.0,
        dh: 8.0,
        image_size: (32, 16),
    };

    let refined = refine_simple(&mask, &geometry, &[block(Rect::new(0, 0, 32, 16))])
        .expect("well-formed input");

    assert_eq!(refined.dimensions(), (32, 16));
    assert_binary(&refined);
}
