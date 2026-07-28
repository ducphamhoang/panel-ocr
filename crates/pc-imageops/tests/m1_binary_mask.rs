//! Task M1 -- spec §10.2, §10.3 steps 0-2, §16.9 items 1-4. FROZEN.

use image::{GrayImage, Luma};
use pc_core::Rect;
use pc_imageops::mask::{rasterize_boxes, BinaryMask, PIL_BINARY_THRESHOLD};

fn set_pixels(mask: &BinaryMask) -> Vec<(u32, u32)> {
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

// ------------------------------------------------------------ rasterisation (§16.9 item 1)

#[test]
fn rasterize_boxes_is_exclusive_on_x2_and_y2() {
    // spec §16.9 item 1 (superseding §10.3 step 1's "inclusive"): x2/y2 are exclusive,
    // exactly like `Rect::to_crop` and `pc_detect::mask::rasterize_union`. A 2x2 rect
    // covers 4 pixels, not 9.
    let mask = rasterize_boxes(&[Rect::new(1, 1, 3, 3)], (5, 5));

    assert_eq!(set_pixels(&mask), vec![(1, 1), (2, 1), (1, 2), (2, 2)]);
    assert_eq!(mask.count_set(), 4);
}

#[test]
fn rasterize_boxes_unions_and_clamps_to_the_canvas() {
    // A rect overhanging the right/bottom edge contributes only its in-canvas part;
    // a fully out-of-bounds or empty rect contributes nothing.
    let mask = rasterize_boxes(
        &[
            Rect::new(0, 0, 2, 1),
            Rect::new(3, 3, 10, 10),
            Rect::new(20, 20, 30, 30),
            Rect::new(1, 1, 1, 4),
        ],
        (4, 4),
    );

    assert_eq!(set_pixels(&mask), vec![(0, 0), (1, 0), (3, 3)]);
}

#[test]
fn rasterize_boxes_with_no_rects_is_blank() {
    let mask = rasterize_boxes(&[], (3, 3));

    assert!(mask.is_blank());
    assert_eq!(mask.dimensions(), (3, 3));
}

// ------------------------------------------------------------ thresholding (§16.9 item 4)

#[test]
fn from_gray_threshold_is_strict_and_pil_uses_127() {
    // spec §10.3 step 0: PIL's `L -> 1` without dither is `value > 127`. Strictly
    // greater, so 127 itself is background.
    assert_eq!(PIL_BINARY_THRESHOLD, 127);
    let image = GrayImage::from_fn(4, 1, |x, _| Luma([[0_u8, 127, 128, 255][x as usize]]));

    let mask = BinaryMask::from_gray_threshold(&image, PIL_BINARY_THRESHOLD);

    assert_eq!(set_pixels(&mask), vec![(2, 0), (3, 0)]);
}

#[test]
fn to_gray_emits_only_0_and_255_and_round_trips() {
    let mask = BinaryMask::from_fn(3, 2, |x, y| (x + y) % 2 == 0);

    let gray = mask.to_gray();

    assert!(gray.pixels().all(|pixel| matches!(pixel.0[0], 0 | 255)));
    assert_eq!(BinaryMask::from_gray_threshold(&gray, 127), mask);
}

// ------------------------------------------------------------ set algebra

#[test]
fn and_or_are_pixel_wise() {
    // spec §10.3 step 2: `cut = precise AND box_mask`.
    let left = BinaryMask::from_fn(2, 1, |x, _| x == 0);
    let right = BinaryMask::from_fn(2, 1, |_, _| true);

    assert_eq!(set_pixels(&left.and(&right)), vec![(0, 0)]);
    assert_eq!(set_pixels(&left.or(&right)), vec![(0, 0), (1, 0)]);
}

#[test]
#[should_panic(expected = "binary mask dimension mismatch")]
fn and_panics_on_a_dimension_mismatch() {
    // spec §16.9 item 3: a programming error, not a per-image condition.
    let _ = BinaryMask::new(2, 2).and(&BinaryMask::new(3, 2));
}

// ------------------------------------------------------------ bbox / blankness

#[test]
fn bbox_uses_the_exclusive_convention_and_is_none_when_blank() {
    // spec §16.9 item 3: a single set pixel at (2,3) gives Rect::new(2,3,3,4).
    let mut mask = BinaryMask::new(6, 6);
    assert_eq!(mask.bbox(), None);
    assert!(mask.is_blank());

    mask.set(2, 3, true);
    assert_eq!(mask.bbox(), Some(Rect::new(2, 3, 3, 4)));

    mask.set(4, 1, true);
    assert_eq!(mask.bbox(), Some(Rect::new(2, 1, 5, 4)));
    assert!(!mask.is_blank());
}

#[test]
fn get_outside_the_canvas_is_false_not_a_panic() {
    // Neighbour scans in `border::is_edge` rely on this.
    let mask = BinaryMask::from_fn(2, 2, |_, _| true);

    assert!(!mask.get(2, 0));
    assert!(!mask.get(0, 99));
}

// ------------------------------------------------------------ crop_into (§16.9 item 3)

#[test]
fn crop_into_places_the_window_at_the_offset() {
    // spec §10.3 step 3: the masking-rect window of the page-sized cut mask is pasted
    // at (x_offset, y_offset) into a zeroed mask the size of the reference crop.
    let page = BinaryMask::from_fn(10, 10, |x, y| x == 5 && y == 6);

    let cropped = page.crop_into(Rect::new(4, 5, 8, 9), (6, 6), (1, 1));

    assert_eq!(cropped.dimensions(), (6, 6));
    // source (5,6) - rect origin (4,5) = (1,1), plus offset (1,1) => (2,2)
    assert_eq!(set_pixels(&cropped), vec![(2, 2)]);
}

#[test]
fn crop_into_clamps_the_source_and_drops_out_of_target_writes() {
    // A rect that overhangs the canvas still lands correctly: the *source* read is
    // clamped, the destination mapping is unchanged (§16.9 item 3).
    let page = BinaryMask::from_fn(4, 4, |x, y| x == 0 && y == 0);

    let cropped = page.crop_into(Rect::new(-2, -2, 2, 2), (4, 4), (0, 0));
    // source (0,0) - rect origin (-2,-2) = (2,2)
    assert_eq!(set_pixels(&cropped), vec![(2, 2)]);

    // Same window, but the destination is too small to hold it.
    let clipped = page.crop_into(Rect::new(-2, -2, 2, 2), (2, 2), (0, 0));
    assert!(clipped.is_blank());
}

#[test]
fn crop_into_of_a_degenerate_rect_is_blank() {
    let page = BinaryMask::from_fn(4, 4, |_, _| true);

    assert!(page
        .crop_into(Rect::new(2, 2, 2, 4), (4, 4), (0, 0))
        .is_blank());
    assert!(page
        .crop_into(Rect::new(90, 90, 95, 95), (4, 4), (0, 0))
        .is_blank());
}
