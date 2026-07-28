//! C4 tests -- pixel sets, IoU and dilation. Frozen gates.
//!
//! §10.7(B)'s shape metric is `IoU(G, O)` plus `O subset dilate(G, 2px)`, where
//! `G = {p : raw[p] != clean[p]}` and `O = {p : raw[p] != ours[p]}`. The definitions
//! pinned here are spec §16.5 items 8 (Chebyshev structuring element) and 9 (IoU of
//! two empty sets is 1.0).

use approx::assert_abs_diff_eq;
use pc_testkit::images::{gray_from_rows, solid_gray};
use pc_testkit::metrics::{diff_set_gray, iou, PixelSet};

const EPS: f64 = 1e-12;
const DIMS: (u32, u32) = (16, 16);

fn set(pixels: &[(u32, u32)]) -> PixelSet {
    PixelSet::from_pixels(DIMS, pixels.iter().copied())
}

// ------------------------------------------------------------------- IoU

#[test]
// IoU of a set with itself is exactly 1.0 -- the identity case §10.7(B)'s
// "IoU >= 0.99" target is measured against.
fn iou_of_identical_sets_is_one() {
    let a = set(&[(0, 0), (1, 0), (5, 7)]);
    assert_abs_diff_eq!(iou(&a, &a), 1.0, epsilon = EPS);
    assert_abs_diff_eq!(iou(&a, &set(&[(5, 7), (0, 0), (1, 0)])), 1.0, epsilon = EPS);
}

#[test]
// IoU of disjoint sets is exactly 0.0.
fn iou_of_disjoint_sets_is_zero() {
    let a = set(&[(0, 0), (1, 0)]);
    let b = set(&[(10, 10), (11, 10)]);
    assert_abs_diff_eq!(iou(&a, &b), 0.0, epsilon = EPS);
}

#[test]
// Hand-computed partial overlap:
//   A = {(0,0),(1,0),(0,1),(1,1)}   B = {(1,0),(1,1),(2,0),(2,1)}
//   |A intersect B| = 2, |A union B| = 6  ->  IoU = 1/3
fn iou_partial_overlap_hand_computed() {
    let a = set(&[(0, 0), (1, 0), (0, 1), (1, 1)]);
    let b = set(&[(1, 0), (1, 1), (2, 0), (2, 1)]);
    assert_eq!(a.intersection_len(&b), 2);
    assert_eq!(a.union_len(&b), 6);
    assert_abs_diff_eq!(iou(&a, &b), 1.0 / 3.0, epsilon = EPS);
}

#[test]
// One set contained in the other: |A intersect B| = |A| = 2, |A union B| = |B| = 4 -> 0.5
fn iou_of_a_subset_hand_computed() {
    let a = set(&[(0, 0), (1, 0)]);
    let b = set(&[(0, 0), (1, 0), (2, 0), (3, 0)]);
    assert_abs_diff_eq!(iou(&a, &b), 0.5, epsilon = EPS);
    assert!(a.is_subset_of(&b));
    assert!(!b.is_subset_of(&a));
}

#[test]
// spec §16.5 item 9 (decided here): IoU(empty, empty) == 1.0. This case is REACHABLE
// in §10.7(B) -- an image neither upstream nor we changed at all -- and 0/0 must not
// become NaN, which would silently fail every threshold comparison.
fn iou_of_two_empty_sets_is_one() {
    let empty = PixelSet::new(DIMS);
    assert_abs_diff_eq!(iou(&empty, &empty), 1.0, epsilon = EPS);
}

#[test]
// One empty, one not, is total disagreement: 0.0 in both argument orders.
fn iou_of_empty_against_non_empty_is_zero() {
    let empty = PixelSet::new(DIMS);
    let a = set(&[(3, 3)]);
    assert_abs_diff_eq!(iou(&empty, &a), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(iou(&a, &empty), 0.0, epsilon = EPS);
}

#[test]
// IoU is symmetric.
fn iou_is_symmetric() {
    let a = set(&[(0, 0), (1, 0), (0, 1), (1, 1)]);
    let b = set(&[(1, 0), (1, 1), (2, 0), (2, 1)]);
    assert_abs_diff_eq!(iou(&a, &b), iou(&b, &a), epsilon = EPS);
}

#[test]
// Comparing sets from different-sized canvases is meaningless and must panic rather
// than produce a plausible-looking number.
#[should_panic(expected = "dimension")]
fn iou_panics_on_canvas_mismatch() {
    let _ = iou(&PixelSet::new((4, 4)), &PixelSet::new((5, 4)));
}

// -------------------------------------------------------------- set basics

#[test]
// Sets deduplicate and iterate deterministically (§5.7) -- failure messages in the
// golden tests must be reproducible run to run.
fn sets_deduplicate_and_iterate_in_sorted_order() {
    let mut s = PixelSet::new(DIMS);
    assert!(s.insert((2, 1)));
    assert!(s.insert((0, 3)));
    assert!(!s.insert((2, 1)), "re-inserting must report false");
    assert_eq!(s.len(), 2);
    assert!(s.contains((2, 1)));
    assert!(!s.contains((1, 2)), "(x,y) order must not be transposed");

    let collected: Vec<(u32, u32)> = s.iter().collect();
    assert_eq!(collected, vec![(0, 3), (2, 1)]);
}

#[test]
// An out-of-bounds coordinate is a bug in the caller, caught immediately.
#[should_panic(expected = "out of bounds")]
fn inserting_out_of_bounds_panics() {
    let mut s = PixelSet::new((4, 4));
    s.insert((4, 0));
}

// ----------------------------------------------------------------- dilate

#[test]
// spec §16.5 item 8 (decided here): dilation uses a CHEBYSHEV (square) structuring
// element. A single pixel dilated by radius 2 becomes a full 5x5 block = 25 pixels.
// A Euclidean disc would give 21 and a 4-connected diamond 13, so this test
// distinguishes all three.
fn dilate_uses_a_chebyshev_square() {
    let s = PixelSet::from_pixels((11, 11), [(5, 5)]);
    let d = s.dilate(2);
    assert_eq!(d.len(), 25);
    assert!(d.contains((3, 3)), "corners of the square must be included");
    assert!(d.contains((7, 7)));
    assert!(!d.contains((2, 5)), "radius 2 must not reach 3 away");
}

#[test]
// Radius 0 is the identity -- §10.7(B) may be run with a 0 radius for a strict check.
fn dilate_by_zero_is_the_identity() {
    let s = PixelSet::from_pixels((11, 11), [(5, 5), (1, 2)]);
    assert_eq!(s.dilate(0), s);
}

#[test]
// Dilation clamps to the canvas: a corner pixel at (0,0) with radius 1 yields only
// the 4 in-bounds pixels, never negative or out-of-range coordinates.
fn dilate_clamps_at_the_canvas_edge() {
    let s = PixelSet::from_pixels((8, 8), [(0, 0)]);
    let d = s.dilate(1);
    assert_eq!(d.len(), 4);
    for p in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        assert!(d.contains(p), "missing {p:?}");
    }
    assert_eq!(d.dims(), (8, 8));
}

#[test]
// Dilation is monotone: the original set is always a subset of its dilation, so
// `O subset dilate(G, 2)` is trivially true whenever O subset G.
fn dilation_contains_the_original() {
    let s = PixelSet::from_pixels((16, 16), [(2, 3), (9, 9), (15, 0)]);
    assert!(s.is_subset_of(&s.dilate(1)));
    assert!(s.is_subset_of(&s.dilate(3)));
}

#[test]
// The §10.7(B) shape gate, end to end on a hand-built case: a 1px shift of the changed
// region is inside dilate(G, 1), but a 3px shift is not.
fn subset_of_dilated_discriminates_a_small_shift_from_a_large_one() {
    let g = PixelSet::from_pixels((16, 16), [(5, 5)]);
    let near = PixelSet::from_pixels((16, 16), [(6, 5)]);
    let far = PixelSet::from_pixels((16, 16), [(8, 5)]);
    assert!(near.is_subset_of(&g.dilate(1)));
    assert!(!far.is_subset_of(&g.dilate(1)));
    assert!(far.is_subset_of(&g.dilate(3)));
}

// --------------------------------------------------------------- diff sets

#[test]
// `diff_set_gray` is exactly `{p : a[p] != b[p]}` -- the definition §10.7(B) uses for
// both G and O.
fn diff_set_is_the_set_of_differing_pixels() {
    let a = gray_from_rows(&[&[0, 10, 20], &[30, 40, 50]]);
    let b = gray_from_rows(&[&[0, 11, 20], &[30, 40, 51]]);
    let d = diff_set_gray(&a, &b);
    assert_eq!(d.dims(), (3, 2));
    assert_eq!(d.len(), 2);
    assert!(d.contains((1, 0)));
    assert!(d.contains((2, 1)));
}

#[test]
// Identical images produce an empty diff set -- combined with the item-9 convention
// this makes "we changed nothing, upstream changed nothing" score IoU 1.0 rather than NaN.
fn diff_set_of_identical_images_is_empty() {
    let img = solid_gray(4, 4, 77);
    let d = diff_set_gray(&img, &img);
    assert!(d.is_empty());
    assert_abs_diff_eq!(iou(&d, &d), 1.0, epsilon = EPS);
}
