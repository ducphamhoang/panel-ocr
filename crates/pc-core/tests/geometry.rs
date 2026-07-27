//! C1 tests — spec §2.1 `Rect` semantics. Every assertion below is a frozen gate.

use pc_core::Rect;

// ---------------------------------------------------------------- basic accessors

#[test]
// spec §2.1: width = x2-x1, height = y2-y1, area = width*height widened to i64
fn dimensions_and_area() {
    let r = Rect::new(10, 20, 40, 60);
    assert_eq!(r.width(), 30);
    assert_eq!(r.height(), 40);
    assert_eq!(r.area(), 1_200i64);
}

#[test]
// spec §2.1: area() returns i64 because upstream's Python int is unbounded — a
// 60_000 x 60_000 rect must not overflow the way i32 multiplication would
fn area_does_not_overflow_i32() {
    let r = Rect::new(0, 0, 60_000, 60_000);
    assert_eq!(r.area(), 3_600_000_000i64);
}

#[test]
// spec §2.1: is_empty() is width <= 0 || height <= 0
fn is_empty_covers_zero_and_inverted() {
    assert!(!Rect::new(0, 0, 1, 1).is_empty());
    assert!(Rect::new(0, 0, 0, 10).is_empty()); // zero width
    assert!(Rect::new(0, 0, 10, 0).is_empty()); // zero height
    assert!(Rect::new(10, 0, 5, 10).is_empty()); // inverted width
    assert!(Rect::new(0, 10, 10, 5).is_empty()); // inverted height
}

// ---------------------------------------------------------------- center

#[test]
// spec §2.1: center is ((x1+x2)/2, (y1+y2)/2) with Python floor division —
// implemented as (x1+x2).div_euclid(2)
fn center_floor_divides() {
    assert_eq!(Rect::new(0, 0, 10, 10).center(), (5, 5));
    // odd sums floor, they do not round
    assert_eq!(Rect::new(0, 0, 5, 7).center(), (2, 3));
    assert_eq!(Rect::new(1, 1, 2, 2).center(), (1, 1));
}

#[test]
// spec §2.1: floor-division, NOT truncation — a negative sum must floor toward
// -inf like Python's `//` (-3 // 2 == -2), which `/ 2` in Rust would get wrong (-1)
fn center_floors_negative_sums() {
    assert_eq!(Rect::new(-3, -3, 0, 0).center(), (-2, -2));
    assert_eq!(Rect::new(-5, -7, -2, -2).center(), (-4, -5));
}

// ---------------------------------------------------------------- contains

#[test]
// spec §2.1: contains() is INCLUSIVE on both ends (x1 <= x <= x2), deliberately
// mismatching the exclusive x2 used for cropping — do not "fix" this
fn contains_is_inclusive_on_both_ends() {
    let r = Rect::new(0, 0, 10, 10);
    assert!(r.contains((0, 0)));
    assert!(r.contains((10, 10))); // the far corner IS contained
    assert!(r.contains((5, 5)));
    assert!(!r.contains((11, 10)));
    assert!(!r.contains((10, 11)));
    assert!(!r.contains((-1, 0)));
    assert!(!r.contains((0, -1)));
}

#[test]
// spec §2.5 invariant support: reference superset-of masking needs a rect-containment test
fn contains_rect_is_inclusive_and_directional() {
    let outer = Rect::new(0, 0, 100, 100);
    let inner = Rect::new(10, 10, 90, 90);
    assert!(outer.contains_rect(&inner));
    assert!(!inner.contains_rect(&outer));
    // coincident edges still count as contained
    assert!(outer.contains_rect(&outer));
    assert!(outer.contains_rect(&Rect::new(0, 0, 100, 50)));
    // one coordinate outside is enough to fail
    assert!(!outer.contains_rect(&Rect::new(0, 0, 101, 50)));
    assert!(!outer.contains_rect(&Rect::new(-1, 0, 100, 50)));
}

// ---------------------------------------------------------------- merge

#[test]
// spec §2.1: merge() is the bounding union
fn merge_is_bounding_union() {
    let a = Rect::new(0, 0, 10, 10);
    let b = Rect::new(20, 5, 30, 40);
    assert_eq!(a.merge(&b), Rect::new(0, 0, 30, 40));
    assert_eq!(b.merge(&a), Rect::new(0, 0, 30, 40)); // commutative
    // a contained box changes nothing
    assert_eq!(a.merge(&Rect::new(2, 2, 4, 4)), a);
}

// ---------------------------------------------------------------- overlaps

#[test]
// spec §2.1 + §9.7(A)2: overlaps() threshold is STRICTLY greater — a ratio of
// exactly 0.20 against threshold_percent 20.0 must NOT overlap
fn overlaps_threshold_is_strictly_greater_at_the_boundary() {
    // A and B are both 100x100 = area 10_000; x overlap is 20px over full height
    // => intersection 2_000; min area 10_000 => ratio exactly 0.20
    let a = Rect::new(0, 0, 100, 100);
    let b = Rect::new(80, 0, 180, 100);
    assert!(!a.overlaps(&b, 20.0), "ratio 0.20 must not exceed threshold 20.0");
    assert!(!b.overlaps(&a, 20.0), "must be symmetric");
    // nudge the threshold below the ratio: now strictly greater holds
    assert!(a.overlaps(&b, 19.99));
    // nudge the ratio above the threshold instead: 21px overlap => 0.21
    let c = Rect::new(79, 0, 179, 100);
    assert!(a.overlaps(&c, 20.0));
}

#[test]
// spec §2.1: overlaps uses min(area, other.area) as the divisor, so a small box
// mostly inside a large one overlaps strongly even though the ratio against the
// large box's area would be tiny
fn overlaps_divides_by_the_smaller_area() {
    let big = Rect::new(0, 0, 1000, 1000); // area 1_000_000
    let small = Rect::new(0, 0, 10, 10); // area 100, fully inside
    // intersection 100 / min area 100 = 1.0
    assert!(big.overlaps(&small, 99.0));
    assert!(small.overlaps(&big, 99.0));
}

#[test]
// spec §2.1 + §9.7(A)2: a zero-area box must not panic or produce NaN — the
// divisor is forced to 1 when min(area, other.area) == 0
fn overlaps_zero_area_divisor_is_forced_to_one() {
    let degenerate = Rect::new(5, 5, 5, 5); // area 0
    let normal = Rect::new(0, 0, 10, 10);
    // 0 / 1 = 0.0, which is never strictly greater than a non-negative threshold
    assert!(!degenerate.overlaps(&normal, 0.0));
    assert!(!normal.overlaps(&degenerate, 0.0));
    assert!(!degenerate.overlaps(&degenerate, 0.0));
    // zero-height and zero-width variants behave the same
    assert!(!Rect::new(0, 5, 10, 5).overlaps(&normal, 0.0));
    assert!(!Rect::new(5, 0, 5, 10).overlaps(&normal, 0.0));
}

#[test]
// spec §2.1: no intersection => ratio 0 => never overlaps, and the clamping of
// negative overlap extents to 0 must not let a negative product read as positive
fn overlaps_disjoint_boxes_never_overlap() {
    let a = Rect::new(0, 0, 10, 10);
    // separated on BOTH axes: naive x_ov*y_ov without the max(0,..) clamp would
    // multiply two negatives into a positive intersection
    let far = Rect::new(20, 20, 30, 30);
    assert!(!a.overlaps(&far, 0.0));
    // separated on one axis only
    assert!(!a.overlaps(&Rect::new(20, 0, 30, 10), 0.0));
    assert!(!a.overlaps(&Rect::new(0, 20, 10, 30), 0.0));
}

// ---------------------------------------------------------------- overlaps_center

#[test]
// spec §2.1: overlaps_center is other.contains(self.center()) OR
// self.contains(other.center()) — the OR makes it asymmetric-tolerant
fn overlaps_center_is_a_disjunction() {
    let a = Rect::new(0, 0, 100, 100); // center (50,50)
    // b contains a's center but a does not contain b's center
    let b = Rect::new(40, 40, 400, 400); // center (220,220)
    assert!(b.contains(a.center()));
    assert!(!a.contains(b.center()));
    assert!(a.overlaps_center(&b));
    assert!(b.overlaps_center(&a));
}

#[test]
// spec §2.1: mutual containment of centers, and the disjoint negative case
fn overlaps_center_mutual_and_disjoint() {
    let a = Rect::new(0, 0, 10, 10); // center (5,5)
    let b = Rect::new(4, 4, 14, 14); // center (9,9)
    assert!(a.overlaps_center(&b));
    let c = Rect::new(100, 100, 110, 110);
    assert!(!a.overlaps_center(&c));
    assert!(!c.overlaps_center(&a));
}

#[test]
// spec §2.1: overlaps_center relies on contains(), which is inclusive — a center
// landing exactly on the far edge counts
fn overlaps_center_edge_hit_counts() {
    let a = Rect::new(0, 0, 10, 10); // center (5,5)
    // b's x1,y1 == a's center: a's center is on b's inclusive boundary
    let b = Rect::new(5, 5, 25, 25);
    assert!(a.overlaps_center(&b));
}

// ---------------------------------------------------------------- padding

#[test]
// spec §2.1: pad() clamps x1,y1 up to 0 and x2,y2 down to the canvas
fn pad_clamps_all_four_sides() {
    let interior = Rect::new(20, 20, 30, 30);
    assert_eq!(interior.pad(5, (100, 100)), Rect::new(15, 15, 35, 35));

    let touching = Rect::new(5, 5, 95, 95);
    // x1: 5-10 = -5 -> 0 ; x2: 95+10 = 105 -> 100
    assert_eq!(touching.pad(10, (100, 100)), Rect::new(0, 0, 100, 100));

    // already-flush box cannot grow past the canvas
    let flush = Rect::new(0, 0, 100, 100);
    assert_eq!(flush.pad(50, (100, 100)), flush);
}

#[test]
// spec §2.1: pad(0) is identity; a negative amount shrinks but still clamps
fn pad_zero_is_identity() {
    let r = Rect::new(20, 20, 30, 30);
    assert_eq!(r.pad(0, (100, 100)), r);
}

#[test]
// spec §2.1 + §9.7(A)6: right_pad() touches ONLY x2, and clamps it to the canvas
// width — a box flush against the right edge yields x2 == width, never width+a
fn right_pad_only_extends_x2_and_clamps() {
    let r = Rect::new(20, 20, 98, 30);
    assert_eq!(r.right_pad(5, (100, 100)), Rect::new(20, 20, 100, 30));

    let interior = Rect::new(20, 20, 30, 30);
    assert_eq!(interior.right_pad(5, (100, 100)), Rect::new(20, 20, 35, 30));

    // y2 must be untouched even when it would exceed the canvas height
    let tall = Rect::new(0, 0, 10, 100);
    assert_eq!(tall.right_pad(5, (100, 100)), Rect::new(0, 0, 15, 100));
}

#[test]
// spec §9.3 step 5: the initial padding tier applies pad() then right_pad(), and
// the composition must clamp at each step, not once at the end
fn padding_tiers_compose() {
    // upstream defaults: box_padding_initial = 2, box_right_padding_initial = 3
    let r = Rect::new(1, 1, 96, 50);
    let padded = r.pad(2, (100, 100)).right_pad(3, (100, 100));
    // pad: x1 max(1-2,0)=0, y1 0, x2 min(98,100)=98, y2 52 ; right_pad: x2 min(101,100)=100
    assert_eq!(padded, Rect::new(0, 0, 100, 52));
}

// ---------------------------------------------------------------- scale

#[test]
// spec §2.1: scale() truncates toward zero per coordinate (Python int()), it does
// NOT round — 4.5 becomes 4
fn scale_truncates_toward_zero() {
    let r = Rect::new(1, 1, 3, 3);
    // 1*1.5 = 1.5 -> 1 ; 3*1.5 = 4.5 -> 4 (NOT 5)
    assert_eq!(r.scale(1.5), Rect::new(1, 1, 4, 4));
}

#[test]
// spec §2.1: truncation is toward ZERO, not floor — int(-4.5) == -4 in Python,
// so a negative coordinate must round up, which floor() would get wrong (-5)
fn scale_truncation_is_toward_zero_for_negatives() {
    let r = Rect::new(-3, -3, 3, 3);
    assert_eq!(r.scale(1.5), Rect::new(-4, -4, 4, 4));
}

#[test]
// spec §9.7(A)9: with page scale 0.5, a box at (10,10,30,30) in base coordinates
// is recorded in ORIGINAL coordinates as (20,20,60,60) via scale(1.0/0.5)
fn scale_maps_base_coords_to_original_coords() {
    let r = Rect::new(10, 10, 30, 30);
    assert_eq!(r.scale(1.0 / 0.5), Rect::new(20, 20, 60, 60));
}

#[test]
// spec §2.1: scale(1.0) is identity (the scale == 1.0 no-resize path must not
// perturb coordinates through float round-tripping)
fn scale_one_is_identity() {
    let r = Rect::new(7, 13, 199, 401);
    assert_eq!(r.scale(1.0), r);
}

#[test]
// spec §2.1: scaling down truncates too — 0.5 on odd coordinates loses the half
fn scale_down_truncates() {
    let r = Rect::new(3, 5, 9, 11);
    // 1.5->1, 2.5->2, 4.5->4, 5.5->5
    assert_eq!(r.scale(0.5), Rect::new(1, 2, 4, 5));
}

// ---------------------------------------------------------------- translate

#[test]
// spec §2.1: translate shifts all four coordinates, with no clamping
fn translate_shifts_without_clamping() {
    let r = Rect::new(10, 10, 20, 20);
    assert_eq!(r.translate(5, -5), Rect::new(15, 5, 25, 15));
    assert_eq!(r.translate(-100, 0), Rect::new(-90, 10, -80, 20));
    assert_eq!(r.translate(0, 0), r);
}

// ---------------------------------------------------------------- to_crop

#[test]
// spec §2.1: to_crop returns a canvas-clamped (x, y, w, h) for the `image` crate,
// with x2/y2 treated as EXCLUSIVE (the cropping convention)
fn to_crop_clamps_to_canvas() {
    // fully interior: w/h come straight from the exclusive coordinates
    assert_eq!(Rect::new(10, 20, 30, 50).to_crop((100, 100)), Some((10, 20, 20, 30)));
    // negative origin clamps up to 0 and shrinks the extent
    assert_eq!(Rect::new(-5, -5, 5, 5).to_crop((100, 100)), Some((0, 0, 5, 5)));
    // far edge clamps down to the canvas
    assert_eq!(Rect::new(90, 90, 110, 110).to_crop((100, 100)), Some((90, 90, 10, 10)));
    // exactly flush is a full-canvas crop, not an error
    assert_eq!(Rect::new(0, 0, 100, 100).to_crop((100, 100)), Some((0, 0, 100, 100)));
}

#[test]
// spec §2.1: an empty or fully out-of-bounds rect has no crop
fn to_crop_returns_none_when_degenerate() {
    assert_eq!(Rect::new(200, 200, 210, 210).to_crop((100, 100)), None);
    assert_eq!(Rect::new(0, 0, 0, 10).to_crop((100, 100)), None);
    assert_eq!(Rect::new(0, 0, 10, 0).to_crop((100, 100)), None);
    assert_eq!(Rect::new(-20, 0, -10, 10).to_crop((100, 100)), None);
    assert_eq!(Rect::new(10, 0, 5, 10).to_crop((100, 100)), None); // inverted
}

// ---------------------------------------------------------------- serde

#[test]
// spec §2: JSON is snake_case; Rect's four fields serialize as a flat object in
// declaration order (this is a persisted format, so the exact shape is frozen)
fn rect_json_shape_is_stable() {
    let r = Rect::new(1, 2, 3, 4);
    let json = serde_json::to_string(&r).expect("serialize");
    assert_eq!(json, r#"{"x1":1,"y1":2,"x2":3,"y2":4}"#);
    let back: Rect = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, r);
}

#[test]
// spec §2: negative coordinates survive a round-trip (analytics rects scaled into
// original coordinates can legitimately be negative)
fn rect_json_round_trips_negatives() {
    let r = Rect::new(-1, -2, 3, 4);
    let back: Rect = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    assert_eq!(back, r);
}
