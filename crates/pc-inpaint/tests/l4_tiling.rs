//! L4 — spec §16.38 items 5(b)–(e): the merged cover, the 512-lattice, the emptiness drop, and pixel
//! ownership. `DEVIATION(24)`'s geometry lives here.

mod common;

use pc_core::Rect;
use pc_imageops::BinaryMask;
use pc_inpaint::{merge_rects, owner_map, tile_cover, tile_windows, TileWindow, UNOWNED};

fn windows(cover: &[TileWindow]) -> Vec<Rect> {
    cover.iter().map(|tile| tile.window).collect()
}

fn mask_with_pixels(size: (u32, u32), pixels: &[(u32, u32)]) -> BinaryMask {
    let mut mask = BinaryMask::new(size.0, size.1);
    for (x, y) in pixels {
        mask.set(*x, *y, true);
    }
    mask
}

fn mask_with_band(
    size: (u32, u32),
    x: std::ops::Range<u32>,
    y: std::ops::Range<u32>,
) -> BinaryMask {
    let mut mask = BinaryMask::new(size.0, size.1);
    for row in y {
        for column in x.clone() {
            mask.set(column, row, true);
        }
    }
    mask
}

/// §16.38 item 5(b): rectangles are closed transitively under intersection, each intersecting group
/// replaced by its bounding union. A chain `A∩B`, `B∩C` with `A∩C` empty collapses to one rectangle;
/// a disjoint fourth stays separate.
#[test]
fn intersecting_rectangles_collapse_transitively_into_their_bounding_union() {
    let merged = merge_rects(&[
        Rect::new(0, 0, 10, 10),
        Rect::new(5, 5, 15, 15),
        Rect::new(12, 12, 20, 20),
        Rect::new(60, 60, 70, 70),
    ]);

    assert_eq!(
        merged,
        vec![Rect::new(0, 0, 20, 20), Rect::new(60, 60, 70, 70)],
        "A-B-C chain into one union, plus one untouched rectangle, sorted by (y1, x1)"
    );
}

/// §16.38 item 5(b)'s **corrected** order-independence argument, made falsifiable.
///
/// The clause is careful about this: "replacing a group by its bounding union can create
/// intersections that no pair in the original relation had, so the closure is taken over a relation
/// the procedure itself edits", and the conclusion survives only because the merge is monotone and
/// the procedure runs to a **least fixpoint**.
///
/// The geometry here is exactly that case. `A = (0,0,18,18)` is the bounding union of the two
/// diagonal rectangles, and `C` sits inside that union but intersects **neither** of them: `C.x1 =
/// 12` is past `A`'s right edge and `C.y2 = 5` is above `B`'s top edge. A single pairwise pass
/// therefore leaves two rectangles; only running to a fixpoint leaves one.
#[test]
fn a_rectangle_that_only_intersects_after_a_merge_is_still_absorbed() {
    let a = Rect::new(0, 0, 10, 10);
    let b = Rect::new(8, 8, 18, 18);
    let c = Rect::new(12, 0, 17, 5);

    // The premise, asserted so the test cannot rot into a different case: C meets neither input.
    let meets = |p: &Rect, q: &Rect| p.x1 < q.x2 && q.x1 < p.x2 && p.y1 < q.y2 && q.y1 < p.y2;
    assert!(meets(&a, &b), "A and B must intersect");
    assert!(!meets(&a, &c), "C must NOT intersect A");
    assert!(!meets(&b, &c), "C must NOT intersect B");
    assert!(
        meets(&a.merge(&b), &c),
        "C must intersect the union of A and B — that is the whole point"
    );

    let merged = merge_rects(&[a, b, c]);

    assert_eq!(merged.len(), 1, "a single pairwise pass would leave 2");
    assert_eq!(merged, vec![Rect::new(0, 0, 18, 18)]);
}

/// §16.38 item 5(b): "`x2`/`y2` are exclusive" (§16.9 item 1), so two rectangles that merely touch
/// along an edge do not intersect and are not merged. Without this, an entire page of adjacent boxes
/// would collapse into one rectangle and the lattice branch would fire where the centred branch
/// should.
#[test]
fn rectangles_that_only_touch_along_an_edge_are_not_merged() {
    let merged = merge_rects(&[Rect::new(0, 0, 10, 10), Rect::new(10, 0, 20, 10)]);
    assert_eq!(
        merged,
        vec![Rect::new(0, 0, 10, 10), Rect::new(10, 0, 20, 10)]
    );
}

/// §16.38 item 5(b)'s explicit obligation: *"L4 must still pin it empirically: a test that permutes
/// `#mask_data.json` region order and asserts an identical cover."*
///
/// All 24 permutations of four padded boxes, two of which intersect. The cover is asserted equal to a
/// hard-coded literal — not merely equal across permutations, which would be satisfied by a cover
/// that is consistently wrong (or consistently empty).
#[test]
fn permuting_the_region_order_yields_an_identical_tile_cover() {
    let canvas = (1600_u32, 1600_u32);
    let rects = [
        Rect::new(100, 100, 150, 150),
        Rect::new(140, 140, 190, 190), // intersects the first
        Rect::new(800, 800, 850, 850),
        Rect::new(1400, 1400, 1450, 1450),
    ];
    let fill = mask_with_pixels(canvas, &[(120, 120), (160, 160), (820, 820), (1420, 1420)]);

    let expected = vec![
        Rect::new(0, 0, 512, 512), // merged (100,100,190,190), centred then clamped
        Rect::new(569, 569, 1081, 1081), // merged (800,800,850,850), centred, in frame
        Rect::new(1088, 1088, 1600, 1600), // merged (1400,..), centred then clamped inward
    ];

    let mut permutations = 0_usize;
    for order in permutations_of_four() {
        let permuted: Vec<Rect> = order.iter().map(|index| rects[*index]).collect();
        let cover = tile_cover(&merge_rects(&permuted), &fill, canvas);
        assert_eq!(
            windows(&cover),
            expected,
            "region order {order:?} produced a different cover"
        );
        permutations += 1;
    }
    assert_eq!(permutations, 24, "all 4! orders must be exercised");
}

fn permutations_of_four() -> Vec<[usize; 4]> {
    let mut out = Vec::new();
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let order = [a, b, c, d];
                    let mut seen = [false; 4];
                    if order
                        .iter()
                        .all(|index| !std::mem::replace(&mut seen[*index], true))
                    {
                        out.push(order);
                    }
                }
            }
        }
    }
    out
}

/// §16.38 item 5(c): "A merged rectangle whose width and height are both `<= 512` gets one 512×512
/// window centred on it."
///
/// Hand-derived: a 100×80 rectangle at `(300,300)` on a 1000×1000 page pads by `(512-100)/2 = 206`
/// on the x axis and `(512-80)/2 = 216` on the y axis, giving `(94, 84)`.
#[test]
fn a_rectangle_that_fits_gets_one_window_centred_on_it() {
    let cover = tile_windows(&[Rect::new(300, 300, 400, 380)], (1000, 1000));
    assert_eq!(windows(&cover), vec![Rect::new(94, 84, 606, 596)]);
    assert_eq!(cover[0].merged_index, 0);
}

/// §16.38 item 5(c): "and then translated minimally to lie inside the frame". Both ends, on one page
/// each, with hard-coded anchors: the top-left case clamps to 0, and the bottom-right case clamps to
/// `canvas - 512 = 88`.
#[test]
fn a_centred_window_is_translated_minimally_to_stay_in_frame() {
    assert_eq!(
        windows(&tile_windows(&[Rect::new(10, 10, 110, 90)], (1000, 1000))),
        vec![Rect::new(0, 0, 512, 512)],
        "the centred anchor would be (-196, -206); the minimal translation is to (0, 0)"
    );
    assert_eq!(
        windows(&tile_windows(&[Rect::new(500, 500, 560, 560)], (600, 600))),
        vec![Rect::new(88, 88, 600, 600)],
        "the centred anchor would be 274; the frame allows at most 600 - 512 = 88"
    );
}

/// §16.38 item 5(c): "A merged rectangle exceeding 512 on either axis is covered by a stride-512
/// lattice anchored at its own top-left corner."
///
/// Note what the clause does **not** say: it does not switch per axis. A 1000×100 rectangle takes the
/// lattice branch on *both* axes, so its single `y` anchor is its own `y1 = 0` — not the centred
/// `0 - (512-100)/2`. If the implementation centred the short axis this test would read
/// `(0, -206, …)` clamped to `(0, 0, …)` and pass by accident, which is why the rectangle below is
/// placed at `y1 = 60`: the lattice anchor is `60` and a centred anchor would be `60 - 206 = -146 ->
/// 0`. The two answers differ.
#[test]
fn a_rectangle_wider_than_512_takes_a_stride_512_lattice_anchored_at_its_own_top_left() {
    let cover = tile_windows(&[Rect::new(0, 60, 1000, 160)], (1200, 800));
    assert_eq!(
        windows(&cover),
        vec![
            Rect::new(0, 60, 512, 572),
            Rect::new(512, 60, 1024, 572),
        ],
        "x anchors 0 and 512; the single y anchor is the rectangle's own y1 = 60, not a centred one"
    );
}

/// §16.38 item 5(c): "with the final row and column translated inward to stay in frame".
///
/// Same rectangle as above on a page exactly 1000 wide: the second x anchor `512` would put the
/// window's right edge at `1024`, so it translates inward to `1000 - 512 = 488`.
#[test]
fn the_final_lattice_column_is_translated_inward_to_stay_in_frame() {
    let cover = tile_windows(&[Rect::new(0, 0, 1000, 100)], (1000, 600));
    assert_eq!(
        windows(&cover),
        vec![Rect::new(0, 0, 512, 512), Rect::new(488, 0, 1000, 512)]
    );
}

/// §16.38 item 5(c)'s last clause: "windows whose intersection with the fill mask is empty are
/// dropped".
///
/// Asserted as a **pair on identical geometry**, so neither half can pass by accident: with fill only
/// on the left, the cover is the *first* window alone (identity, not "one window"); adding a fill
/// pixel on the right brings the second window back.
#[test]
fn a_window_with_no_fill_pixel_is_dropped_and_returns_when_a_fill_pixel_appears() {
    let canvas = (1000_u32, 600_u32);
    let merged = merge_rects(&[Rect::new(0, 0, 1000, 100)]);

    let left_only = mask_with_pixels(canvas, &[(50, 50)]);
    assert_eq!(
        windows(&tile_cover(&merged, &left_only, canvas)),
        vec![Rect::new(0, 0, 512, 512)],
        "the second window (488..1000) holds no fill pixel and is dropped"
    );

    let both = mask_with_pixels(canvas, &[(50, 50), (900, 50)]);
    assert_eq!(
        windows(&tile_cover(&merged, &both, canvas)),
        vec![Rect::new(0, 0, 512, 512), Rect::new(488, 0, 1000, 512)],
        "a fill pixel at x = 900 is only reachable from the second window"
    );
}

/// §16.38 item 5(e): "Every fill pixel is written exactly once … L4 must pin 'exactly once' as an
/// assertion over the whole page, not as a comment."
///
/// The page is a 1000×600 canvas with a 1000-wide, 20-tall fill band — wider than one tile, so the
/// lattice's two windows overlap in `x ∈ [488, 512)`. Every number below is hand-derived from the two
/// window anchors and is a literal:
///
///   * total fill pixels `= 1000 × 20 = 20000`;
///   * window 0 spans `x ∈ [0, 512)` and, being first in reading order, owns `512 × 20 = 10240`;
///   * window 1 spans `x ∈ [488, 1000)` and owns only what is left, `488 × 20 = 9760`.
///
/// An off-by-one in the lattice, a reversed ownership order, or a cover with a hole all move at least
/// one of those three numbers.
#[test]
fn every_fill_pixel_has_exactly_one_owning_window_and_the_split_is_the_hand_derived_one() {
    let canvas = (1000_u32, 600_u32);
    let fill = mask_with_band(canvas, 0..1000, 40..60);
    let cover = tile_cover(&merge_rects(&[Rect::new(0, 0, 1000, 100)]), &fill, canvas);
    assert_eq!(windows(&cover).len(), 2);

    let owners = owner_map(&cover, &fill);

    let mut unowned_fill = 0_usize;
    let mut owned_by = [0_usize; 2];
    let mut owned_non_fill = 0_usize;
    for y in 0..canvas.1 {
        for x in 0..canvas.0 {
            let owner = owners[(y as usize) * (canvas.0 as usize) + (x as usize)];
            match (fill.get(x, y), owner) {
                (true, UNOWNED) => unowned_fill += 1,
                (true, index) => owned_by[index as usize] += 1,
                (false, UNOWNED) => {}
                (false, _) => owned_non_fill += 1,
            }
        }
    }

    assert_eq!(fill.count_set(), 20000, "1000 x 20 fill band");
    assert_eq!(
        unowned_fill, 0,
        "a hole in the cover leaves fill pixels unwritten"
    );
    assert_eq!(
        owned_non_fill, 0,
        "ownership is only ever assigned to fill pixels"
    );
    assert_eq!(owned_by[0], 10240, "window 0 covers x in [0, 512)");
    assert_eq!(owned_by[1], 9760, "window 1 gets only x in [512, 1000)");
    assert_eq!(owned_by[0] + owned_by[1], 20000);
}

/// §16.38 item 5(e): "a fill pixel belongs to the **first** window in that order". Asserted on a
/// pixel that provably lies in both windows — the premise is checked first, so the test cannot pass
/// because the overlap vanished.
#[test]
fn a_pixel_in_two_windows_is_owned_by_the_earlier_one_in_reading_order() {
    let canvas = (1000_u32, 600_u32);
    let fill = mask_with_band(canvas, 0..1000, 40..60);
    let cover = tile_cover(&merge_rects(&[Rect::new(0, 0, 1000, 100)]), &fill, canvas);

    let contested = (500_u32, 50_u32);
    let inside = |window: &Rect, p: (u32, u32)| {
        (p.0 as i32) >= window.x1
            && (p.0 as i32) < window.x2
            && (p.1 as i32) >= window.y1
            && (p.1 as i32) < window.y2
    };
    assert!(
        inside(&cover[0].window, contested),
        "premise: window 0 covers it"
    );
    assert!(
        inside(&cover[1].window, contested),
        "premise: window 1 covers it too"
    );

    let owners = owner_map(&cover, &fill);
    assert_eq!(
        owners[(contested.1 as usize) * (canvas.0 as usize) + (contested.0 as usize)],
        0,
        "the earlier window in reading order owns the overlap"
    );
}

/// §16.38 item 5(f) makes a quantitative claim about when the centred window removes the seam risk:
/// "a box of **≤ 462 px** on an axis always fits (512 − 2×25), a box of **≥ 489 px** never fits (512
/// − 2×12 = 488)". Those thresholds are properties of the *padded* extent, so what is checkable here
/// is the branch boundary itself: a merged rectangle of exactly 512 still takes the single centred
/// window, and 513 takes the lattice.
#[test]
fn the_branch_boundary_between_one_window_and_a_lattice_is_at_exactly_512() {
    let fits = tile_windows(&[Rect::new(100, 100, 612, 612)], (2000, 2000));
    assert_eq!(fits.len(), 1, "512 x 512 exactly still fits in one window");
    assert_eq!(windows(&fits), vec![Rect::new(100, 100, 612, 612)]);

    let lattice = tile_windows(&[Rect::new(100, 100, 613, 612)], (2000, 2000));
    assert_eq!(
        windows(&lattice),
        vec![
            Rect::new(100, 100, 612, 612),
            Rect::new(612, 100, 1124, 612)
        ],
        "513 wide takes the lattice branch on both axes"
    );
}

/// A canvas exactly `TILE` wide collapses every lattice anchor onto 0. A duplicated window would be a
/// second inference call producing the same tile and owning nothing, so the anchors are de-duplicated.
#[test]
fn clamped_lattice_anchors_that_collapse_onto_each_other_are_not_duplicated() {
    let cover = tile_windows(&[Rect::new(0, 0, 512, 600)], (512, 512));
    assert_eq!(windows(&cover), vec![Rect::new(0, 0, 512, 512)]);
}
