//! Task P3 — spec §9.3 steps 4 and 9, §9.7(A)2–4, §14.2, §14.3, §16.8 item 2. FROZEN.

mod common;

use common::text_box;
use pc_core::{Language, Rect};
use pc_preprocess::{resolve_overlaps, resolve_total_overlaps};

// ------------------------------------------------------------ step 4 (§9.7(A)3)

#[test]
fn a3_mutually_centered_boxes_merge_and_the_earliest_language_wins() {
    // spec §9.7(A)3: A and B contain each other's centers, C is disjoint => 2 boxes,
    // A∪B first (input order preserved), merged language == A's.
    // §14.3: upstream would desynchronize `boxes` from `box_language` here; our
    // `TextBox` pairing makes the earliest box's language the specified winner.
    let a = Rect::new(0, 0, 100, 100); // center (50, 50)
    let b = Rect::new(40, 40, 140, 140); // center (90, 90), inside A
    let c = Rect::new(500, 500, 600, 600);
    assert!(a.overlaps_center(&b), "fixture must be mutually centered");
    assert!(!a.overlaps_center(&c), "fixture C must be disjoint");

    let resolved = resolve_total_overlaps(vec![
        text_box(a, Some(Language::Japanese)),
        text_box(b, Some(Language::English)),
        text_box(c, None),
    ]);

    assert_eq!(common::rects(&resolved), vec![Rect::new(0, 0, 140, 140), c]);
    assert_eq!(
        common::languages(&resolved),
        vec![Some(Language::Japanese), None],
        "the popped (earliest) box's language must win the merge"
    );
}

#[test]
fn a3_non_overlapping_boxes_pass_through_unchanged_and_in_order() {
    let boxes = vec![
        text_box(Rect::new(0, 0, 10, 10), Some(Language::Japanese)),
        text_box(Rect::new(500, 0, 510, 10), Some(Language::English)),
        text_box(Rect::new(0, 500, 10, 510), None),
    ];

    let resolved = resolve_total_overlaps(boxes.clone());

    assert_eq!(resolved, boxes);
}

#[test]
fn total_overlap_resolution_of_an_empty_page_is_empty() {
    assert!(resolve_total_overlaps(Vec::new()).is_empty());
}

#[test]
fn total_overlap_resolution_absorbs_several_partners_into_one_box() {
    // Three boxes all mutually centered on the first: one output, bounding all of them.
    let a = Rect::new(0, 0, 100, 100);
    let b = Rect::new(20, 20, 120, 120);
    let c = Rect::new(10, 10, 110, 160);

    let resolved = resolve_total_overlaps(vec![
        text_box(a, Some(Language::Japanese)),
        text_box(b, None),
        text_box(c, None),
    ]);

    assert_eq!(common::rects(&resolved), vec![Rect::new(0, 0, 120, 160)]);
    assert_eq!(resolved[0].language, Some(Language::Japanese));
}

// ------------------------------------------------------------ step 9 (§9.7(A)2, §9.7(A)4)

#[test]
fn a2_overlap_threshold_is_strictly_greater() {
    // spec §9.7(A)2: a ratio of exactly 0.20 against threshold 20.0 does NOT merge;
    // 0.2001 does. Both fixtures use a 100x100 pair so `min(area) == 10_000` and the
    // ratio is exact in f64.
    let a = Rect::new(0, 0, 100, 100);

    // intersection 100 x 20 = 2_000 => 2_000 / 10_000 == 0.2000 exactly.
    let exactly_twenty_percent = Rect::new(0, 80, 100, 180);
    assert_eq!(
        resolve_overlaps(vec![a, exactly_twenty_percent], 20.0),
        vec![a, exactly_twenty_percent],
        "exactly 20% must not merge (strict `>`)"
    );

    // intersection 69 x 29 = 2_001 => 2_001 / 10_000 == 0.2001.
    let just_over = Rect::new(31, 71, 131, 171);
    assert_eq!(
        resolve_overlaps(vec![a, just_over], 20.0),
        vec![Rect::new(0, 0, 131, 171)],
        "0.2001 must merge"
    );
}

#[test]
fn a2_a_zero_area_box_does_not_panic() {
    // spec §9.7(A)2: `Rect::overlaps` forces the divisor to 1 when the smaller area is
    // 0, so a degenerate box is simply never a merge partner.
    let degenerate = Rect::new(10, 10, 10, 10);
    let normal = Rect::new(0, 0, 100, 100);

    assert_eq!(
        resolve_overlaps(vec![degenerate, normal], 20.0),
        vec![degenerate, normal]
    );
    assert_eq!(
        resolve_overlaps(vec![normal, degenerate], 20.0),
        vec![normal, degenerate]
    );
    assert_eq!(
        resolve_overlaps(vec![degenerate, degenerate], 0.0),
        vec![degenerate, degenerate]
    );
}

#[test]
fn a4_overlap_resolution_is_a_single_pass_with_no_transitive_chaining() {
    // spec §9.7(A)4: A~B and B~C but not A~C => exactly [A∪B, C], NOT [A∪B∪C].
    // §16.8 item 2: this only holds under snapshot partner selection — C is tested
    // against A as popped, never against the widened A∪B (which WOULD overlap C by the
    // same 10%).
    let a = Rect::new(0, 0, 100, 100);
    let b = Rect::new(90, 0, 190, 100);
    let c = Rect::new(180, 0, 280, 100);

    // 10% overlap between neighbours, none between the ends.
    assert!(a.overlaps(&b, 5.0));
    assert!(b.overlaps(&c, 5.0));
    assert!(!a.overlaps(&c, 5.0));
    let merged_ab = a.merge(&b);
    assert!(
        merged_ab.overlaps(&c, 5.0),
        "the fixture is only meaningful if the merged box WOULD have absorbed C"
    );

    assert_eq!(resolve_overlaps(vec![a, b, c], 5.0), vec![merged_ab, c]);
}

#[test]
fn a4_the_partner_set_is_snapshotted_before_any_merge_widens_the_box() {
    // The mirror of the above stated directly: a box that only overlaps the *merged*
    // result must survive on its own, in its original input position.
    let a = Rect::new(0, 0, 100, 100);
    let b = Rect::new(50, 0, 150, 100); // 50% with A
    let c = Rect::new(120, 0, 220, 100); // 30% with B, 0% with A

    assert!(!a.overlaps(&c, 20.0));

    let resolved = resolve_overlaps(vec![a, b, c], 20.0);

    assert_eq!(resolved, vec![Rect::new(0, 0, 150, 100), c]);
}

#[test]
fn overlap_resolution_preserves_index_order_of_the_survivors() {
    // §14.2 / §5.7: index order (FIFO), not the nondeterministic set iteration upstream
    // uses. Running the same input repeatedly must give the same answer, and the
    // survivors must come out in the order they went in.
    let rects = vec![
        Rect::new(600, 600, 700, 700),
        Rect::new(0, 0, 100, 100),
        Rect::new(300, 300, 400, 400),
    ];

    let resolved = resolve_overlaps(rects.clone(), 20.0);

    assert_eq!(resolved, rects);
    for _ in 0..10 {
        assert_eq!(resolve_overlaps(rects.clone(), 20.0), resolved);
    }
}

#[test]
fn overlap_resolution_of_an_empty_list_is_empty() {
    assert!(resolve_overlaps(Vec::new(), 20.0).is_empty());
}
