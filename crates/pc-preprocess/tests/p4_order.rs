//! Task P4 — spec §9.3 step 6, §9.7(A)5, §16.8 item 14. FROZEN.

mod common;

use common::text_box;
use pc_config::ReadingOrder;
use pc_core::{Language, Rect};
use pc_preprocess::{is_right_to_left, sort_key, sort_reading_order};

fn at(x: i32, y: i32) -> Rect {
    Rect::new(x, y, x + 50, y + 50)
}

fn sorted(
    order: ReadingOrder,
    page_language: Option<Language>,
    origins: &[(i32, i32)],
) -> Vec<(i32, i32)> {
    let mut boxes = origins
        .iter()
        .map(|(x, y)| text_box(at(*x, *y), page_language))
        .collect::<Vec<_>>();
    sort_reading_order(&mut boxes, order, page_language);
    boxes
        .iter()
        .map(|text_box| (text_box.rect.x1, text_box.rect.y1))
        .collect()
}

// ------------------------------------------------------------ direction resolution

#[test]
fn a5_direction_resolution_covers_every_reading_order_and_page_language() {
    // spec §9.3 step 6: RTL unless Comic, or Auto with a page language outside
    // RTL_BOX_ORDER_LANGUAGES. An unknown page language is not in that set.
    assert!(is_right_to_left(ReadingOrder::Manga, None));
    assert!(is_right_to_left(
        ReadingOrder::Manga,
        Some(Language::English)
    ));

    assert!(!is_right_to_left(ReadingOrder::Comic, None));
    assert!(!is_right_to_left(
        ReadingOrder::Comic,
        Some(Language::Japanese)
    ));

    assert!(is_right_to_left(
        ReadingOrder::Auto,
        Some(Language::Japanese)
    ));
    assert!(!is_right_to_left(
        ReadingOrder::Auto,
        Some(Language::English)
    ));
    assert!(
        !is_right_to_left(ReadingOrder::Auto, None),
        "an unknown page language must read left-to-right under Auto"
    );
}

// ------------------------------------------------------------ the sort (§9.7(A)5)

#[test]
fn a5_manga_and_auto_japanese_read_right_to_left() {
    // spec §9.7(A)5: boxes at (0,0), (500,0), (0,500) => (500,0), (0,0), (0,500).
    let origins = [(0, 0), (500, 0), (0, 500)];
    let expected = vec![(500, 0), (0, 0), (0, 500)];

    assert_eq!(sorted(ReadingOrder::Manga, None, &origins), expected);
    assert_eq!(
        sorted(ReadingOrder::Auto, Some(Language::Japanese), &origins),
        expected
    );
}

#[test]
fn a5_comic_and_auto_english_read_left_to_right() {
    // spec §9.7(A)5: the same boxes => (0,0), (500,0), (0,500).
    let origins = [(0, 0), (500, 0), (0, 500)];
    let expected = vec![(0, 0), (500, 0), (0, 500)];

    assert_eq!(sorted(ReadingOrder::Comic, None, &origins), expected);
    assert_eq!(
        sorted(ReadingOrder::Auto, Some(Language::English), &origins),
        expected
    );
}

#[test]
fn a5_equal_keys_retain_detector_order() {
    // spec §9.3 step 6: the sort must be STABLE. Same (x1, y1), different extents =>
    // byte-identical keys, so only stability can decide the order.
    let mut boxes = vec![
        text_box(Rect::new(100, 100, 150, 150), Some(Language::Japanese)),
        text_box(Rect::new(100, 100, 160, 160), Some(Language::English)),
        text_box(Rect::new(100, 100, 170, 170), None),
    ];
    let before = boxes.clone();

    sort_reading_order(&mut boxes, ReadingOrder::Manga, Some(Language::Japanese));
    assert_eq!(boxes, before, "manga sort must be stable on equal keys");

    sort_reading_order(&mut boxes, ReadingOrder::Comic, Some(Language::English));
    assert_eq!(boxes, before, "comic sort must be stable on equal keys");
}

#[test]
fn sort_key_uses_the_specified_factors() {
    // spec §9.3 step 6: `x_factor * x1 + 1.0 * y1`, with |x_factor| == 0.4.
    let rect = Rect::new(500, 100, 550, 150);

    pc_testkit::assert_close(sort_key(rect, true), -0.4 * 500.0 + 100.0, 1e-12);
    pc_testkit::assert_close(sort_key(rect, false), 0.4 * 500.0 + 100.0, 1e-12);
}

#[test]
fn sorting_reads_row_by_row_when_the_vertical_offset_dominates() {
    // The y_factor of 1.0 against |x_factor| of 0.4 means a row difference of 500px
    // always beats any column difference on a 1000px-wide page. Locks the relative
    // weighting, not just the sign.
    let origins = [(0, 500), (900, 0), (0, 0)];

    assert_eq!(
        sorted(ReadingOrder::Comic, None, &origins),
        vec![(0, 0), (900, 0), (0, 500)]
    );
    assert_eq!(
        sorted(ReadingOrder::Manga, None, &origins),
        vec![(900, 0), (0, 0), (0, 500)]
    );
}

#[test]
fn sorting_an_empty_or_single_box_page_is_a_no_op() {
    let mut empty: Vec<pc_core::TextBox> = Vec::new();
    sort_reading_order(&mut empty, ReadingOrder::Auto, None);
    assert!(empty.is_empty());

    let mut single = vec![text_box(at(7, 9), None)];
    let before = single.clone();
    sort_reading_order(&mut single, ReadingOrder::Auto, None);
    assert_eq!(single, before);
}
