//! Task P4 — spec §9.3 step 6: the reading-order sort key and the auto/manga/comic
//! resolution.

use pc_config::ReadingOrder;
use pc_core::{Language, Rect, TextBox, RTL_BOX_ORDER_LANGUAGES};

/// spec §9.3 step 6: `|x_factor| == 0.4`, negative for right-to-left pages.
pub const READING_ORDER_X_MAGNITUDE: f64 = 0.4;
/// spec §9.3 step 6.
pub const READING_ORDER_Y_FACTOR: f64 = 1.0;

/// spec §9.3 step 6: right-to-left unless `reading_order == Comic`, or `Auto` with a
/// page language outside [`RTL_BOX_ORDER_LANGUAGES`]. An unknown page language (`None`)
/// is not in that set, so `Auto` + unknown reads left-to-right.
pub fn is_right_to_left(order: ReadingOrder, page_language: Option<Language>) -> bool {
    match order {
        ReadingOrder::Manga => true,
        ReadingOrder::Comic => false,
        ReadingOrder::Auto => {
            page_language.is_some_and(|language| RTL_BOX_ORDER_LANGUAGES.contains(&language))
        }
    }
}

/// spec §9.3 step 6: `x_factor * x1 + y_factor * y1`.
pub fn sort_key(rect: Rect, right_to_left: bool) -> f64 {
    let x_factor = if right_to_left {
        -READING_ORDER_X_MAGNITUDE
    } else {
        READING_ORDER_X_MAGNITUDE
    };
    x_factor * f64::from(rect.x1) + READING_ORDER_Y_FACTOR * f64::from(rect.y1)
}

/// spec §9.3 step 6: ascending by [`sort_key`], **stably**, so equal keys keep detector
/// order. `total_cmp` rather than `partial_cmp().unwrap()` per spec §16.8 item 14.
pub fn sort_reading_order(
    boxes: &mut [TextBox],
    order: ReadingOrder,
    page_language: Option<Language>,
) {
    let right_to_left = is_right_to_left(order, page_language);
    boxes.sort_by(|a, b| {
        sort_key(a.rect, right_to_left).total_cmp(&sort_key(b.rect, right_to_left))
    });
}
