//! L4 — spec §16.38 item 3(e): the growth arithmetic and the padded box.

mod common;

use common::radii;
use pc_config::InpainterConfig;
use pc_core::Rect;
use pc_inpaint::{growth, padded_box};

/// §16.38 item 3(e) / upstream `inpainting.py:106-108`:
/// `growth = min(min_inpainting_radius + int(deviation * inpainting_radius_multiplier),
/// max_inpainting_radius)`.
///
/// The table is hand-computed at the ratified defaults (`min = 7`, `max = 20`, `multiplier = 0.2`),
/// column by column, and every expected value is a literal. Two rows carry specific traps:
///
///   * `9.9 -> 8`: `9.9 * 0.2 = 1.98`, and `int()` truncates toward zero. Rounding would give `9`,
///     so this row alone falsifies a `.round()`.
///   * `1000.0 -> 20`: the cap is applied **after** the addition, so a large deviation saturates.
#[test]
fn growth_is_the_hand_computed_table_at_the_ratified_defaults() {
    let config = InpainterConfig::default();

    let table: &[(f64, u32)] = &[
        (0.0, 7),     // 7 + int(0.0)   = 7
        (4.9, 7),     // 7 + int(0.98)  = 7
        (5.0, 8),     // 7 + int(1.0)   = 8
        (9.9, 8),     // 7 + int(1.98)  = 8   <- rounding would say 9
        (14.999, 9),  // 7 + int(2.9998) = 9
        (15.0, 10),   // 7 + int(3.0)   = 10
        (65.0, 20),   // 7 + int(13.0)  = 20  (exactly the cap)
        (66.0, 20),   // 7 + int(13.2)  = 20  (exactly the cap)
        (70.0, 20),   // 7 + int(14.0)  = 21  -> capped to 20
        (1000.0, 20), // 7 + int(200.0) = 207 -> capped to 20
    ];

    for (deviation, expected) in table {
        assert_eq!(
            growth(*deviation, &config),
            *expected,
            "growth({deviation}) at the ratified defaults"
        );
    }
}

/// §16.38 item 3(e): the cap can never drag the result below `min_inpainting_radius` on a profile
/// §16.38 item 13(b) validated, because that clause enforces upstream's own `config.py:932`
/// invariant `max_inpainting_radius >= min_inpainting_radius`. This pins the boundary case where
/// the two are equal — the tightest configuration the validator admits.
#[test]
fn a_zero_slack_radius_range_pins_growth_to_that_single_value() {
    let config = radii(12, 12, 5.0, 0, 0);
    for deviation in [0.0, 1.0, 50.0, 1e9] {
        assert_eq!(growth(deviation, &config), 12);
    }
}

/// §16.38 item 3(e) / upstream `:111` + `:113` via `structures.py:98-110`: the box is padded by
/// `growth + inpainting_isolation_radius` and **clamped to the canvas**.
///
/// Three rows: interior (no clamping), against the top-left corner (both low ends clamp), and
/// against the bottom-right corner (both high ends clamp). Every expected rect is a literal.
#[test]
fn the_padded_box_adds_growth_plus_the_isolation_radius_and_clamps_to_the_canvas() {
    let config = radii(0, 100, 0.0, 5, 0);
    let canvas = (1000, 1000);

    assert_eq!(
        padded_box(Rect::new(100, 200, 150, 260), 8, &config, canvas),
        Rect::new(87, 187, 163, 273),
        "pad = growth 8 + isolation 5 = 13, applied on all four sides"
    );
    assert_eq!(
        padded_box(Rect::new(5, 3, 50, 40), 8, &config, canvas),
        Rect::new(0, 0, 63, 53),
        "the low ends clamp at 0, and only the low ends"
    );
    assert_eq!(
        padded_box(Rect::new(950, 960, 1000, 1000), 8, &config, canvas),
        Rect::new(937, 947, 1000, 1000),
        "the high ends clamp at the canvas, and only the high ends"
    );
}

/// §16.38 item 3(e)'s isolation radius is added *ahead of time*, and that sizing is what lets
/// `fill::padded_region` build both masks in the padded box's own frame. Stated as an assertion
/// rather than as a comment, because the frame choice depends on it: the padded box must be at
/// least `growth + isolation` wider than the box on every unclamped side.
#[test]
fn the_padded_box_is_wide_enough_for_the_fill_growth_and_a_further_isolation_growth() {
    let config = radii(0, 100, 0.0, 5, 0);
    let rect = Rect::new(400, 400, 420, 420);
    let growth = 13_u32;

    let padded = padded_box(rect, growth, &config, (1000, 1000));
    let slack_left = rect.x1 - padded.x1;
    let slack_right = padded.x2 - rect.x2;

    assert_eq!(slack_left, 18, "growth 13 + isolation 5");
    assert_eq!(slack_right, 18);
    assert!(
        slack_left >= (growth + config.inpainting_isolation_radius) as i32,
        "the padded frame must hold the fill growth AND the later isolation growth"
    );
}
