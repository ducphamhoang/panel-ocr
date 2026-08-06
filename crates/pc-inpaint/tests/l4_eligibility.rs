//! L4 — spec §16.38 items 3(c), 3(d) and 7(d): which regions are inpainted, in which order, from
//! which mask, and which config key must make no difference at all.

mod common;

use common::{radii, region};
use pc_config::InpainterConfig;
use pc_core::Rect;
use pc_inpaint::{combined_fill_binary, select_regions, FillSource};

fn box_at(x: i32, y: i32) -> Rect {
    Rect::new(x, y, x + 10, y + 10)
}

/// §16.38 item 3(c) + 3(d) + upstream `inpainting.py:76-101`'s build order.
///
/// The assertion is the exact `(index, FillSource)` **sequence**, because three separate claims
/// ride on it and a count would verify none of them: which rows are in, which mask each takes its
/// fill from, and that all failed rows precede all poorly-fitted ones (upstream appends the second
/// list to the first, and `analytics_thicknesses` at `:119` is emitted in that order).
#[test]
fn failed_rows_come_first_from_the_raw_mask_then_poorly_fitted_rows_from_the_combined_mask() {
    let config = InpainterConfig {
        inpainting_min_std_dev: 15.0,
        min_inpainting_radius: 7,
        ..InpainterConfig::default()
    };
    let regions = vec![
        // 0: poorly fitted — not failed, deviation over the floor, thin enough.
        region(box_at(0, 0), 20.0, false, Some(3)),
        // 1: failed.
        region(box_at(20, 0), 99.0, true, Some(4)),
        // 2: not failed and under the deviation floor — out.
        region(box_at(40, 0), 1.0, false, Some(3)),
        // 3: failed, and `thickness` is None — failure alone is enough (`:76-80` reads no
        //    thickness at all), so this one being IN is the point.
        region(box_at(60, 0), 50.0, true, None),
        // 4: poorly fitted exactly on both inclusive boundaries.
        region(box_at(80, 0), 15.0, false, Some(7)),
    ];

    let selected = select_regions(&regions, &config);

    // Anti-vacuity literal: hard-coded, not computed from `regions`.
    assert_eq!(selected.len(), 4, "expected exactly four eligible rows");
    let actual: Vec<(usize, FillSource)> = selected
        .iter()
        .map(|region| (region.index, region.source))
        .collect();
    assert_eq!(
        actual,
        vec![
            (1, FillSource::RawMask),
            (3, FillSource::RawMask),
            (0, FillSource::CombinedMask),
            (4, FillSource::CombinedMask),
        ],
        "failed rows first (raw mask), then poorly-fitted rows (combined mask), \
         each in `regions` order"
    );
}

/// §16.38 item 3(c)'s `thickness is not None` clause, carrying upstream's own comment: *"For box
/// masks, this is none. We don't need to inpaint those, they are always good."*
///
/// Asserted as a **pair** so that neither half can pass vacuously: the same region with a
/// thickness is eligible, and the only difference between the two rows is that field.
#[test]
fn a_box_mask_row_with_no_thickness_is_never_poorly_fitted_but_the_same_row_with_one_is() {
    let config = InpainterConfig {
        inpainting_min_std_dev: 15.0,
        min_inpainting_radius: 7,
        ..InpainterConfig::default()
    };

    let without = vec![region(box_at(0, 0), 999.0, false, None)];
    let with = vec![region(box_at(0, 0), 999.0, false, Some(0))];

    assert!(
        select_regions(&without, &config).is_empty(),
        "a `None` thickness is upstream's box-mask marker and is never eligible"
    );
    assert_eq!(
        select_regions(&with, &config)
            .iter()
            .map(|region| (region.index, region.source))
            .collect::<Vec<_>>(),
        vec![(0, FillSource::CombinedMask)],
        "the identical row with a thickness IS eligible, so the clause above is doing the work"
    );
}

/// §16.38 item 3(c): both comparisons are **inclusive** — upstream's `deviation >=
/// inpainting_min_std_dev` and `thickness <= min_inpainting_radius` (`inpainting.py:86`, `:89`).
///
/// Each row sits one step off a boundary in exactly one field, so a `>` or a `<` in either place
/// changes the answer set.
#[test]
fn the_deviation_floor_and_the_thickness_ceiling_are_both_inclusive() {
    let config = InpainterConfig {
        inpainting_min_std_dev: 15.0,
        min_inpainting_radius: 7,
        ..InpainterConfig::default()
    };
    let regions = vec![
        region(box_at(0, 0), 15.0, false, Some(7)), // 0: exactly on both bounds — IN
        region(box_at(20, 0), 14.999, false, Some(7)), // 1: a hair under the floor — OUT
        region(box_at(40, 0), 15.0, false, Some(8)), // 2: a hair over the ceiling — OUT
        region(box_at(60, 0), 15.001, false, Some(6)), // 3: inside both — IN
    ];

    let indices: Vec<usize> = select_regions(&regions, &config)
        .iter()
        .map(|region| region.index)
        .collect();

    assert_eq!(indices, vec![0, 3]);
}

/// §16.38 item 7(d), quoted: *"What L4 owes is a gate that keeps it inert: a test asserting that
/// two runs differing only in `inpainting_max_mask_radius` select the identical eligible-region
/// set. A comment saying 'unused' enforces nothing."*
///
/// The sweep spans the key's whole plausible range, including its own default (`6`) and both
/// values upstream's two disagreeing files ship. The hard-coded expected length keeps the gate from
/// passing by selecting nothing under every setting.
#[test]
fn inpainting_max_mask_radius_never_changes_the_eligible_set() {
    let regions = vec![
        region(box_at(0, 0), 99.0, true, Some(4)),
        region(box_at(20, 0), 20.0, false, Some(3)),
        region(box_at(40, 0), 1.0, false, Some(3)),
        region(box_at(60, 0), 30.0, false, Some(2)),
    ];

    let baseline: Vec<(usize, FillSource)> = select_regions(
        &regions,
        &InpainterConfig {
            inpainting_min_std_dev: 15.0,
            min_inpainting_radius: 7,
            inpainting_max_mask_radius: InpainterConfig::DEFAULT_MAX_MASK_RADIUS,
            ..InpainterConfig::default()
        },
    )
    .iter()
    .map(|region| (region.index, region.source))
    .collect();

    assert_eq!(
        baseline.len(),
        3,
        "the eligible set must be non-trivial, or the sweep below proves nothing"
    );

    for radius in [0_u32, 1, 2, 5, 6, 7, 20, 100, u32::MAX] {
        let swept: Vec<(usize, FillSource)> = select_regions(
            &regions,
            &InpainterConfig {
                inpainting_min_std_dev: 15.0,
                min_inpainting_radius: 7,
                inpainting_max_mask_radius: radius,
                ..InpainterConfig::default()
            },
        )
        .iter()
        .map(|region| (region.index, region.source))
        .collect();
        assert_eq!(
            swept, baseline,
            "inpainting_max_mask_radius = {radius} changed the eligible set; \
             §16.38 item 7(d) requires it to be inert"
        );
    }
}

/// §16.38 item 3(d), and the reading is now **ratified**: `DEVIATION(28)` at §14 item 28, per
/// §16.38 item 21 (Fable tie-break, 2026-08-07). It is deliberately its **own** register entry and
/// not `DEVIATION(5)` propagating — item 5 names `denoiser.py:62`'s `mask.convert("L")`, a
/// different upstream call in a different file feeding a different stage.
///
/// Upstream binarises the combined fill mask with `mask_image.convert("1")`
/// (`inpainting.py:63`), which PIL routes through `L` — the luma of the RGB channels, alpha
/// discarded — so a black opaque fill reads as *uncovered* and every poorly-fitted region in the
/// `black_bubble` demo page — a demo fixture, not a profile — silently gets nothing inpainted. What item 5 supplies is the ground,
/// reused verbatim rather than widened: *"A mask's 'is this pixel covered' signal must not depend
/// on the brightness of its fill color"*. This test pins the alpha reading, so switching to
/// upstream's luma path turns it red instead of changing behaviour quietly. See
/// `fill::combined_fill_binary`, which carries the `DEVIATION(28)` site comment.
#[test]
fn a_black_but_opaque_combined_mask_pixel_counts_as_covered_and_a_transparent_white_one_does_not() {
    use image::{Rgba, RgbaImage};

    let mut mask = RgbaImage::from_pixel(3, 1, Rgba([0, 0, 0, 0]));
    mask.put_pixel(0, 0, Rgba([0, 0, 0, 255])); // black, opaque   -> covered
    mask.put_pixel(1, 0, Rgba([255, 255, 255, 0])); // white, transparent -> not covered
    mask.put_pixel(2, 0, Rgba([255, 255, 255, 255])); // white, opaque   -> covered

    let binary = combined_fill_binary(&mask);

    assert!(
        binary.get(0, 0),
        "a black opaque fill is coverage; upstream's luma path would drop it (DEVIATION(28))"
    );
    assert!(!binary.get(1, 0), "alpha 0 is never coverage");
    assert!(binary.get(2, 0));
}

/// §16.38 item 3(e)'s inputs come from the config, so a hand-built `radii` helper must not silently
/// disagree with the ratified defaults this suite leans on (item 13(a)).
#[test]
fn the_ratified_inpainter_defaults_are_what_this_suite_assumes() {
    let default = InpainterConfig::default();
    assert!(!default.inpainting_enabled);
    assert_eq!(default.inpainting_min_std_dev, 15.0);
    assert_eq!(default.inpainting_max_mask_radius, 6);
    assert_eq!(default.min_inpainting_radius, 7);
    assert_eq!(default.max_inpainting_radius, 20);
    assert_eq!(default.inpainting_radius_multiplier, 0.2);
    assert_eq!(default.inpainting_isolation_radius, 5);
    assert_eq!(default.inpainting_fade_radius, 4);

    // The helper overrides exactly the five radii and nothing else.
    let overridden = radii(1, 2, 3.0, 4, 5);
    assert_eq!(overridden.min_inpainting_radius, 1);
    assert_eq!(overridden.max_inpainting_radius, 2);
    assert_eq!(overridden.inpainting_radius_multiplier, 3.0);
    assert_eq!(overridden.inpainting_isolation_radius, 4);
    assert_eq!(overridden.inpainting_fade_radius, 5);
    assert_eq!(overridden.inpainting_min_std_dev, 15.0);
}
