//! Task T5 (§16.35 item 8) -- wire the already-frozen rescue policy into `fit_region`.
//! FROZEN. T4's config surface is a prerequisite for this suite to compile.

mod common;

use common::{box_mask_of, canvas_of, cut_of, memory_input, page, text_raw_mask, PAGE_SIZE};
use image::{DynamicImage, GrayImage, Luma};
use pc_config::MaskerConfig;
use pc_core::{PageData, Rect};
use pc_mask::fit::{fit_accepted, fit_region, fit_region_scored};

const MASKING: Rect = Rect {
    x1: 70,
    y1: 50,
    x2: 130,
    y2: 110,
};
const REFERENCE: Rect = Rect {
    x1: 50,
    y1: 30,
    x2: 150,
    y2: 130,
};

/// The rescue-eligible real-pixel ladder frozen first in q2. It is repeated here because
/// q2 intentionally tests the pure policy without making a T5 end-to-end claim.
fn banded_page() -> PageData {
    let text = [Rect::new(80, 60, 120, 100)];
    let mut image = GrayImage::from_pixel(PAGE_SIZE.0, PAGE_SIZE.1, Luma([200]));
    for y in 54..57 {
        for x in 0..PAGE_SIZE.0 {
            image.put_pixel(x, y, Luma([90]));
        }
    }
    for y in 60..100 {
        for x in 80..120 {
            image.put_pixel(x, y, Luma([0]));
        }
    }
    page(
        DynamicImage::ImageLuma8(image),
        text_raw_mask(PAGE_SIZE, &text),
        &[MASKING],
        20,
        1.0,
    )
}

fn disabled() -> MaskerConfig {
    MaskerConfig {
        mask_fallback_to_lowest_deviation: false,
        ..MaskerConfig::default()
    }
}

#[test]
fn default_on_fit_region_rescues_the_real_ladder_and_reports_the_rescued_candidate() {
    // §16.35 items 1, 4 and 7. q2 hand-traces this ladder: greedy index 6 fails at
    // ~15.819, while the already-scored argmin at index 10 passes at ~14.319.
    let page = banded_page();
    let config = MaskerConfig::default();
    assert!(
        config.mask_fallback_to_lowest_deviation,
        "T4's shipped default"
    );
    let canvas = canvas_of(&page);
    let cut = cut_of(&page);
    let boxes = box_mask_of(&page);

    let report = fit_region_scored(&canvas, &cut, &boxes, MASKING, REFERENCE, &config)
        .expect("the region reaches fitting");
    let direct = fit_region(&canvas, &cut, &boxes, MASKING, REFERENCE, &config)
        .expect("the wrapper returns the fitment");

    assert_eq!(report.greedy_index, 6);
    assert!(!fit_accepted(
        report.candidate_deviations[6],
        config.mask_max_standard_deviation
    ));
    assert!(report.rescued);
    assert_eq!(report.fitment.candidate_index, 10);
    assert_eq!(
        report.fitment.std_deviation,
        report.candidate_deviations[10]
    );
    assert!(fit_accepted(
        report.fitment.std_deviation,
        config.mask_max_standard_deviation
    ));
    assert!(report.fitment.mask.is_some());
    assert!(!report.fitment.failed());
    assert_eq!(direct, report.fitment, "fit_region is the thin wrapper");
}

#[test]
fn disabling_the_flag_reproduces_the_pre_t5_selection_output_on_the_same_fixture() {
    // §16.35 item 4's mandatory DEVIATION(21) control: not merely `mask: None`, but the
    // old greedy candidate's exact index/deviation/median/thickness output.
    let page = banded_page();
    let config = disabled();
    let canvas = canvas_of(&page);
    let cut = cut_of(&page);
    let boxes = box_mask_of(&page);
    let report = fit_region_scored(&canvas, &cut, &boxes, MASKING, REFERENCE, &config)
        .expect("the region reaches fitting");
    let direct = fit_region(&canvas, &cut, &boxes, MASKING, REFERENCE, &config)
        .expect("failed fits are still reported");

    assert!(!report.rescued);
    assert_eq!(report.greedy_index, 6);
    assert_eq!(report.fitment.candidate_index, 6);
    assert_eq!(report.fitment.std_deviation, report.candidate_deviations[6]);
    assert_eq!(report.fitment.thickness, Some(16));
    assert!(report.fitment.mask.is_none());
    assert!(report.fitment.failed());
    assert_eq!(
        direct, report.fitment,
        "the disabled wrapper must not re-resolve"
    );
}

#[test]
fn run_propagates_the_rescue_to_mask_data_analytics_and_pixels_without_schema_growth() {
    // §16.35 item 7's cross-crate consequence starts here: a rescued region is an ordinary
    // successful fit in MaskRegionStats/MaskFittingAnalytic, while flag=false retains the
    // old failed row. The diagnostics themselves do not enter either schema (q3).
    let enabled = pc_mask::run(memory_input(banded_page(), MaskerConfig::default())).unwrap();
    let disabled = pc_mask::run(memory_input(banded_page(), disabled())).unwrap();

    assert_eq!(enabled.mask_data.regions.len(), 1);
    assert_eq!(enabled.analytics.len(), 1);
    assert!(!enabled.mask_data.regions[0].failed);
    assert!(enabled.analytics[0].fit_found);
    assert_eq!(enabled.analytics[0].candidate_index, 10);
    assert_eq!(
        enabled.mask_data.regions[0].std_deviation,
        enabled.analytics[0].std_deviation
    );
    assert!(enabled
        .combined_mask
        .load()
        .unwrap()
        .to_rgba8()
        .pixels()
        .any(|pixel| pixel.0[3] == 255));

    assert!(disabled.mask_data.regions[0].failed);
    assert!(!disabled.analytics[0].fit_found);
    assert_eq!(disabled.analytics[0].candidate_index, 6);
    assert!(disabled
        .combined_mask
        .load()
        .unwrap()
        .to_rgba8()
        .pixels()
        .all(|pixel| pixel.0[3] == 0));
}

#[test]
fn the_rescued_fit_preserves_failed_equivalence_and_the_existing_threshold() {
    // §16.35 items 2(P1) and 7: wiring may change which already-scored candidate is
    // selected, never the acceptance predicate.
    for config in [MaskerConfig::default(), disabled()] {
        let page = banded_page();
        let fitment = fit_region(
            &canvas_of(&page),
            &cut_of(&page),
            &box_mask_of(&page),
            MASKING,
            REFERENCE,
            &config,
        )
        .expect("the region reaches fitting");
        assert_eq!(
            fitment.failed(),
            fitment.std_deviation > config.mask_max_standard_deviation
        );
        if fitment.mask.is_some() {
            assert!(fit_accepted(
                fitment.std_deviation,
                config.mask_max_standard_deviation
            ));
        }
    }
}
