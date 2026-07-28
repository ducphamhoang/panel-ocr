//! Task M4 -- spec §10.3 step 3, §10.7(A)8, §10.7(A)9, §10.7(A)10, §10.7(A)11,
//! §16.9 items 9-12. FROZEN.

mod common;

use common::{
    box_mask_of, canvas_of, cut_of, page, set_pixels, simple_page, text_raw_mask, uniform_base,
    PAGE_SIZE,
};
use image::{DynamicImage, GrayImage, Luma};
use pc_config::MaskerConfig;
use pc_core::Rect;
use pc_mask::border::{BlankMask, BorderStats};
use pc_mask::{fit_region, select_candidate};
use std::cell::RefCell;

fn stats(std_deviation: f64) -> Result<BorderStats, BlankMask> {
    Ok(BorderStats {
        std_deviation,
        median_color: [1, 2, 3],
    })
}

/// The masking region of [`simple_page`], and its reference box.
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

// ------------------------------------------------------------ selection policy (§10.7(A)8)

#[test]
fn a8_selection_requires_a_ten_percent_improvement() {
    // spec §10.7(A)8, hand-traced with mask_improvement_threshold = 0.1:
    //   i=0: dev 10.0 -> accepted unconditionally
    //   i=1: needs <= 10.0*0.9 = 9.0; 9.5 -> rejected
    //   i=2: needs <= 9.0;            8.0 -> accepted
    //   i=3: needs <= 8.0*0.9 = 7.2;  8.0 -> rejected
    // Chosen index = 2.
    let deviations = [10.0, 9.5, 8.0, 8.0];

    let selected = select_candidate(deviations.len(), false, 0.1, |index| {
        stats(deviations[index])
    })
    .expect("no blank candidate");

    assert_eq!(selected.index, 2);
    assert_eq!(selected.std_deviation, 8.0);
    assert_eq!(selected.median_color, [1, 2, 3]);
}

#[test]
fn a8_a_second_perfect_candidate_wins_because_the_comparison_is_inclusive() {
    // spec §10.7(A)8: with [0.0, 0.0], index 1 needs `<= 0.0 * 0.9 == 0.0` and 0.0 <= 0.0
    // holds, so the LARGER mask wins. This edge case decides whether a perfect fit
    // prefers the largest mask; it must be locked.
    let deviations = [0.0, 0.0];

    let selected = select_candidate(deviations.len(), false, 0.1, |index| {
        stats(deviations[index])
    })
    .expect("no blank candidate");

    assert_eq!(selected.index, 1);
    assert_eq!(selected.std_deviation, 0.0);
}

#[test]
fn a8_nothing_beats_a_perfect_first_candidate_except_another_perfect_one() {
    // The other half of the same rule: once best_dev is 0.0, `best_dev * (1 - t)` is 0.0,
    // so any positive deviation is rejected forever.
    let deviations = [0.0, 0.001, 5.0];

    let selected = select_candidate(deviations.len(), false, 0.1, |index| {
        stats(deviations[index])
    })
    .expect("no blank candidate");

    assert_eq!(selected.index, 0);
}

#[test]
fn a8_a_blank_candidate_abandons_the_whole_region() {
    // spec §10.3 step 8: an empty edge set on ANY candidate propagates out as BlankMask,
    // which `fit_region` turns into `None` for the region (§16.9 item 8).
    let result = select_candidate(3, false, 0.1, |index| {
        if index == 1 {
            Err(BlankMask)
        } else {
            stats(1.0)
        }
    });

    assert_eq!(result, Err(BlankMask));
}

// ------------------------------------------------------- fast-mode early break (§10.7(A)11)

#[test]
fn a11_fast_mode_evaluates_exactly_one_candidate_when_the_box_scores_zero() {
    // spec §10.7(A)11 / §16.9 item 10: with mask_selection_fast the box candidate is
    // first; scoring exactly 0.0 breaks out immediately. The instrumented scorer panics
    // if any later candidate is scored.
    let calls = RefCell::new(0_usize);

    let selected = select_candidate(12, true, 0.1, |index| {
        *calls.borrow_mut() += 1;
        assert_eq!(
            index, 0,
            "fast mode must not score any candidate past the first"
        );
        stats(0.0)
    })
    .expect("no blank candidate");

    assert_eq!(selected.index, 0);
    assert_eq!(*calls.borrow(), 1);
}

#[test]
fn a11_fast_mode_keeps_scoring_while_deviations_are_nonzero() {
    let calls = RefCell::new(0_usize);
    let deviations = [3.0, 2.0, 0.0, 9.0];

    let selected = select_candidate(deviations.len(), true, 0.1, |index| {
        *calls.borrow_mut() += 1;
        stats(deviations[index])
    })
    .expect("no blank candidate");

    assert_eq!(selected.index, 2);
    assert_eq!(
        *calls.borrow(),
        3,
        "the break happens at the zero, not before"
    );
}

#[test]
fn a11_slow_mode_never_breaks_early() {
    // The default (mask_selection_fast = false) scores every candidate even after a
    // perfect one -- that is what lets a later perfect-but-larger mask win (§10.7(A)8).
    let calls = RefCell::new(0_usize);
    let deviations = [0.0, 0.0, 0.0, 0.0];

    let selected = select_candidate(deviations.len(), false, 0.1, |index| {
        *calls.borrow_mut() += 1;
        stats(deviations[index])
    })
    .expect("no blank candidate");

    assert_eq!(selected.index, 3);
    assert_eq!(*calls.borrow(), 4);
}

// ------------------------------------------------------------ fit_region (§10.3 step 3)

#[test]
fn fit_region_on_a_uniform_background_picks_the_box_candidate() {
    // Hand-traced against `simple_page`: a 200-valued background with a 40x40 black text
    // block, masking (70,50,130,110), reference (50,30,150,130) -> a 100x100 analysis
    // canvas. Every candidate's border ring lies on the uniform 200 background, so every
    // deviation is exactly 0.0; by §10.7(A)8's inclusive comparison the LAST candidate
    // wins, and in slow mode that is the box candidate (thickness None).
    let page = simple_page();
    let canvas = canvas_of(&page);
    let cut = cut_of(&page);
    let boxes = box_mask_of(&page);

    let fitment = fit_region(
        &canvas,
        &cut,
        &boxes,
        MASKING,
        REFERENCE,
        &MaskerConfig::default(),
    )
    .expect("a non-blank region fits");

    assert_eq!(fitment.candidate_index, 11);
    assert_eq!(fitment.thickness, None);
    assert_eq!(fitment.std_deviation, 0.0);
    assert_eq!(fitment.median_color, [200, 200, 200]);
    assert_eq!(fitment.coords, (50, 30));
    assert_eq!(fitment.masking_rect, MASKING);

    // The chosen mask is the box candidate: the masking rect placed into the reference
    // crop at (masking.x1 - reference.x1, masking.y1 - reference.y1) == (20, 20).
    let mask = fitment.mask.expect("a fit was found");
    assert_eq!(mask.dimensions(), (100, 100));
    assert_eq!(mask.count_set(), 60 * 60);
    assert_eq!(mask.bbox(), Some(Rect::new(20, 20, 80, 80)));
}

#[test]
fn fit_region_offsets_the_precise_cut_into_the_reference_frame() {
    // spec §10.3 step 3 steps 1-3, isolated: with mask_growth_steps = 1 and
    // min_mask_thickness = 0 the only growth candidate IS the precise cut, and it scores
    // 0.0 on the uniform background, so slow-mode ordering still lands on the box
    // candidate -- but a fast-mode run stops on the box candidate at index 0 with the
    // same result. What this pins is the geometry: the text block at page (80,60)-(120,100)
    // must land at crop (30,30)-(70,70).
    let page = simple_page();
    let cut = cut_of(&page);

    let precise_cut = cut.crop_into(MASKING, (100, 100), (20, 20));

    assert_eq!(precise_cut.bbox(), Some(Rect::new(30, 30, 70, 70)));
    assert_eq!(set_pixels(&precise_cut).len(), 40 * 40);
}

#[test]
fn a10_a_blank_precise_cut_drops_the_region() {
    // spec §10.7(A)10: a region whose `cut` crop is empty is detector noise -- fit_region
    // returns None (and logs WARN); the region contributes no MaskData and no analytics.
    let text = [Rect::new(10, 10, 30, 30)];
    let page = page(
        uniform_base(PAGE_SIZE, 200, &text),
        text_raw_mask(PAGE_SIZE, &text),
        &[Rect::new(120, 100, 160, 140)],
        20,
        1.0,
    );
    let canvas = canvas_of(&page);
    let cut = cut_of(&page);
    let boxes = box_mask_of(&page);

    let fitment = fit_region(
        &canvas,
        &cut,
        &boxes,
        Rect::new(120, 100, 160, 140),
        Rect::new(100, 80, 180, 160),
        &MaskerConfig::default(),
    );

    assert!(fitment.is_none());
}

#[test]
fn a9_a_deviation_above_the_maximum_fails_the_fit_but_keeps_its_numbers() {
    // spec §10.7(A)9 / §16.9 item 12: when the best candidate's deviation exceeds
    // mask_max_standard_deviation the mask is dropped (`mask: None`) but the region is
    // still reported -- deviation and median colour included. Here the base is a hard
    // vertical ramp, so every candidate's border straddles a wide range of values and
    // the deviation is far above the default 15.0.
    let text = [Rect::new(80, 60, 120, 100)];
    let ramp = GrayImage::from_fn(PAGE_SIZE.0, PAGE_SIZE.1, |x, _| Luma([(x * 5 % 256) as u8]));
    let page = page(
        DynamicImage::ImageLuma8(ramp),
        text_raw_mask(PAGE_SIZE, &text),
        &[Rect::new(70, 50, 130, 110)],
        20,
        1.0,
    );
    let canvas = canvas_of(&page);
    let cut = cut_of(&page);
    let boxes = box_mask_of(&page);

    let fitment = fit_region(
        &canvas,
        &cut,
        &boxes,
        MASKING,
        REFERENCE,
        &MaskerConfig::default(),
    )
    .expect("the precise cut is not blank, so the region is reported");

    assert!(
        fitment.mask.is_none(),
        "the fit failed, so no mask is painted"
    );
    assert!(fitment.failed());
    assert!(
        fitment.std_deviation > 15.0,
        "deviation {} should exceed mask_max_standard_deviation",
        fitment.std_deviation
    );
    assert!(fitment.candidate_index < 12);
}

#[test]
fn a9_raising_the_maximum_turns_the_same_region_into_a_success() {
    // The failure threshold is the ONLY difference between the two outcomes: same page,
    // same candidates, same chosen index -- only `mask` changes.
    let text = [Rect::new(80, 60, 120, 100)];
    let ramp = GrayImage::from_fn(PAGE_SIZE.0, PAGE_SIZE.1, |x, _| Luma([(x * 5 % 256) as u8]));
    let page = page(
        DynamicImage::ImageLuma8(ramp),
        text_raw_mask(PAGE_SIZE, &text),
        &[Rect::new(70, 50, 130, 110)],
        20,
        1.0,
    );
    let canvas = canvas_of(&page);
    let cut = cut_of(&page);
    let boxes = box_mask_of(&page);
    let strict = fit_region(
        &canvas,
        &cut,
        &boxes,
        MASKING,
        REFERENCE,
        &MaskerConfig::default(),
    )
    .expect("reported region");

    let lenient = fit_region(
        &canvas,
        &cut,
        &boxes,
        MASKING,
        REFERENCE,
        &MaskerConfig {
            mask_max_standard_deviation: 1000.0,
            ..MaskerConfig::default()
        },
    )
    .expect("reported region");

    assert!(lenient.mask.is_some());
    assert_eq!(lenient.candidate_index, strict.candidate_index);
    assert_eq!(lenient.std_deviation, strict.std_deviation);
    assert_eq!(lenient.median_color, strict.median_color);
}

#[test]
fn fit_region_skips_a_degenerate_rect_instead_of_erroring() {
    // spec §16.9 item 11: an empty or fully out-of-canvas rect (reachable from a
    // hand-edited #clean.json) is a skipped region, never a StageError.
    let page = simple_page();
    let canvas = canvas_of(&page);
    let cut = cut_of(&page);
    let boxes = box_mask_of(&page);

    assert!(fit_region(
        &canvas,
        &cut,
        &boxes,
        Rect::new(70, 50, 70, 110),
        Rect::new(50, 30, 50, 130),
        &MaskerConfig::default()
    )
    .is_none());
    assert!(fit_region(
        &canvas,
        &cut,
        &boxes,
        Rect::new(900, 900, 950, 950),
        Rect::new(880, 880, 970, 970),
        &MaskerConfig::default()
    )
    .is_none());
}

#[test]
fn fit_region_in_fast_mode_takes_the_box_candidate_first() {
    // spec §10.3 step 7 + §10.7(A)11 end to end: with mask_selection_fast the box
    // candidate is index 0, scores 0.0 on the uniform background, and the loop stops
    // there -- same painted mask as slow mode, reached in one scoring call.
    let page = simple_page();
    let canvas = canvas_of(&page);
    let cut = cut_of(&page);
    let boxes = box_mask_of(&page);

    let fitment = fit_region(
        &canvas,
        &cut,
        &boxes,
        MASKING,
        REFERENCE,
        &MaskerConfig {
            mask_selection_fast: true,
            ..MaskerConfig::default()
        },
    )
    .expect("a non-blank region fits");

    assert_eq!(fitment.candidate_index, 0);
    assert_eq!(fitment.thickness, None);
    assert_eq!(
        fitment.mask.expect("a fit was found").bbox(),
        Some(Rect::new(20, 20, 80, 80))
    );
}
