//! Task T1 (§16.35 item 8) -- `fit_region_scored` / `FitReport`, the introspection half
//! of §16.35 item 7, and T1's zero-behaviour-change condition. FROZEN.
//!
//! What this file is for, precisely: item 7 makes `fit_region` "a thin wrapper over"
//! `fit_region_scored`, so the risk T1 carries is (a) the wrapper diverging from the
//! function it wraps and (b) the report describing a different selection than the one the
//! `Fitment` was built from. Every expectation below is either a literal hand-traced in
//! the frozen `m4_fit.rs` (an oracle that predates this refactor) or a structural
//! invariant of the report itself.
//!
//! T1 does NOT wire the rescue: `[masker] mask_fallback_to_lowest_deviation` does not
//! exist yet (T4) and nothing here can switch it on, so `FitReport.rescued` is asserted
//! `false` throughout and the fallback-firing gates live in `q1_fallback_policy.rs` (on
//! the pure policy) and, later, in T5's own suite (end to end).

mod common;

use common::{box_mask_of, canvas_of, cut_of, page, simple_page, text_raw_mask, PAGE_SIZE};
use image::{DynamicImage, GrayImage, Luma};
use pc_config::MaskerConfig;
use pc_core::{PageData, Rect};
use pc_mask::fit::{fit_region, fit_region_scored, FitReport};

/// The masking region of the `m4_fit.rs` fixtures, and its reference box.
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

/// `mask_growth_steps = 11` growth candidates plus the box candidate (§10.3 steps 6-7,
/// with the shipped `[masker]` defaults).
const CANDIDATE_COUNT: usize = 12;

/// The `a9_*` fixture of `m4_fit.rs`: a hard vertical ramp, so every candidate's border
/// straddles a wide range of values and no candidate can pass the 15.0 fail-safe.
fn ramp_page() -> PageData {
    let text = [Rect::new(80, 60, 120, 100)];
    let ramp = GrayImage::from_fn(PAGE_SIZE.0, PAGE_SIZE.1, |x, _| Luma([(x * 5 % 256) as u8]));
    page(
        DynamicImage::ImageLuma8(ramp),
        text_raw_mask(PAGE_SIZE, &text),
        &[Rect::new(70, 50, 130, 110)],
        20,
        1.0,
    )
}

fn report(
    page: &PageData,
    config: &MaskerConfig,
    masking: Rect,
    reference: Rect,
) -> Option<FitReport> {
    fit_region_scored(
        &canvas_of(page),
        &cut_of(page),
        &box_mask_of(page),
        masking,
        reference,
        config,
    )
}

fn fast() -> MaskerConfig {
    MaskerConfig {
        mask_selection_fast: true,
        ..MaskerConfig::default()
    }
}

// ------------------------------------------------ the report describes the real selection

#[test]
fn every_candidate_is_reported_and_the_greedy_index_indexes_it() {
    // §16.35 item 7: `FitReport.candidate_deviations` is the per-candidate vector and
    // `greedy_index` is §10.3 step 9's pick within it. The literals are the frozen trace of
    // `m4_fit.rs::fit_region_on_a_uniform_background_picks_the_box_candidate`: 12
    // candidates, every border on the uniform 200 background so every deviation is exactly
    // 0.0, and the inclusive `<=` carrying the ratchet to the LAST candidate, index 11.
    let report = report(&simple_page(), &MaskerConfig::default(), MASKING, REFERENCE)
        .expect("a non-blank region fits");

    assert_eq!(report.candidate_deviations.len(), CANDIDATE_COUNT);
    assert_eq!(report.candidate_deviations, vec![0.0; CANDIDATE_COUNT]);
    assert_eq!(report.greedy_index, 11);
    assert_eq!(
        report.candidate_deviations[report.greedy_index],
        report.fitment.std_deviation
    );
    assert_eq!(report.fitment.candidate_index, 11);
    assert_eq!(report.fitment.thickness, None);
    assert_eq!(report.fitment.median_color, [200, 200, 200]);
    assert_eq!(report.fitment.coords, (50, 30));
    assert_eq!(report.fitment.masking_rect, MASKING);
    assert_eq!(
        report
            .fitment
            .mask
            .as_ref()
            .expect("a fit was found")
            .bbox(),
        Some(Rect::new(20, 20, 80, 80))
    );
}

#[test]
fn fast_mode_reports_only_the_candidates_it_actually_scored() {
    // §16.35 item 1 restricts the rescue to candidates "already scored in §10.3 step 9",
    // so the report must be truthful about how short the fast-mode ladder is: the box
    // candidate comes first, scores 0.0, and the loop breaks (§16.9 item 10) -- ONE entry,
    // not 12. The chosen index 0 is the frozen trace of
    // `m4_fit.rs::fit_region_in_fast_mode_takes_the_box_candidate_first`.
    let report =
        report(&simple_page(), &fast(), MASKING, REFERENCE).expect("a non-blank region fits");

    assert_eq!(report.candidate_deviations, vec![0.0]);
    assert_eq!(report.greedy_index, 0);
    assert_eq!(report.fitment.candidate_index, 0);
    assert_eq!(report.fitment.thickness, None);
}

#[test]
fn a_failed_region_still_reports_its_whole_candidate_ladder() {
    // §16.9 item 12 (the region is reported even when the fit fails) extended to the new
    // diagnostic: the vector is what §16.35 item 1's rescue would later choose from, so it
    // must be present on exactly the branch that returns `mask: None`.
    //
    // The ramp's 12 deviations all sit between 50 and 75 (measured off the FROZEN
    // `grow`/`border` code that T1 does not touch, gated by `m2_grow.rs`/`m3_border.rs`):
    // the ratchet stops at index 0 because every later candidate would need
    // <= 55.7 * 0.9 == 50.16 and the smallest, at index 3, is 50.96. That index-3 minimum
    // is why this fixture is the interesting one -- the argmin genuinely differs from the
    // greedy pick, and yet it is still above 15.0, so no rescue could ever save it.
    let report = report(&ramp_page(), &MaskerConfig::default(), MASKING, REFERENCE)
        .expect("the precise cut is not blank, so the region is reported");

    assert_eq!(report.candidate_deviations.len(), CANDIDATE_COUNT);
    assert_eq!(report.greedy_index, 0);
    assert!(report.fitment.mask.is_none());
    assert!(report.fitment.failed());
    assert!(
        report
            .candidate_deviations
            .iter()
            .all(|deviation| *deviation > 15.0),
        "no candidate of the ramp fixture may pass the fail-safe: {:?}",
        report.candidate_deviations
    );

    let argmin = report
        .candidate_deviations
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("twelve candidates");
    assert_eq!(
        argmin, 3,
        "the ramp's argmin is candidate 3, not the greedy pick"
    );
    assert_ne!(argmin, report.greedy_index);
}

#[test]
fn the_reported_deviations_are_consistent_with_the_fitment_on_every_fixture() {
    // The structural invariant the diagnostic's usefulness rests on, checked on a passing
    // region, a failing region and a truncated fast-mode ladder.
    for (label, page, config) in [
        ("uniform slow", simple_page(), MaskerConfig::default()),
        ("uniform fast", simple_page(), fast()),
        ("ramp slow", ramp_page(), MaskerConfig::default()),
        ("ramp fast", ramp_page(), fast()),
    ] {
        let report = report(&page, &config, MASKING, REFERENCE)
            .unwrap_or_else(|| panic!("{label}: the region is reported"));

        assert!(
            !report.candidate_deviations.is_empty(),
            "{label}: at least the box candidate is always scored"
        );
        assert!(
            report.greedy_index < report.candidate_deviations.len(),
            "{label}: greedy index out of range"
        );
        assert_eq!(
            report.candidate_deviations[report.greedy_index], report.fitment.std_deviation,
            "{label}: the Fitment reports a deviation the ladder does not contain"
        );
        assert_eq!(
            report.greedy_index, report.fitment.candidate_index,
            "{label}: the report and the Fitment disagree on which candidate was chosen"
        );
    }
}

#[test]
fn fitment_failed_is_equivalent_to_the_deviation_exceeding_the_maximum() {
    // §16.35 item 7, verbatim: "keeping `Fitment::failed() == (std_deviation >
    // mask_max_standard_deviation)` true on both branches". `failed()` reads `mask.is_none()`
    // (`fit.rs:37-39`), so this asserts the two definitions still coincide -- the property
    // T5's rescue must preserve, pinned before the rescue exists.
    let lenient = MaskerConfig {
        mask_max_standard_deviation: 1000.0,
        ..MaskerConfig::default()
    };
    for (label, page, config) in [
        ("uniform passes", simple_page(), MaskerConfig::default()),
        ("ramp fails", ramp_page(), MaskerConfig::default()),
        (
            "ramp passes with a raised maximum",
            ramp_page(),
            lenient.clone(),
        ),
    ] {
        let report = report(&page, &config, MASKING, REFERENCE)
            .unwrap_or_else(|| panic!("{label}: the region is reported"));

        assert_eq!(
            report.fitment.failed(),
            report.fitment.std_deviation > config.mask_max_standard_deviation,
            "{label}: failed() and the threshold disagree"
        );
    }
    // Anti-vacuity: the loop above must actually visit both truths, or it proves nothing.
    assert!(
        report(&ramp_page(), &MaskerConfig::default(), MASKING, REFERENCE)
            .expect("reported")
            .fitment
            .failed()
    );
    assert!(!report(&ramp_page(), &lenient, MASKING, REFERENCE)
        .expect("reported")
        .fitment
        .failed());
}

// ------------------------------------------------------- T1's zero-behaviour-change gate

#[test]
fn fit_region_returns_exactly_the_fitment_field_of_fit_region_scored() {
    // §16.35 item 7: "`fit_region` becomes a thin wrapper over" `fit_region_scored`. Both
    // the `Some` and the `None` branches are covered, because the wrapper could plausibly
    // diverge only on one of them (a degenerate rect and a blank precise cut are the two
    // reasons a region is dropped, §16.9 item 11 / §10.7(A)10).
    let blank_cut_page = {
        let text = [Rect::new(10, 10, 30, 30)];
        page(
            common::uniform_base(PAGE_SIZE, 200, &text),
            text_raw_mask(PAGE_SIZE, &text),
            &[Rect::new(120, 100, 160, 140)],
            20,
            1.0,
        )
    };

    let cases: [(&str, PageData, MaskerConfig, Rect, Rect); 6] = [
        (
            "uniform slow",
            simple_page(),
            MaskerConfig::default(),
            MASKING,
            REFERENCE,
        ),
        ("uniform fast", simple_page(), fast(), MASKING, REFERENCE),
        (
            "ramp slow (mask: None)",
            ramp_page(),
            MaskerConfig::default(),
            MASKING,
            REFERENCE,
        ),
        (
            "blank precise cut (region dropped)",
            blank_cut_page,
            MaskerConfig::default(),
            Rect::new(120, 100, 160, 140),
            Rect::new(100, 80, 180, 160),
        ),
        (
            "degenerate rect (region dropped)",
            simple_page(),
            MaskerConfig::default(),
            Rect::new(70, 50, 70, 110),
            Rect::new(50, 30, 50, 130),
        ),
        (
            "out-of-canvas rect (region dropped)",
            simple_page(),
            MaskerConfig::default(),
            Rect::new(900, 900, 950, 950),
            Rect::new(880, 880, 970, 970),
        ),
    ];

    let mut fitted = 0_usize;
    let mut dropped = 0_usize;
    for (label, page, config, masking, reference) in cases {
        let canvas = canvas_of(&page);
        let cut = cut_of(&page);
        let boxes = box_mask_of(&page);

        let direct = fit_region(&canvas, &cut, &boxes, masking, reference, &config);
        let via_report = fit_region_scored(&canvas, &cut, &boxes, masking, reference, &config)
            .map(|report| report.fitment);

        assert_eq!(direct, via_report, "{label}: wrapper divergence");
        if direct.is_some() {
            fitted += 1;
        } else {
            dropped += 1;
        }
    }
    // Anti-vacuity: `None == None` is a cheap pass, so pin how many of each branch ran.
    assert_eq!(fitted, 3);
    assert_eq!(dropped, 3);
}

#[test]
fn no_rescue_is_reported_on_a_fixture_where_no_rescue_is_possible() {
    // §16.35 item 8's T1 ("zero behaviour change, rescue not wired") observed through
    // `FitReport.rescued`. Both fixtures are chosen so the assertion stays true after T5
    // wires the flag on by default, rather than being a claim T5 must invert: the uniform
    // page's ratchet pick already passes the fail-safe (item 2's P2 leaves it alone), and
    // every one of the ramp's candidates fails it (item 1's "if none qualifies, behaviour
    // is unchanged"). The rescue-eligible fixture deliberately does NOT appear here -- see
    // `q2_fallback_on_real_ladders.rs`, which asserts the policy on it without asserting
    // what T1's unwired `fit_region` paints.
    for (label, page, config) in [
        ("uniform slow", simple_page(), MaskerConfig::default()),
        ("uniform fast", simple_page(), fast()),
        ("ramp slow", ramp_page(), MaskerConfig::default()),
        ("ramp fast", ramp_page(), fast()),
    ] {
        let report = report(&page, &config, MASKING, REFERENCE)
            .unwrap_or_else(|| panic!("{label}: the region is reported"));

        assert!(!report.rescued, "{label}: T1 wires no rescue");
    }
}

#[test]
fn a_dropped_region_produces_no_report_at_all() {
    // §10.7(A)10 / §16.9 item 11: the drop decision happens before any scoring, so there
    // is no ladder to report -- `fit_region_scored` returns `None`, it does not return a
    // report with an empty vector.
    let text = [Rect::new(10, 10, 30, 30)];
    let blank_cut = page(
        common::uniform_base(PAGE_SIZE, 200, &text),
        text_raw_mask(PAGE_SIZE, &text),
        &[Rect::new(120, 100, 160, 140)],
        20,
        1.0,
    );

    assert!(report(
        &blank_cut,
        &MaskerConfig::default(),
        Rect::new(120, 100, 160, 140),
        Rect::new(100, 80, 180, 160)
    )
    .is_none());
    assert!(report(
        &simple_page(),
        &MaskerConfig::default(),
        Rect::new(70, 50, 70, 110),
        Rect::new(50, 30, 50, 130)
    )
    .is_none());
    assert!(report(
        &simple_page(),
        &MaskerConfig::default(),
        Rect::new(900, 900, 950, 950),
        Rect::new(880, 880, 970, 970)
    )
    .is_none());
}
