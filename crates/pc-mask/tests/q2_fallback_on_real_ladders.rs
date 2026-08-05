//! Task T1 (§16.35 item 8) -- the rescue policy applied to candidate ladders produced by
//! REAL pixels rather than canned numbers. FROZEN.
//!
//! Why this file exists separately from `q1_fallback_policy.rs`: every ladder there is a
//! literal array, so nothing in it establishes that a ladder the masker actually produces
//! can reach §16.35 item 1's rescue branch at all. §16.35 item 4's binding condition asks
//! for "a rescue-eligible fixture"; `banded_page` below IS that fixture, found by sweeping
//! synthetic canvases and recorded here with its full hand-traced ratchet.
//!
//! How the ladder is obtained, stated because it decides what these tests prove: the
//! deviations come from `fit_region_scored`'s own `candidate_deviations` (real candidate
//! masks over real pixels, via the frozen `grow`/`border` code that T1 does not touch),
//! and are then replayed through `select_candidate_with_fallback` and `resolve_fallback`.
//! That composition is deliberate -- it avoids re-implementing `fit_region`'s crop/
//! candidate geometry in the test, which would gate a copy instead of the original
//! (cookbook rule 12). The scorer's `median_color` is a placeholder; the policy under test
//! never reads it.
//!
//! Scope limit, deliberate: T1 wires no rescue and `[masker]
//! mask_fallback_to_lowest_deviation` does not exist until T4, so NOTHING here asserts
//! what `fit_region` paints on `banded_page`. That end-to-end assertion is T5's, and
//! writing it now would freeze a claim T5 is required to invert.

mod common;

use common::{box_mask_of, canvas_of, cut_of, page, simple_page, text_raw_mask, PAGE_SIZE};
use image::{DynamicImage, GrayImage, Luma};
use pc_config::MaskerConfig;
use pc_core::{PageData, Rect};
use pc_mask::border::{BlankMask, BorderStats};
use pc_mask::fit::{
    fit_accepted, fit_region_scored, resolve_fallback, select_candidate_with_fallback, Selection,
};

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
const CANDIDATE_COUNT: usize = 12;
const THRESHOLD: f64 = 0.1;
const MAX_DEVIATION: f64 = 15.0;

/// The rescue-eligible fixture: a uniform 200 background with one 3-pixel horizontal band
/// of 90 at `y = 54..57`, i.e. just above the masking rect's top edge at `y = 50`, plus
/// the usual 40x40 black text block. As the candidate masks grow, their border rings sweep
/// across the band, so the ladder falls steadily -- but in steps smaller than the 10%
/// `mask_improvement_threshold`, which is exactly the shape §16.35 item 1 describes.
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
        &[Rect::new(70, 50, 130, 110)],
        20,
        1.0,
    )
}

/// The `a9_*` ramp fixture of `m4_fit.rs`: no candidate can pass the fail-safe.
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

/// The real candidate ladder of `page`, and the `Selection` the policy computes from it.
fn real_ladder(page: &PageData) -> (Vec<f64>, usize, Selection) {
    let config = MaskerConfig::default();
    let report = fit_region_scored(
        &canvas_of(page),
        &cut_of(page),
        &box_mask_of(page),
        MASKING,
        REFERENCE,
        &config,
    )
    .expect("the region is reported");

    let deviations = report.candidate_deviations.clone();
    let selection = select_candidate_with_fallback(deviations.len(), false, THRESHOLD, |index| {
        Ok::<BorderStats, BlankMask>(BorderStats {
            std_deviation: deviations[index],
            median_color: [1, 2, 3],
        })
    })
    .expect("a real ladder holds no blank candidate");

    (deviations, report.greedy_index, selection)
}

#[test]
fn the_policy_and_the_report_agree_on_which_candidate_the_ratchet_chose() {
    // The cross-check the other tests in this file rest on: `fit_region_scored`'s
    // `greedy_index` (§16.35 item 7) and `select_candidate_with_fallback`'s `greedy.index`
    // are two independent computations of §10.3 step 9's pick over the same numbers, and a
    // refactor that changed one and not the other would show up here.
    for (label, page, expected_greedy) in [
        ("uniform", simple_page(), 11_usize),
        ("ramp", ramp_page(), 0),
        ("banded", banded_page(), 6),
    ] {
        let (deviations, greedy_index, selection) = real_ladder(&page);

        assert_eq!(deviations.len(), CANDIDATE_COUNT, "{label}");
        assert_eq!(greedy_index, expected_greedy, "{label}: reported greedy");
        assert_eq!(
            selection.greedy.index, expected_greedy,
            "{label}: policy greedy"
        );
        assert_eq!(
            selection.greedy.std_deviation, deviations[expected_greedy],
            "{label}"
        );
    }
}

#[test]
fn the_rescue_fires_on_a_real_candidate_ladder_and_picks_its_argmin() {
    // §16.35 item 4's required "rescue-eligible fixture", on real pixels. Hand-traced
    // ratchet over `banded_page`'s twelve deviations (values from the frozen `grow`/`border`
    // code, gated by `m2_grow.rs`/`m3_border.rs`), threshold 0.1:
    //   i=0  45.019 accepted           i=6  15.819 <= 19.286*0.9 = 17.357 -> accepted
    //   i=1  48.529 > 40.517 rejected  i=7  15.400 > 15.819*0.9 = 14.237 -> rejected
    //   i=2  22.913 <= 40.517 accepted i=8  15.013 > 14.237 rejected
    //   i=3  22.158 > 20.621 rejected  i=9  14.653 > 14.237 rejected
    //   i=4  19.286 <= 20.621 accepted i=10 14.319 > 14.237 rejected  <-- the argmin
    //   i=5  18.719 > 17.357 rejected  i=11 17.315 > 14.237 rejected
    // So the ratchet stops at index 6 with 15.819, which FAILS the 15.0 fail-safe and is
    // left unmasked today -- while index 10 at 14.319, already scored and passed over,
    // passes it.
    let (deviations, greedy_index, selection) = real_ladder(&banded_page());

    assert_eq!(greedy_index, 6);
    assert!(
        !fit_accepted(deviations[6], MAX_DEVIATION),
        "the fixture must be one the fail-safe rejects today: {:?}",
        deviations[6]
    );
    assert_eq!(selection.lowest.index, 10);
    assert!(fit_accepted(selection.lowest.std_deviation, MAX_DEVIATION));

    let rescued = resolve_fallback(&selection, MAX_DEVIATION, true);

    assert_eq!(rescued.index, 10);
    assert_eq!(rescued.std_deviation, deviations[10]);
    assert!(fit_accepted(rescued.std_deviation, MAX_DEVIATION));
    // Discrimination the canned ladders cannot provide: indices 9 and 10 BOTH pass the
    // fail-safe here, so "pick the first candidate that passes" and "pick the argmin" are
    // different answers on this fixture, and §16.35 item 1 mandates the argmin.
    assert!(fit_accepted(deviations[9], MAX_DEVIATION));
    assert!(deviations[10] < deviations[9] && deviations[9] < deviations[8]);
}

#[test]
fn switching_the_rescue_off_reproduces_the_ratchets_pick_on_the_rescue_eligible_fixture() {
    // §16.35 item 4's DEVIATION(21) control: with the flag off, the selection is the one
    // today's `select_candidate` makes -- index 6, still failing the fail-safe -- on the one
    // committed fixture where the flag provably changes the answer.
    let (deviations, _, selection) = real_ladder(&banded_page());

    let disabled = resolve_fallback(&selection, MAX_DEVIATION, false);
    let enabled = resolve_fallback(&selection, MAX_DEVIATION, true);

    assert_eq!(disabled.index, 6);
    assert_eq!(disabled.std_deviation, deviations[6]);
    assert!(!fit_accepted(disabled.std_deviation, MAX_DEVIATION));
    assert_ne!(disabled.index, enabled.index);
}

#[test]
fn the_rescue_declines_the_ramp_because_even_its_argmin_fails_the_fail_safe() {
    // §16.35 item 2's P1 on real pixels: the ramp's argmin (index 3, ~50.96) is a genuinely
    // different candidate from the ratchet's pick (index 0, ~55.73), so the rescue branch is
    // actually reached -- and it must still decline, because 50.96 > 15.0. This is the
    // fail-safe surviving the polish, which is the whole safety claim of §16.35.
    let (deviations, greedy_index, selection) = real_ladder(&ramp_page());

    assert_eq!(greedy_index, 0);
    assert_eq!(selection.lowest.index, 3);
    assert_ne!(selection.lowest.index, selection.greedy.index);
    assert!(
        deviations
            .iter()
            .all(|deviation| *deviation > MAX_DEVIATION),
        "no ramp candidate may pass the fail-safe: {deviations:?}"
    );

    let resolved = resolve_fallback(&selection, MAX_DEVIATION, true);

    assert_eq!(resolved.index, 0);
    assert_eq!(resolved.std_deviation, deviations[0]);
    assert!(!fit_accepted(resolved.std_deviation, MAX_DEVIATION));
}

#[test]
fn the_rescue_leaves_the_uniform_pages_pick_alone_although_its_argmin_is_candidate_zero() {
    // §16.35 item 2's P2 on real pixels, on the fixture that would expose the most
    // damaging plausible bug: every one of the uniform page's twelve deviations is exactly
    // 0.0, so the argmin is candidate 0 -- the THINNEST mask -- while the ratchet's
    // inclusive `<=` carries it to candidate 11, the box candidate. An implementation that
    // took the argmin unconditionally would silently shrink every already-successful mask
    // on every uniform page. The rescue must return candidate 11 with the flag either way.
    let (deviations, greedy_index, selection) = real_ladder(&simple_page());

    assert_eq!(deviations, vec![0.0; CANDIDATE_COUNT]);
    assert_eq!(greedy_index, 11);
    assert_eq!(selection.lowest.index, 0, "the argmin is the thinnest mask");

    let disabled = resolve_fallback(&selection, MAX_DEVIATION, false);
    let enabled = resolve_fallback(&selection, MAX_DEVIATION, true);

    assert_eq!(disabled.index, 11);
    assert_eq!(enabled.index, 11);
}
