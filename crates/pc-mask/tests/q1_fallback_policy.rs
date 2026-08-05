//! Task T1 (§16.35 item 8) -- the pure selection/rescue policy introduced by §16.35
//! items 1, 2, 4, 5 and 7. FROZEN.
//!
//! Every test here runs on canned deviation ladders, so the rule is observable without a
//! canvas, exactly as `m4_fit.rs` does for `select_candidate` (§16.9 item 9).
//!
//! Scope note, stated so it is not read as more than it is: T1 adds these functions and
//! *does not wire them* -- `fit_region`'s observable output is unchanged, and the
//! `[masker] mask_fallback_to_lowest_deviation` config key does not exist yet (T4). The
//! `enabled` flag is therefore a plain parameter here, and every claim below is about
//! `resolve_fallback` as a function, never about what `fit_region` paints.
//!
//! Residual gap, recorded rather than papered over: §16.35 item 4's binding condition
//! says `mask_fallback_to_lowest_deviation = false` must "reproduce upstream's exact
//! selection output on a rescue-eligible fixture". Nothing in T1 runs upstream
//! PanelCleaner, so what the `enabled = false` tests below actually pin is the weaker
//! (and T1-appropriate) claim: identical to *our* frozen pre-rescue `select_candidate`,
//! which is the artifact `m4_fit.rs` froze against upstream parity. The upstream run
//! itself belongs to T3/§16.35 item 6.

use pc_mask::border::{BlankMask, BorderStats};
use pc_mask::fit::{
    fit_accepted, resolve_fallback, select_candidate, select_candidate_with_fallback, Scored,
    Selected, Selection,
};
use std::cell::RefCell;

/// The default `[masker]` values this whole section is written against
/// (`crates/pc-config/src/profile.rs`, verified 2026-08-05 and quoted in §16.35 item 1:
/// "the default `mask_improvement_threshold = 0.1`" and item 3: "`mask_max_standard_deviation = 15`").
const THRESHOLD: f64 = 0.1;
const MAX_DEVIATION: f64 = 15.0;

fn stats(std_deviation: f64) -> Result<BorderStats, BlankMask> {
    Ok(BorderStats {
        std_deviation,
        median_color: [1, 2, 3],
    })
}

/// Score a canned ladder in candidate order, slow mode unless `fast`.
fn ladder(deviations: &[f64], fast: bool) -> Selection {
    select_candidate_with_fallback(deviations.len(), fast, THRESHOLD, |index| {
        stats(deviations[index])
    })
    .expect("no blank candidate in a canned ladder")
}

// -------------------------------------------------- what the scored ladder reports (item 7)

#[test]
fn the_scored_ladder_reports_every_candidate_and_the_ratchets_own_pick() {
    // §16.35 item 7: `Selection` carries "everything §10.3 step 9's loop learned".
    // The greedy trace is the one hand-traced in the frozen
    // `m4_fit.rs::a8_selection_requires_a_ten_percent_improvement`, which is the
    // independent oracle for this expectation:
    //   i=0: 10.0 accepted unconditionally
    //   i=1: needs <= 9.0;  9.5 rejected
    //   i=2: needs <= 9.0;  8.0 accepted
    //   i=3: needs <= 7.2;  8.0 rejected
    let selection = ladder(&[10.0, 9.5, 8.0, 8.0], false);

    assert_eq!(selection.deviations, vec![10.0, 9.5, 8.0, 8.0]);
    assert_eq!(selection.greedy.index, 2);
    assert_eq!(selection.greedy.std_deviation, 8.0);
    assert_eq!(selection.greedy.median_color, [1, 2, 3]);
}

#[test]
fn the_lowest_field_is_the_argmin_and_can_differ_from_the_ratchets_pick() {
    // §16.35 item 1's whole premise, on the item's own worked shape extended by two
    // candidates. Hand-traced at threshold 0.1:
    //   i=0: 16.0 accepted; every later candidate needs <= 14.4 and none reaches it
    //        (15.5, 15.4, 14.9, 15.0) -> the ratchet's final pick is index 0 at 16.0,
    //        even though it already scored and passed over 14.9 at index 3.
    let selection = ladder(&[16.0, 15.5, 15.4, 14.9, 15.0], false);

    assert_eq!(selection.greedy.index, 0);
    assert_eq!(selection.greedy.std_deviation, 16.0);
    assert_eq!(selection.lowest.index, 3);
    assert_eq!(selection.lowest.std_deviation, 14.9);
    assert_eq!(selection.deviations, vec![16.0, 15.5, 15.4, 14.9, 15.0]);
}

#[test]
fn fast_mode_reports_only_the_candidates_it_actually_scored() {
    // §16.35 item 7 read against §16.9 item 10: the rescue may only consider candidates
    // "already scored in §10.3 step 9" (item 1), so a fast-mode run that broke early must
    // report a SHORTER vector -- three entries, not four. The greedy index 2 and the call
    // count 3 are the frozen trace of
    // `m4_fit.rs::a11_fast_mode_keeps_scoring_while_deviations_are_nonzero`.
    let deviations = [3.0, 2.0, 0.0, 9.0];
    let calls = RefCell::new(0_usize);

    let selection = select_candidate_with_fallback(deviations.len(), true, THRESHOLD, |index| {
        *calls.borrow_mut() += 1;
        stats(deviations[index])
    })
    .expect("no blank candidate");

    assert_eq!(*calls.borrow(), 3);
    assert_eq!(selection.deviations, vec![3.0, 2.0, 0.0]);
    assert_eq!(selection.greedy.index, 2);
    assert_eq!(selection.lowest.index, 2);
    assert_eq!(selection.lowest.std_deviation, 0.0);
}

#[test]
fn the_greedy_index_always_indexes_the_reported_deviation_vector() {
    // The consistency `fit_region_scored`'s `greedy_index`/`candidate_deviations` pair
    // depends on (§16.35 item 7). Both modes, including one that breaks early.
    for (deviations, fast) in [
        (vec![10.0, 9.5, 8.0, 8.0], false),
        (vec![3.0, 2.0, 0.0, 9.0], true),
        (vec![16.0, 15.5, 15.4, 14.9, 15.0], false),
        (vec![7.0], false),
    ] {
        let selection = ladder(&deviations, fast);

        assert!(
            selection.greedy.index < selection.deviations.len(),
            "greedy index {} is out of range for {:?}",
            selection.greedy.index,
            selection.deviations
        );
        assert_eq!(
            selection.deviations[selection.greedy.index],
            selection.greedy.std_deviation
        );
        assert_eq!(
            selection.deviations[selection.lowest.index],
            selection.lowest.std_deviation
        );
    }
}

#[test]
fn a_blank_candidate_abandons_the_region_before_any_rescue_can_apply() {
    // §10.3 step 8 / §16.9 item 8, unchanged by §16.35: the scorer's `Err(BlankMask)`
    // propagates out of the new entry point exactly as it does out of `select_candidate`,
    // so there is no `Selection` for a rescue to be computed from.
    let result = select_candidate_with_fallback(3, false, THRESHOLD, |index| {
        if index == 1 {
            Err(BlankMask)
        } else {
            stats(1.0)
        }
    });

    assert_eq!(result, Err(BlankMask));
}

// ----------------------------------------------- the wrapper contract (item 7, T1's gate)

#[test]
fn select_candidate_agrees_with_the_scored_ladders_greedy_field_on_every_shape() {
    // §16.35 item 7: "`select_candidate` becomes a thin wrapper over
    // `select_candidate_with_fallback` (`.map(|s| s.greedy.into())`)". This is T1's
    // zero-behaviour-change condition stated at the policy level; the end-to-end half is
    // the frozen `m4_fit.rs` suite staying green.
    for (deviations, fast) in [
        (vec![10.0, 9.5, 8.0, 8.0], false),
        (vec![0.0, 0.0], false),
        (vec![0.0, 0.001, 5.0], false),
        (vec![3.0, 2.0, 0.0, 9.0], true),
        (vec![0.0, 1.0, 2.0], true),
        (vec![16.0, 14.5, 14.5], false),
        (vec![16.0, 15.5, 15.4, 14.9, 15.0], false),
    ] {
        let expected = select_candidate(deviations.len(), fast, THRESHOLD, |index| {
            stats(deviations[index])
        })
        .expect("no blank candidate");
        let selection = ladder(&deviations, fast);

        assert_eq!(
            Selected::from(selection.greedy),
            expected,
            "wrapper divergence on {deviations:?} (fast = {fast})"
        );
    }
}

#[test]
fn a_scored_candidate_converts_into_the_frozen_selected_type_field_for_field() {
    // §16.35 item 7's `.into()`: `Selected` keeps its frozen three fields
    // (`m4_fit.rs:55-57` reads all three), so the conversion must be lossless.
    let scored = Scored {
        index: 7,
        std_deviation: 12.25,
        median_color: [9, 8, 7],
    };

    let selected = Selected::from(scored);

    assert_eq!(selected.index, 7);
    assert_eq!(selected.std_deviation, 12.25);
    assert_eq!(selected.median_color, [9, 8, 7]);
}

// ----------------------------------------------------- the fail-safe predicate (item 7)

#[test]
fn fit_accepted_is_inclusive_at_the_maximum_and_exclusive_just_above_it() {
    // §16.35 item 7 names the predicate "so it has exactly one call site"; the predicate
    // itself is `fit.rs:188`'s existing `selected.std_deviation <= mask_max_standard_deviation`,
    // which must not change sense or strictness in the refactor.
    assert!(fit_accepted(0.0, MAX_DEVIATION));
    assert!(fit_accepted(14.999, MAX_DEVIATION));
    assert!(
        fit_accepted(MAX_DEVIATION, MAX_DEVIATION),
        "the comparison is <=, not <"
    );
    assert!(!fit_accepted(15.000000000000002, MAX_DEVIATION));
    assert!(!fit_accepted(16.0, MAX_DEVIATION));
}

// ---------------------------------------------------------- the rescue policy (items 1, 2, 5)

#[test]
fn the_rescue_paints_the_lowest_candidate_only_when_the_ratchet_would_be_discarded() {
    // §16.35 item 1, on its own worked example: greedy 16.0 fails the 15.0 fail-safe, the
    // already-scored 14.9 at index 3 passes it, so the rescue paints index 3.
    let selection = ladder(&[16.0, 15.5, 15.4, 14.9, 15.0], false);

    let rescued = resolve_fallback(&selection, MAX_DEVIATION, true);

    assert_eq!(rescued.index, 3);
    assert_eq!(rescued.std_deviation, 14.9);
    assert!(fit_accepted(rescued.std_deviation, MAX_DEVIATION));
}

#[test]
fn switching_the_rescue_off_returns_the_ratchets_pick_on_the_very_same_ladder() {
    // §16.35 item 4's DEVIATION(21) control: `enabled = false` "reproduces today's
    // `select_candidate` output bit for bit" (item 7) -- on the one ladder where the two
    // branches provably differ, so this cannot pass by both branches being the same.
    let selection = ladder(&[16.0, 15.5, 15.4, 14.9, 15.0], false);

    let disabled = resolve_fallback(&selection, MAX_DEVIATION, false);
    let enabled = resolve_fallback(&selection, MAX_DEVIATION, true);

    assert_eq!(disabled.index, 0);
    assert_eq!(disabled.std_deviation, 16.0);
    assert_eq!(Selected::from(disabled), Selected::from(selection.greedy));
    assert_ne!(
        disabled.index, enabled.index,
        "the fixture must be rescue-eligible or this test proves nothing"
    );
}

#[test]
fn the_rescue_leaves_an_acceptable_ratchet_pick_alone_even_when_a_lower_one_was_discarded() {
    // §16.35 item 2 (P2): "every region masked today is unaffected -- the rescue only
    // fires on the branch that currently returns `None`". Hand-traced: i=1 needs
    // <= 12.6 and 13.0 is rejected, so greedy is index 0 at 14.0, which already passes
    // the 15.0 fail-safe; the discarded 13.0 must NOT replace it.
    let selection = ladder(&[14.0, 13.0], false);

    let resolved = resolve_fallback(&selection, MAX_DEVIATION, true);

    assert_eq!(selection.greedy.index, 0);
    assert_eq!(
        selection.lowest.index, 1,
        "a strictly lower candidate exists"
    );
    assert_eq!(resolved.index, 0);
    assert_eq!(resolved.std_deviation, 14.0);
}

#[test]
fn ties_on_the_lowest_deviation_resolve_to_the_lowest_index() {
    // §16.35 item 5 (Fable): "When two or more already-scored candidates share the
    // exact-equal lowest `std_deviation`, the rescue picks the **lowest index**", and
    // "a frozen test with two equal passing deviations (e.g. `[16.0, 14.5, 14.5]`,
    // threshold `15.0`) must assert the lower index wins".
    //
    // Hand-traced greedy: i=0 accepts 16.0; i=1 and i=2 each need <= 14.4 and 14.5 misses,
    // so greedy is index 0 and fails the fail-safe. Both 14.5 candidates pass it; index 1
    // wins, and painting index 2 instead would be the overruled direction.
    let selection = ladder(&[16.0, 14.5, 14.5], false);

    let rescued = resolve_fallback(&selection, MAX_DEVIATION, true);

    assert_eq!(selection.lowest.index, 1);
    assert_eq!(rescued.index, 1);
    assert_eq!(rescued.std_deviation, 14.5);
}

#[test]
fn a_tie_at_the_very_end_of_the_ladder_still_resolves_to_the_earlier_candidate() {
    // The same ruling with the two tied minima separated by a higher candidate, so a
    // "last minimum" implementation cannot slip through by coincidence of ordering.
    // Hand-traced: index 0 accepts 16.0 and every later candidate needs <= 14.4, which
    // 15.9, 14.5, 15.8 and 14.5 all miss, so greedy is index 0 and fails the fail-safe;
    // the tied 14.5 minima at indices 2 and 4 both pass it, and index 2 must win.
    let selection = ladder(&[16.0, 15.9, 14.5, 15.8, 14.5], false);

    let rescued = resolve_fallback(&selection, MAX_DEVIATION, true);

    assert_eq!(selection.greedy.index, 0, "14.5 > 16.0 * 0.9 == 14.4");
    assert_eq!(selection.lowest.index, 2);
    assert_eq!(rescued.index, 2);
    assert_eq!(rescued.std_deviation, 14.5);
}

#[test]
fn the_rescue_cannot_paint_a_candidate_above_the_maximum() {
    // §16.35 item 2 (P1): "every painted mask still satisfies
    // `std_deviation <= mask_max_standard_deviation`". Hand-traced: i=1 needs <= 18.0 and
    // 18.0 is accepted inclusively, so greedy is index 1 at 18.0; the argmin is that same
    // 18.0, which fails the fail-safe, so nothing qualifies and behaviour is unchanged
    // (`mask: None`, per item 1's last clause).
    let selection = ladder(&[20.0, 18.0, 19.0], false);

    let resolved = resolve_fallback(&selection, MAX_DEVIATION, true);

    assert_eq!(selection.greedy.index, 1);
    assert_eq!(resolved.index, 1);
    assert_eq!(resolved.std_deviation, 18.0);
    assert!(!fit_accepted(resolved.std_deviation, MAX_DEVIATION));
}

#[test]
fn a_fast_mode_run_that_broke_on_a_zero_deviation_can_never_rescue() {
    // §16.35 item 5's parenthetical, which the tie-break ruling leans on: "a scored `0.0`
    // is always accepted by the greedy loop and always passes the fail-safe, since
    // `mask_max_standard_deviation > 0` is a validated invariant, so the rescue never runs
    // on a region whose candidate list contains one". Here the break also truncates the
    // ladder at two entries, so the 99.0 is never even scored.
    let deviations = [20.0, 0.0, 99.0];

    let selection = ladder(&deviations, true);
    let resolved = resolve_fallback(&selection, MAX_DEVIATION, true);

    assert_eq!(selection.deviations, vec![20.0, 0.0]);
    assert_eq!(selection.greedy.index, 1);
    assert_eq!(selection.greedy.std_deviation, 0.0);
    assert_eq!(resolved.index, 1);
    assert!(fit_accepted(resolved.std_deviation, MAX_DEVIATION));
}

#[test]
fn the_rescue_holds_p1_and_p2_across_every_ladder_in_a_six_by_three_grid() {
    // §16.35 item 2's P1 and P2 as an exhaustive property over 216 ladders (6 values x 3
    // positions), at the default threshold 0.1 and maximum 15.0.
    //
    // The two totals below are ANTI-VACUITY LITERALS: they were computed by an independent
    // oracle (a standalone script implementing §16.35 items 1 and 5 directly from the spec
    // text, run 2026-08-05, `scratchpad`-local, not from this crate's code), and they
    // cannot be recomputed from anything the assertions here observe. Without them the
    // property below would pass on an implementation that never rescues anything.
    const VALUES: [f64; 6] = [0.0, 10.0, 14.5, 15.0, 16.0, 20.0];
    const EXPECTED_GREEDY_FAILURES: usize = 22;
    const EXPECTED_RESCUES: usize = 14;

    let mut ladders = 0_usize;
    let mut greedy_failures = 0_usize;
    let mut rescues = 0_usize;

    for a in VALUES {
        for b in VALUES {
            for c in VALUES {
                let deviations = [a, b, c];
                ladders += 1;
                let selection = ladder(&deviations, false);
                let greedy_ok = fit_accepted(selection.greedy.std_deviation, MAX_DEVIATION);
                if !greedy_ok {
                    greedy_failures += 1;
                }

                let disabled = resolve_fallback(&selection, MAX_DEVIATION, false);
                assert_eq!(
                    Selected::from(disabled),
                    Selected::from(selection.greedy),
                    "enabled = false must be the ratchet's pick on {deviations:?}"
                );

                let resolved = resolve_fallback(&selection, MAX_DEVIATION, true);
                if resolved.index != selection.greedy.index {
                    rescues += 1;
                    // P1: nothing above the maximum is ever painted.
                    assert!(
                        fit_accepted(resolved.std_deviation, MAX_DEVIATION),
                        "P1 violated on {deviations:?}: rescued {resolved:?}"
                    );
                    // P2: the rescue only fires where today's code returns None.
                    assert!(
                        !greedy_ok,
                        "P2 violated on {deviations:?}: rescued an already-acceptable pick"
                    );
                    // Item 1: the rescue picks from the already-scored candidates.
                    assert_eq!(
                        selection.deviations[resolved.index], resolved.std_deviation,
                        "the rescued candidate must be one of the scored ones"
                    );
                    // Item 5: lowest index among the exact-equal minima.
                    let minimum = selection
                        .deviations
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min);
                    let first_minimum = selection
                        .deviations
                        .iter()
                        .position(|deviation| *deviation == minimum)
                        .expect("the minimum is one of the deviations");
                    assert_eq!(
                        resolved.index, first_minimum,
                        "tie-break must be lowest index on {deviations:?}"
                    );
                } else if !greedy_ok {
                    // The only way a failing greedy pick survives is that nothing
                    // qualifies (item 1's "if none qualifies, behaviour is unchanged").
                    let minimum = selection
                        .deviations
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min);
                    assert!(
                        !fit_accepted(minimum, MAX_DEVIATION)
                            || minimum == selection.greedy.std_deviation,
                        "an acceptable lower candidate existed on {deviations:?} but was not used"
                    );
                }
            }
        }
    }

    assert_eq!(ladders, 216);
    assert_eq!(greedy_failures, EXPECTED_GREEDY_FAILURES);
    assert_eq!(rescues, EXPECTED_RESCUES);
}
