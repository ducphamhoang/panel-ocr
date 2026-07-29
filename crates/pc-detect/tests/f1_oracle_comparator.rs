//! spec §16.20 item 3(b) + item 10, as ratified by §16.24 items 4–8 — the comparator's own
//! negative controls.
//!
//! Every input is constructed here. These tests are fully meaningful in a checkout with no
//! recorded detector fixture, which is the point: §16.20 item 10 requires the comparator to "ship
//! with its own negative controls", and controls committed under `tests/fixtures/` would need
//! their own PROVENANCE entry and an edit to `recorded_provenance.rs:14-27`'s frozen constants —
//! for artifacts whose whole purpose is to be wrong. **No real-page gate test exists yet**; it
//! lands only in the atomic recording commit (§16.24 item 6).

use pc_core::{Language, Rect};
use pc_detect::oracle::{
    self, ComparisonReport, DerivedAccounting, Divergence, DivergenceClass, Edge, Expectations,
    ExpectedPair, IdentityBranch, Mechanism, OracleBlock, OracleCoverage, OursSide, Side, Totals,
    UnmatchedEntry, UpstreamOracle,
};
use pc_detect::RawBlock;

const FRAME_SCALE: f64 = 0.617;
const FRAME_SIZE: (u32, u32) = (1024, 1434);

// ── the synthetic pair (full derivation in the plan's section 4) ──────────────

fn ours_truth() -> Vec<RawBlock> {
    vec![
        RawBlock {
            rect: Rect::new(100, 200, 160, 240),
            class_index: 0,
            confidence: 0.775,
        }, // A
        RawBlock {
            rect: Rect::new(300, 400, 366, 441),
            class_index: 1,
            confidence: 0.512,
        }, // B
        RawBlock {
            rect: Rect::new(500, 600, 617, 672),
            class_index: 0,
            confidence: 0.704,
        }, // C
        RawBlock {
            rect: Rect::new(700, 800, 760, 830),
            class_index: 1,
            confidence: 0.437,
        }, // D
        RawBlock {
            rect: Rect::new(900, 1000, 966, 1041),
            class_index: 0,
            confidence: 0.564,
        }, // E
    ]
}

fn oracle_block(
    xyxy: [i32; 4],
    lines: Vec<Vec<[i32; 2]>>,
    confidence: f32,
    language: Language,
) -> OracleBlock {
    OracleBlock {
        xyxy,
        lines,
        confidence: Some(confidence),
        language: Some(language),
        base_xyxy_pretruncation: None,
    }
}

/// A line-less oracle block with no optional fields recorded — the shape the degenerate branch and
/// the NO-ORACLE rows are written against.
fn bare_block(xyxy: [i32; 4]) -> OracleBlock {
    OracleBlock {
        xyxy,
        lines: vec![],
        confidence: None,
        language: None,
        base_xyxy_pretruncation: None,
    }
}

fn upstream_truth() -> UpstreamOracle {
    UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![
            // A — line-informed. bbox(lines) = (104,206,150,232), strictly inside ours, so the
            // union is ours: (100,200,160,240).
            oracle_block(
                [100, 200, 160, 240],
                vec![vec![[104, 206], [150, 206], [150, 232], [104, 232]]],
                0.775,
                Language::English,
            ),
            // B — line-less: identity degenerates to `upstream.xyxy == ours.rect`.
            oracle_block([300, 400, 366, 441], vec![], 0.512, Language::Japanese),
            // C — line-informed. bbox(lines) = (505,610,600,660); union with ours = ours.
            oracle_block(
                [500, 600, 617, 672],
                vec![vec![[505, 610], [600, 610], [600, 660], [505, 660]]],
                0.704,
                Language::English,
            ),
            // D — line-less.
            oracle_block([700, 800, 760, 830], vec![], 0.437, Language::Japanese),
            // E — line-less.
            oracle_block([900, 1000, 966, 1041], vec![], 0.564, Language::English),
        ],
        pre_filter_blocks: vec![],
    }
}

fn ours_side(blocks: &[RawBlock]) -> OursSide<'_> {
    OursSide {
        blocks,
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
    }
}

/// The signed pairing for the unperturbed pair: index-aligned, branches per the data above.
fn expectations_truth() -> Expectations {
    use IdentityBranch::{LineInformed, LineLess};
    Expectations {
        pairs: vec![
            ExpectedPair {
                ours: 0,
                upstream: 0,
                branch: LineInformed,
            },
            ExpectedPair {
                ours: 1,
                upstream: 1,
                branch: LineLess,
            },
            ExpectedPair {
                ours: 2,
                upstream: 2,
                branch: LineInformed,
            },
            ExpectedPair {
                ours: 3,
                upstream: 3,
                branch: LineLess,
            },
            ExpectedPair {
                ours: 4,
                upstream: 4,
                branch: LineLess,
            },
        ],
        unmatched_ours: vec![],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 5,
            ours_total: 5,
            upstream_total: 5,
        },
    }
}

/// §16.20 item 10's three faults applied to the truth pair, with the pairing **re-authored for
/// the perturbed artifacts** — which is what a signed pairing means: the author sees 4 of ours
/// against 5 upstream and writes the table for what is there.
fn negative_control() -> (Vec<RawBlock>, UpstreamOracle, Expectations) {
    use IdentityBranch::{LineInformed, LineLess};

    let mut ours = ours_truth();
    ours.pop(); // fault 3: block E deleted from OUR side.

    let mut upstream = upstream_truth();
    upstream.blocks[2].xyxy[2] += 1; // fault 1: 1 px outward on C's x2.
    upstream.blocks[3].confidence = Some(0.438); // fault 2: +0.001 on D.

    let expectations = Expectations {
        pairs: vec![
            ExpectedPair {
                ours: 0,
                upstream: 0,
                branch: LineInformed,
            },
            ExpectedPair {
                ours: 1,
                upstream: 1,
                branch: LineLess,
            },
            ExpectedPair {
                ours: 2,
                upstream: 2,
                branch: LineInformed,
            },
            ExpectedPair {
                ours: 3,
                upstream: 3,
                branch: LineLess,
            },
        ],
        unmatched_ours: vec![],
        // Upstream block 4 has no counterpart and no legitimate mechanism, so the honest authored
        // entry is `Open` — which blocks (§16.24 item 4).
        unmatched_upstream: vec![UnmatchedEntry {
            index: 4,
            mechanism: Mechanism::Open,
        }],
        totals: Totals {
            pairs: 4,
            ours_total: 4,
            upstream_total: 5,
        },
    };
    (ours, upstream, expectations)
}

// ── the convention, hand-computed ────────────────────────────────────────────

#[test]
// spec §2.1 + §16.24 item 6: `rect_to_xyxy` is isolated so §2.1's deliberately mixed convention
// has exactly one home. It must be the IDENTITY: upstream slices `mask[by1:by2, bx1:bx2]`
// (`textblock.py:485-490`) with Python's exclusive stop, and §2.1 makes our `x2`/`y2` exclusive
// for cropping/rasterising, so no ±1 is correct.
//
// What would break it: any "helpful" ±1. The values are hand-computed from a 60x40 box at
// (100,200) — width 160-100 = 60 and height 240-200 = 40 are asserted alongside, so an
// inclusive-convention conversion (which would make the box 61x41) fails here rather than
// silently shifting every recorded coordinate by a pixel. `Rect::contains` is asserted in the
// same test purely to keep the mixed convention visible at the one site that depends on it.
fn rect_to_xyxy_is_the_identity_and_keeps_the_exclusive_convention() {
    let rect = Rect::new(100, 200, 160, 240);

    assert_eq!(oracle::rect_to_xyxy(rect), [100, 200, 160, 240]);
    assert_eq!(rect.width(), 60);
    assert_eq!(rect.height(), 40);

    // Round trip, both directions, on a box whose coordinates are all distinct.
    assert_eq!(oracle::xyxy_to_rect([1, 2, 3, 4]), Rect::new(1, 2, 3, 4));
    assert_eq!(
        oracle::rect_to_xyxy(oracle::xyxy_to_rect([1, 2, 3, 4])),
        [1, 2, 3, 4]
    );

    // §2.1's other half, stated where it can mislead: `x2` is INCLUSIVE for `contains`, so the
    // exclusive reading above is a cropping/rasterising fact and not a general one.
    assert!(
        rect.contains((160, 240)),
        "§2.1: contains() is inclusive on x2/y2"
    );
}

// ── controls, in both directions ─────────────────────────────────────────────

#[test]
// spec §16.20 item 3(b) — the POSITIVE control, and the precondition for every "and no others"
// assertion below: if the unperturbed pair produced any divergence, the exact-vector assertions
// would be unfalsifiable (they would pass against a comparator that always emitted that row).
//
// Also asserts the branch ACTUALLY taken per pair (§16.24 item 6), not merely that no branch
// mismatch was reported — a comparator that ignored `lines` entirely would report `LineLess`
// everywhere and be caught here.
fn the_unperturbed_pair_produces_no_divergences_and_compares_five_pairs() {
    let ours = ours_truth();
    let report = oracle::compare(ours_side(&ours), &upstream_truth(), &expectations_truth());

    assert_eq!(report.divergences, Vec::new());
    assert_eq!(report.pairs_compared, 5);
    assert_eq!(
        report.branches,
        vec![
            (0, 0, IdentityBranch::LineInformed),
            (1, 1, IdentityBranch::LineLess),
            (2, 2, IdentityBranch::LineInformed),
            (3, 3, IdentityBranch::LineLess),
            (4, 4, IdentityBranch::LineLess),
        ]
    );

    // §16.24 item 18(h)(ii): `residual_full` is all-zero for green, on every pair. Asserted here
    // rather than only where it fails, so "green" has a positive definition.
    assert_eq!(report.residuals.len(), 5);
    for residual in &report.residuals {
        assert_eq!(
            residual.residual_full,
            [0, 0, 0, 0],
            "pair ({}, {}) must satisfy the identity exactly",
            residual.ours_index,
            residual.upstream_index
        );
    }
}

#[test]
// spec §16.20 item 10, verbatim: "a synthetic artifact pair with one box perturbed by 1 px, one
// confidence by 0.001, and one box deleted, asserting the comparator reports exactly those three
// divergences and no others" — read per §16.24 item 7 as a MULTISET ACROSS CLASSES: one gating
// geometry row, one DIAGNOSTIC confidence row, one gating unmatched row. A control expecting
// three *gating* failures would contradict items 3(c)/(d) and item 5's 0.079–0.089 noise floor.
//
// The assertion is on the whole ordered `Vec<Divergence>`, so it fails if the comparator emits a
// fourth row, drops one of the three, or reorders them. Every field is an exactly-copied integer
// or f32 literal, so `assert_eq!` is exact. Expected values derived in the plan's section 4;
// nothing here was read out of an implementation's output (cookbook rule 7).
fn the_three_injected_faults_are_reported_exactly() {
    let (ours, upstream, expectations) = negative_control();
    let report = oracle::compare(ours_side(&ours), &upstream, &expectations);

    assert_eq!(
        report.divergences,
        vec![
            Divergence::GeometryIdentity {
                ours: Rect::new(500, 600, 617, 672),
                expected: Rect::new(500, 600, 617, 672),
                upstream: Rect::new(500, 600, 618, 672),
            },
            Divergence::ConfidenceDelta {
                ours: Rect::new(700, 800, 760, 830),
                ours_confidence: 0.437,
                upstream_confidence: 0.438,
            },
            Divergence::OpenMechanism {
                side: Side::Upstream,
                index: 4
            },
        ]
    );
    assert_eq!(report.pairs_compared, 4);

    // §16.20 item 3(d): confidence is DIAGNOSTIC and must never gate, so exactly two of the three
    // faults may fail CI. Asserting the split, not just the total, is what pins item 3(d).
    assert_eq!(report.gating().len(), 2, "gating: {:?}", report.gating());
    assert_eq!(report.diagnostics().len(), 1);
    assert!(report
        .gating()
        .iter()
        .all(|row| !matches!(row, Divergence::ConfidenceDelta { .. })));

    // No accounting row: the two equations close. 4 + 0 + 1 = 5 upstream; 4 + 0 = 4 ours. This is
    // the plan's §1.3 reading, and it is what keeps the third fault to ONE row.
    assert!(!report
        .divergences
        .iter()
        .any(|row| matches!(row, Divergence::AccountingMismatch { .. })));
}

#[test]
// spec §16.20 item 10 — each fault in isolation, so the three-fault assertion above cannot pass
// by coincidence (a comparator emitting a fixed three-row report would fail here). Each input
// yields exactly ONE row, which also proves the faults do not interact.
fn each_fault_alone_produces_exactly_its_own_divergence() {
    let ours = ours_truth();

    // Fault 1: 1 px outward on C's x2. Under signed pairing the pair still forms — the table says
    // so — which is why this is one geometry row and not an unmatched pair on both sides.
    let mut upstream = upstream_truth();
    upstream.blocks[2].xyxy[2] += 1;
    assert_eq!(
        oracle::compare(ours_side(&ours), &upstream, &expectations_truth()).divergences,
        vec![Divergence::GeometryIdentity {
            ours: Rect::new(500, 600, 617, 672),
            expected: Rect::new(500, 600, 617, 672),
            upstream: Rect::new(500, 600, 618, 672),
        }]
    );

    // Fault 2: +0.001 on D's confidence — one LSB of the 3-dp persisted field (§8.3 step 4's
    // `np.round(..., 3)`), i.e. the smallest representable perturbation.
    let mut upstream = upstream_truth();
    upstream.blocks[3].confidence = Some(0.438);
    assert_eq!(
        oracle::compare(ours_side(&ours), &upstream, &expectations_truth()).divergences,
        vec![Divergence::ConfidenceDelta {
            ours: Rect::new(700, 800, 760, 830),
            ours_confidence: 0.437,
            upstream_confidence: 0.438,
        }]
    );

    // Fault 3: E deleted from our side.
    let (short_ours, _, expectations) = negative_control();
    assert_eq!(
        oracle::compare(ours_side(&short_ours), &upstream_truth(), &expectations).divergences,
        vec![Divergence::OpenMechanism {
            side: Side::Upstream,
            index: 4
        }]
    );
}

#[test]
// Cookbook rule 13's iteration 4→5: "coverage ran one direction only." §16.20 item 10 names only
// the deletion from OUR side, so a comparator that walked upstream's blocks and never the
// leftovers of ours would pass every test above. This is the missing direction, and it is where a
// false positive of ours would appear.
fn an_extra_block_on_our_side_is_reported_too() {
    let mut ours = ours_truth();
    ours.push(RawBlock {
        rect: Rect::new(50, 50, 90, 90),
        class_index: 0,
        confidence: 0.9,
    });

    let mut expectations = expectations_truth();
    expectations.unmatched_ours = vec![UnmatchedEntry {
        index: 5,
        mechanism: Mechanism::Open,
    }];
    expectations.totals.ours_total = 6;

    assert_eq!(
        oracle::compare(ours_side(&ours), &upstream_truth(), &expectations).divergences,
        vec![Divergence::OpenMechanism {
            side: Side::Ours,
            index: 5
        }]
    );
}

// ── the identity's two branches ──────────────────────────────────────────────

#[test]
// spec §16.20 item 3: "The identity must handle a line-less upstream block, where
// `bbox(upstream.lines)` is undefined and the identity degenerates to `upstream.xyxy ==
// ours.rect` — v1 synthesizes no lines, and upstream's line-less branch is exactly where §14.17's
// filter divergence lives, so this case is not hypothetical."
//
// Both directions: the degenerate identity HOLDS for an exactly-equal rect and FAILS for a 1-px
// one, with no line union available to absorb the difference. The box is §16.20 item 9's measured
// one, so the coordinates are real.
fn the_line_less_identity_degenerates_to_plain_equality() {
    let ours = vec![RawBlock {
        rect: Rect::new(438, 1407, 498, 1446),
        class_index: 0,
        confidence: 0.5,
    }];
    let expectations = Expectations {
        pairs: vec![ExpectedPair {
            ours: 0,
            upstream: 0,
            branch: IdentityBranch::LineLess,
        }],
        unmatched_ours: vec![],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 1,
            ours_total: 1,
            upstream_total: 1,
        },
    };
    let frame = |blocks: Vec<OracleBlock>| UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks,
        pre_filter_blocks: vec![],
    };

    let exact = frame(vec![bare_block([438, 1407, 498, 1446])]);
    let report = oracle::compare(ours_side(&ours), &exact, &expectations);
    assert_eq!(report.pairs_compared, 1);
    assert!(report.gating().is_empty(), "{:?}", report.gating());
    // §16.25 items 2 and 4: with neither field recorded there is no DIVERGENCE — nothing diverges,
    // one side simply has no data. The absence is an artifact property and is reported on the third
    // channel, unconditionally and once per field rather than once per block. Equal in strength to
    // the two rows this replaces: same content, no longer keyed by our rect.
    assert_eq!(report.divergences, Vec::new());
    assert_eq!(report.coverage.confidence, OracleCoverage::Absent);
    assert_eq!(report.coverage.language, OracleCoverage::Absent);

    let off_by_one = frame(vec![bare_block([438, 1407, 499, 1446])]);
    let report = oracle::compare(ours_side(&ours), &off_by_one, &expectations);
    assert!(report.divergences.contains(&Divergence::GeometryIdentity {
        ours: Rect::new(438, 1407, 498, 1446),
        expected: Rect::new(438, 1407, 498, 1446),
        upstream: Rect::new(438, 1407, 499, 1446),
    }));
}

#[test]
// spec §16.20 item 3(b) — the identity is `upstream.xyxy == bbox(ours.rect ∪
// bbox(upstream.lines))`, NOT `upstream.xyxy == ours.rect`. This is the one case proving the union
// genuinely WIDENS the box rather than being a no-op: §16.20 item 12's measured pair, ours
// `(29,105,86,261)` against upstream `[29,105,90,261]`, "the trailing-edge gap being upstream's
// line union". A comparator ignoring `lines` would report this as a divergence and the real gate
// would be permanently red on real data.
//
// Falsifiable in the other direction too: the same pair with the line polygon removed must FAIL,
// since then nothing explains upstream's larger x2.
fn the_line_union_widens_the_box_and_satisfies_the_identity() {
    let ours = vec![RawBlock {
        rect: Rect::new(29, 105, 86, 261),
        class_index: 0,
        confidence: 0.564,
    }];
    let expectations = Expectations {
        pairs: vec![ExpectedPair {
            ours: 0,
            upstream: 0,
            branch: IdentityBranch::LineInformed,
        }],
        unmatched_ours: vec![],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 1,
            ours_total: 1,
            upstream_total: 1,
        },
    };
    let frame = |blocks: Vec<OracleBlock>| UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks,
        pre_filter_blocks: vec![],
    };

    // bbox(lines) = (33,110,90,250); union with ours (29,105,86,261) = (29,105,90,261) = upstream.
    let with_lines = frame(vec![OracleBlock {
        xyxy: [29, 105, 90, 261],
        lines: vec![vec![[33, 110], [90, 110], [90, 250], [33, 250]]],
        confidence: None,
        language: None,
        base_xyxy_pretruncation: None,
    }]);
    let report = oracle::compare(ours_side(&ours), &with_lines, &expectations);
    assert!(
        !report
            .divergences
            .iter()
            .any(|row| matches!(row, Divergence::GeometryIdentity { .. })),
        "the line union must satisfy the identity, not diverge: {:?}",
        report.divergences
    );
    assert_eq!(report.branches, vec![(0, 0, IdentityBranch::LineInformed)]);

    // Same geometry, no lines: now nothing explains x2 = 90, so the identity must fail. Without
    // this half, a comparator that always returned "identity holds" would pass the first half.
    let without_lines = frame(vec![bare_block([29, 105, 90, 261])]);
    let mut line_less = expectations.clone();
    line_less.pairs[0].branch = IdentityBranch::LineLess;
    assert!(
        oracle::compare(ours_side(&ours), &without_lines, &line_less)
            .divergences
            .contains(&Divergence::GeometryIdentity {
                ours: Rect::new(29, 105, 86, 261),
                expected: Rect::new(29, 105, 86, 261),
                upstream: Rect::new(29, 105, 90, 261),
            })
    );
}

#[test]
// spec §16.24 item 6: the branch is "returned and asserted per pair, not merely taken — otherwise
// a bug treating empty `lines` as `bbox = (0,0,0,0)` and unioning it would corrupt every line-ful
// pair while line-less pairs still passed."
//
// Two directions. (a) A declared branch that contradicts the data is reported, so the authored
// table cannot drift from the artifact. (b) A list of EMPTY polygons is `LineLess`: the
// `(0,0,0,0)` bug this guards would drag `x1`/`y1` to 0 on every box, so the assertion is that
// the reconstructed rect is unchanged rather than merely that no error was raised.
fn the_declared_identity_branch_is_verified_and_an_empty_polygon_list_is_line_less() {
    let ours = ours_truth();

    // (a) declare LineLess for a block that has lines.
    let mut expectations = expectations_truth();
    expectations.pairs[0].branch = IdentityBranch::LineLess;
    let report = oracle::compare(ours_side(&ours), &upstream_truth(), &expectations);
    assert_eq!(
        report.divergences,
        vec![Divergence::IdentityBranchMismatch {
            ours: Rect::new(100, 200, 160, 240),
            declared: IdentityBranch::LineLess,
            actual: IdentityBranch::LineInformed,
        }]
    );

    // (b) a polygon list containing only empty polygons has no points at all.
    let degenerate = OracleBlock {
        xyxy: [100, 200, 160, 240],
        lines: vec![vec![], vec![]],
        confidence: Some(0.775),
        language: Some(Language::English),
        base_xyxy_pretruncation: None,
    };
    assert_eq!(degenerate.lines_bbox(), None);
    assert_eq!(degenerate.identity_branch(), IdentityBranch::LineLess);
    assert_eq!(
        degenerate.reconstruct(Rect::new(100, 200, 160, 240)),
        (Rect::new(100, 200, 160, 240), IdentityBranch::LineLess),
        "an empty polygon list must not be unioned as a point at the origin"
    );
}

#[test]
// spec §16.24 item 18(i) + cookbook rule 15 — built on the pair that STARTED this: §16.20 item 12
// offered ours `(40,23,111,207)` against upstream `[40,23,110,207]` as evidence FOR the identity,
// and it refutes the leg it was cited to support. `Rect::merge` (`geometry.rs:60-67`) is
// `x1.min, y1.min, x2.max, y2.max`, so a bounding union can never narrow an edge — upstream's
// `x2 = 110` against ours `111` is one pixel in the direction a union cannot produce, whatever
// upstream's lines are.
//
// What this test buys is the MESSAGE, not the power: `GeometryIdentity` already fails here. Without
// the monotonicity row a reader sees "identity failed" and reasonably suspects the line union;
// with it, the failure is attributed to **leg 1** (the two engines' regression heads disagreeing),
// which is where item 18(b) locates it. Had this existed as prose, item 12's contradiction would
// have been caught when it was written.
//
// Both rows are asserted, in emission order, so a comparator that reported only one fails.
fn the_refuting_pair_is_attributed_to_leg_one_by_the_monotonicity_message() {
    let ours = vec![RawBlock {
        rect: Rect::new(40, 23, 111, 207),
        class_index: 0,
        confidence: 0.696,
    }];
    let expectations = Expectations {
        pairs: vec![ExpectedPair {
            ours: 0,
            upstream: 0,
            branch: IdentityBranch::LineLess,
        }],
        unmatched_ours: vec![],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 1,
            ours_total: 1,
            upstream_total: 1,
        },
    };
    let upstream = UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![bare_block([40, 23, 110, 207])],
        pre_filter_blocks: vec![],
    };

    let report = oracle::compare(ours_side(&ours), &upstream, &expectations);

    assert_eq!(
        report.divergences,
        vec![
            Divergence::GeometryIdentity {
                ours: Rect::new(40, 23, 111, 207),
                expected: Rect::new(40, 23, 111, 207),
                upstream: Rect::new(40, 23, 110, 207),
            },
            Divergence::UnionMonotonicity {
                ours: Rect::new(40, 23, 111, 207),
                upstream: Rect::new(40, 23, 110, 207),
                edge: Edge::X2,
            },
        ]
    );

    // The sign pattern is the attribution (§16.24 item 18(h)(iii)): a NEGATIVE `x2` entry in
    // `residual_leg1` is what no union can produce.
    assert_eq!(report.residuals.len(), 1);
    assert_eq!(report.residuals[0].residual_leg1, [0, 0, -1, 0]);
    assert_eq!(report.residuals[0].residual_full, [0, 0, -1, 0]);

    // And a widening of the same edge must NOT raise the monotonicity row, or it would fire on
    // every legitimate line-informed pair — the falsification in the other direction.
    let widened = UpstreamOracle {
        blocks: vec![bare_block([40, 23, 112, 207])],
        ..upstream
    };
    assert!(
        !oracle::compare(ours_side(&ours), &widened, &expectations)
            .divergences
            .iter()
            .any(|row| matches!(row, Divergence::UnionMonotonicity { .. })),
        "a widened edge is what a union DOES produce; only narrowing is impossible"
    );

    // §16.24 item 21 R6 — the OPPOSITE polarity, because the impossible direction is not uniform
    // across edges. On `x1` a union can only move DOWN (`x1.min`), so `upstream.x1 > ours.x1` is the
    // union-impossible case and `residual_leg1.x1` is POSITIVE. A monotonicity check written with
    // one uniform comparison passes the `x2` case above and fails here; that is the sign error
    // cookbook rule 15 exists for.
    let x1_inward = UpstreamOracle {
        blocks: vec![bare_block([41, 23, 111, 207])],
        ..upstream_truth()
    };
    let report = oracle::compare(ours_side(&ours), &x1_inward, &expectations);
    assert!(
        report.divergences.contains(&Divergence::UnionMonotonicity {
            ours: Rect::new(40, 23, 111, 207),
            upstream: Rect::new(41, 23, 111, 207),
            edge: Edge::X1,
        }),
        "got {:?}",
        report.divergences
    );
    assert_eq!(
        report.residuals[0].residual_leg1,
        [1, 0, 0, 0],
        "x1's impossible direction is POSITIVE"
    );
}

#[test]
// spec §16.24 item 18(i): equality implies the four inequalities, so a `UnionMonotonicity` row
// without a `GeometryIdentity` row is not a finding about the artifacts — it is a contradiction
// inside the comparator. Sweeping every control in this file is the cheapest way to assert that
// invariant, and it would catch a monotonicity check written with a flipped comparison (which
// would otherwise fire on every well-behaved pair and be dismissed as "the new noisy row").
fn a_monotonicity_row_never_appears_without_an_identity_row() {
    let ours = ours_truth();
    let (short_ours, perturbed, perturbed_expectations) = negative_control();

    let cases: Vec<ComparisonReport> = vec![
        oracle::compare(ours_side(&ours), &upstream_truth(), &expectations_truth()),
        oracle::compare(ours_side(&short_ours), &perturbed, &perturbed_expectations),
    ];

    for report in &cases {
        let monotonicity = report
            .divergences
            .iter()
            .filter(|row| matches!(row, Divergence::UnionMonotonicity { .. }))
            .count();
        let identity = report
            .divergences
            .iter()
            .filter(|row| matches!(row, Divergence::GeometryIdentity { .. }))
            .count();
        assert!(
            monotonicity <= identity,
            "monotonicity rows ({monotonicity}) exceeded identity rows ({identity}): {:?}",
            report.divergences
        );
    }
}

#[test]
// spec §16.24 item 18(h)(iii): "Report both residual forms plus ... `len(lines)`, because the sign
// pattern is what attributes a failure to leg 1 versus leg 2 — a composite-only report would have
// left this contradiction as undiagnosable as the record did."
//
// The load-bearing assertion is the SECOND one: on a well-behaved line-informed pair
// `residual_leg1` is NON-ZERO (`[0,0,+4,0]` — the union widened `x2`, exactly item 12's measured
// pair) while `residual_full` is zero and NOTHING gates. A reading of 18(h)(ii) that gated any
// non-zero residual would fail every line-informed page; see the plan's §5 flag (ii).
fn every_pair_reports_both_residual_forms_and_leg_one_is_not_a_gate() {
    let ours = vec![RawBlock {
        rect: Rect::new(29, 105, 86, 261),
        class_index: 0,
        confidence: 0.564,
    }];
    let expectations = Expectations {
        pairs: vec![ExpectedPair {
            ours: 0,
            upstream: 0,
            branch: IdentityBranch::LineInformed,
        }],
        unmatched_ours: vec![],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 1,
            ours_total: 1,
            upstream_total: 1,
        },
    };
    let upstream = UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![OracleBlock {
            xyxy: [29, 105, 90, 261],
            lines: vec![vec![[33, 110], [90, 110], [90, 250], [33, 250]]],
            confidence: None,
            language: None,
            base_xyxy_pretruncation: Some([29.0, 105.4, 90.2, 261.8]),
        }],
        pre_filter_blocks: vec![],
    };

    let report = oracle::compare(ours_side(&ours), &upstream, &expectations);

    assert_eq!(report.residuals.len(), 1);
    let residual = &report.residuals[0];
    assert_eq!(residual.ours_index, 0);
    assert_eq!(residual.upstream_index, 0);
    assert_eq!(residual.branch, IdentityBranch::LineInformed);
    // Leg 2 did its job: the composite residual is zero...
    assert_eq!(residual.residual_full, [0, 0, 0, 0]);
    // ...while leg 1's is not, because the union legitimately widened `x2` by 4.
    assert_eq!(residual.residual_leg1, [0, 0, 4, 0]);
    // §16.24 item 18(k): the line census, the first real evidence on the degenerate branch.
    assert_eq!(residual.upstream_line_count, 1);
    // 18(e)'s discriminator, passed through untouched when the recording captured it.
    assert_eq!(
        residual.upstream_base_xyxy_pretruncation,
        Some([29.0, 105.4, 90.2, 261.8])
    );

    // The point of the whole test: a non-zero `residual_leg1` is NOT a divergence.
    assert_eq!(report.divergences, Vec::new());
    assert!(report.gating().is_empty());
}

// ── the enumeration: nothing may go unaccounted for ──────────────────────────

#[test]
// spec §16.20 item 3: "a comparator reporting 'compared 0 boxes, 0 divergences → PASS' is cookbook
// rule 1's defect". Under signed pairing an empty pairing is something a caller WROTE, so the
// structural guard is that the comparator refuses it outright rather than relying on every future
// caller to remember a count assertion. Falsifiable both ways: a non-empty pairing must not raise
// this row (asserted by the positive control).
fn an_empty_authored_pairing_can_never_pass() {
    let ours = ours_truth();
    let empty = Expectations {
        pairs: vec![],
        unmatched_ours: vec![],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 0,
            ours_total: 0,
            upstream_total: 0,
        },
    };
    let report = oracle::compare(ours_side(&ours), &upstream_truth(), &empty);

    assert!(report.divergences.contains(&Divergence::EmptyPairing));
    assert!(!report.gating().is_empty());
    assert_eq!(report.pairs_compared, 0);
}

#[test]
// spec §16.20 item 3(b): the count exists because "a renamed field silently yields an empty
// pairing rather than an error". Simulates exactly that — an oracle artifact whose `blocks`
// deserialized into nothing while the authored table still claims five pairs. `pairs_compared`
// must be 0 (no pair resolved) against a declared 5, and every unresolvable pair must be named.
fn an_empty_oracle_is_an_accounting_failure_not_a_pass() {
    let ours = ours_truth();
    let empty_oracle = UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![],
        pre_filter_blocks: vec![],
    };
    let report = oracle::compare(ours_side(&ours), &empty_oracle, &expectations_truth());

    assert_eq!(report.pairs_compared, 0);
    assert!(report
        .divergences
        .contains(&Divergence::AccountingMismatch {
            field: "pairs",
            declared: 5,
            actual: 0,
        }));
    assert!(report
        .divergences
        .contains(&Divergence::PairingIndexOutOfRange {
            side: Side::Upstream,
            index: 0,
            len: 0,
        }));
    // §16.24 item 21 R5: a sibling `upstream_total` row is TOLERATED, not asserted absent — the
    // authored total of 5 genuinely contradicts an artifact of 0, so a comparator that also reports
    // it is right to. Per R4 a structural-defect control asserts containment plus gating-non-empty
    // and never sibling-absence, because "no other row appeared" is not a property of a table this
    // broken.
    assert!(
        !report.gating().is_empty(),
        "an empty oracle must never pass"
    );
}

#[test]
// Cookbook rule 13 — "for each thing the gate protects, what does it iterate to find them?" This
// is the answer, and the reason signed pairing loses nothing to computed pairing: BOTH full block
// lists are walked independently and every index must be paired or listed exactly once. An
// authored table that quietly omits a box is the failure this catches, in both directions.
fn every_block_on_both_sides_must_be_paired_or_listed() {
    let ours = ours_truth();

    // Drop the last pair from the table while both artifacts still have five blocks: index 4 is
    // now unaccounted for on BOTH sides.
    let mut expectations = expectations_truth();
    expectations.pairs.pop();
    expectations.totals.pairs = 4;
    let report = oracle::compare(ours_side(&ours), &upstream_truth(), &expectations);

    assert!(report.divergences.contains(&Divergence::NotAccountedFor {
        side: Side::Ours,
        index: 4
    }));
    assert!(report.divergences.contains(&Divergence::NotAccountedFor {
        side: Side::Upstream,
        index: 4
    }));
    // §16.24 item 21 R5: the equation second-signal is DROPPED. With mechanism counts derived, the
    // equations' left side is `|pairs| + |unmatched|` by construction, so "the equations no longer
    // close" is a restatement of the two assertions above rather than independent evidence. Per R4
    // this control asserts containment plus gating-non-empty, never sibling-absence.
    assert!(!report.gating().is_empty());

    // Other direction: the complete table produces no `NotAccountedFor` at all.
    assert!(
        !oracle::compare(ours_side(&ours), &upstream_truth(), &expectations_truth())
            .divergences
            .iter()
            .any(|row| matches!(row, Divergence::NotAccountedFor { .. }))
    );
}

#[test]
// spec §16.20 item 3(b)'s box accounting. A block paired twice, or a pairing naming a nonexistent
// block, would inflate `pairs_compared` and let a real box hide behind a duplicate citation —
// authoring errors that a signed table makes possible and must therefore detect. Both are pinned,
// and the positive control proves neither fires on a well-formed table.
fn a_malformed_authored_pairing_is_rejected() {
    let ours = ours_truth();

    // Pair our block 0 twice; upstream 4 then goes unpaired and our block 4 is unaccounted for.
    let mut twice = expectations_truth();
    twice.pairs[4] = ExpectedPair {
        ours: 0,
        upstream: 4,
        branch: IdentityBranch::LineLess,
    };
    let report = oracle::compare(ours_side(&ours), &upstream_truth(), &twice);
    assert!(report.divergences.contains(&Divergence::PairedTwice {
        side: Side::Ours,
        index: 0
    }));
    assert!(report.divergences.contains(&Divergence::NotAccountedFor {
        side: Side::Ours,
        index: 4
    }));

    // An out-of-range index on our side.
    let mut out_of_range = expectations_truth();
    out_of_range.pairs[4] = ExpectedPair {
        ours: 9,
        upstream: 4,
        branch: IdentityBranch::LineLess,
    };
    let report = oracle::compare(ours_side(&ours), &upstream_truth(), &out_of_range);
    assert!(report
        .divergences
        .contains(&Divergence::PairingIndexOutOfRange {
            side: Side::Ours,
            index: 9,
            len: 5,
        }));
    assert_eq!(
        report.pairs_compared, 4,
        "an unresolvable pair is not a compared pair"
    );
}

#[test]
// spec §16.24 item 21 R3 — the mirror of `NotAccountedFor`, closing the set identity
// `declared_unmatched == artifact − paired` in BOTH directions. `NotAccountedFor` catches a block
// the table forgot; this catches a table entry that names a block which is not unmatched at all.
//
// Why it must exist once the mechanism counts are derived: a phantom entry inflates a derived count
// and lets both equations close over a block that either is already paired or does not exist. With
// authored counts that showed up as an `AccountingMismatch`; with derived counts the arithmetic is
// self-consistent and the only remaining signal is this row.
//
// The computed set is taken over the DECLARED pairs, so this is verification of the signed table
// against itself and the artifacts — not a pairing the comparator invented (item 4).
//
// Containment-style per R4: `contains` plus gating-non-empty, never sibling-absence, because a
// table this broken legitimately raises other rows too.
fn a_phantom_unmatched_entry_is_rejected() {
    let ours = ours_truth();

    // (a) The entry names a block that IS paired — it is not unmatched.
    let mut already_paired = expectations_truth();
    already_paired.unmatched_upstream = vec![UnmatchedEntry {
        index: 0,
        mechanism: Mechanism::ClassDuplicateOf { upstream_index: 1 },
    }];
    let report = oracle::compare(ours_side(&ours), &upstream_truth(), &already_paired);
    assert!(
        report.divergences.contains(&Divergence::PhantomUnmatched {
            side: Side::Upstream,
            index: 0
        }),
        "got {:?}",
        report.divergences
    );
    assert!(!report.gating().is_empty());

    // (b) The entry names an index outside the artifact entirely.
    let mut out_of_range = expectations_truth();
    out_of_range.unmatched_ours = vec![UnmatchedEntry {
        index: 9,
        mechanism: Mechanism::Open,
    }];
    let report = oracle::compare(ours_side(&ours), &upstream_truth(), &out_of_range);
    assert!(
        report.divergences.contains(&Divergence::PhantomUnmatched {
            side: Side::Ours,
            index: 9
        }),
        "got {:?}",
        report.divergences
    );
    assert!(!report.gating().is_empty());

    // Other direction: a sound table produces no phantom row, so neither half above can pass on a
    // comparator that always emits one.
    assert!(
        !oracle::compare(ours_side(&ours), &upstream_truth(), &expectations_truth())
            .divergences
            .iter()
            .any(|row| matches!(row, Divergence::PhantomUnmatched { .. }))
    );
}

// ── mechanisms: every unmatched block is explained, and the explanation is checked ──

#[test]
// spec §16.24 item 8 + §16.20 item 9 — the case the left-hand-side ruling exists to make VISIBLE
// instead of hidden. Upstream's line-less coverage filter fired once on the candidate page, on
// `[438,1407,498,1446]` at `mask_score` 0.0359, so that box is in our pre-filter
// `_detector_blocks.json` and absent from the post-`group_output` oracle. It closes as a
// documented unmatched-ours entry citing §14.17 — with **verified evidence**: the comparator
// checks that the cited `pre_filter_blocks` entry satisfies the identity against our rect, so
// "upstream had this box and then filtered it" is proven rather than asserted.
//
// Result must be ZERO divergences: a fully explained mechanism is not a divergence. That is what
// distinguishes this test from `a_coverage_claim_whose_evidence_does_not_match_is_rejected`.
fn the_upstream_coverage_filter_case_closes_as_documented_with_verified_evidence() {
    let filtered = Rect::new(438, 1407, 498, 1446);
    let ours = vec![
        RawBlock {
            rect: Rect::new(100, 200, 160, 240),
            class_index: 0,
            confidence: 0.775,
        },
        RawBlock {
            rect: filtered,
            class_index: 0,
            confidence: 0.5,
        },
    ];
    let upstream = UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        // Post-filter: the box is gone.
        blocks: vec![bare_block([100, 200, 160, 240])],
        // Pre-filter: upstream DID have it, line-less, geometrically agreeing.
        pre_filter_blocks: vec![bare_block([438, 1407, 498, 1446])],
    };
    let expectations = Expectations {
        pairs: vec![ExpectedPair {
            ours: 0,
            upstream: 0,
            branch: IdentityBranch::LineLess,
        }],
        unmatched_ours: vec![UnmatchedEntry {
            index: 1,
            mechanism: Mechanism::CoverageFilteredUpstream {
                pre_filter_index: 0,
            },
        }],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 1,
            ours_total: 2,
            upstream_total: 1,
        },
    };

    let report = oracle::compare(ours_side(&ours), &upstream, &expectations);
    assert_eq!(
        report.divergences,
        Vec::new(),
        "a verified mechanism is not a divergence"
    );
    assert_eq!(report.pairs_compared, 1);
}

#[test]
// Cookbook rule 12 one level up: a mechanism nobody can falsify is a rubber stamp. Three ways to
// claim `CoverageFilteredUpstream` wrongly, each rejected: an out-of-range citation, a citation
// whose pre-filter box is a DIFFERENT box, and the mechanism used on the upstream side where it
// makes no sense. Without this test the mechanism would be a free pass for any unmatched box.
fn a_coverage_claim_whose_evidence_does_not_match_is_rejected() {
    let ours = vec![RawBlock {
        rect: Rect::new(438, 1407, 498, 1446),
        class_index: 0,
        confidence: 0.5,
    }];
    let with_pre_filter = |pre: Vec<OracleBlock>| UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![],
        pre_filter_blocks: pre,
    };
    let claim = |pre_filter_index: usize| Expectations {
        // A single unmatched block and no pairs: `EmptyPairing` will also fire, so these
        // assertions use `contains` and the point is the mechanism row, not the whole vector.
        pairs: vec![],
        unmatched_ours: vec![UnmatchedEntry {
            index: 0,
            mechanism: Mechanism::CoverageFilteredUpstream { pre_filter_index },
        }],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 0,
            ours_total: 1,
            upstream_total: 0,
        },
    };

    // (a) citation out of range.
    let report = oracle::compare(ours_side(&ours), &with_pre_filter(vec![]), &claim(0));
    assert!(report.divergences.iter().any(|row| matches!(
        row,
        Divergence::UnverifiableMechanism {
            side: Side::Ours,
            index: 0,
            ..
        }
    )));

    // (b) citation points at a different box: the identity against our rect fails, so the claim
    // "upstream had THIS box" is unsupported.
    let wrong_box = with_pre_filter(vec![bare_block([10, 20, 30, 40])]);
    let report = oracle::compare(ours_side(&ours), &wrong_box, &claim(0));
    assert!(report.divergences.iter().any(|row| matches!(
        row,
        Divergence::UnverifiableMechanism {
            side: Side::Ours,
            index: 0,
            ..
        }
    )));

    // (c) the same mechanism on the upstream side is meaningless — upstream's filter cannot
    // explain a box upstream has and we lack.
    let mut wrong_side = claim(0);
    wrong_side.unmatched_ours.clear();
    wrong_side.unmatched_upstream = vec![UnmatchedEntry {
        index: 0,
        mechanism: Mechanism::CoverageFilteredUpstream {
            pre_filter_index: 0,
        },
    }];
    wrong_side.totals.ours_total = 1;
    wrong_side.totals.upstream_total = 1;
    let upstream_has_one = UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![bare_block([438, 1407, 498, 1446])],
        pre_filter_blocks: vec![bare_block([438, 1407, 498, 1446])],
    };
    let report = oracle::compare(ours_side(&ours), &upstream_has_one, &wrong_side);
    assert!(report.divergences.iter().any(|row| matches!(
        row,
        Divergence::UnverifiableMechanism {
            side: Side::Upstream,
            index: 0,
            ..
        }
    )));
}

#[test]
// spec §14.13 — upstream runs PER-CLASS NMS (`agnostic=False`), "which can emit duplicate
// overlapping boxes for one balloon detected under two language classes", and v1 deliberately
// runs class-agnostic NMS instead. So an upstream-only duplicate is a legitimate mechanism, and
// its citation must name a block that is actually paired: a duplicate of nothing explains nothing.
//
// Threshold-free by construction — verifying the two upstream boxes overlap would need the IoU
// §16.20 item 3(c) forbids, so the check is that the cited index is in range and paired. Both
// directions: a valid citation yields no divergence, a citation naming an unpaired block does.
fn a_class_duplicate_must_cite_a_paired_upstream_block() {
    let ours = vec![RawBlock {
        rect: Rect::new(100, 200, 160, 240),
        class_index: 0,
        confidence: 0.775,
    }];
    let upstream = UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![
            bare_block([100, 200, 160, 240]),
            // The per-class duplicate: same balloon, other language class.
            bare_block([100, 200, 160, 240]),
        ],
        pre_filter_blocks: vec![],
    };
    let base = Expectations {
        pairs: vec![ExpectedPair {
            ours: 0,
            upstream: 0,
            branch: IdentityBranch::LineLess,
        }],
        unmatched_ours: vec![],
        unmatched_upstream: vec![UnmatchedEntry {
            index: 1,
            mechanism: Mechanism::ClassDuplicateOf { upstream_index: 0 },
        }],
        totals: Totals {
            pairs: 1,
            ours_total: 1,
            upstream_total: 2,
        },
    };

    // Valid: cites the paired block 0. Equations (§16.24 item 11, six derived terms):
    // 1 + 1 + 0 + 0 = 2 upstream; 1 + 0 + 0 + 0 = 1 ours.
    assert_eq!(
        oracle::compare(ours_side(&ours), &upstream, &base).divergences,
        Vec::new()
    );

    // Invalid: cites itself, which is not paired.
    let mut self_cite = base.clone();
    self_cite.unmatched_upstream[0].mechanism = Mechanism::ClassDuplicateOf { upstream_index: 1 };
    // The block index, not the list position: `index` is `entry.index` at every emission site, and
    // three sibling assertions in this file depend on that reading. Exact vector rather than
    // `.any(..)`, and the `reason` is PINNED: `oracle.rs` has two distinct class-duplicate reasons,
    // so `..` would let the wrong-SIDE rejection satisfy a test whose subject is the citation check.
    assert_eq!(
        oracle::compare(ours_side(&ours), &upstream, &self_cite).divergences,
        vec![Divergence::UnverifiableMechanism {
            side: Side::Upstream,
            index: 1,
            reason: "class duplicate citation is not a paired upstream block",
        }]
    );
}

#[test]
// spec §16.24 item 4: "`Open` is a blocking violation, never a verdict." An unexplained residue
// must block the fixture commit, so it is GATING — and it must still be a single row, since the
// accounting terms partition the unmatched set (plan §1.3). Falsifiable: reclassify `Open` as
// diagnostic and the gating assertion fails.
fn an_open_mechanism_blocks_and_is_never_a_verdict() {
    let (ours, upstream, expectations) = negative_control();
    let report = oracle::compare(ours_side(&ours), &upstream, &expectations);

    let open = Divergence::OpenMechanism {
        side: Side::Upstream,
        index: 4,
    };
    assert!(report.divergences.contains(&open));
    assert_eq!(open.class(), DivergenceClass::Gating);
    assert!(report.gating().contains(&&open));
}

#[test]
// spec §16.20 item 3(b), the anti-bypass property, adopted from my earlier draft. The
// `documented_*` terms close the accounting ARITHMETIC; they must not silence the per-box report,
// or a maintainer could make a missing text box disappear by incrementing a counter — cookbook
// rule 12's "a gate that exists, looks rigorous, and guards a copy".
fn declaring_a_documented_difference_does_not_silence_the_row() {
    let (ours, upstream, expectations) = negative_control();
    let report = oracle::compare(ours_side(&ours), &upstream, &expectations);

    // The arithmetic closes...
    assert!(!report
        .divergences
        .iter()
        .any(|row| matches!(row, Divergence::AccountingMismatch { .. })));
    // ...and the box is still reported.
    assert!(report.divergences.contains(&Divergence::OpenMechanism {
        side: Side::Upstream,
        index: 4
    }));
}

#[test]
// spec §16.24 items 11 and 21 R5 — the ratified TWO-EQUATION accounting, with the six mechanism
// terms DERIVED and only the three totals authored:
//
//   pairs + class_duplicates + documented_split_merge_upstream + open_upstream == upstream_total
//   pairs + coverage_filtered_ours + documented_split_merge_ours + open_ours   == ours_total
//
// The literal "equals both totals" is unsatisfiable with one-sided extras, which is why item 11 is
// two equations. What this test verifies: the derived partition is correct, both sums close against
// the authored totals, and an authored total contradicting the artifact is a gating row.
//
// Renamed from `..._and_every_declared_term_are_checked`: there are no declared terms left to check
// (cookbook rule 1 — the name may not claim more than the assertions verify).
fn the_derived_accounting_partitions_the_entries_and_both_equations_close() {
    let ours = vec![RawBlock {
        rect: Rect::new(100, 200, 160, 240),
        class_index: 0,
        confidence: 0.775,
    }];
    let upstream = UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![
            bare_block([100, 200, 160, 240]),
            bare_block([100, 200, 160, 240]),
        ],
        pre_filter_blocks: vec![],
    };
    let sound = Expectations {
        pairs: vec![ExpectedPair {
            ours: 0,
            upstream: 0,
            branch: IdentityBranch::LineLess,
        }],
        unmatched_ours: vec![],
        unmatched_upstream: vec![UnmatchedEntry {
            index: 1,
            mechanism: Mechanism::ClassDuplicateOf { upstream_index: 0 },
        }],
        totals: Totals {
            pairs: 1,
            ours_total: 1,
            upstream_total: 2,
        },
    };
    let report = oracle::compare(ours_side(&ours), &upstream, &sound);
    assert_eq!(report.divergences, Vec::new());

    // The mechanism counts are DERIVED, so the arithmetic is reported rather than authored, and
    // each term names one mechanism class on one side. Equation 1: 1 + 1 + 0 + 0 = 2 upstream;
    // equation 2: 1 + 0 + 0 + 0 = 1 ours.
    assert_eq!(
        report.derived,
        DerivedAccounting {
            pairs: 1,
            class_duplicates: 1,
            documented_split_merge_upstream: 0,
            open_upstream: 0,
            coverage_filtered_ours: 0,
            documented_split_merge_ours: 0,
            open_ours: 0,
        }
    );
    assert_eq!(report.derived.upstream_sum(), sound.totals.upstream_total);
    assert_eq!(report.derived.ours_sum(), sound.totals.ours_total);

    // A declared total that contradicts the artifact — the half of the old test that survives,
    // because `*_total` is the only accounting quantity still authored.
    let mut wrong_total = sound.clone();
    wrong_total.totals.upstream_total = 3;
    let report = oracle::compare(ours_side(&ours), &upstream, &wrong_total);
    assert!(report
        .divergences
        .contains(&Divergence::AccountingMismatch {
            field: "upstream_total",
            declared: 3,
            actual: 2,
        }));
    assert!(!report.gating().is_empty());

    // The other half — a MISLABELLED mechanism count — is now unrepresentable, which is why this
    // test no longer claims to check "every declared term" (§16.24 item 21 R5, and cookbook rule 1:
    // the name must not claim more than the assertions verify). There is no field to mislabel:
    // `Totals` has three members and none of them is a mechanism count. The derived counts above
    // are the replacement, and `a_phantom_unmatched_entry_is_rejected` covers the remaining way a
    // signed table could inflate one.
}

// ── page-level rows and the class partition ──────────────────────────────────

#[test]
// spec §16.24 item 8 + cookbook rule 7: "upstream is a real oracle for `scale`, `image_size` and
// box geometry", and these two rows "are what pins the two sides to one coordinate frame" — so
// both are GATING. Without them every box comparison could be right in the wrong frame, which
// §8.2's resize path makes a live possibility (§16.24 item 17 puts the committed page at
// letterbox ratio ≈ 0.617, so the resize runs on both sides).
//
// Both directions, and the rows come FIRST in emission order so a frame mismatch is read before
// the box rows it would explain.
fn scale_and_image_size_are_gated_exactly() {
    let ours = ours_truth();
    let mut upstream = upstream_truth();

    assert_eq!(
        oracle::compare(ours_side(&ours), &upstream, &expectations_truth()).divergences,
        Vec::new()
    );

    upstream.scale = 1.0;
    let report = oracle::compare(ours_side(&ours), &upstream, &expectations_truth());
    assert_eq!(
        report.divergences,
        vec![Divergence::ScaleMismatch {
            ours: FRAME_SCALE,
            upstream: 1.0
        }]
    );
    assert!(report.gating().len() == 1);

    let mut upstream = upstream_truth();
    upstream.image_size = (1024, 1433);
    assert_eq!(
        oracle::compare(ours_side(&ours), &upstream, &expectations_truth()).divergences,
        vec![Divergence::ImageSizeMismatch {
            ours: FRAME_SIZE,
            upstream: (1024, 1433)
        }]
    );
}

#[test]
// spec §16.20 item 3(d): "`confidence`, `language` and `raw_mask` are DIAGNOSTIC or NO-ORACLE rows
// only, never gated", and §16.24 item 9 keeps `confidence: Option<_>` with NO-ORACLE rows when
// absent. §16.20 item 5 is why: the noise floor from unavoidable rounding is ~0.08 while §8.3 step
// 4's gates sit at 0.4, so "a tolerance absorbing the noise spans 'kept' and 'dropped'".
//
// This also proves the comparator applies §8.3 step 4's class→language mapping
// (`yolo::class_to_language`, 0 => English, 1 => Japanese) rather than comparing raw integers.
fn confidence_and_language_are_diagnostic_and_absence_is_no_oracle() {
    let ours = vec![RawBlock {
        rect: Rect::new(100, 200, 160, 240),
        class_index: 0,
        confidence: 0.775,
    }];
    let expectations = Expectations {
        pairs: vec![ExpectedPair {
            ours: 0,
            upstream: 0,
            branch: IdentityBranch::LineLess,
        }],
        unmatched_ours: vec![],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 1,
            ours_total: 1,
            upstream_total: 1,
        },
    };
    let frame = |block: OracleBlock| UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![block],
        pre_filter_blocks: vec![],
    };

    // Disagreement on both: two DIAGNOSTIC rows, zero gating.
    let disagreeing = frame(OracleBlock {
        xyxy: [100, 200, 160, 240],
        lines: vec![],
        confidence: Some(0.696),
        language: Some(Language::Japanese),
        base_xyxy_pretruncation: None,
    });
    let report = oracle::compare(ours_side(&ours), &disagreeing, &expectations);
    assert!(report.gating().is_empty(), "{:?}", report.gating());
    assert_eq!(
        report.divergences,
        vec![
            Divergence::ConfidenceDelta {
                ours: Rect::new(100, 200, 160, 240),
                ours_confidence: 0.775,
                upstream_confidence: 0.696,
            },
            Divergence::LanguageDelta {
                ours: Rect::new(100, 200, 160, 240),
                ours_language: Some(Language::English),
                upstream_language: Some(Language::Japanese),
            },
        ]
    );

    // §16.25 item 7 (additive strengthening): the disagreeing artifact DOES carry both fields, so
    // coverage is `Present`. Asserting it here is what makes `Absent` falsifiable against `Present`
    // inside one test rather than only across tests.
    assert_eq!(report.coverage.confidence, OracleCoverage::Present);
    assert_eq!(report.coverage.language, OracleCoverage::Present);

    // Absent on both: a closed NO-ORACLE verdict (§16.24 item 9 — "either outcome is a closed
    // verdict; neither blocks the fixture"), reported as coverage and NOT as divergences.
    // Exact equality replaces the old `no_oracle().len() == 2`: a bare `len()` among exact-vector
    // siblings is cookbook rule 13's "cardinality is not identity" (§16.25 item 7).
    let absent = frame(bare_block([100, 200, 160, 240]));
    let report = oracle::compare(ours_side(&ours), &absent, &expectations);
    assert!(report.gating().is_empty());
    assert_eq!(report.divergences, Vec::new());
    assert_eq!(report.coverage.confidence, OracleCoverage::Absent);
    assert_eq!(report.coverage.language, OracleCoverage::Absent);

    // Agreement on both: no row at all, so the two above are falsifiable.
    let agreeing = frame(OracleBlock {
        xyxy: [100, 200, 160, 240],
        lines: vec![],
        confidence: Some(0.775),
        language: Some(Language::English),
        base_xyxy_pretruncation: None,
    });
    let report = oracle::compare(ours_side(&ours), &agreeing, &expectations);
    assert_eq!(report.divergences, Vec::new());
    assert_eq!(report.coverage.confidence, OracleCoverage::Present);
    assert_eq!(report.coverage.language, OracleCoverage::Present);
}

#[test]
// spec §16.25 item 5 — `Partial` coverage is GATING, and the derivation is what makes that
// defensible without bending §16.20 item 3(d): `Partial` ⟹ "the script captured upstream's scores"
// is neither true nor false of the document ⟹ §16.24 item 9's two-way branch does not resolve ⟹ the
// `confidence` partition row is `OPEN` ⟹ item 3(e) makes it blocking. The row is not a verdict about
// confidence; it is the report stating that the partition cannot be completed.
//
// The artifact is structurally SOUND — every block paired or listed, totals matching, identity
// holding — so this belongs in the exact-whole-vector family per §16.24 item 21(e), not the
// containment family.
//
// The second half is §16.25 item 7's additive strengthening and is the point of the test: the
// *with*-block's `ConfidenceDelta` must STILL be emitted. That is what proves the subset is
// reported rather than suppressed — the exact confusion that produced the rejected curve-fit, which
// made the signal disappear so the check went green.
fn partial_oracle_coverage_gates_and_still_reports_the_covered_subset() {
    let ours = vec![
        RawBlock {
            rect: Rect::new(100, 200, 160, 240),
            class_index: 0,
            confidence: 0.775,
        },
        RawBlock {
            rect: Rect::new(300, 400, 366, 441),
            class_index: 0,
            confidence: 0.512,
        },
    ];
    // Block 0 carries confidence and DISAGREES with ours; block 1 carries none.
    let upstream = UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![
            OracleBlock {
                xyxy: [100, 200, 160, 240],
                lines: vec![],
                confidence: Some(0.696),
                language: Some(Language::English),
                base_xyxy_pretruncation: None,
            },
            OracleBlock {
                xyxy: [300, 400, 366, 441],
                lines: vec![],
                confidence: None,
                language: Some(Language::English),
                base_xyxy_pretruncation: None,
            },
        ],
        pre_filter_blocks: vec![],
    };
    let expectations = Expectations {
        pairs: vec![
            ExpectedPair {
                ours: 0,
                upstream: 0,
                branch: IdentityBranch::LineLess,
            },
            ExpectedPair {
                ours: 1,
                upstream: 1,
                branch: IdentityBranch::LineLess,
            },
        ],
        unmatched_ours: vec![],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 2,
            ours_total: 2,
            upstream_total: 2,
        },
    };

    let report = oracle::compare(ours_side(&ours), &upstream, &expectations);

    // Exact vector, in the contractual emission order of §16.25 item 6(c): the coverage caveat
    // precedes every block row it qualifies.
    assert_eq!(
        report.divergences,
        vec![
            Divergence::InconsistentOracleCoverage {
                field: "confidence",
                with: 1,
                without: 1,
                missing: vec![1],
            },
            Divergence::ConfidenceDelta {
                ours: Rect::new(100, 200, 160, 240),
                ours_confidence: 0.775,
                upstream_confidence: 0.696,
            },
        ]
    );

    // The subset IS reported, not suppressed — the whole point.
    assert_eq!(report.diagnostics().len(), 1);
    assert_eq!(report.pairs_compared, 2);

    // Gating, and gating on coverage alone: the `ConfidenceDelta` beside it is diagnostic.
    let gating = report.gating();
    assert_eq!(gating.len(), 1);
    assert!(matches!(
        gating[0],
        Divergence::InconsistentOracleCoverage { .. }
    ));

    // `language` is on every block, so it is `Present` — one field `Partial` must not drag another
    // with it.
    assert_eq!(
        report.coverage.confidence,
        OracleCoverage::Partial {
            with: 1,
            without: 1
        }
    );
    assert_eq!(report.coverage.language, OracleCoverage::Present);
    // §16.25 item 6(a): the whole-artifact figures are diagnostic and here coincide with the paired
    // subject, since every block is paired.
    assert_eq!(
        report.coverage.confidence_whole_artifact,
        OracleCoverage::Partial {
            with: 1,
            without: 1
        }
    );
}

#[test]
// spec §16.24 item 6: the class system is what reconciles item 3(d) ("never gated") with item 10
// ("exactly three divergences"), so the partition must be TOTAL — every variant lands in exactly
// one class, and `gating()` must exclude precisely the diagnostic and no-oracle rows. A variant
// added later without a `class()` arm would fall into the catch-all and silently become gating;
// that is the safe direction, and this test documents it as deliberate rather than accidental.
fn the_divergence_class_partition_is_total_and_gating_excludes_diagnostics() {
    let samples = [
        Divergence::GeometryIdentity {
            ours: Rect::new(0, 0, 1, 1),
            expected: Rect::new(0, 0, 1, 1),
            upstream: Rect::new(0, 0, 2, 1),
        },
        // §16.24 item 18(i) — gating, because it can only accompany a failed identity and the
        // identity gates. Its value is the attribution, not the verdict.
        Divergence::UnionMonotonicity {
            ours: Rect::new(0, 0, 2, 1),
            upstream: Rect::new(0, 0, 1, 1),
            edge: Edge::X2,
        },
        Divergence::IdentityBranchMismatch {
            ours: Rect::new(0, 0, 1, 1),
            declared: IdentityBranch::LineLess,
            actual: IdentityBranch::LineInformed,
        },
        Divergence::EmptyPairing,
        Divergence::PairingIndexOutOfRange {
            side: Side::Ours,
            index: 9,
            len: 1,
        },
        Divergence::PairedTwice {
            side: Side::Upstream,
            index: 0,
        },
        Divergence::NotAccountedFor {
            side: Side::Ours,
            index: 0,
        },
        // §16.24 item 21 R3 — gating, and the mirror of `NotAccountedFor`.
        Divergence::PhantomUnmatched {
            side: Side::Upstream,
            index: 0,
        },
        Divergence::OpenMechanism {
            side: Side::Ours,
            index: 0,
        },
        Divergence::UnverifiableMechanism {
            side: Side::Upstream,
            index: 0,
            reason: "class duplicate citation is not a paired upstream block",
        },
        Divergence::AccountingMismatch {
            field: "pairs",
            declared: 1,
            actual: 0,
        },
        Divergence::ScaleMismatch {
            ours: 1.0,
            upstream: 0.5,
        },
        Divergence::ImageSizeMismatch {
            ours: (1024, 1434),
            upstream: (1024, 1433),
        },
        // §16.25 item 5 — GATING, derived from item 3(e)'s OPEN-blocks rule rather than as an
        // exception to item 3(d): `Partial` means item 9's branch does not resolve, so the
        // partition row is OPEN, so it blocks.
        Divergence::InconsistentOracleCoverage {
            field: "confidence",
            with: 1,
            without: 1,
            missing: vec![1],
        },
        Divergence::ConfidenceDelta {
            ours: Rect::new(0, 0, 1, 1),
            ours_confidence: 0.4,
            upstream_confidence: 0.5,
        },
        Divergence::LanguageDelta {
            ours: Rect::new(0, 0, 1, 1),
            ours_language: None,
            upstream_language: Some(Language::English),
        },
    ];

    let gating = samples.iter().filter(|row| row.is_gating()).count();
    let diagnostic = samples
        .iter()
        .filter(|row| row.class() == DivergenceClass::Diagnostic)
        .count();

    // §16.25 item 11: the sample is now EXHAUSTIVE — 16 of 16 variants — because a test whose name
    // claims totality while covering 10 of 16 is cookbook rule 1's dominant defect class sitting in
    // the very test whose job is to prevent it. The literal 16 is the ratchet that makes adding a
    // variant without classifying it a failure here.
    assert_eq!(samples.len(), 16, "every `Divergence` variant must appear");
    assert_eq!((gating, diagnostic), (14, 2));
    assert_eq!(
        gating + diagnostic,
        samples.len(),
        "the partition must be total"
    );
    let confidence_row = samples
        .iter()
        .find(|row| matches!(row, Divergence::ConfidenceDelta { .. }))
        .expect("the sample set includes a confidence row");
    assert!(
        confidence_row.class() == DivergenceClass::Diagnostic && !confidence_row.is_gating(),
        "§16.20 item 3(d): confidence is DIAGNOSTIC and never gates"
    );
}

#[test]
// spec §16.20 item 3(a): the oracle artifact is a committed JSON file a third party must be able
// to read, so its shape has to survive a round trip and must tolerate absent optional fields —
// `confidence` and `language` are absent whenever the recording could not capture them (§16.24
// item 9), and `pre_filter_blocks` is absent on any artifact recorded before item 8's adjustment.
// `deny_unknown_fields` is asserted too: a renamed field must fail loudly rather than
// deserializing into an empty block list, which is item 3(b)'s named failure mode.
fn the_oracle_artifact_shape_round_trips_and_rejects_unknown_fields() {
    let original = upstream_truth();
    let text = serde_json::to_string_pretty(&original).expect("serialisable");
    let parsed: UpstreamOracle = serde_json::from_str(&text).expect("round-trips");
    assert_eq!(parsed, original);

    let minimal: UpstreamOracle = serde_json::from_str(
        r#"{"scale":1.0,"image_size":[1024,1434],"blocks":[{"xyxy":[1,2,3,4]}]}"#,
    )
    .expect("optional fields may be absent");
    assert_eq!(minimal.blocks[0].lines, Vec::<Vec<[i32; 2]>>::new());
    assert_eq!(minimal.blocks[0].confidence, None);
    assert_eq!(minimal.blocks[0].language, None);
    assert_eq!(minimal.blocks[0].base_xyxy_pretruncation, None);
    assert!(minimal.pre_filter_blocks.is_empty());

    // The FULL-SHAPE literal — cookbook rule 14 / plan §1.7 gate 2. `xtask/scripts/
    // record_detector_oracle.py` writes this format and no type system spans the two ends, so this
    // string is the canonical example the script must match, and the only Rust-side gate on the
    // Python writer until the script exists. Every optional field is populated here on purpose: a
    // *dropped* optional field is the quiet direction of drift, since it defaults silently.
    let full: UpstreamOracle = serde_json::from_str(
        r#"{
          "scale": 0.617,
          "image_size": [1024, 1434],
          "blocks": [
            {
              "xyxy": [29, 105, 90, 261],
              "lines": [[[33, 110], [90, 110], [90, 250], [33, 250]]],
              "confidence": 0.564,
              "language": "english",
              "base_xyxy_pretruncation": [29.0, 105.4, 90.2, 261.8]
            }
          ],
          "pre_filter_blocks": [
            { "xyxy": [438, 1407, 498, 1446], "lines": [] }
          ]
        }"#,
    )
    .expect("the full recorded shape must parse");
    assert_eq!(full.blocks[0].language, Some(Language::English));
    assert_eq!(full.blocks[0].lines.len(), 1);
    assert_eq!(
        full.blocks[0].base_xyxy_pretruncation,
        Some([29.0, 105.4, 90.2, 261.8])
    );
    assert_eq!(full.pre_filter_blocks.len(), 1);
    // `Language` serialises lowercase (`pc-core/src/language.rs:5-10`), so the script must emit
    // `"english"`/`"japanese"` and not upstream's own language codes. Pinning it here is what
    // stops that mapping from being invented twice.
    assert!(serde_json::from_str::<UpstreamOracle>(
        r#"{"scale":1.0,"image_size":[1,1],"blocks":[{"xyxy":[1,2,3,4],"language":"eng"}]}"#
    )
    .is_err());

    let renamed = serde_json::from_str::<UpstreamOracle>(
        r#"{"scale":1.0,"image_size":[1024,1434],"boxes":[{"xyxy":[1,2,3,4]}]}"#,
    );
    assert!(
        renamed.is_err(),
        "a renamed field must not silently yield an empty block list"
    );
}
