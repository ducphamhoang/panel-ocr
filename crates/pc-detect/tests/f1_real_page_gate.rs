//! F1 real-page oracle gate. FROZEN per CLAUDE.md.
//!
//! spec §16.20 item 3(b) / §16.24 item 6 / §16.28 — the gate that `oracle.rs`'s module doc says
//! "lands only in the atomic recording commit". It compares our committed detector output for
//! `ja_Pepper-and-Carrot_by-David-Revoy_E01P01` against the committed upstream PanelCleaner oracle
//! run (upstream `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`, `comictextdetector.pt.onnx`,
//! page sha256 `3bef9922…`, all pinned in `tests/fixtures/recorded/detector/PROVENANCE.json`).
//!
//! **The [`Expectations`] below is a SIGNED INPUT (§16.24 item 4), hand-derived from the two
//! recorded documents and the comparator's stated rules — not read back from any run of
//! `oracle::compare`, and not copied from any tool's own derivation of what the pairing "should"
//! be.** The full derivation is written out beside each entry so a reader can re-check the
//! arithmetic without running anything. The one field copied verbatim rather than derived is
//! `derivation`, which `oracle.rs` documents as diagnostic passthrough from the recorder.
//!
//! §16.24 item 6's CI contract is `gating().is_empty()` AND `pairs_compared == <literal>`; both
//! are asserted, plus the branch/residual/accounting/coverage channels, because a green
//! `gating()` over the wrong pairing is exactly what cookbook rule 1 warns about.

use pc_core::{Language, Rect};
use pc_detect::oracle::{
    self, Derivation, DerivedAccounting, Divergence, Expectations, ExpectedPair, FieldCoverage,
    IdentityBranch, Mechanism, OracleCoverage, OursSide, PairResidual, Side, Totals,
    UnmatchedEntry, UpstreamOracle,
};
use pc_detect::RawBlock;

const STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";

// ── the recorded artifacts ───────────────────────────────────────────────────

fn ours_blocks() -> Vec<RawBlock> {
    // §16.24 item 8: OUR side of the comparison is the PRE-coverage-filter replay fixture, not
    // `#raw.json`. `#raw.json` supplies only the coordinate frame.
    let path = pc_testkit::paths::recorded(format!("detector/{STEM}_detector_blocks.json"));
    let bytes = std::fs::read(&path).expect("recorded detector blocks are readable");
    serde_json::from_slice(&bytes).expect("recorded detector blocks parse as Vec<RawBlock>")
}

fn ours_frame() -> (f64, (u32, u32)) {
    let path = pc_testkit::paths::recorded(format!("detector/{STEM}#raw.json"));
    let bytes = std::fs::read(&path).expect("recorded #raw.json is readable");
    let page: pc_core::PageDataRaw =
        serde_json::from_slice(&bytes).expect("recorded #raw.json parses as PageDataRaw");
    (page.scale, page.image_size)
}

fn upstream_oracle() -> UpstreamOracle {
    let path = pc_testkit::paths::recorded(format!("detector/{STEM}_upstream_oracle.json"));
    let bytes = std::fs::read(&path).expect("recorded upstream oracle is readable");
    serde_json::from_slice(&bytes).expect("recorded upstream oracle parses as UpstreamOracle")
}

/// Every index in the signed table below is a POSITION in one of these three lists, so the
/// positions themselves are pinned. Without this a re-recorded fixture in a different order would
/// silently re-point every pair while `gating()` stayed empty — cookbook rule 8's "indexing a
/// sample by position" hazard, applied to a signed pairing.
fn assert_artifact_layout(ours: &[RawBlock], upstream: &UpstreamOracle) {
    assert_eq!(
        ours.iter()
            .map(|block| (block.rect, block.class_index, block.confidence))
            .collect::<Vec<_>>(),
        vec![
            (Rect::new(674, 1397, 740, 1438), 0, 0.713),
            (Rect::new(567, 74, 663, 123), 0, 0.704),
            (Rect::new(607, 631, 724, 703), 1, 0.643),
            (Rect::new(438, 1407, 498, 1446), 0, 0.627),
        ],
        "ours-side pre-filter block list changed shape or order"
    );
    assert_eq!(
        upstream
            .blocks
            .iter()
            .map(|block| block.xyxy)
            .collect::<Vec<_>>(),
        vec![
            [567, 74, 663, 123],
            [607, 630, 723, 705],
            [607, 631, 724, 703],
            [674, 1397, 740, 1438],
        ],
        "upstream post-group_output block list changed shape or order"
    );
    assert_eq!(
        upstream
            .pre_filter_blocks
            .iter()
            .map(|block| block.xyxy)
            .collect::<Vec<_>>(),
        vec![
            [567, 74, 663, 123],
            [674, 1397, 740, 1438],
            [607, 630, 723, 703],
            [607, 631, 724, 703],
            [438, 1407, 498, 1446],
        ],
        "upstream pre-filter list changed shape or order; `pre_filter_index: 4` is a position in it"
    );
}

// ── the signed pairing, hand-derived ─────────────────────────────────────────

/// The hand-derivation, in full. Ours-side indices are positions in
/// `_detector_blocks.json`; upstream indices are positions in `_upstream_oracle.json`'s `blocks`.
///
/// **Pair (ours 0, upstream 3)** — ours `(674,1397,740,1438)`, upstream `[674,1397,740,1438]`.
/// `lines_pre_expand` is recorded (`eng_expanded: true`, `expand_size: 4`), so §16.29 item 1 makes
/// it the identity operand: one polygon `(674,1397) (740,1397) (740,1438) (674,1438)`, whose bbox
/// is `(674,1397,740,1438)`. Points exist ⇒ `LineInformed`. Union with ours:
/// `x1 min(674,674)=674`, `y1 min(1397,1397)=1397`, `x2 max(740,740)=740`, `y2 max(1438,1438)=1438`
/// ⇒ `(674,1397,740,1438)` == upstream. Note this is the pair where the operand choice bites: the
/// *served* `lines` are `(674,1393)…(740,1442)` (post-expansion), which would give
/// `(674,1393,740,1442)` and fail. `derivation: yolo_synthesized_corners`;
/// `rect_yolo [674,1397,740,1438]` == ours, so leg 1 holds.
///
/// **Pair (ours 1, upstream 0)** — ours `(567,74,663,123)`, upstream `[567,74,663,123]`.
/// `lines_pre_expand` recorded (`expand_size: 2`): two polygons, points
/// `(577,74) (651,74) (651,97) (577,97)` and `(569,98) (660,98) (660,121) (569,121)`; bbox over all
/// eight = `x1 569, y1 74, x2 660, y2 121`. ⇒ `LineInformed`. Union with ours:
/// `min(567,569)=567`, `min(74,74)=74`, `max(663,660)=663`, `max(123,121)=123` ⇒ `(567,74,663,123)`
/// == upstream. `derivation: yolo_unioned`; `rect_yolo [567,74,663,123]` == ours.
///
/// **Pair (ours 2, upstream 2)** — ours `(607,631,724,703)`, upstream `[607,631,724,703]`, both
/// japanese. `lines_pre_expand` is `null` (`eng_expanded: false`), so the operand is the served
/// `lines`: one polygon `(607,631) (724,631) (724,703) (607,703)`, bbox `(607,631,724,703)`.
/// ⇒ `LineInformed` (a synthesized corner polygon is still points). Union with ours is ours ⇒
/// equal. `derivation: yolo_synthesized_corners`; `rect_yolo [607,631,724,703]` == ours.
///
/// **Unmatched ours 3** — `(438,1407,498,1446)`. Upstream's post-`group_output` list has no block
/// there, and the reason is recorded: `pre_filter_blocks[4]` is exactly `[438,1407,498,1446]`, and
/// PROVENANCE's `upstream_pre_filter_mask_scores[4]` gives `mask_score 0.03446…` with
/// `len_lines 0` — below upstream's `mask_score_thresh` of 0.1, on upstream's line-less branch.
/// That is §14.17 / §16.20 item 9 verbatim, so the mechanism is
/// `CoverageFilteredUpstream { pre_filter_index: 4 }`. **It must appear here even though our own
/// coverage filter also dropped the box downstream**: this comparison's ours-side list is the
/// PRE-filter fixture, index 3 exists in it, and `oracle.rs` walks `0..ours.blocks.len()`
/// independently — an undeclared index 3 is `NotAccountedFor`, and its absence from the mechanism
/// partition breaks `ours_sum`. Our own filter's decision (coverage 0.0? — not recorded; the box
/// is simply absent from `#raw.json`) is a *different* operation on a *different* operand (§14
/// item 17) and is not what this entry cites.
///
/// **Unmatched upstream 1** — `[607,630,723,705]`, english, `yolo_unioned`. Upstream emitted TWO
/// blocks over the same balloon, one per language class: `pre_filter_blocks[2]`
/// `[607,630,723,703]` english and `pre_filter_blocks[3]` `[607,631,724,703]` japanese. Upstream
/// runs per-class NMS (`agnostic=False`); §14.13 makes v1's NMS class-agnostic, so our detector
/// emitted one box for that balloon — the japanese one, paired at upstream index 2. The english
/// sibling therefore has no counterpart by a documented mechanism:
/// `ClassDuplicateOf { upstream_index: 2 }`. The citation is checkable and checked — `oracle.rs`
/// requires index 2 to be in range and itself paired, which it is.
///
/// **Accounting** — `pairs 3`; ours `3 + 1 coverage_filtered = 4` = the pre-filter list length;
/// upstream `3 + 1 class_duplicate = 4` = the oracle `blocks` length. Both equations close.
fn signed_expectations() -> Expectations {
    Expectations {
        pairs: vec![
            ExpectedPair {
                ours: 0,
                upstream: 3,
                branch: IdentityBranch::LineInformed,
                derivation: Some(Derivation::YoloSynthesizedCorners),
            },
            ExpectedPair {
                ours: 1,
                upstream: 0,
                branch: IdentityBranch::LineInformed,
                derivation: Some(Derivation::YoloUnioned),
            },
            ExpectedPair {
                ours: 2,
                upstream: 2,
                branch: IdentityBranch::LineInformed,
                derivation: Some(Derivation::YoloSynthesizedCorners),
            },
        ],
        unmatched_ours: vec![UnmatchedEntry {
            index: 3,
            mechanism: Mechanism::CoverageFilteredUpstream {
                pre_filter_index: 4,
            },
        }],
        unmatched_upstream: vec![UnmatchedEntry {
            index: 1,
            mechanism: Mechanism::ClassDuplicateOf { upstream_index: 2 },
        }],
        totals: Totals {
            pairs: 3,
            ours_total: 4,
            upstream_total: 4,
        },
    }
}

/// The three DIAGNOSTIC confidence rows, in the comparator's emission order (authored pair order,
/// each pair's confidence row after its geometry rows). Every value is copied from the two
/// artifacts: ours from `_detector_blocks.json`, upstream from the oracle's `confidence`, whose
/// recorded f64s are the exact widenings of these f32s.
fn expected_confidence_rows() -> Vec<Divergence> {
    vec![
        Divergence::ConfidenceDelta {
            ours: Rect::new(674, 1397, 740, 1438),
            ours_confidence: 0.713,
            upstream_confidence: 0.715,
        },
        Divergence::ConfidenceDelta {
            ours: Rect::new(567, 74, 663, 123),
            ours_confidence: 0.704,
            upstream_confidence: 0.774,
        },
        Divergence::ConfidenceDelta {
            ours: Rect::new(607, 631, 724, 703),
            ours_confidence: 0.643,
            upstream_confidence: 0.624,
        },
    ]
}

// ── the gate ─────────────────────────────────────────────────────────────────

#[test]
// spec §16.20 item 3(b) + §16.24 item 6 — THE real-page gate. Asserts the whole ordered
// `divergences` vector, so a fourth row, a dropped row or a reordering fails here; `gating()` and
// `pairs_compared` are asserted separately because those two together are §16.24 item 6's stated
// CI contract and must be readable as such.
//
// What turns this red: any coordinate moving on either side; the identity operand reverting from
// `lines_pre_expand` to the served `lines` (pair (0,3) alone catches that, 4 px on two edges); a
// re-recorded fixture in a different order (`assert_artifact_layout`); a dropped `derivation`
// (signature check); the ours-side coverage-filtered box losing its mechanism (`ours_sum`); the
// class-duplicate citation pointing at an unpaired block.
fn the_recorded_page_matches_the_upstream_oracle_under_the_signed_pairing() {
    let ours = ours_blocks();
    let upstream = upstream_oracle();
    let (scale, image_size) = ours_frame();
    assert_artifact_layout(&ours, &upstream);

    // The frame, hard-coded rather than taken from either document (§16.24 item 8 gates both).
    assert_eq!(scale, 1.0, "the recording requires scale == 1.0");
    assert_eq!(image_size, (1200, 1660));

    let report = oracle::compare(
        OursSide {
            blocks: &ours,
            scale,
            image_size,
        },
        &upstream,
        &signed_expectations(),
    );

    // §16.24 item 6's CI contract, both halves.
    assert_eq!(
        report.gating(),
        Vec::<&Divergence>::new(),
        "gating rows on the real page: {:?}",
        report.gating()
    );
    assert_eq!(report.pairs_compared, 3);

    // …and the whole vector, so "and no others" is a real assertion. The only rows are the three
    // DIAGNOSTIC confidence deltas, which §16.20 item 3(d) forbids from ever gating.
    assert_eq!(report.divergences, expected_confidence_rows());
    assert_eq!(report.diagnostics().len(), 3);

    // The branch ACTUALLY taken per pair (§16.24 item 6): a comparator that ignored `lines`
    // entirely would report `LineLess` three times and be caught here, not by the identity.
    assert_eq!(
        report.branches,
        vec![
            (0, 3, IdentityBranch::LineInformed),
            (1, 0, IdentityBranch::LineInformed),
            (2, 2, IdentityBranch::LineInformed),
        ]
    );

    // §16.28 item 5: leg 1 (`ours.rect == rect_yolo`) was evaluated on all three pairs — no pair
    // has `derivation: None` and none is unpairable.
    assert_eq!(report.leg1_rows_checked, 3);

    // §16.24 item 11's seven terms, derived from the signed lists. 3+1 = 4 on each side.
    assert_eq!(
        report.derived,
        DerivedAccounting {
            pairs: 3,
            class_duplicates: 1,
            documented_split_merge_upstream: 0,
            dbnet_scattered: 0,
            open_upstream: 0,
            coverage_filtered_ours: 1,
            documented_split_merge_ours: 0,
            open_ours: 0,
        }
    );
    assert_eq!(report.derived.ours_sum(), 4);
    assert_eq!(report.derived.upstream_sum(), 4);

    // §16.24 item 18(h): the per-pair residual row. `residual_full` all-zero is the positive
    // definition of green; `residual_leg1` is all-zero too on this page because every paired
    // upstream box equals its `rect_yolo` and no union widened an edge. `upstream_line_count` is
    // the SERVED line count (2/1/1), deliberately not the pre-expansion count the identity used.
    assert_eq!(
        report.residuals,
        vec![
            PairResidual {
                ours_index: 0,
                upstream_index: 3,
                branch: IdentityBranch::LineInformed,
                residual_full: [0, 0, 0, 0],
                residual_leg1: [0, 0, 0, 0],
                upstream_line_count: 1,
                upstream_base_xyxy_pretruncation: Some([
                    674.5721435546875,
                    1397.5679931640625,
                    740.8538208007812,
                    1438.76953125,
                ]),
            },
            PairResidual {
                ours_index: 1,
                upstream_index: 0,
                branch: IdentityBranch::LineInformed,
                residual_full: [0, 0, 0, 0],
                residual_leg1: [0, 0, 0, 0],
                upstream_line_count: 2,
                upstream_base_xyxy_pretruncation: Some([
                    567.109375,
                    74.40827941894531,
                    663.7955932617188,
                    123.6535873413086,
                ]),
            },
            PairResidual {
                ours_index: 2,
                upstream_index: 2,
                branch: IdentityBranch::LineInformed,
                residual_full: [0, 0, 0, 0],
                residual_leg1: [0, 0, 0, 0],
                upstream_line_count: 1,
                upstream_base_xyxy_pretruncation: Some([
                    607.2233276367188,
                    631.926025390625,
                    724.68408203125,
                    703.7631225585938,
                ]),
            },
        ]
    );

    // §16.25 item 4: an artifact property, not a comparison finding. This recording captured
    // `confidence` and `language` on every block, so §16.24 item 9's partition closes PRESENT
    // rather than NO-ORACLE, and no `InconsistentOracleCoverage` row is possible.
    assert_eq!(
        report.coverage,
        FieldCoverage {
            confidence: OracleCoverage::Present,
            language: OracleCoverage::Present,
            confidence_whole_artifact: OracleCoverage::Present,
            language_whole_artifact: OracleCoverage::Present,
        }
    );

    // The pairing carries our class -> language mapping across to upstream's, exactly: no
    // `LanguageDelta` row exists above, and these are the values that made that true.
    assert_eq!(
        upstream.blocks[3].language,
        Some(Language::English),
        "pair (0,3)"
    );
    assert_eq!(
        upstream.blocks[0].language,
        Some(Language::English),
        "pair (1,0)"
    );
    assert_eq!(
        upstream.blocks[2].language,
        Some(Language::Japanese),
        "pair (2,2)"
    );
}

// ── controls on the real artifacts ───────────────────────────────────────────

#[test]
// spec §16.29 item 1 — the identity operand, falsified ON THE REAL PAGE. `oracle.rs` says the
// operand is `lines_pre_expand` when recorded, else the served `lines`; upstream block 3 is the
// one real block where the two differ (`expand_size: 4` moved every edge). Dropping the recorded
// pre-expansion list must turn the gate red, or the green above would be green under either rule
// and would prove nothing about which one is implemented.
//
// Hand-derived expectation: served `lines` = (674,1393) (740,1393) (740,1442) (674,1442), bbox
// (674,1393,740,1442); union with ours (674,1397,740,1438) = (674,1393,740,1442); upstream's
// recorded `xyxy` is still (674,1397,740,1438), so exactly one `GeometryIdentity` row appears,
// before that pair's confidence row. No monotonicity row: a union only ever widens, and every
// upstream edge here is inside or equal to ours.
fn dropping_the_pre_expansion_lines_breaks_the_identity_on_the_real_page() {
    let ours = ours_blocks();
    let mut upstream = upstream_oracle();
    let (scale, image_size) = ours_frame();

    let removed = upstream.blocks[3].lines_pre_expand.take();
    assert!(
        removed.is_some(),
        "the mutation must actually remove something (cookbook rule 6a)"
    );
    assert_eq!(
        upstream.blocks[3].lines,
        vec![vec![[674, 1393], [740, 1393], [740, 1442], [674, 1442]]],
        "the fallback operand this control exercises"
    );

    let report = oracle::compare(
        OursSide {
            blocks: &ours,
            scale,
            image_size,
        },
        &upstream,
        &signed_expectations(),
    );

    let mut expected = vec![Divergence::GeometryIdentity {
        ours: Rect::new(674, 1397, 740, 1438),
        expected: Rect::new(674, 1393, 740, 1442),
        upstream: Rect::new(674, 1397, 740, 1438),
    }];
    expected.extend(expected_confidence_rows());
    assert_eq!(report.divergences, expected);
    assert_eq!(report.gating().len(), 1);
    assert_eq!(report.pairs_compared, 3);
}

#[test]
// spec §16.24 item 21 R3 / cookbook rule 13 — the ours-side box upstream's own filter dropped is
// ACCOUNTED FOR, not invisible. Deleting its `unmatched_ours` entry (and nothing else) must
// produce two gating rows, which is what proves the entry in the signed table is load-bearing
// rather than decorative.
//
// Hand-derived expectation: index 3 is neither paired nor declared, so the independent walk over
// `0..4` emits `NotAccountedFor { Ours, 3 }`; and `coverage_filtered_ours` drops to 0, so
// `ours_sum` becomes `3 + 0 + 0 + 0 = 3` against the declared `ours_total` of 4. `ours_total`
// itself still matches the artifact (4 blocks), so that row does NOT fire — only `ours_sum`.
fn removing_the_coverage_filtered_entry_leaves_the_ours_side_unaccounted() {
    let ours = ours_blocks();
    let upstream = upstream_oracle();
    let (scale, image_size) = ours_frame();

    let mut expectations = signed_expectations();
    let removed = std::mem::take(&mut expectations.unmatched_ours);
    assert_eq!(
        removed,
        vec![UnmatchedEntry {
            index: 3,
            mechanism: Mechanism::CoverageFilteredUpstream {
                pre_filter_index: 4
            },
        }],
        "the mutation must actually remove the entry under test"
    );

    let report = oracle::compare(
        OursSide {
            blocks: &ours,
            scale,
            image_size,
        },
        &upstream,
        &expectations,
    );

    let mut expected = expected_confidence_rows();
    expected.push(Divergence::NotAccountedFor {
        side: Side::Ours,
        index: 3,
    });
    expected.push(Divergence::AccountingMismatch {
        field: "ours_sum",
        declared: 4,
        actual: 3,
    });
    assert_eq!(report.divergences, expected);
    assert_eq!(report.gating().len(), 2);
}
