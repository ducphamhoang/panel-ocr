//! spec §16.27 items 1(a), 2, 2(c) and 3 + §16.28 items 2, 4, 5 and 6 — the derivation axis's own
//! controls. Kept separate from f1_oracle_comparator.rs (which is frozen and only receives the
//! exact null-completion edits §16.28 authorizes) so a reviewer diffing that file sees only those
//! edits and nothing else.

use pc_core::Rect;
use pc_detect::oracle::{
    self, Derivation, DerivedAccounting, Divergence, Expectations, ExpectedPair, FieldCoverage,
    IdentityBranch, Mechanism, OracleBlock, OursSide, Side, Totals, UnmatchedEntry, UpstreamOracle,
};
use pc_detect::RawBlock;

const FRAME_SCALE: f64 = 0.617;
const FRAME_SIZE: (u32, u32) = (1024, 1434);

fn ours_side(blocks: &[RawBlock]) -> OursSide<'_> {
    OursSide {
        blocks,
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
    }
}

fn one_block_frame(block: OracleBlock) -> UpstreamOracle {
    UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![block],
        pre_filter_blocks: vec![],
    }
}

fn one_pair(branch: IdentityBranch, derivation: Option<Derivation>) -> Expectations {
    Expectations {
        pairs: vec![ExpectedPair {
            ours: 0,
            upstream: 0,
            branch,
            derivation,
        }],
        unmatched_ours: vec![],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 1,
            ours_total: 1,
            upstream_total: 1,
        },
    }
}

fn ours_item12() -> Vec<RawBlock> {
    vec![RawBlock {
        rect: Rect::new(29, 105, 86, 261),
        class_index: 0,
        confidence: 0.564,
    }]
}

fn item12_lines() -> Vec<Vec<[i32; 2]>> {
    vec![vec![[33, 110], [90, 110], [90, 250], [33, 250]]]
}

/// spec §16.27 item 2 — the exact, epsilon-free `ours.rect == rect_yolo` leg-1 gate fires when
/// the recorded YOLO geometry differs from ours.
#[test]
fn the_leg_one_row_fires_when_our_box_differs_from_the_recorded_yolo_box() {
    let ours = ours_item12();
    let upstream = one_block_frame(OracleBlock {
        xyxy: [29, 105, 90, 261],
        lines: item12_lines(),
        confidence: None,
        language: None,
        base_xyxy_pretruncation: None,
        derivation: Some(Derivation::YoloUnioned),
        rect_yolo: Some([29, 105, 85, 261]),
        eng_expanded: false,
        lines_pre_expand: Some(item12_lines()),
        expand_size: None,
    });
    let expectations = one_pair(IdentityBranch::LineInformed, Some(Derivation::YoloUnioned));

    let report = oracle::compare(ours_side(&ours), &upstream, &expectations);

    assert_eq!(
        report.divergences,
        vec![Divergence::Leg1YoloGeometry {
            ours: Rect::new(29, 105, 86, 261),
            rect_yolo: Some([29, 105, 85, 261]),
            derivation: Derivation::YoloUnioned,
        }]
    );
    assert_eq!(report.gating().len(), 1, "gating: {:?}", report.gating());
    assert_eq!(report.leg1_rows_checked, 1);
    assert!(
        !report
            .divergences
            .iter()
            .any(|row| matches!(row, Divergence::GeometryIdentity { .. })),
        "the identity holds on this pair; the new row must catch what it cannot: {:?}",
        report.divergences
    );
    assert_eq!(report.residuals.len(), 1);
    assert_eq!(report.residuals[0].residual_full, [0, 0, 0, 0]);
}

/// spec §16.27 item 2 and §16.28 item 5 — the leg-1 row is evaluated and can pass non-trivially,
/// with the evaluation count preventing a vacuous green result.
#[test]
fn the_leg_one_row_is_evaluated_and_passes_where_the_yolo_box_is_not_the_upstream_box() {
    let ours = ours_item12();
    let rect_yolo = [29, 105, 86, 261];
    let xyxy = [29, 105, 90, 261];
    assert_ne!(
        rect_yolo, xyxy,
        "a pass where rect_yolo == xyxy would be implied by GeometryIdentity and prove nothing"
    );

    let upstream = one_block_frame(OracleBlock {
        xyxy,
        lines: item12_lines(),
        confidence: None,
        language: None,
        base_xyxy_pretruncation: None,
        derivation: Some(Derivation::YoloUnioned),
        rect_yolo: Some(rect_yolo),
        eng_expanded: false,
        lines_pre_expand: Some(item12_lines()),
        expand_size: None,
    });
    let expectations = one_pair(IdentityBranch::LineInformed, Some(Derivation::YoloUnioned));

    let report = oracle::compare(ours_side(&ours), &upstream, &expectations);

    assert_eq!(report.divergences, Vec::new());
    assert!(report.gating().is_empty());
    assert_eq!(report.pairs_compared, 1);
    assert_eq!(
        report.leg1_rows_checked, 1,
        "a green run with the row never evaluated is the vacuous pass this counter forbids"
    );
}

/// spec §16.27 item 2(c) and §16.28 item 6 — `YoloSplit` and `DbnetScattered` pairs are rejected
/// structurally, while pairable derivations retain the leg-1 evaluation.
#[test]
fn a_split_or_scattered_derivation_may_not_be_paired_and_the_leg_one_row_is_not_run() {
    let ours_split = vec![RawBlock {
        rect: Rect::new(300, 400, 365, 440),
        class_index: 0,
        confidence: 0.5,
    }];
    let sub_lines = vec![vec![[300, 400], [360, 400], [360, 440], [300, 440]]];
    let split = one_block_frame(OracleBlock {
        xyxy: [300, 400, 360, 440],
        lines: sub_lines.clone(),
        confidence: None,
        language: None,
        base_xyxy_pretruncation: None,
        derivation: Some(Derivation::YoloSplit),
        rect_yolo: Some([280, 390, 400, 460]),
        eng_expanded: false,
        lines_pre_expand: Some(sub_lines),
        expand_size: None,
    });
    let report = oracle::compare(
        ours_side(&ours_split),
        &split,
        &one_pair(IdentityBranch::LineInformed, Some(Derivation::YoloSplit)),
    );
    assert_eq!(
        report.divergences,
        vec![Divergence::UnpairableDerivation {
            ours_index: 0,
            upstream_index: 0,
            derivation: Derivation::YoloSplit,
        }]
    );
    assert_eq!(report.gating().len(), 1);
    assert_eq!(report.leg1_rows_checked, 0);

    let ours_scattered = vec![RawBlock {
        rect: Rect::new(500, 600, 560, 640),
        class_index: 0,
        confidence: 0.5,
    }];
    let scattered_lines = vec![vec![[500, 600], [560, 600], [560, 640], [500, 640]]];
    let ours_scattered_misaligned = vec![RawBlock {
        rect: Rect::new(495, 600, 560, 640),
        class_index: 0,
        confidence: 0.5,
    }];
    let scattered = one_block_frame(OracleBlock {
        xyxy: [500, 600, 560, 640],
        lines: scattered_lines.clone(),
        confidence: None,
        language: None,
        base_xyxy_pretruncation: None,
        derivation: Some(Derivation::DbnetScattered),
        rect_yolo: None,
        eng_expanded: false,
        lines_pre_expand: Some(scattered_lines),
        expand_size: None,
    });
    let report = oracle::compare(
        ours_side(&ours_scattered_misaligned),
        &scattered,
        &one_pair(
            IdentityBranch::LineInformed,
            Some(Derivation::DbnetScattered),
        ),
    );
    assert_eq!(
        report.divergences,
        vec![Divergence::UnpairableDerivation {
            ours_index: 0,
            upstream_index: 0,
            derivation: Derivation::DbnetScattered,
        }]
    );
    assert_eq!(report.gating().len(), 1);
    assert_eq!(report.leg1_rows_checked, 0);

    for derivation in [Derivation::YoloUnioned, Derivation::YoloSynthesizedCorners] {
        let corners = vec![vec![[500, 600], [560, 600], [560, 640], [500, 640]]];
        let pairable = one_block_frame(OracleBlock {
            xyxy: [500, 600, 560, 640],
            lines: corners.clone(),
            confidence: None,
            language: None,
            base_xyxy_pretruncation: None,
            derivation: Some(derivation),
            rect_yolo: Some([500, 600, 560, 640]),
            eng_expanded: false,
            lines_pre_expand: Some(corners),
            expand_size: None,
        });
        let report = oracle::compare(
            ours_side(&ours_scattered),
            &pairable,
            &one_pair(IdentityBranch::LineInformed, Some(derivation)),
        );
        assert_eq!(report.divergences, Vec::new(), "{derivation:?}");
        assert_eq!(report.leg1_rows_checked, 1, "{derivation:?}");
    }
}

/// spec §16.28 items 2 and 4 — a dropped recorded derivation is caught by the signed signature,
/// while the derivation-conditional geometry machinery is disabled for that pair.
#[test]
fn a_derivation_the_recorder_dropped_is_caught_by_the_signed_pair_and_not_by_silence() {
    let ours = ours_item12();

    let upstream: UpstreamOracle = serde_json::from_str(
        r#"{
          "scale": 0.617,
          "image_size": [1024, 1434],
          "blocks": [
            {
              "xyxy": [29, 105, 90, 261],
              "lines": [[[33, 110], [90, 110], [90, 250], [33, 250]]],
              "rect_yolo": [29, 105, 86, 261]
            }
          ]
        }"#,
    )
    .expect("a block omitting `derivation` must still parse");
    assert_eq!(
        upstream.blocks[0].derivation, None,
        "the premise of this test is that the field really is absent"
    );

    let expectations = one_pair(IdentityBranch::LineInformed, Some(Derivation::YoloUnioned));
    let report = oracle::compare(ours_side(&ours), &upstream, &expectations);

    assert_eq!(
        report.divergences,
        vec![Divergence::DerivationSignatureMismatch {
            ours: Rect::new(29, 105, 86, 261),
            signed: Some(Derivation::YoloUnioned),
            recorded: None,
        }]
    );
    assert_eq!(report.gating().len(), 1);
    assert_eq!(
        report.leg1_rows_checked, 0,
        "an absent derivation disables the leg-1 row — the silent pass this variant catches"
    );
    assert!(!report
        .divergences
        .iter()
        .any(|row| matches!(row, Divergence::GeometryIdentity { .. })));
}

/// spec §16.28 item 2 — the signed and recorded derivations must match by exact bidirectional
/// equality, including both directions involving `None`.
#[test]
fn the_signed_derivation_check_is_exact_equality_in_both_directions() {
    use Derivation::{YoloSynthesizedCorners, YoloUnioned};

    let ours = vec![RawBlock {
        rect: Rect::new(100, 200, 160, 240),
        class_index: 0,
        confidence: 0.5,
    }];

    let cases: [(Option<Derivation>, Option<Derivation>, bool, usize); 5] = [
        (Some(YoloUnioned), Some(YoloSynthesizedCorners), true, 1),
        (Some(YoloUnioned), None, true, 0),
        (None, Some(YoloUnioned), true, 1),
        (Some(YoloUnioned), Some(YoloUnioned), false, 1),
        (None, None, false, 0),
    ];

    for (signed, recorded, expect_row, expect_checked) in cases {
        let upstream = one_block_frame(OracleBlock {
            xyxy: [100, 200, 160, 240],
            lines: vec![],
            confidence: None,
            language: None,
            base_xyxy_pretruncation: None,
            derivation: recorded,
            rect_yolo: Some([100, 200, 160, 240]),
            eng_expanded: false,
            lines_pre_expand: None,
            expand_size: None,
        });
        let report = oracle::compare(
            ours_side(&ours),
            &upstream,
            &one_pair(IdentityBranch::LineLess, signed),
        );

        let expected = if expect_row {
            vec![Divergence::DerivationSignatureMismatch {
                ours: Rect::new(100, 200, 160, 240),
                signed,
                recorded,
            }]
        } else {
            Vec::new()
        };
        assert_eq!(
            report.divergences, expected,
            "signed {signed:?} vs recorded {recorded:?}"
        );
        assert_eq!(
            report.leg1_rows_checked, expect_checked,
            "keys on the RECORDED value: signed {signed:?} vs recorded {recorded:?}"
        );
    }
}

/// spec §16.27 item 3(a) and (b) — `DbnetScattered` is an upstream-only mechanism with its own
/// derived accounting term, and cannot be used to explain an ours-side unmatched block.
#[test]
fn a_scattered_upstream_block_closes_on_its_own_mechanism_and_its_own_accounting_term() {
    let ours = vec![RawBlock {
        rect: Rect::new(100, 200, 160, 240),
        class_index: 0,
        confidence: 0.5,
    }];
    let scattered_lines = vec![vec![[700, 800], [760, 800], [760, 830], [700, 830]]];
    let upstream = UpstreamOracle {
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
        blocks: vec![
            OracleBlock {
                xyxy: [100, 200, 160, 240],
                lines: vec![],
                confidence: None,
                language: None,
                base_xyxy_pretruncation: None,
                derivation: None,
                rect_yolo: None,
                eng_expanded: false,
                lines_pre_expand: None,
                expand_size: None,
            },
            OracleBlock {
                xyxy: [700, 800, 760, 830],
                lines: scattered_lines.clone(),
                confidence: None,
                language: None,
                base_xyxy_pretruncation: None,
                derivation: Some(Derivation::DbnetScattered),
                rect_yolo: None,
                eng_expanded: false,
                lines_pre_expand: Some(scattered_lines),
                expand_size: None,
            },
        ],
        pre_filter_blocks: vec![],
    };
    let expectations = Expectations {
        pairs: vec![ExpectedPair {
            ours: 0,
            upstream: 0,
            branch: IdentityBranch::LineLess,
            derivation: None,
        }],
        unmatched_ours: vec![],
        unmatched_upstream: vec![UnmatchedEntry {
            index: 1,
            mechanism: Mechanism::DbnetScattered {
                register_entry: "§16.27 item 3",
            },
        }],
        totals: Totals {
            pairs: 1,
            ours_total: 1,
            upstream_total: 2,
        },
    };

    let report = oracle::compare(ours_side(&ours), &upstream, &expectations);

    assert_eq!(report.divergences, Vec::new());

    assert_eq!(
        report.derived,
        DerivedAccounting {
            pairs: 1,
            class_duplicates: 0,
            documented_split_merge_upstream: 0,
            dbnet_scattered: 1,
            open_upstream: 0,
            coverage_filtered_ours: 0,
            documented_split_merge_ours: 0,
            open_ours: 0,
        }
    );
    assert_eq!(report.derived.upstream_sum(), 2);
    assert_eq!(report.derived.ours_sum(), 1);

    let ours_two = vec![
        RawBlock {
            rect: Rect::new(100, 200, 160, 240),
            class_index: 0,
            confidence: 0.5,
        },
        RawBlock {
            rect: Rect::new(700, 800, 760, 830),
            class_index: 0,
            confidence: 0.5,
        },
    ];
    let upstream_one = one_block_frame(OracleBlock {
        xyxy: [100, 200, 160, 240],
        lines: vec![],
        confidence: None,
        language: None,
        base_xyxy_pretruncation: None,
        derivation: None,
        rect_yolo: None,
        eng_expanded: false,
        lines_pre_expand: None,
        expand_size: None,
    });
    let wrong_side = Expectations {
        pairs: vec![ExpectedPair {
            ours: 0,
            upstream: 0,
            branch: IdentityBranch::LineLess,
            derivation: None,
        }],
        unmatched_ours: vec![UnmatchedEntry {
            index: 1,
            mechanism: Mechanism::DbnetScattered {
                register_entry: "§16.27 item 3",
            },
        }],
        unmatched_upstream: vec![],
        totals: Totals {
            pairs: 1,
            ours_total: 2,
            upstream_total: 1,
        },
    };
    let report = oracle::compare(ours_side(&ours_two), &upstream_one, &wrong_side);
    assert!(
        report.divergences.iter().any(|d| matches!(
            d,
            Divergence::UnverifiableMechanism {
                side: Side::Ours,
                index: 1,
                ..
            }
        )),
        "got {:?}",
        report.divergences
    );
    assert!(!report.gating().is_empty());
}

/// spec §16.27 item 1(a) — the recorded derivation is a closed enum with exactly the four pinned
/// snake-case spellings and no catch-all value.
#[test]
fn the_derivation_enum_is_closed_and_its_four_spellings_are_pinned() {
    for (spelling, expected) in [
        ("yolo_unioned", Derivation::YoloUnioned),
        (
            "yolo_synthesized_corners",
            Derivation::YoloSynthesizedCorners,
        ),
        ("yolo_split", Derivation::YoloSplit),
        ("dbnet_scattered", Derivation::DbnetScattered),
    ] {
        let doc = format!(
            r#"{{"scale":1.0,"image_size":[1,1],"blocks":[{{"xyxy":[1,2,3,4],"derivation":"{spelling}"}}]}}"#
        );
        let parsed: UpstreamOracle =
            serde_json::from_str(&doc).unwrap_or_else(|e| panic!("`{spelling}` must parse: {e}"));
        assert_eq!(parsed.blocks[0].derivation, Some(expected));
    }

    for rejected in ["other", "unknown", "YoloUnioned", "yolounioned", ""] {
        let doc = format!(
            r#"{{"scale":1.0,"image_size":[1,1],"blocks":[{{"xyxy":[1,2,3,4],"derivation":"{rejected}"}}]}}"#
        );
        assert!(
            serde_json::from_str::<UpstreamOracle>(&doc).is_err(),
            "`{rejected}` must not parse as a derivation"
        );
    }
}

/// spec §16.28 item 2 — derivation signatures are carried on the signed pair, while the existing
/// `FieldCoverage` channel remains limited to its ratified coverage fields.
#[test]
fn field_coverage_carries_no_derivation_member() {
    let ours = ours_item12();
    let upstream = one_block_frame(OracleBlock {
        xyxy: [29, 105, 86, 261],
        lines: vec![],
        confidence: None,
        language: None,
        base_xyxy_pretruncation: None,
        derivation: None,
        rect_yolo: None,
        eng_expanded: false,
        lines_pre_expand: None,
        expand_size: None,
    });
    let report = oracle::compare(
        ours_side(&ours),
        &upstream,
        &one_pair(IdentityBranch::LineLess, None),
    );

    let FieldCoverage {
        confidence: _,
        language: _,
        confidence_whole_artifact: _,
        language_whole_artifact: _,
    } = report.coverage;
}
