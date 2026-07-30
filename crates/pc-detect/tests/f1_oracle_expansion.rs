//! F1 oracle expansion tests. FROZEN per CLAUDE.md.
//!
//! spec §16.27 item 1(d) + §16.20 item 3(b) + §16.29 item 1 — which line list the identity is
//! evaluated over.
//!
//! §16.27 item 1(d) measured, on the ratified oracle page, that the English-expansion loop
//! (`textblock.py:518-532`) rewrites `blk.lines` AFTER the last `adjust_bbox` and never re-adjusts
//! `xyxy`, so on an `eng`-classified horizontal block the SERVED lines overhang `xyxy`. A bounding
//! union is monotone (`Rect::merge`), so evaluating item 3(b)'s identity over the served lines
//! makes `residual_full` non-zero on exactly the blocks §16.27 item 4 says close at `[0,0,0,0]`.
//! These controls pin the operand rule (§16.29 item 1) in both directions.
//!
//! Kept out of `f1_oracle_comparator.rs` (frozen, receives only §16.28's authorized null-completion
//! edits) so a reviewer diffing that file sees only those. `f1_oracle_derivation.rs` DOES change in
//! the same commit as this file (§16.27 item 7's four reachable-shape controls) — the two files
//! are separate for topic, not to keep either one's diff empty.

use pc_core::Rect;
use pc_detect::oracle::{
    self, Derivation, Divergence, Expectations, ExpectedPair, IdentityBranch, OracleBlock,
    OursSide, Totals, UpstreamOracle,
};
use pc_detect::RawBlock;

/// §16.24 item 17: P01 is 1200x1660 and §8.3 step 2 leaves it unresized (1660 <= the 4000
/// `input_height_upper_target` default), so `scale == 1.0`.
const FRAME_SCALE: f64 = 1.0;
const FRAME_SIZE: (u32, u32) = (1200, 1660);

fn ours_side(blocks: &[RawBlock]) -> OursSide<'_> {
    OursSide {
        blocks,
        scale: FRAME_SCALE,
        image_size: FRAME_SIZE,
    }
}

fn one_block(block: OracleBlock) -> UpstreamOracle {
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

/// A bbox over polygon points computed HERE, independently of `OracleBlock::lines_bbox` — the
/// function whose operand these tests are about.
fn bbox_of(lines: &[Vec<[i32; 2]>]) -> Option<Rect> {
    let mut points = lines.iter().flatten();
    let first = points.next()?;
    let (mut x1, mut y1, mut x2, mut y2) = (first[0], first[1], first[0], first[1]);
    for point in points {
        x1 = x1.min(point[0]);
        y1 = y1.min(point[1]);
        x2 = x2.max(point[0]);
        y2 = y2.max(point[1]);
    }
    Some(Rect::new(x1, y1, x2, y2))
}

fn corners(xyxy: [i32; 4]) -> Vec<Vec<[i32; 2]>> {
    vec![vec![
        [xyxy[0], xyxy[1]],
        [xyxy[2], xyxy[1]],
        [xyxy[2], xyxy[3]],
        [xyxy[0], xyxy[3]],
    ]]
}

/// P01's block 3 as §16.27 item 1(d) measured it: the `YoloSynthesizedCorners` case, `xyxy =
/// [674,1397,740,1438]`, `expand_size = 4`, served `bbox(lines) = [674,1393,740,1442]`.
const P01_BLOCK3_XYXY: [i32; 4] = [674, 1397, 740, 1438];
const P01_BLOCK3_SERVED_LINES_BBOX: [i32; 4] = [674, 1393, 740, 1442];

fn p01_block3() -> OracleBlock {
    OracleBlock {
        xyxy: P01_BLOCK3_XYXY,
        lines: corners(P01_BLOCK3_SERVED_LINES_BBOX),
        confidence: None,
        language: None,
        base_xyxy_pretruncation: None,
        derivation: Some(Derivation::YoloSynthesizedCorners),
        rect_yolo: Some(P01_BLOCK3_XYXY),
        eng_expanded: true,
        lines_pre_expand: Some(corners(P01_BLOCK3_XYXY)),
        expand_size: Some(4),
    }
}

#[test]
// The identity is evaluated over `lines_pre_expand` when the recorder captured it, so the block
// §16.27 item 4 says closes at `residual_full == [0,0,0,0]` actually does.
fn the_identity_is_evaluated_over_the_pre_expansion_lines_when_they_were_recorded() {
    let upstream = one_block(p01_block3());

    assert_eq!(
        bbox_of(&upstream.blocks[0].lines),
        Some(oracle::xyxy_to_rect(P01_BLOCK3_SERVED_LINES_BBOX))
    );
    assert_eq!(
        bbox_of(
            upstream.blocks[0]
                .lines_pre_expand
                .as_deref()
                .expect("recorded")
        ),
        Some(oracle::xyxy_to_rect(P01_BLOCK3_XYXY))
    );

    let ours = vec![RawBlock {
        rect: oracle::xyxy_to_rect(P01_BLOCK3_XYXY),
        class_index: 0,
        confidence: 0.775,
    }];
    let report = oracle::compare(
        ours_side(&ours),
        &upstream,
        &one_pair(
            IdentityBranch::LineInformed,
            Some(Derivation::YoloSynthesizedCorners),
        ),
    );

    assert_eq!(report.divergences, Vec::new());
    assert_eq!(report.pairs_compared, 1);
    assert_eq!(report.residuals.len(), 1);
    assert_eq!(report.residuals[0].residual_full, [0, 0, 0, 0]);
    assert_eq!(report.residuals[0].residual_leg1, [0, 0, 0, 0]);
    assert_eq!(report.leg1_rows_checked, 1);
    assert_eq!(report.residuals[0].upstream_line_count, 1);
    assert_eq!(
        report.branches,
        vec![(0, 0, IdentityBranch::LineInformed)],
        "the pre-expansion list is non-empty, so the branch is line-informed"
    );
}

#[test]
// Without a recorded pre-expansion list the operand is the served `lines`, unchanged — this is
// what keeps every existing frozen ExpectedPair/OracleBlock literal in f1_oracle_comparator.rs
// unaffected by this task (all have lines_pre_expand: None or identical-to-lines).
fn without_recorded_pre_expansion_lines_the_served_lines_remain_the_operand() {
    let mut block = p01_block3();
    block.lines_pre_expand = None;
    block.expand_size = None;
    block.eng_expanded = false;

    let ours = vec![RawBlock {
        rect: oracle::xyxy_to_rect(P01_BLOCK3_XYXY),
        class_index: 0,
        confidence: 0.775,
    }];
    let report = oracle::compare(
        ours_side(&ours),
        &one_block(block),
        &one_pair(
            IdentityBranch::LineInformed,
            Some(Derivation::YoloSynthesizedCorners),
        ),
    );

    assert_eq!(
        report.divergences,
        vec![Divergence::GeometryIdentity {
            ours: oracle::xyxy_to_rect(P01_BLOCK3_XYXY),
            expected: oracle::xyxy_to_rect(P01_BLOCK3_SERVED_LINES_BBOX),
            upstream: oracle::xyxy_to_rect(P01_BLOCK3_XYXY),
        }]
    );
    assert_eq!(report.residuals[0].residual_full, [0, 4, 0, -4]);
}

#[test]
// The line census is over the SERVED polygons, not the pre-expansion ones (§16.27 item 8's claim
// is about what upstream serves) — so upstream_line_count must not follow the identity operand.
fn the_line_census_counts_the_served_polygons_not_the_pre_expansion_ones() {
    let mut block = p01_block3();
    block.lines = vec![
        corners(P01_BLOCK3_SERVED_LINES_BBOX).remove(0),
        vec![[700, 1400], [710, 1400], [710, 1410], [700, 1410]],
    ];

    let ours = vec![RawBlock {
        rect: oracle::xyxy_to_rect(P01_BLOCK3_XYXY),
        class_index: 0,
        confidence: 0.775,
    }];
    let report = oracle::compare(
        ours_side(&ours),
        &one_block(block),
        &one_pair(
            IdentityBranch::LineInformed,
            Some(Derivation::YoloSynthesizedCorners),
        ),
    );

    assert_eq!(report.residuals[0].upstream_line_count, 2);
    assert_eq!(report.divergences, Vec::new());
}

#[test]
// The degenerate case on the NEW operand: a recorded-but-empty `lines_pre_expand` must be
// LineLess and must NOT be unioned as a point at the origin.
fn a_recorded_but_empty_pre_expansion_list_is_line_less_and_is_not_unioned_at_the_origin() {
    let mut block = p01_block3();
    block.lines_pre_expand = Some(vec![vec![]]);

    let ours_rect = oracle::xyxy_to_rect(P01_BLOCK3_XYXY);
    let ours = vec![RawBlock {
        rect: ours_rect,
        class_index: 0,
        confidence: 0.775,
    }];
    let report = oracle::compare(
        ours_side(&ours),
        &one_block(block),
        &one_pair(
            IdentityBranch::LineLess,
            Some(Derivation::YoloSynthesizedCorners),
        ),
    );
    assert_eq!(report.divergences, Vec::new());
}
