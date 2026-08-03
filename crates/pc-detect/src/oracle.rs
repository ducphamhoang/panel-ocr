//! spec §16.20 item 3(b), as ratified by §16.24 items 4–8 — the committed-oracle comparator.
//!
//! Both sides are frozen committed files, so this is pure arithmetic over two JSON documents and
//! belongs in CI on every run (§16.20 item 11: "producing the two artifacts runs both detectors
//! ... comparing them runs no model"). Nothing here loads a model, decodes an image, or touches
//! the filesystem.
//!
//! **There is deliberately NO real-page gate yet, and no committed oracle artifacts. That is the
//! ratified staging, not an unfinished edge.** Every test against this module builds its inputs
//! in-code. §16.20 item 10 requires the comparator to "ship with its own negative controls"
//! precisely so that it is *not* first exercised on the real pair — a comparator whose failure
//! modes were never observed is the thing that item warns about. And §16.24 item 6 fixes the other
//! half: **"the recording commit is atomic … until then no real-page gate test exists"** — the
//! fixture, its provenance, the frozen `EXPECTED_*` edits, the real-page gate with its literal
//! [`Expectations`], `docs/DETECTOR_ORACLE.md`'s verdicts and §16.20 item 3(f)'s three signatures
//! all land in one commit or none.
//!
//! So "synthetic controls only" is the correct state for this file to be committed in, and it stays
//! correct until that single commit lands. A reviewer reading the absence of a real-page gate as
//! incompleteness has found the design, not a defect — this note exists because that reading was
//! made once already.
//!
//! **Pairing is a SIGNED INPUT (§16.24 item 4).** [`Expectations`] is a required by-value
//! argument and there is deliberately no constructor that discovers a pairing from the data: an
//! empty pairing is something a caller wrote, never something this module can produce. No
//! distance constant appears anywhere in this file — §16.24 item 4(a) rejected
//! `PAIRING_CENTRE_L1_LIMIT` as an unmeasured magnitude threshold on geometry, and §16.20 item 5
//! is the reason (no defensible magnitude exists in this territory). §16.20 item 3(c) is scoped to
//! gating verdicts, which is not license for a correspondence bound.
//!
//! **What the caller signs, and what this module verifies.** The caller authors the pairing, each
//! pair's [`IdentityBranch`], every unmatched block with a [`Mechanism`], and the totals — that is
//! the judgment §16.20 item 3(f)'s three signatures exist to cover. This module then checks that
//! the authored table is *internally sound and consistent with the artifacts*: every index resolves,
//! nothing is paired twice, **every block on both sides is either paired or listed**, each
//! mechanism's citation is verifiable against recorded evidence, the declared branch matches the
//! data, the identity holds on every pair, and both accounting equations close.
//!
//! **No boolean success channel (§16.24 item 6).** There is no `is_ok()`. A CI gate reads
//! `report.gating().is_empty()` AND `report.pairs_compared == <authored literal>`.
//!
//! A `derivation == None` artifact block disables the leg-1 row, the structural guard, and the
//! derivation-conditional §16.24 item 18(i) message for its pair — but NOT the signature check
//! below, which is unconditional. That's what makes the None-disabling safe: a dropped derivation
//! is still caught by the signature mismatch even though the geometry checks it would have fed are
//! off.
//!
//! The comparator does NOT re-check the four derivation-law equations from spec §16.27 item 1(c)
//! — those are the recorder's responsibility, not the comparator's.

use crate::detector::RawBlock;
use crate::yolo::class_to_language;
use pc_core::{Language, Rect};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ─────────────────────────────────────────────────────── the oracle artifact

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Derivation {
    YoloUnioned,
    YoloSynthesizedCorners,
    YoloSplit,
    DbnetScattered,
}

/// One upstream block as recorded by the oracle script (`upstream + cv2.dnn`, §16.20 item 6).
///
/// The identity operand is `lines_pre_expand` when the artifact recorded it (`Some`), otherwise
/// the served `lines`. [`lines_bbox`], [`identity_branch`], and [`reconstruct`] all use that same
/// operand; the served `lines` remain authoritative for [`PairResidual::upstream_line_count`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleBlock {
    /// upstream `TextBlock.xyxy`, `[x1, y1, x2, y2]`, `x2`/`y2` **exclusive** — see
    /// [`rect_to_xyxy`] for why that needs no adjustment against our `Rect`.
    pub xyxy: [i32; 4],
    /// upstream `TextBlock.lines`: zero or more polygons of `[x, y]` points. The identity must
    /// still handle an empty list (§16.20 item 3(b)'s `LineLess` branch) as an artifact-grammar
    /// property, not because a real *output* block is ever line-less — §16.27 item 8 measured no
    /// gated upstream block carries zero lines (0 of 14 on the three candidate pages; every path
    /// through `group_output` guarantees at least one line, §16.27 erratum 4). What §14.17's
    /// filter divergence actually exercises is upstream's line-less **pre-filter** branch
    /// (`mask_score` 0.0359 on `[438,1407,498,1446]`, §16.20 item 9) — a different thing from a
    /// line-less *output* block, a conflation §16.27 erratum 4 supersedes this comment's earlier
    /// wording for.
    #[serde(default)]
    pub lines: Vec<Vec<[i32; 2]>>,
    /// §16.20 item 3(d) DIAGNOSTIC, never gated. `None` when the recording could not capture it
    /// (§16.24 item 9: upstream's `#raw.json` does not persist confidence, so the script attempts
    /// capture from `postprocess_yolo` and the row closes NO-ORACLE if that fails).
    #[serde(default)]
    pub confidence: Option<f32>,
    /// §16.20 item 3(d) DIAGNOSTIC, never gated.
    #[serde(default)]
    pub language: Option<Language>,
    /// §16.24 item 18(e)'s discriminator, DIAGNOSTIC and optional: upstream's box in base-image
    /// space **before** `astype(np.int32)` truncation. `None` when the recording did not probe for
    /// it. Passed through into [`PairResidual`] untouched — the comparator cannot compute it, and
    /// our own side's equivalent needs a probe inside `yolo::rescale` that lives in the
    /// commissioned demo_bubbles run, not here (plan §1.6).
    #[serde(default)]
    pub base_xyxy_pretruncation: Option<[f64; 4]>,
    #[serde(default)]
    pub derivation: Option<Derivation>,
    #[serde(default)]
    pub rect_yolo: Option<[i32; 4]>,
    /// §16.28 item 1(e)'s "REQUIRED together" condition binds the recorder, not this Rust
    /// representation, so the three expansion fields remain plain fields rather than a grouped
    /// type that would make a malformed recording unrepresentable here.
    #[serde(default)]
    pub eng_expanded: bool,
    /// Required together with [`Self::expand_size`] when [`Self::eng_expanded`] is true by
    /// §16.28 item 1(e); the requirement is recorded-data validation, not a Rust type invariant.
    #[serde(default)]
    pub lines_pre_expand: Option<Vec<Vec<[i32; 2]>>>,
    /// Required together with [`Self::lines_pre_expand`] when [`Self::eng_expanded`] is true by
    /// §16.28 item 1(e); it stays an `Option` because that condition binds the recorder.
    #[serde(default)]
    pub expand_size: Option<i32>,
}

/// The committed upstream oracle artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamOracle {
    /// Cookbook rule 7 names `scale` and `image_size` as genuine upstream oracles, so both are
    /// GATED (§16.24 item 8) — they are what pins the two sides to one coordinate frame.
    pub scale: f64,
    pub image_size: (u32, u32),
    /// **Post-`group_output`** — the gated side (§16.24 item 8), because that is where `lines`
    /// exist and where the identity of item 3(b) is defined.
    pub blocks: Vec<OracleBlock>,
    /// **Pre-`group_output`** — a DIAGNOSTIC list, never gated (§16.24 item 8's adjustment). It
    /// exists so a [`Mechanism::CoverageFilteredUpstream`] entry points at recorded evidence
    /// rather than at an inference.
    #[serde(default)]
    pub pre_filter_blocks: Vec<OracleBlock>,
}

/// Our side. Note the two artifacts: blocks come from the PRE-filter replay fixture, the frame
/// from `#raw.json` (§16.24 item 8).
#[derive(Debug, Clone, Copy)]
pub struct OursSide<'a> {
    /// `<stem>_detector_blocks.json` — §16.20 item 2 calls it "the artifact that genuinely
    /// accepts output as truth", and it is what `ReplayDetector` replays. **Pre** coverage
    /// filter: `crates/pc-detect/src/lib.rs:106-118` filters on `mask_coverage` before assembling
    /// `PageDataRaw`, so `#raw.json`'s blocks are a subset and are locked separately by
    /// §8.7(A)6/(B)9.
    pub blocks: &'a [RawBlock],
    /// From `<stem>#raw.json`.
    pub scale: f64,
    pub image_size: (u32, u32),
}

// ───────────────────────────────────────────────────── the identity itself

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityBranch {
    /// `upstream.xyxy == bbox(ours.rect ∪ OracleBlock::identity_lines())` (§16.29 item 1:
    /// `lines_pre_expand` when recorded, else the served `lines` — see `OracleBlock`'s doc).
    LineInformed,
    /// §16.20 item 3's degenerate case: `OracleBlock::identity_lines()` is undefined (no points at
    /// all), so the identity is `upstream.xyxy == ours.rect`.
    LineLess,
}

/// spec §2.1 — the ONE place our `Rect` becomes upstream's `xyxy`.
///
/// It is the identity map, and that is the claim: upstream slices `mask[by1:by2, bx1:bx2]`
/// (`textblock.py:485-490`) where Python's stop is exclusive, and §2.1 makes our `x2`/`y2`
/// exclusive for cropping/rasterising. No ±1 adjustment is correct. This is isolated in one
/// function with a hand-computed test (§16.24 item 6) because §2.1's convention is deliberately
/// mixed — `x2` is *inclusive* for `Rect::contains` (`geometry.rs:46-49`) — so the next reader has
/// exactly one place to check instead of an assumption spread over the comparator.
pub fn rect_to_xyxy(rect: Rect) -> [i32; 4] {
    [rect.x1, rect.y1, rect.x2, rect.y2]
}

pub fn xyxy_to_rect(xyxy: [i32; 4]) -> Rect {
    Rect::new(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
}

impl OracleBlock {
    pub fn rect(&self) -> Rect {
        xyxy_to_rect(self.xyxy)
    }

    /// The line polygons used by the geometry identity: pre-expansion lines when recorded, or
    /// the served lines when no pre-expansion list was recorded.
    pub fn identity_lines(&self) -> &[Vec<[i32; 2]>] {
        self.lines_pre_expand.as_deref().unwrap_or(&self.lines)
    }

    /// Which branch of the identity this block's data selects. An empty polygon list — and a list
    /// of empty polygons — is `LineLess`; treating either as a point at the origin and unioning it
    /// would drag every box's `x1`/`y1` to 0.
    pub fn identity_branch(&self) -> IdentityBranch {
        match self.lines_bbox() {
            Some(_) => IdentityBranch::LineInformed,
            None => IdentityBranch::LineLess,
        }
    }

    /// `bbox` over every point of every polygon; `None` when there are no points at all.
    pub fn lines_bbox(&self) -> Option<Rect> {
        let mut points = self.identity_lines().iter().flatten();
        let first = points.next()?;
        let mut bbox = Rect::new(first[0], first[1], first[0], first[1]);
        for point in points {
            bbox = bbox.merge(&Rect::new(point[0], point[1], point[0], point[1]));
        }
        Some(bbox)
    }

    /// §16.20 item 3(b)'s right-hand side, plus the branch it took so the caller can assert it
    /// rather than trust it (§16.24 item 6).
    pub fn reconstruct(&self, ours: Rect) -> (Rect, IdentityBranch) {
        match self.lines_bbox() {
            Some(lines) => (ours.merge(&lines), IdentityBranch::LineInformed),
            None => (ours, IdentityBranch::LineLess),
        }
    }
}

// ───────────────────────────────────────────── the signed input

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Ours,
    Upstream,
}

/// Why a block on one side has no counterpart on the other. Every unmatched block carries one
/// (§16.24 item 4), and each variant's citation is *checkable* — a mechanism nobody can falsify
/// is a rubber stamp, which is cookbook rule 12's failure one level up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mechanism {
    /// §14.13 — upstream's per-class NMS emitted a second box for one balloon and our
    /// class-agnostic NMS collapsed it. **Upstream side only.** Cites the upstream block it
    /// duplicates; the citation is verified to be in range and itself paired, which is
    /// threshold-free (an overlap test would need the IoU §16.20 item 3(c) forbids).
    ClassDuplicateOf { upstream_index: usize },
    /// §14.17 / §16.20 item 9 — upstream's line-less coverage filter dropped this box, so it is
    /// present in our pre-filter list and absent from the post-`group_output` oracle. **Ours side
    /// only.** Cites the index in [`UpstreamOracle::pre_filter_blocks`] carrying the evidence;
    /// the comparator verifies that block satisfies the identity against our rect, so the claim
    /// "upstream had this box and then filtered it" is proven, not asserted.
    CoverageFilteredUpstream { pre_filter_index: usize },
    /// A split or merge documented by a §14 / §16.x register entry, named here. The anchor's
    /// existence in the spec is checked by F1-E's `EXPLAINED`-anchor doc gate, not here — this
    /// module reads no documents, and saying so is better than implying otherwise.
    DocumentedSplitMerge { register_entry: &'static str },
    /// Unexplained. **Blocking**: §16.24 item 4 — "`Open` is a blocking violation, never a
    /// verdict."
    Open,
    /// spec §16.27 item 3 — a block upstream constructed from a single unassigned DBNet line
    /// then merged. Upstream side only, by construction: v1 synthesizes no DBNet line polygons,
    /// so we can never produce one. Cites its register entry; until a real §14 register entry
    /// exists, the anchor is "§16.27 item 3" itself (that item's own text).
    DbnetScattered { register_entry: &'static str },
}

impl Mechanism {
    /// Return the Rust variant name using an exhaustive match, so adding a variant requires this
    /// maintenance point to be updated.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::ClassDuplicateOf { .. } => "ClassDuplicateOf",
            Self::CoverageFilteredUpstream { .. } => "CoverageFilteredUpstream",
            Self::DocumentedSplitMerge { .. } => "DocumentedSplitMerge",
            Self::Open => "Open",
            Self::DbnetScattered { .. } => "DbnetScattered",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedPair {
    pub ours: usize,
    pub upstream: usize,
    /// Hard-coded per pair and **verified**, not taken (§16.24 item 6): a bug treating empty
    /// `lines` as `bbox = (0,0,0,0)` and unioning it would corrupt every line-ful pair while
    /// line-less pairs still passed, so the branch has to be an assertion.
    pub branch: IdentityBranch,
    pub derivation: Option<Derivation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnmatchedEntry {
    pub index: usize,
    pub mechanism: Mechanism,
}

/// §16.20 item 3(b)'s "hard-coded expected count", authored in the test source and never read
/// from a file beside the artifacts (§16.24 item 5: that would be derived data one hop from the
/// gate). Declared *alongside* the pairing list on purpose — the two are cross-checked, so a
/// copy-paste error in either is caught.
///
/// **Exactly three fields (§16.24 item 21 R5).** The seven mechanism counts of item 11's equations
/// are **derived** from the signed entry lists into [`DerivedAccounting`] and never authored, so a
/// mislabelled count is unrepresentable rather than merely detectable. Adding a mechanism-count
/// field here would reintroduce the class of defect the rename and the derivation removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Totals {
    pub pairs: usize,
    pub ours_total: usize,
    pub upstream_total: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expectations {
    pub pairs: Vec<ExpectedPair>,
    pub unmatched_ours: Vec<UnmatchedEntry>,
    pub unmatched_upstream: Vec<UnmatchedEntry>,
    pub totals: Totals,
}

// ─────────────────────────────────────────────── divergences and the report

/// spec §16.25 item 8 — **two** classes. §16.24 item 6 adopted a three-class system
/// (`Gating` / `Diagnostic` / `NoOracle`) verbatim; §16.25 supersedes that enumeration in place.
/// The class is deleted rather than kept empty: a `no_oracle()` accessor that provably always
/// returns empty is cookbook rule 1's decoration with a name claiming a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceClass {
    Gating,
    Diagnostic,
}

/// spec §16.25 item 4 — whether the oracle document carries a field, per field.
///
/// This is an **artifact property**, not a comparison finding, which is why it lives on the report
/// and not in `divergences` (item 2's discriminator: does the row's truth depend on the relation
/// between two sides, or on one artifact alone?). It is the computable witness for §16.24 item 9's
/// actual decision procedure — a per-field, per-document branch, not a per-block one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleCoverage {
    /// Every block in the subject set carries the field.
    Present,
    /// No block carries it. **Also the empty case** (§16.25 item 6(b)): `with == 0 && without == 0`
    /// is `Absent`, never `Present` — `Present` over an empty set is a vacuous truth of exactly the
    /// shape cookbook rule 1 hunts, and two live test sites reach it.
    Absent,
    /// Some do and some do not, so §16.24 item 9's two-way branch does not resolve, so the
    /// `confidence` partition row is `OPEN`, so §16.20 item 3(e) makes it blocking. The gating row
    /// is therefore not a verdict about confidence — it is the report stating that the partition
    /// cannot be completed (§16.25 item 5).
    Partial { with: usize, without: usize },
}

/// spec §16.25 items 4 and 6(a). The **paired** fields are the gating subject: the compared set is
/// the paired set, since an unmatched block contributes no `ConfidenceDelta` whether or not it
/// carries a score, and already has its own row and `Mechanism`. The whole-artifact fields are
/// DIAGNOSTIC numbers only and never gate.
///
/// `pre_filter_blocks` are excluded from both — §16.24 item 8 makes them DIAGNOSTIC and
/// pre-`group_output`.
///
/// No coverage field exists for `base_xyxy_pretruncation` (§16.25 item 6(c)): it is per-pair
/// visible in [`PairResidual`] and carries no partition row, so it needs no per-document witness.
/// The asymmetry is deliberate rather than an oversight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldCoverage {
    pub confidence: OracleCoverage,
    pub language: OracleCoverage,
    pub confidence_whole_artifact: OracleCoverage,
    pub language_whole_artifact: OracleCoverage,
}

/// Which of the four monotonicity inequalities failed (§16.24 item 18(i)). Named per edge because
/// the *edge* is the diagnostic: `X2` narrowing is the shape item 18 found, and a `Y1` violation
/// would point somewhere else entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    X1,
    Y1,
    X2,
    Y2,
}

/// No variant carries a COMPUTED float (§16.24 item 6). `ScaleMismatch` and `ConfidenceDelta`
/// hold values copied verbatim from the artifacts, so `assert_eq!` over the whole ordered vector
/// is exact — which is what makes §16.20 item 10's "and no others" a real assertion instead of a
/// `len()` check. `ConfidenceDelta` deliberately has no `delta` field for the same reason.
#[derive(Debug, Clone, PartialEq)]
pub enum Divergence {
    // ── GATING ──
    /// §16.20 item 3(b)'s identity failed on a matched pair. Equivalent to a non-zero
    /// `residual_full`; the two are the same condition and must never both gate (plan §1.6(ii)).
    GeometryIdentity {
        ours: Rect,
        expected: Rect,
        upstream: Rect,
    },
    /// §16.24 item 18(i) — leg 2's monotonicity. Adds no power over the equality; adds the
    /// attribution: a bounding union cannot narrow an edge (`Rect::merge`, `geometry.rs:60-67`),
    /// so this is a leg-1 engine-geometry divergence only when the pair's upstream derivation is
    /// `YoloUnioned`; for `YoloSplit` and `DbnetScattered` a narrowed edge is expected rather than
    /// a divergence. Never appears without `GeometryIdentity`, which is itself asserted.
    UnionMonotonicity {
        ours: Rect,
        upstream: Rect,
        edge: Edge,
    },
    IdentityBranchMismatch {
        ours: Rect,
        declared: IdentityBranch,
        actual: IdentityBranch,
    },
    /// The authored pairing is empty. §16.20 item 3: "a comparator reporting 'compared 0 boxes,
    /// 0 divergences → PASS' is cookbook rule 1's defect" — structural, so it cannot depend on a
    /// caller remembering to assert a count.
    EmptyPairing,
    PairingIndexOutOfRange {
        side: Side,
        index: usize,
        len: usize,
    },
    PairedTwice {
        side: Side,
        index: usize,
    },
    /// A block that is neither paired nor listed as unmatched. **This is the enumeration that
    /// replaces computed discovery** (cookbook rule 13): both full block lists are walked
    /// independently and every index must be accounted for exactly once.
    ///
    /// One direction of the set identity `declared_unmatched == artifact − paired`; the other is
    /// [`Divergence::PhantomUnmatched`].
    NotAccountedFor {
        side: Side,
        index: usize,
    },
    /// §16.24 item 21 R3 — the mirror of `NotAccountedFor`: an entry in `unmatched_*` that is NOT
    /// in `artifact − paired`, either because the index is paired (so it is not unmatched at all)
    /// or because it is out of range for the artifact.
    ///
    /// The computed set is taken over the **declared** pairs, so this is verification of the signed
    /// table against itself and the artifacts — not a pairing the comparator invented. Without it,
    /// a phantom entry would inflate a derived mechanism count and let the accounting equations
    /// close over a block that does not exist.
    PhantomUnmatched {
        side: Side,
        index: usize,
    },
    OpenMechanism {
        side: Side,
        index: usize,
    },
    UnverifiableMechanism {
        side: Side,
        index: usize,
        reason: &'static str,
    },
    AccountingMismatch {
        field: &'static str,
        declared: usize,
        actual: usize,
    },
    /// Cookbook rule 7: upstream IS an oracle for `scale`.
    ScaleMismatch {
        ours: f64,
        upstream: f64,
    },
    /// GATING — §16.25 item 5. The oracle document carries the field on some **paired** upstream
    /// blocks and not others, so §16.24 item 9's branch does not resolve and the partition row is
    /// `OPEN`, which §16.20 item 3(e) makes blocking. Routed through `divergences` rather than left
    /// only on the report so the CI contract stays one channel: `gating().is_empty()` AND
    /// `pairs_compared == <literal>`, with no second thing a caller must remember to check.
    ///
    /// `missing` carries the **indices**, not just counts — counts send a reader hunting, indices
    /// are a fix (cookbook rule 13's corollary).
    InconsistentOracleCoverage {
        field: &'static str,
        with: usize,
        without: usize,
        missing: Vec<usize>,
    },
    ImageSizeMismatch {
        ours: (u32, u32),
        upstream: (u32, u32),
    },
    /// spec §16.27 item 2 — GATING. The new geometry gate `ours.rect == rect_yolo`, exact and
    /// epsilon-free (no tolerance, unlike the old identity-based check). `rect_yolo: None` here
    /// means the recorded derivation requires a parent yolo box and none was recorded, which
    /// itself gates rather than silently passing.
    Leg1YoloGeometry {
        ours: Rect,
        rect_yolo: Option<[i32; 4]>,
        derivation: Derivation,
    },
    /// spec §16.27 item 2(c) — GATING structural guard. A pair whose upstream derivation is
    /// `YoloSplit` or `DbnetScattered` may not be paired at all. This replaces the leg-1 equality
    /// row for these two derivations.
    UnpairableDerivation {
        ours_index: usize,
        upstream_index: usize,
        derivation: Derivation,
    },
    /// spec §16.28 item 2 (Fable ruling, Amendment 1) — GATING. Exact bidirectional equality
    /// between the signed and recorded derivations.
    DerivationSignatureMismatch {
        ours: Rect,
        signed: Option<Derivation>,
        recorded: Option<Derivation>,
    },
    // ── DIAGNOSTIC (§16.20 item 3(d): never gated) ──
    ConfidenceDelta {
        ours: Rect,
        ours_confidence: f32,
        upstream_confidence: f32,
    },
    LanguageDelta {
        ours: Rect,
        ours_language: Option<Language>,
        upstream_language: Option<Language>,
    },
}

impl Divergence {
    pub fn class(&self) -> DivergenceClass {
        match self {
            Self::ConfidenceDelta { .. } | Self::LanguageDelta { .. } => {
                DivergenceClass::Diagnostic
            }
            _ => DivergenceClass::Gating,
        }
    }

    /// Return the Rust variant name using an exhaustive match, so adding a variant requires this
    /// maintenance point to be updated even though [`Self::class`] deliberately remains wildcarded.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::GeometryIdentity { .. } => "GeometryIdentity",
            Self::UnionMonotonicity { .. } => "UnionMonotonicity",
            Self::IdentityBranchMismatch { .. } => "IdentityBranchMismatch",
            Self::EmptyPairing => "EmptyPairing",
            Self::PairingIndexOutOfRange { .. } => "PairingIndexOutOfRange",
            Self::PairedTwice { .. } => "PairedTwice",
            Self::NotAccountedFor { .. } => "NotAccountedFor",
            Self::PhantomUnmatched { .. } => "PhantomUnmatched",
            Self::OpenMechanism { .. } => "OpenMechanism",
            Self::UnverifiableMechanism { .. } => "UnverifiableMechanism",
            Self::AccountingMismatch { .. } => "AccountingMismatch",
            Self::ScaleMismatch { .. } => "ScaleMismatch",
            Self::InconsistentOracleCoverage { .. } => "InconsistentOracleCoverage",
            Self::ImageSizeMismatch { .. } => "ImageSizeMismatch",
            Self::Leg1YoloGeometry { .. } => "Leg1YoloGeometry",
            Self::UnpairableDerivation { .. } => "UnpairableDerivation",
            Self::DerivationSignatureMismatch { .. } => "DerivationSignatureMismatch",
            Self::ConfidenceDelta { .. } => "ConfidenceDelta",
            Self::LanguageDelta { .. } => "LanguageDelta",
        }
    }

    pub fn is_gating(&self) -> bool {
        self.class() == DivergenceClass::Gating
    }
}

/// §16.24 item 11's seven terms, **derived** by partitioning the signed entry lists by `Mechanism`
/// (item 21 R5). Reported, never authored. Each term names exactly one mechanism class on one side:
/// `class_duplicates` is unsuffixed because §14.13 duplicates are upstream-only by definition, and
/// `coverage_filtered_ours` is ours-only for the mirror reason (§14.17 / §16.20 item 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DerivedAccounting {
    pub pairs: usize,
    pub class_duplicates: usize,
    pub documented_split_merge_upstream: usize,
    pub dbnet_scattered: usize,
    pub open_upstream: usize,
    pub coverage_filtered_ours: usize,
    pub documented_split_merge_ours: usize,
    pub open_ours: usize,
}

impl DerivedAccounting {
    /// `pairs + class_duplicates + documented_split_merge_upstream + dbnet_scattered + open_upstream`.
    pub fn upstream_sum(&self) -> usize {
        self.pairs
            + self.class_duplicates
            + self.documented_split_merge_upstream
            + self.dbnet_scattered
            + self.open_upstream
    }
    /// `pairs + coverage_filtered_ours + documented_split_merge_ours + open_ours`.
    pub fn ours_sum(&self) -> usize {
        self.pairs + self.coverage_filtered_ours + self.documented_split_merge_ours + self.open_ours
    }
}

/// §16.24 item 18(h)(iii) — the signed residual row, one per compared pair, **DIAGNOSTIC only**.
/// Nothing in `Expectations` can reach these and no epsilon exists anywhere near them (18(h)(i)).
#[derive(Debug, Clone, PartialEq)]
pub struct PairResidual {
    pub ours_index: usize,
    pub upstream_index: usize,
    /// The branch actually taken, so a residual can be read against the right leg.
    pub branch: IdentityBranch,
    /// `upstream.xyxy − bbox(ours.rect ∪ OracleBlock::identity_lines())`, per coordinate (§16.29
    /// item 1's operand). **All-zero for green.** Non-zero here IS `GeometryIdentity` — the same
    /// condition, reported once as a verdict and once as a number (plan §1.6(ii)).
    pub residual_full: [i32; 4],
    /// `upstream.xyxy − ours.rect`, per coordinate. Leg 1's residual. **Non-zero in the normal
    /// case** — item 12's well-behaved pair gives `[0,0,+4,0]` because the union widened `x2` — so
    /// this NEVER gates. Its sign pattern is the attribution: all-zero `residual_full` beside a
    /// non-zero `residual_leg1` is leg 2 working; a negative `x2` entry is leg 1 failing.
    pub residual_leg1: [i32; 4],
    /// §16.27 item 8 (superseding §16.24 item 18(k)'s "rare-but-real", which conflated upstream's
    /// line-less *filter* branch — real, fired once — with a line-less *output* block, which is
    /// structurally impossible): no gated upstream block carries zero lines, 0 of 14 measured on
    /// the three candidate pages. The per-block census is the evidence for that claim, not for a
    /// branch expected to fire.
    pub upstream_line_count: usize,
    /// Passed through from [`OracleBlock::base_xyxy_pretruncation`]; `None` when unrecorded.
    pub upstream_base_xyxy_pretruncation: Option<[f64; 4]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonReport {
    /// Pairs whose two indices both resolved and were actually compared. NOT
    /// `expectations.pairs.len()` — an authored pair naming a nonexistent block raises
    /// `PairingIndexOutOfRange` and does not count here, which is what makes the count a real
    /// anti-vacuity number.
    pub pairs_compared: usize,
    /// `(ours index, upstream index, branch ACTUALLY taken)`, in authored order.
    pub branches: Vec<(usize, usize, IdentityBranch)>,
    /// One per compared pair, in authored order (§16.24 item 18(h)).
    pub residuals: Vec<PairResidual>,
    /// Item 11's seven terms, derived from the signed entry lists (§16.24 item 21 R5). Reported so a
    /// human sees the arithmetic; not an independent gate, since with derived counts the equations'
    /// left side is `|pairs| + |unmatched|` by construction.
    pub derived: DerivedAccounting,
    /// spec §16.25 item 4 — the third channel. `divergences` carries *findings about the
    /// comparison*; `residuals`/`branches` carry *per-pair measurements*; this carries *artifact
    /// properties*. `residual_leg1` is the precedent: non-zero in the healthy case, and outside
    /// `divergences` for exactly the reason coverage must be.
    pub coverage: FieldCoverage,
    /// spec §16.28 item 5 — count of pairs on which the leg-1 `ours.rect == rect_yolo` row was
    /// actually EVALUATED (regardless of whether it passed). Zero for a pair whose recorded
    /// `derivation` is `None`, and zero for a pair caught by the structural guard below. No
    /// specific expected value is ratified anywhere yet — each test authors its own expected count.
    pub leg1_rows_checked: usize,
    pub divergences: Vec<Divergence>,
}

impl ComparisonReport {
    pub fn gating(&self) -> Vec<&Divergence> {
        self.divergences
            .iter()
            .filter(|row| row.is_gating())
            .collect()
    }
    pub fn diagnostics(&self) -> Vec<&Divergence> {
        self.divergences
            .iter()
            .filter(|row| row.class() == DivergenceClass::Diagnostic)
            .collect()
    }
}

/// Coverage of one optional field over a set of upstream blocks (§16.25 items 4 and 6(b)).
/// `present` reports whether the block at that index carries the field.
fn coverage_of(
    indices: &[usize],
    upstream: &UpstreamOracle,
    present: impl Fn(&OracleBlock) -> bool,
) -> (OracleCoverage, Vec<usize>) {
    let mut with = 0usize;
    let mut missing = Vec::new();
    for &index in indices {
        match upstream.blocks.get(index) {
            Some(block) if present(block) => with += 1,
            // An out-of-range index is not a coverage fact; `PairingIndexOutOfRange` /
            // `PhantomUnmatched` own it. Skipped so coverage never double-reports a structural
            // defect as a content one.
            Some(_) => missing.push(index),
            None => {}
        }
    }
    let without = missing.len();
    let coverage = match (with, without) {
        // §16.25 item 6(b): no block carries it — INCLUDING the empty set, where `with == 0 &&
        // without == 0`. That case is `Absent`, never `Present`: a vacuous truth over zero blocks
        // is exactly the shape cookbook rule 1 hunts, and two live test sites reach it.
        (0, _) => OracleCoverage::Absent,
        (_, 0) => OracleCoverage::Present,
        (with, without) => OracleCoverage::Partial { with, without },
    };
    (coverage, missing)
}

/// Emission order is specified in the plan's §1.2 and is part of the contract.
pub fn compare(
    ours: OursSide<'_>,
    upstream: &UpstreamOracle,
    expectations: &Expectations,
) -> ComparisonReport {
    let mut divergences = Vec::new();

    // Page rows establish the coordinate frame before any block-level result is emitted.
    if ours.scale != upstream.scale {
        divergences.push(Divergence::ScaleMismatch {
            ours: ours.scale,
            upstream: upstream.scale,
        });
    }
    if ours.image_size != upstream.image_size {
        divergences.push(Divergence::ImageSizeMismatch {
            ours: ours.image_size,
            upstream: upstream.image_size,
        });
    }

    // Validate the signed table's structure, accumulating into a LOCAL vec rather than emitting
    // directly: §16.25 item 6(c) makes the coverage row's position contractual — immediately after
    // the page frame rows and before any block row — and coverage needs `paired_upstream`, which
    // only this pass can produce. Assembling in contract order is what keeps the frozen vectors
    // stable.
    let mut structural = Vec::new();
    let mut paired_ours = HashSet::new();
    let mut paired_upstream = HashSet::new();
    if expectations.pairs.is_empty() {
        structural.push(Divergence::EmptyPairing);
    }

    for pair in &expectations.pairs {
        if pair.ours >= ours.blocks.len() {
            structural.push(Divergence::PairingIndexOutOfRange {
                side: Side::Ours,
                index: pair.ours,
                len: ours.blocks.len(),
            });
        } else if !paired_ours.insert(pair.ours) {
            structural.push(Divergence::PairedTwice {
                side: Side::Ours,
                index: pair.ours,
            });
        }

        if pair.upstream >= upstream.blocks.len() {
            structural.push(Divergence::PairingIndexOutOfRange {
                side: Side::Upstream,
                index: pair.upstream,
                len: upstream.blocks.len(),
            });
        } else if !paired_upstream.insert(pair.upstream) {
            structural.push(Divergence::PairedTwice {
                side: Side::Upstream,
                index: pair.upstream,
            });
        }
    }
    // §16.25 items 4, 6(a) and 6(c): coverage is an ARTIFACT property, so it is computed over the
    // paired upstream blocks (the compared set, and the gating subject) and emitted here —
    // immediately after the page frame rows and BEFORE any block row, because a caveat that the
    // rows below cover a subset must precede the rows it qualifies.
    let mut paired_sorted: Vec<usize> = paired_upstream.iter().copied().collect();
    paired_sorted.sort_unstable();
    let all_upstream: Vec<usize> = (0..upstream.blocks.len()).collect();

    let (confidence, confidence_missing) =
        coverage_of(&paired_sorted, upstream, |block| block.confidence.is_some());
    let (language, language_missing) =
        coverage_of(&paired_sorted, upstream, |block| block.language.is_some());
    let coverage = FieldCoverage {
        confidence: confidence.clone(),
        language: language.clone(),
        confidence_whole_artifact: coverage_of(&all_upstream, upstream, |block| {
            block.confidence.is_some()
        })
        .0,
        language_whole_artifact: coverage_of(&all_upstream, upstream, |block| {
            block.language.is_some()
        })
        .0,
    };

    // Only `Partial` gates, and only over the paired subject. `Absent` is a closed verdict
    // (§16.24 item 9) and `Present` is the healthy case; neither is a finding.
    for (field, state, missing) in [
        ("confidence", confidence, confidence_missing),
        ("language", language, language_missing),
    ] {
        if let OracleCoverage::Partial { with, without } = state {
            divergences.push(Divergence::InconsistentOracleCoverage {
                field,
                with,
                without,
                missing,
            });
        }
    }

    // Structural rows follow the document-level rows, per the assembly note above.
    divergences.append(&mut structural);

    let mut branches = Vec::new();
    let mut residuals = Vec::new();
    let mut pairs_compared = 0;
    let mut leg1_rows_checked = 0;

    // Compare only pairs whose two signed indices resolve. The authored order is observable.
    for pair in &expectations.pairs {
        let (Some(ours_block), Some(upstream_block)) = (
            ours.blocks.get(pair.ours),
            upstream.blocks.get(pair.upstream),
        ) else {
            continue;
        };

        pairs_compared += 1;
        let ours_rect = ours_block.rect;
        let (expected_rect, actual_branch) = upstream_block.reconstruct(ours_rect);
        branches.push((pair.ours, pair.upstream, actual_branch));

        if pair.branch != actual_branch {
            divergences.push(Divergence::IdentityBranchMismatch {
                ours: ours_rect,
                declared: pair.branch,
                actual: actual_branch,
            });
        }

        // Signature verification is unconditional; only the recorded derivation controls whether
        // the derivation-specific geometry machinery below is enabled.
        if pair.derivation != upstream_block.derivation {
            divergences.push(Divergence::DerivationSignatureMismatch {
                ours: ours_rect,
                signed: pair.derivation,
                recorded: upstream_block.derivation,
            });
        }
        let unpairable_derivation = match upstream_block.derivation {
            Some(derivation)
                if matches!(
                    derivation,
                    Derivation::YoloSplit | Derivation::DbnetScattered
                ) =>
            {
                divergences.push(Divergence::UnpairableDerivation {
                    ours_index: pair.ours,
                    upstream_index: pair.upstream,
                    derivation,
                });
                true
            }
            Some(derivation) => {
                leg1_rows_checked += 1;
                let rect_yolo = upstream_block.rect_yolo;
                if rect_yolo != Some(rect_to_xyxy(ours_rect)) {
                    divergences.push(Divergence::Leg1YoloGeometry {
                        ours: ours_rect,
                        rect_yolo,
                        derivation,
                    });
                }
                false
            }
            None => false,
        };

        let upstream_rect = upstream_block.rect();
        if !unpairable_derivation {
            // §16.27 items 2(c) and 6: an unpairable split/scattered box is already a gating
            // row, and its narrowed edges are expected rather than union divergences.
            if expected_rect != upstream_rect {
                divergences.push(Divergence::GeometryIdentity {
                    ours: ours_rect,
                    expected: expected_rect,
                    upstream: upstream_rect,
                });
            }

            // A union cannot move x1/y1 inward or x2/y2 inward. These are deliberately four
            // independent comparisons: the impossible polarity differs between the leading and
            // trailing edges.
            if upstream_rect.x1 > ours_rect.x1 {
                divergences.push(Divergence::UnionMonotonicity {
                    ours: ours_rect,
                    upstream: upstream_rect,
                    edge: Edge::X1,
                });
            }
            if upstream_rect.y1 > ours_rect.y1 {
                divergences.push(Divergence::UnionMonotonicity {
                    ours: ours_rect,
                    upstream: upstream_rect,
                    edge: Edge::Y1,
                });
            }
            if upstream_rect.x2 < ours_rect.x2 {
                divergences.push(Divergence::UnionMonotonicity {
                    ours: ours_rect,
                    upstream: upstream_rect,
                    edge: Edge::X2,
                });
            }
            if upstream_rect.y2 < ours_rect.y2 {
                divergences.push(Divergence::UnionMonotonicity {
                    ours: ours_rect,
                    upstream: upstream_rect,
                    edge: Edge::Y2,
                });
            }
        }

        residuals.push(PairResidual {
            ours_index: pair.ours,
            upstream_index: pair.upstream,
            branch: actual_branch,
            residual_full: subtract_rect(upstream_rect, expected_rect),
            residual_leg1: subtract_rect(upstream_rect, ours_rect),
            upstream_line_count: upstream_block.lines.len(),
            upstream_base_xyxy_pretruncation: upstream_block.base_xyxy_pretruncation,
        });

        if let Some(upstream_confidence) = upstream_block.confidence {
            if ours_block.confidence != upstream_confidence {
                divergences.push(Divergence::ConfidenceDelta {
                    ours: ours_rect,
                    ours_confidence: ours_block.confidence,
                    upstream_confidence,
                });
            }
        }

        let ours_language = class_to_language(ours_block.class_index);
        if let Some(upstream_language) = upstream_block.language {
            if ours_language != Some(upstream_language) {
                divergences.push(Divergence::LanguageDelta {
                    ours: ours_rect,
                    ours_language,
                    upstream_language: Some(upstream_language),
                });
            }
        }
    }

    // Every declared unmatched entry is checked independently. Mechanism rows are emitted in
    // signed-list order, before the direct set-accounting rows.
    for entry in &expectations.unmatched_ours {
        validate_mechanism(
            Side::Ours,
            entry,
            &paired_upstream,
            ours.blocks,
            upstream,
            &mut divergences,
        );
    }
    for entry in &expectations.unmatched_upstream {
        validate_mechanism(
            Side::Upstream,
            entry,
            &paired_upstream,
            ours.blocks,
            upstream,
            &mut divergences,
        );
    }

    let declared_ours: HashSet<usize> = expectations
        .unmatched_ours
        .iter()
        .map(|e| e.index)
        .collect();
    let declared_upstream: HashSet<usize> = expectations
        .unmatched_upstream
        .iter()
        .map(|e| e.index)
        .collect();

    for index in 0..ours.blocks.len() {
        if !paired_ours.contains(&index) && !declared_ours.contains(&index) {
            divergences.push(Divergence::NotAccountedFor {
                side: Side::Ours,
                index,
            });
        }
    }
    for index in 0..upstream.blocks.len() {
        if !paired_upstream.contains(&index) && !declared_upstream.contains(&index) {
            divergences.push(Divergence::NotAccountedFor {
                side: Side::Upstream,
                index,
            });
        }
    }

    // The reverse half of the set identity rejects an entry for a paired or nonexistent block.
    for entry in &expectations.unmatched_ours {
        if !paired_ours.contains(&entry.index) && entry.index < ours.blocks.len() {
            continue;
        }
        divergences.push(Divergence::PhantomUnmatched {
            side: Side::Ours,
            index: entry.index,
        });
    }
    for entry in &expectations.unmatched_upstream {
        if !paired_upstream.contains(&entry.index) && entry.index < upstream.blocks.len() {
            continue;
        }
        divergences.push(Divergence::PhantomUnmatched {
            side: Side::Upstream,
            index: entry.index,
        });
    }

    let derived = derive_accounting(
        pairs_compared,
        &expectations.unmatched_ours,
        &expectations.unmatched_upstream,
    );

    // These are the authored anti-vacuity literals: compare them with resolved/artifact counts,
    // rather than deriving them from either document.
    if expectations.totals.pairs != pairs_compared {
        divergences.push(Divergence::AccountingMismatch {
            field: "pairs",
            declared: expectations.totals.pairs,
            actual: pairs_compared,
        });
    }
    if expectations.totals.ours_total != ours.blocks.len() {
        divergences.push(Divergence::AccountingMismatch {
            field: "ours_total",
            declared: expectations.totals.ours_total,
            actual: ours.blocks.len(),
        });
    }
    if expectations.totals.upstream_total != upstream.blocks.len() {
        divergences.push(Divergence::AccountingMismatch {
            field: "upstream_total",
            declared: expectations.totals.upstream_total,
            actual: upstream.blocks.len(),
        });
    }
    if derived.upstream_sum() != expectations.totals.upstream_total {
        divergences.push(Divergence::AccountingMismatch {
            field: "upstream_sum",
            declared: expectations.totals.upstream_total,
            actual: derived.upstream_sum(),
        });
    }
    if derived.ours_sum() != expectations.totals.ours_total {
        divergences.push(Divergence::AccountingMismatch {
            field: "ours_sum",
            declared: expectations.totals.ours_total,
            actual: derived.ours_sum(),
        });
    }

    ComparisonReport {
        pairs_compared,
        branches,
        residuals,
        derived,
        coverage,
        leg1_rows_checked,
        divergences,
    }
}

fn subtract_rect(left: Rect, right: Rect) -> [i32; 4] {
    [
        left.x1 - right.x1,
        left.y1 - right.y1,
        left.x2 - right.x2,
        left.y2 - right.y2,
    ]
}

fn derive_accounting(
    pairs: usize,
    unmatched_ours: &[UnmatchedEntry],
    unmatched_upstream: &[UnmatchedEntry],
) -> DerivedAccounting {
    let mut derived = DerivedAccounting {
        pairs,
        class_duplicates: 0,
        documented_split_merge_upstream: 0,
        dbnet_scattered: 0,
        open_upstream: 0,
        coverage_filtered_ours: 0,
        documented_split_merge_ours: 0,
        open_ours: 0,
    };

    for entry in unmatched_upstream {
        match entry.mechanism {
            Mechanism::ClassDuplicateOf { .. } => derived.class_duplicates += 1,
            Mechanism::DocumentedSplitMerge { .. } => derived.documented_split_merge_upstream += 1,
            Mechanism::DbnetScattered { .. } => derived.dbnet_scattered += 1,
            Mechanism::Open => derived.open_upstream += 1,
            Mechanism::CoverageFilteredUpstream { .. } => {}
        }
    }
    for entry in unmatched_ours {
        match entry.mechanism {
            Mechanism::CoverageFilteredUpstream { .. } => derived.coverage_filtered_ours += 1,
            Mechanism::DocumentedSplitMerge { .. } => derived.documented_split_merge_ours += 1,
            Mechanism::Open => derived.open_ours += 1,
            Mechanism::ClassDuplicateOf { .. } => {}
            Mechanism::DbnetScattered { .. } => {}
        }
    }
    derived
}

fn validate_mechanism(
    side: Side,
    entry: &UnmatchedEntry,
    paired_upstream: &HashSet<usize>,
    ours_blocks: &[RawBlock],
    upstream: &UpstreamOracle,
    divergences: &mut Vec<Divergence>,
) {
    let invalid = |reason: &'static str, divergences: &mut Vec<Divergence>| {
        divergences.push(Divergence::UnverifiableMechanism {
            side,
            index: entry.index,
            reason,
        });
    };

    match (side, entry.mechanism) {
        (_, Mechanism::Open) => {
            divergences.push(Divergence::OpenMechanism {
                side,
                index: entry.index,
            });
        }
        (_, Mechanism::DocumentedSplitMerge { .. }) => {}
        (Side::Upstream, Mechanism::DbnetScattered { .. }) => {}
        (Side::Ours, Mechanism::DbnetScattered { .. }) => {
            invalid(
                "dbnet-scattered mechanism is upstream-side-only",
                divergences,
            );
        }
        (Side::Upstream, Mechanism::ClassDuplicateOf { upstream_index }) => {
            if upstream_index >= upstream.blocks.len() || !paired_upstream.contains(&upstream_index)
            {
                invalid(
                    "class duplicate citation is not a paired upstream block",
                    divergences,
                );
            }
        }
        (Side::Ours, Mechanism::ClassDuplicateOf { .. }) => {
            invalid("class duplicate is upstream-side-only", divergences);
        }
        (Side::Ours, Mechanism::CoverageFilteredUpstream { pre_filter_index }) => {
            let Some(ours_block) = ours_blocks.get(entry.index) else {
                invalid("ours unmatched index is out of range", divergences);
                return;
            };
            let Some(pre_filter_block) = upstream.pre_filter_blocks.get(pre_filter_index) else {
                invalid("coverage-filter citation is out of range", divergences);
                return;
            };
            let (expected, _) = pre_filter_block.reconstruct(ours_block.rect);
            if expected != pre_filter_block.rect() {
                invalid(
                    "coverage-filter evidence does not match the ours block",
                    divergences,
                );
            }
        }
        (Side::Upstream, Mechanism::CoverageFilteredUpstream { .. }) => {
            invalid("coverage-filter mechanism is ours-side-only", divergences);
        }
    }
}
