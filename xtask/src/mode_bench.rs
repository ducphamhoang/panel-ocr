//! `cargo xtask mode-bench` — the Simple/Annotation/LaMa non-gating comparison
//! benchmark (spec §16.43).
//!
//! The pure layer (cell/segment types, CLI argument parsing, device-disclosure rendering
//! and the report renderer) came from §16.43 item 9's "simple" task. [`measure`] and the
//! per-`(page, cell)` driver below are item 9's **first heavy** task: the three input
//! sources and the `pc_detect::run` → `pc_preprocess::run` → `pc_mask::run` sequence per
//! cell. Item 9's **second heavy** task adds the LaMa stage, under §16.44's ruling:
//!
//!   * on `--replay`, a LaMa cell **never seeks the model at all** — mode-bench's own
//!     source policy, not an unavailability (§16.44 item 2). Its row is a real mask-stage
//!     measurement carrying [`INPAINT_NOT_ATTEMPTED`];
//!   * on `--demo-bubbles`/`--pages`, a LaMa cell calls `pc_pipeline::run_inpaint`
//!     for real, through `pc_cli::inpainter::build_provider`'s provider;
//!   * a cell that could not inpaint **keeps its mask-stage measurement** rather than
//!     collapsing to a bare `**BLOCKED**` row (§16.44 item 1), and the three "did not
//!     inpaint" states are textually distinct (§16.44 item 3);
//!   * a LaMa cell that inpainted on **zero** pages contributes no mean to the
//!     eligibility-restricted segment 4.3 (§16.44 item 4).

use crate::paths;
use anyhow::{anyhow, bail, Context, Result};
use pc_cli::inpainter::build_provider;
use pc_config::{
    InpainterConfig, MaskRefineMode, MaskerConfig, PreprocessorConfig, TextDetectorConfig,
};
use pc_core::device::{resolve, Device, DevicePolicy, DeviceSupport};
use pc_core::ImageHandle;
use pc_pipeline::single::{run_inpaint, InpaintDests};
use pc_pipeline::{InpainterProvider, PipelineError};
use pc_testkit::golden::GoldenReport;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// D1's ruling (§16.43 item 3): the default cell set varies one factor at a time.
pub(crate) const DEFAULT_CELLS_SPEC: &str = "simple,annotation,simple+lama";

/// The one page `--replay` measures — the committed detector fixture, same stem
/// `mask-sweep --replay` uses, so the two tools measure the same input.
const REPLAY_STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";

/// `--demo-bubbles` compares our cleaned output against the vendored `_clean.png` with
/// the same dilation radius `calibrate-goldens` uses (`xtask/src/calibrate.rs`'s
/// `GoldenReport::compare_gray_with_shape(..., 2)`), so the two reports' shape columns
/// mean the same thing.
const REFERENCE_DILATE_RADIUS: u32 = 2;

/// §16.47 item 6: corrects §16.43 item 6's graft, false since GPU-2's G2-C landed real
/// detector CUDA registration. The detector's session constructor now takes the resolved
/// device policy directly and attempts real registration when a provider is requested.
const DETECTOR_DEVICE_MECHANISM: &str =
    "its session constructor takes the resolved device policy directly and attempts \
     real registration when a provider is requested — refusal now happens only when \
     the stage has no ratified path for the requested device (§16.47 item 4), not as \
     a substitute for registration";

/// §16.43 item 6's graft, quoted: the inpainter's row must state this.
const INPAINTER_DEVICE_MECHANISM: &str =
    "its construction route (`from_path_for_device`) re-resolves the same requested \
     device through the same resolver";

/// D5's binding conditions (§16.43 item 7), all three.
const REFERENCE_CAVEATS: &str = "\
- The reference is the vendored `demo_bubbles/*_clean.png`. Its producing PanelCleaner\n\
\u{20} version, profile and environment are **unrecorded**, so §16.37 item 3's pins are\n\
\u{20} unsatisfiable for the reference side; only the ours-side numbers here are reproducible.\n\
- The reference column is headed **agreement with reference**, never \"quality\": a LaMa\n\
\u{20} cell can be visually better while further from a non-inpainted reference.\n\
- No pass/fail assertion is made against `_clean.png`, ever — §15.2 stays fully in force.\n";

const REFERENCE_COLUMN_HEADER: &str = "Reference agreement (not quality)";

const NOT_CROSS_COMPARABLE: &str = "not cross-comparable";

/// §16.43 item 8's last bullet.
const D2_NOT_CLOSED: &str = "\
This report does **not** close §16.38 item 17(b) decision point D2. That decision needs a\n\
side-by-side run of upstream PanelCleaner and this port on a real page with a human\n\
verdict under §15.10(a)'s independence rule; mode-bench compares this port's own cells to\n\
each other and runs no upstream.\n";

// ---------------------------------------------------------------------------
// Cells (§16.43 items 2(a) and 3)
// ---------------------------------------------------------------------------

/// One `(mask mode, inpainting on/off)` combination.
///
/// §16.43 item 2(a): LaMa is not a peer mode — it is a stage that composes with whichever
/// mask mode ran, so the space is *(mask mode) × (inpainting on/off)*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum CellId {
    Simple,
    Annotation,
    SimpleLama,
    AnnotationLama,
}

impl CellId {
    /// All four accepted cells, in D1's stated order.
    pub(crate) const ALL: &'static [CellId] = &[
        CellId::Simple,
        CellId::Annotation,
        CellId::SimpleLama,
        CellId::AnnotationLama,
    ];

    /// D1's default subset: Simple, Annotation, Simple+LaMa.
    pub(crate) const DEFAULT: &'static [CellId] =
        &[CellId::Simple, CellId::Annotation, CellId::SimpleLama];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Annotation => "annotation",
            Self::SimpleLama => "simple+lama",
            Self::AnnotationLama => "annotation+lama",
        }
    }

    pub(crate) const fn mask_mode(self) -> &'static str {
        match self {
            Self::Simple | Self::SimpleLama => "Simple",
            Self::Annotation | Self::AnnotationLama => "Annotation",
        }
    }

    /// The `pc_config` enum value the cell's mask-mode component names, matched directly
    /// rather than round-tripped through [`Self::mask_mode`]'s display string.
    ///
    /// §16.43 item 2(a): a `+lama` cell's *mask mode* is the same as its non-LaMa peer's,
    /// because LaMa composes with a mask mode instead of being a peer of one.
    pub(crate) const fn refine_mode(self) -> MaskRefineMode {
        match self {
            Self::Simple | Self::SimpleLama => MaskRefineMode::Simple,
            Self::Annotation | Self::AnnotationLama => MaskRefineMode::Annotation,
        }
    }

    pub(crate) const fn inpainting(self) -> bool {
        matches!(self, Self::SimpleLama | Self::AnnotationLama)
    }

    fn from_name(value: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|cell| cell.name() == value)
    }
}

/// The parsed `--cells` value. A newtype so clap can carry the whole list through one
/// `value_parser`, which is what lets `all` expand to four cells inside argument parsing
/// (and therefore report an unknown name as a clap usage error, not a runtime one).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CellSet(pub(crate) Vec<CellId>);

/// Accepted: any comma-separated mix of the four cell names plus `all`. Repeats collapse,
/// keeping first-occurrence order. Unknown names are rejected, naming the accepted set.
pub(crate) fn parse_cells(value: &str) -> std::result::Result<CellSet, String> {
    let mut out: Vec<CellId> = Vec::new();
    let push = |cell: CellId, out: &mut Vec<CellId>| {
        if !out.contains(&cell) {
            out.push(cell);
        }
    };

    for token in value.split(',') {
        let token = token.trim();
        if token.is_empty() {
            return Err(format!(
                "empty cell name in `--cells {value}`; {}",
                accepted()
            ));
        }
        if token == "all" {
            for cell in CellId::ALL {
                push(*cell, &mut out);
            }
            continue;
        }
        match CellId::from_name(token) {
            Some(cell) => push(cell, &mut out),
            None => return Err(format!("unknown cell `{token}`; {}", accepted())),
        }
    }

    if out.is_empty() {
        return Err(format!("`--cells` selected no cells; {}", accepted()));
    }
    Ok(CellSet(out))
}

fn accepted() -> String {
    let join = |cells: &[CellId]| {
        cells
            .iter()
            .map(|cell| cell.name())
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "expected a comma-separated list of: {}, all (default: {})",
        join(CellId::ALL),
        join(CellId::DEFAULT),
    )
}

// ---------------------------------------------------------------------------
// Measurements (§16.43 item 2(a))
// ---------------------------------------------------------------------------

/// §16.44 item 3, binding: the `--replay` "not attempted — source policy" state must be
/// **textually distinct** from "eligible regions = 0, nothing to inpaint" and from a real
/// failed acquisition on `--demo-bubbles`/`--pages`. Fable's ruling, quoted: *"Don't
/// collapse all three into one bare 'no'."*
///
/// Its wording deliberately repeats `Source::Replay.describe()`'s shipped phrase *"no
/// model is loaded"* (§16.44 item 3: *"under this ruling it becomes load-bearing and the
/// per-row note should agree with it"*), so the two sentences cannot drift into
/// contradicting each other.
pub(crate) const INPAINT_NOT_ATTEMPTED: &str = "inpainting not attempted — `--replay` does \
not exercise LaMa on any cell, so no model is loaded and no acquisition is tried (§16.44 \
item 2)";

/// The second of §16.44 item 3's three states: acquisition was reachable, but
/// `pc_pipeline::run_inpaint` short-circuited on its own empty eligible set.
pub(crate) const INPAINT_NOTHING_ELIGIBLE: &str = "inpainting was reachable but nothing was \
eligible on this page — `pc_pipeline::run_inpaint` returned `Ok(None)` before any model was \
sought";

/// The third of §16.44 item 3's three states: the model was genuinely sought and the
/// attempt failed. Whole-cell, because the provider declares its failures run-fatal.
pub(crate) const INPAINT_ACQUISITION_FAILED: &str =
    "inpainting model acquisition was attempted for this cell and failed — ";

/// Not one of item 3's three states: the model was acquired and inference failed on this
/// one page. Per-page, so the cell's other pages still inpaint.
pub(crate) const INPAINT_INFERENCE_FAILED: &str =
    "the inpainting model was acquired, but inference failed on this page — ";

/// Why a row's `Inpainting ran` column says what it says (§16.44 item 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InpaintStatus {
    /// A non-LaMa cell. Inpainting is not a factor of this cell at all, so there is no
    /// "did not run" to explain and the row carries no note.
    NotRequested,
    /// `--replay` (§16.44 item 2): mode-bench's own source policy, not an unavailability.
    NotAttempted,
    /// `run_inpaint` returned `Ok(None)`.
    NothingEligible,
    /// The provider declared its failure run-fatal (`PipelineError::RunFatal`), which is
    /// how `run_inpaint` reports a failed *acquisition* — cookbook rule 4: the fatality is
    /// **declared** by `InpainterProvider::failures_are_run_fatal`, never inferred here.
    AcquisitionFailed(String),
    /// `PipelineError::Stage` out of `run_inpaint` after the provider handed over an
    /// inpainter — a per-page inference failure.
    InferenceFailed(String),
    /// `run_inpaint` returned `Ok(Some(..))`: inpainting genuinely executed on this page.
    Ran { tiles_inferred: usize },
}

impl InpaintStatus {
    /// Did inpainting genuinely execute? The **only** predicate §16.44 item 4's
    /// eligibility-restricted exclusion may key on.
    pub(crate) const fn ran(&self) -> bool {
        matches!(self, Self::Ran { .. })
    }

    pub(crate) const fn tiles_inferred(&self) -> Option<usize> {
        match self {
            Self::Ran { tiles_inferred } => Some(*tiles_inferred),
            _ => None,
        }
    }

    /// The row's note, or `None` when there is nothing to explain (a non-LaMa cell, or a
    /// cell that did inpaint).
    pub(crate) fn note(&self) -> Option<String> {
        match self {
            Self::NotRequested | Self::Ran { .. } => None,
            Self::NotAttempted => Some(INPAINT_NOT_ATTEMPTED.to_owned()),
            Self::NothingEligible => Some(INPAINT_NOTHING_ELIGIBLE.to_owned()),
            Self::AcquisitionFailed(reason) => {
                Some(format!("{INPAINT_ACQUISITION_FAILED}{reason}"))
            }
            Self::InferenceFailed(reason) => Some(format!("{INPAINT_INFERENCE_FAILED}{reason}")),
        }
    }
}

/// Everything one cell measured on one page.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Measured {
    pub(crate) detected_boxes: usize,
    pub(crate) masking_regions: usize,
    pub(crate) succeeded_regions: usize,
    pub(crate) failed_regions: usize,
    pub(crate) dropped_regions: usize,
    /// `pc_inpaint::select_regions`' output size — computed for **every** cell, LaMa or
    /// not (§16.43 item 8), since that function reads only `MaskRegionStats`.
    pub(crate) eligible_regions: usize,
    /// What the inpainting stage did, and — when it did nothing — which of §16.44 item 3's
    /// three distinct states applies.
    pub(crate) inpaint: InpaintStatus,
    /// `None` for sources with no reference (`--replay`, `--pages`).
    pub(crate) reference: Option<GoldenReport>,
}

/// What happened for one `(page, cell)` pair.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Outcome {
    /// Boxed so the enum is not dominated by the measurement variant.
    Measured(Box<Measured>),
    /// A whole cell was unavailable — e.g. the LaMa model is missing (§16.43 item 8).
    Blocked { reason: String },
    /// One page failed inside an otherwise-running cell.
    Failed { reason: String },
}

impl Outcome {
    fn measured(&self) -> Option<&Measured> {
        match self {
            Self::Measured(measured) => Some(measured),
            _ => None,
        }
    }
}

/// Which input source the run measured (§16.43 item 8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Source {
    Replay,
    DemoBubbles,
    Pages,
}

impl Source {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Replay => "--replay (committed detector fixture)",
            Self::DemoBubbles => "--demo-bubbles (7 vendored crops)",
            Self::Pages => "--pages (maintainer-local full pages)",
        }
    }

    pub(crate) const fn has_reference(self) -> bool {
        matches!(self, Self::DemoBubbles)
    }

    /// §16.44 item 2, the disputed question Fable resolved: **`--replay` never attempts
    /// LaMa acquisition for any cell**, by mode-bench's own deliberate source policy —
    /// *"not because the model is unavailable, but because that source does not exercise
    /// LaMa at all"*. Fable's decisive ground, quoted: *"Under the architect's ruling,
    /// item 8's antecedent ('whose model is unavailable') simply never fires on
    /// `--replay`, because unavailability is only meaningful relative to a need; both the
    /// ratified sentence and the frozen test hold as written, on every source."*
    ///
    /// This is the predicate that makes §16.43 item 8's ratified clause *"`--pages` … the
    /// only source that reaches multi-tile LaMa"* true **by construction** rather than
    /// contingent on the replay fixture's eligible-region geometry (item 2's second
    /// ground).
    pub(crate) const fn attempts_inpainting(self) -> bool {
        match self {
            Self::Replay => false,
            Self::DemoBubbles | Self::Pages => true,
        }
    }

    pub(crate) const fn describe(self) -> &'static str {
        match self {
            Self::Replay => "The committed detector fixture; no model is loaded and this source has no reference image.",
            Self::DemoBubbles => "The 7 vendored crops — the only source carrying a reference. Every crop is under 512px on both axes, so each LaMa cell here is a single edge-replicated tile (§16.38 items 5(c), 5(d)) and this source cannot evidence `DEVIATION(24)`'s tiling behavior.",
            Self::Pages => "Maintainer-local full pages; this source has no reference image, and it is the only source that reaches multi-tile LaMa.",
        }
    }
}

/// The scalars a segment mean may be taken over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Metric {
    FailedRegions,
    EligibleRegions,
    ReferenceMeanAbsDiff,
}

impl Metric {
    const ALL: &'static [Metric] = &[
        Metric::FailedRegions,
        Metric::EligibleRegions,
        Metric::ReferenceMeanAbsDiff,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::FailedRegions => "Mean failed regions",
            Self::EligibleRegions => "Mean eligible regions",
            Self::ReferenceMeanAbsDiff => "Mean reference mean-abs-diff (lower = closer agreement)",
        }
    }

    fn value(self, measured: &Measured) -> Option<f64> {
        match self {
            Self::FailedRegions => Some(measured.failed_regions as f64),
            Self::EligibleRegions => Some(measured.eligible_regions as f64),
            Self::ReferenceMeanAbsDiff => measured
                .reference
                .as_ref()
                .map(|report| report.mean_abs_diff),
        }
    }
}

/// Every measurement in one invocation, plus which cells that invocation ran.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Benchmark {
    source: Source,
    cells: Vec<CellId>,
    pages: Vec<String>,
    rows: BTreeMap<(String, CellId), Outcome>,
    blocked_cells: BTreeMap<CellId, String>,
}

impl Benchmark {
    pub(crate) fn new(source: Source, cells: &[CellId]) -> Self {
        Self {
            source,
            cells: cells.to_vec(),
            pages: Vec::new(),
            rows: BTreeMap::new(),
            blocked_cells: BTreeMap::new(),
        }
    }

    pub(crate) fn cells(&self) -> &[CellId] {
        &self.cells
    }

    pub(crate) fn pages(&self) -> &[String] {
        &self.pages
    }

    /// Record one `(page, cell)` outcome, registering the page on first sight.
    pub(crate) fn record(&mut self, page: &str, cell: CellId, outcome: Outcome) {
        if !self.pages.iter().any(|known| known == page) {
            self.pages.push(page.to_owned());
        }
        self.rows.insert((page.to_owned(), cell), outcome);
    }

    /// Register a page even when no cell measured it, so a blocked cell still gets rows.
    pub(crate) fn register_page(&mut self, page: &str) {
        if !self.pages.iter().any(|known| known == page) {
            self.pages.push(page.to_owned());
        }
    }

    /// Mark a whole cell unavailable. Every other cell in the run still reports
    /// (§16.43 item 8).
    //
    // **§16.44 item 1 removed this function's production caller, deliberately.** Both joint
    // rulings converged, quoted: a LaMa cell that cannot inpaint "must **keep its own real
    // mask-stage measurement** (detected boxes, masking regions, succeeded/failed/dropped/
    // eligible counts) — never erase it to render a bare `**BLOCKED**` row with no other
    // data." So the LaMa driver records an acquisition failure as a NOTE on a `Measured`
    // row (`InpaintStatus::AcquisitionFailed`), not as a blocked cell. The mechanism stays
    // for a future whole-cell condition that genuinely precedes measurement, and this
    // module's tests still exercise the rendering path.
    #[allow(dead_code)]
    pub(crate) fn block_cell(&mut self, cell: CellId, reason: impl Into<String>) {
        self.blocked_cells.insert(cell, reason.into());
    }

    /// A recorded row **wins over** a blocked cell (§16.44 item 1): once a page has a real
    /// measurement, no cell-level condition may erase it. A blocked cell with no recorded
    /// row still renders `**BLOCKED**`, which is what keeps §16.43 item 8's rule intact for
    /// a condition that really did precede measurement.
    pub(crate) fn outcome(&self, page: &str, cell: CellId) -> Option<Outcome> {
        if let Some(recorded) = self.rows.get(&(page.to_owned(), cell)) {
            return Some(recorded.clone());
        }
        self.blocked_cells
            .get(&cell)
            .map(|reason| Outcome::Blocked {
                reason: reason.clone(),
            })
    }

    /// Segment 2: this cell's own `{page : eligible_regions > 0}`.
    pub(crate) fn eligible_pages(&self, cell: CellId) -> BTreeSet<String> {
        self.pages
            .iter()
            .filter(|page| {
                self.outcome(page, cell)
                    .as_ref()
                    .and_then(Outcome::measured)
                    .is_some_and(|measured| measured.eligible_regions > 0)
            })
            .cloned()
            .collect()
    }

    /// Segment 1: every page, regardless of eligibility.
    pub(crate) fn full_population(&self) -> BTreeSet<String> {
        self.pages.iter().cloned().collect()
    }

    /// Did this cell produce at least one real measurement anywhere?
    ///
    /// §16.44 item 1, binding and converged on by both joint rulings: the "actually run"
    /// predicate *"must stop keying on absence from the blocked-cells map, and key it on
    /// **having produced at least one `Measured` row** instead"*.
    fn has_measured_row(&self, cell: CellId) -> bool {
        self.pages.iter().any(|page| {
            self.outcome(page, cell)
                .as_ref()
                .and_then(Outcome::measured)
                .is_some()
        })
    }

    /// Did inpainting genuinely execute for this cell on **any** page?
    ///
    /// §16.44 item 4's predicate, quoted: a LaMa cell must have *"genuinely run inpainting
    /// on at least one page (i.e. `run_inpaint` returned `Ok(Some(...))` somewhere in that
    /// cell's pages, not merely `Ok(None)` or a `Measured` row with `inpainting_ran: false`
    /// everywhere)"*.
    fn inpainting_ran_on_any_page(&self, cell: CellId) -> bool {
        self.pages.iter().any(|page| {
            self.outcome(page, cell)
                .as_ref()
                .and_then(Outcome::measured)
                .is_some_and(|measured| measured.inpaint.ran())
        })
    }

    /// Segment 3: the intersection of the eligible sets over the cells actually run.
    ///
    /// **§16.44 item 1** replaced this predicate's original "not in `blocked_cells`" form
    /// with "produced at least one `Measured` row". The two existing unit tests below
    /// (`a_blocked_cell_does_not_poison_the_intersection_even_though_it_was_requested`,
    /// `an_intersection_over_only_blocked_cells_is_empty_not_a_false_full_match`) pass
    /// unchanged under it, which is what both rulings independently verified before
    /// ratifying it. The original ground is unchanged and still applies: a cell that never
    /// produced a measurement contributes no genuine eligible set to intersect against, so
    /// treating its absence as an empty set would silently collapse this segment to empty
    /// on the most common real-world run shape.
    ///
    /// **This is the eligible-page-set intersection, not the mean.** §16.44 item 4's
    /// exclusion of a never-inpainted LaMa cell is applied to the *mean* — see
    /// [`Self::contributes_to_eligibility_restricted_mean`] — because Fable's ruling scoped
    /// item 4 to *"the eligibility-restricted cross-cell mean"* and closed with *"I decide
    /// nothing further about section 4.3's captions beyond D2's existing requirements."*
    pub(crate) fn common_eligible(&self) -> BTreeSet<String> {
        let mut contributing = self
            .cells
            .iter()
            .copied()
            .filter(|cell| self.has_measured_row(*cell));
        let Some(first) = contributing.next() else {
            return BTreeSet::new();
        };
        let mut common = self.eligible_pages(first);
        for cell in contributing {
            let next = self.eligible_pages(cell);
            common.retain(|page| next.contains(page));
        }
        common
    }

    /// The cells this intersection was actually computed over, for the segment-3 caption
    /// to name (§16.43 item 4's "the caption must name the cell set the intersection was
    /// computed over").
    pub(crate) fn common_eligible_contributing_cells(&self) -> Vec<CellId> {
        self.cells
            .iter()
            .copied()
            .filter(|cell| self.has_measured_row(*cell))
            .collect()
    }

    /// §16.44 item 4, the hazard Fable found by combining two of the converged-on points
    /// and the binding condition that closes it, quoted: *"a cell whose inpainting factor
    /// was requested but ran on zero pages must not contribute to the eligibility-
    /// restricted cross-cell mean (disclosure alone is not enough;
    /// disclosure-instead-of-exclusion is what D2 already rejected)."*
    ///
    /// Without this, a LaMa cell that kept its mask-stage measurement (item 1) but never
    /// inpainted would re-enter segment 4.3's mean, putting *"'inpainted' output [that] is
    /// actually the bare masking output … inside a segment whose caption claims
    /// comparability"* — the different-populations bug, hidden.
    ///
    /// A **non-LaMa** cell's predicate is explicitly unaffected: *"it is still 'produced at
    /// least one `Measured` row,' since it has no inpainting factor to have run or not."*
    pub(crate) fn contributes_to_eligibility_restricted_mean(&self, cell: CellId) -> bool {
        if cell.inpainting() {
            self.inpainting_ran_on_any_page(cell)
        } else {
            self.has_measured_row(cell)
        }
    }

    /// Mean of `metric` for `cell` over `pages`. `None` when no row contributes a value —
    /// never a mean over zero rows.
    pub(crate) fn mean(
        &self,
        cell: CellId,
        pages: &BTreeSet<String>,
        metric: Metric,
    ) -> Option<f64> {
        let values: Vec<f64> = pages
            .iter()
            .filter_map(|page| self.outcome(page, cell))
            .filter_map(|outcome| outcome.measured().and_then(|m| metric.value(m)))
            .collect();
        if values.is_empty() {
            return None;
        }
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }

    fn cell_list(&self) -> String {
        self.cells
            .iter()
            .map(|cell| cell.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

// ---------------------------------------------------------------------------
// Device disclosure (§16.43 item 6)
// ---------------------------------------------------------------------------

/// Only the genuinely per-stage facts. The device statement is **not** among them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StageDisclosure {
    pub(crate) stage: &'static str,
    pub(crate) model_path: String,
    pub(crate) expected_sha256: &'static str,
    pub(crate) digest_verified: bool,
    pub(crate) session_constructed: bool,
    pub(crate) constructing_function: &'static str,
    pub(crate) device_mechanism: &'static str,
}

pub(crate) fn detector_disclosure(
    model_path: Option<&str>,
    digest_verified: bool,
    session_constructed: bool,
) -> StageDisclosure {
    StageDisclosure {
        stage: "detector",
        model_path: model_path
            .unwrap_or("(not resolved in this run)")
            .to_owned(),
        expected_sha256: pc_models::COMIC_TEXT_DETECTOR.sha256,
        digest_verified,
        session_constructed,
        constructing_function: "`pc_detect::onnx::OnnxDetector::from_path_with_config`",
        device_mechanism: DETECTOR_DEVICE_MECHANISM,
    }
}

pub(crate) fn inpainter_disclosure(
    model_path: Option<&str>,
    digest_verified: bool,
    session_constructed: bool,
) -> StageDisclosure {
    StageDisclosure {
        stage: "inpainter",
        model_path: model_path
            .unwrap_or("(not resolved in this run)")
            .to_owned(),
        expected_sha256: pc_models::LAMA_MANGA_INPAINTER.sha256,
        digest_verified,
        session_constructed,
        constructing_function: "`pc_inpaint::onnx::OnnxInpainter::from_path_for_device`",
        device_mechanism: INPAINTER_DEVICE_MECHANISM,
    }
}

/// The stage rows this invocation needs: always the detector, plus the inpainter when any
/// requested cell inpaints.
///
/// **Placeholder facts** (`None, false, false`) — used by the pure-layer tests, which never
/// run a real detector or a real inpainter. [`run`] uses [`stage_disclosures_with_facts`]
/// instead, which carries what [`measure`] actually observed. Test-only: since the fix for
/// F2 (detector) and this task (inpainter), no non-test code needs the placeholder.
#[cfg(test)]
fn stage_disclosures(cells: &[CellId]) -> Vec<StageDisclosure> {
    stage_disclosures_with_facts(cells, &RunFacts::default())
}

/// What the detector stage actually did in this invocation, so the report's per-stage
/// disclosure row states real facts rather than a placeholder (post-review fix, §16.43
/// item 6: a row claiming `no`/`no`/`(not resolved)` when a real ONNX session was in fact
/// resolved, digest-verified and constructed is a false disclosure, not a nit).
#[derive(Debug, Clone, Default)]
struct DetectorFacts {
    model_path: Option<String>,
    digest_verified: bool,
    session_constructed: bool,
}

/// The same real-facts treatment extended to the inpainter row (brief point 5).
///
/// `digest_verified` and `session_constructed` are set **only** from an acquisition that
/// actually succeeded: `OnnxInpainterProvider::initialize` verifies the artifact's sha256
/// (§16.38 item 19(b)) and then builds the session, in that order, so a successful
/// `run_inpaint` is evidence of both and nothing else is. A run that never sought the model
/// — every `--replay` run (§16.44 item 2), and any run where no cell had an eligible region
/// — reports `attempted: false` and leaves the other three at their honest defaults.
#[derive(Debug, Clone, Default)]
struct InpainterFacts {
    /// The path the provider was pointed at, stated only once a session was really built
    /// from it.
    model_path: Option<String>,
    digest_verified: bool,
    session_constructed: bool,
    /// Was acquisition sought at all in this invocation?
    attempted: bool,
}

/// What both model-backed stages actually did, carried from [`measure`] to the renderer.
#[derive(Debug, Clone, Default)]
struct RunFacts {
    detector: DetectorFacts,
    inpainter: InpainterFacts,
}

/// The inpainter row's model-path cell: the honest three-way distinction §16.44 item 3
/// asks the *report* to preserve, carried into the disclosure table too.
fn inpainter_model_path_cell(facts: &InpainterFacts) -> String {
    match (&facts.model_path, facts.attempted) {
        (Some(path), _) => path.clone(),
        (None, true) => "(acquisition attempted; no session was built from it)".to_owned(),
        (None, false) => "(not attempted in this run)".to_owned(),
    }
}

fn stage_disclosures_with_facts(cells: &[CellId], facts: &RunFacts) -> Vec<StageDisclosure> {
    let mut stages = vec![detector_disclosure(
        facts.detector.model_path.as_deref(),
        facts.detector.digest_verified,
        facts.detector.session_constructed,
    )];
    if cells.iter().any(|cell| cell.inpainting()) {
        let mut row = inpainter_disclosure(
            None,
            facts.inpainter.digest_verified,
            facts.inpainter.session_constructed,
        );
        row.model_path = inpainter_model_path_cell(&facts.inpainter);
        stages.push(row);
    }
    stages
}

pub(crate) fn render_device_section(policy: &DevicePolicy, stages: &[StageDisclosure]) -> String {
    let mut body = String::from("## 2. Device\n\n");
    // §16.43 item 6, binding: state `DevicePolicy::report()` verbatim, once.
    let _ = writeln!(body, "{}\n", policy.report());
    body.push_str(
        "That is the **one** device statement for this whole invocation: one `--device` flag, \
         resolved once through `pc_core::device::resolve`, one resolved policy. The per-stage \
         rows below carry only per-stage facts and cross-reference the statement above; they do \
         not restate it, and there is no per-session device heading.\n\n",
    );
    body.push_str(
        "| Stage | Model path | Expected sha256 | Digest verified? | Session constructed? | Constructing function |\n",
    );
    body.push_str("|---|---|---|---|---|---|\n");
    for stage in stages {
        let _ = writeln!(
            body,
            "| {} | {} | `{}` | {} | {} | {} |",
            stage.stage,
            escape_cell(&stage.model_path),
            stage.expected_sha256,
            yes_no(stage.digest_verified),
            yes_no(stage.session_constructed),
            stage.constructing_function,
        );
    }
    body.push_str("\nMechanism disclosure per stage — not a second policy statement:\n\n");
    for stage in stages {
        let _ = writeln!(body, "- **{}:** {}", stage.stage, stage.device_mechanism);
    }
    body.push('\n');
    body
}

// ---------------------------------------------------------------------------
// Report renderer (§16.43 items 1, 4, 7, 8)
// ---------------------------------------------------------------------------

pub(crate) fn render_document(
    benchmark: &Benchmark,
    policy: &DevicePolicy,
    stages: &[StageDisclosure],
) -> String {
    let mut body = String::new();
    body.push_str(
        "# Mode comparison benchmark (spec §16.43)\n\n\
         **Generated in full by `cargo xtask mode-bench` — do not hand-edit.** Every value here\n\
         is measured at generation time. This is a **non-gating** report: it gates no CI job and\n\
         blocks no merge, the same status as `cargo xtask calibrate-goldens` (§7.3) and\n\
         `cargo xtask mask-sweep` (§16.35).\n\n",
    );

    body.push_str("## 1. Method and provenance\n\n");
    let _ = writeln!(body, "- **Source:** {}", benchmark.source.label());
    let _ = writeln!(body, "- **Source detail:** {}", benchmark.source.describe());
    let _ = writeln!(body, "- **Cells run:** {}", benchmark.cell_list());
    body.push_str(
        "- **Stage sequence:** `pc_detect::run` → `pc_preprocess::run` → `pc_mask::run` → \
         `pc_pipeline::run_inpaint`, invoked directly (not `pc-cli`'s `run_clean`).\n",
    );
    body.push_str(
        "- **Metric:** `pc_testkit::golden::GoldenReport` (`compare_gray_with_shape` / \
         `compare_gray`) — no new metric type.\n",
    );
    body.push_str(
        "- **Eligibility:** `pc_inpaint::select_regions`, computed and reported for every cell \
         (it reads only `MaskRegionStats`), LaMa or not.\n\n",
    );

    body.push_str(&render_device_section(policy, stages));
    body.push_str(&render_measurements(benchmark));
    body.push_str(&render_segments(benchmark));
    body.push_str(&render_reference_section(benchmark));

    body.push_str("## 6. Verdict\n\n");
    body.push_str(
        "**Non-gating.** This report ranks nothing and asserts nothing. It records what each \
         cell measured on the same inputs so a maintainer can read the differences directly. \
         Cells are not ordered by their distance to any reference image, and no cell is declared \
         better or worse than another here.\n\n",
    );
    body.push_str(D2_NOT_CLOSED);
    body
}

fn render_measurements(benchmark: &Benchmark) -> String {
    let mut body = String::from("## 3. Per-page, per-cell measurements\n\n");
    body.push_str(
        "| Page | Cell | Mask mode | Detected boxes | Masking regions | Succeeded | Failed | Dropped | Eligible | Inpainting ran | Tiles inferred | ",
    );
    let _ = writeln!(body, "{REFERENCE_COLUMN_HEADER} | Outcome |");
    body.push_str("|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|\n");

    if benchmark.pages().is_empty() {
        body.push_str("| _(no pages measured)_ |  |  |  |  |  |  |  |  |  |  |  |  |\n");
    }
    for page in benchmark.pages() {
        for cell in benchmark.cells() {
            let _ = writeln!(body, "{}", measurement_row(benchmark, page, *cell));
        }
    }
    body.push('\n');
    body
}

fn measurement_row(benchmark: &Benchmark, page: &str, cell: CellId) -> String {
    let head = format!(
        "| {} | {} | {} |",
        escape_cell(page),
        cell.name(),
        cell.mask_mode()
    );
    match benchmark.outcome(page, cell) {
        Some(Outcome::Measured(measured)) => format!(
            "{head} {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            measured.detected_boxes,
            measured.masking_regions,
            measured.succeeded_regions,
            measured.failed_regions,
            measured.dropped_regions,
            measured.eligible_regions,
            yes_no(measured.inpaint.ran()),
            measured
                .inpaint
                .tiles_inferred()
                .map_or_else(|| "—".to_owned(), |tiles| tiles.to_string()),
            measured
                .reference
                .as_ref()
                .map_or_else(|| "—".to_owned(), render_agreement),
            // §16.44 item 1: a cell that could not inpaint keeps this row's real
            // mask-stage numbers; the reason rides in the outcome column as a note
            // rather than replacing them. §16.44 item 3: the three "did not inpaint"
            // states each get their own sentence here.
            measured.inpaint.note().map_or_else(
                || "measured".to_owned(),
                |note| format!("measured; {}", escape_cell(&note))
            ),
        ),
        Some(Outcome::Blocked { reason }) => format!(
            "{head} — | — | — | — | — | — | — | — | — | **BLOCKED** — {} |",
            escape_cell(&reason)
        ),
        Some(Outcome::Failed { reason }) => format!(
            "{head} — | — | — | — | — | — | — | — | — | **FAILED** — {} |",
            escape_cell(&reason)
        ),
        None => format!("{head} — | — | — | — | — | — | — | — | — | **NOT RECORDED** |"),
    }
}

fn render_agreement(report: &GoldenReport) -> String {
    format!(
        "exact {:.6} / mad {:.6} / ssim {:.6}",
        report.exact_fraction, report.mean_abs_diff, report.ssim
    )
}

fn render_segments(benchmark: &Benchmark) -> String {
    let mut body = String::from("## 4. Eligibility segments\n\n");

    // Segment 1 — no eligibility conditioning applies, so a cross-cell mean IS permitted.
    body.push_str("### 4.1 Full population — every page, every cell\n\n");
    body.push_str(
        "No eligibility conditioning applies to this segment, so a **cross-cell mean is \
         permitted** here: every cell reports on every page, whether or not that page was \
         eligible for that cell.\n\n",
    );
    body.push_str(&segment_table(
        benchmark,
        |_| benchmark.full_population(),
        &BTreeSet::new(),
    ));

    // Segment 2 — each cell's own eligible set. Explicitly NOT comparable across cells.
    body.push_str("### 4.2 Per-cell eligible subsets\n\n");
    let _ = writeln!(
        body,
        "Each row is that cell's own `{{page : eligible_regions > 0}}`. These subsets are \
         **{NOT_CROSS_COMPARABLE}**: the eligibility predicate is mode-dependent, so two cells' \
         rows here describe different populations and averaging across them would compare \
         inpainted output against bare masking output under a caption claiming comparability.\n"
    );
    body.push_str(&segment_table(
        benchmark,
        |cell| benchmark.eligible_pages(cell),
        &BTreeSet::new(),
    ));

    // Segment 3 — the only eligibility-restricted segment a cross-cell mean may use.
    // The heading names the cells the intersection actually ran over (requested minus
    // BLOCKED), not the full requested set — a BLOCKED cell contributed no real
    // measurement to intersect against, so naming it here would misstate which rows
    // this table's mean actually reflects.
    let common = benchmark.common_eligible();
    let contributing = benchmark.common_eligible_contributing_cells();
    let contributing_list = contributing
        .iter()
        .map(|cell| cell.name())
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        body,
        "### 4.3 Common-eligible intersection over ({contributing_list})\n"
    );
    let blocked_note = if contributing.len() < benchmark.cells().len() {
        " (excluding blocked cells, which contributed no measurement to intersect against)"
    } else {
        ""
    };
    let _ = writeln!(
        body,
        "The intersection of the eligible sets over the cells actually run in this invocation \
         ({contributing_list}){blocked_note}. This is the only *eligibility-restricted* segment \
         a cross-cell mean may be computed over. A later run with a different cell set \
         re-segments this table visibly, because the cell set is named in the heading above.\n"
    );
    // §16.44 item 4: a LaMa cell that inpainted on zero pages is excluded from THIS
    // segment's mean — the only eligibility-restricted one — because including it would
    // compare bare masking output against genuinely inpainted output under a caption
    // claiming comparability.
    let excluded: BTreeSet<CellId> = benchmark
        .cells()
        .iter()
        .copied()
        .filter(|cell| !benchmark.contributes_to_eligibility_restricted_mean(*cell))
        .collect();
    if common.is_empty() {
        body.push_str(
            "The common-eligible intersection is **empty**; no mean is printed for this \
             segment, rather than a mean over zero rows.\n\n",
        );
    } else {
        let _ = writeln!(
            body,
            "Pages in the intersection: {}\n",
            common.iter().cloned().collect::<Vec<_>>().join(", ")
        );
        body.push_str(&segment_table(benchmark, |_| common.clone(), &excluded));
        if !excluded.is_empty() {
            let names = excluded
                .iter()
                .map(|cell| cell.name())
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                body,
                "{EXCLUDED_FROM_MEAN_HEAD} {names}. {EXCLUDED_FROM_MEAN_WHY}\n"
            );
        }
    }
    body
}

/// §16.44 item 4's exclusion, stated in the report so the omission is legible rather than
/// silent. The exclusion itself is the requirement — this sentence accompanies it, it does
/// not replace it (*"disclosure alone is not enough"*).
const EXCLUDED_FROM_MEAN_HEAD: &str = "**No mean is computed in this segment for:**";
const EXCLUDED_FROM_MEAN_WHY: &str = "Inpainting ran on zero pages for that cell, so its \
rows here are bare masking output. Averaging them alongside a genuinely inpainted cell's \
rows, inside a segment whose caption claims comparability, would compare different \
populations under one heading (§16.44 item 4).";

/// What an excluded cell's metric columns render instead of a number.
const EXCLUDED_CELL: &str = "excluded";

fn segment_table(
    benchmark: &Benchmark,
    pages_for: impl Fn(CellId) -> BTreeSet<String>,
    excluded: &BTreeSet<CellId>,
) -> String {
    let mut body = String::from("| Cell | Pages in segment |");
    for metric in Metric::ALL {
        let _ = write!(body, " {} |", metric.label());
    }
    body.push_str("\n|---|---:|---:|---:|---:|\n");
    for cell in benchmark.cells() {
        let pages = pages_for(*cell);
        let _ = write!(body, "| {} | {} |", cell.name(), pages.len());
        for metric in Metric::ALL {
            let rendered = if excluded.contains(cell) {
                EXCLUDED_CELL.to_owned()
            } else {
                benchmark
                    .mean(*cell, &pages, *metric)
                    .map_or_else(|| "—".to_owned(), |value| format!("{value:.6}"))
            };
            let _ = write!(body, " {rendered} |");
        }
        body.push('\n');
    }
    body.push('\n');
    body
}

fn render_reference_section(benchmark: &Benchmark) -> String {
    let mut body = String::from("## 5. Reference comparison\n\n");
    if benchmark.source.has_reference() {
        let _ = writeln!(
            body,
            "The `{REFERENCE_COLUMN_HEADER}` column in section 3 reports agreement with the \
             vendored reference, under all three of D5's binding conditions (§16.43 item 7):\n"
        );
        body.push_str(REFERENCE_CAVEATS);
        body.push('\n');
    } else {
        let _ = writeln!(
            body,
            "This source has **no reference** image, so the `{REFERENCE_COLUMN_HEADER}` column \
             in section 3 is empty for every row and no comparison against `_clean.png` is made \
             in this run.\n"
        );
    }
    body
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn escape_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

// ---------------------------------------------------------------------------
// CLI (§16.43 items 2(a), 6, 8)
// ---------------------------------------------------------------------------

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    /// Replay the committed detector fixture (CI-runnable; no model, no reference).
    #[arg(
        long,
        conflicts_with_all = ["demo_bubbles", "pages"],
        required_unless_present_any = ["demo_bubbles", "pages"]
    )]
    replay: bool,
    /// Measure the 7 vendored `demo_bubbles` crops — the only source with a reference.
    ///
    /// Requires `--detector`: `ReplayDetector` carries exactly one recorded page, so a
    /// faithful per-crop detection needs the real ONNX detector, not a replay of some
    /// other image's boxes.
    #[arg(long, requires = "detector", conflicts_with_all = ["replay", "pages"])]
    demo_bubbles: bool,
    /// Directory of maintainer-local full pages (no reference; the only multi-tile source).
    #[arg(
        long,
        value_name = "DIR",
        requires = "detector",
        conflicts_with_all = ["replay", "demo_bubbles"]
    )]
    pages: Option<PathBuf>,
    /// Real detector specification in the form `onnx:<path>` (with --pages or --demo-bubbles).
    #[arg(long, value_name = "SPEC", conflicts_with = "replay")]
    detector: Option<String>,
    /// Cells to run: any comma-separated mix of `simple`, `annotation`, `simple+lama`,
    /// `annotation+lama`, or `all`.
    #[arg(long, default_value = DEFAULT_CELLS_SPEC, value_parser = parse_cells)]
    cells: CellSet,
    /// Execution device for the whole invocation. `cuda` is refused on this build.
    #[arg(long, default_value = "cpu", value_parser = crate::parse_device)]
    device: Device,
    /// Report destination (default: `docs/MODE_COMPARISON.md`).
    #[arg(long)]
    out: Option<PathBuf>,
}

pub(crate) fn run(args: Args) -> Result<()> {
    // §16.43 item 2(b) / item 8: a device this build cannot provide is a hard refusal
    // carrying `pc_core`'s own message verbatim — never a silent downgrade to CPU. It is
    // resolved before anything is measured or written, so a refused run produces no report.
    let policy = resolve(args.device, DeviceSupport::compiled())
        .map_err(|refusal| anyhow::anyhow!(refusal.message()))?;

    let source = if args.replay {
        Source::Replay
    } else if args.demo_bubbles {
        Source::DemoBubbles
    } else {
        Source::Pages
    };

    let target = args
        .out
        .unwrap_or_else(|| paths::docs_root().join("MODE_COMPARISON.md"));

    let (benchmark, facts) = measure(
        source,
        &args.cells.0,
        args.pages.as_deref(),
        args.detector.as_deref(),
        args.device,
    )?;
    let stages = stage_disclosures_with_facts(benchmark.cells(), &facts);
    let document = render_document(&benchmark, &policy, &stages);

    std::fs::write(&target, &document).with_context(|| format!("writing {}", target.display()))?;
    println!(
        "wrote {} ({} bytes)",
        paths::display_relative(&target),
        document.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Measurement driver (§16.43 items 8 and 9's first heavy task)
// ---------------------------------------------------------------------------

/// One page to measure, plus the reference to compare against when the source has one.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PageJob {
    /// The name the report's row is keyed by.
    name: String,
    image: PathBuf,
    /// `Some` only for `--demo-bubbles` (§16.43 item 8: the other two sources carry no
    /// reference image at all).
    reference: Option<PathBuf>,
}

/// The inpainter config every LaMa cell runs under: the shipped defaults with **only**
/// `inpainting_enabled` turned on, the mirror of [`detector_config`]'s one-field move.
fn inpainter_config() -> InpainterConfig {
    InpainterConfig {
        inpainting_enabled: true,
        ..InpainterConfig::default()
    }
}

/// The LaMa stage for one invocation: the source policy, the shared provider, and the
/// per-cell acquisition latch.
///
/// **Acquisition is a whole-cell condition; inference is a per-page one.** The
/// discriminator is `PipelineError`'s own variant, which `run_inpaint` derives from
/// `InpainterProvider::failures_are_run_fatal()` — cookbook rule 4's "fatality is declared,
/// not inferred": `RunFatal` is only ever produced by a provider that declared its
/// acquisition failures fatal, and every failure after the provider handed over an
/// inpainter is a `Stage` error. So this needs no string matching and no heuristic.
struct InpaintRunner<'a> {
    source: Source,
    provider: Option<&'a dyn InpainterProvider>,
    /// Where the provider was told to look, for the disclosure row.
    model_path: Option<String>,
    /// Cells whose acquisition already failed. §16.38 item 8(d)'s `OnceLock` latch inside
    /// the provider already prevents a second *attempt*; this map is what makes the
    /// failure show up on every one of the cell's pages rather than only the first.
    acquisition_failures: BTreeMap<CellId, String>,
    /// Whether any cell genuinely inpainted, and whether acquisition was sought at all.
    any_ran: bool,
    attempted: bool,
}

/// One page's inpainting attempt: the status the report prints, **and** the image that
/// attempt produced.
///
/// The second half exists because the status alone is not enough to measure the cell. A
/// LaMa cell's final output is `InpaintOutput.clean_inpaint` — the mask-stage page with the
/// inpainted regions composited over it — and comparing that cell against the reference
/// while reading `MaskOutput.cleaned` measures the *pre*-inpaint image, making every LaMa
/// row's reference numbers byte-identical to its non-LaMa sibling's however much the
/// inpainter changed. That is exactly the defect this type was introduced to remove, and it
/// was invisible for as long as `inpaint` returned only a status.
struct InpaintRun {
    status: InpaintStatus,
    /// `Some` exactly when `status.ran()`; `None` in every other state, where the cell's
    /// final output is still the mask stage's `cleaned`.
    ///
    /// That pairing is a convention held by two things, neither of which is the type system:
    /// every non-`Ran` value in [`InpaintRunner::inpaint`] is built through the `From` impl
    /// below, whose `debug_assert` rejects a `Ran`, and the single `Ran` site fills this
    /// field. A hand-written struct literal could still break it. **The `debug_assert` is
    /// compiled out under `--release` — the profile the real benchmark runs under — so in
    /// that build the convention rests entirely on there being one `Ran` construction site.**
    /// If a second one is ever added, add a real (non-debug) check or encode the pairing in
    /// the type (e.g. `enum InpaintRun { Ran { clean_inpaint: ImageHandle, .. }, DidNot(...) }`)
    /// rather than trusting this comment.
    clean_inpaint: Option<ImageHandle>,
}

impl From<InpaintStatus> for InpaintRun {
    /// Every state except `Ran`: no image was produced, so there is nothing to carry.
    fn from(status: InpaintStatus) -> Self {
        debug_assert!(
            !status.ran(),
            "`Ran` must carry its composited page; build it directly, not through `From`"
        );
        Self {
            status,
            clean_inpaint: None,
        }
    }
}

impl InpaintRunner<'_> {
    fn inpaint(
        &mut self,
        cell: CellId,
        original: &ImageHandle,
        raw_mask: &ImageHandle,
        mask_data: &pc_core::MaskData,
    ) -> InpaintRun {
        if !cell.inpainting() {
            return InpaintStatus::NotRequested.into();
        }
        // §16.44 item 2: `--replay` never seeks the model, for any cell.
        if !self.source.attempts_inpainting() {
            return InpaintStatus::NotAttempted.into();
        }
        if let Some(message) = self.acquisition_failures.get(&cell) {
            return InpaintStatus::AcquisitionFailed(message.clone()).into();
        }
        // Handled here rather than left to `run_inpaint`, which reports a missing provider
        // as a `Stage` error and so would land in the *inference*-failure bucket — a
        // misclassification of exactly the distinction §16.44 item 3 asks this enum to keep.
        if self.provider.is_none() {
            return InpaintStatus::AcquisitionFailed(
                "no inpainter provider was built for this invocation".to_owned(),
            )
            .into();
        }
        match run_inpaint(
            original,
            raw_mask,
            mask_data,
            // mode-bench runs no denoise stage, so there is no noise mask to isolate against.
            None,
            MaskerConfig::default().min_mask_thickness,
            &inpainter_config(),
            // Nothing is persisted: mode-bench reports numbers, not images.
            InpaintDests::default(),
            self.provider,
        ) {
            Ok(Some(output)) => {
                self.any_ran = true;
                self.attempted = true;
                InpaintRun {
                    status: InpaintStatus::Ran {
                        tiles_inferred: output.tiles_inferred,
                    },
                    // The composited page, not the raw `inpainting` layer: this is what the
                    // export precedence hands a LaMa run as its final image
                    // (`pc-pipeline`'s `export_sources`, `inpainted: clean_inpaint`), so it
                    // is what this cell must be measured on.
                    clean_inpaint: Some(output.clean_inpaint),
                }
            }
            // `run_inpaint`'s own short-circuit: zero eligible regions, so no model is
            // sought even here. Accurate on these sources, and distinct from `--replay`'s
            // source policy (§16.44 item 3).
            Ok(None) => InpaintStatus::NothingEligible.into(),
            Err(PipelineError::RunFatal(error)) => {
                self.attempted = true;
                let message = error.to_string();
                self.acquisition_failures.insert(cell, message.clone());
                InpaintStatus::AcquisitionFailed(message).into()
            }
            Err(PipelineError::Stage(error)) => {
                self.attempted = true;
                InpaintStatus::InferenceFailed(error.to_string()).into()
            }
        }
    }

    fn facts(&self) -> InpainterFacts {
        InpainterFacts {
            // Stated only when a session was really built from it: the provider verifies
            // the sha256 and then constructs, in that order, so a successful inpaint is
            // evidence of both and nothing weaker is.
            model_path: self.any_ran.then(|| self.model_path.clone()).flatten(),
            digest_verified: self.any_ran,
            session_constructed: self.any_ran,
            attempted: self.attempted,
        }
    }
}

/// mode-bench needs only the in-memory `MaskOutput`, never a persisted per-page image, so
/// every destination stays `None`.
///
/// This is also what makes a cross-cell scratch collision impossible here: the
/// `calibrate-goldens` analog (`planned_demo_bubbles_destinations`) builds a path with no
/// cell component, so reusing it across cells would have had the second cell's
/// `pc_mask::run` overwrite — and effectively re-read — the first cell's files. Writing
/// nothing removes the shared name entirely. If a later task needs these files on disk,
/// `mask_dests` must take the cell and the page into the path, and the collision test
/// this property currently makes unnecessary has to be written then.
fn mask_dests(_cell: CellId, _page: &str) -> pc_mask::MaskDests {
    pc_mask::MaskDests::default()
}

/// Measure one `(page, cell)` pair: `pc_detect::run` → `pc_preprocess::run` →
/// `pc_mask::run` with the cell's own `MaskRefineMode`.
///
/// Every error inside is per-`(page, cell)` (§16.43 item 8's own shape, and this
/// project's per-image classification rule): it becomes [`Outcome::Failed`] for this one
/// row and the caller continues with the next page and the next cell. Nothing here is
/// run-fatal.
fn measure_cell(
    job: &PageJob,
    cell: CellId,
    detector: &dyn pc_detect::TextDetector,
    runner: &mut InpaintRunner<'_>,
) -> Outcome {
    match measure_cell_inner(job, cell, detector, runner) {
        Ok(measured) => Outcome::Measured(Box::new(measured)),
        Err(error) => Outcome::Failed {
            reason: format!("{error:#}"),
        },
    }
}

/// The one call in this module that runs the detector, always with the cell's own
/// `detector_config(cell)`. [`measure_cell_inner`] (the production path) and the test
/// `the_two_cells_configs_take_different_detector_branches_on_the_replay_fixture` both go
/// through this same function — a mutation that hard-codes a mode at either call site can
/// no longer diverge from what the test observes, closing the gap a post-review pass found
/// (the test previously called `pc_detect::run` a second, independent time, so a mutation
/// at `measure_cell_inner`'s own call site went undetected).
fn detect_page(
    job: &PageJob,
    cell: CellId,
    detector: &dyn pc_detect::TextDetector,
) -> Result<pc_detect::DetectOutput> {
    pc_detect::run(
        pc_detect::DetectInput {
            schema_version: pc_core::SCHEMA_VERSION,
            source: ImageHandle::from_path(&job.image),
            original_path: job.image.clone(),
            target_height_lower: 1000,
            target_height_upper: 4000,
            base_image_dest: None,
            raw_mask_dest: None,
            min_mask_coverage: pc_detect::DEFAULT_MIN_MASK_COVERAGE,
            config: detector_config(cell),
        },
        detector,
    )
    .map_err(|error| anyhow!(error.to_string()))
    .with_context(|| "running the detector")
}

/// The detector config one cell runs under: the shipped defaults with **only**
/// `mask_refine_mode` moved to that cell's own mode (§16.43 item 2(a)).
fn detector_config(cell: CellId) -> TextDetectorConfig {
    TextDetectorConfig {
        mask_refine_mode: cell.refine_mode(),
        ..TextDetectorConfig::default()
    }
}

fn measure_cell_inner(
    job: &PageJob,
    cell: CellId,
    detector: &dyn pc_detect::TextDetector,
    runner: &mut InpaintRunner<'_>,
) -> Result<Measured> {
    let detect_output = detect_page(job, cell, detector)?;
    let detected_boxes = detect_output.analytics.blocks_detected;
    // The same `_raw_mask` the production call site passes (`pc-pipeline`'s
    // `single.rs:436-447`), captured before `page` moves into preprocessing.
    let raw_mask = detect_output.page.raw_mask.clone();

    let preprocess_output = pc_preprocess::run(
        pc_preprocess::PreprocessInput {
            schema_version: pc_core::SCHEMA_VERSION,
            page: detect_output.page,
            config: PreprocessorConfig::default(),
            performing_ocr: false,
        },
        None,
    )
    .map_err(|error| anyhow!(error.to_string()))
    .with_context(|| "running preprocessing")?;
    let masking_regions = preprocess_output.page.masking_regions.len();

    let mask_output = pc_mask::run(pc_mask::MaskInput {
        schema_version: pc_core::SCHEMA_VERSION,
        page: preprocess_output.page,
        original_image: ImageHandle::from_path(&job.image),
        config: MaskerConfig::default(),
        extract_text: false,
        debug_outputs: false,
        dests: mask_dests(cell, &job.name),
    })
    .map_err(|error| anyhow!(error.to_string()))
    .with_context(|| "running masking")?;

    let regions = &mask_output.mask_data.regions;
    let failed_regions = regions.iter().filter(|region| region.failed).count();
    let succeeded_regions = regions.len() - failed_regions;
    // A region whose precise mask was blank never reaches `regions` at all (§2.6, and
    // `MaskData::regions`' own doc comment) — the same computation `mask_sweep.rs` makes.
    let dropped_regions = masking_regions.saturating_sub(regions.len());
    // §16.43 item 8: computed for EVERY cell, LaMa or not, by calling the real function —
    // task 2's local mirror of it is gone, so there is no second copy to drift.
    let eligible_regions =
        pc_inpaint::eligible::select_regions(regions, &InpainterConfig::default()).len();

    let InpaintRun {
        status: inpaint,
        clean_inpaint,
    } = runner.inpaint(
        cell,
        &ImageHandle::from_path(&job.image),
        &raw_mask,
        &mask_output.mask_data,
    );

    // This cell's **actual final output** on this page: the composited inpaint when
    // inpainting genuinely ran, and the mask stage's cleaned page in every other state —
    // including `NothingEligible` and a failed inference, where no inpainted image exists and
    // the pipeline's own export precedence would likewise fall back to the masked page.
    // Comparing a LaMa cell against `mask_output.cleaned` unconditionally is what made every
    // LaMa row's reference numbers identical to its non-LaMa sibling's.
    let final_output = clean_inpaint.as_ref().unwrap_or(&mask_output.cleaned);

    let reference = match &job.reference {
        Some(clean_path) => Some(compare_against_reference(
            &job.name,
            &job.image,
            clean_path,
            final_output,
        )?),
        None => None,
    };

    Ok(Measured {
        detected_boxes,
        masking_regions,
        succeeded_regions,
        failed_regions,
        dropped_regions,
        eligible_regions,
        inpaint,
        reference,
    })
}

/// Compare one cell's final output for one page against the vendored reference.
///
/// `final_output` is the cell's **own** last-stage image — `InpaintOutput.clean_inpaint` for
/// a LaMa cell that inpainted, `MaskOutput.cleaned` otherwise — never unconditionally the
/// mask stage's.
fn compare_against_reference(
    name: &str,
    raw_path: &Path,
    clean_path: &Path,
    final_output: &ImageHandle,
) -> Result<GoldenReport> {
    let raw_luma = image::open(raw_path)
        .map(|image| image.to_luma8())
        .with_context(|| format!("decoding source `{}`", paths::display_relative(raw_path)))?;
    let clean_luma = image::open(clean_path)
        .map(|image| image.to_luma8())
        .with_context(|| {
            format!(
                "decoding reference `{}`",
                paths::display_relative(clean_path)
            )
        })?;
    let cleaned_luma = final_output
        .load()
        .map(|image| image.to_luma8())
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| "loading the cell's final output")?;
    if raw_luma.dimensions() != clean_luma.dimensions()
        || raw_luma.dimensions() != cleaned_luma.dimensions()
    {
        bail!(
            "dimension mismatch: source {:?}, reference {:?}, cleaned output {:?}",
            raw_luma.dimensions(),
            clean_luma.dimensions(),
            cleaned_luma.dimensions()
        );
    }
    Ok(GoldenReport::compare_gray_with_shape(
        name,
        &raw_luma,
        &clean_luma,
        &cleaned_luma,
        REFERENCE_DILATE_RADIUS,
    ))
}

/// Run every requested cell over every page, recording one outcome per pair.
///
/// The page is registered even when every cell on it failed, so a failing page still gets
/// its rows in the report instead of vanishing from the table.
fn measure_jobs(
    benchmark: &mut Benchmark,
    jobs: &[PageJob],
    cells: &[CellId],
    detector: &dyn pc_detect::TextDetector,
    runner: &mut InpaintRunner<'_>,
) {
    for job in jobs {
        benchmark.register_page(&job.name);
        for cell in cells {
            benchmark.record(&job.name, *cell, measure_cell(job, *cell, detector, runner));
        }
    }
}

fn demo_bubble_jobs() -> Vec<PageJob> {
    pc_testkit::paths::DEMO_BUBBLES
        .iter()
        .map(|bubble| PageJob {
            name: bubble.name.to_owned(),
            image: bubble.path(pc_testkit::paths::BubbleKind::Raw),
            reference: Some(bubble.path(pc_testkit::paths::BubbleKind::Clean)),
        })
        .collect()
}

/// The three input sources (§16.43 item 8).
///
/// Everything that fails here is **command-fatal**, not per-image: an unparseable
/// `--detector` spec, a detector model that fails its digest check, an unreadable
/// `--pages` directory. Per-page and per-cell failures are handled one level down, in
/// [`measure_cell`], and never reach this signature.
fn measure(
    source: Source,
    cells: &[CellId],
    pages: Option<&Path>,
    detector: Option<&str>,
    device: Device,
) -> Result<(Benchmark, RunFacts)> {
    let mut benchmark = Benchmark::new(source, cells);

    // §16.43 item 5 (D3): reuse `pc_cli::inpainter::build_provider` rather than hand-rolling
    // a second acquisition mechanism. `build_provider` itself loads nothing — the 207 MB
    // artifact is only touched by the provider's own lazy latch, and `run_inpaint` only
    // reaches that latch on a page with a non-empty eligible set, so a run where no cell has
    // an eligible region pays nothing. The `enabled` flag carries mode-bench's own source
    // policy (§16.44 item 2): on `--replay` no provider is built at all, so there is nothing
    // for a stray call to acquire through.
    let wants_inpainting =
        source.attempts_inpainting() && cells.iter().any(|cell| cell.inpainting());
    let cache_root = pc_cli::paths::default_cache_dir();
    let provider = build_provider(wants_inpainting, None, &cache_root, device);
    let mut runner = InpaintRunner {
        source,
        provider: provider.as_deref(),
        model_path: wants_inpainting.then(|| {
            pc_cli::paths::models_dir(&cache_root)
                .join(pc_models::LAMA_MANGA_INPAINTER.file_name)
                .display()
                .to_string()
        }),
        acquisition_failures: BTreeMap::new(),
        any_ran: false,
        attempted: false,
    };

    let detector_facts = match source {
        Source::Replay => {
            let fixture_dir = paths::recorded_root().join("detector");
            let jobs = vec![PageJob {
                name: REPLAY_STEM.to_owned(),
                image: fixture_dir.join(format!("{REPLAY_STEM}.jpg")),
                reference: None,
            }];
            let detector = pc_detect::ReplayDetector::new(&fixture_dir, REPLAY_STEM);
            measure_jobs(&mut benchmark, &jobs, cells, &detector, &mut runner);
            // No real model is ever resolved for `--replay` — `ReplayDetector` replays a
            // committed fixture, so the placeholder facts are the honest ones here, not a
            // stand-in for something unimplemented.
            DetectorFacts::default()
        }
        Source::DemoBubbles => {
            let spec = detector.ok_or_else(|| {
                anyhow!("--demo-bubbles requires --detector onnx:<path> (clap enforces this)")
            })?;
            measure_with_real_detector(
                &mut benchmark,
                &demo_bubble_jobs(),
                cells,
                spec,
                &mut runner,
            )?
        }
        Source::Pages => {
            let dir = pages.ok_or_else(|| anyhow!("--pages is required for this source"))?;
            let spec = detector.ok_or_else(|| {
                anyhow!("--pages requires --detector onnx:<path> (clap enforces this)")
            })?;
            let jobs = local_page_jobs(dir)?;
            measure_with_real_detector(&mut benchmark, &jobs, cells, spec, &mut runner)?
        }
    };
    let facts = RunFacts {
        detector: detector_facts,
        inpainter: runner.facts(),
    };
    Ok((benchmark, facts))
}

fn local_page_jobs(dir: &Path) -> Result<Vec<PageJob>> {
    if !dir.is_dir() {
        bail!("--pages is not a directory: {}", dir.display());
    }
    let mut paths = std::fs::read_dir(dir)
        .with_context(|| format!("reading pages directory {}", dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|path| {
        path.is_file()
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    matches!(
                        extension.to_ascii_lowercase().as_str(),
                        "png" | "jpg" | "jpeg" | "webp" | "tif" | "tiff" | "bmp" | "pnm"
                    )
                })
    });
    paths.sort();
    if paths.is_empty() {
        bail!("--pages contains no supported images: {}", dir.display());
    }
    paths
        .into_iter()
        .map(|path| {
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| anyhow!("page has no UTF-8 file stem: {}", path.display()))?
                .to_owned();
            Ok(PageJob {
                name,
                image: path,
                reference: None,
            })
        })
        .collect()
}

#[cfg(feature = "onnx")]
fn measure_with_real_detector(
    benchmark: &mut Benchmark,
    jobs: &[PageJob],
    cells: &[CellId],
    detector_spec: &str,
    runner: &mut InpaintRunner<'_>,
) -> Result<DetectorFacts> {
    let model = crate::env::parse_detector_spec(detector_spec)?;
    let model = if model.is_relative() {
        paths::workspace_root().join(model)
    } else {
        model
    };
    pc_models::verify_sha256(&model, pc_models::COMIC_TEXT_DETECTOR.sha256)
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| format!("verifying detector model {}", model.display()))?;
    let detector = pc_detect::onnx::OnnxDetector::from_path_with_config(
        &model,
        &TextDetectorConfig::default(),
    )
    .map_err(|error| anyhow!(error.to_string()))
    .with_context(|| "constructing the ONNX detector")?;
    measure_jobs(benchmark, jobs, cells, &detector, runner);
    // Reached only after verify_sha256 and from_path_with_config both succeeded above, so
    // both facts are genuinely true here — not asserted ahead of the work that earns them.
    Ok(DetectorFacts {
        model_path: Some(model.display().to_string()),
        digest_verified: true,
        session_constructed: true,
    })
}

#[cfg(not(feature = "onnx"))]
fn measure_with_real_detector(
    _benchmark: &mut Benchmark,
    _jobs: &[PageJob],
    _cells: &[CellId],
    detector_spec: &str,
    _runner: &mut InpaintRunner<'_>,
) -> Result<DetectorFacts> {
    let _ = crate::env::parse_detector_spec(detector_spec)?;
    bail!(
        "--demo-bubbles and --pages require the real ONNX detector; rerun with \
         `cargo xtask --features onnx mode-bench ...`"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(failed: usize, eligible: usize) -> Outcome {
        const SUCCEEDED: usize = 3;
        Outcome::Measured(Box::new(Measured {
            detected_boxes: 7,
            masking_regions: SUCCEEDED + failed,
            succeeded_regions: SUCCEEDED,
            failed_regions: failed,
            dropped_regions: 1,
            eligible_regions: eligible,
            inpaint: InpaintStatus::NotRequested,
            reference: None,
        }))
    }

    /// The same fabricated measurement, with an explicit inpainting outcome — for the
    /// §16.44 item 3 and item 4 tests, which are entirely about that field.
    fn measured_with(failed: usize, eligible: usize, inpaint: InpaintStatus) -> Outcome {
        let mut outcome = measured(failed, eligible);
        if let Outcome::Measured(inner) = &mut outcome {
            inner.inpaint = inpaint;
        }
        outcome
    }

    fn with_reference(mean_abs_diff: f64) -> Outcome {
        let mut outcome = measured(2, 1);
        if let Outcome::Measured(inner) = &mut outcome {
            inner.reference = Some(GoldenReport {
                name: "page".into(),
                dims: (10, 10),
                exact_fraction: 0.5,
                max_delta: 9,
                mean_abs_diff,
                ssim: 0.75,
                shape_iou: None,
                shape_subset_of_dilated: None,
            });
        }
        outcome
    }

    use pc_core::StageError;
    use pc_inpaint::stub::{StubInpainter, StubMode};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A runner with no provider at all — the shape `--replay` uses (§16.44 item 2: no
    /// provider is built for that source at all).
    fn no_inpaint_runner(source: Source) -> InpaintRunner<'static> {
        runner_with(source, None)
    }

    fn runner_with(source: Source, provider: Option<&dyn InpainterProvider>) -> InpaintRunner<'_> {
        InpaintRunner {
            source,
            provider,
            model_path: Some("/models/lama-manga.onnx".to_owned()),
            acquisition_failures: BTreeMap::new(),
            any_ran: false,
            attempted: false,
        }
    }

    /// A provider whose acquisition always fails and which **declares** that failure
    /// run-fatal — the same declaration `pc_cli::inpainter::OnnxInpainterProvider` makes.
    /// The counter is the assertion: a whole-cell condition must be sought once, not once
    /// per page.
    struct FailingAcquisition {
        attempts: AtomicUsize,
    }

    impl InpainterProvider for FailingAcquisition {
        fn inpainter(&self) -> Result<Arc<dyn pc_inpaint::Inpainter>, StageError> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            Err(StageError::Model(
                "the optional LaMa inpainting model is missing".to_owned(),
            ))
        }

        fn failures_are_run_fatal(&self) -> bool {
            true
        }
    }

    /// An inpainter that fails on its **first** tile and succeeds on every later one, so a
    /// two-page run has exactly one failing page. Wrapping the `testkit` stub keeps the
    /// success path a real `inpaint_page` composite rather than a second hand-rolled one.
    struct FailFirstTile {
        calls: AtomicUsize,
        inner: StubInpainter,
    }

    impl pc_inpaint::Inpainter for FailFirstTile {
        fn inpaint_tile(
            &self,
            tile: &image::RgbImage,
            mask: &pc_imageops::BinaryMask,
        ) -> Result<image::RgbImage, StageError> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(StageError::Inference("tile 0 refused by the test".into()));
            }
            self.inner.inpaint_tile(tile, mask)
        }
    }

    /// A provider that hands over an already-built inpainter. Acquisition never fails here,
    /// so anything that does fail is unambiguously an inference failure.
    struct WorkingProvider {
        inpainter: Arc<dyn pc_inpaint::Inpainter>,
        attempts: AtomicUsize,
    }

    impl WorkingProvider {
        fn new(inpainter: Arc<dyn pc_inpaint::Inpainter>) -> Self {
            Self {
                inpainter,
                attempts: AtomicUsize::new(0),
            }
        }
    }

    impl InpainterProvider for WorkingProvider {
        fn inpainter(&self) -> Result<Arc<dyn pc_inpaint::Inpainter>, StageError> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::clone(&self.inpainter))
        }

        fn failures_are_run_fatal(&self) -> bool {
            true
        }
    }

    /// A synthetic page whose one detected block is surrounded by **noise**, so the masker
    /// cannot fit a clean mask and records the region `failed` — which is
    /// `select_regions`' first eligibility set. `eligible = false` uses a flat background,
    /// which fits cleanly and yields zero eligible regions.
    ///
    /// Both facts are asserted, not assumed, by
    /// `the_two_synthetic_pages_really_do_differ_in_eligibility` below — without that, every
    /// test using these pages could pass for the wrong reason.
    fn synthetic_page(dir: &Path, name: &str, eligible: bool) -> PageJob {
        let path = dir.join(format!("{name}.png"));
        image::GrayImage::from_fn(240, 1000, |x, y| {
            let inside = (20..200).contains(&x) && (20..400).contains(&y);
            if inside {
                image::Luma([20])
            } else if eligible {
                image::Luma([((x * 37 + y * 91) % 256) as u8])
            } else {
                image::Luma([235])
            }
        })
        .save(&path)
        .expect("writing the synthetic page");
        PageJob {
            name: name.to_owned(),
            image: path,
            reference: None,
        }
    }

    fn synthetic_detector() -> pc_detect::MockDetector {
        pc_detect::MockDetector::new()
            .with_blocks(vec![pc_detect::RawBlock {
                rect: pc_core::Rect::new(20, 20, 200, 400),
                class_index: 0,
                confidence: 0.9,
            }])
            .with_block_fill(255)
    }

    fn status_of(benchmark: &Benchmark, page: &str, cell: CellId) -> InpaintStatus {
        match benchmark.outcome(page, cell) {
            Some(Outcome::Measured(measured)) => measured.inpaint.clone(),
            other => panic!("{page}/{} is not a Measured row: {other:?}", cell.name()),
        }
    }

    fn eligible_of(benchmark: &Benchmark, page: &str, cell: CellId) -> usize {
        match benchmark.outcome(page, cell) {
            Some(Outcome::Measured(measured)) => measured.eligible_regions,
            other => panic!("{page}/{} is not a Measured row: {other:?}", cell.name()),
        }
    }

    fn cpu_policy() -> DevicePolicy {
        resolve(Device::Cpu, DeviceSupport::compiled()).expect("cpu resolves on every build")
    }

    fn section<'a>(document: &'a str, heading: &str) -> &'a str {
        let start = document
            .find(heading)
            .unwrap_or_else(|| panic!("missing heading {heading}\n{document}"));
        let rest = &document[start + heading.len()..];
        match rest.find("\n## ") {
            Some(end) => &rest[..end],
            None => rest,
        }
    }

    // --- §16.43 item 3 (D1): the cell space and its default ------------------

    #[test]
    fn default_cells_spec_parses_to_simple_annotation_and_simple_plus_lama() {
        assert_eq!(
            parse_cells(DEFAULT_CELLS_SPEC)
                .expect("default spec parses")
                .0,
            vec![CellId::Simple, CellId::Annotation, CellId::SimpleLama],
        );
        // The default really is the three-cell subset, not all four.
        assert_eq!(
            CellId::DEFAULT,
            &[CellId::Simple, CellId::Annotation, CellId::SimpleLama]
        );
        assert!(!CellId::DEFAULT.contains(&CellId::AnnotationLama));
    }

    #[test]
    fn annotation_plus_lama_is_an_accepted_value_and_all_expands_to_the_four_cells() {
        assert_eq!(
            parse_cells("annotation+lama").expect("accepted").0,
            vec![CellId::AnnotationLama],
        );
        assert_eq!(
            parse_cells("all").expect("accepted").0,
            vec![
                CellId::Simple,
                CellId::Annotation,
                CellId::SimpleLama,
                CellId::AnnotationLama,
            ],
        );
    }

    #[test]
    fn an_unknown_cell_name_is_rejected_and_the_error_names_every_accepted_value() {
        let error = parse_cells("simple,bogus").expect_err("unknown name must be rejected");
        assert!(error.contains("bogus"), "{error}");
        for accepted in [
            "simple",
            "annotation",
            "simple+lama",
            "annotation+lama",
            "all",
        ] {
            assert!(
                error.contains(accepted),
                "error omitted `{accepted}`: {error}"
            );
        }
    }

    #[test]
    fn an_empty_cell_token_is_rejected_rather_than_silently_dropped() {
        assert!(parse_cells("").is_err());
        assert!(parse_cells("simple,,annotation").is_err());
    }

    #[test]
    fn repeated_cell_names_collapse_keeping_first_occurrence_order() {
        assert_eq!(
            parse_cells("simple+lama,all,simple").expect("accepted").0,
            vec![
                CellId::SimpleLama,
                CellId::Simple,
                CellId::Annotation,
                CellId::AnnotationLama,
            ],
        );
    }

    #[test]
    fn a_cells_mask_mode_and_inpainting_flag_decompose_the_two_factors() {
        // §16.43 item 2(a): the space is (mask mode) x (inpainting on/off).
        assert_eq!(CellId::SimpleLama.mask_mode(), "Simple");
        assert_eq!(CellId::AnnotationLama.mask_mode(), "Annotation");
        assert!(CellId::SimpleLama.inpainting());
        assert!(CellId::AnnotationLama.inpainting());
        assert!(!CellId::Simple.inpainting());
        assert!(!CellId::Annotation.inpainting());
    }

    // --- §16.43 item 4 (D2): the three eligibility segments -------------------

    fn two_cell_benchmark() -> Benchmark {
        let mut benchmark = Benchmark::new(Source::Replay, &[CellId::Simple, CellId::Annotation]);
        // p1 is eligible only under Simple; p2 only under Annotation.
        benchmark.record("p1", CellId::Simple, measured(2, 3));
        benchmark.record("p1", CellId::Annotation, measured(0, 0));
        benchmark.record("p2", CellId::Simple, measured(4, 0));
        benchmark.record("p2", CellId::Annotation, measured(6, 2));
        benchmark
    }

    #[test]
    fn full_population_holds_every_page_and_its_cross_cell_mean_is_computable() {
        let benchmark = two_cell_benchmark();
        assert_eq!(
            benchmark.full_population(),
            BTreeSet::from(["p1".to_owned(), "p2".to_owned()]),
        );
        // Hand-computed: Simple failed 2 and 4 -> 3.0; Annotation failed 0 and 6 -> 3.0.
        let full = benchmark.full_population();
        assert_eq!(
            benchmark.mean(CellId::Simple, &full, Metric::FailedRegions),
            Some(3.0),
        );
        assert_eq!(
            benchmark.mean(CellId::Annotation, &full, Metric::EligibleRegions),
            Some(1.0),
        );
    }

    #[test]
    fn each_cells_eligible_subset_is_its_own_page_set_not_a_shared_one() {
        let benchmark = two_cell_benchmark();
        // Set identity, not cardinality: both sets have size 1, so a swapped lookup would
        // pass a count assertion and fails this one.
        assert_eq!(
            benchmark.eligible_pages(CellId::Simple),
            BTreeSet::from(["p1".to_owned()]),
        );
        assert_eq!(
            benchmark.eligible_pages(CellId::Annotation),
            BTreeSet::from(["p2".to_owned()]),
        );
    }

    #[test]
    fn the_per_cell_subset_table_is_captioned_not_cross_comparable() {
        let benchmark = two_cell_benchmark();
        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        let subsets = section(&document, "## 4. Eligibility segments");
        let heading = subsets
            .find("### 4.2 Per-cell eligible subsets")
            .expect("per-cell subsection");
        let after = &subsets[heading..];
        let end = after.find("### 4.3").unwrap_or(after.len());
        assert!(
            after[..end].contains(NOT_CROSS_COMPARABLE),
            "per-cell subsets are not captioned `{NOT_CROSS_COMPARABLE}`: {}",
            &after[..end]
        );
    }

    #[test]
    fn common_eligible_is_the_intersection_over_the_cells_actually_run() {
        let mut benchmark = two_cell_benchmark();
        assert_eq!(benchmark.common_eligible(), BTreeSet::new());
        benchmark.record("p3", CellId::Simple, measured(1, 5));
        benchmark.record("p3", CellId::Annotation, measured(1, 4));
        assert_eq!(
            benchmark.common_eligible(),
            BTreeSet::from(["p3".to_owned()]),
        );
    }

    /// Post-review fix (§16.43 item 4): "actually run" excludes BLOCKED cells, not
    /// just unrequested ones. A requested-but-blocked cell (e.g. LaMa when the model
    /// is absent — the common CI case) must not poison the intersection to always-
    /// empty just because it was requested. This is a DIFFERENT claim from the test
    /// above, which never distinguishes requested from run since nothing is blocked
    /// there — that gap is exactly what let the pre-fix version through review clean.
    #[test]
    fn a_blocked_cell_does_not_poison_the_intersection_even_though_it_was_requested() {
        let mut benchmark = Benchmark::new(
            Source::Replay,
            &[CellId::Simple, CellId::Annotation, CellId::SimpleLama],
        );
        // Both non-blocked cells are eligible on p1; SimpleLama is requested but
        // never measured (blocked) rather than measured-and-ineligible.
        benchmark.record("p1", CellId::Simple, measured(2, 3));
        benchmark.record("p1", CellId::Annotation, measured(0, 4));
        benchmark.block_cell(CellId::SimpleLama, "the LaMa model is not available");

        assert_eq!(
            benchmark.common_eligible(),
            BTreeSet::from(["p1".to_owned()]),
            "a blocked cell must not collapse the intersection to empty"
        );
        assert_eq!(
            benchmark.common_eligible_contributing_cells(),
            vec![CellId::Simple, CellId::Annotation],
            "the blocked cell must not appear in the contributing set the caption names"
        );
    }

    /// The degenerate case the fix must still get right: if EVERY requested cell is
    /// blocked, nothing actually ran, and the intersection must still be empty via
    /// `contributing.is_empty()` — not accidentally "empty because iteration never
    /// started" versus "empty because nothing overlapped" being conflated.
    #[test]
    fn an_intersection_over_only_blocked_cells_is_empty_not_a_false_full_match() {
        let mut benchmark = Benchmark::new(Source::Replay, &[CellId::SimpleLama]);
        benchmark.block_cell(CellId::SimpleLama, "the LaMa model is not available");
        assert_eq!(benchmark.common_eligible(), BTreeSet::new());
        assert!(benchmark.common_eligible_contributing_cells().is_empty());
    }

    #[test]
    fn an_empty_common_eligible_intersection_states_so_and_prints_no_mean() {
        let benchmark = two_cell_benchmark();
        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        let segments = section(&document, "## 4. Eligibility segments");
        let start = segments.find("### 4.3").expect("intersection subsection");
        let intersection = &segments[start..];
        assert!(
            intersection.contains("intersection is **empty**"),
            "empty intersection not stated: {intersection}"
        );
        assert!(
            !intersection.contains("Mean failed regions"),
            "a mean table was printed over an empty intersection: {intersection}"
        );
    }

    #[test]
    fn the_intersection_caption_names_the_cell_set_it_was_computed_over() {
        let two = two_cell_benchmark();
        let mut three = Benchmark::new(
            Source::Replay,
            &[CellId::Simple, CellId::Annotation, CellId::SimpleLama],
        );
        three.record("p1", CellId::Simple, measured(2, 3));
        three.record("p1", CellId::Annotation, measured(0, 0));
        three.record("p1", CellId::SimpleLama, measured(0, 1));

        let render = |benchmark: &Benchmark| {
            render_document(
                benchmark,
                &cpu_policy(),
                &stage_disclosures(benchmark.cells()),
            )
        };
        let two_doc = render(&two);
        let three_doc = render(&three);
        assert!(two_doc.contains("### 4.3 Common-eligible intersection over (simple, annotation)"));
        assert!(three_doc.contains(
            "### 4.3 Common-eligible intersection over (simple, annotation, simple+lama)"
        ));
        // Adding a cell must visibly re-segment: the two captions are not the same string.
        assert_ne!(
            two_doc.contains("(simple, annotation, simple+lama)"),
            three_doc.contains("(simple, annotation, simple+lama)"),
        );
    }

    #[test]
    fn a_mean_over_zero_contributing_rows_is_none_rather_than_zero() {
        let benchmark = two_cell_benchmark();
        assert_eq!(
            benchmark.mean(CellId::Simple, &BTreeSet::new(), Metric::FailedRegions),
            None,
        );
        // No cell has a reference on this source, so the reference metric contributes nothing.
        assert_eq!(
            benchmark.mean(
                CellId::Simple,
                &benchmark.full_population(),
                Metric::ReferenceMeanAbsDiff
            ),
            None,
        );
    }

    // --- §16.43 item 6 (D4): device disclosure --------------------------------

    #[test]
    fn the_device_policy_report_is_printed_verbatim_exactly_once() {
        let policy = cpu_policy();
        let benchmark = two_cell_benchmark();
        let document = render_document(&benchmark, &policy, &stage_disclosures(benchmark.cells()));

        assert_eq!(
            document.matches(&policy.report()).count(),
            1,
            "DevicePolicy::report() must appear exactly once, verbatim"
        );
        // Anti-vacuity literal: the CPU statement's own text, hard-coded here so this
        // cannot pass by rendering some other (or empty) string that `report()` also
        // happens to produce.
        assert!(document.contains(
            "device: requested cpu; no execution provider registered explicitly \
             (ONNX Runtime's built-in CPU provider is implicit); per-node operator placement \
             is not claimed"
        ));
        // And it is never restated per session.
        assert_eq!(document.matches("device: requested").count(), 1);
    }

    #[test]
    fn per_stage_rows_carry_only_per_stage_facts_and_cross_reference_the_one_statement() {
        let benchmark = Benchmark::new(Source::Replay, &[CellId::Simple, CellId::SimpleLama]);
        let stages = stage_disclosures(benchmark.cells());
        let device = render_device_section(&cpu_policy(), &stages);

        // The stage table header names exactly the five per-stage facts §16.43 item 6 lists.
        assert!(device.contains(
            "| Stage | Model path | Expected sha256 | Digest verified? | Session constructed? | Constructing function |"
        ));
        // Hard-coded pins from pc_models, not read back out of the renderer.
        assert!(device.contains("1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"));
        assert!(device.contains("50a1abae0d73bd46d08eae36c8590cd59ad09029494c9698702b050ef00b0100"));
        assert!(device.contains("cross-reference the statement above"));
    }

    #[test]
    fn the_inpainter_stage_row_appears_only_when_an_inpainting_cell_was_requested() {
        let without = stage_disclosures(&[CellId::Simple, CellId::Annotation]);
        let with = stage_disclosures(&[CellId::Simple, CellId::SimpleLama]);
        assert_eq!(
            without.iter().map(|s| s.stage).collect::<Vec<_>>(),
            vec!["detector"]
        );
        assert_eq!(
            with.iter().map(|s| s.stage).collect::<Vec<_>>(),
            vec!["detector", "inpainter"]
        );
    }

    #[test]
    fn the_detector_and_inpainter_rows_disclose_their_two_device_mechanisms_verbatim() {
        let stages = stage_disclosures(&[CellId::Simple, CellId::SimpleLama]);
        let device = render_device_section(&cpu_policy(), &stages);
        assert!(
            device.contains(DETECTOR_DEVICE_MECHANISM),
            "detector mechanism sentence missing: {device}"
        );
        assert!(
            device.contains(
                "its construction route (`from_path_for_device`) re-resolves the same requested \
                 device through the same resolver"
            ),
            "inpainter mechanism sentence missing: {device}"
        );
        // The mechanism sentences name the real constructors, so a renamed constructor
        // does not leave a stale claim standing.
        assert!(device.contains("`pc_detect::onnx::OnnxDetector::from_path_with_config`"));
        assert!(device.contains("`pc_inpaint::onnx::OnnxInpainter::from_path_for_device`"));
    }

    // --- §16.43 item 8: a blocked cell does not take the others down ----------

    #[test]
    fn a_blocked_cell_renders_as_blocked_without_removing_other_cells_rows() {
        let mut benchmark = Benchmark::new(Source::Replay, &[CellId::Simple, CellId::SimpleLama]);
        benchmark.record("p1", CellId::Simple, measured(2, 3));
        benchmark.register_page("p1");
        benchmark.block_cell(
            CellId::SimpleLama,
            "lama-manga.onnx is absent; run `panel-ocr models download --include-optional`",
        );

        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        let table = section(&document, "## 3. Per-page, per-cell measurements");

        // The surviving cell keeps its real numbers.
        assert!(
            table.contains(
                "| p1 | simple | Simple | 7 | 5 | 3 | 2 | 1 | 3 | no | — | — | measured |"
            ),
            "the unblocked cell's row is missing or changed: {table}"
        );
        // The blocked cell is present as a BLOCKED row carrying its reason.
        assert!(
            table.contains("| p1 | simple+lama | Simple |")
                && table.contains(
                    "**BLOCKED** — lama-manga.onnx is absent; run `panel-ocr models download --include-optional`"
                ),
            "the blocked cell's row or reason is missing: {table}"
        );
        // Both requested cells appear for the page — identity, not a count.
        let cells_in_table: BTreeSet<&str> = ["simple", "simple+lama"]
            .into_iter()
            .filter(|name| table.contains(&format!("| p1 | {name} |")))
            .collect();
        assert_eq!(
            cells_in_table,
            BTreeSet::from(["simple", "simple+lama"]),
            "a blocked cell removed rows from the report: {table}"
        );
    }

    #[test]
    fn a_per_page_failure_marks_only_that_page_leaving_the_cells_other_pages_measured() {
        let mut benchmark = Benchmark::new(Source::Replay, &[CellId::Simple]);
        benchmark.record("p1", CellId::Simple, measured(2, 3));
        benchmark.record(
            "p2",
            CellId::Simple,
            Outcome::Failed {
                reason: "decoding p2 failed".to_owned(),
            },
        );
        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        let table = section(&document, "## 3. Per-page, per-cell measurements");
        assert!(table
            .contains("| p1 | simple | Simple | 7 | 5 | 3 | 2 | 1 | 3 | no | — | — | measured |"));
        assert!(table.contains("**FAILED** — decoding p2 failed"));
        assert!(
            !table.contains("**BLOCKED**"),
            "a page failure blocked the cell: {table}"
        );
    }

    // --- §16.43 item 7 (D5): the reference column and the verdict -------------

    #[test]
    fn the_reference_column_is_headed_as_agreement_and_never_as_quality() {
        let mut benchmark = Benchmark::new(Source::DemoBubbles, &[CellId::Simple]);
        benchmark.record("crop", CellId::Simple, with_reference(0.25));
        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );

        assert!(document.contains("Reference agreement (not quality)"));
        for forbidden in ["Reference quality", "reference quality", "Quality |"] {
            assert!(
                !document.contains(forbidden),
                "report headed the reference column as quality (`{forbidden}`)"
            );
        }
        assert!(document.contains("exact 0.500000 / mad 0.250000 / ssim 0.750000"));
    }

    #[test]
    fn the_reference_section_states_all_three_of_d5s_binding_conditions() {
        let mut benchmark = Benchmark::new(Source::DemoBubbles, &[CellId::Simple]);
        benchmark.record("crop", CellId::Simple, with_reference(0.25));
        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        let reference = section(&document, "## 5. Reference comparison");

        // (i) the reference's producing version/profile/environment are unrecorded.
        assert!(reference.contains("**unrecorded**") && reference.contains("§16.37 item 3"));
        // (ii) agreement, never quality.
        assert!(reference.contains("never \"quality\""));
        // (iii) no pass/fail assertion, ever.
        assert!(reference.contains("No pass/fail assertion is made against `_clean.png`, ever"));
        assert!(reference.contains("§15.2 stays fully in force"));
    }

    #[test]
    fn a_source_without_a_reference_says_so_instead_of_claiming_a_comparison() {
        let benchmark = two_cell_benchmark();
        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        let reference = section(&document, "## 5. Reference comparison");
        assert!(reference.contains("no reference"), "{reference}");
        assert!(
            !reference.contains("**unrecorded**"),
            "a reference caveat was printed for a source with no reference: {reference}"
        );
    }

    #[test]
    fn the_verdict_never_ranks_cells_on_distance_to_the_reference() {
        let mut benchmark =
            Benchmark::new(Source::DemoBubbles, &[CellId::Simple, CellId::SimpleLama]);
        benchmark.record("crop", CellId::Simple, with_reference(0.25));
        benchmark.record("crop", CellId::SimpleLama, with_reference(0.90));
        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        let verdict = section(&document, "## 6. Verdict");

        for ranking in [
            "closest to",
            "furthest from",
            "best match",
            "ranked by",
            "highest agreement",
            "lowest mean abs",
            "winner",
            "wins",
            "PASS",
            "FAIL",
        ] {
            assert!(
                !verdict.contains(ranking),
                "verdict ranks or gates on the reference (`{ranking}`): {verdict}"
            );
        }
        assert!(verdict.contains("ranks nothing and asserts nothing"));
        assert!(verdict.contains("not ordered by their distance to any reference image"));
    }

    #[test]
    fn the_report_states_that_it_does_not_close_decision_point_d2() {
        let benchmark = two_cell_benchmark();
        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        assert!(document.contains("does **not** close §16.38 item 17(b) decision point D2"));
        assert!(document.contains("§15.10(a)'s independence rule"));
        assert!(document.contains("runs no upstream"));
    }

    // --- §16.43 item 1: the non-gating banner convention -----------------------

    #[test]
    fn the_report_carries_the_generated_in_full_and_non_gating_banner() {
        let benchmark = two_cell_benchmark();
        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        assert!(document.starts_with("# Mode comparison benchmark (spec §16.43)\n"));
        assert!(document
            .contains("**Generated in full by `cargo xtask mode-bench` — do not hand-edit.**"));
        assert!(document.contains("**non-gating**"));
        assert!(document.contains("gates no CI job and\nblocks no merge"));
    }

    // --- §16.43 item 8: the eligibility predicate ------------------------------

    fn region(
        failed: bool,
        std_deviation: f64,
        thickness: Option<u32>,
    ) -> pc_core::MaskRegionStats {
        pc_core::MaskRegionStats {
            rect: pc_core::Rect::new(0, 0, 4, 4),
            std_deviation,
            failed,
            thickness,
        }
    }

    /// Hand-derived oracle for **`pc_inpaint::eligible::select_regions` itself**
    /// (`crates/pc-inpaint/src/eligible.rs:64-97`).
    ///
    /// Task 2 could only assert this table against xtask's own local mirror of that
    /// function, because §16.43 item 9 reserved the `pc-inpaint` dependency for this task.
    /// The mirror is now deleted and the dependency exists, so the same hand-derived table
    /// is asserted against the real function — a genuine equivalence check instead of a
    /// check of a duplicate against itself.
    ///
    /// Every expectation below is derived from `select_regions`' *text* plus
    /// `InpainterConfig::default()`'s two literals, never from running it. The two boundary
    /// rows catch a `>=`→`>` or `<=`→`<` drift, and the last row is the
    /// `thickness.is_some()` clause upstream comments as *"For box masks, this is none. We
    /// don't need to inpaint those, they are always good."*
    #[test]
    fn the_eligibility_predicate_matches_the_hand_derived_upstream_cases() {
        let config = InpainterConfig::default();
        // The boundary rows below mean what their names say only at these two values.
        assert_eq!(config.inpainting_min_std_dev, 15.0);
        assert_eq!(config.min_inpainting_radius, 7);

        let cases: [(&str, pc_core::MaskRegionStats, bool); 5] = [
            (
                "a failed region is eligible whatever its std dev or thickness",
                region(true, 0.0, None),
                true,
            ),
            (
                "both bounds met inclusively: std dev == 15.0 and thickness == 7",
                region(false, 15.0, Some(7)),
                true,
            ),
            (
                "std dev just under the inclusive lower bound",
                region(false, 14.999, Some(7)),
                false,
            ),
            (
                "thickness just over the inclusive upper bound",
                region(false, 15.0, Some(8)),
                false,
            ),
            (
                "a box mask (thickness None) is never eligible, however high its std dev",
                region(false, 1_000.0, None),
                false,
            ),
        ];

        for (why, stats, expected) in &cases {
            assert_eq!(
                pc_inpaint::eligible::select_regions(std::slice::from_ref(stats), &config).len(),
                usize::from(*expected),
                "{why}"
            );
        }

        // Anti-vacuity literal: the whole table counted at once is 2 — a number a
        // predicate that answered all-true (5) or all-false (0) cannot produce.
        let all: Vec<pc_core::MaskRegionStats> =
            cases.iter().map(|(_, stats, _)| stats.clone()).collect();
        assert_eq!(pc_inpaint::eligible::select_regions(&all, &config).len(), 2);
        // Identity, not cardinality: the two selected rows are case 0 (the failed region)
        // and case 1 (both bounds met inclusively), in `select_regions`' own documented
        // order — failed first, then poorly-fitted. A predicate that selected cases 2 and 3
        // instead would still count 2 and pass the assertion above.
        assert_eq!(
            pc_inpaint::eligible::select_regions(&all, &config)
                .iter()
                .map(|region| region.index)
                .collect::<Vec<_>>(),
            vec![0, 1],
        );
    }

    // --- §16.43 item 9: per-cell mask-mode selection ---------------------------

    #[test]
    fn the_detector_config_a_cell_runs_under_carries_that_cells_own_refine_mode() {
        // §16.43 item 2(a): a `+lama` cell's mask mode is its non-LaMa peer's.
        assert_eq!(
            detector_config(CellId::Simple).mask_refine_mode,
            MaskRefineMode::Simple
        );
        assert_eq!(
            detector_config(CellId::SimpleLama).mask_refine_mode,
            MaskRefineMode::Simple
        );
        assert_eq!(
            detector_config(CellId::Annotation).mask_refine_mode,
            MaskRefineMode::Annotation
        );
        assert_eq!(
            detector_config(CellId::AnnotationLama).mask_refine_mode,
            MaskRefineMode::Annotation
        );
        // Only that one field moves; everything else stays the shipped default.
        let default = TextDetectorConfig::default();
        let annotation = detector_config(CellId::Annotation);
        assert_eq!(annotation.model_path, default.model_path);
        assert_eq!(annotation.concurrent_models, default.concurrent_models);
        assert_eq!(annotation.intra_threads, default.intra_threads);
        assert_eq!(annotation.inter_threads, default.inter_threads);
    }

    /// The teeth behind the test above: it asserts a *field value*, which would still
    /// hold if `MaskRefineMode::Annotation` were a no-op. This runs the two configs the
    /// driver actually builds through `pc_detect::run` on the committed replay fixture
    /// and shows they take different branches — so the Annotation cell is a genuinely
    /// different run, not a relabelled Simple one.
    #[test]
    fn the_two_cells_configs_take_different_detector_branches_on_the_replay_fixture() {
        let fixture_dir = paths::recorded_root().join("detector");
        let page = fixture_dir.join(format!("{REPLAY_STEM}.jpg"));
        let detector = pc_detect::ReplayDetector::new(&fixture_dir, REPLAY_STEM);
        let job = PageJob {
            name: REPLAY_STEM.to_owned(),
            image: page,
            reference: None,
        };

        // Post-review fix (F1): calls `detect_page`, the SAME function
        // `measure_cell_inner` — the real production path — calls with `detector_config
        // (cell)`. Previously this test called `pc_detect::run` a second, independent
        // time, so a mutation at the production call site went undetected; routing
        // through `detect_page` closes that gap by construction.
        let raw_mask_for = |cell: CellId| {
            detect_page(&job, cell, &detector)
                .expect("the committed replay fixture detects")
                .page
                .raw_mask
                .load()
                .expect("raw mask")
                .to_luma8()
                .into_raw()
        };

        let simple = raw_mask_for(CellId::Simple);
        let annotation = raw_mask_for(CellId::Annotation);
        assert_eq!(
            simple.len(),
            annotation.len(),
            "the two modes produced different-sized masks; the comparison below is meaningless"
        );
        assert_ne!(
            simple, annotation,
            "the Simple and Annotation cells produced byte-identical raw masks, so the \
             cell's `mask_refine_mode` is not reaching `pc_detect::run`"
        );
    }

    // --- §16.43 item 9 / brief point 4: no scratch path to collide on ----------

    /// The scratch-collision risk named in §16.43's planning history is removed by
    /// construction rather than by naming paths carefully: mode-bench needs only the
    /// in-memory `MaskOutput`, so it asks `pc_mask::run` to write nothing at all. There
    /// is therefore no path any two cells could share. If a later task starts writing
    /// files, this test goes red and that task owes the cell-and-page-qualified path plus
    /// the collision test this one currently makes unnecessary.
    #[test]
    fn no_cell_asks_pc_mask_to_write_any_file_so_no_two_cells_can_share_an_output_path() {
        for cell in CellId::ALL {
            for page in ["p1", "p2"] {
                let dests = mask_dests(*cell, page);
                let named: [(&str, &Option<PathBuf>); 6] = [
                    ("combined_mask", &dests.combined_mask),
                    ("cleaned", &dests.cleaned),
                    ("text_layer", &dests.text_layer),
                    ("box_mask", &dests.box_mask),
                    ("cut_mask", &dests.cut_mask),
                    ("mask_overlay", &dests.mask_overlay),
                ];
                for (field, value) in named {
                    assert_eq!(
                        *value,
                        None,
                        "{}/{page} asks for a `{field}` output path",
                        cell.name()
                    );
                }
            }
        }
    }

    // --- §16.43 item 8: a per-page failure is per-(page, cell) ------------------

    /// Task 1 already has a *renderer* test for this shape against fabricated
    /// `Outcome::Failed` values. This one drives the real
    /// `pc_detect::run` → `pc_preprocess::run` → `pc_mask::run` sequence over two pages,
    /// one of which cannot be decoded, and asserts the **driver** produces that outcome:
    /// the bad page fails in every cell, the good page still measures in every cell, and
    /// the run does not abort.
    #[test]
    fn an_undecodable_page_fails_only_its_own_rows_and_leaves_the_other_page_measured() {
        let temp = tempfile::tempdir().expect("tempdir");
        let good = temp.path().join("good.png");
        image::GrayImage::from_fn(240, 1000, |x, y| {
            let inside = (20..200).contains(&x) && (20..400).contains(&y);
            image::Luma([if inside { 20 } else { 235 }])
        })
        .save(&good)
        .expect("writing the decodable page");
        let bad = temp.path().join("bad.png");
        std::fs::write(&bad, b"this file is not a PNG at all").expect("writing the bad page");

        let detector = pc_detect::MockDetector::new()
            .with_blocks(vec![pc_detect::RawBlock {
                rect: pc_core::Rect::new(20, 20, 200, 400),
                class_index: 0,
                confidence: 0.9,
            }])
            .with_block_fill(255);

        let cells = [CellId::Simple, CellId::SimpleLama];
        let jobs = vec![
            PageJob {
                name: "bad".to_owned(),
                image: bad,
                reference: None,
            },
            PageJob {
                name: "good".to_owned(),
                image: good,
                reference: None,
            },
        ];
        let mut benchmark = Benchmark::new(Source::Pages, &cells);
        // A working provider, so nothing here can be mistaken for an inpainting problem:
        // this test is about the mask-stage failure path only.
        let provider = WorkingProvider::new(Arc::new(StubInpainter::flat(200, 200, 200)));
        measure_jobs(
            &mut benchmark,
            &jobs,
            &cells,
            &detector,
            &mut runner_with(Source::Pages, Some(&provider)),
        );

        for cell in cells {
            match benchmark.outcome("bad", cell) {
                Some(Outcome::Failed { reason }) => assert!(
                    reason.contains("detector"),
                    "the failure does not name the stage that failed: {reason}"
                ),
                other => panic!("bad page under {}: {other:?}", cell.name()),
            }
            assert!(
                matches!(benchmark.outcome("good", cell), Some(Outcome::Measured(_))),
                "the good page did not measure under {}: {:?}",
                cell.name(),
                benchmark.outcome("good", cell)
            );
        }
        // Both pages are still in the report — identity, not a count.
        assert_eq!(
            benchmark.full_population(),
            BTreeSet::from(["bad".to_owned(), "good".to_owned()]),
        );
    }

    // --- §16.43 items 7 and 8: --demo-bubbles' reference-loading path -----------

    /// The 7 vendored crops and their sizes, transcribed from
    /// `tests/fixtures/upstream/ATTRIBUTION.md` via §7.1 — an oracle independent of the
    /// driver, which reads `pc_testkit::paths::DEMO_BUBBLES` itself.
    #[cfg(feature = "onnx")]
    const VENDORED_CROPS: [(&str, (u32, u32)); 7] = [
        ("black", (202, 319)),
        ("darkrays", (208, 320)),
        ("handwritten", (72, 132)),
        ("nightmare", (219, 343)),
        ("ray", (256, 329)),
        ("spikey", (354, 354)),
        ("square", (144, 270)),
    ];

    /// `--demo-bubbles` needs the real ONNX detector and a real model artifact, so this
    /// skips with an actionable message on a machine with neither, exactly as
    /// `xtask/src/bench.rs` does, rather than failing the whole `onnx` tier there.
    ///
    /// **No pass/fail assertion is made against `_clean.png`** (§15.2, and §16.43 item
    /// 7's third binding condition): this asserts that the reference was *found, decoded
    /// and compared*, never that the comparison met any threshold.
    #[cfg(feature = "onnx")]
    #[test]
    fn demo_bubbles_attaches_a_reference_report_to_every_vendored_crop() {
        let Some(model) = std::env::var_os("PANEL_OCR_ONNX_MODEL")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
        else {
            println!(
                "SKIPPED: no detector model; set PANEL_OCR_ONNX_MODEL=<comictextdetector.pt.onnx>"
            );
            return;
        };
        let spec = format!("onnx:{}", model.display());
        let cells = [CellId::Simple];
        let (benchmark, facts) =
            measure(Source::DemoBubbles, &cells, None, Some(&spec), Device::Cpu)
                .expect("--demo-bubbles is a complete run");
        // Post-review fix (F2): the real detector's facts must reach the disclosure row —
        // confirm this run actually observed them, not the `--replay` placeholder.
        assert!(
            facts.detector.model_path.is_some(),
            "no model path recorded for a real detector run"
        );
        assert!(
            facts.detector.digest_verified,
            "digest verification was not observed"
        );
        assert!(
            facts.detector.session_constructed,
            "session construction was not observed"
        );

        // Identity, not cardinality: all 7 crops by name.
        assert_eq!(
            benchmark.full_population(),
            VENDORED_CROPS
                .iter()
                .map(|(name, _)| (*name).to_owned())
                .collect::<BTreeSet<String>>(),
        );
        for (name, size) in VENDORED_CROPS {
            match benchmark.outcome(name, CellId::Simple) {
                Some(Outcome::Measured(measured)) => {
                    let report = measured
                        .reference
                        .as_ref()
                        .unwrap_or_else(|| panic!("`{name}` carries no reference report"));
                    assert_eq!(report.name, name);
                    // The compared images really are the vendored crop, at its
                    // independently transcribed size.
                    assert_eq!(report.dims, size, "`{name}` compared the wrong image");
                    // `compare_gray_with_shape` was used, not the shapeless `compare_gray`.
                    assert!(
                        report.shape_iou.is_some() && report.shape_subset_of_dilated.is_some(),
                        "`{name}` has no shape columns, so the source-vs-reference change \
                         sets were never computed"
                    );
                }
                other => panic!("`{name}` did not measure: {other:?}"),
            }
        }
    }

    // --- §16.44: the LaMa stage --------------------------------------------------

    /// Falsification control for every test below that uses [`synthetic_page`]. Those
    /// tests mean what their names say only if the "eligible" page really has an eligible
    /// region and the other really has none — otherwise "inpainting did not run" would be
    /// true for a reason none of them are about.
    #[test]
    fn the_two_synthetic_pages_really_do_differ_in_eligibility() {
        let temp = tempfile::tempdir().expect("tempdir");
        let detector = synthetic_detector();
        let cells = [CellId::SimpleLama];
        let jobs = vec![
            synthetic_page(temp.path(), "eligible", true),
            synthetic_page(temp.path(), "barren", false),
        ];
        let mut benchmark = Benchmark::new(Source::Replay, &cells);
        measure_jobs(
            &mut benchmark,
            &jobs,
            &cells,
            &detector,
            &mut no_inpaint_runner(Source::Replay),
        );
        // Hard-coded, from `select_regions`' first set: the noisy page's single region is
        // `failed`, so exactly one region is eligible; the flat page's fits, so none is.
        assert_eq!(eligible_of(&benchmark, "eligible", CellId::SimpleLama), 1);
        assert_eq!(eligible_of(&benchmark, "barren", CellId::SimpleLama), 0);
    }

    /// §16.44 item 2, the ruling's core: *"`--replay` never attempts LaMa acquisition for
    /// any cell, by mode-bench's own deliberate source policy — not because the model is
    /// unavailable."*
    ///
    /// The assertion is a **side-effect count on the provider**, not a status string: the
    /// page fed in is the genuinely-eligible one, so under any implementation that reaches
    /// `run_inpaint` the provider would be asked at least once. The second half is the
    /// falsification control — the identical page and provider on `Source::Pages` DO ask —
    /// without which a provider that was simply never reachable would pass the first half.
    #[test]
    fn replay_never_asks_the_provider_for_a_model_while_pages_with_the_same_input_does() {
        let temp = tempfile::tempdir().expect("tempdir");
        let detector = synthetic_detector();
        let cells = [CellId::SimpleLama];
        let jobs = vec![synthetic_page(temp.path(), "eligible", true)];

        let attempts_under = |source: Source| {
            let provider = FailingAcquisition {
                attempts: AtomicUsize::new(0),
            };
            let mut benchmark = Benchmark::new(source, &cells);
            measure_jobs(
                &mut benchmark,
                &jobs,
                &cells,
                &detector,
                &mut runner_with(source, Some(&provider)),
            );
            (
                provider.attempts.load(Ordering::SeqCst),
                status_of(&benchmark, "eligible", CellId::SimpleLama),
            )
        };

        let (replay_attempts, replay_status) = attempts_under(Source::Replay);
        assert_eq!(
            replay_attempts, 0,
            "`--replay` sought the inpainting model; §16.44 item 2 says it never does"
        );
        assert_eq!(replay_status, InpaintStatus::NotAttempted);

        let (pages_attempts, pages_status) = attempts_under(Source::Pages);
        assert_eq!(
            pages_attempts, 1,
            "`--pages` did not seek the model on a genuinely eligible page, so the \
             assertion above passes for the wrong reason"
        );
        assert!(matches!(pages_status, InpaintStatus::AcquisitionFailed(_)));
    }

    /// §16.44 item 2 again, from the report's side rather than the provider's: a
    /// `--replay` invocation's inpainter facts must say **not attempted**, on a source
    /// whose one page genuinely has an eligible region (the false premise that produced
    /// §16.44 was the belief that it has none — the hard-coded `1` here is the corrected
    /// fact, re-derived by `cargo xtask mask-sweep --replay`'s own `failed = 1` row).
    #[test]
    fn a_replay_invocation_reports_the_inpainter_as_not_attempted_over_an_eligible_page() {
        let cells = [CellId::Simple, CellId::SimpleLama];
        let (benchmark, facts) =
            measure(Source::Replay, &cells, None, None, Device::Cpu).expect("--replay runs");
        assert_eq!(eligible_of(&benchmark, REPLAY_STEM, CellId::SimpleLama), 1);
        assert_eq!(
            status_of(&benchmark, REPLAY_STEM, CellId::SimpleLama),
            InpaintStatus::NotAttempted
        );
        assert!(
            !facts.inpainter.attempted,
            "the disclosure claims acquisition was attempted on --replay"
        );
        assert!(!facts.inpainter.digest_verified);
        assert!(!facts.inpainter.session_constructed);
        assert_eq!(facts.inpainter.model_path, None);
        assert_eq!(
            inpainter_model_path_cell(&facts.inpainter),
            "(not attempted in this run)"
        );
    }

    /// The brief's first requested test, under §16.44 item 1's correction: a failed
    /// acquisition is a **whole-cell** condition — both pages carry it — but it **does not
    /// erase either page's mask-stage measurement**.
    ///
    /// Both halves matter, and each can fail on its own: a per-page retry design fails the
    /// `attempts == 1` assertion, and a §16.43-item-8-as-written design (render
    /// `**BLOCKED**`) fails the "still Measured, still carries its real numbers"
    /// assertions.
    #[test]
    fn a_failed_acquisition_marks_every_page_of_the_cell_once_and_erases_no_measurement() {
        let temp = tempfile::tempdir().expect("tempdir");
        let detector = synthetic_detector();
        let cells = [CellId::Simple, CellId::SimpleLama];
        let jobs = vec![
            synthetic_page(temp.path(), "page-a", true),
            synthetic_page(temp.path(), "page-b", true),
        ];
        let provider = FailingAcquisition {
            attempts: AtomicUsize::new(0),
        };
        let mut benchmark = Benchmark::new(Source::Pages, &cells);
        measure_jobs(
            &mut benchmark,
            &jobs,
            &cells,
            &detector,
            &mut runner_with(Source::Pages, Some(&provider)),
        );

        assert_eq!(
            provider.attempts.load(Ordering::SeqCst),
            1,
            "acquisition was retried per page instead of latching for the whole cell"
        );
        for page in ["page-a", "page-b"] {
            assert!(
                matches!(
                    status_of(&benchmark, page, CellId::SimpleLama),
                    InpaintStatus::AcquisitionFailed(_)
                ),
                "{page} does not carry the whole-cell acquisition failure"
            );
            // §16.44 item 1: the row keeps its own real mask-stage measurement.
            assert_eq!(eligible_of(&benchmark, page, CellId::SimpleLama), 1);
            // ...and the non-LaMa cell is untouched by the LaMa cell's failure.
            assert_eq!(
                status_of(&benchmark, page, CellId::Simple),
                InpaintStatus::NotRequested
            );
        }

        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        let table = section(&document, "## 3. Per-page, per-cell measurements");
        assert!(
            !table.contains("**BLOCKED**"),
            "a failed acquisition erased the cell's measurements: {table}"
        );
        assert_eq!(
            table.matches(INPAINT_ACQUISITION_FAILED).count(),
            2,
            "the acquisition failure is not stated on both of the cell's rows: {table}"
        );
    }

    /// The brief's second requested test: a per-page **inference** failure blocks only that
    /// page. The inpainter refuses its first tile and serves every later one, so page-a
    /// fails and page-b genuinely inpaints — under one acquisition.
    #[test]
    fn a_per_page_inference_failure_leaves_the_cells_other_page_genuinely_inpainted() {
        let temp = tempfile::tempdir().expect("tempdir");
        let detector = synthetic_detector();
        let cells = [CellId::SimpleLama];
        let jobs = vec![
            synthetic_page(temp.path(), "page-a", true),
            synthetic_page(temp.path(), "page-b", true),
        ];
        let provider = WorkingProvider::new(Arc::new(FailFirstTile {
            calls: AtomicUsize::new(0),
            inner: StubInpainter::new(StubMode::Flat(image::Rgb([200, 30, 30]))),
        }));
        let mut benchmark = Benchmark::new(Source::Pages, &cells);
        measure_jobs(
            &mut benchmark,
            &jobs,
            &cells,
            &detector,
            &mut runner_with(Source::Pages, Some(&provider)),
        );

        assert!(
            matches!(
                status_of(&benchmark, "page-a", CellId::SimpleLama),
                InpaintStatus::InferenceFailed(_)
            ),
            "page-a: {:?}",
            status_of(&benchmark, "page-a", CellId::SimpleLama)
        );
        match status_of(&benchmark, "page-b", CellId::SimpleLama) {
            InpaintStatus::Ran { tiles_inferred } => assert!(
                tiles_inferred > 0,
                "page-b reports Ran with zero tiles inferred"
            ),
            other => panic!("page-b did not inpaint: {other:?}"),
        }
        // Both rows keep their mask-stage numbers either way (§16.44 item 1).
        assert_eq!(eligible_of(&benchmark, "page-a", CellId::SimpleLama), 1);
        assert_eq!(eligible_of(&benchmark, "page-b", CellId::SimpleLama), 1);
        // The provider is consulted once per page — cheap, because the real provider's
        // `OnceLock` latch (§16.38 item 8(d)) makes every call after the first free. What
        // this number pins is that page-a's failure did **not** latch the cell out: page-b
        // still reached the provider, which is exactly what an *acquisition* failure
        // prevents (contrast the `1` asserted in the acquisition test above).
        assert_eq!(
            provider.attempts.load(Ordering::SeqCst),
            2,
            "an inference failure on page-a latched the whole cell out"
        );
    }

    // --- The reference comparison measures the cell's OWN final output -----------
    //
    // Found by running the real benchmark: every LaMa row in section 3 carried a
    // `Reference agreement` cell byte-identical to its non-LaMa sibling's on all 6 pages
    // where inpainting actually ran, because `compare_against_reference` was handed
    // `mask_output` unconditionally — the mask stage's image, computed before inpainting.
    // The two tests below are the pair that distinguishes "identical because inpainting
    // changed nothing here" (correct) from "identical because the inpainted image was never
    // looked at" (the bug).

    /// The whole-page reference these two tests compare against: a constant **white** page
    /// at [`synthetic_page`]'s dimensions.
    ///
    /// White is chosen so the expectation below is an argument, not a measurement. The stub
    /// inpainter also writes pure white, and `to_luma8` maps `Rgb([255,255,255])` to `255`
    /// for any weighting whose coefficients sum to one — so every pixel the inpainter wrote
    /// *must* agree exactly with this reference, while the same pixels in the non-inpainted
    /// sibling carry page content (luma 20 inside the detected block). The direction of both
    /// inequalities below therefore follows from the construction, and is not read off the
    /// artifact under test.
    fn white_reference(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(format!("{name}_reference.png"));
        image::GrayImage::from_pixel(240, 1000, image::Luma([255]))
            .save(&path)
            .expect("writing the white reference");
        path
    }

    fn reference_of(benchmark: &Benchmark, page: &str, cell: CellId) -> GoldenReport {
        match benchmark.outcome(page, cell) {
            Some(Outcome::Measured(measured)) => measured
                .reference
                .clone()
                .unwrap_or_else(|| panic!("{page}/{} carries no reference", cell.name())),
            other => panic!("{page}/{} is not a Measured row: {other:?}", cell.name()),
        }
    }

    /// Run `simple` and `simple+lama` over one synthetic page carrying a white reference,
    /// with a stub inpainter that paints pure white.
    fn measure_against_white_reference(
        temp: &Path,
        page: &str,
        eligible: bool,
    ) -> (Benchmark, Arc<StubInpainter>) {
        let mut job = synthetic_page(temp, page, eligible);
        job.reference = Some(white_reference(temp, page));
        let cells = [CellId::Simple, CellId::SimpleLama];
        let inpainter = Arc::new(StubInpainter::new(StubMode::Flat(image::Rgb([
            255, 255, 255,
        ]))));
        let provider =
            WorkingProvider::new(Arc::clone(&inpainter) as Arc<dyn pc_inpaint::Inpainter>);
        let mut benchmark = Benchmark::new(Source::DemoBubbles, &cells);
        measure_jobs(
            &mut benchmark,
            &[job],
            &cells,
            &synthetic_detector(),
            &mut runner_with(Source::DemoBubbles, Some(&provider)),
        );
        (benchmark, inpainter)
    }

    /// The LaMa cell's reference numbers are taken from the **inpainted** page, so on a page
    /// where inpainting genuinely ran they agree with the white reference strictly better
    /// than its non-LaMa sibling's do.
    ///
    /// Verified capable of failing: reverting the fix — passing `&mask_output.cleaned` for
    /// both cells — makes the two reports identical (`exact_fraction` 0.0027791666… for each)
    /// and the `exact_fraction` assertion below fails. The status and `dims` assertions above
    /// it are guards and stay green either way.
    #[test]
    fn a_lama_cell_that_inpainted_scores_strictly_closer_to_the_white_reference_than_its_sibling() {
        let temp = tempfile::tempdir().expect("tempdir");
        let (benchmark, inpainter) = measure_against_white_reference(temp.path(), "eligible", true);

        // Guards first: without these, "the numbers differ" could mean anything.
        match status_of(&benchmark, "eligible", CellId::SimpleLama) {
            InpaintStatus::Ran { tiles_inferred } => assert!(
                tiles_inferred >= 1,
                "the LaMa cell reports Ran with zero tiles inferred"
            ),
            other => {
                panic!("the LaMa cell did not inpaint, so this test proves nothing: {other:?}")
            }
        }
        assert_eq!(
            status_of(&benchmark, "eligible", CellId::Simple),
            InpaintStatus::NotRequested
        );
        assert!(
            inpainter.calls() >= 1,
            "the stub inpainter was never asked for a tile"
        );

        let simple = reference_of(&benchmark, "eligible", CellId::Simple);
        let lama = reference_of(&benchmark, "eligible", CellId::SimpleLama);
        // Both cells measured the same page at the same size, so nothing below is a
        // dimension artifact. The literal is `synthetic_page`'s own hard-coded shape.
        assert_eq!(simple.dims, (240, 1000));
        assert_eq!(lama.dims, (240, 1000));

        // The detected block is 20..200 x 20..400 = 68_400 px of a 240x1000 = 240_000 px
        // page, i.e. 28.5% of it. The inpaint mask is that region's mask (grown), and every
        // pixel of it is written pure white, which matches this reference exactly while the
        // sibling's same pixels hold the block's luma-20 content. Even if only a fifth of
        // the block is written, the exactly-equal fraction must rise by more than 0.05.
        // This literal is derived from the fixture's geometry, never from a measured run.
        assert!(
            lama.exact_fraction - simple.exact_fraction > 0.05,
            "the LaMa cell's agreement with the reference barely moved, so its numbers are \
             not being taken from the inpainted page: simple {:?}, lama {:?}",
            simple.exact_fraction,
            lama.exact_fraction
        );
        assert!(
            lama.mean_abs_diff < simple.mean_abs_diff,
            "painting white over a white reference did not lower the mean abs diff: \
             simple {:?}, lama {:?}",
            simple.mean_abs_diff,
            lama.mean_abs_diff
        );
    }

    /// The control for the test above: when nothing is eligible the LaMa cell's final output
    /// **is** the mask stage's output, so identical reference numbers are correct there.
    ///
    /// This is what keeps the fix from being "make the LaMa row differ": a change that
    /// perturbed the LaMa cell unconditionally would pass the test above and fail this one.
    #[test]
    fn a_lama_cell_with_nothing_eligible_reports_the_same_reference_numbers_as_its_sibling() {
        let temp = tempfile::tempdir().expect("tempdir");
        let (benchmark, inpainter) =
            measure_against_white_reference(temp.path(), "ineligible", false);

        assert_eq!(
            status_of(&benchmark, "ineligible", CellId::SimpleLama),
            InpaintStatus::NothingEligible
        );
        assert_eq!(
            inpainter.calls(),
            0,
            "nothing was eligible, yet a tile was inferred"
        );

        let simple = reference_of(&benchmark, "ineligible", CellId::Simple);
        let lama = reference_of(&benchmark, "ineligible", CellId::SimpleLama);
        assert_eq!(
            simple, lama,
            "no inpainting happened, so both cells' final output is the same image"
        );
        // Anti-vacuity: the two reports are equal because both measured a real page against
        // the white reference, not because both are empty. A masked page over a white
        // reference cannot be an exact match — the block's content is not white.
        assert!(
            simple.mean_abs_diff > 0.0 && simple.exact_fraction < 1.0,
            "both cells reported a perfect match, so the equality above is vacuous: {simple:?}"
        );
    }

    /// §16.44 item 3, binding graft from the losing position: the three "inpainting did not
    /// run" states must be **textually distinct** in the report — Fable: *"don't collapse
    /// all three into one bare 'no'."*
    ///
    /// Pairwise distinctness is asserted directly, so two notes that were accidentally made
    /// equal cannot pass. The last assertion is item 3's other half: the `--replay` note
    /// must **agree with** `Source::Replay.describe()`'s shipped "no model is loaded".
    #[test]
    fn the_three_reasons_inpainting_did_not_run_render_as_three_distinct_sentences() {
        let mut benchmark = Benchmark::new(Source::Pages, &[CellId::SimpleLama]);
        benchmark.record(
            "not-attempted",
            CellId::SimpleLama,
            measured_with(1, 1, InpaintStatus::NotAttempted),
        );
        benchmark.record(
            "nothing-eligible",
            CellId::SimpleLama,
            measured_with(0, 0, InpaintStatus::NothingEligible),
        );
        benchmark.record(
            "acquisition-failed",
            CellId::SimpleLama,
            measured_with(
                1,
                1,
                InpaintStatus::AcquisitionFailed("lama-manga.onnx is missing".to_owned()),
            ),
        );
        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        let table = section(&document, "## 3. Per-page, per-cell measurements");

        let row = |page: &str| {
            table
                .lines()
                .find(|line| line.starts_with(&format!("| {page} | simple+lama |")))
                .unwrap_or_else(|| panic!("no row for {page}: {table}"))
                .to_owned()
        };
        let not_attempted = row("not-attempted");
        let nothing_eligible = row("nothing-eligible");
        let acquisition_failed = row("acquisition-failed");

        assert!(
            not_attempted.contains(INPAINT_NOT_ATTEMPTED),
            "{not_attempted}"
        );
        assert!(
            nothing_eligible.contains(INPAINT_NOTHING_ELIGIBLE),
            "{nothing_eligible}"
        );
        assert!(
            acquisition_failed.contains("lama-manga.onnx is missing")
                && acquisition_failed.contains(INPAINT_ACQUISITION_FAILED),
            "{acquisition_failed}"
        );

        // Pairwise distinct — the property item 3 actually asks for. Each note is also
        // checked NOT to appear on the other two rows, so a renderer that concatenated all
        // three onto every row fails.
        let notes = [
            INPAINT_NOT_ATTEMPTED,
            INPAINT_NOTHING_ELIGIBLE,
            INPAINT_ACQUISITION_FAILED,
        ];
        for (index, note) in notes.iter().enumerate() {
            assert_eq!(
                table.matches(note).count(),
                1,
                "note {index} appears on more than one row: {table}"
            );
        }

        // §16.44 item 3's other half: the per-row note agrees with the shipped source
        // description instead of contradicting it.
        assert!(Source::Replay.describe().contains("no model is loaded"));
        assert!(INPAINT_NOT_ATTEMPTED.contains("no model is loaded"));
    }

    /// §16.44 item 4, the hidden hazard: a LaMa cell with `Measured` rows but **zero**
    /// genuine inpaint executions must not contribute to segment 4.3's mean, while a
    /// non-LaMa cell in the identical shape must.
    ///
    /// "Identical shape" is what gives this teeth: `simple` and `simple+lama` here carry
    /// the same numbers and the same `Measured` status, and differ only in whether their
    /// cell has an inpainting factor that ran. A predicate that keyed on `Measured` alone —
    /// §16.44 item 1's own wording, which item 4 corrects — includes both and fails here.
    #[test]
    fn a_lama_cell_that_inpainted_nowhere_is_excluded_from_the_eligibility_restricted_mean() {
        let cells = [CellId::Simple, CellId::SimpleLama, CellId::AnnotationLama];
        let mut benchmark = Benchmark::new(Source::Pages, &cells);
        for page in ["p1", "p2"] {
            benchmark.record(
                page,
                CellId::Simple,
                measured_with(1, 2, InpaintStatus::NotRequested),
            );
            benchmark.record(
                page,
                CellId::SimpleLama,
                measured_with(1, 2, InpaintStatus::NothingEligible),
            );
        }
        // The one cell that genuinely inpainted, and only on one of its two pages.
        benchmark.record(
            "p1",
            CellId::AnnotationLama,
            measured_with(1, 2, InpaintStatus::Ran { tiles_inferred: 3 }),
        );
        benchmark.record(
            "p2",
            CellId::AnnotationLama,
            measured_with(1, 2, InpaintStatus::NothingEligible),
        );

        assert!(
            benchmark.contributes_to_eligibility_restricted_mean(CellId::Simple),
            "a non-LaMa cell with Measured rows must still contribute (§16.44 item 4's \
             explicit carve-out)"
        );
        assert!(
            !benchmark.contributes_to_eligibility_restricted_mean(CellId::SimpleLama),
            "a LaMa cell that inpainted on zero pages contributed to the mean"
        );
        assert!(
            benchmark.contributes_to_eligibility_restricted_mean(CellId::AnnotationLama),
            "one genuine inpaint on one page is enough to contribute"
        );

        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        let segments = section(&document, "## 4. Eligibility segments");
        let intersection = &segments[segments.find("### 4.3").expect("4.3 subsection")..];
        // Exclusion, not disclosure-instead-of-exclusion: the excluded cell prints no
        // number at all in this segment.
        assert!(
            intersection.contains("| simple+lama | 2 | excluded | excluded | excluded |"),
            "the excluded LaMa cell still printed a mean: {intersection}"
        );
        // The two cells that DO contribute print real means — hand-computed: every row
        // above has failed_regions = 1 and eligible_regions = 2.
        assert!(
            intersection.contains("| simple | 2 | 1.000000 | 2.000000 | — |"),
            "the non-LaMa cell was excluded too: {intersection}"
        );
        assert!(
            intersection.contains("| annotation+lama | 2 | 1.000000 | 2.000000 | — |"),
            "the genuinely-inpainting cell was excluded: {intersection}"
        );
        assert!(intersection.contains(EXCLUDED_FROM_MEAN_HEAD));
        assert!(intersection.contains("simple+lama"));
    }

    /// §16.44 item 1's other half, at the `Benchmark` level: once a page has a real
    /// measurement, a cell-level block may not erase it. The pre-§16.44 `outcome` returned
    /// `Blocked` unconditionally and fails this.
    #[test]
    fn a_recorded_measurement_wins_over_a_cell_level_block() {
        let mut benchmark = Benchmark::new(Source::Pages, &[CellId::SimpleLama]);
        benchmark.record("p1", CellId::SimpleLama, measured(2, 3));
        benchmark.register_page("p2");
        benchmark.block_cell(CellId::SimpleLama, "the LaMa model is not available");

        assert!(
            matches!(
                benchmark.outcome("p1", CellId::SimpleLama),
                Some(Outcome::Measured(_))
            ),
            "a cell-level block erased a real measurement (§16.44 item 1)"
        );
        // The page that never measured still renders BLOCKED, so §16.43 item 8's rule
        // survives for a condition that really did precede measurement.
        assert!(matches!(
            benchmark.outcome("p2", CellId::SimpleLama),
            Some(Outcome::Blocked { .. })
        ));
    }

    /// A real LaMa session, real tiles, real `tiles_inferred` — the only test here that
    /// proves the wiring reaches ONNX at all. Skips with an actionable message when the
    /// optional 207 MB artifact is absent, exactly as the detector tests do.
    ///
    /// It drives `measure_jobs` with `pc_cli::inpainter::build_provider`'s own provider —
    /// the production acquisition path (§16.43 item 5/D3) — over a synthetic page whose
    /// eligibility is independently asserted by
    /// `the_two_synthetic_pages_really_do_differ_in_eligibility`.
    #[cfg(feature = "onnx")]
    #[test]
    fn a_real_lama_session_inpaints_an_eligible_page_and_reports_the_tiles_it_inferred() {
        let cache_root = pc_cli::paths::default_cache_dir();
        let model =
            pc_cli::paths::models_dir(&cache_root).join(pc_models::LAMA_MANGA_INPAINTER.file_name);
        if !model.is_file() {
            println!(
                "SKIPPED: the optional LaMa model is absent at {}; run \
                 `panel-ocr models download --include-optional`",
                model.display()
            );
            return;
        }

        let temp = tempfile::tempdir().expect("tempdir");
        let detector = synthetic_detector();
        let cells = [CellId::SimpleLama];
        let jobs = vec![synthetic_page(temp.path(), "eligible", true)];
        let provider = build_provider(true, None, &cache_root, Device::Cpu)
            .expect("build_provider returns a provider when inpainting is enabled");
        let mut runner = runner_with(Source::Pages, Some(provider.as_ref()));
        // The same path `measure` records: where the provider was pointed.
        runner.model_path = Some(model.display().to_string());
        let mut benchmark = Benchmark::new(Source::Pages, &cells);
        measure_jobs(&mut benchmark, &jobs, &cells, &detector, &mut runner);

        match status_of(&benchmark, "eligible", CellId::SimpleLama) {
            InpaintStatus::Ran { tiles_inferred } => assert!(
                tiles_inferred >= 1,
                "a real session ran but inferred no tiles"
            ),
            other => panic!("real LaMa did not run: {other:?}"),
        }
        let facts = runner.facts();
        assert!(facts.attempted && facts.digest_verified && facts.session_constructed);
        assert_eq!(
            facts.model_path.as_deref(),
            Some(
                pc_cli::paths::models_dir(&cache_root)
                    .join(pc_models::LAMA_MANGA_INPAINTER.file_name)
                    .display()
                    .to_string()
                    .as_str()
            ),
        );
    }

    #[test]
    fn every_requested_cell_gets_a_row_for_every_page() {
        let benchmark = two_cell_benchmark();
        let document = render_document(
            &benchmark,
            &cpu_policy(),
            &stage_disclosures(benchmark.cells()),
        );
        let table = section(&document, "## 3. Per-page, per-cell measurements");
        let found: BTreeSet<String> = ["p1", "p2"]
            .into_iter()
            .flat_map(|page| {
                ["simple", "annotation"]
                    .into_iter()
                    .map(move |cell| (page, cell))
            })
            .filter(|(page, cell)| table.contains(&format!("| {page} | {cell} |")))
            .map(|(page, cell)| format!("{page}/{cell}"))
            .collect();
        // Hard-coded expected set: 2 pages x 2 cells, named individually.
        assert_eq!(
            found,
            BTreeSet::from([
                "p1/simple".to_owned(),
                "p1/annotation".to_owned(),
                "p2/simple".to_owned(),
                "p2/annotation".to_owned(),
            ])
        );
    }
}
