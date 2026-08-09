//! `cargo xtask mode-bench` — the Simple/Annotation/LaMa non-gating comparison
//! benchmark (spec §16.43).
//!
//! **This module is the pure layer only** (§16.43 item 9's "simple" task): cell/segment
//! types, CLI argument parsing, device-disclosure rendering and the report renderer.
//! It performs no stage calls and no measurement I/O. The Simple/Annotation measurement
//! driver and the LaMa integration are the two ratified **heavy** tasks that follow, and
//! [`measure`] below is a deliberate placeholder until the first of them lands.

use crate::paths;
use anyhow::{Context, Result};
use pc_core::device::{resolve, Device, DevicePolicy, DeviceSupport};
use pc_testkit::golden::GoldenReport;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// D1's ruling (§16.43 item 3): the default cell set varies one factor at a time.
pub(crate) const DEFAULT_CELLS_SPEC: &str = "simple,annotation,simple+lama";

/// The reason [`measure`] records for every cell until the measurement driver lands.
const MEASUREMENT_DRIVER_UNIMPLEMENTED: &str =
    "the Simple/Annotation measurement driver is not implemented yet (§16.43 item 9, \
     heavy task 2); this run rendered the report skeleton only";

/// §16.43 item 6's graft, quoted: the detector's row must state this.
const DETECTOR_DEVICE_MECHANISM: &str =
    "its session constructor takes no device argument at all — device reaches the \
     detector path only as an up-front refusal, never as a registration";

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
    pub(crate) inpainting_ran: bool,
    /// `None` when this cell did not inpaint at all.
    pub(crate) tiles_inferred: Option<usize>,
    /// `None` for sources with no reference (`--replay`, `--pages`).
    pub(crate) reference: Option<GoldenReport>,
}

/// What happened for one `(page, cell)` pair.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Outcome {
    /// Boxed so the enum is not dominated by the measurement variant.
    // Constructed by the measurement driver (§16.43 item 9, heavy task 2).
    #[allow(dead_code)]
    Measured(Box<Measured>),
    /// A whole cell was unavailable — e.g. the LaMa model is missing (§16.43 item 8).
    Blocked { reason: String },
    /// One page failed inside an otherwise-running cell.
    // Constructed by the measurement driver (§16.43 item 9, heavy task 2).
    #[allow(dead_code)]
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
    // Called by the measurement driver (§16.43 item 9, heavy task 2).
    #[allow(dead_code)]
    pub(crate) fn record(&mut self, page: &str, cell: CellId, outcome: Outcome) {
        if !self.pages.iter().any(|known| known == page) {
            self.pages.push(page.to_owned());
        }
        self.rows.insert((page.to_owned(), cell), outcome);
    }

    /// Register a page even when no cell measured it, so a blocked cell still gets rows.
    // Called by the measurement driver (§16.43 item 9, heavy task 2).
    #[allow(dead_code)]
    pub(crate) fn register_page(&mut self, page: &str) {
        if !self.pages.iter().any(|known| known == page) {
            self.pages.push(page.to_owned());
        }
    }

    /// Mark a whole cell unavailable. Every other cell in the run still reports
    /// (§16.43 item 8).
    pub(crate) fn block_cell(&mut self, cell: CellId, reason: impl Into<String>) {
        self.blocked_cells.insert(cell, reason.into());
    }

    pub(crate) fn outcome(&self, page: &str, cell: CellId) -> Option<Outcome> {
        if let Some(reason) = self.blocked_cells.get(&cell) {
            return Some(Outcome::Blocked {
                reason: reason.clone(),
            });
        }
        self.rows.get(&(page.to_owned(), cell)).cloned()
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

    /// Segment 3: the intersection of the eligible sets over the cells actually run.
    ///
    /// **Design decision (post-review, §16.43 item 4): "actually run" excludes
    /// BLOCKED cells, not just unrequested ones.** A requested-but-blocked cell (e.g.
    /// the LaMa cell when the model artifact is absent) never produced a real
    /// measurement, so it contributes no genuine eligible set to intersect against —
    /// treating its absence as an empty set would silently collapse this segment to
    /// empty on the single most common real-world run shape (LaMa unavailable in CI),
    /// which is exactly the "silent re-segmentation" failure item 4's caption
    /// requirement exists to surface, not hide behind an always-empty table. A run
    /// requesting only blocked cells (nothing actually ran) still correctly yields an
    /// empty intersection via `contributing.is_empty()` below, not through blocked
    /// cells poisoning an otherwise-real intersection.
    pub(crate) fn common_eligible(&self) -> BTreeSet<String> {
        let mut contributing = self
            .cells
            .iter()
            .copied()
            .filter(|cell| !self.blocked_cells.contains_key(cell));
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

    /// The cells this intersection was actually computed over — i.e. requested minus
    /// blocked — for the segment-3 caption to name (item 4's "the caption must name
    /// the cell set the intersection was computed over").
    pub(crate) fn common_eligible_contributing_cells(&self) -> Vec<CellId> {
        self.cells
            .iter()
            .copied()
            .filter(|cell| !self.blocked_cells.contains_key(cell))
            .collect()
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
fn stage_disclosures(cells: &[CellId]) -> Vec<StageDisclosure> {
    let mut stages = vec![detector_disclosure(None, false, false)];
    if cells.iter().any(|cell| cell.inpainting()) {
        stages.push(inpainter_disclosure(None, false, false));
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
            "{head} {} | {} | {} | {} | {} | {} | {} | {} | {} | measured |",
            measured.detected_boxes,
            measured.masking_regions,
            measured.succeeded_regions,
            measured.failed_regions,
            measured.dropped_regions,
            measured.eligible_regions,
            yes_no(measured.inpainting_ran),
            measured
                .tiles_inferred
                .map_or_else(|| "—".to_owned(), |tiles| tiles.to_string()),
            measured
                .reference
                .as_ref()
                .map_or_else(|| "—".to_owned(), render_agreement),
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
    body.push_str(&segment_table(benchmark, |_| benchmark.full_population()));

    // Segment 2 — each cell's own eligible set. Explicitly NOT comparable across cells.
    body.push_str("### 4.2 Per-cell eligible subsets\n\n");
    let _ = writeln!(
        body,
        "Each row is that cell's own `{{page : eligible_regions > 0}}`. These subsets are \
         **{NOT_CROSS_COMPARABLE}**: the eligibility predicate is mode-dependent, so two cells' \
         rows here describe different populations and averaging across them would compare \
         inpainted output against bare masking output under a caption claiming comparability.\n"
    );
    body.push_str(&segment_table(benchmark, |cell| {
        benchmark.eligible_pages(cell)
    }));

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
        body.push_str(&segment_table(benchmark, |_| common.clone()));
    }
    body
}

fn segment_table(benchmark: &Benchmark, pages_for: impl Fn(CellId) -> BTreeSet<String>) -> String {
    let mut body = String::from("| Cell | Pages in segment |");
    for metric in Metric::ALL {
        let _ = write!(body, " {} |", metric.label());
    }
    body.push_str("\n|---|---:|---:|---:|---:|\n");
    for cell in benchmark.cells() {
        let pages = pages_for(*cell);
        let _ = write!(body, "| {} | {} |", cell.name(), pages.len());
        for metric in Metric::ALL {
            let _ = write!(
                body,
                " {} |",
                benchmark
                    .mean(*cell, &pages, *metric)
                    .map_or_else(|| "—".to_owned(), |value| format!("{value:.6}"))
            );
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
    #[arg(long, conflicts_with_all = ["replay", "pages"])]
    demo_bubbles: bool,
    /// Directory of maintainer-local full pages (no reference; the only multi-tile source).
    #[arg(
        long,
        value_name = "DIR",
        requires = "detector",
        conflicts_with_all = ["replay", "demo_bubbles"]
    )]
    pages: Option<PathBuf>,
    /// Real detector specification in the form `onnx:<path>` (requires --pages).
    #[arg(long, value_name = "SPEC", requires = "pages")]
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

    let benchmark = measure(
        source,
        &args.cells.0,
        args.pages.as_deref(),
        args.detector.as_deref(),
    )?;
    let stages = stage_disclosures(benchmark.cells());
    let document = render_document(&benchmark, &policy, &stages);

    std::fs::write(&target, &document).with_context(|| format!("writing {}", target.display()))?;
    println!(
        "wrote {} ({} bytes)",
        paths::display_relative(&target),
        document.len()
    );
    Ok(())
}

/// **Placeholder for the two heavy tasks.**
///
/// §16.43 item 9 ratifies the Simple/Annotation measurement driver and the LaMa
/// integration as two separate implementation calls *after* this one. Until the first of
/// them lands there is no measurement to report, so every requested cell is recorded as
/// BLOCKED with a reason that says exactly that — rather than inventing numbers or
/// panicking somewhere the report renderer cannot be exercised.
fn measure(
    source: Source,
    cells: &[CellId],
    _pages: Option<&Path>,
    _detector: Option<&str>,
) -> Result<Benchmark> {
    let mut benchmark = Benchmark::new(source, cells);
    for cell in cells {
        benchmark.block_cell(*cell, MEASUREMENT_DRIVER_UNIMPLEMENTED);
    }
    Ok(benchmark)
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
            inpainting_ran: false,
            tiles_inferred: None,
            reference: None,
        }))
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
            device.contains(
                "its session constructor takes no device argument at all — device reaches the \
                 detector path only as an up-front refusal, never as a registration"
            ),
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
