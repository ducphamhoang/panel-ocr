//! `cargo xtask mode-bench` — the Simple/Annotation/LaMa non-gating comparison
//! benchmark (spec §16.43).
//!
//! The pure layer (cell/segment types, CLI argument parsing, device-disclosure rendering
//! and the report renderer) came from §16.43 item 9's "simple" task. [`measure`] and the
//! per-`(page, cell)` driver below are item 9's **first heavy** task: the three input
//! sources and the `pc_detect::run` → `pc_preprocess::run` → `pc_mask::run` sequence per
//! cell. The **LaMa integration is still the second heavy task** — nothing here calls an
//! inpainting function or depends on `pc-inpaint`, and every cell is therefore recorded
//! with `inpainting_ran: false` / `tiles_inferred: None`, which is what actually happened.

use crate::paths;
use anyhow::{anyhow, bail, Context, Result};
use pc_config::{
    InpainterConfig, MaskRefineMode, MaskerConfig, PreprocessorConfig, TextDetectorConfig,
};
use pc_core::device::{resolve, Device, DevicePolicy, DeviceSupport};
use pc_core::{ImageHandle, MaskRegionStats};
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
    // Exercised by this module's tests; the *production* caller is the LaMa integration
    // (§16.43 item 9's second heavy task), which blocks a LaMa cell whose model is
    // absent. The Simple/Annotation driver blocks no cell — a LaMa cell's masking-stage
    // numbers are real measurements here, only `inpainting_ran` is false.
    #[allow(dead_code)]
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
///
/// **Placeholder detector facts** (`None, false, false`) — used by the pure-layer tests,
/// which never run a real detector. [`run`] uses [`stage_disclosures_with_detector_facts`]
/// instead, which carries what [`measure`] actually observed. Test-only: since the fix
/// for F2, no non-test code needs the placeholder.
#[cfg(test)]
fn stage_disclosures(cells: &[CellId]) -> Vec<StageDisclosure> {
    stage_disclosures_with_detector_facts(cells, DetectorFacts::default())
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

fn stage_disclosures_with_detector_facts(
    cells: &[CellId],
    detector: DetectorFacts,
) -> Vec<StageDisclosure> {
    let mut stages = vec![detector_disclosure(
        detector.model_path.as_deref(),
        detector.digest_verified,
        detector.session_constructed,
    )];
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

    let (benchmark, detector_facts) = measure(
        source,
        &args.cells.0,
        args.pages.as_deref(),
        args.detector.as_deref(),
    )?;
    let stages = stage_disclosures_with_detector_facts(benchmark.cells(), detector_facts);
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

/// **Mirror of `pc_inpaint::eligible::select_regions`**
/// (`crates/pc-inpaint/src/eligible.rs:64-97`), duplicated deliberately.
///
/// §16.43 item 8 requires `eligible_regions` for **every** cell, LaMa or not, since that
/// function reads only `MaskRegionStats`. §16.43 item 9 reserves the `xtask/Cargo.toml`
/// edit — and therefore any new dependency, including `pc-inpaint` — for the LaMa task.
/// So this counts the same two-set union over types xtask already depends on.
///
/// The predicate, quoted verbatim from `select_regions`' two loops:
///
/// ```text
/// if region.failed { out.push(...) }                       // set 1: failed
///
/// let poorly_fitted = !region.failed
///     && region.std_deviation >= config.inpainting_min_std_dev
///     && region
///         .thickness
///         .is_some_and(|thickness| thickness <= config.min_inpainting_radius);
/// ```
///
/// Both comparisons are inclusive, matching upstream's `>=` and `<=`; `thickness.is_some()`
/// carries upstream's own comment *"For box masks, this is none. We don't need to inpaint
/// those, they are always good."* The two sets are disjoint by construction (`failed` vs
/// `!failed`) and `select_regions` pushes one entry per member, so this count is exactly
/// `select_regions(regions, config).len()`.
///
/// **Accepted risk, named rather than solved:** this is a second independent copy of one
/// predicate, and no cross-crate equivalence test can be written without the dependency
/// this task must not add. `eligibility_mirror_*` below is a hand-derived-oracle drift
/// guard local to this copy only.
fn eligible_region_count(regions: &[MaskRegionStats], config: &InpainterConfig) -> usize {
    regions
        .iter()
        .filter(|region| {
            region.failed
                || (region.std_deviation >= config.inpainting_min_std_dev
                    && region
                        .thickness
                        .is_some_and(|thickness| thickness <= config.min_inpainting_radius))
        })
        .count()
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
fn measure_cell(job: &PageJob, cell: CellId, detector: &dyn pc_detect::TextDetector) -> Outcome {
    match measure_cell_inner(job, cell, detector) {
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
) -> Result<Measured> {
    let detect_output = detect_page(job, cell, detector)?;
    let detected_boxes = detect_output.analytics.blocks_detected;

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
    let eligible_regions = eligible_region_count(regions, &InpainterConfig::default());

    let reference = match &job.reference {
        Some(clean_path) => Some(compare_against_reference(
            &job.name,
            &job.image,
            clean_path,
            &mask_output,
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
        // §16.43 item 9: the LaMa integration is the *second* heavy task. Nothing in this
        // module calls an inpainting function, so "did not inpaint" is the measured fact.
        inpainting_ran: false,
        tiles_inferred: None,
        reference,
    })
}

fn compare_against_reference(
    name: &str,
    raw_path: &Path,
    clean_path: &Path,
    mask_output: &pc_mask::MaskOutput,
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
    let cleaned_luma = mask_output
        .cleaned
        .load()
        .map(|image| image.to_luma8())
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| "loading the cleaned output")?;
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
) {
    for job in jobs {
        benchmark.register_page(&job.name);
        for cell in cells {
            benchmark.record(&job.name, *cell, measure_cell(job, *cell, detector));
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
) -> Result<(Benchmark, DetectorFacts)> {
    let mut benchmark = Benchmark::new(source, cells);
    let facts = match source {
        Source::Replay => {
            let fixture_dir = paths::recorded_root().join("detector");
            let jobs = vec![PageJob {
                name: REPLAY_STEM.to_owned(),
                image: fixture_dir.join(format!("{REPLAY_STEM}.jpg")),
                reference: None,
            }];
            let detector = pc_detect::ReplayDetector::new(&fixture_dir, REPLAY_STEM);
            measure_jobs(&mut benchmark, &jobs, cells, &detector);
            // No real model is ever resolved for `--replay` — `ReplayDetector` replays a
            // committed fixture, so the placeholder facts are the honest ones here, not a
            // stand-in for something unimplemented.
            DetectorFacts::default()
        }
        Source::DemoBubbles => {
            let spec = detector.ok_or_else(|| {
                anyhow!("--demo-bubbles requires --detector onnx:<path> (clap enforces this)")
            })?;
            measure_with_real_detector(&mut benchmark, &demo_bubble_jobs(), cells, spec)?
        }
        Source::Pages => {
            let dir = pages.ok_or_else(|| anyhow!("--pages is required for this source"))?;
            let spec = detector.ok_or_else(|| {
                anyhow!("--pages requires --detector onnx:<path> (clap enforces this)")
            })?;
            let jobs = local_page_jobs(dir)?;
            measure_with_real_detector(&mut benchmark, &jobs, cells, spec)?
        }
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
    measure_jobs(benchmark, jobs, cells, &detector);
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

    // --- §16.43 item 8: the mirrored eligibility predicate ---------------------

    fn region(failed: bool, std_deviation: f64, thickness: Option<u32>) -> MaskRegionStats {
        MaskRegionStats {
            rect: pc_core::Rect::new(0, 0, 4, 4),
            std_deviation,
            failed,
            thickness,
        }
    }

    /// Hand-derived oracle for [`eligible_region_count`], the local mirror of
    /// `pc_inpaint::eligible::select_regions` (`crates/pc-inpaint/src/eligible.rs:64-97`).
    ///
    /// Every expectation below is derived from that function's *text* plus
    /// `InpainterConfig::default()`'s two literals, never from running the mirror. The
    /// two boundary rows are the ones that catch a `>=`→`>` or `<=`→`<` drift, and the
    /// last row is the `thickness.is_some()` clause upstream comments as *"For box masks,
    /// this is none. We don't need to inpaint those, they are always good."*
    ///
    /// **This cannot catch cross-crate drift** — asserting the two copies agree needs a
    /// `pc-inpaint` dependency, which §16.43 item 9 reserves for the LaMa task. It is a
    /// guard on this copy only, and that limitation is real, not solved.
    #[test]
    fn the_mirrored_eligibility_predicate_matches_the_hand_derived_upstream_cases() {
        let config = InpainterConfig::default();
        // The boundary rows below mean what their names say only at these two values.
        assert_eq!(config.inpainting_min_std_dev, 15.0);
        assert_eq!(config.min_inpainting_radius, 7);

        let cases: [(&str, MaskRegionStats, bool); 5] = [
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
                eligible_region_count(std::slice::from_ref(stats), &config),
                usize::from(*expected),
                "{why}"
            );
        }

        // Anti-vacuity literal: the whole table counted at once is 2 — a number a
        // predicate that answered all-true (5) or all-false (0) cannot produce.
        let all: Vec<MaskRegionStats> = cases.iter().map(|(_, stats, _)| stats.clone()).collect();
        assert_eq!(eligible_region_count(&all, &config), 2);
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
        measure_jobs(&mut benchmark, &jobs, &cells, &detector);

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
        let (benchmark, facts) = measure(Source::DemoBubbles, &cells, None, Some(&spec))
            .expect("--demo-bubbles is a complete run");
        // Post-review fix (F2): the real detector's facts must reach the disclosure row —
        // confirm this run actually observed them, not the `--replay` placeholder.
        assert!(
            facts.model_path.is_some(),
            "no model path recorded for a real detector run"
        );
        assert!(
            facts.digest_verified,
            "digest verification was not observed"
        );
        assert!(
            facts.session_constructed,
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
