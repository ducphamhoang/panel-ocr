//! Task T3 (§16.35 item 8): non-gating mask-rescue calibration.

use crate::paths;
use anyhow::{anyhow, bail, Context, Result};
use pc_config::{MaskerConfig, PreprocessorConfig, TextDetectorConfig};
use pc_core::ImageHandle;
use pc_mask::fit::{fit_accepted, fit_region_scored, resolve_fallback, Scored, Selection};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const REPLAY_STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";
const PROVENANCE_PATH: &str = "tests/fixtures/recorded/detector/PROVENANCE.json";

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    /// Replay the committed detector fixture (CI-runnable; no model is loaded).
    #[arg(
        long,
        conflicts_with_all = ["pages", "detector"],
        required_unless_present = "pages"
    )]
    replay: bool,
    /// Directory of maintainer-local pages to measure with a real ONNX detector.
    #[arg(
        long,
        value_name = "DIR",
        requires = "detector",
        conflicts_with = "replay"
    )]
    pages: Option<PathBuf>,
    /// Real detector specification in the form `onnx:<path>` (requires --pages).
    #[arg(
        long,
        value_name = "SPEC",
        requires = "pages",
        conflicts_with = "replay"
    )]
    detector: Option<String>,
    /// Person or automation that reviewed this calibration run.
    #[arg(long)]
    reviewer: String,
    /// Review date in YYYY-MM-DD form.
    #[arg(long)]
    date: String,
    /// Literal description of the measurement method.
    #[arg(long)]
    method: String,
    /// Report destination (default: docs/MASK_QUALITY_CALIBRATION.md).
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Accepted,
    Rescued,
    FailedEvenWithRescue,
}

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rescued => "rescued",
            Self::FailedEvenWithRescue => "failed-even-with-rescue",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RegionMeasurement {
    page: String,
    region: usize,
    greedy_index: usize,
    greedy_deviation: f64,
    lowest_index: usize,
    lowest_deviation: f64,
    outcome: Outcome,
    candidate_deviations: Vec<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Summary {
    pages: usize,
    masking_regions: usize,
    reached_fitting: usize,
    already_accepted: usize,
    rescue_eligible: usize,
    failed_even_with_rescue: usize,
    dropped: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct Sweep {
    summary: Summary,
    regions: Vec<RegionMeasurement>,
}

impl Sweep {
    fn add_page(
        &mut self,
        page: &str,
        masking_regions: usize,
        reports: Vec<(usize, pc_mask::fit::FitReport)>,
        threshold: f64,
    ) {
        self.summary.pages += 1;
        self.summary.masking_regions += masking_regions;
        self.summary.reached_fitting += reports.len();
        self.summary.dropped += masking_regions - reports.len();

        for (region, report) in reports {
            let selection = selection_from_report(&report);
            let resolved = resolve_fallback(&selection, threshold, true);
            let greedy_accepted = fit_accepted(selection.greedy.std_deviation, threshold);
            let outcome = if greedy_accepted {
                self.summary.already_accepted += 1;
                Outcome::Accepted
            } else if resolved.index != selection.greedy.index
                && fit_accepted(resolved.std_deviation, threshold)
            {
                self.summary.rescue_eligible += 1;
                Outcome::Rescued
            } else {
                self.summary.failed_even_with_rescue += 1;
                Outcome::FailedEvenWithRescue
            };
            self.regions.push(RegionMeasurement {
                page: page.to_owned(),
                region,
                greedy_index: selection.greedy.index,
                greedy_deviation: selection.greedy.std_deviation,
                lowest_index: selection.lowest.index,
                lowest_deviation: selection.lowest.std_deviation,
                outcome,
                candidate_deviations: selection.deviations,
            });
        }
    }
}

pub(crate) fn run(args: Args) -> Result<()> {
    parse_date(&args.date).map_err(anyhow::Error::msg)?;

    let target = args
        .out
        .unwrap_or_else(|| paths::docs_root().join("MASK_QUALITY_CALIBRATION.md"));
    let (sweep, detector_provenance) = if args.replay {
        run_replay()?
    } else {
        run_local(
            args.pages
                .as_deref()
                .expect("clap requires --pages outside replay mode"),
            args.detector
                .as_deref()
                .expect("clap requires --detector with --pages"),
        )?
    };
    let document = render_document(
        &sweep,
        &args.reviewer,
        &args.date,
        &args.method,
        &detector_provenance,
    );
    std::fs::write(&target, &document).with_context(|| format!("writing {}", target.display()))?;
    println!(
        "wrote {} ({} bytes)",
        paths::display_relative(&target),
        document.len()
    );
    Ok(())
}

fn run_replay() -> Result<(Sweep, String)> {
    let fixture_dir = paths::recorded_root().join("detector");
    let page = fixture_dir.join(format!("{REPLAY_STEM}.jpg"));
    let detector = pc_detect::ReplayDetector::new(&fixture_dir, REPLAY_STEM);
    let mut sweep = Sweep::default();
    measure_page(&mut sweep, &page, REPLAY_STEM, &detector)?;

    let provenance_path = paths::workspace_root().join(PROVENANCE_PATH);
    let provenance = std::fs::read_to_string(&provenance_path)
        .with_context(|| format!("reading {}", provenance_path.display()))?;
    Ok((
        sweep,
        format!(
            "ReplayDetector fixture.\n\n<details><summary>Detector provenance (`{PROVENANCE_PATH}`)</summary>\n\n```json\n{}\n```\n\n</details>",
            provenance.trim_end()
        ),
    ))
}

#[cfg(feature = "onnx")]
fn run_local(pages: &Path, detector_spec: &str) -> Result<(Sweep, String)> {
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

    let page_paths = collect_page_paths(pages)?;
    let mut sweep = Sweep::default();
    for page in page_paths {
        let name = page
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| anyhow!("page has no UTF-8 file stem: {}", page.display()))?;
        measure_page(&mut sweep, &page, name, &detector)?;
    }
    Ok((
        sweep,
        format!("Real ONNX detector: `onnx:{}`.", model.display()),
    ))
}

#[cfg(not(feature = "onnx"))]
fn run_local(_pages: &Path, detector_spec: &str) -> Result<(Sweep, String)> {
    let _ = crate::env::parse_detector_spec(detector_spec)?;
    bail!(
        "--pages requires the real ONNX detector; rerun with `cargo xtask --features onnx mask-sweep ...`"
    )
}

#[cfg(feature = "onnx")]
fn collect_page_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        bail!("--pages is not a directory: {}", dir.display());
    }
    let mut pages = std::fs::read_dir(dir)
        .with_context(|| format!("reading pages directory {}", dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    pages.retain(|path| {
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
    pages.sort();
    if pages.is_empty() {
        bail!("--pages contains no supported images: {}", dir.display());
    }
    Ok(pages)
}

fn measure_page(
    sweep: &mut Sweep,
    page_path: &Path,
    page_name: &str,
    detector: &dyn pc_detect::TextDetector,
) -> Result<()> {
    let detect_output = pc_detect::run(
        pc_detect::DetectInput {
            schema_version: pc_core::SCHEMA_VERSION,
            source: ImageHandle::from_path(page_path),
            original_path: page_path.to_path_buf(),
            target_height_lower: 1000,
            target_height_upper: 4000,
            base_image_dest: None,
            raw_mask_dest: None,
            min_mask_coverage: pc_detect::DEFAULT_MIN_MASK_COVERAGE,
            config: TextDetectorConfig::default(),
        },
        detector,
    )
    .map_err(|error| anyhow!(error.to_string()))
    .with_context(|| format!("detecting {}", page_path.display()))?;
    let page = pc_preprocess::run(
        pc_preprocess::PreprocessInput {
            schema_version: pc_core::SCHEMA_VERSION,
            page: detect_output.page,
            config: PreprocessorConfig::default(),
            performing_ocr: false,
        },
        None,
    )
    .map_err(|error| anyhow!(error.to_string()))
    .with_context(|| format!("preprocessing {}", page_path.display()))?
    .page;

    let base_image = page
        .base_image
        .load()
        .map_err(|error| anyhow!(error.to_string()))?;
    let raw_mask = page
        .raw_mask
        .load()
        .map_err(|error| anyhow!(error.to_string()))?;
    let base = pc_mask::BaseCanvas::from_dynamic(&base_image);
    let precise = pc_imageops::BinaryMask::from_gray_threshold(
        &raw_mask.to_luma8(),
        pc_imageops::PIL_BINARY_THRESHOLD,
    );
    let box_mask = pc_imageops::rasterize_boxes(&page.extended_boxes, page.image_size);
    let cut = precise.and(&box_mask);
    let masker = MaskerConfig::default();
    let reports = page
        .masking_regions
        .iter()
        .enumerate()
        .filter_map(|(index, region)| {
            fit_region_scored(
                &base,
                &cut,
                &box_mask,
                region.masking,
                region.reference,
                &masker,
            )
            .map(|report| (index, report))
        })
        .collect();
    sweep.add_page(
        page_name,
        page.masking_regions.len(),
        reports,
        masker.mask_max_standard_deviation,
    );
    Ok(())
}

fn selection_from_report(report: &pc_mask::fit::FitReport) -> Selection {
    let lowest_index = report
        .candidate_deviations
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .expect("fit_region_scored reports at least one scored candidate");
    let scored = |index| Scored {
        index,
        std_deviation: report.candidate_deviations[index],
        median_color: report.fitment.median_color,
    };
    Selection {
        greedy: scored(report.greedy_index),
        lowest: scored(lowest_index),
        deviations: report.candidate_deviations.clone(),
    }
}

fn render_document(
    sweep: &Sweep,
    reviewer: &str,
    date: &str,
    method: &str,
    detector_provenance: &str,
) -> String {
    let summary = &sweep.summary;
    let mut body = String::new();
    body.push_str(
        "# Mask quality calibration (spec §16.35, task T3)\n\n\
         **Generated in full by `cargo xtask mask-sweep` — do not hand-edit.** Every value\n\
         here is measured at generation time. This is a **non-gating** calibration report;\n\
         it records evidence for the lowest-deviation rescue without changing tolerances.\n\n\
         ## 1. Method and provenance\n\n",
    );
    let _ = writeln!(body, "- **Reviewer:** {reviewer}");
    let _ = writeln!(body, "- **Date:** {date}");
    let _ = writeln!(body, "- **Method:** {method}");
    let _ = writeln!(
        body,
        "- **Rescue policy:** `resolve_fallback(selection, mask_max_standard_deviation, true)` (measured independently of T4 configuration).\n"
    );
    body.push_str(detector_provenance.trim_end());
    body.push_str("\n\n## 2. Summary\n\n");
    body.push_str("| Pages | Masking regions | Reached fitting | Already accepted | Rescue-eligible | Failed even with rescue | Dropped |\n");
    body.push_str("|---:|---:|---:|---:|---:|---:|---:|\n");
    let _ = writeln!(
        body,
        "| {} | {} | {} | {} | {} | {} | {} |",
        summary.pages,
        summary.masking_regions,
        summary.reached_fitting,
        summary.already_accepted,
        summary.rescue_eligible,
        summary.failed_even_with_rescue,
        summary.dropped
    );
    body.push_str("\n## 3. Region measurements\n\n");
    body.push_str("| Page | Region | Greedy index | Greedy deviation | Lowest index | Lowest deviation | Outcome | Candidate deviations |\n");
    body.push_str("|---|---:|---:|---:|---:|---:|---|---|\n");
    for region in &sweep.regions {
        let ladder = region
            .candidate_deviations
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            body,
            "| {} | {} | {} | {:.6} | {} | {:.6} | {} | [{}] |",
            region.page,
            region.region,
            region.greedy_index,
            region.greedy_deviation,
            region.lowest_index,
            region.lowest_deviation,
            region.outcome.label(),
            ladder
        );
    }
    body.push_str("\n## 4. Verdict\n\n");
    let _ = writeln!(
        body,
        "**Non-gating verdict:** measured {} rescue-eligible region(s); {} region(s) still failed the unchanged fail-safe even with rescue. No threshold was loosened.",
        summary.rescue_eligible, summary.failed_even_with_rescue
    );
    body
}

fn parse_date(value: &str) -> std::result::Result<String, String> {
    let bytes = value.as_bytes();
    let valid = bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit());
    if valid {
        Ok(value.to_owned())
    } else {
        Err("expected YYYY-MM-DD".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_mask::fit::{FitReport, Fitment};

    fn report(greedy_index: usize, deviations: &[f64]) -> FitReport {
        FitReport {
            fitment: Fitment {
                mask: None,
                median_color: [1, 2, 3],
                coords: (0, 0),
                std_deviation: deviations[greedy_index],
                candidate_index: greedy_index,
                thickness: None,
                masking_rect: pc_core::Rect::new(0, 0, 1, 1),
            },
            candidate_deviations: deviations.to_vec(),
            greedy_index,
            rescued: false,
        }
    }

    #[test]
    fn aggregation_distinguishes_accepted_rescued_failed_and_dropped_regions() {
        let mut sweep = Sweep::default();
        sweep.add_page(
            "page",
            4,
            vec![
                (0, report(0, &[12.0, 13.0])),
                (1, report(0, &[16.0, 14.5])),
                (2, report(0, &[30.0, 29.0])),
            ],
            15.0,
        );
        assert_eq!(
            sweep.summary,
            Summary {
                pages: 1,
                masking_regions: 4,
                reached_fitting: 3,
                already_accepted: 1,
                rescue_eligible: 1,
                failed_even_with_rescue: 1,
                dropped: 1,
            }
        );
        assert_eq!(
            sweep
                .regions
                .iter()
                .map(|region| region.outcome)
                .collect::<Vec<_>>(),
            vec![
                Outcome::Accepted,
                Outcome::Rescued,
                Outcome::FailedEvenWithRescue
            ]
        );
    }

    #[test]
    fn renderer_uses_six_decimals_and_keeps_signed_fields_literal() {
        let mut sweep = Sweep::default();
        sweep.add_page("page", 1, vec![(0, report(0, &[12.3456789]))], 15.0);
        let document = render_document(&sweep, "reviewer", "2026-08-05", "method", "provenance");
        assert!(document.contains("- **Reviewer:** reviewer"));
        assert!(document.contains("- **Date:** 2026-08-05"));
        assert!(document.contains("- **Method:** method"));
        assert!(document
            .contains("| page | 0 | 0 | 12.345679 | 0 | 12.345679 | accepted | [12.345679] |"));
    }
}
