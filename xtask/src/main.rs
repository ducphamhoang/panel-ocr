//! panel-ocr maintainer tooling — spec §7.2 (task **F1**) and §7.3 (task **F2**).
//!
//! Never shipped: this crate is not in `pc-cli`'s dependency graph and is `publish = false`.
//!
//! Design rule, from §7.2's "maintainer machine, real models present": every recording
//! here must be produced by the **real** third-party implementation it is a reference
//! for (`cv2.fastNlMeansDenoising`, `cv2.INTER_AREA`, `PIL.FIND_EDGES`, the real ONNX
//! detector). A Rust re-implementation standing in for any of them would make the
//! corresponding parity gate circular and worthless, so a missing tool is always a
//! reported skip, never a fallback.

mod bench;
mod calibrate;
mod device;
mod env;
mod mask_sweep;
mod model_signature;
mod ocr_model_signature;
mod paths;
mod record;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use pc_core::device::{resolve, Device, DeviceSupport};
use record::{Group, Outcome};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "xtask",
    about = "panel-ocr maintainer tasks (spec §7.2 F1, §7.3 F2)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report what this environment can and cannot record, and exit.
    Probe,
    /// Task F1 (§7.2): record reference fixtures from the real third-party tools.
    RecordFixtures {
        /// Execution device. Committed recordings are restricted to CPU.
        #[arg(long, default_value = "cpu", value_parser = parse_device)]
        device: Device,
        /// Restrict to these groups (default: all). Unavailable groups are reported, not run.
        #[arg(long, value_delimiter = ',')]
        only: Vec<Group>,
        /// Python interpreter that has `cv2`, `PIL` and `numpy` importable.
        #[arg(long)]
        python: Option<PathBuf>,
        /// Model path used by the detector recording preflight, e.g. `onnx:/path/model.onnx`.
        #[arg(long, value_name = "SPEC")]
        detector: Option<String>,
        /// Pinned PanelCleaner checkout used by the detector oracle recorder.
        #[arg(long)]
        detector_upstream: Option<PathBuf>,
        /// Verified ONNX model path for the dependency-free `model-signature` group.
        #[arg(long)]
        model_signature: Option<PathBuf>,
        /// Verified manga-ocr encoder ONNX path for the dependency-free OCR signature group.
        #[arg(long)]
        ocr_encoder: Option<PathBuf>,
        /// Verified manga-ocr decoder ONNX path for the dependency-free OCR signature group.
        #[arg(long)]
        ocr_decoder: Option<PathBuf>,
        /// Re-record even when the output already exists.
        #[arg(long)]
        force: bool,
    },
    /// Task F2 (§7.3): measure the goldens and write `docs/GOLDEN_CALIBRATION.md`.
    CalibrateGoldens {
        /// Execution device. Committed calibration is restricted to CPU.
        #[arg(long, default_value = "cpu", value_parser = parse_device)]
        device: Device,
        /// Write somewhere other than `docs/GOLDEN_CALIBRATION.md`.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Detector model, using `onnx:<path>`; falls back to PANEL_OCR_ONNX_MODEL.
        #[arg(long, value_name = "SPEC")]
        detector: Option<String>,
    },
    /// Task T3 (§16.35): measure lowest-deviation mask-rescue opportunities.
    MaskSweep(mask_sweep::Args),
    /// Measure ONNX detector tuning candidates in fresh child processes.
    BenchDetector {
        /// Detector model, using `onnx:<path>`; falls back to PANEL_OCR_ONNX_MODEL.
        #[arg(long, value_name = "SPEC")]
        detector: Option<String>,
        /// Number of measured repetitions per candidate.
        #[arg(long, default_value_t = 5)]
        reps: usize,
        /// Warmup repetitions excluded from the summary.
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// `all` or a comma-separated list of built-in labels.
        #[arg(long, default_value = "all")]
        variants: String,
        /// Write the non-gating report here.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    #[command(hide = true)]
    BenchDetectorChild {
        #[arg(long)]
        detector: PathBuf,
        #[arg(long)]
        label: String,
        #[arg(long)]
        reps: usize,
        #[arg(long)]
        warmup: usize,
        #[arg(long)]
        raw_out: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Probe => probe(),
        Command::RecordFixtures {
            device,
            only,
            python,
            detector,
            detector_upstream,
            model_signature,
            ocr_encoder,
            ocr_decoder,
            force,
        } => {
            device::ensure_recording_request(device)?;
            let policy = resolve(device, DeviceSupport::compiled())
                .map_err(|refusal| anyhow::anyhow!(refusal.message()))?;
            let groups = if only.is_empty() {
                Group::ALL.to_vec()
            } else {
                only
            };
            let detector_path = detector
                .as_deref()
                .map(env::parse_detector_spec)
                .transpose()?;
            let model_signature = model_signature.or(detector_path);
            let results = record::run(
                &groups,
                python.as_deref(),
                detector.as_deref(),
                detector_upstream.as_deref(),
                model_signature.as_deref(),
                (ocr_encoder.as_deref(), ocr_decoder.as_deref()),
                force,
                &policy,
            )?;
            summarize(&results);
            let failed = results
                .iter()
                .filter(|(_, outcome)| matches!(outcome, Outcome::Failed { .. }))
                .count();
            if failed > 0 {
                bail!("{failed} group(s) failed during fixture recording");
            }
            Ok(())
        }
        Command::CalibrateGoldens {
            device,
            out,
            detector,
        } => {
            device::ensure_recording_request(device)?;
            let policy = resolve(device, DeviceSupport::compiled())
                .map_err(|refusal| anyhow::anyhow!(refusal.message()))?;
            calibrate::run(out.as_deref(), detector.as_deref(), &policy)
        }
        Command::MaskSweep(args) => mask_sweep::run(args),
        Command::BenchDetector {
            detector,
            reps,
            warmup,
            variants,
            out,
        } => {
            let detector = detector
                .as_deref()
                .map(env::parse_detector_spec)
                .transpose()?;
            bench::run(detector.as_deref(), reps, warmup, &variants, out.as_deref())
        }
        Command::BenchDetectorChild {
            detector,
            label,
            reps,
            warmup,
            raw_out,
        } => bench::run_child(&detector, &label, reps, warmup, &raw_out),
    }
}

fn parse_device(value: &str) -> std::result::Result<Device, String> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda),
        _ => Err(format!(
            "invalid device `{value}`; expected `cpu` or `cuda`"
        )),
    }
}

fn probe() -> Result<()> {
    println!("panel-ocr xtask capability probe\n");
    match env::PythonTooling::discover(None)? {
        Some(found) => println!("cv2 + PIL: AVAILABLE — {}", found.describe()),
        None => println!("cv2 + PIL: MISSING\n{}", env::NO_PYTHON_HELP),
    }
    println!("\nONNX detector: UNAVAILABLE");
    // `probe` has no detector CLI option; `None` deliberately means consult only
    // PANEL_OCR_ONNX_MODEL when the ONNX feature is enabled.
    println!("{}", env::detector_backend_status(None)?.explain());
    println!(
        "\nmodel-signature: {}",
        model_signature::capability_status(None)
    );
    println!(
        "ocr-model-signature: {}",
        ocr_model_signature::capability_status(None, None)
    );
    Ok(())
}

fn summarize(results: &[(Group, Outcome)]) {
    println!("\n═══ summary ═══");
    let mut skipped = 0;
    let mut failed = 0;
    for (group, outcome) in results {
        let (tag, detail) = match outcome {
            Outcome::Recorded { detail } => ("ok     ", detail.as_str()),
            Outcome::Skipped { .. } => {
                skipped += 1;
                (
                    "SKIPPED",
                    "see above for the reason and the follow-up command",
                )
            }
            Outcome::Failed { reason } => {
                failed += 1;
                ("FAILED ", reason.as_str())
            }
        };
        println!("{tag} {group:?}: {detail}");
    }
    if skipped > 0 {
        println!(
            "\n{skipped} group(s) skipped. The tests that depend on them MUST stay `#[ignore]`d —\n\
             producing a stand-in fixture would defeat the purpose of a parity gate (§7.3)."
        );
    }
    if failed > 0 {
        println!("\n{failed} group(s) failed. The recording run will exit non-zero.");
    }
}
