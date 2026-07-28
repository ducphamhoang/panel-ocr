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

mod calibrate;
mod env;
mod paths;
mod record;

use anyhow::Result;
use clap::{Parser, Subcommand};
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
        /// Restrict to these groups (default: all). Unavailable groups are reported, not run.
        #[arg(long, value_delimiter = ',')]
        only: Vec<Group>,
        /// Python interpreter that has `cv2`, `PIL` and `numpy` importable.
        #[arg(long)]
        python: Option<PathBuf>,
        /// Detector backend for the `detector` group, e.g. `onnx:/path/comictextdetector.pt.onnx`.
        /// Accepted but not yet usable: see the group's skip message (spec §8.5 D1/D4).
        #[arg(long, value_name = "SPEC")]
        detector: Option<String>,
        /// Re-record even when the output already exists.
        #[arg(long)]
        force: bool,
    },
    /// Task F2 (§7.3): measure the goldens and write `docs/GOLDEN_CALIBRATION.md`.
    CalibrateGoldens {
        /// Write somewhere other than `docs/GOLDEN_CALIBRATION.md`.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Probe => probe(),
        Command::RecordFixtures {
            only,
            python,
            detector,
            force,
        } => {
            let groups = if only.is_empty() {
                Group::ALL.to_vec()
            } else {
                only
            };
            if let Some(spec) = &detector {
                // Explicitly acknowledged rather than silently ignored: a maintainer who
                // supplies weights must be told why nothing happened with them.
                println!(
                    "note: --detector {spec} was supplied, but the backend it would drive does \
                     not exist yet (spec §8.5 D1/D4). See the `detector` group below."
                );
            }
            let results = record::run(&groups, python.as_deref(), force)?;
            summarize(&results);
            Ok(())
        }
        Command::CalibrateGoldens { out } => calibrate::run(out.as_deref()),
    }
}

fn probe() -> Result<()> {
    println!("panel-ocr xtask capability probe\n");
    match env::PythonTooling::discover(None)? {
        Some(found) => println!("cv2 + PIL: AVAILABLE — {}", found.describe()),
        None => println!("cv2 + PIL: MISSING\n{}", env::NO_PYTHON_HELP),
    }
    println!("\nONNX detector: UNAVAILABLE");
    println!("{}", env::detector_backend_status().explain());
    Ok(())
}

fn summarize(results: &[(Group, Outcome)]) {
    println!("\n═══ summary ═══");
    let mut skipped = 0;
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
        };
        println!("{tag} {group:?}: {detail}");
    }
    if skipped > 0 {
        println!(
            "\n{skipped} group(s) skipped. The tests that depend on them MUST stay `#[ignore]`d —\n\
             producing a stand-in fixture would defeat the purpose of a parity gate (§7.3)."
        );
    }
}
