//! `pc_cli` — the library half of the `panel-ocr` binary (spec §13.1, §16.12 item 1).
//!
//! Split out from `main.rs` so the `clap` surface, the detector-spec grammar, the
//! verbosity mapping and the config discovery are unit-testable without spawning a
//! process. §1 rule 3: no algorithm code lives here.
//!
//! `run_clean` / `run_ocr` / the three management subcommands are `todo!()` skeletons
//! for task X1; their signatures are frozen with the tests.

pub mod args;
pub mod detector;
pub mod logging;
pub mod paths;
pub mod setup;

pub use args::{
    CacheCommand, CleanArgs, Cli, Command, DetectorSpec, ModelsCommand, OcrArgs, ProfileCommand,
    ReportFormatArg,
};

use anyhow::Result;
use pc_pipeline::{EXIT_FATAL, EXIT_OK};

/// Dispatch a parsed command line and return the process exit code (§5.5).
///
/// Any `Err` bubbling out of a subcommand is a **fatal** condition: it is reported on
/// stderr and becomes exit code 1. Per-image failures never reach here — they are
/// summarised by `pc_pipeline::BatchSummary` and become exit code 2.
pub fn run(cli: Cli) -> i32 {
    logging::init(cli.verbose, cli.quiet);

    let result = match cli.command {
        Command::Clean(args) => run_clean(args),
        Command::Ocr(args) => run_ocr(args),
        Command::Profile { command } => run_profile(command),
        Command::Cache { command } => run_cache(command),
        Command::Models { command } => run_models(command),
    };

    match result {
        Ok(code) => code,
        Err(error) => {
            tracing::error!("{error:#}");
            eprintln!("error: {error:#}");
            EXIT_FATAL
        }
    }
}

/// spec §13.1's `clean` (task X1).
///
/// Contract Codex must satisfy:
///   1. expand `paths` (`pc_pipeline::expand_inputs`); an empty result is fatal (§5.3);
///   2. load the app config + profile (`setup::load_app_config`/`load_profile`) and
///      build options (`setup::build_clean_options`) — all three already implemented;
///   3. build the detector provider (`detector::build_provider`); an `Err` is fatal and
///      must print [`detector::ONNX_UNAVAILABLE`] verbatim for `--detector onnx`
///      (§16.12 item 2);
///   4. create the cache dir (fatal on failure, §5.3) and run
///      `pc_pipeline::run_batch`;
///   5. always print `BatchSummary::render()`; print the analytics tables unless
///      `--hide-analytics`; show an `indicatif` progress bar when stderr is a terminal;
///   6. delete the cache dir afterwards unless `--keep-cache` (and never in Memory
///      mode, where there is none);
///   7. return `summary.exit_code()`.
pub fn run_clean(args: CleanArgs) -> Result<i32> {
    let _ = args;
    todo!("task X1 (spec §13.1): `clean` wiring")
}

/// spec §13.1's `ocr` (task X1). Runs stages 1–2 with `performing_ocr = true` and
/// renders `pc_export::render_ocr_report`. §16.12 item 4: with no engine in v1 the
/// report is empty and a `WARN` says so.
pub fn run_ocr(args: OcrArgs) -> Result<i32> {
    let _ = args;
    todo!("task X1 (spec §13.1): `ocr` wiring")
}

/// spec §13.1's `profile new|show|list|validate|edit` — thin wrappers over `pc-config`.
pub fn run_profile(command: ProfileCommand) -> Result<i32> {
    let _ = command;
    todo!("task X1 (spec §13.1): `profile` subcommands")
}

/// spec §13.1's `cache show|clear`.
pub fn run_cache(command: CacheCommand) -> Result<i32> {
    let _ = command;
    todo!("task X1 (spec §13.1): `cache` subcommands")
}

/// spec §13.1's `models download|verify|path`. `download`/`verify` require task D1 and
/// must fail with the same kind of explicit "not in this build" message as
/// [`detector::ONNX_UNAVAILABLE`] (§16.12 item 2); `path` works today.
pub fn run_models(command: ModelsCommand) -> Result<i32> {
    let _ = command;
    let _ = EXIT_OK;
    todo!("task X1 (spec §13.1): `models` subcommands")
}
