//! `pc_cli` — the library half of the `panel-ocr` binary (spec §13.1, §16.12 item 1).
//!
//! Split out from `main.rs` so the `clap` surface, the detector-spec grammar, the
//! verbosity mapping and the config discovery are unit-testable without spawning a
//! process. §1 rule 3: no algorithm code lives here.
//!
//! `run_clean` / `run_ocr` / the three management subcommands are implemented here (task
//! X1); their signatures are frozen with the tests.

pub mod args;
pub mod detector;
pub mod logging;
pub mod models;
pub mod ocr;
pub mod paths;
pub mod setup;

pub use args::{
    CacheCommand, CleanArgs, Cli, Command, DetectorSpec, ModelsCommand, OcrArgs, ProfileCommand,
    ReportFormatArg,
};

use anyhow::{anyhow, bail, Context, Result};
use paths::{EnvSource, Platform};
use pc_pipeline::{Checkpointing, ImageOutcome, PipelineCtx, PipelineOptions, EXIT_FATAL, EXIT_OK};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

/// Resolve `$EDITOR` when set, else the platform's fallback, else `None`.
pub fn resolve_editor(platform: Platform, env: &dyn EnvSource) -> Option<std::ffi::OsString> {
    env.var("EDITOR").or_else(|| match platform {
        Platform::Windows => Some(std::ffi::OsString::from("notepad.exe")),
        Platform::Linux | Platform::MacOs => None,
    })
}

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
            eprintln!("error: {error:#}");
            EXIT_FATAL
        }
    }
}

/// spec §13.1's `clean` (task X1).
///
/// Contract (each clause is a frozen test):
///   1. expand `paths` (`pc_pipeline::expand_inputs`); an empty result is fatal (§5.3);
///   2. load the app config + profile (`setup::load_app_config`/`load_profile`) and
///      build options (`setup::build_clean_options`) — all three already implemented;
///   3. build the detector provider (`detector::build_provider`); ONNX resolves and verifies
///      its model on first detection use, while an `Err` at construction remains fatal and
///      must print [`detector::ONNX_UNAVAILABLE`] verbatim for `--detector onnx`
///      (§16.12 item 2, §16.18);
///   4. create the cache dir (fatal on failure, §5.3) and run
///      `pc_pipeline::run_batch`;
///   5. always print `BatchSummary::render()`; print the analytics tables unless
///      `--hide-analytics`; show an `indicatif` progress bar when stderr is a terminal;
///   6. delete the cache dir afterwards unless `--keep-cache` (and never in Memory
///      mode, where there is none);
///   7. return `summary.exit_code()`.
pub fn run_clean(args: CleanArgs) -> Result<i32> {
    let images = pc_pipeline::expand_inputs(&args.paths)?;
    ensure_inputs(&images)?;

    let config = setup::load_app_config()?;
    let profile = setup::load_profile(
        args.profile.as_deref(),
        args.profile_path.as_deref(),
        &config,
    )?;
    let device = profile.general.device;
    let cache_root = paths::resolve_cache_root(args.cache_dir.as_deref(), &config);
    let provider = detector::build_provider(
        &args.detector,
        args.model_path.as_deref(),
        profile.text_detector.model_path(),
        &cache_root,
        &profile.text_detector,
        device,
    )?;
    let ocr_factory = if profile.preprocessor.ocr_enabled {
        Some(ocr::build_factory_for_device(&cache_root, device)?)
    } else {
        None
    };
    let options = setup::build_clean_options(&args, profile, images.len(), &cache_root);
    run_pipeline(
        &images,
        options,
        provider.as_ref(),
        ocr_factory.as_deref(),
        !args.hide_analytics,
    )
}

/// spec §13.1's `ocr` (task X1). Runs stages 1–2 with `performing_ocr = true` and
/// renders `pc_export::render_ocr_report` using one eagerly constructed OCR factory.
///
/// `performing_ocr` stops the chain after preprocessing (see
/// `pc_pipeline::process_image`), so the report — stdout or `--output FILE` — is the run's
/// *only* product: `ocr` has no `--output-dir` and must never write cleaning artifacts.
pub fn run_ocr(args: OcrArgs) -> Result<i32> {
    let images = pc_pipeline::expand_inputs(&args.paths)?;
    ensure_inputs(&images)?;

    let config = setup::load_app_config()?;
    let mut profile = setup::load_profile(
        args.profile.as_deref(),
        args.profile_path.as_deref(),
        &config,
    )?;
    ocr::apply_report_overrides(&mut profile);
    let device = profile.general.device;
    let cache_root = paths::resolve_cache_root(args.cache_dir.as_deref(), &config);
    let ocr_factory = ocr::build_factory_for_device(&cache_root, device)?;
    let provider = detector::build_provider(
        &args.detector,
        None,
        profile.text_detector.model_path(),
        &cache_root,
        &profile.text_detector,
        device,
    )?;
    let cache_dir = paths::image_cache_dir(&cache_root);
    let options = PipelineOptions {
        threads: pc_pipeline::resolve_threads(profile.general.max_threads, images.len()),
        profile,
        cache_dir,
        performing_ocr: true,
        ..PipelineOptions::default()
    };
    let ctx = PipelineCtx::new(provider.as_ref()).with_ocr(ocr_factory.as_ref());
    let summary = pc_pipeline::run_batch(&images, &options, &ctx);
    if let Some(message) = summary.fatal_model_message() {
        cleanup_cache(&options)?;
        return Err(anyhow!(message));
    }
    let report = ocr_report(&summary, args.format.into());
    match args.output {
        Some(path) => std::fs::write(&path, report)
            .with_context(|| format!("failed to write OCR report `{}`", path.display()))?,
        None => print!("{report}"),
    }
    print!("{}", summary.render());
    cleanup_cache(&options)?;
    Ok(summary.exit_code())
}

/// spec §13.1's `profile new|show|list|validate|edit` — thin wrappers over `pc-config`.
pub fn run_profile(command: ProfileCommand) -> Result<i32> {
    match command {
        ProfileCommand::New { path } => {
            pc_config::ProfileDocument::new_default()
                .save(&path)
                .with_context(|| format!("failed to write profile `{}`", path.display()))?;
        }
        ProfileCommand::Show {
            profile,
            profile_path,
        } => {
            if profile.is_some() && profile_path.is_some() {
                bail!("--profile and --profile-path cannot be used together");
            }
            let config = setup::load_app_config()?;
            let loaded = setup::load_profile(profile.as_deref(), profile_path.as_deref(), &config)?;
            print!(
                "{}",
                pc_config::ProfileDocument::from_profile(&loaded).to_toml_string()
            );
        }
        ProfileCommand::List => {
            let config = setup::load_app_config()?;
            for name in config.saved_profiles.keys() {
                println!("{name}");
            }
        }
        ProfileCommand::Validate { path } => {
            let config = pc_config::Config::default();
            setup::load_profile(None, Some(&path), &config)?;
            println!("valid: {}", path.display());
        }
        ProfileCommand::Edit { profile } => {
            let config = setup::load_app_config()?;
            let name = profile
                .as_deref()
                .or(config.default_profile.as_deref())
                .context("no profile selected; pass --profile NAME or configure default_profile")?;
            let path = config
                .profile_path(name)
                .with_context(|| format!("no profile named `{name}` in the app config"))?;
            let editor = resolve_editor(paths::Platform::HOST, &paths::ProcessEnv)
                .context("$EDITOR is not set")?;
            let status = std::process::Command::new(editor)
                .arg(path)
                .status()
                .context("failed to start $EDITOR")?;
            if !status.success() {
                bail!("$EDITOR exited with {status}");
            }
        }
    }
    Ok(EXIT_OK)
}

/// spec §13.1's `cache show|clear`.
pub fn run_cache(command: CacheCommand) -> Result<i32> {
    let config = setup::load_app_config()?;
    let cli_override = match &command {
        CacheCommand::Show { cache_dir } | CacheCommand::Clear { cache_dir, .. } => {
            cache_dir.as_deref()
        }
    };
    let cache_dir = paths::resolve_cache_root(cli_override, &config);
    match command {
        CacheCommand::Show { .. } => {
            println!(
                "{}\t{} bytes",
                cache_dir.display(),
                directory_size(&cache_dir)?
            );
        }
        CacheCommand::Clear { models, images, .. } => {
            let clear_all = !models && !images;
            // Per category, never the whole root: `--images` must not take `models/` with
            // it (task D1's weights are expensive to re-download), and `--models` must not
            // take the per-image artifacts.
            if clear_all || images {
                remove_dir_contents(&paths::image_cache_dir(&cache_dir))?;
            }
            if clear_all || models {
                remove_dir_contents(&paths::models_dir(&cache_dir))?;
            }
        }
    }
    Ok(EXIT_OK)
}

/// spec §13.1's `models download|verify|path`.
pub fn run_models(command: ModelsCommand) -> Result<i32> {
    match command {
        ModelsCommand::Download { cache_dir } => {
            let models_dir = models::resolve_managed_models_dir(cache_dir.as_deref())?;
            let fetcher = pc_models::ReqwestFetcher::new();
            let mut progress = models::progress_sink();
            for spec in pc_models::ALL {
                let path = pc_models::ensure_available(
                    spec,
                    &models_dir,
                    None,
                    &fetcher,
                    progress.as_mut(),
                )
                .with_context(|| format!("failed to install model `{}`", spec.name))?;
                println!("{}\t{}", spec.name, path.display());
            }
            Ok(EXIT_OK)
        }
        ModelsCommand::Verify { cache_dir } => {
            let models_dir = models::resolve_managed_models_dir(cache_dir.as_deref())?;
            let mut all_ok = true;
            for spec in pc_models::ALL {
                let resolution = pc_models::resolve(spec, &models_dir, None)?;
                match resolution {
                    pc_models::Resolution::Missing(path) => {
                        all_ok = false;
                        println!("{}\tMISSING\t{}", spec.name, path.display());
                    }
                    pc_models::Resolution::Cached(path) => {
                        if let Some(expected_size) = models::expected_size(spec) {
                            let actual_size = std::fs::metadata(&path)
                                .with_context(|| {
                                    format!("failed to inspect model `{}`", spec.name)
                                })?
                                .len();
                            if actual_size != expected_size {
                                all_ok = false;
                                println!(
                                    "{}\tSIZE MISMATCH\t{}\tactual={}\texpected={}",
                                    spec.name,
                                    path.display(),
                                    actual_size,
                                    expected_size
                                );
                                continue;
                            }
                        }
                        match pc_models::verify_sha256(&path, spec.sha256) {
                            Ok(()) => println!("{}\tOK\t{}", spec.name, path.display()),
                            Err(pc_models::ModelError::HashMismatch {
                                actual, expected, ..
                            }) => {
                                all_ok = false;
                                println!(
                                    "{}\tHASH MISMATCH\t{}\tactual={}\texpected={}",
                                    spec.name,
                                    path.display(),
                                    actual,
                                    expected
                                );
                            }
                            Err(error) => {
                                all_ok = false;
                                println!("{}\tERROR\t{}\t{}", spec.name, path.display(), error);
                            }
                        }
                    }
                    pc_models::Resolution::Override(_) => {
                        unreachable!("managed model verification never supplies an override")
                    }
                }
            }
            Ok(if all_ok { EXIT_OK } else { EXIT_FATAL })
        }
        ModelsCommand::Path { cache_dir } => {
            let models_dir = models::resolve_managed_models_dir(cache_dir.as_deref())?;
            println!("{}", models_dir.display());
            for spec in pc_models::ALL {
                let resolution = pc_models::resolve(spec, &models_dir, None)?;
                let present = matches!(resolution, pc_models::Resolution::Cached(_));
                println!(
                    "{}\t{}\t{}",
                    spec.name,
                    resolution.path().display(),
                    if present { "present" } else { "missing" }
                );
            }
            Ok(EXIT_OK)
        }
    }
}

fn ensure_inputs(images: &[PathBuf]) -> Result<()> {
    if images.is_empty() {
        bail!("no input images found");
    }
    Ok(())
}

fn run_pipeline(
    images: &[PathBuf],
    options: PipelineOptions,
    provider: &dyn pc_pipeline::DetectorProvider,
    ocr: Option<&dyn pc_ocr::OcrEngineFactory>,
    show_analytics: bool,
) -> Result<i32> {
    if options.checkpointing == Checkpointing::Disk {
        std::fs::create_dir_all(&options.cache_dir).with_context(|| {
            format!(
                "failed to create cache directory `{}`",
                options.cache_dir.display()
            )
        })?;
    }
    let progress = std::io::stderr().is_terminal().then(|| {
        let bar = indicatif::ProgressBar::new(images.len() as u64);
        bar.set_message("cleaning pages");
        bar
    });
    let mut ctx = PipelineCtx::new(provider);
    if let Some(ocr) = ocr {
        ctx = ctx.with_ocr(ocr);
    }
    let summary = pc_pipeline::run_batch(images, &options, &ctx);
    if let Some(bar) = progress {
        bar.finish_with_message("cleaning complete");
    }
    if let Some(message) = summary.fatal_model_message() {
        cleanup_cache(&options)?;
        return Err(anyhow!(message));
    }
    print!("{}", summary.render());
    if show_analytics {
        render_analytics(&summary);
    }
    cleanup_cache(&options)?;
    Ok(summary.exit_code())
}

/// Delete this run's per-image cache directory (`{root}/images`, never the root itself, so
/// a future `{root}/models` survives an ordinary `clean`).
fn cleanup_cache(options: &PipelineOptions) -> Result<()> {
    if options.checkpointing == Checkpointing::Disk
        && !options.keep_cache
        && options.cache_dir.exists()
    {
        std::fs::remove_dir_all(&options.cache_dir).with_context(|| {
            format!(
                "failed to remove cache directory `{}`",
                options.cache_dir.display()
            )
        })?;
    }
    Ok(())
}

fn render_analytics(summary: &pc_pipeline::BatchSummary) {
    for outcome in &summary.outcomes {
        let ImageOutcome::Completed { analytics, .. } = outcome else {
            continue;
        };
        if let Some(detect) = &analytics.detect {
            println!(
                "detect\t{}\t{}\t{}",
                detect.path.display(),
                detect.blocks_detected,
                detect.blocks_kept
            );
        }
        if let Some(ocr) = &analytics.ocr {
            println!("ocr\t{}\t{}", ocr.path.display(), ocr.num_boxes);
        }
        for mask in &analytics.mask_fitting {
            println!(
                "mask\t{}\t{}\t{}",
                mask.path.display(),
                mask.fit_found,
                mask.std_deviation
            );
        }
        if let Some(denoise) = &analytics.denoise {
            println!(
                "denoise\t{}\t{}",
                denoise.path.display(),
                denoise.boxes_denoised
            );
        }
    }
}

pub fn ocr_report(summary: &pc_pipeline::BatchSummary, format: pc_export::ReportFormat) -> String {
    let analytics = summary
        .outcomes
        .iter()
        .filter_map(|outcome| match outcome {
            ImageOutcome::Completed { analytics, .. } => {
                analytics.ocr.as_ref().filter(|ocr| !ocr.removed.is_empty())
            }
            _ => None,
        })
        .cloned()
        .collect::<Vec<_>>();
    pc_export::render_ocr_report(format, &analytics)
}

fn directory_size(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let mut bytes = 0;
    for entry in std::fs::read_dir(path)
        .with_context(|| format!("failed to read cache directory `{}`", path.display()))?
    {
        let entry = entry?;
        let child = entry.path();
        if child.is_dir() {
            bytes += directory_size(&child)?;
        } else {
            bytes += entry.metadata()?.len();
        }
    }
    Ok(bytes)
}

fn remove_dir_contents(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(path)
        .with_context(|| format!("failed to read cache directory `{}`", path.display()))?
    {
        let child = entry?.path();
        if child.is_dir() {
            std::fs::remove_dir_all(&child)?;
        } else {
            std::fs::remove_file(&child)?;
        }
    }
    Ok(())
}
