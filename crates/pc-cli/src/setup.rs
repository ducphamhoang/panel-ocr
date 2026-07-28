//! Config discovery and option assembly (spec §6, §13.1, §16.12 items 6, 14, 15, 17).
//!
//! Fully pinned; implemented, not stubbed. §1 rule 3 holds: this is lookup and
//! translation, no algorithms.

use crate::args::CleanArgs;
use crate::paths;
use anyhow::{Context, Result};
use pc_config::{Config, ConfigDocument, Profile, ProfileDocument};
use pc_pipeline::{resolve_threads, select_checkpointing, PipelineOptions};
use std::path::Path;

/// Load the app-level config, or the empty default when the file is absent.
pub fn load_app_config() -> Result<Config> {
    let path = paths::default_config_path();
    if !path.exists() {
        return Ok(Config::default());
    }
    let document = ConfigDocument::load(&path)
        .with_context(|| format!("failed to load app config `{}`", path.display()))?;
    for warning in document.warnings() {
        tracing::warn!("{}", warning.message());
    }
    let config = document.config().clone();
    config.validate()?;
    Ok(config)
}

/// spec §13.1 — `--profile-path FILE` wins, then `--profile NAME` through the app
/// config, then the app config's `default_profile`, then the built-in §6 defaults.
///
/// Validation failures are **fatal** (§5.3, §6: "fail config load, not per-image").
pub fn load_profile(
    name: Option<&str>,
    explicit_path: Option<&Path>,
    config: &Config,
) -> Result<Profile> {
    let path = match (explicit_path, name) {
        (Some(path), _) => Some(path.to_path_buf()),
        (None, Some(name)) => Some(
            config
                .profile_path(name)
                .with_context(|| format!("no profile named `{name}` in the app config"))?
                .to_path_buf(),
        ),
        (None, None) => match config.default_profile.as_deref() {
            Some(default) => Some(
                config
                    .profile_path(default)
                    .with_context(|| {
                        format!("app config's default_profile `{default}` has no path")
                    })?
                    .to_path_buf(),
            ),
            None => None,
        },
    };

    let Some(path) = path else {
        return Ok(Profile::default());
    };

    let document = ProfileDocument::load(&path)
        .with_context(|| format!("failed to load profile `{}`", path.display()))?;
    for warning in document.warnings() {
        tracing::warn!("{}", warning.message());
    }
    let profile = document.profile().clone();
    profile
        .validate()
        .with_context(|| format!("profile `{}` is invalid", path.display()))?;
    Ok(profile)
}

/// Translate parsed arguments + a loaded profile into [`PipelineOptions`].
///
/// Emits §16.12 item 6's normalisation `WARN` and §16.12 item 4's "no OCR engine in v1"
/// `WARN`; both are the CLI's job, since the pipeline receives already-resolved options.
pub fn build_clean_options(
    args: &CleanArgs,
    profile: Profile,
    image_count: usize,
) -> PipelineOptions {
    let skips = args.skip_flags();
    if skips.implies_more() {
        tracing::warn!(
            "a --skip-* flag implies the earlier steps as well; \
             loading every stage up to the requested one from the cache (spec §16.12 item 6)"
        );
    }
    if profile.preprocessor.ocr_enabled {
        tracing::warn!(
            "ocr_enabled is true but v1 ships no OCR engine (task P7); \
             OCR-based box discarding is inactive (spec §16.12 item 4)"
        );
    }

    let debug_outputs = args.cache_masks || profile.general.always_cache_masks;
    let checkpointing = select_checkpointing(image_count, debug_outputs, args.no_cache);
    let configured_threads = args.threads.unwrap_or(profile.general.max_threads);
    let threads = resolve_threads(configured_threads, image_count);
    let cache_dir = args
        .cache_dir
        .clone()
        .unwrap_or_else(paths::default_cache_dir);

    PipelineOptions {
        profile,
        cache_dir,
        output_dir: args.output_dir.clone(),
        skips: skips.normalized(),
        checkpointing,
        save_only: args.save_only(),
        extract_text: args.extract_text,
        debug_outputs,
        keep_cache: args.keep_cache,
        fail_fast: args.fail_fast,
        threads,
        performing_ocr: false,
    }
}
