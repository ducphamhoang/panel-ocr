//! CLI-owned model-path resolution and download presentation (spec §6 and §8.3).

use crate::args::DetectorSpec;
use anyhow::Result;
use pc_core::StageError;
use std::path::{Path, PathBuf};

/// The current registry API carries the digest but not the published byte length.
/// Keep the size check local to the CLI until the registry exposes that metadata.
pub fn expected_size(spec: &pc_models::ModelSpec) -> Option<u64> {
    match spec.file_name {
        "comictextdetector.pt.onnx" => Some(94_669_756),
        "encoder_model.onnx" => Some(343_454_249),
        "decoder_model.onnx" => Some(117_480_262),
        _ => None,
    }
}

/// Resolve the managed model cache using the app-level cache precedence.
pub fn resolve_managed_models_dir(cli_override: Option<&Path>) -> Result<PathBuf> {
    let config = crate::setup::load_app_config()?;
    let cache_dir = crate::paths::resolve_cache_root(cli_override, &config);
    Ok(crate::paths::models_dir(&cache_dir))
}

/// Resolve the model only for an ONNX detector.
///
/// Explicit paths are existence-checked but deliberately not digest-checked. The managed
/// path is resolved from the cache and verified, but never acquired here: provisioning belongs
/// only to the explicit `models download` subcommand.
pub fn resolve_detector_model(
    detector: &DetectorSpec,
    cli_override: Option<&Path>,
    profile_override: Option<&Path>,
    cache_root: &Path,
) -> Result<Option<PathBuf>, StageError> {
    if !matches!(detector, DetectorSpec::Onnx) {
        return Ok(None);
    }

    #[cfg(not(feature = "onnx"))]
    {
        let _ = (cli_override, profile_override, cache_root);
        Ok(None)
    }

    #[cfg(feature = "onnx")]
    {
        let models_dir = crate::paths::models_dir(cache_root);
        if let Some(override_path) = cli_override.or(profile_override) {
            let source = if cli_override.is_some() {
                "--model-path"
            } else {
                "profile [text_detector].model_path"
            };
            let resolution = pc_models::resolve(
                &pc_models::COMIC_TEXT_DETECTOR,
                &models_dir,
                Some(override_path),
            )
            .map_err(|error| match error {
                pc_models::ModelError::OverrideMissing { path } => StageError::Model(format!(
                    "model path `{}` supplied by {source} does not exist",
                    path.display()
                )),
                other => other.into(),
            })?;
            return Ok(Some(resolution.path().to_path_buf()));
        }

        let resolution = pc_models::resolve(&pc_models::COMIC_TEXT_DETECTOR, &models_dir, None)?;
        match resolution {
            pc_models::Resolution::Missing(path) => Err(StageError::Model(format!(
                "managed model is missing at `{}`; run `{}`",
                path.display(),
                models_download_command(Some(cache_root))
            ))),
            pc_models::Resolution::Cached(path) => {
                match pc_models::verify_sha256(&path, pc_models::COMIC_TEXT_DETECTOR.sha256) {
                    Ok(()) => Ok(Some(path)),
                    Err(pc_models::ModelError::HashMismatch {
                        path,
                        expected,
                        actual,
                    }) => Err(StageError::Model(format!(
                        "managed model sha256 mismatch for `{}`: expected {expected}, actual {actual}; run `{}`",
                        path.display(),
                        models_download_command(Some(cache_root))
                    ))),
                    Err(error) => Err(error.into()),
                }
            }
            pc_models::Resolution::Override(_) => {
                unreachable!("managed model resolution never supplies an override")
            }
        }
    }
}

#[cfg(feature = "onnx")]
/// Quote a path using POSIX single-quote rules so a pasted suggestion is safe and
/// round-trips to the original path. Everything inside single quotes is literal;
/// an embedded single quote is represented as '\'' by closing the quote, escaping
/// the literal quote, and reopening it.
fn shell_quote(path: &Path) -> String {
    format!("'{}'", path.display().to_string().replace('\'', "'\\''"))
}

#[cfg(feature = "onnx")]
#[doc(hidden)]
/// Build the recovery command for the managed model cache.
///
/// The optional path is the resolved cache root used by the failing operation, not
/// merely a CLI override. `None` preserves the bare command for callers without a
/// resolved root.
pub fn models_download_command(resolved_cache_root: Option<&Path>) -> String {
    let Some(cache_root) = resolved_cache_root else {
        return "panel-ocr models download".to_owned();
    };

    let config = crate::setup::load_app_config().unwrap_or_default();
    let default_cache_root = crate::paths::resolve_cache_root(None, &config);
    if cache_root == default_cache_root {
        "panel-ocr models download".to_owned()
    } else {
        format!(
            "panel-ocr models download --cache-dir {}",
            shell_quote(cache_root)
        )
    }
}

pub fn progress_sink() -> Box<dyn pc_models::ProgressSink> {
    use std::io::{self, IsTerminal};

    if io::stderr().is_terminal() {
        Box::new(IndicatifProgress::default())
    } else {
        Box::new(pc_models::NoProgress)
    }
}

#[derive(Default)]
struct IndicatifProgress {
    bar: Option<indicatif::ProgressBar>,
}

impl pc_models::ProgressSink for IndicatifProgress {
    fn start(&mut self, total: Option<u64>) {
        let bar = total
            .map(indicatif::ProgressBar::new)
            .unwrap_or_else(indicatif::ProgressBar::new_spinner);
        bar.set_message("downloading ONNX model");
        self.bar = Some(bar);
    }

    fn advance(&mut self, bytes: u64) {
        if let Some(bar) = &self.bar {
            bar.inc(bytes);
        }
    }

    fn finish(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_with_message("ONNX model ready");
        }
    }
}
