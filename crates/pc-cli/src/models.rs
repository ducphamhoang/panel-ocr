//! CLI-owned model-path resolution and download presentation (spec §6 and §8.3).

use crate::args::DetectorSpec;
use crate::paths::Shell;
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
        // spec §16.38 item 1(a), re-verified for that entry against Hugging Face's
        // `X-Linked-Size` header on the pinned revision URL.
        "lama-manga.onnx" => Some(207_482_644),
        _ => None,
    }
}

/// One `models verify` output row. Pure, so the shape of every status line is assertable
/// without running the subcommand (spec §13.1, §16.38 item 19(d)).
pub fn verify_row(name: &str, verification: &pc_models::Verification) -> String {
    let path = verification.path.display();
    let label = verification.status.label();
    match &verification.status {
        pc_models::VerifyStatus::SizeMismatch { actual, expected } => {
            format!("{name}\t{label}\t{path}\tactual={actual}\texpected={expected}")
        }
        pc_models::VerifyStatus::HashMismatch { actual, expected } => {
            format!("{name}\t{label}\t{path}\tactual={actual}\texpected={expected}")
        }
        pc_models::VerifyStatus::Error(message) => {
            format!("{name}\t{label}\t{path}\t{message}")
        }
        pc_models::VerifyStatus::Ok
        | pc_models::VerifyStatus::Missing
        | pc_models::VerifyStatus::NotInstalled => format!("{name}\t{label}\t{path}"),
    }
}

/// The line `models download` prints for an optional model it deliberately did not fetch
/// (spec §16.38 item 19(c) — a skip is announced, never silent).
pub fn skipped_optional_notice(spec: &pc_models::ModelSpec) -> String {
    format!(
        "{}\tSKIPPED (optional)\trun `panel-ocr models download --include-optional` to fetch it",
        spec.name
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_size_matches_the_recorded_detector_signature_and_ocr_pins() {
        // A third, previously unbound literal copy of these byte counts (found during the
        // P7/P8 post-implementation review) — nothing compared them to the recorded
        // signature (detector) or the pc-testkit pins (OCR), so a typo here would only
        // surface as a spurious "size mismatch" on an otherwise-correct file, since the
        // sha256 check that follows in `resolve_managed_model`/`resolve_detector_model`
        // would still catch a genuinely wrong artifact.
        let detector_signature = pc_testkit::model_signature::comic_text_detector_signature();
        assert_eq!(
            expected_size(&pc_models::COMIC_TEXT_DETECTOR),
            Some(detector_signature.size_bytes)
        );

        assert_eq!(
            expected_size(&pc_models::MANGA_OCR_ENCODER),
            Some(pc_testkit::ocr_model_signature::MANGA_OCR_ENCODER.size_bytes)
        );
        assert_eq!(
            expected_size(&pc_models::MANGA_OCR_DECODER),
            Some(pc_testkit::ocr_model_signature::MANGA_OCR_DECODER.size_bytes)
        );
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

/// The `--cache-dir <quoted>` suffix a recovery command needs, or `None` when the bare
/// command is already followable (no resolved root, or the default one).
///
/// Shared by [`models_download_command`] and [`models_download_optional_command`] so the
/// two cannot drift on quoting — which is what the hostile-path test in `x1_args.rs`
/// exists to police for the first of them.
fn cache_dir_suffix(resolved_cache_root: Option<&Path>) -> Option<String> {
    let cache_root = resolved_cache_root?;
    let config = crate::setup::load_app_config().unwrap_or_default();
    let default_cache_root = crate::paths::resolve_cache_root(None, &config);
    (cache_root != default_cache_root)
        .then(|| format!(" --cache-dir {}", Shell::HOST.quote(cache_root)))
}

#[cfg(feature = "onnx")]
#[doc(hidden)]
/// Build the recovery command for the managed model cache.
///
/// The optional path is the resolved cache root used by the failing operation, not
/// merely a CLI override. `None` preserves the bare command for callers without a
/// resolved root.
pub fn models_download_command(resolved_cache_root: Option<&Path>) -> String {
    format!(
        "panel-ocr models download{}",
        cache_dir_suffix(resolved_cache_root).unwrap_or_default()
    )
}

#[doc(hidden)]
/// Build the recovery command for an **optional** managed model — the LaMa inpainting
/// weights are the only one today (spec §16.38 item 19(g)).
///
/// A separate function rather than a parameter on [`models_download_command`], because
/// that one's exact output is pinned by frozen tests and by `ModelError::Unavailable`'s
/// message, and because a required model's refusal must never suggest a flag that fetches
/// 207 MB the user did not ask for.
///
/// Deliberately **not** `#[cfg(feature = "onnx")]`, unlike its sibling: it has no caller
/// until L5 wires the inpainter, and gating it would put its only tests in the tier
/// cookbook rule 6 records as the one that is easy to leave unexecuted.
pub fn models_download_optional_command(resolved_cache_root: Option<&Path>) -> String {
    format!(
        "panel-ocr models download --include-optional{}",
        cache_dir_suffix(resolved_cache_root).unwrap_or_default()
    )
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
