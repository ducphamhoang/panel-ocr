//! `--ocr-enabled`'s effect: constructing an `OcrEngineFactory` (spec §16.30/§16.31).

use pc_config::Profile;
use pc_core::StageError;
use std::path::Path;
#[cfg(feature = "onnx")]
use std::path::PathBuf;

#[cfg(not(feature = "onnx"))]
pub const ONNX_UNAVAILABLE: &str = "the manga-ocr engine is not available in this build \
(pc-cli was compiled without the `onnx` feature, so the OCR model provisioning and ONNX \
session are not linked in). Rebuild with `--features onnx`, or disable OCR \
(`ocr_enabled = false` in the profile) to run without it.";

/// Build the OCR engine factory. `Err` here is fatal (§5.3) -- construction is eager and
/// image-independent, matching upstream's `MangaOcr()` (constructed once, up front, with
/// no per-image lazy resolution -- unlike the detector, `OcrEngineFactory::engine_for`
/// takes no image path, so there is nothing to bind lazily).
pub fn build_factory(cache_root: &Path) -> Result<Box<dyn pc_ocr::OcrEngineFactory>, StageError> {
    #[cfg(not(feature = "onnx"))]
    {
        let _ = cache_root;
        Err(StageError::Model(ONNX_UNAVAILABLE.to_string()))
    }
    #[cfg(feature = "onnx")]
    {
        let models_dir = crate::paths::models_dir(cache_root);
        let encoder_path =
            resolve_managed_model(&pc_models::MANGA_OCR_ENCODER, &models_dir, cache_root)?;
        let decoder_path =
            resolve_managed_model(&pc_models::MANGA_OCR_DECODER, &models_dir, cache_root)?;
        let engine = pc_ocr::manga::MangaOcrEngine::from_paths(&encoder_path, &decoder_path)?;
        Ok(Box::new(pc_ocr::manga::MangaOcrFactory::new(engine)))
    }
}

#[cfg(feature = "onnx")]
fn resolve_managed_model(
    spec: &pc_models::ModelSpec,
    models_dir: &Path,
    cache_root: &Path,
) -> Result<PathBuf, StageError> {
    let resolution = pc_models::resolve(spec, models_dir, None)?;
    match resolution {
        pc_models::Resolution::Missing(path) => Err(StageError::Model(format!(
            "managed model is missing at `{}`; run `{}`",
            path.display(),
            crate::models::models_download_command(Some(cache_root))
        ))),
        pc_models::Resolution::Cached(path) => {
            match pc_models::verify_sha256(&path, spec.sha256) {
                Ok(()) => Ok(path),
                Err(pc_models::ModelError::HashMismatch {
                    path,
                    expected,
                    actual,
                }) => Err(StageError::Model(format!(
                    "managed model sha256 mismatch for `{}`: expected {expected}, actual {actual}; run `{}`",
                    path.display(),
                    crate::models::models_download_command(Some(cache_root))
                ))),
                Err(error) => Err(error.into()),
            }
        }
        pc_models::Resolution::Override(_) => {
            unreachable!("managed model resolution never supplies an override")
        }
    }
}

/// Apply the upstream `ocr` report-path overrides from §15.5.
pub(crate) fn apply_report_overrides(profile: &mut Profile) {
    profile.preprocessor.ocr_blacklist_pattern = ".*".to_string();
    profile.preprocessor.ocr_max_size = 10_000_000_000;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn build_factory_reports_the_verbatim_onnx_unavailable_message() {
        let cache = tempfile::tempdir().unwrap();

        let error = match build_factory(cache.path()) {
            Ok(_) => panic!("the backend is not linked"),
            Err(error) => error,
        };
        match error {
            StageError::Model(message) => assert_eq!(message, ONNX_UNAVAILABLE),
            other => panic!("expected a model error, got {other:?}"),
        }
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn missing_managed_model_names_the_download_command() {
        let cache = tempfile::tempdir().unwrap();

        let error = match build_factory(cache.path()) {
            Ok(_) => panic!("the managed models are absent"),
            Err(error) => error,
        };
        match error {
            StageError::Model(message) => {
                assert!(message.contains("managed model is missing"), "{message}");
                assert!(message.contains("panel-ocr models download"), "{message}");
            }
            other => panic!("expected a model error, got {other:?}"),
        }
    }

    #[test]
    fn report_profile_overrides_disable_report_filtering() {
        let mut profile = Profile::default();

        apply_report_overrides(&mut profile);

        assert_eq!(profile.preprocessor.ocr_blacklist_pattern, ".*");
        assert_eq!(profile.preprocessor.ocr_max_size, 10_000_000_000);
    }
}
