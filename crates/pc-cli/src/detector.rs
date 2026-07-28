//! `--detector` → a [`DetectorProvider`] (spec §16.12 items 2 and 3).
//!
//! The default build keeps ONNX opt-in. With the feature enabled, model resolution and
//! ONNX session construction are deferred until the first image actually needs detection.
//!
//! Fully pinned; implemented, not stubbed.

use crate::args::DetectorSpec;
use pc_core::StageError;
use pc_detect::{MockDetector, ReplayDetector, TextDetector};
use pc_pipeline::{DetectorProvider, SharedDetector};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(feature = "onnx")]
use std::sync::{Mutex, OnceLock};

/// The message emitted when the default build has no ONNX backend (§5.3 → exit 1).
#[cfg(not(feature = "onnx"))]
pub const ONNX_UNAVAILABLE: &str = "the ONNX text detector is not available in this build \
(pc-cli was compiled without the `onnx` feature, so D1 model provisioning and D4 `ort` \
session are not linked in). Rebuild with `--features onnx`, or pass \
`--detector replay:<DIR>` to run against recorded detector fixtures, or \
`--detector mock` for a no-text pass-through run.";

/// spec §7.2.1 — one stem-bound `ReplayDetector` per image.
pub struct ReplayProvider {
    dir: PathBuf,
}

#[cfg(feature = "onnx")]
#[cfg(test)]
type InitHook = Box<dyn Fn() -> Result<Arc<dyn TextDetector>, String> + Send + Sync>;

#[cfg(feature = "onnx")]
// DEVIATION(16): upstream constructs `TextDetector(...)` once before each per-image loop
// (`pcleaner/ctd_interface.py::process_image_batch` and its single-process path), with no
// `try/except`, so construction failure ends the run. v1 constructs lazily so §4.4 resume
// runs with cached `#raw.json` do not resolve a detector that stage 1 never reads (§16.19
// item 1; there is no `--resume` flag, §16.12 item 7), then latches the one `detector_for`
// attempt's `outcome` — session or rendered refusal, including an init panic — for all rayon workers; without it,
// each page would re-resolve, re-hash ~90 MB and rebuild a session. The latch also makes
// `fatal_model_message`'s first run-fatal in input order byte-identical: without the latch,
// two workers could render different text for the same cause and make stdout scheduling-
// dependent (§5.7), while the latch preserves upstream's once-per-run guarantee.
// This is not an in-run retry: upstream unlinks a hash-mismatched file and tells the user to
// download it manually; here retry means re-running `panel-ocr models download` (§16.19 items
// 2-3, §5.3, `crates/pc-models/tests/d1_install.rs`). Latch the message, not `StageError`,
// because `StageError` is deliberately not `Clone`.
struct OnnxProvider {
    cli_override: Option<PathBuf>,
    profile_override: Option<PathBuf>,
    cache_root: PathBuf,
    outcome: OnceLock<Result<Arc<dyn TextDetector>, String>>,
    initializing: Mutex<()>,
    // Deliberate test instrumentation proving failed initialization is attempted once.
    #[cfg(test)]
    attempts: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    init_hook: Option<InitHook>,
}

#[cfg(feature = "onnx")]
impl OnnxProvider {
    fn new(
        cli_override: Option<&Path>,
        profile_override: Option<&Path>,
        cache_root: &Path,
    ) -> Self {
        Self {
            cli_override: cli_override.map(Path::to_path_buf),
            profile_override: profile_override.map(Path::to_path_buf),
            cache_root: cache_root.to_path_buf(),
            outcome: OnceLock::new(),
            initializing: Mutex::new(()),
            #[cfg(test)]
            attempts: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            init_hook: None,
        }
    }

    fn initialize_detector(&self) -> Result<Arc<dyn TextDetector>, String> {
        #[cfg(test)]
        {
            self.attempts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if let Some(init_hook) = &self.init_hook {
                return init_hook();
            }
        }

        // Keep this order: a model refusal must be actionable even when the runtime is absent.
        // All provisioning refusals stay Model errors so the pipeline treats them as run-fatal.
        let model_path = crate::models::resolve_detector_model(
            &DetectorSpec::Onnx,
            self.cli_override.as_deref(),
            self.profile_override.as_deref(),
            &self.cache_root,
        )
        .map_err(|error| match error {
            StageError::Model(message) => message,
            other => other.to_string(),
        })?
        .ok_or_else(|| "ONNX detector model path was not resolved".to_string())?;
        pc_detect::onnx::ensure_model_file(&model_path).map_err(|error| error.to_string())?;
        if !pc_detect::onnx::runtime_available() {
            return Err("ONNX Runtime is not loadable in this environment".to_string());
        }
        let detector = pc_detect::onnx::OnnxDetector::from_path(&model_path)
            .map_err(|error| error.to_string())?;
        Ok(Arc::new(detector))
    }
}

#[cfg(feature = "onnx")]
impl DetectorProvider for OnnxProvider {
    fn detector_for(&self, _original: &Path) -> Result<Arc<dyn TextDetector>, StageError> {
        if let Some(outcome) = self.outcome.get() {
            return outcome
                .as_ref()
                .map(Arc::clone)
                .map_err(|message| StageError::Model(message.clone()));
        }

        let _initializing = self
            .initializing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(outcome) = self.outcome.get() {
            return outcome
                .as_ref()
                .map(Arc::clone)
                .map_err(|message| StageError::Model(message.clone()));
        }

        // initialize_detector takes no image, so an init panic is independent of the original
        // by construction. §16.19 item 5(b)'s causal criterion classifies it run-fatal, and
        // §16.12 item 2's "neither outcome is retried within a run" binds the panicking attempt
        // as much as the erroring one. §5.2's boundary is not relocated: process_image_isolated
        // still owns panic conversion for every image-dependent unit; only the image-independent
        // init unwind is converted here.
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.initialize_detector()))
                .unwrap_or_else(|payload| Err(pc_pipeline::panic_message(payload.as_ref())));
        match outcome {
            Ok(detector) => {
                let _ = self.outcome.set(Ok(Arc::clone(&detector)));
                Ok(detector)
            }
            Err(message) => {
                let _ = self.outcome.set(Err(message.clone()));
                Err(StageError::Model(message))
            }
        }
    }

    fn failures_are_run_fatal(&self) -> bool {
        true
    }
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Barrier};

    #[test]
    fn failed_initialization_is_latched_across_threads() {
        for _ in 0..25 {
            let cache = tempfile::tempdir().unwrap();
            let provider = Arc::new(OnnxProvider::new(None, None, cache.path()));
            let barrier = Arc::new(Barrier::new(4));

            let results = std::thread::scope(|scope| {
                let workers = (0..4)
                    .map(|_| {
                        let barrier = Arc::clone(&barrier);
                        let provider = Arc::clone(&provider);
                        scope.spawn(move || {
                            barrier.wait();
                            (0..2)
                                .map(|_| provider.detector_for(Path::new("image.png")))
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect::<Vec<_>>();

                workers
                    .into_iter()
                    .flat_map(|worker| worker.join().unwrap())
                    .collect::<Vec<_>>()
            });

            assert_eq!(provider.attempts.load(Ordering::SeqCst), 1);
            assert_eq!(results.len(), 8);
            let messages = results
                .into_iter()
                .map(|result| match result {
                    Err(StageError::Model(message)) => message,
                    Err(error) => panic!("expected model error, got {error:?}"),
                    Ok(_) => panic!("expected initialization to fail"),
                })
                .collect::<Vec<_>>();
            assert!(messages.windows(2).all(|pair| pair[0] == pair[1]));
        }
    }

    #[test]
    fn panicking_initialization_is_latched_across_threads() {
        for _ in 0..25 {
            let cache = tempfile::tempdir().unwrap();
            let mut provider = OnnxProvider::new(None, None, cache.path());
            provider.init_hook = Some(Box::new(|| panic!("initialization exploded")));
            let provider = Arc::new(provider);
            let barrier = Arc::new(Barrier::new(4));

            let results = std::thread::scope(|scope| {
                let workers = (0..4)
                    .map(|_| {
                        let barrier = Arc::clone(&barrier);
                        let provider = Arc::clone(&provider);
                        scope.spawn(move || {
                            barrier.wait();
                            (0..2)
                                .map(|_| {
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        provider.detector_for(Path::new("image.png"))
                                    }))
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect::<Vec<_>>();

                workers
                    .into_iter()
                    .flat_map(|worker| worker.join().unwrap())
                    .collect::<Vec<_>>()
            });

            assert_eq!(provider.attempts.load(Ordering::SeqCst), 1);
            assert_eq!(results.len(), 8);
            let messages = results
                .into_iter()
                .map(|result| match result {
                    Ok(Err(StageError::Model(message))) => message,
                    Ok(Err(error)) => panic!("expected model error, got {error:?}"),
                    Ok(Ok(_)) => panic!("expected initialization to fail"),
                    Err(payload) => panic!(
                        "detector_for unexpectedly unwound: {}",
                        pc_pipeline::panic_message(payload.as_ref())
                    ),
                })
                .collect::<Vec<_>>();
            assert!(messages.windows(2).all(|pair| pair[0] == pair[1]));
            assert!(messages[0].contains("panicked: "));
        }
    }
}

impl ReplayProvider {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
}

impl DetectorProvider for ReplayProvider {
    fn detector_for(&self, original: &Path) -> Result<Arc<dyn TextDetector>, StageError> {
        let stem = original
            .file_stem()
            .ok_or_else(|| {
                StageError::InvalidInput(format!(
                    "input path `{}` has no file stem",
                    original.display()
                ))
            })?
            .to_string_lossy()
            .into_owned();

        // `ReplayDetector::new` panics on a missing fixture (§7.2.1's "fail at
        // construction"). At the pipeline boundary a missing fixture must be a
        // per-image error instead, so probe first. The probe converts the constructor panic
        // to an error, and `failures_are_run_fatal() == false` keeps it per-image.
        let mask = pc_detect::mock::detector_mask_path(&self.dir, &stem);
        let blocks = pc_detect::mock::detector_blocks_path(&self.dir, &stem);
        for path in [&mask, &blocks] {
            if !path.exists() {
                return Err(StageError::Model(format!(
                    "no replay fixture `{}`; run `cargo xtask record-fixtures`",
                    path.display()
                )));
            }
        }
        Ok(Arc::new(ReplayDetector::new(&self.dir, &stem)))
    }
}

/// Build the provider a run will use. `Err` here is fatal (§5.3).
pub fn build_provider(
    spec: &DetectorSpec,
    cli_override: Option<&Path>,
    profile_override: Option<&Path>,
    cache_root: &Path,
) -> Result<Box<dyn DetectorProvider>, StageError> {
    match spec {
        DetectorSpec::Onnx => {
            #[cfg(not(feature = "onnx"))]
            {
                let _ = (cli_override, profile_override, cache_root);
                Err(StageError::Model(ONNX_UNAVAILABLE.to_string()))
            }
            #[cfg(feature = "onnx")]
            {
                Ok(Box::new(OnnxProvider::new(
                    cli_override,
                    profile_override,
                    cache_root,
                )))
            }
        }
        DetectorSpec::Mock => Ok(Box::new(SharedDetector::new(Arc::new(MockDetector::new())))),
        DetectorSpec::Replay(dir) => Ok(Box::new(ReplayProvider::new(dir.clone()))),
    }
}
