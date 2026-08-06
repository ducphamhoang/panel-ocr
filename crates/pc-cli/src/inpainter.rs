//! The inpainting provider: `[inpainter].inpainting_enabled` → a lazily built, latched
//! [`pc_inpaint::Inpainter`] (spec §16.38 items 8, 9, 19(b) and 19(g)).
//!
//! **This file is `DEVIATION(27)`'s declared site** (§14 item 27, §16.38 item 8): *"The
//! implementation site is the inpainter provider in `crates/pc-cli/` and must carry
//! `DEVIATION(27)`."*
//!
//! **Scope, so an absence is not read as an oversight.** Task L5 lands the provider and its
//! latch. It does **not** wire anything into `pc-pipeline`: `Step::Inpaint`, the cache
//! suffixes, `ExportSources` and `--skip-inpaint` are all L6 (§16.38 item 16(e)), so nothing
//! here has a caller yet — the same "added now, unwired" shape item 19(g) used for
//! [`crate::models::models_download_optional_command`], which this module is the first caller
//! of.

use pc_core::device::Device;
use pc_core::StageError;
use pc_inpaint::Inpainter;
use std::path::Path;
#[cfg(feature = "onnx")]
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(feature = "onnx")]
use std::sync::{Mutex, OnceLock};

/// The message emitted when the default build has no ONNX backend, mirroring
/// [`crate::detector::ONNX_UNAVAILABLE`]'s shape.
///
/// §16.38 item 13(c) is why this is a *stage* refusal and not a config error:
/// *"`inpainting_enabled = true` must load and validate successfully even in a build with no
/// ONNX, exactly as §16.36 item 6 ruled for `device = "cuda"`: config accepts, the stage
/// refuses."*
pub const INPAINT_ONNX_UNAVAILABLE: &str = "LaMa inpainting is not available in this build \
(pc-cli was compiled without the `onnx` feature, so the `ort` session behind \
`[inpainter].inpainting_enabled` is not linked in). Rebuild with `--features onnx`, or set \
`inpainting_enabled = false` under `[inpainter]` in your profile.";

/// A source of the run's single [`Inpainter`].
///
/// Deliberately a trait even though L5 has one real implementation and one refusing one: L6
/// hands this to the pipeline, and the boundary is what keeps `--skip-inpaint` and the
/// no-`onnx` build from needing two different call sites.
pub trait InpainterProvider: Send + Sync {
    /// The run's inpainter, built at most once (§16.38 item 8(d)).
    fn inpainter(&self) -> Result<Arc<dyn Inpainter>, StageError>;

    /// **Fatality is declared, not inferred** (cookbook rule 4; §16.38 item 9(g) declares
    /// both halves of this stage's split).
    ///
    /// `true` for every implementation here, and the reason is item 9(c) rather than item
    /// 9(b): construction takes no image, which only *licences* the classification, while
    /// what chooses it is that `Step::Inpaint` sits before `Step::Export`, so a per-image
    /// `Failed { step: Inpaint }` suppresses that page's export anyway — *"The outcome is the
    /// same zero exported files as run-fatal, at the cost of N error lines instead of one, N
    /// provisioning attempts instead of one, and exit 2 instead of 1."*
    ///
    /// This says nothing about failures **inside** [`Inpainter::inpaint_tile`], which receive
    /// the image and are per-image `StageError::Inference` (item 9(g)).
    fn failures_are_run_fatal(&self) -> bool;
}

/// The provider for a build without the `onnx` feature, and for a run whose model could never
/// be reached. Refuses on first use rather than at construction, so a run that never reaches
/// an eligible region never sees the message (§16.38 item 8(c)).
pub struct UnavailableInpainterProvider {
    message: String,
}

impl UnavailableInpainterProvider {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl InpainterProvider for UnavailableInpainterProvider {
    fn inpainter(&self) -> Result<Arc<dyn Inpainter>, StageError> {
        Err(StageError::Model(self.message.clone()))
    }

    fn failures_are_run_fatal(&self) -> bool {
        true
    }
}

#[cfg(feature = "onnx")]
#[cfg(test)]
type InitHook = Box<dyn Fn() -> Result<Arc<dyn Inpainter>, String> + Send + Sync>;

// DEVIATION(27): upstream constructs its `InpaintingModel` once, conditional on
// `inpainting_enabled`, BEFORE its per-page loop — `pcleaner/main.py:390-392` sets the skip
// flag, `:625-626` calls `md.ensure_inpainting_available(config)` and constructs inside the
// `if not skip_inpainting:` block at `:605`, and the page loop is at `:642`. So upstream is
// conditional on the flag and EAGER with respect to the loop. v1.5 defers construction to the
// first page that has an eligible region and latches that single attempt's success or rendered
// refusal for the whole run.
//
// The config flag alone would NOT force this — it forces conditionality, and §16.38 item 8(b)
// says so explicitly: an eager-but-flag-gated construction already avoids the 207 MB download
// for every default run, since `inpainting_enabled` defaults to false. What forces laziness is
// item 8(c): a batch may have `inpainting_enabled = true` and ZERO eligible regions on every
// page — upstream's own `if boxes_to_inpaint:` guard at `inpainting.py:131` shows the empty
// case is expected rather than pathological — and such a run must not pay a 207 MB download
// plus a ~6.3 s session build (item 1(f)) to inpaint nothing. §16.19 item 1's ground for the
// detector ("a run whose `#raw.json` is cached never executes stage 1") transfers as well.
//
// This is a SEPARATE register item from DEVIATION(16) and not a widening of it: item 16's text
// is scoped to the detector, and §16.38 item 8(e) declines to widen it, because widening a
// ratified claim past the subject it names is a separate, argued step.
//
// The latch is §16.19 item 2's mechanism unchanged — one `OnceLock<Result<_, String>>` behind
// a double-checked `Mutex<()>`, storing the rendered MESSAGE rather than the error because
// `StageError` is deliberately not `Clone`. Item 2(a)'s cost argument applies at 207 MB, and
// 2(b)'s determinism argument applies verbatim: without the latch two rayon workers can render
// different text for one cause and stdout becomes scheduling-dependent, violating §5.7.
#[cfg(feature = "onnx")]
pub struct OnnxInpainterProvider {
    cli_override: Option<PathBuf>,
    cache_root: PathBuf,
    device: Device,
    outcome: OnceLock<Result<Arc<dyn Inpainter>, String>>,
    initializing: Mutex<()>,
    // Instrumentation proving the single attempt is a single attempt, not a claim in a
    // comment. §16.38 item 8(d)'s latch is the assertion, and a counter is the only way to
    // observe it.
    #[cfg(test)]
    attempts: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    init_hook: Option<InitHook>,
}

#[cfg(feature = "onnx")]
impl OnnxInpainterProvider {
    /// `cli_override`, when present, bypasses the managed cache and its digest check — the
    /// same latitude `--model-path` already has for the detector
    /// (`crate::models::resolve_detector_model`: *"Explicit paths are existence-checked but
    /// deliberately not digest-checked"*).
    pub fn new(cli_override: Option<&Path>, cache_root: &Path, device: Device) -> Self {
        Self {
            cli_override: cli_override.map(Path::to_path_buf),
            cache_root: cache_root.to_path_buf(),
            device,
            outcome: OnceLock::new(),
            initializing: Mutex::new(()),
            #[cfg(test)]
            attempts: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            init_hook: None,
        }
    }

    /// Resolve the optional LaMa artifact and **verify its digest**, then build the session.
    fn initialize(&self) -> Result<Arc<dyn Inpainter>, String> {
        #[cfg(test)]
        {
            self.attempts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if let Some(init_hook) = &self.init_hook {
                return init_hook();
            }
        }

        let model_path = self.resolve_model()?;
        pc_inpaint::onnx::ensure_model_file(&model_path).map_err(|error| error.to_string())?;
        if !pc_inpaint::onnx::runtime_available() {
            return Err("ONNX Runtime is not loadable in this environment".to_owned());
        }
        let inpainter =
            pc_inpaint::onnx::OnnxInpainter::from_path_for_device(&model_path, self.device)
                .map_err(|error| match error {
                    StageError::Model(message) => message,
                    other => other.to_string(),
                })?;
        Ok(Arc::new(inpainter))
    }

    /// §16.38 item 19(b)'s RULING, quoted because it is the reason this function hashes 207 MB
    /// instead of trusting the last `models verify`: *"L5's inpainter runtime path must
    /// independently verify the LaMa artifact's integrity before use, regardless of what
    /// `models verify` last reported, so a corrupt optional model yields a run-fatal refusal at
    /// the stage rather than corrupted inpainting"*. That is what makes the same item's
    /// accepted tradeoff — a plain `models verify` not reporting an `Optional` row at all —
    /// low-severity, so this check is load-bearing for a ratified decision and not belt-and-
    /// braces.
    ///
    /// A **missing** artifact names `--include-optional` (item 19(g)); a **corrupt** one does
    /// not get to be optional at all (item 19(d): *"optionality licenses **absence only, never
    /// corruption**"*).
    fn resolve_model(&self) -> Result<PathBuf, String> {
        let models_dir = crate::paths::models_dir(&self.cache_root);
        let spec = &pc_models::LAMA_MANGA_INPAINTER;
        let resolution = pc_models::resolve(spec, &models_dir, self.cli_override.as_deref())
            .map_err(|error| match error {
                // Deliberately names no CLI flag: there is no `--inpaint-model-path` yet, and
                // §16.38 item 16(e) leaves the flag surface to L6. Naming a flag that does not
                // exist would be worse than naming none.
                pc_models::ModelError::OverrideMissing { path } => format!(
                    "the explicit inpainting model path `{}` does not exist",
                    path.display()
                ),
                other => other.to_string(),
            })?;

        match resolution {
            // An explicit path is the user's own artifact; existence-checked, not hashed.
            pc_models::Resolution::Override(path) => Ok(path),
            pc_models::Resolution::Missing(path) => Err(format!(
                "the optional LaMa inpainting model is missing at `{}`; run `{}`",
                path.display(),
                crate::models::models_download_optional_command(Some(&self.cache_root))
            )),
            pc_models::Resolution::Cached(path) => {
                match pc_models::verify_sha256(&path, spec.sha256) {
                    Ok(()) => Ok(path),
                    Err(pc_models::ModelError::HashMismatch {
                        path,
                        expected,
                        actual,
                    }) => Err(format!(
                        "inpainting model sha256 mismatch for `{}`: expected {expected}, \
                         actual {actual}; run `{}`",
                        path.display(),
                        crate::models::models_download_optional_command(Some(&self.cache_root))
                    )),
                    Err(error) => Err(error.to_string()),
                }
            }
        }
    }
}

#[cfg(feature = "onnx")]
impl InpainterProvider for OnnxInpainterProvider {
    fn inpainter(&self) -> Result<Arc<dyn Inpainter>, StageError> {
        if let Some(outcome) = self.outcome.get() {
            return latched(outcome);
        }

        let _initializing = self
            .initializing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(outcome) = self.outcome.get() {
            return latched(outcome);
        }

        // `initialize` takes no image, so an init panic is image-independent by construction;
        // §16.19 item 5(b)'s criterion classifies it run-fatal and §16.12 item 2's "neither
        // outcome is retried within a run" binds the panicking attempt as much as the erroring
        // one. Same treatment as `crate::detector::OnnxProvider`; §5.2's per-image panic
        // boundary is not relocated by it.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.initialize()))
            .unwrap_or_else(|payload| Err(pc_pipeline::panic_message(payload.as_ref())));
        match outcome {
            Ok(inpainter) => {
                let _ = self.outcome.set(Ok(Arc::clone(&inpainter)));
                Ok(inpainter)
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

#[cfg(feature = "onnx")]
fn latched(outcome: &Result<Arc<dyn Inpainter>, String>) -> Result<Arc<dyn Inpainter>, StageError> {
    outcome
        .as_ref()
        .map(Arc::clone)
        .map_err(|message| StageError::Model(message.clone()))
}

/// Build the provider a run will use.
///
/// `enabled` is `[inpainter].inpainting_enabled` (§16.38 item 13(a), default `false`) already
/// combined with whatever L6's `--skip-inpaint` decides. With it false the caller has no
/// inpainter and never asks for one, which is why this returns `None` rather than a provider
/// that refuses: a refusing provider would be indistinguishable from the no-`onnx` build.
pub fn build_provider(
    enabled: bool,
    cli_override: Option<&Path>,
    cache_root: &Path,
    device: Device,
) -> Option<Box<dyn InpainterProvider>> {
    if !enabled {
        return None;
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (cli_override, cache_root, device);
        Some(Box::new(UnavailableInpainterProvider::new(
            INPAINT_ONNX_UNAVAILABLE,
        )))
    }
    #[cfg(feature = "onnx")]
    {
        Some(Box::new(OnnxInpainterProvider::new(
            cli_override,
            cache_root,
            device,
        )))
    }
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::sync::Barrier;

    /// §16.38 item 8(d): *"One `OnceLock<Result<…, String>>` behind a double-checked
    /// `Mutex<()>`, storing the rendered message rather than the error"*, and §16.19 item
    /// 2(b)'s determinism ground: *"without the latch, two workers can render different text
    /// for one cause and stdout becomes scheduling-dependent, violating §5.7."*
    ///
    /// The attempt **counter** is the assertion. "Trust the code" cannot distinguish a latch
    /// from an unsynchronised `OnceLock::get_or_init` race, and 25 repetitions with a barrier
    /// is what the equivalent detector test uses.
    #[test]
    fn a_failing_initialization_is_attempted_exactly_once_and_every_caller_gets_that_message() {
        for _ in 0..25 {
            let cache = tempfile::tempdir().unwrap();
            // No model in the cache, so initialization fails for a real reason.
            let provider = Arc::new(OnnxInpainterProvider::new(None, cache.path(), Device::Cpu));
            let barrier = Arc::new(Barrier::new(4));

            let results = std::thread::scope(|scope| {
                let workers = (0..4)
                    .map(|_| {
                        let barrier = Arc::clone(&barrier);
                        let provider = Arc::clone(&provider);
                        scope.spawn(move || {
                            barrier.wait();
                            (0..2).map(|_| provider.inpainter()).collect::<Vec<_>>()
                        })
                    })
                    .collect::<Vec<_>>();
                workers
                    .into_iter()
                    .flat_map(|worker| worker.join().unwrap())
                    .collect::<Vec<_>>()
            });

            assert_eq!(
                provider.attempts.load(Ordering::SeqCst),
                1,
                "eight calls across four threads, one provisioning attempt"
            );
            assert_eq!(results.len(), 8);
            let messages = results
                .into_iter()
                .map(|result| match result {
                    Err(StageError::Model(message)) => message,
                    Err(error) => panic!("expected a run-fatal Model error, got {error:?}"),
                    Ok(_) => panic!("expected initialization to fail with no model present"),
                })
                .collect::<Vec<_>>();
            assert!(
                messages.windows(2).all(|pair| pair[0] == pair[1]),
                "every caller must get byte-identical text"
            );
        }
    }

    /// The success half of the latch, which the failure test above cannot cover: a session is
    /// expensive (§16.38 item 1(f) measured ~6.3 s), so a second page must reuse the first
    /// page's `Arc` rather than build again.
    ///
    /// The hook stands in for the real session so this runs without the 207 MB artifact. What
    /// it proves is the latch, which is provider logic and has nothing to do with `ort`.
    #[test]
    fn a_successful_initialization_builds_once_and_hands_the_same_arc_to_every_caller() {
        let cache = tempfile::tempdir().unwrap();
        let mut provider = OnnxInpainterProvider::new(None, cache.path(), Device::Cpu);
        provider.init_hook = Some(Box::new(|| {
            Ok(Arc::new(pc_inpaint::stub::StubInpainter::flat(1, 2, 3)) as Arc<dyn Inpainter>)
        }));
        let provider = Arc::new(provider);
        let barrier = Arc::new(Barrier::new(4));

        let handles = std::thread::scope(|scope| {
            let workers = (0..4)
                .map(|_| {
                    let barrier = Arc::clone(&barrier);
                    let provider = Arc::clone(&provider);
                    scope.spawn(move || {
                        barrier.wait();
                        (0..2)
                            .map(|_| provider.inpainter().expect("the hook succeeds"))
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
        assert_eq!(handles.len(), 8);
        // Pointer identity, not equality: eight `Arc`s to ONE session, which is what "built
        // once" means. Eight equal-but-distinct sessions would satisfy a weaker assertion.
        let first = Arc::as_ptr(&handles[0]);
        assert!(
            handles
                .iter()
                .all(|handle| std::ptr::addr_eq(Arc::as_ptr(handle), first)),
            "every caller must receive the same session"
        );
    }

    /// §16.38 item 8(c)'s ground, made observable: *"a batch may have `inpainting_enabled =
    /// true` and **zero eligible regions on every page** … and such a run must not pay a 207 MB
    /// download plus a ~6.3 s session build (item 1(f)) to inpaint nothing."*
    ///
    /// Constructing the provider must therefore touch nothing. Asserted through the attempt
    /// counter, because "no session was built" is otherwise invisible.
    #[test]
    fn constructing_the_provider_provisions_nothing_until_a_page_actually_asks() {
        let cache = tempfile::tempdir().unwrap();
        let provider = OnnxInpainterProvider::new(None, cache.path(), Device::Cpu);

        assert_eq!(
            provider.attempts.load(Ordering::SeqCst),
            0,
            "eager construction would defeat item 8(c)"
        );
        // And the flag being off yields no provider at all, so a default run cannot even ask.
        assert!(
            build_provider(false, None, cache.path(), Device::Cpu).is_none(),
            "`inpainting_enabled = false` (the §16.38 item 13(a) default) means no provider"
        );
        assert!(build_provider(true, None, cache.path(), Device::Cpu).is_some());
    }

    /// A panicking initialization is latched with the same single-attempt guarantee, so a
    /// panic cannot become a per-page retry. Same shape as
    /// `detector::tests::panicking_initialization_is_latched_across_threads`.
    #[test]
    fn a_panicking_initialization_is_also_attempted_once_and_rendered_identically() {
        let cache = tempfile::tempdir().unwrap();
        let mut provider = OnnxInpainterProvider::new(None, cache.path(), Device::Cpu);
        provider.init_hook = Some(Box::new(|| panic!("session build exploded")));
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
                                    provider.inpainter()
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
                Ok(Err(error)) => panic!("expected a Model error, got {error:?}"),
                Ok(Ok(_)) => panic!("expected initialization to fail"),
                Err(payload) => panic!(
                    "inpainter() must not unwind: {}",
                    pc_pipeline::panic_message(payload.as_ref())
                ),
            })
            .collect::<Vec<_>>();
        assert!(messages.windows(2).all(|pair| pair[0] == pair[1]));
        assert!(messages[0].contains("panicked: "), "got {}", messages[0]);
    }
}
