//! `pc-pipeline` — the orchestrator (spec §4, §5).
//!
//! Every stage crate is a pure function over typed structs; this crate is what makes
//! them a *pipeline*: it owns all path construction (§1 rule 1), all checkpoint
//! persistence, the two checkpointing modes (§4.1), the skip/resume levels (§4.4),
//! per-image parallelism (§4.5) and the error policy (§5).
//!
//! Module map (§13 rows 24–26, §16.12 item 20):
//!   * `cache`      — `CachePaths` (§4.2) + `discover` (§16.12 item 8)
//!   * `checkpoint` — typed JSON checkpoint read/write with the two validations
//!   * `options`    — `Checkpointing`, `SkipFlags`, requested outputs, thread budget
//!   * `outcome`    — `ImageOutcome`, `BatchSummary`, exit codes (§5.1, §5.5)
//!   * `discovery`  — `<PATHS>...` → a deterministic image list
//!   * `ctx`        — `DetectorProvider` (§16.12 item 3) and the injected OCR factory
//!   * `single`     — the five-stage chain for one image, plus §4.3's `[split?]` branch
//!     above it (G1-chain + §16.14 item 1)
//!   * `batch`      — rayon + `catch_unwind` + `--fail-fast` (G2)
//!   * `strip`      — long-strip split orchestration and merged export (D9/E5)
//!
//! Two v1 realities this crate encodes, both decided in §16.12:
//!   * **there is no working default detector yet** (D1/D4 deferred) — the pipeline is
//!     detector-agnostic via `DetectorProvider`, and `pc-cli` is where `--detector`
//!     turns that into a fatal, well-explained error for `onnx`;
//!   * **there is no OCR engine yet** (P7) — `PipelineCtx.ocr` is `Option`, and `None`
//!     is a supported, warned-about configuration.

pub mod batch;
pub mod cache;
pub mod checkpoint;
pub mod ctx;
pub mod discovery;
pub mod options;
pub mod outcome;
pub mod single;
pub mod strip;

pub use batch::{process_image_isolated, run_batch};
pub use cache::{CachePaths, SEGMENT_INFIX, SPLITS_SUFFIX};
pub use ctx::{DetectorProvider, PipelineCtx, SharedDetector};
pub use discovery::{expand_inputs, input_suffix, is_supported_input, SUPPORTED_INPUT_SUFFIXES};
pub use options::{
    requested_outputs, resolve_threads, select_checkpointing, Checkpointing, PipelineOptions,
    SaveOnly, SkipFlags,
};
pub use outcome::{
    panic_message, BatchSummary, ImageAnalytics, ImageOutcome, SkipReason, EXIT_FATAL, EXIT_OK,
    EXIT_PARTIAL,
};
pub use single::{process_image, process_image_with_splitting, run_stages, ChainOutputs};
pub use strip::{should_split, SplitManifest};
