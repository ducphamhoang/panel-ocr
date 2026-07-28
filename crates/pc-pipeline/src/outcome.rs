//! spec §5 — the per-image outcome type, the batch summary, and the exit codes.
//!
//! Fully pinned (§5.1, §5.5, §16.12 items 12 and 18); implemented, not stubbed.

use pc_core::{
    DenoiseAnalytic, DetectAnalytic, MaskFittingAnalytic, OcrAnalytic, StageError, Step,
};
use std::path::PathBuf;

/// Errors crossing the pipeline's stage boundary retain whether they are run-fatal.
#[derive(Debug)]
pub enum PipelineError {
    Stage(StageError),
    RunFatal(StageError),
}

/// A run-fatal condition must be carried by a type, not by a string prefix on a
/// user-facing message — message text is for humans, control flow needs a type.
fn run_fatal_model_message(error: &ImageOutcome) -> Option<String> {
    match error {
        ImageOutcome::RunFatal { error, .. } => Some(error.to_string()),
        _ => None,
    }
}

/// spec §5.5.
pub const EXIT_OK: i32 = 0;
/// spec §5.5 — a fatal condition (§5.3).
pub const EXIT_FATAL: i32 = 1;
/// spec §5.5 — at least one image failed, others succeeded.
pub const EXIT_PARTIAL: i32 = 2;

/// spec §5.6 / §16.12 item 12 — closed set. Everything else is a `Failed`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    UnsupportedFormat { suffix: String },
    NoTextDetected,
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkipReason::UnsupportedFormat { suffix } => {
                write!(f, "unsupported image format `{suffix}`")
            }
            SkipReason::NoTextDetected => write!(f, "no text detected"),
        }
    }
}

/// Everything the analytics printout needs for one image (§2.7).
#[derive(Debug, Clone, Default)]
pub struct ImageAnalytics {
    pub detect: Option<DetectAnalytic>,
    pub ocr: Option<OcrAnalytic>,
    pub mask_fitting: Vec<MaskFittingAnalytic>,
    pub denoise: Option<DenoiseAnalytic>,
}

/// spec §5.1.
#[derive(Debug)]
pub enum ImageOutcome {
    Completed {
        original: PathBuf,
        files_written: Vec<PathBuf>,
        analytics: Box<ImageAnalytics>,
    },
    /// §16.12 item 12: a skipped page may still have exported files (§5.6's
    /// `NoTextDetected` page is exported untouched).
    Skipped {
        original: PathBuf,
        reason: SkipReason,
        files_written: Vec<PathBuf>,
    },
    RunFatal {
        original: PathBuf,
        step: Step,
        error: StageError,
    },
    Failed {
        original: PathBuf,
        step: Step,
        error: StageError,
    },
}

impl ImageOutcome {
    pub fn original(&self) -> &PathBuf {
        match self {
            ImageOutcome::Completed { original, .. }
            | ImageOutcome::Skipped { original, .. }
            | ImageOutcome::RunFatal { original, .. }
            | ImageOutcome::Failed { original, .. } => original,
        }
    }

    pub fn files_written(&self) -> &[PathBuf] {
        match self {
            ImageOutcome::Completed { files_written, .. }
            | ImageOutcome::Skipped { files_written, .. } => files_written,
            ImageOutcome::RunFatal { .. } | ImageOutcome::Failed { .. } => &[],
        }
    }

    pub fn is_failed(&self) -> bool {
        matches!(
            self,
            ImageOutcome::RunFatal { .. } | ImageOutcome::Failed { .. }
        )
    }

    pub(crate) fn is_run_fatal(&self) -> bool {
        matches!(self, ImageOutcome::RunFatal { .. })
    }

    pub fn is_completed(&self) -> bool {
        matches!(self, ImageOutcome::Completed { .. })
    }

    pub fn is_skipped(&self) -> bool {
        matches!(self, ImageOutcome::Skipped { .. })
    }
}

/// The result of a whole batch. `outcomes` is in **input order** regardless of thread
/// count (§5.7 / §16.12 item 17).
#[derive(Debug, Default)]
pub struct BatchSummary {
    pub outcomes: Vec<ImageOutcome>,
}

impl BatchSummary {
    pub fn new(outcomes: Vec<ImageOutcome>) -> Self {
        Self { outcomes }
    }

    pub fn completed(&self) -> usize {
        self.outcomes.iter().filter(|o| o.is_completed()).count()
    }

    pub fn skipped(&self) -> usize {
        self.outcomes.iter().filter(|o| o.is_skipped()).count()
    }

    pub fn failed(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|outcome| outcome.is_failed() && !outcome.is_run_fatal())
            .count()
    }

    /// The first run-fatal error, if one occurred during detection.
    pub fn fatal_model_message(&self) -> Option<String> {
        self.outcomes.iter().find_map(run_fatal_model_message)
    }

    /// Every file any image wrote, in outcome order.
    pub fn files_written(&self) -> Vec<PathBuf> {
        self.outcomes
            .iter()
            .flat_map(|outcome| outcome.files_written().iter().cloned())
            .collect()
    }

    /// spec §5.5 / §16.12 item 18. Fatal conditions abort the run at first detection use;
    /// the provider-declared carve-out in `single.rs` routes them out of per-image reporting.
    pub fn exit_code(&self) -> i32 {
        if self.fatal_model_message().is_some() {
            EXIT_FATAL
        } else if self.failed() > 0 {
            EXIT_PARTIAL
        } else {
            EXIT_OK
        }
    }

    /// spec §5.5's "a summary table is always printed listing failed/skipped images
    /// with reasons".
    pub fn render(&self) -> String {
        use std::fmt::Write as _;

        if let Some(message) = self.fatal_model_message() {
            return format!("fatal: {message}\n");
        }

        let mut text = String::new();
        let _ = writeln!(
            text,
            "{} completed, {} skipped, {} failed",
            self.completed(),
            self.skipped(),
            self.failed()
        );
        for outcome in &self.outcomes {
            match outcome {
                ImageOutcome::Completed { .. } => {}
                ImageOutcome::Skipped {
                    original, reason, ..
                } => {
                    let _ = writeln!(text, "  SKIPPED {}: {reason}", original.display());
                }
                ImageOutcome::RunFatal { .. } => {}
                ImageOutcome::Failed {
                    original,
                    step,
                    error,
                } => {
                    let _ = writeln!(
                        text,
                        "  FAILED  {} at {step:?}: {error}",
                        original.display()
                    );
                }
            }
        }
        text
    }
}

/// spec §5.2 / §16.12 item 18 — the message form a caught panic becomes.
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        format!("panicked: {text}")
    } else if let Some(text) = payload.downcast_ref::<String>() {
        format!("panicked: {text}")
    } else {
        "panicked: <non-string panic payload>".to_string()
    }
}
