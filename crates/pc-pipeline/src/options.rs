//! spec §4.1/§4.4/§4.5 + §16.12 items 5, 6, 14–17 — everything the orchestrator needs
//! to know about *how* to run, resolved once per invocation.
//!
//! Fully pinned; implemented, not stubbed.

use pc_config::Profile;
use pc_core::{Output, Step};
use std::path::PathBuf;

/// spec §4.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Checkpointing {
    /// Persist every intermediate to the cache dir; enables `--skip-*` and the debug
    /// outputs. The default (§4.1).
    #[default]
    Disk,
    /// Keep intermediates in memory; write only the requested final exports. Debug
    /// outputs are unavailable.
    Memory,
}

/// `--save-only-{cleaned,mask,text}` (§12.3 step 2, §16.12 item 16). Mutually exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveOnly {
    Cleaned,
    Mask,
    Text,
}

/// The four `--skip-*` flags. Note §16.12 item 5: `denoise` is *disable*, the other
/// three are *load from cache*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SkipFlags {
    pub text_detection: bool,
    pub preprocess: bool,
    pub mask: bool,
    pub denoise: bool,
}

impl SkipFlags {
    /// §16.12 item 6 — skipping step *n* implies skipping every earlier step. Returns
    /// the normalised flags; the caller logs the `WARN` (this function is pure).
    pub fn normalized(self) -> Self {
        let mask = self.mask;
        let preprocess = self.preprocess || mask;
        let text_detection = self.text_detection || preprocess;
        Self {
            text_detection,
            preprocess,
            mask,
            denoise: self.denoise,
        }
    }

    /// `true` iff normalisation would change anything (i.e. a `WARN` is due).
    pub fn implies_more(self) -> bool {
        self.normalized() != self
    }

    /// The first step actually executed (§16.12 item 6). `Denoise` even when denoising
    /// is disabled — `--skip-denoise` is not a resume level, and the chain then simply
    /// falls through to export.
    pub fn start_step(self) -> Step {
        let flags = self.normalized();
        if flags.mask {
            Step::Denoise
        } else if flags.preprocess {
            Step::Mask
        } else if flags.text_detection {
            Step::Preprocess
        } else {
            Step::Detect
        }
    }
}

/// spec §16.12 item 16 — which export categories a run requests, as the `Vec<Output>`
/// `pc_export::ExportInput` expects (read through `pc_export::discover::Category`).
pub fn requested_outputs(save_only: Option<SaveOnly>, extract_text: bool) -> Vec<Output> {
    let cleaned = [Output::MaskedOutput, Output::DenoisedOutput];
    let mask = [Output::FinalMask, Output::DenoiseMask];
    let text = [Output::IsolatedText];

    match save_only {
        Some(SaveOnly::Cleaned) => cleaned.to_vec(),
        Some(SaveOnly::Mask) => mask.to_vec(),
        Some(SaveOnly::Text) => text.to_vec(),
        None => {
            let mut outputs = cleaned.to_vec();
            outputs.extend(mask);
            if extract_text {
                outputs.extend(text);
            }
            outputs
        }
    }
}

/// spec §4.1 / §16.12 item 14.
pub fn select_checkpointing(
    image_count: usize,
    debug_outputs: bool,
    no_cache: bool,
) -> Checkpointing {
    if image_count == 1 && !debug_outputs && no_cache {
        Checkpointing::Memory
    } else {
        Checkpointing::Disk
    }
}

/// spec §4.5 / §16.12 item 17: `0` means "all cores"; the result is bounded to
/// `1..=image_count`.
pub fn resolve_threads(configured: usize, image_count: usize) -> usize {
    let cores = if configured == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        configured
    };
    cores.clamp(1, image_count.max(1))
}

/// Everything the orchestrator needs, resolved once. Built by `pc-cli` from the parsed
/// arguments plus the loaded `Profile`; `pc-pipeline` never reads a config file itself.
#[derive(Debug, Clone)]
pub struct PipelineOptions {
    pub profile: Profile,
    pub cache_dir: PathBuf,
    /// Passed through to `pc_export::ExportInput::output_dir` (absolute => used as-is,
    /// relative => relative to `export_path.parent()`; §16.11 item 4 owns the rule).
    pub output_dir: PathBuf,
    pub skips: SkipFlags,
    pub checkpointing: Checkpointing,
    pub save_only: Option<SaveOnly>,
    pub extract_text: bool,
    /// `--cache-masks || general.always_cache_masks` (§16.12 item 15).
    pub debug_outputs: bool,
    pub keep_cache: bool,
    pub fail_fast: bool,
    /// Already resolved through [`resolve_threads`]; always `>= 1`.
    pub threads: usize,
    /// `true` for `panel-ocr ocr` (§9.3 step 2's strict-language rule).
    pub performing_ocr: bool,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            profile: Profile::default(),
            cache_dir: PathBuf::from("cache"),
            output_dir: PathBuf::from("cleaned"),
            skips: SkipFlags::default(),
            checkpointing: Checkpointing::Disk,
            save_only: None,
            extract_text: false,
            debug_outputs: false,
            keep_cache: false,
            fail_fast: false,
            threads: 1,
            performing_ocr: false,
        }
    }
}

impl PipelineOptions {
    /// spec §16.12 item 5 — `--skip-denoise` disables the stage *and* suppresses stale
    /// cached denoise artifacts at export time.
    pub fn denoising_enabled(&self) -> bool {
        self.profile.denoiser.denoising_enabled && !self.skips.denoise
    }

    pub fn requested_outputs(&self) -> Vec<Output> {
        requested_outputs(self.save_only, self.extract_text)
    }

    pub fn start_step(&self) -> Step {
        self.skips.start_step()
    }

    /// Debug artifacts are a `Disk`-mode-only feature (§4.1).
    pub fn debug_outputs_active(&self) -> bool {
        self.debug_outputs && self.checkpointing == Checkpointing::Disk
    }
}
