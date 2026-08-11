//! spec §13.1 — the `clap` surface. Fresh and idiomatic, with no docopt compatibility
//! (decisions doc #6).
//!
//! Fully pinned; implemented, not stubbed. §1 rule 3: nothing here does any work beyond
//! parsing and validating what the user typed.

use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use pc_pipeline::{SaveOnly, SkipFlags};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Parser)]
#[command(
    name = "panel-ocr",
    version,
    about = "Detect and clean text bubbles in manga/comic pages",
    propagate_version = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Increase log verbosity (repeatable): -v info, -vv debug, -vvv trace.
    #[arg(short = 'v', long = "verbose", global = true, action = ArgAction::Count)]
    pub verbose: u8,

    /// Log errors only.
    #[arg(short = 'q', long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Clean pages: detect → preprocess → mask → denoise → export.
    Clean(CleanArgs),
    /// Run OCR over the detected boxes and write a CSV/TXT report.
    Ocr(OcrArgs),
    /// Inpaint a user-supplied mask over an image directly — no cache/uuid state (spec-adjacent,
    /// not part of the `clean` pipeline's five stages).
    Inpaint(InpaintArgs),
    /// Inspect and manage profiles.
    Profile {
        #[command(subcommand)]
        command: ProfileCommand,
    },
    /// Inspect and clear the cache.
    Cache {
        #[command(subcommand)]
        command: CacheCommand,
    },
    /// Manage model files.
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
}

#[derive(Debug, Args)]
pub struct CleanArgs {
    /// Image files and/or directories of images.
    #[arg(required = true, value_name = "PATHS")]
    pub paths: Vec<PathBuf>,

    /// Where exports go. Absolute => used as-is; relative => next to each input.
    #[arg(long, value_name = "DIR", default_value = "cleaned")]
    pub output_dir: PathBuf,

    /// A profile name from the app config.
    #[arg(long, value_name = "NAME", conflicts_with = "profile_path")]
    pub profile: Option<String>,
    /// A profile TOML file, bypassing the app config.
    #[arg(long, value_name = "FILE")]
    pub profile_path: Option<PathBuf>,

    /// Load `#raw.json` from the cache instead of running detection.
    #[arg(long)]
    pub skip_text_detection: bool,
    /// Load `#clean.json` from the cache instead of running preprocessing.
    #[arg(long)]
    pub skip_preprocess: bool,
    /// Load `#mask_data.json` from the cache instead of running masking.
    #[arg(long)]
    pub skip_mask: bool,
    /// Disable denoising entirely (spec §16.12 item 5 — this one is *disable*, not
    /// *load from cache*).
    #[arg(long)]
    pub skip_denoise: bool,

    /// Disable inpainting entirely (spec §16.38 item 16(e) — this one is *disable*, not
    /// *load from cache*).
    #[arg(long)]
    pub skip_inpaint: bool,

    /// Write only the cleaned image.
    #[arg(long, group = "save_only")]
    pub save_only_cleaned: bool,
    /// Write only the mask image.
    #[arg(long, group = "save_only")]
    pub save_only_mask: bool,
    /// Write only the isolated-text image.
    #[arg(long, group = "save_only", requires = "extract_text")]
    pub save_only_text: bool,

    /// Also produce the isolated text layer (`_text.png`).
    #[arg(long)]
    pub extract_text: bool,

    /// Keep the debug mask artifacts (`_box_mask`, `_cut_mask`, `_with_masks`).
    #[arg(long, conflicts_with = "no_cache")]
    pub cache_masks: bool,
    /// Do not delete the cache directory when the run finishes.
    #[arg(long, conflicts_with = "no_cache")]
    pub keep_cache: bool,
    /// Never touch the cache: a single image runs entirely in memory (spec §4.1).
    #[arg(long)]
    pub no_cache: bool,

    /// Abort on the first failed image instead of isolating it (spec §5.4).
    #[arg(long)]
    pub fail_fast: bool,

    /// Worker threads; 0 or omitted uses the profile's `general.max_threads`.
    #[arg(long, value_name = "N")]
    pub threads: Option<usize>,

    /// Override the detector model file.
    #[arg(long, value_name = "FILE")]
    pub model_path: Option<PathBuf>,

    /// Suppress the per-stage analytics tables.
    #[arg(long)]
    pub hide_analytics: bool,

    /// Detector backend (spec §16.12 items 2 and 21). Hidden: `onnx` is the documented
    /// default and is available when pc-cli is built with its non-default `onnx` feature.
    #[arg(long, value_name = "SPEC", default_value = "onnx", hide = true)]
    pub detector: DetectorSpec,

    /// Cache directory override (spec §16.12 item 21).
    #[arg(long, value_name = "DIR", hide = true)]
    pub cache_dir: Option<PathBuf>,
}

impl CleanArgs {
    /// spec §16.12 item 16 — at most one category narrowing.
    pub fn save_only(&self) -> Option<SaveOnly> {
        if self.save_only_cleaned {
            Some(SaveOnly::Cleaned)
        } else if self.save_only_mask {
            Some(SaveOnly::Mask)
        } else if self.save_only_text {
            Some(SaveOnly::Text)
        } else {
            None
        }
    }

    /// The raw flags, **before** §16.12 item 6's prefix normalisation (the pipeline
    /// normalises and warns).
    pub fn skip_flags(&self) -> SkipFlags {
        SkipFlags {
            text_detection: self.skip_text_detection,
            preprocess: self.skip_preprocess,
            mask: self.skip_mask,
            denoise: self.skip_denoise,
        }
    }
}

#[derive(Debug, Args)]
pub struct OcrArgs {
    #[arg(required = true, value_name = "PATHS")]
    pub paths: Vec<PathBuf>,

    #[arg(long, value_enum, default_value_t = ReportFormatArg::Csv)]
    pub format: ReportFormatArg,

    /// Write the report here instead of to stdout.
    #[arg(long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    #[arg(long, value_name = "NAME", conflicts_with = "profile_path")]
    pub profile: Option<String>,
    #[arg(long, value_name = "FILE")]
    pub profile_path: Option<PathBuf>,

    #[arg(long, value_name = "SPEC", default_value = "onnx", hide = true)]
    pub detector: DetectorSpec,

    #[arg(long, value_name = "DIR", hide = true)]
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct InpaintArgs {
    /// The image to inpaint.
    #[arg(required = true, value_name = "IMAGE")]
    pub image: PathBuf,

    /// RGBA PNG the same pixel dimensions as IMAGE. Alpha > 0 marks a painted pixel;
    /// RGB is ignored. Disjoint painted blobs become separate inpaint regions.
    #[arg(long, value_name = "FILE")]
    pub mask: PathBuf,

    /// Where to write the inpainted result (format inferred from the extension; use `.png`).
    #[arg(long, value_name = "FILE")]
    pub output: PathBuf,

    #[arg(long, value_name = "NAME", conflicts_with = "profile_path")]
    pub profile: Option<String>,
    #[arg(long, value_name = "FILE")]
    pub profile_path: Option<PathBuf>,

    /// Override the LaMa inpainting model file (bypasses the managed cache and its digest
    /// check, same latitude `clean --model-path` gives the detector).
    #[arg(long, value_name = "FILE")]
    pub model_path: Option<PathBuf>,

    /// Cache directory override — used only to locate/download the managed LaMa model,
    /// never for pipeline checkpoints (this command has none). Hidden like `clean`'s.
    #[arg(long, value_name = "DIR", hide = true)]
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ReportFormatArg {
    Csv,
    Txt,
}

impl From<ReportFormatArg> for pc_export::ReportFormat {
    fn from(value: ReportFormatArg) -> Self {
        match value {
            ReportFormatArg::Csv => pc_export::ReportFormat::Csv,
            ReportFormatArg::Txt => pc_export::ReportFormat::Txt,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum ProfileCommand {
    /// Write the default profile to a file.
    New { path: PathBuf },
    /// Print a profile as TOML.
    Show {
        #[arg(long, value_name = "NAME")]
        profile: Option<String>,
        #[arg(long, value_name = "FILE")]
        profile_path: Option<PathBuf>,
    },
    /// List the profiles the app config knows about.
    List,
    /// Validate a profile file.
    Validate { path: PathBuf },
    /// Open a profile in `$EDITOR`.
    Edit {
        #[arg(long, value_name = "NAME")]
        profile: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
pub enum CacheCommand {
    /// Report the cache location and size.
    Show {
        /// Cache directory override (spec §16.12 item 21).
        #[arg(long, value_name = "DIR", hide = true)]
        cache_dir: Option<PathBuf>,
    },
    /// Delete cached files.
    Clear {
        #[arg(long)]
        models: bool,
        #[arg(long)]
        images: bool,
        /// Cache directory override (spec §16.12 item 21).
        #[arg(long, value_name = "DIR", hide = true)]
        cache_dir: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    /// Download the required detector/OCR/inpainting models into the managed cache, repairing entries with mismatched digests. Requires network access; 763,086,911 bytes in total (~763 MB) across all four required models — detector 95 MB, OCR encoder 343 MB, OCR decoder 117 MB, LaMa inpainter 207 MB. No `onnx` feature or ONNX Runtime is required.
    Download {
        /// Cache directory override (spec §16.12 item 21).
        #[arg(long, value_name = "DIR", hide = true)]
        cache_dir: Option<PathBuf>,
        /// Also fetch optional models. No model is currently optional -- the LaMa inpainting
        /// weights became required with §16.46 item 11(b) -- so this fetches nothing extra
        /// today. Spec §13.1 as superseded by §16.38 item 19 and §16.46 item 11(c). Visible on
        /// purpose, unlike `--cache-dir`.
        #[arg(long)]
        include_optional: bool,
    },
    /// Verify model availability and digests without modifying the cache, reporting each model's status. Exits 1 if any required model is missing or if any reported model fails verification, making it useful as a preflight or CI check. No `onnx` feature or ONNX Runtime is required.
    Verify {
        /// Cache directory override (spec §16.12 item 21).
        #[arg(long, value_name = "DIR", hide = true)]
        cache_dir: Option<PathBuf>,
        /// Also report optional models. An absent optional model is reported and is not a
        /// failure; a corrupt one still is. No model is currently optional (§16.46 item
        /// 11(b)/(c)), so this adds no row today. Spec §13.1 as superseded by §16.38 item 19.
        #[arg(long)]
        include_optional: bool,
    },
    /// Print where models are looked up.
    Path {
        /// Cache directory override (spec §16.12 item 21).
        #[arg(long, value_name = "DIR", hide = true)]
        cache_dir: Option<PathBuf>,
    },
}

/// spec §16.12 item 2 — `--detector`'s value grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectorSpec {
    /// The real ONNX backend. Not buildable in v1 (tasks D1/D4).
    Onnx,
    /// Zero blocks, blank mask — a page with no detected text (§5.6 still exports it).
    Mock,
    /// `pc-detect`'s `ReplayDetector`, keyed per image stem out of `<DIR>` (§7.2.1).
    Replay(PathBuf),
}

impl FromStr for DetectorSpec {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "onnx" => Ok(DetectorSpec::Onnx),
            "mock" => Ok(DetectorSpec::Mock),
            other => match other.split_once(':') {
                Some(("replay", dir)) if !dir.is_empty() => {
                    Ok(DetectorSpec::Replay(PathBuf::from(dir)))
                }
                _ => Err(format!(
                    "unknown detector `{other}`; expected `onnx`, `mock` or `replay:<DIR>`"
                )),
            },
        }
    }
}

impl std::fmt::Display for DetectorSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectorSpec::Onnx => write!(f, "onnx"),
            DetectorSpec::Mock => write!(f, "mock"),
            DetectorSpec::Replay(dir) => write!(f, "replay:{}", dir.display()),
        }
    }
}
