//! spec §6 -- the five per-stage config sections and their upstream-exact defaults.
//!
//! Every struct is `#[serde(default, rename_all = "snake_case")]` so that a missing
//! key takes its default (§6) and an unknown key is *ignored* by serde -- unknown keys
//! are detected separately by `round_trip`'s key registry so they can be WARNed about
//! and preserved rather than silently dropped.

use crate::error::ConfigError;
use pc_core::{device::Device, Language};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct Profile {
    pub general: GeneralConfig,
    pub text_detector: TextDetectorConfig,
    pub preprocessor: PreprocessorConfig,
    pub masker: MaskerConfig,
    pub denoiser: DenoiserConfig,
}

impl Profile {
    /// Table name -> known key names, in spec §6 declaration order. The single source
    /// of truth for "is this key unknown?" (§6 round-trip WARN) and for `profile show`.
    /// Frozen against `DEFAULT_PROFILE_TOML` by a test.
    pub const TABLES: &'static [(&'static str, &'static [&'static str])] = &[
        (
            "general",
            &[
                "preferred_file_type",
                "preferred_mask_file_type",
                "input_height_lower_target",
                "input_height_upper_target",
                "split_long_strips",
                "preferred_split_height",
                "split_tolerance_margin",
                "long_strip_aspect_ratio",
                "merge_after_split",
                "max_threads",
                "always_cache_masks",
                "device",
            ],
        ),
        (
            "text_detector",
            &[
                "model_path",
                "concurrent_models",
                "intra_threads",
                "inter_threads",
                "mask_refine_mode",
            ],
        ),
        (
            "preprocessor",
            &[
                "box_min_size",
                "suspicious_box_min_size",
                "box_overlap_threshold",
                "ocr_enabled",
                "ocr_language",
                "reading_order",
                "ocr_max_size",
                "ocr_blacklist_pattern",
                "ocr_strict_language",
                "box_padding_initial",
                "box_right_padding_initial",
                "box_padding_extended",
                "box_right_padding_extended",
                "box_reference_padding",
            ],
        ),
        (
            "masker",
            &[
                "mask_growth_step_pixels",
                "mask_growth_steps",
                "min_mask_thickness",
                "allow_colored_masks",
                "off_white_max_threshold",
                "mask_max_standard_deviation",
                "mask_improvement_threshold",
                "mask_selection_fast",
                "mask_fallback_to_lowest_deviation",
                "debug_mask_color",
            ],
        ),
        (
            "denoiser",
            &[
                "denoising_enabled",
                "noise_min_standard_deviation",
                "noise_outline_size",
                "noise_fade_radius",
                "colored_images",
                "filter_strength",
                "color_filter_strength",
                "template_window_size",
                "search_window_size",
            ],
        ),
    ];

    /// spec §6 validation paragraph. Returns the **first** violation in `TABLES` order
    /// (deterministic). Callers that want everything use `validate_all`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.validate_all().into_iter().next().map_or(Ok(()), Err)
    }

    /// Every violation, in `TABLES` order. Empty == valid.
    pub fn validate_all(&self) -> Vec<ConfigError> {
        let mut errors = Vec::new();
        crate::validate::validate_general(&self.general, &mut errors);
        crate::validate::validate_text_detector(&self.text_detector, &mut errors);
        crate::validate::validate_preprocessor(&self.preprocessor, &mut errors);
        crate::validate::validate_masker(&self.masker, &mut errors);
        crate::validate::validate_denoiser(&self.denoiser, &mut errors);
        errors
    }
}

// ------------------------------------------------------------------ [general]

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct GeneralConfig {
    /// Empty = keep the original suffix (§6). Otherwise must be in
    /// `SUPPORTED_OUTPUT_SUFFIXES` (§6 / §12.3 step 6 / §12.7(A)9).
    pub preferred_file_type: String,
    /// Never empty; must be in `SUPPORTED_OUTPUT_SUFFIXES` (spec §16.5 item 6).
    pub preferred_mask_file_type: String,
    pub input_height_lower_target: u32,
    pub input_height_upper_target: u32,
    pub split_long_strips: bool,
    pub preferred_split_height: u32,
    pub split_tolerance_margin: u32,
    pub long_strip_aspect_ratio: f64,
    pub merge_after_split: bool,
    /// `0` = all cores (§4.5).
    pub max_threads: usize,
    pub always_cache_masks: bool,
    pub device: Device,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            preferred_file_type: String::new(),
            preferred_mask_file_type: ".png".into(),
            input_height_lower_target: 1000,
            input_height_upper_target: 4000,
            split_long_strips: true,
            preferred_split_height: 2000,
            split_tolerance_margin: 500,
            long_strip_aspect_ratio: 0.33,
            merge_after_split: true,
            max_threads: 0,
            always_cache_masks: false,
            device: Device::Cpu,
        }
    }
}

impl GeneralConfig {
    /// `None` when `preferred_file_type` is empty ("keep original suffix"), else the
    /// normalised (ASCII-lowercased, dot-prefixed) suffix.
    pub fn cleaned_suffix(&self) -> Option<String> {
        (!self.preferred_file_type.is_empty())
            .then(|| self.preferred_file_type.to_ascii_lowercase())
    }
}

// ------------------------------------------------------------- [text_detector]

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct TextDetectorConfig {
    /// Empty = use the managed cache (`pc-models`). Kept as `String`, not
    /// `Option<PathBuf>`, so the TOML text round-trips verbatim; use `model_path()`.
    pub model_path: String,
    pub concurrent_models: usize,
    /// `0` delegates intra-op parallelism to ONNX Runtime; positive values pin it.
    pub intra_threads: usize,
    /// `0` delegates inter-op parallelism to ONNX Runtime; positive values pin it.
    pub inter_threads: usize,
    pub mask_refine_mode: MaskRefineMode,
}

impl Default for TextDetectorConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            concurrent_models: 1,
            intra_threads: 0,
            inter_threads: 0,
            mask_refine_mode: MaskRefineMode::default(),
        }
    }
}

impl TextDetectorConfig {
    /// `None` when `model_path` is empty.
    pub fn model_path(&self) -> Option<&std::path::Path> {
        (!self.model_path.is_empty()).then(|| std::path::Path::new(&self.model_path))
    }
}

/// spec §8.3 step 5 / §15.2. Only `Simple` is implemented in v1; `Annotation` is
/// accepted by config (spec §16.5 item 3) and rejected by `pc-detect` with
/// `StageError::InvalidInput`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskRefineMode {
    #[default]
    Simple,
    Annotation,
}

// ------------------------------------------------------------- [preprocessor]

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct PreprocessorConfig {
    /// `i64` to line up with `Rect::area() -> i64` (§2.1, §9.7(A)1).
    pub box_min_size: i64,
    pub suspicious_box_min_size: i64,
    /// Percent, 0..=100 (§6). Compared strictly-greater in `Rect::overlaps`.
    pub box_overlap_threshold: f64,
    pub ocr_enabled: bool,
    pub ocr_language: OcrLanguageSetting,
    pub reading_order: ReadingOrder,
    pub ocr_max_size: i64,
    /// Must compile as a regex (§6). Applied as a **full match** (§9.7(A)8).
    pub ocr_blacklist_pattern: String,
    pub ocr_strict_language: bool,
    pub box_padding_initial: i32,
    pub box_right_padding_initial: i32,
    pub box_padding_extended: i32,
    pub box_right_padding_extended: i32,
    pub box_reference_padding: i32,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            box_min_size: 400,
            suspicious_box_min_size: 40_000,
            box_overlap_threshold: 20.0,
            ocr_enabled: true,
            ocr_language: OcrLanguageSetting::default(),
            reading_order: ReadingOrder::default(),
            ocr_max_size: 3000,
            ocr_blacklist_pattern: "[～．ー！？０-９~.!?0-9-]*".into(),
            ocr_strict_language: false,
            box_padding_initial: 2,
            box_right_padding_initial: 3,
            box_padding_extended: 5,
            box_right_padding_extended: 5,
            box_reference_padding: 20,
        }
    }
}

impl PreprocessorConfig {
    /// Compiles `ocr_blacklist_pattern` anchored for full-match use (§9.7(A)8).
    /// `validate` has already proven this succeeds, so callers may `expect` it.
    pub fn compile_blacklist(&self) -> Result<regex::Regex, regex::Error> {
        regex::Regex::new(&format!("^(?:{})$", self.ocr_blacklist_pattern))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrLanguageSetting {
    #[default]
    DetectBox,
    DetectPage,
    Jpn,
    Eng,
}

impl OcrLanguageSetting {
    /// `Some` for the pinned variants, `None` for the two detect modes.
    pub fn fixed_language(self) -> Option<Language> {
        match self {
            Self::Jpn => Some(Language::Japanese),
            Self::Eng => Some(Language::English),
            Self::DetectBox | Self::DetectPage => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadingOrder {
    #[default]
    Auto,
    Manga,
    Comic,
}

// ------------------------------------------------------------------- [masker]

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct MaskerConfig {
    pub mask_growth_step_pixels: u32,
    pub mask_growth_steps: u32,
    pub min_mask_thickness: u32,
    pub allow_colored_masks: bool,
    pub off_white_max_threshold: u8,
    /// f64 mandated by §15.9 -- no `f32` anywhere on this path.
    pub mask_max_standard_deviation: f64,
    pub mask_improvement_threshold: f64,
    pub mask_selection_fast: bool,
    pub mask_fallback_to_lowest_deviation: bool,
    /// RGBA.
    pub debug_mask_color: [u8; 4],
}

impl Default for MaskerConfig {
    fn default() -> Self {
        Self {
            mask_growth_step_pixels: 2,
            mask_growth_steps: 11,
            min_mask_thickness: 4,
            allow_colored_masks: true,
            off_white_max_threshold: 240,
            mask_max_standard_deviation: 15.0,
            mask_improvement_threshold: 0.1,
            mask_selection_fast: false,
            mask_fallback_to_lowest_deviation: true,
            debug_mask_color: [108, 30, 240, 127],
        }
    }
}

// ----------------------------------------------------------------- [denoiser]

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct DenoiserConfig {
    pub denoising_enabled: bool,
    pub noise_min_standard_deviation: f64,
    pub noise_outline_size: u32,
    pub noise_fade_radius: u32,
    /// §15.7: setting this to `true` triggers a one-time `WARN` at load -- v1 uses a
    /// joint-channel approximation and ignores `color_filter_strength` until v1.5.
    pub colored_images: bool,
    pub filter_strength: f64,
    /// Ignored in v1 (§14.7). Still validated and still round-tripped.
    pub color_filter_strength: f64,
    /// Odd and >= 3 (§6).
    pub template_window_size: u32,
    /// Odd and >= 3 (§6).
    pub search_window_size: u32,
}

impl Default for DenoiserConfig {
    fn default() -> Self {
        Self {
            denoising_enabled: true,
            noise_min_standard_deviation: 0.25,
            noise_outline_size: 5,
            noise_fade_radius: 1,
            colored_images: false,
            filter_strength: 10.0,
            color_filter_strength: 10.0,
            template_window_size: 7,
            search_window_size: 21,
        }
    }
}
