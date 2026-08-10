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
    /// v1.5, ratified by §16.38 item 13. Declared last so `TABLES` order, the default
    /// document's table order and `validate_all`'s reporting order all agree.
    pub inpainter: InpainterConfig,
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
        (
            "inpainter",
            &[
                "inpainting_enabled",
                "inpainting_min_std_dev",
                "inpainting_max_mask_radius",
                "min_inpainting_radius",
                "max_inpainting_radius",
                "inpainting_radius_multiplier",
                "inpainting_isolation_radius",
                "inpainting_fade_radius",
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
        crate::validate::validate_inpainter(&self.inpainter, &mut errors);
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
/// spec §8.3 step 5 / §15.2. Simple remains the shipped default; `Annotation` is
/// accepted by config and opts into upstream refinement. The default remains a deliberate divergence from upstream's
/// unconditional refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskRefineMode {
    /// DEVIATION(12): the shipped default is `Simple`; parity is reachable only by opting in
    /// with `mask_refine_mode = "annotation"`.
    #[default]
    Simple,
    Annotation,
}

impl GeneralConfig {
    /// `None` when `preferred_file_type` is empty ("keep original suffix"), else the
    /// normalised (ASCII-lowercased, dot-prefixed) suffix.
    pub fn cleaned_suffix(&self) -> Option<String> {
        (!self.preferred_file_type.is_empty())
            .then(|| self.preferred_file_type.to_ascii_lowercase())
    }
}

impl TextDetectorConfig {
    /// `None` when `model_path` is empty.
    pub fn model_path(&self) -> Option<&std::path::Path> {
        (!self.model_path.is_empty()).then(|| std::path::Path::new(&self.model_path))
    }
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

// ---------------------------------------------------------------- [inpainter]

/// spec §6 as superseded by §16.38 item 13 — the eight-key v1.5 inpainting block.
///
/// Every default is `config.py:817-824`'s **dataclass** default, which §16.38 item 7(b)
/// established is upstream's runtime authority: `media/default.conf` disagrees on two keys
/// and is read by zero upstream Python files.
///
/// The five radius keys are `u32` rather than a signed type, matching this crate's
/// existing treatment of `masker.min_mask_thickness` and `denoiser.noise_outline_size`:
/// §6's `>= 0` rule for them is then enforced by the *type*, and a negative literal fails
/// the load as a `ConfigError::Parse` naming the key (§16.5 item 5's precedent). The two
/// non-`Pixels` keys are `f64` and carry real `validate_inpainter` rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct InpainterConfig {
    /// §16.38 item 13(c): `true` must load and validate even in a build with no ONNX —
    /// config accepts, the stage refuses. Same split §16.36 item 6 ruled for `device`.
    pub inpainting_enabled: bool,
    pub inpainting_min_std_dev: f64,
    /// **INERT, in this port and upstream.** `pcleaner/inpainting.py` never reads it; the
    /// gate its upstream comment describes is `inpainting.py:89`'s
    /// `thickness <= min_inpainting_radius`. Parsed, validated and round-tripped, never
    /// read by the eligibility filter. Setting it away from this default emits a one-time
    /// WARN — `DEVIATION(26)`, §16.38 item 7(e).
    pub inpainting_max_mask_radius: u32,
    pub min_inpainting_radius: u32,
    pub max_inpainting_radius: u32,
    /// Live: it feeds §16.38 item 3(e)'s growth formula, which is why its `>= 0.0` rule is
    /// checked rather than left to the type (it is the block's only non-`Pixels`,
    /// non-threshold key).
    pub inpainting_radius_multiplier: f64,
    pub inpainting_isolation_radius: u32,
    pub inpainting_fade_radius: u32,
}

impl InpainterConfig {
    /// `config.py:819`'s dataclass default for the inert key. The WARN of `DEVIATION(26)`
    /// fires when a loaded profile differs from this, and never at this value.
    pub const DEFAULT_MAX_MASK_RADIUS: u32 = 6;
}

impl Default for InpainterConfig {
    fn default() -> Self {
        Self {
            inpainting_enabled: false,
            inpainting_min_std_dev: 15.0,
            inpainting_max_mask_radius: Self::DEFAULT_MAX_MASK_RADIUS,
            min_inpainting_radius: 7,
            max_inpainting_radius: 20,
            inpainting_radius_multiplier: 0.2,
            inpainting_isolation_radius: 5,
            inpainting_fade_radius: 4,
        }
    }
}
