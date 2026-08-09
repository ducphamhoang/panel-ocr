//! spec §6's validation paragraph, one function per rule group. Every failure is a
//! **config-load** error (§6, §12.7(A)9) -- never a per-image runtime error.

use crate::error::ConfigError;
use crate::profile::{
    DenoiserConfig, GeneralConfig, InpainterConfig, MaskerConfig, PreprocessorConfig,
    TextDetectorConfig,
};
use std::cmp::Ordering;

/// §6 / §12.3 step 6. `.jp2` is deliberately absent: it is a legal *input* suffix but
/// the `image` crate has no JPEG2000 encoder, so it is rejected as an output suffix.
pub const SUPPORTED_OUTPUT_SUFFIXES: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".dib", ".ppm",
];

/// Human-readable rendering of `SUPPORTED_OUTPUT_SUFFIXES`, e.g.
/// `".png, .jpg, .jpeg, .webp, .tif, .tiff, .bmp, .dib, .ppm"`. §6 requires the error
/// to name the supported list.
pub fn supported_suffix_list() -> String {
    SUPPORTED_OUTPUT_SUFFIXES.join(", ")
}

/// Validates one output-suffix field. `allow_empty` is true only for
/// `general.preferred_file_type` (§6: empty = keep original suffix).
/// Normalisation: ASCII-lowercase; the leading `.` is required.
pub fn validate_output_suffix(
    field: &str,
    value: &str,
    allow_empty: bool,
) -> Result<(), ConfigError> {
    if allow_empty && value.is_empty() {
        return Ok(());
    }
    let normalized = value.to_ascii_lowercase();
    if SUPPORTED_OUTPUT_SUFFIXES.contains(&normalized.as_str()) {
        return Ok(());
    }
    let detail = if normalized == ".jp2" {
        ".jp2 is supported as input only and cannot be used as an output suffix".into()
    } else if value.is_empty() {
        format!(
            "must not be empty; supported suffixes: {}",
            supported_suffix_list()
        )
    } else {
        format!(
            "unsupported suffix `{value}`; supported suffixes: {}",
            supported_suffix_list()
        )
    };
    Err(ConfigError::Invalid {
        field: field.into(),
        message: detail,
    })
}

pub fn validate_general(cfg: &GeneralConfig, out: &mut Vec<ConfigError>) {
    if let Err(error) = validate_output_suffix(
        "general.preferred_file_type",
        &cfg.preferred_file_type,
        true,
    ) {
        out.push(error);
    }
    if let Err(error) = validate_output_suffix(
        "general.preferred_mask_file_type",
        &cfg.preferred_mask_file_type,
        false,
    ) {
        out.push(error);
    }
    if cfg.long_strip_aspect_ratio.partial_cmp(&0.0) != Some(Ordering::Greater) {
        out.push(ConfigError::Invalid {
            field: "general.long_strip_aspect_ratio".into(),
            message: "must be greater than 0".into(),
        });
    }
}

/// ONNX Runtime's thread-count setters receive a signed 32-bit integer in the C API.
/// Zero is the runtime-default sentinel; positive values are explicit thread counts.
pub fn validate_text_detector(cfg: &TextDetectorConfig, out: &mut Vec<ConfigError>) {
    let max = i32::MAX as usize;
    for (field, value) in [
        ("text_detector.intra_threads", cfg.intra_threads),
        ("text_detector.inter_threads", cfg.inter_threads),
    ] {
        if value > max {
            out.push(ConfigError::Invalid {
                field: field.into(),
                message: format!("must be 0 or at most {max}"),
            });
        }
    }
}

pub fn validate_preprocessor(cfg: &PreprocessorConfig, out: &mut Vec<ConfigError>) {
    if !(0.0..=100.0).contains(&cfg.box_overlap_threshold) {
        out.push(ConfigError::Invalid {
            field: "preprocessor.box_overlap_threshold".into(),
            message: "must be between 0 and 100 inclusive".into(),
        });
    }
    if let Err(error) = cfg.compile_blacklist() {
        out.push(ConfigError::Invalid {
            field: "preprocessor.ocr_blacklist_pattern".into(),
            message: format!("must be a valid regular expression: {error}"),
        });
    }
}

pub fn validate_masker(cfg: &MaskerConfig, out: &mut Vec<ConfigError>) {
    if cfg.mask_growth_step_pixels < 1 {
        out.push(ConfigError::Invalid {
            field: "masker.mask_growth_step_pixels".into(),
            message: "must be at least 1".into(),
        });
    }
    if cfg.mask_growth_steps < 1 {
        out.push(ConfigError::Invalid {
            field: "masker.mask_growth_steps".into(),
            message: "must be at least 1".into(),
        });
    }
    if cfg.mask_max_standard_deviation.partial_cmp(&0.0) != Some(Ordering::Greater) {
        out.push(ConfigError::Invalid {
            field: "masker.mask_max_standard_deviation".into(),
            message: "must be greater than 0".into(),
        });
    }
    if !(0.0..1.0).contains(&cfg.mask_improvement_threshold) {
        out.push(ConfigError::Invalid {
            field: "masker.mask_improvement_threshold".into(),
            message: "must be at least 0 and less than 1".into(),
        });
    }
}

pub fn validate_denoiser(cfg: &DenoiserConfig, out: &mut Vec<ConfigError>) {
    if cfg.filter_strength.partial_cmp(&0.0) != Some(Ordering::Greater) {
        out.push(ConfigError::Invalid {
            field: "denoiser.filter_strength".into(),
            message: "must be greater than 0".into(),
        });
    }
    if cfg.color_filter_strength.partial_cmp(&0.0) != Some(Ordering::Greater) {
        out.push(ConfigError::Invalid {
            field: "denoiser.color_filter_strength".into(),
            message: "must be greater than 0".into(),
        });
    }
    if !is_odd_at_least_three(cfg.template_window_size) {
        out.push(ConfigError::Invalid {
            field: "denoiser.template_window_size".into(),
            message: "must be odd and at least 3".into(),
        });
    }
    if !is_odd_at_least_three(cfg.search_window_size) {
        out.push(ConfigError::Invalid {
            field: "denoiser.search_window_size".into(),
            message: "must be odd and at least 3".into(),
        });
    }
}

/// spec §6 as superseded by §16.38 item 13(b): the `[inpainter]` rules, mirroring
/// `config.py:917-932`'s `fix()` as **errors** rather than silent clamps (this crate
/// rejects, upstream repairs).
///
/// The five radius keys' `>= 0` rules are enforced by their `u32` type rather than by a
/// check here — a `>= 0` comparison on an unsigned integer is a line that cannot fail, and
/// this crate already handles `min_mask_thickness` / `noise_outline_size` /
/// `noise_fade_radius` the same way. A negative literal fails the load as
/// `ConfigError::Parse` naming the key; `validation.rs` asserts that at the load level.
pub fn validate_inpainter(cfg: &InpainterConfig, out: &mut Vec<ConfigError>) {
    if cfg.inpainting_min_std_dev.partial_cmp(&0.0) == Some(Ordering::Less)
        || cfg.inpainting_min_std_dev.is_nan()
    {
        out.push(ConfigError::Invalid {
            field: "inpainter.inpainting_min_std_dev".into(),
            message: "must be at least 0".into(),
        });
    }
    if cfg.inpainting_radius_multiplier.partial_cmp(&0.0) == Some(Ordering::Less)
        || cfg.inpainting_radius_multiplier.is_nan()
    {
        out.push(ConfigError::Invalid {
            field: "inpainter.inpainting_radius_multiplier".into(),
            message: "must be at least 0".into(),
        });
    }
    // upstream's own `config.py:932` invariant.
    if cfg.max_inpainting_radius < cfg.min_inpainting_radius {
        out.push(ConfigError::Invalid {
            field: "inpainter.max_inpainting_radius".into(),
            message: format!(
                "must be at least `min_inpainting_radius` ({})",
                cfg.min_inpainting_radius
            ),
        });
    }
}

/// `n >= 3 && n % 2 == 1` (§6, for both NLM window sizes).
pub fn is_odd_at_least_three(n: u32) -> bool {
    n >= 3 && n % 2 == 1
}
