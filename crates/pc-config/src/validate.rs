//! spec §6's validation paragraph, one function per rule group. Every failure is a
//! **config-load** error (§6, §12.7(A)9) -- never a per-image runtime error.

use crate::error::ConfigError;
use crate::profile::{DenoiserConfig, GeneralConfig, MaskerConfig, PreprocessorConfig};
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

/// `n >= 3 && n % 2 == 1` (§6, for both NLM window sizes).
pub fn is_odd_at_least_three(n: u32) -> bool {
    n >= 3 && n % 2 == 1
}
