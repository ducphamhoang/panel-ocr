//! spec §6's validation paragraph, one function per rule group. Every failure is a
//! **config-load** error (§6, §12.7(A)9) -- never a per-image runtime error.

use crate::error::ConfigError;
use crate::profile::{DenoiserConfig, GeneralConfig, MaskerConfig, PreprocessorConfig};

/// §6 / §12.3 step 6. `.jp2` is deliberately absent: it is a legal *input* suffix but
/// the `image` crate has no JPEG2000 encoder, so it is rejected as an output suffix.
pub const SUPPORTED_OUTPUT_SUFFIXES: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".dib", ".ppm",
];

/// Human-readable rendering of `SUPPORTED_OUTPUT_SUFFIXES`, e.g.
/// `".png, .jpg, .jpeg, .webp, .tif, .tiff, .bmp, .dib, .ppm"`. §6 requires the error
/// to name the supported list.
pub fn supported_suffix_list() -> String {
    todo!()
}

/// Validates one output-suffix field. `allow_empty` is true only for
/// `general.preferred_file_type` (§6: empty = keep original suffix).
/// Normalisation: ASCII-lowercase; the leading `.` is required.
pub fn validate_output_suffix(
    field: &str,
    value: &str,
    allow_empty: bool,
) -> Result<(), ConfigError> {
    todo!()
}

pub fn validate_general(cfg: &GeneralConfig, out: &mut Vec<ConfigError>) {
    todo!()
}

pub fn validate_preprocessor(cfg: &PreprocessorConfig, out: &mut Vec<ConfigError>) {
    todo!()
}

pub fn validate_masker(cfg: &MaskerConfig, out: &mut Vec<ConfigError>) {
    todo!()
}

pub fn validate_denoiser(cfg: &DenoiserConfig, out: &mut Vec<ConfigError>) {
    todo!()
}

/// `n >= 3 && n % 2 == 1` (§6, for both NLM window sizes).
pub fn is_odd_at_least_three(n: u32) -> bool {
    todo!()
}
