//! `pc-config` -- spec §6. The TOML `Profile` (per-stage settings), the app-level
//! `Config`, their defaults, their validation, and format-preserving round-trip via
//! `toml_edit`.
//!
//! Three rules this crate exists to enforce:
//!   1. Defaults are **exactly** upstream's (`config.py`), transcribed in
//!      `default_profile.toml` and mirrored by the `Default` impls.
//!   2. Validation happens at **load** time, never per-image (§6, §12.7(A)9).
//!   3. Unknown keys are a `WARN` and are **preserved** on round-trip -- that is the
//!      entire reason this crate uses `toml_edit` and not `toml`.
//!
//! NOTE (spec §16.5 item 1): every stage crate depends on this crate, since the
//! per-stage config structs live here and are embedded by value in each stage's
//! `Input` (§4.3).

pub mod config;
pub mod error;
pub mod profile;
pub mod round_trip;
pub mod validate;

pub use config::Config;
pub use error::{ConfigError, ConfigWarning};
pub use profile::{
    DenoiserConfig, GeneralConfig, InpainterConfig, MaskRefineMode, MaskerConfig,
    OcrLanguageSetting, PreprocessorConfig, Profile, ReadingOrder, TextDetectorConfig,
};
pub use round_trip::{ConfigDocument, ProfileDocument};
pub use validate::SUPPORTED_OUTPUT_SUFFIXES;

/// The literal spec §6 profile, byte-for-byte. `panel-ocr profile new` writes this;
/// `Profile::default()` must parse out of it unchanged.
pub const DEFAULT_PROFILE_TOML: &str = include_str!("default_profile.toml");
