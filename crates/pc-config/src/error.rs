//! spec §6 error + warning vocabulary.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The text is not well-formed TOML, or a value has the wrong type / is outside
    /// the range of its Rust type (spec §16.5 item 5: `off_white_max_threshold = 256`
    /// lands here, not in `Invalid`). `message` must name the offending key.
    #[error("failed to parse config: {message}")]
    Parse { message: String },

    /// A §6 validation rule was violated. `field` is the fully-qualified TOML path,
    /// e.g. `masker.mask_growth_step_pixels`.
    #[error("invalid config value for `{field}`: {message}")]
    Invalid { field: String, message: String },
}

impl ConfigError {
    /// `Some(field)` for `Invalid`, `None` otherwise. Lets tests assert on the rule
    /// that fired without matching on message text.
    pub fn field(&self) -> Option<&str> {
        match self {
            Self::Invalid { field, .. } => Some(field),
            Self::Io { .. } | Self::Parse { .. } => None,
        }
    }
}

/// Non-fatal load-time findings. Every variant is *also* emitted through `tracing`
/// at `WARN` when it is produced; this enum is the deterministic, testable mirror.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigWarning {
    /// §6: unknown keys WARN and are preserved.
    UnknownKey { table: String, key: String },
    /// §6/§16.5 item 11: an entire unrecognised table. Also preserved.
    UnknownTable { table: String },
    /// §15.7 / §14.7: `denoiser.colored_images = true` is honoured, but v1 uses a
    /// joint-channel approximation and ignores `color_filter_strength` until v1.5.
    /// The `tracing` WARN for this one fires **once per process** (§15.7 "one-time");
    /// this value is still returned on every parse.
    ColoredImagesApproximation,
    /// §16.38 item 7(e) / `DEVIATION(26)`: `inpainter.inpainting_max_mask_radius` is set
    /// away from its default, and the key is inert — in this port *and* upstream. The
    /// `tracing` WARN fires **once per process**; this value is returned on every parse.
    InertInpaintingMaxMaskRadius,
}

impl ConfigWarning {
    /// The exact user-facing text logged at `WARN`. Frozen: §15.7 requires the
    /// colored-images notice to state both the approximation and the ignored key.
    pub fn message(&self) -> String {
        match self {
            Self::UnknownKey { table, key } if table.is_empty() => {
                format!("unknown config key `{key}`; preserving it")
            }
            Self::UnknownKey { table, key } => {
                format!("unknown config key `{table}.{key}`; preserving it")
            }
            Self::UnknownTable { table } => {
                format!("unknown config table `[{table}]`; preserving it")
            }
            Self::ColoredImagesApproximation => concat!(
                "`denoiser.colored_images = true` uses the v1 joint-channel ",
                "approximation; `color_filter_strength` is ignored until the ",
                "Lab-split implementation in v1.5"
            )
            .to_owned(),
            Self::InertInpaintingMaxMaskRadius => concat!(
                "`inpainter.inpainting_max_mask_radius` has no effect: it is inert in this ",
                "port and in upstream PanelCleaner, whose `inpainting.py` never reads it. ",
                "The key that actually gates inpainting eligibility is ",
                "`inpainter.min_inpainting_radius`"
            )
            .to_owned(),
        }
    }
}
