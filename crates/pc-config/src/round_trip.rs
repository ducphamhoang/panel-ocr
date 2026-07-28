//! spec §6 round-trip: `toml_edit::DocumentMut` is retained alongside the typed
//! `Profile`, so comments, key order, whitespace and **unknown keys** all survive a
//! load/save cycle. Unknown keys additionally produce a `ConfigWarning` (and a
//! `tracing::warn!`), never a silent drop and never an error.

use crate::config::Config;
use crate::error::{ConfigError, ConfigWarning};
use crate::profile::Profile;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ProfileDocument {
    profile: Profile,
    doc: toml_edit::DocumentMut,
    warnings: Vec<ConfigWarning>,
}

impl ProfileDocument {
    /// Parse -> collect unknown-key warnings -> deserialize -> **validate** (§6).
    /// A validation failure returns `Err`; unknown keys do not.
    ///
    /// §15.7: if the parsed profile has `denoiser.colored_images = true`, pushes
    /// `ConfigWarning::ColoredImagesApproximation` and emits the process-wide
    /// one-time `tracing::warn!`.
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        todo!()
    }

    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        todo!()
    }

    /// A fresh document from `DEFAULT_PROFILE_TOML` -- comments included. This, not
    /// `from_profile(&Profile::default())`, is what `panel-ocr profile new` writes.
    pub fn new_default() -> Self {
        todo!()
    }

    /// A fresh, comment-free document serialised from `profile`.
    pub fn from_profile(profile: &Profile) -> Self {
        todo!()
    }

    pub fn profile(&self) -> &Profile {
        todo!()
    }

    pub fn warnings(&self) -> &[ConfigWarning] {
        todo!()
    }

    /// Underlying `toml_edit` document, for callers that want to inspect formatting.
    pub fn document(&self) -> &toml_edit::DocumentMut {
        todo!()
    }

    /// Write `profile`'s values back into the retained document **in place**: known
    /// keys are updated (decor preserved), unknown keys are left untouched, keys
    /// absent from the document are appended to their table. Re-validates.
    pub fn set_profile(&mut self, profile: &Profile) -> Result<(), ConfigError> {
        todo!()
    }

    pub fn to_toml_string(&self) -> String {
        todo!()
    }

    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        todo!()
    }
}

/// Same contract as `ProfileDocument`, for the app-level `Config` (§6, spec gap
/// noted in §16.5 item 2).
#[derive(Debug, Clone)]
pub struct ConfigDocument {
    config: Config,
    doc: toml_edit::DocumentMut,
    warnings: Vec<ConfigWarning>,
}

impl ConfigDocument {
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        todo!()
    }
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        todo!()
    }
    pub fn from_config(config: &Config) -> Self {
        todo!()
    }
    pub fn config(&self) -> &Config {
        todo!()
    }
    pub fn warnings(&self) -> &[ConfigWarning] {
        todo!()
    }
    pub fn to_toml_string(&self) -> String {
        todo!()
    }
    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        todo!()
    }
}

/// §15.7 "one-time": the `tracing::warn!` fires at most once per process.
/// `ConfigWarning::ColoredImagesApproximation` is still returned on every parse.
pub(crate) fn warn_colored_images_once() {
    todo!()
}
