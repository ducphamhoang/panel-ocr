//! spec §6's app-level `Config`.
//!
//! SPEC GAP (§16.5 item 2): §6 names this type and then specifies no fields, no
//! defaults and no validation rules for it. Everything below is provisional, modelled
//! on upstream `config.py`'s `Config` minus GUI/v2 keys, and must be confirmed by the
//! architects before its tests are treated as frozen.

use crate::error::ConfigError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct Config {
    /// Name of the profile to use when `--profile` is absent. `None` = built-in
    /// defaults (`Profile::default()`).
    pub default_profile: Option<String>,
    /// Name -> path. `BTreeMap` (not `HashMap`) so listing and serialisation order are
    /// deterministic, per §5.7.
    pub saved_profiles: BTreeMap<String, PathBuf>,
    /// `None` = the platform cache dir.
    pub cache_dir: Option<PathBuf>,
}

impl Config {
    /// Construct an empty app configuration.
    ///
    /// This mirrors the `Default` implementation while giving integration tests and
    /// callers a direct constructor.
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        <Self as Default>::default()
    }

    /// Provisional rule: `default_profile`, when `Some`, must name a key of
    /// `saved_profiles`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if let Some(name) = &self.default_profile {
            if !self.saved_profiles.contains_key(name) {
                return Err(ConfigError::Invalid {
                    field: "default_profile".into(),
                    message: format!("`{name}` is not present in `saved_profiles`"),
                });
            }
        }
        Ok(())
    }

    pub fn profile_path(&self, name: &str) -> Option<&std::path::Path> {
        self.saved_profiles.get(name).map(PathBuf::as_path)
    }
}
