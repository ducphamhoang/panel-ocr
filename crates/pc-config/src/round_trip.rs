//! spec §6 round-trip: `toml_edit::DocumentMut` is retained alongside the typed
//! `Profile`, so comments, key order, whitespace and **unknown keys** all survive a
//! load/save cycle. Unknown keys additionally produce a `ConfigWarning` (and a
//! `tracing::warn!`), never a silent drop and never an error.

use crate::config::Config;
use crate::error::{ConfigError, ConfigWarning};
use crate::profile::Profile;
use std::path::Path;
use std::sync::Once;

fn parse_document(text: &str) -> Result<toml_edit::DocumentMut, ConfigError> {
    text.parse::<toml_edit::DocumentMut>()
        .map_err(|error| ConfigError::Parse {
            message: error.to_string(),
        })
}

fn emit_warning(warning: &ConfigWarning) {
    tracing::warn!("{}", warning.message());
}

fn profile_warnings(doc: &toml_edit::DocumentMut) -> Vec<ConfigWarning> {
    let mut warnings = Vec::new();
    for (table_name, item) in doc.iter() {
        let Some((_, known_keys)) = Profile::TABLES
            .iter()
            .find(|(known_table, _)| *known_table == table_name)
        else {
            let warning = if item.is_table() {
                ConfigWarning::UnknownTable {
                    table: table_name.into(),
                }
            } else {
                ConfigWarning::UnknownKey {
                    table: String::new(),
                    key: table_name.into(),
                }
            };
            emit_warning(&warning);
            warnings.push(warning);
            continue;
        };

        if let Some(table) = item.as_table() {
            for (key, _) in table.iter() {
                if !known_keys.contains(&key) {
                    let warning = ConfigWarning::UnknownKey {
                        table: table_name.into(),
                        key: key.into(),
                    };
                    emit_warning(&warning);
                    warnings.push(warning);
                }
            }
        }
    }
    warnings
}

fn deserialize_profile(doc: &toml_edit::DocumentMut) -> Result<Profile, ConfigError> {
    toml_edit::de::from_document(doc.clone()).map_err(|error| ConfigError::Parse {
        message: error.to_string(),
    })
}

fn serialize_profile(profile: &Profile) -> Result<toml_edit::DocumentMut, ConfigError> {
    toml_edit::ser::to_document(profile).map_err(|error| ConfigError::Parse {
        message: error.to_string(),
    })
}

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
        let doc = parse_document(text)?;
        let mut warnings = profile_warnings(&doc);
        let profile = deserialize_profile(&doc)?;
        profile.validate()?;
        if profile.denoiser.colored_images {
            warnings.push(ConfigWarning::ColoredImagesApproximation);
            warn_colored_images_once();
        }
        Ok(Self {
            profile,
            doc,
            warnings,
        })
    }

    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&text)
    }

    /// A fresh document from `DEFAULT_PROFILE_TOML` -- comments included. This, not
    /// `from_profile(&Profile::default())`, is what `panel-ocr profile new` writes.
    pub fn new_default() -> Self {
        Self::parse(crate::DEFAULT_PROFILE_TOML)
            .expect("the built-in default profile must parse and validate")
    }

    /// A fresh, comment-free document serialised from `profile`.
    pub fn from_profile(profile: &Profile) -> Self {
        profile
            .validate()
            .expect("cannot create a profile document from an invalid profile");
        Self {
            profile: profile.clone(),
            doc: serialize_profile(profile).expect("profile serialization must succeed"),
            warnings: Vec::new(),
        }
    }

    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    pub fn warnings(&self) -> &[ConfigWarning] {
        &self.warnings
    }

    /// Underlying `toml_edit` document, for callers that want to inspect formatting.
    pub fn document(&self) -> &toml_edit::DocumentMut {
        &self.doc
    }

    /// Write `profile`'s values back into the retained document **in place**: known
    /// keys are updated (decor preserved), unknown keys are left untouched, keys
    /// absent from the document are appended to their table. Re-validates.
    pub fn set_profile(&mut self, profile: &Profile) -> Result<(), ConfigError> {
        profile.validate()?;
        let serialized = serialize_profile(profile)?;

        for (table_name, keys) in Profile::TABLES {
            if !self.doc.contains_key(table_name) {
                self.doc[table_name] = toml_edit::Item::Table(toml_edit::Table::new());
            }
            for key in *keys {
                let new_item = serialized[table_name][key].clone();
                let old_item = &mut self.doc[table_name][key];
                if let (Some(old), Some(new)) = (old_item.as_value_mut(), new_item.as_value()) {
                    let decor = old.decor().clone();
                    *old = new.clone();
                    *old.decor_mut() = decor;
                } else {
                    *old_item = new_item;
                }
            }
        }
        self.profile = profile.clone();
        Ok(())
    }

    pub fn to_toml_string(&self) -> String {
        self.doc.to_string()
    }

    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        std::fs::write(path, self.to_toml_string()).map_err(|source| ConfigError::Io {
            path: path.to_owned(),
            source,
        })
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
        let doc = parse_document(text)?;
        let mut warnings = Vec::new();
        for (key, item) in doc.iter() {
            if matches!(key, "default_profile" | "saved_profiles" | "cache_dir") {
                continue;
            }
            let warning = if item.is_table() {
                ConfigWarning::UnknownTable { table: key.into() }
            } else {
                ConfigWarning::UnknownKey {
                    table: String::new(),
                    key: key.into(),
                }
            };
            emit_warning(&warning);
            warnings.push(warning);
        }
        let config: Config =
            toml_edit::de::from_document(doc.clone()).map_err(|error| ConfigError::Parse {
                message: error.to_string(),
            })?;
        config.validate()?;
        Ok(Self {
            config,
            doc,
            warnings,
        })
    }
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::parse(&text)
    }
    pub fn from_config(config: &Config) -> Self {
        config
            .validate()
            .expect("cannot create a config document from an invalid config");
        let doc = toml_edit::ser::to_document(config)
            .expect("app configuration serialization must succeed");
        Self {
            config: config.clone(),
            doc,
            warnings: Vec::new(),
        }
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn warnings(&self) -> &[ConfigWarning] {
        &self.warnings
    }
    pub fn to_toml_string(&self) -> String {
        self.doc.to_string()
    }
    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        std::fs::write(path, self.to_toml_string()).map_err(|source| ConfigError::Io {
            path: path.to_owned(),
            source,
        })
    }
}

/// §15.7 "one-time": the `tracing::warn!` fires at most once per process.
/// `ConfigWarning::ColoredImagesApproximation` is still returned on every parse.
pub(crate) fn warn_colored_images_once() {
    static WARN_ONCE: Once = Once::new();
    WARN_ONCE.call_once(|| emit_warning(&ConfigWarning::ColoredImagesApproximation));
}
