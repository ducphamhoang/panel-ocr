//! Config and cache location discovery (spec §16.12 item 21).
//!
//! v1 was Linux + macOS only; v1.1 adds Windows (§16.33), and `dirs` is not in
//! `[workspace.dependencies]`, so this is done by hand against all three platforms' conventions.
//! Fully pinned; implemented.

use pc_config::Config;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub const APP_DIR_NAME: &str = "panel-ocr";
pub const CONFIG_FILE_NAME: &str = "config.toml";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Linux,
    MacOs,
    Windows,
}

impl Platform {
    /// The platform this binary was compiled for.
    pub const HOST: Platform = if cfg!(target_os = "windows") {
        Platform::Windows
    } else if cfg!(target_os = "macos") {
        Platform::MacOs
    } else {
        Platform::Linux
    };
}

/// An injectable environment lookup.
pub trait EnvSource {
    /// `None` means that the variable is unset or empty.
    fn var(&self, key: &str) -> Option<OsString>;
}

/// Reads the real process environment.
pub struct ProcessEnv;

impl EnvSource for ProcessEnv {
    fn var(&self, key: &str) -> Option<OsString> {
        std::env::var_os(key).filter(|value| !value.is_empty())
    }
}

/// A fixed environment table for tests.
#[derive(Debug, Default, Clone)]
pub struct MapEnv(BTreeMap<String, OsString>);

impl MapEnv {
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        Self(
            pairs
                .iter()
                .map(|(key, value)| ((*key).to_owned(), OsString::from(*value)))
                .collect(),
        )
    }
}

impl EnvSource for MapEnv {
    fn var(&self, key: &str) -> Option<OsString> {
        self.0.get(key).cloned().filter(|value| !value.is_empty())
    }
}

pub struct DirEnv<'a> {
    platform: Platform,
    env: &'a dyn EnvSource,
}

impl<'a> DirEnv<'a> {
    pub fn new(platform: Platform, env: &'a dyn EnvSource) -> Self {
        Self { platform, env }
    }

    pub fn cache_dir(&self) -> PathBuf {
        if let Some(xdg) = self.env_path("XDG_CACHE_HOME") {
            return xdg.join(APP_DIR_NAME);
        }

        let platform_root = match self.platform {
            Platform::Linux => self.env_path("HOME").map(|home| home.join(".cache")),
            Platform::MacOs => self
                .env_path("HOME")
                .map(|home| home.join("Library").join("Caches")),
            Platform::Windows => {
                // DEVIATION(19): Windows cache uses %LOCALAPPDATA% and config uses %APPDATA%, diverging from upstream's %APPDATA% for both; see §14 item 19 and §16.33 item 3.
                self.env_path("LOCALAPPDATA")
            }
        };

        platform_root
            .map(|root| root.join(APP_DIR_NAME))
            .unwrap_or_else(|| PathBuf::from(".panel-ocr-cache"))
    }

    pub fn config_dir(&self) -> PathBuf {
        if let Some(xdg) = self.env_path("XDG_CONFIG_HOME") {
            return xdg.join(APP_DIR_NAME);
        }

        let platform_root = match self.platform {
            Platform::Linux => self.env_path("HOME").map(|home| home.join(".config")),
            Platform::MacOs => self
                .env_path("HOME")
                .map(|home| home.join("Library").join("Application Support")),
            Platform::Windows => self.env_path("APPDATA"),
        };

        platform_root
            .map(|root| root.join(APP_DIR_NAME))
            .unwrap_or_else(|| PathBuf::from(".panel-ocr"))
    }

    pub fn config_path(&self) -> PathBuf {
        self.config_dir().join(CONFIG_FILE_NAME)
    }

    fn env_path(&self, key: &str) -> Option<PathBuf> {
        self.env.var(key).map(PathBuf::from)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shell {
    Posix,
    PowerShell,
}

impl Shell {
    pub const HOST: Shell = Shell::for_target(Platform::HOST);

    pub const fn for_target(platform: Platform) -> Shell {
        match platform {
            Platform::Linux | Platform::MacOs => Shell::Posix,
            Platform::Windows => Shell::PowerShell,
        }
    }

    pub fn quote(self, path: &Path) -> String {
        let path = path.display().to_string();
        match self {
            Shell::Posix => format!("'{}'", path.replace('\'', "'\\''")),
            Shell::PowerShell => format!("'{}'", path.replace('\'', "''")),
        }
    }
}

/// Per-image cache artifacts (§4.2's `{uuid}_{stem}{suffix}` entries) live in this
/// subdirectory of the cache root, **not** in the root itself: a `clean` run without
/// `--keep-cache` deletes its whole cache directory, and the (future, task D1) model cache
/// at [`MODELS_SUBDIR`] shares the same root. Keeping the two in sibling subdirectories is
/// what stops an ordinary run from deleting downloaded model weights.
pub const IMAGES_SUBDIR: &str = "images";

/// Where task D1's downloaded models live, relative to the cache root. `models path`
/// prints this, and only `cache clear [--models]` may remove it.
pub const MODELS_SUBDIR: &str = "models";

/// The pipeline's `PipelineOptions::cache_dir` for a run rooted at `cache_root`.
pub fn image_cache_dir(cache_root: &std::path::Path) -> PathBuf {
    cache_root.join(IMAGES_SUBDIR)
}

/// The model cache directory for a run rooted at `cache_root`.
pub fn models_dir(cache_root: &std::path::Path) -> PathBuf {
    cache_root.join(MODELS_SUBDIR)
}

/// Resolve the shared cache root: an explicit CLI override wins over the app config,
/// which wins over the platform default (spec §16.18 item 1).
pub fn resolve_cache_root(cli_override: Option<&Path>, config: &Config) -> PathBuf {
    cli_override
        .map(Path::to_path_buf)
        .or_else(|| config.cache_dir.clone())
        .unwrap_or_else(default_cache_dir)
}

pub fn default_cache_dir() -> PathBuf {
    DirEnv::new(Platform::HOST, &ProcessEnv).cache_dir()
}

pub fn default_config_dir() -> PathBuf {
    DirEnv::new(Platform::HOST, &ProcessEnv).config_dir()
}

pub fn default_config_path() -> PathBuf {
    DirEnv::new(Platform::HOST, &ProcessEnv).config_path()
}
