//! Config and cache location discovery (spec §16.12 item 21).
//!
//! v1 is Linux + macOS only, and `dirs` is not in `[workspace.dependencies]`, so this is
//! done by hand against the two platforms' conventions. Fully pinned; implemented.

use pc_config::Config;
use std::path::{Path, PathBuf};

pub const APP_DIR_NAME: &str = "panel-ocr";
pub const CONFIG_FILE_NAME: &str = "config.toml";

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

/// `$XDG_CACHE_HOME/panel-ocr`, else `~/Library/Caches/panel-ocr` on macOS,
/// else `~/.cache/panel-ocr`, else `./.panel-ocr-cache`.
pub fn default_cache_dir() -> PathBuf {
    if let Some(xdg) = env_path("XDG_CACHE_HOME") {
        return xdg.join(APP_DIR_NAME);
    }
    if let Some(home) = env_path("HOME") {
        if cfg!(target_os = "macos") {
            return home.join("Library").join("Caches").join(APP_DIR_NAME);
        }
        return home.join(".cache").join(APP_DIR_NAME);
    }
    PathBuf::from(".panel-ocr-cache")
}

/// `$XDG_CONFIG_HOME/panel-ocr`, else `~/Library/Application Support/panel-ocr` on
/// macOS, else `~/.config/panel-ocr`, else `./.panel-ocr`.
pub fn default_config_dir() -> PathBuf {
    if let Some(xdg) = env_path("XDG_CONFIG_HOME") {
        return xdg.join(APP_DIR_NAME);
    }
    if let Some(home) = env_path("HOME") {
        if cfg!(target_os = "macos") {
            return home
                .join("Library")
                .join("Application Support")
                .join(APP_DIR_NAME);
        }
        return home.join(".config").join(APP_DIR_NAME);
    }
    PathBuf::from(".panel-ocr")
}

/// `{config_dir}/config.toml`.
pub fn default_config_path() -> PathBuf {
    default_config_dir().join(CONFIG_FILE_NAME)
}

fn env_path(key: &str) -> Option<PathBuf> {
    match std::env::var_os(key) {
        Some(value) if !value.is_empty() => Some(PathBuf::from(value)),
        _ => None,
    }
}
