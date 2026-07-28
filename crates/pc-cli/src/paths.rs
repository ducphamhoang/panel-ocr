//! Config and cache location discovery (spec §16.12 item 21).
//!
//! v1 is Linux + macOS only, and `dirs` is not in `[workspace.dependencies]`, so this is
//! done by hand against the two platforms' conventions. Fully pinned; implemented.

use std::path::PathBuf;

pub const APP_DIR_NAME: &str = "panel-ocr";
pub const CONFIG_FILE_NAME: &str = "config.toml";

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
