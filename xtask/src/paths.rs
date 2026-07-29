//! Repo-root-anchored paths. `CARGO_MANIFEST_DIR` of this crate is `<repo>/xtask`.

use std::path::{Path, PathBuf};

#[allow(unused_imports)]
pub use pc_testkit::paths::{fixtures_root, recorded_root, upstream_root, workspace_root};

pub fn docs_root() -> PathBuf {
    workspace_root().join("docs")
}

/// A committed recording script (§11.6: "the recording script is committed so the
/// provenance is auditable").
pub fn script(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join(name)
}

/// Scratch space for intermediates that must NOT be committed (§16.13 item 6): the
/// lossless JPEG re-decode and the decoder-diagnostic resize are 8 MB of derived data
/// whose only consumer is the recording run itself.
pub fn scratch_dir() -> std::io::Result<PathBuf> {
    let dir = workspace_root().join("target/xtask-scratch");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Rewrite every string value in a recording manifest that names a path inside the
/// workspace into a repo-relative one, so `PROVENANCE.json` is committable and
/// reviewable rather than carrying one machine's home directory.
pub fn relativize_manifest(value: &mut serde_json::Value) {
    let root = workspace_root().display().to_string();
    let prefix = format!("{root}/");
    fn walk(value: &mut serde_json::Value, prefix: &str) {
        match value {
            serde_json::Value::String(text) => {
                if let Some(rest) = text.strip_prefix(prefix) {
                    *text = rest.to_string();
                }
            }
            serde_json::Value::Array(items) => items.iter_mut().for_each(|it| walk(it, prefix)),
            serde_json::Value::Object(map) => map.values_mut().for_each(|it| walk(it, prefix)),
            _ => {}
        }
    }
    walk(value, &prefix);
}

/// Path relative to the workspace root when possible — log lines that paste into a
/// `git add` are worth the three lines this costs.
pub fn display_relative(path: &Path) -> String {
    path.strip_prefix(workspace_root())
        .unwrap_or(path)
        .display()
        .to_string()
}
