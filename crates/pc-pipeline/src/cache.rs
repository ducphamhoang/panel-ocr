//! spec §4.2 — cache path construction and recovery.
//!
//! Format `{uuid}_{stem}{suffix}`, mirroring upstream `OutputPathGenerator`
//! (`output_structures.py:269`): the uuid is clobber protection (two different source
//! files with the same stem must not fight over one cache entry) and is recoverable
//! from the file name via `stem.split('_')[0]`.
//!
//! Everything in this module is fully pinned by §4.2 + §16.12 items 8–10 and is
//! implemented, not stubbed.

use pc_core::{Output, StageError};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// §16.12 item 10: the split manifest's suffix. Not an `Output` variant — `pc-core` is
/// frozen and §2.8's list is closed.
pub const SPLITS_SUFFIX: &str = "#splits.json";

/// §16.38 item 11(d): upstream's intermediate inpainting cache suffix.
pub const INPAINTING_SUFFIX: &str = "_inpainting.png";

/// §16.38 item 11(d): upstream's cleaned inpainting cache suffix.
pub const CLEAN_INPAINT_SUFFIX: &str = "_clean_inpaint.png";

/// §16.12 item 10: `{uuid}_{stem}_seg{index:03}.png`.
pub const SEGMENT_INFIX: &str = "_seg";

/// Every suffix `from_existing` knows how to strip, **longest first** (§16.12 item 9).
pub fn known_suffixes() -> Vec<&'static str> {
    let mut suffixes: Vec<&'static str> = Output::ALL
        .iter()
        .map(|output| output.cache_suffix())
        .collect();
    suffixes.push(SPLITS_SUFFIX);
    suffixes.push(INPAINTING_SUFFIX);
    suffixes.push(CLEAN_INPAINT_SUFFIX);
    suffixes.sort_by_key(|suffix| std::cmp::Reverse(suffix.len()));
    suffixes
}

/// spec §4.2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachePaths {
    cache_dir: PathBuf,
    uuid: Uuid,
    stem: String,
}

impl CachePaths {
    /// A fresh uuid for `original`'s stem.
    pub fn new(original: &Path, cache_dir: &Path) -> Self {
        let stem = original
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default();
        Self {
            cache_dir: cache_dir.to_path_buf(),
            uuid: Uuid::new_v4(),
            stem,
        }
    }

    /// Deterministic constructor for tests and for `discover`.
    pub fn from_parts(cache_dir: &Path, stem: &str, uuid: Uuid) -> Self {
        Self {
            cache_dir: cache_dir.to_path_buf(),
            uuid,
            stem: stem.to_string(),
        }
    }

    /// Recover the uuid + stem from a cache file name (§4.2's uuid recovery).
    ///
    /// `StageError::InvalidInput` when the name has no `_`, when the leading component
    /// is not a uuid, or when the remainder ends in none of [`known_suffixes`].
    pub fn from_existing(path_with_uuid: &Path, cache_dir: &Path) -> Result<Self, StageError> {
        let (uuid, stem) = parse_cache_name(path_with_uuid)?;
        Ok(Self {
            cache_dir: cache_dir.to_path_buf(),
            uuid,
            stem,
        })
    }

    /// spec §16.12 item 8 — find the cache entry an earlier run left for `original`.
    ///
    /// Scans `cache_dir`, keeps entries whose recovered stem matches
    /// `original.file_stem()`, and returns the lexicographically **smallest** uuid among
    /// them (deterministic, §5.7), warning when there is more than one. `Ok(None)` when
    /// nothing matches or the directory does not exist.
    pub fn discover(cache_dir: &Path, original: &Path) -> Result<Option<Self>, StageError> {
        let wanted = original
            .file_stem()
            .ok_or_else(|| {
                StageError::InvalidInput(format!(
                    "input path `{}` has no file stem",
                    original.display()
                ))
            })?
            .to_string_lossy()
            .into_owned();

        if !cache_dir.exists() {
            return Ok(None);
        }

        let entries = std::fs::read_dir(cache_dir).map_err(|source| StageError::Io {
            path: cache_dir.to_path_buf(),
            source,
        })?;

        let mut found: BTreeMap<String, Uuid> = BTreeMap::new();
        for entry in entries {
            let entry = entry.map_err(|source| StageError::Io {
                path: cache_dir.to_path_buf(),
                source,
            })?;
            if let Ok((uuid, stem)) = parse_cache_name(&entry.path()) {
                if stem == wanted {
                    found.insert(uuid.to_string(), uuid);
                }
            }
        }

        if found.len() > 1 {
            tracing::warn!(
                stem = %wanted,
                candidates = found.len(),
                "multiple cache entries for this image; using the lowest uuid (spec §16.12 item 8)"
            );
        }

        Ok(found
            .into_values()
            .next()
            .map(|uuid| Self::from_parts(cache_dir, &wanted, uuid)))
    }

    /// `{cache}/{uuid}_{stem}{suffix}` (§4.2).
    pub fn for_output(&self, out: Output) -> PathBuf {
        self.for_suffix(out.cache_suffix())
    }

    /// `for_output`'s generalisation, for the pipeline-local suffixes of §16.12 item 10.
    pub fn for_suffix(&self, suffix: &str) -> PathBuf {
        self.cache_dir
            .join(format!("{}_{}{}", self.uuid, self.stem, suffix))
    }

    /// `{cache}/{uuid}_{stem}#splits.json`.
    pub fn splits_manifest(&self) -> PathBuf {
        self.for_suffix(SPLITS_SUFFIX)
    }

    /// `{cache}/{uuid}_{stem}_seg{index:03}.png`.
    pub fn segment(&self, index: usize) -> PathBuf {
        self.for_suffix(&format!("{SEGMENT_INFIX}{index:03}.png"))
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn uuid(&self) -> Uuid {
        self.uuid
    }

    pub fn stem(&self) -> &str {
        &self.stem
    }
}

fn parse_cache_name(path: &Path) -> Result<(Uuid, String), StageError> {
    let name = path
        .file_name()
        .ok_or_else(|| StageError::InvalidInput(format!("`{}` has no file name", path.display())))?
        .to_string_lossy()
        .into_owned();

    let (uuid_part, rest) = name.split_once('_').ok_or_else(|| {
        StageError::InvalidInput(format!("cache name `{name}` has no `{{uuid}}_` prefix"))
    })?;
    let uuid = Uuid::parse_str(uuid_part).map_err(|error| {
        StageError::InvalidInput(format!("cache name `{name}` has no uuid prefix: {error}"))
    })?;

    for suffix in known_suffixes() {
        if let Some(stem) = rest.strip_suffix(suffix) {
            return Ok((uuid, stem.to_string()));
        }
    }
    if let Some(index) = rest.rfind(SEGMENT_INFIX) {
        if rest.ends_with(".png") {
            return Ok((uuid, rest[..index].to_string()));
        }
    }

    Err(StageError::InvalidInput(format!(
        "cache name `{name}` ends in no known cache suffix"
    )))
}
