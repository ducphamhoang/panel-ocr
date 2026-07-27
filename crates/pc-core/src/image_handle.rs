//! spec §2.3 — how images cross stage boundaries.

use crate::error::StageError;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize)]
pub struct ImageHandle {
    /// Where the image lives (or will live) on disk. `None` only in pure in-memory runs.
    pub path: Option<PathBuf>,
    #[serde(skip)]
    cached: Option<Arc<DynamicImage>>,
}

/// Hand-written (not derived): fails with a serde error when `path.is_none()`, so a
/// path-less (in-memory-only) handle can never be written into a JSON checkpoint —
/// see `ensure_materialized`'s doc comment for why this is enforced at two points.
impl Serialize for ImageHandle {
    fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        todo!()
    }
}

impl ImageHandle {
    pub fn from_path(_p: impl Into<PathBuf>) -> Self {
        todo!()
    }

    /// `path: None` — an in-memory-only handle. Cannot be checkpointed (see
    /// `ensure_materialized`).
    pub fn from_memory(_img: DynamicImage) -> Self {
        todo!()
    }

    pub fn with_both(_p: impl Into<PathBuf>, _img: DynamicImage) -> Self {
        todo!()
    }

    /// Cached image if present, else decode from `path`.
    /// `path: None` and no cache => `StageError::UnmaterializedHandle`.
    pub fn load(&self) -> Result<Arc<DynamicImage>, StageError> {
        todo!()
    }

    /// Decode-free size query: cache if present, else the image *header* only.
    pub fn dimensions(&self) -> Result<(u32, u32), StageError> {
        todo!()
    }

    /// True iff `path` is `Some` and that path exists on disk.
    pub fn is_materialized(&self) -> bool {
        todo!()
    }

    /// spec §2.3 invariant, in callable form: `Ok(())` iff this handle can survive a
    /// checkpoint round-trip (i.e. `path.is_some()`), else
    /// `StageError::UnmaterializedHandle`.
    ///
    /// Enforced at TWO points (decided during Rust Engineer review): this pre-flight
    /// check, AND a hand-written `Serialize` impl (below) that itself fails when
    /// `path.is_none()` — so no code path can silently smuggle a path-less handle
    /// into a JSON checkpoint, even one that forgets to call this method first.
    pub fn ensure_materialized(&self) -> Result<(), StageError> {
        todo!()
    }
}

/// spec §2.3 (decided): `PartialEq` compares `path` only — `cached` is `#[serde(skip)]`
/// and comparing decoded pixel buffers would be a semantically wrong value comparison.
impl PartialEq for ImageHandle {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

