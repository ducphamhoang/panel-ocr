//! spec §2.9
//!
//! `StageError` is NEVER used for "this box was noise" or "this mask didn't fit" —
//! those are normal outcomes recorded in analytics (§2.9, §5.6). Only conditions that
//! make the *image* unprocessable are errors.
//!
//! This is the ONLY error type `pc-core` defines. An earlier spec draft's `ImageHandle`
//! section referenced a separate `CoreError` type; that was a naming slip and has been
//! corrected in the spec (§2.3) to say `StageError` throughout.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum StageError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to decode image {path}: {source}")]
    Decode {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("unsupported image format: {0}")]
    UnsupportedFormat(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("invalid stage input: {0}")]
    InvalidInput(String),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("image handle is not materialized on disk")]
    UnmaterializedHandle,
    #[error("stage produced no usable output: {0}")]
    Empty(String),
}
