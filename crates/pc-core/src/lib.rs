//! `pc-core` — the shared vocabulary of the panel-ocr pipeline (spec §2, §3).
//!
//! Nothing in this crate performs stage logic. It owns:
//!   * geometry (`Rect`) with upstream-exact arithmetic (§2.1)
//!   * `Language` + the RTL ordering set (§2.2)
//!   * `ImageHandle`, the in-memory-or-on-disk image contract (§2.3)
//!   * the persisted data contracts `PageDataRaw` / `PageData` / `MaskData` (§2.4-§2.6)
//!   * analytics records (§2.7)
//!   * `Step` / `Output` and their upstream-verbatim cache suffixes (§2.8)
//!   * `StageError` (§2.9) and the `Stage` trait (§3)
//!
//! No crate in the workspace may define an alternative spelling of any of these.
//! `StageError` is the only error type this crate defines — an earlier spec draft
//! referenced a `CoreError` type that does not exist; that was a naming slip,
//! resolved during the Rust Engineer's foundational-test pass.

pub mod analytics;
pub mod error;
pub mod geometry;
pub mod image_handle;
pub mod language;
pub mod mask_data;
pub mod output;
pub mod page;
pub mod stage;

pub use analytics::{
    DenoiseAnalytic, DetectAnalytic, MaskFittingAnalytic, OcrAnalytic, RemovedBox,
};
pub use error::StageError;
pub use geometry::Rect;
pub use image_handle::ImageHandle;
pub use language::{Language, RTL_BOX_ORDER_LANGUAGES};
pub use mask_data::{MaskData, MaskRegionStats};
pub use output::{Output, Step};
pub use page::{DetectedBlock, MaskingRegion, PageData, PageDataRaw, TextBox};
pub use stage::Stage;

/// spec §2: every top-level persisted struct carries `schema_version` as its first
/// field, starting at 1.
pub const SCHEMA_VERSION: u32 = 1;
