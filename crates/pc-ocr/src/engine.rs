//! spec §9.2 — the two OCR traits, verbatim from the spec's signature block.

use image::DynamicImage;
use pc_core::{Language, StageError};

/// One recogniser. `Send + Sync` because §4.5 parallelises whole images over rayon and
/// a single engine is shared across those threads (same rule as `TextDetector`).
pub trait OcrEngine: Send + Sync {
    /// Languages this engine claims to handle. Advisory: `OcrEngineFactory` owns the
    /// actual routing decision.
    fn languages(&self) -> &[Language];

    /// `crop`: one text box cut out of `PageData::base_image`, in base-image scale.
    ///
    /// Returning `Err` is **not** fatal to the page: spec §9.3 step 7 / §14.9 make the
    /// preprocess stage fail *open* (the box is kept, the failure is logged at `WARN`).
    fn recognize(&self, crop: &DynamicImage) -> Result<String, StageError>;
}

/// Routes a box's language to an engine. Returning `None` means "no engine handles this
/// language" — per spec §16.8 item 4 the caller then **keeps** the box, un-OCR'd, and
/// does not count it in `OcrAnalytic::box_areas_ocred`.
pub trait OcrEngineFactory: Send + Sync {
    /// `None` language = unknown; the factory picks a best-effort engine or returns `None`.
    fn engine_for(&self, lang: Option<Language>) -> Option<&dyn OcrEngine>;
}
