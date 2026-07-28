//! spec §7.1/§7.2 -- fixture location, independent of the current working directory.
//!
//! Anchor: `CARGO_MANIFEST_DIR` of *this* crate is `<repo>/crates/pc-testkit`, so the
//! workspace root is two levels up. This is stable no matter which crate's test binary
//! is running and no matter what cwd cargo chooses -- which is exactly the property
//! `std::env::current_dir()` does not have.

use pc_core::{MaskData, PageDataRaw};
use std::path::{Path, PathBuf};

/// `<repo>` -- the cargo workspace root.
pub fn workspace_root() -> PathBuf {
    todo!()
}

/// `<repo>/tests/fixtures`
pub fn fixtures_root() -> PathBuf {
    todo!()
}

/// `<repo>/tests/fixtures/upstream` (§7.1, vendored PanelCleaner assets, GPL-3).
pub fn upstream_root() -> PathBuf {
    todo!()
}

/// `<repo>/tests/fixtures/recorded` (§7.2, recorded real-model outputs).
pub fn recorded_root() -> PathBuf {
    todo!()
}

/// `upstream_root().join(rel)`, asserting the file exists -- a missing vendored fixture
/// is a broken checkout, and the panic must say so rather than surfacing later as a
/// confusing decode error.
pub fn upstream(rel: impl AsRef<Path>) -> PathBuf {
    todo!()
}

/// `recorded_root().join(rel)`, asserting the file exists. The panic message must point
/// at `cargo xtask record-fixtures` (§7.2) since these are maintainer-generated.
pub fn recorded(rel: impl AsRef<Path>) -> PathBuf {
    todo!()
}

/// Like `recorded`, but returns `None` instead of panicking when the fixture has not
/// been recorded yet. Lets a test `return` early with a `WARN` in a fresh checkout.
pub fn recorded_opt(rel: impl AsRef<Path>) -> Option<PathBuf> {
    todo!()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BubbleKind {
    /// `<name>_bubble_raw.png` -- the input.
    Raw,
    /// `<name>_bubble_clean.png` -- upstream's output. Per §15.2 / ATTRIBUTION.md this
    /// is a **calibration** fixture: no pass/fail assertion may be written against it.
    Clean,
}

/// spec §7.1 measured properties, transcribed from
/// `tests/fixtures/upstream/ATTRIBUTION.md`. `raw` and `clean` share one size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DemoBubble {
    pub name: &'static str,
    pub width: u32,
    pub height: u32,
}

/// All 7 pairs, in the order §10.6 lists them alphabetically. All are 8-bit
/// **grayscale** PNGs.
pub const DEMO_BUBBLES: &[DemoBubble] = &[
    DemoBubble {
        name: "black",
        width: 202,
        height: 319,
    },
    DemoBubble {
        name: "darkrays",
        width: 208,
        height: 320,
    },
    DemoBubble {
        name: "handwritten",
        width: 72,
        height: 132,
    },
    DemoBubble {
        name: "nightmare",
        width: 219,
        height: 343,
    },
    DemoBubble {
        name: "ray",
        width: 256,
        height: 329,
    },
    DemoBubble {
        name: "spikey",
        width: 354,
        height: 354,
    },
    DemoBubble {
        name: "square",
        width: 144,
        height: 270,
    },
];

/// `long_strip.jpg` is 1000x8000 RGB, progressive JPEG, 300 dpi (§7.1).
pub const LONG_STRIP_SIZE: (u32, u32) = (1000, 8000);
pub const LONG_STRIP_DPI: (u32, u32) = (300, 300);

impl DemoBubble {
    pub fn path(&self, kind: BubbleKind) -> PathBuf {
        todo!()
    }
    pub fn size(&self) -> (u32, u32) {
        todo!()
    }
    /// The §7.2 recorded artifact for this fixture, e.g. `<name>_raw_mask.png`.
    pub fn recorded(&self, suffix: &str) -> Option<PathBuf> {
        todo!()
    }
}

/// Look up a demo bubble by short name (`"black"`, not `"black_bubble_raw"`).
/// Panics on an unknown name, listing the valid ones.
pub fn demo_bubble(name: &str) -> DemoBubble {
    todo!()
}

/// `<repo>/tests/fixtures/upstream/long_strip.jpg`
pub fn long_strip() -> PathBuf {
    todo!()
}

/// `<repo>/tests/fixtures/upstream/ocr_output/<file>`
pub fn ocr_output(file: &str) -> PathBuf {
    todo!()
}

// ------------------------------------------------------- §7.2 path rebasing

/// spec §7.2: "The recorded JSON's `ImageHandle` paths are stored **relative to the
/// fixtures root** and rebased on load by `pc-testkit`." Rewrites every relative
/// handle path to `root.join(path)`; absolute paths are left alone.
pub fn rebase_page_data_raw(page: &mut PageDataRaw, root: &Path) {
    todo!()
}

/// Same contract as `rebase_page_data_raw`, for `MaskData`'s two handles.
pub fn rebase_mask_data(mask: &mut MaskData, root: &Path) {
    todo!()
}

/// Read a recorded `<name>#raw.json`, deserialize, and rebase its handles against
/// `fixtures_root()`. Panics with a `cargo xtask record-fixtures` hint if absent.
pub fn load_recorded_page_raw(rel: impl AsRef<Path>) -> PageDataRaw {
    todo!()
}
