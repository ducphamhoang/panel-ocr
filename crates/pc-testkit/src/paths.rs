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
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("pc-testkit workspace root must exist")
}

/// `<repo>/tests/fixtures`
pub fn fixtures_root() -> PathBuf {
    workspace_root().join("tests/fixtures")
}

/// `<repo>/tests/fixtures/upstream` (§7.1, vendored PanelCleaner assets, GPL-3).
pub fn upstream_root() -> PathBuf {
    fixtures_root().join("upstream")
}

/// `<repo>/tests/fixtures/recorded` (§7.2, recorded real-model outputs).
pub fn recorded_root() -> PathBuf {
    fixtures_root().join("recorded")
}

/// `upstream_root().join(rel)`, asserting the file exists -- a missing vendored fixture
/// is a broken checkout, and the panic must say so rather than surfacing later as a
/// confusing decode error.
pub fn upstream(rel: impl AsRef<Path>) -> PathBuf {
    let path = upstream_root().join(rel);
    assert!(
        path.is_file(),
        "missing upstream fixture `{}` (broken checkout)",
        path.display()
    );
    path
}

/// `recorded_root().join(rel)`, asserting the file exists. The panic message must point
/// at `cargo xtask record-fixtures` (§7.2) since these are maintainer-generated.
pub fn recorded(rel: impl AsRef<Path>) -> PathBuf {
    let path = recorded_root().join(rel);
    assert!(
        path.is_file(),
        "missing recorded fixture `{}`; run `cargo xtask record-fixtures`",
        path.display()
    );
    path
}

/// Like `recorded`, but returns `None` instead of panicking when the fixture has not
/// been recorded yet. Lets a test `return` early with a `WARN` in a fresh checkout.
pub fn recorded_opt(rel: impl AsRef<Path>) -> Option<PathBuf> {
    let path = recorded_root().join(rel);
    path.is_file().then_some(path)
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
        let kind = match kind {
            BubbleKind::Raw => "raw",
            BubbleKind::Clean => "clean",
        };
        upstream(format!("demo_bubbles/{}_bubble_{kind}.png", self.name))
    }
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    /// The §7.2 recorded artifact for this fixture, e.g. `<name>_raw_mask.png`.
    pub fn recorded(&self, suffix: &str) -> Option<PathBuf> {
        recorded_opt(format!("{}_{suffix}", self.name))
    }
}

/// Look up a demo bubble by short name (`"black"`, not `"black_bubble_raw"`).
/// Panics on an unknown name, listing the valid ones.
pub fn demo_bubble(name: &str) -> DemoBubble {
    DEMO_BUBBLES
        .iter()
        .copied()
        .find(|bubble| bubble.name == name)
        .unwrap_or_else(|| {
            let valid = DEMO_BUBBLES
                .iter()
                .map(|bubble| bubble.name)
                .collect::<Vec<_>>()
                .join(", ");
            panic!("unknown demo bubble `{name}`; valid names: {valid}")
        })
}

/// `<repo>/tests/fixtures/upstream/long_strip.jpg`
pub fn long_strip() -> PathBuf {
    upstream("long_strip.jpg")
}

/// `<repo>/tests/fixtures/upstream/ocr_output/<file>`
pub fn ocr_output(file: &str) -> PathBuf {
    upstream(Path::new("ocr_output").join(file))
}

// ------------------------------------------------------- §7.2 path rebasing

/// spec §7.2: "The recorded JSON's `ImageHandle` paths are stored **relative to the
/// fixtures root** and rebased on load by `pc-testkit`." Rewrites every relative
/// handle path to `root.join(path)`; absolute paths are left alone.
pub fn rebase_page_data_raw(page: &mut PageDataRaw, root: &Path) {
    rebase_handle_path(&mut page.base_image.path, root);
    rebase_handle_path(&mut page.raw_mask.path, root);
}

/// Same contract as `rebase_page_data_raw`, for `MaskData`'s two handles.
pub fn rebase_mask_data(mask: &mut MaskData, root: &Path) {
    rebase_handle_path(&mut mask.base_image.path, root);
    rebase_handle_path(&mut mask.combined_mask.path, root);
}

/// Read a recorded `<name>#raw.json`, deserialize, and rebase its handles against
/// `fixtures_root()`. Panics with a `cargo xtask record-fixtures` hint if absent.
pub fn load_recorded_page_raw(rel: impl AsRef<Path>) -> PageDataRaw {
    let path = recorded(rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "failed to read recorded fixture `{}`: {error}",
            path.display()
        )
    });
    let mut page: PageDataRaw = serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        panic!(
            "failed to deserialize recorded fixture `{}`: {error}",
            path.display()
        )
    });
    rebase_page_data_raw(&mut page, &fixtures_root());
    page
}

fn rebase_handle_path(path: &mut Option<PathBuf>, root: &Path) {
    if let Some(relative) = path.as_ref().filter(|path| path.is_relative()) {
        *path = Some(root.join(relative));
    }
}
