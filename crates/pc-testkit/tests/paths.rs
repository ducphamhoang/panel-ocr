//! C4 tests -- spec §7.1/§7.2 fixture resolution. Frozen gates.
//!
//! The whole point of `paths` is cwd-independence, so these tests deliberately assert
//! properties of the resolved paths rather than string-comparing against a relative
//! path that would happen to work from the workspace root.

use pc_testkit::images;
use pc_testkit::paths::{self, BubbleKind, DEMO_BUBBLES, LONG_STRIP_DPI, LONG_STRIP_SIZE};

#[test]
// spec §7.1: fixture roots resolve to absolute, existing directories, anchored on
// CARGO_MANIFEST_DIR rather than the current working directory
fn fixture_roots_are_absolute_and_exist() {
    for dir in [
        paths::workspace_root(),
        paths::fixtures_root(),
        paths::upstream_root(),
        paths::recorded_root(),
    ] {
        assert!(dir.is_absolute(), "{dir:?} must be absolute");
        assert!(dir.is_dir(), "{dir:?} must exist as a directory");
    }
}

#[test]
// spec §1 layout: the roots nest exactly as the workspace tree says
fn fixture_roots_nest_correctly() {
    assert_eq!(
        paths::fixtures_root(),
        paths::workspace_root().join("tests/fixtures")
    );
    assert_eq!(
        paths::upstream_root(),
        paths::fixtures_root().join("upstream")
    );
    assert_eq!(
        paths::recorded_root(),
        paths::fixtures_root().join("recorded")
    );
    assert!(paths::workspace_root().join("Cargo.toml").is_file());
}

#[test]
// spec §7.1: resolution must not depend on the process's current directory -- this is
// the one property that makes pc-testkit usable from every crate's tests
fn resolution_is_independent_of_cwd() {
    let before = paths::upstream_root();
    let tmp = std::env::temp_dir();
    let restore = std::env::current_dir().unwrap();
    std::env::set_current_dir(&tmp).unwrap();
    let after = paths::upstream_root();
    std::env::set_current_dir(restore).unwrap();
    assert_eq!(before, after);
}

#[test]
// spec §7.1: all 7 demo_bubbles raw/clean pairs exist, and their MEASURED sizes match
// ATTRIBUTION.md exactly. This is the fixture-integrity gate: if someone re-vendors
// from a different upstream commit without updating the table, this fails.
fn demo_bubble_sizes_match_attribution() {
    assert_eq!(DEMO_BUBBLES.len(), 7);
    for bubble in DEMO_BUBBLES {
        for kind in [BubbleKind::Raw, BubbleKind::Clean] {
            let path = bubble.path(kind);
            assert!(path.is_file(), "missing fixture {path:?}");
            let img = images::load(&path);
            assert_eq!(
                (img.width(), img.height()),
                (bubble.width, bubble.height),
                "size mismatch for {path:?}"
            );
        }
    }
}

#[test]
// spec §7.1: "all demo_bubbles are 8-bit GRAYSCALE PNGs" -- downstream masking tests
// (§10.6) rely on this, so the colour model is asserted, not assumed
fn demo_bubbles_are_eight_bit_grayscale() {
    for bubble in DEMO_BUBBLES {
        for kind in [BubbleKind::Raw, BubbleKind::Clean] {
            let path = bubble.path(kind);
            let img = images::load(&path);
            assert_eq!(
                img.color(),
                image::ColorType::L8,
                "{path:?} must be 8-bit grayscale"
            );
        }
    }
}

#[test]
// spec §7.1: the raw/clean pair for each fixture shares one size
fn demo_bubble_pairs_share_a_size() {
    for bubble in DEMO_BUBBLES {
        let raw = images::load(bubble.path(BubbleKind::Raw));
        let clean = images::load(bubble.path(BubbleKind::Clean));
        assert_eq!(
            (raw.width(), raw.height()),
            (clean.width(), clean.height()),
            "{} raw/clean size mismatch",
            bubble.name
        );
    }
}

#[test]
// spec §7.1: long_strip.jpg is 1000x8000 (its dimensions drive §8.7(A)1's resize table
// and §8.7(B)8's 3-split expectation, so a wrong file must fail loudly here)
fn long_strip_dimensions() {
    let path = paths::long_strip();
    assert!(path.is_file(), "missing {path:?}");
    let img = images::load(&path);
    assert_eq!((img.width(), img.height()), LONG_STRIP_SIZE);
    assert_eq!(LONG_STRIP_SIZE, (1000, 8000));
    assert_eq!(LONG_STRIP_DPI, (300, 300));
}

#[test]
// spec §12.6: the two OCR-report format goldens are present and non-empty
fn ocr_output_fixtures_exist() {
    for file in ["good_detected_text.csv", "good_detected_text.txt"] {
        let path = paths::ocr_output(file);
        assert!(path.is_file(), "missing {path:?}");
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
    }
}

#[test]
// spec §7.1: a typo'd fixture name must panic naming the path, not silently produce a
// non-existent PathBuf that fails later as a confusing decode error
#[should_panic(expected = "no_such_fixture.png")]
fn missing_upstream_fixture_panics_naming_the_path() {
    let _ = paths::upstream("no_such_fixture.png");
}

#[test]
// spec §7.2: recorded fixtures are maintainer-generated, so a missing one must point
// the reader at `cargo xtask record-fixtures` rather than just saying "not found"
#[should_panic(expected = "record-fixtures")]
fn missing_recorded_fixture_panics_with_the_xtask_hint() {
    let _ = paths::recorded("definitely_not_recorded#raw.json");
}

#[test]
// spec §7.2: `recorded_opt` is the non-panicking probe, so a test can skip gracefully
// in a checkout where `record-fixtures` has not been run
fn recorded_opt_returns_none_for_a_missing_fixture() {
    assert_eq!(
        paths::recorded_opt("definitely_not_recorded#raw.json"),
        None
    );
}

#[test]
// spec §7.1: the demo_bubble lookup rejects unknown names loudly and lists the valid
// ones, so a typo in a stage test is immediately self-diagnosing
#[should_panic(expected = "nightmare")]
fn unknown_demo_bubble_name_panics_listing_valid_names() {
    let _ = paths::demo_bubble("nightmarish");
}
