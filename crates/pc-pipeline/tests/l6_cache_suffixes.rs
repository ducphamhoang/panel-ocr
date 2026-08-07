//! L6-2 tests — §16.38 items 11(d), 11(e), 24(b), and 25(b).
//!
//! FROZEN (CLAUDE.md): once the intended red baseline is observed, only production code
//! may be changed.

use pc_pipeline::cache::{known_suffixes, CachePaths, CLEAN_INPAINT_SUFFIX, INPAINTING_SUFFIX};
use std::path::Path;
use uuid::Uuid;

fn fixed_uuid() -> Uuid {
    Uuid::parse_str("00000000-0000-4000-8000-000000000001").unwrap()
}

/// §16.38 items 11(d) and 24(b): the two upstream-verbatim inpainting suffixes are
/// pipeline-local constants, are known to cache-name recovery, and each recovers the exact
/// UUID and stem from a path constructed with that suffix.
#[test]
fn both_inpainting_suffixes_round_trip_through_cache_name_recovery() {
    assert_eq!(INPAINTING_SUFFIX, "_inpainting.png");
    assert_eq!(CLEAN_INPAINT_SUFFIX, "_clean_inpaint.png");

    let cache = Path::new("/cache");
    let paths = CachePaths::from_parts(cache, "page_with_underscores", fixed_uuid());
    let known = known_suffixes();

    for suffix in [INPAINTING_SUFFIX, CLEAN_INPAINT_SUFFIX] {
        assert!(
            known.contains(&suffix),
            "{suffix:?} is absent from known_suffixes(): {known:?}"
        );
        let recovered = CachePaths::from_existing(&paths.for_suffix(suffix), cache)
            .unwrap_or_else(|error| panic!("{suffix:?} did not round-trip: {error}"));
        assert_eq!(recovered.uuid(), fixed_uuid(), "{suffix:?}");
        assert_eq!(recovered.stem(), "page_with_underscores", "{suffix:?}");
    }
}

/// §16.38 items 11(e), 24(b), and 25(b): after adding the two inpainting suffixes, no
/// suffix recognised by `CachePaths::from_existing` is a proper tail of another recognised
/// suffix. Testing every ordered pair pins the real invariant rather than the nonexistent
/// `_clean.png` collision discussed in item 25(b).
#[test]
fn no_known_cache_suffix_is_a_proper_tail_of_another() {
    let suffixes = known_suffixes();
    assert_eq!(
        suffixes.len(),
        16,
        "13 Output suffixes plus SPLITS_SUFFIX and the two L6-2 suffixes"
    );

    for (index, suffix) in suffixes.iter().enumerate() {
        for (other_index, other) in suffixes.iter().enumerate() {
            if index != other_index {
                assert!(
                    !other.ends_with(suffix),
                    "known suffix {suffix:?} is a proper tail of {other:?}; all suffixes: {suffixes:?}"
                );
            }
        }
    }
}
