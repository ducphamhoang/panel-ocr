//! Task **G1** — `CachePaths` (spec §4.2) and `discover` (§16.12 item 8).
//!
//! FROZEN (CLAUDE.md): these tests may not be edited once written; only the
//! implementation is iterated.

use pc_core::Output;
use pc_pipeline::cache::{known_suffixes, CachePaths, SEGMENT_INFIX, SPLITS_SUFFIX};
use std::path::Path;
use uuid::Uuid;

fn fixed_uuid() -> Uuid {
    Uuid::parse_str("00000000-0000-4000-8000-000000000001").unwrap()
}

/// §4.2: `{cache}/{uuid}_{stem}{suffix}`, with the suffix taken verbatim from §2.8.
#[test]
fn for_output_uses_the_upstream_name_format() {
    let paths = CachePaths::from_parts(Path::new("/cache"), "page01", fixed_uuid());

    assert_eq!(
        paths.for_output(Output::RawJson),
        Path::new("/cache/00000000-0000-4000-8000-000000000001_page01#raw.json")
    );
    assert_eq!(
        paths.for_output(Output::BaseImage),
        Path::new("/cache/00000000-0000-4000-8000-000000000001_page01_base.png")
    );
    assert_eq!(
        paths.for_output(Output::DenoisedOutput),
        Path::new("/cache/00000000-0000-4000-8000-000000000001_page01_clean_denoised.png")
    );
}

/// §16.12 item 10 — the two pipeline-local suffixes are not `Output` variants.
#[test]
fn split_artifacts_have_pipeline_local_names() {
    let paths = CachePaths::from_parts(Path::new("/cache"), "strip", fixed_uuid());

    assert!(paths
        .splits_manifest()
        .to_string_lossy()
        .ends_with(SPLITS_SUFFIX));
    assert!(paths
        .segment(7)
        .to_string_lossy()
        .ends_with(&format!("{SEGMENT_INFIX}007.png")));
    assert!(!Output::ALL
        .iter()
        .any(|output| output.cache_suffix() == SPLITS_SUFFIX));
}

/// §4.2's uuid recovery: every `Output`'s own path parses back to the same uuid + stem.
#[test]
fn from_existing_round_trips_every_output() {
    let cache = Path::new("/cache");
    let paths = CachePaths::from_parts(cache, "some_page_02", fixed_uuid());

    for output in Output::ALL {
        let path = paths.for_output(*output);
        let recovered = CachePaths::from_existing(&path, cache)
            .unwrap_or_else(|error| panic!("{output:?} did not round-trip: {error}"));
        assert_eq!(recovered.uuid(), fixed_uuid(), "{output:?}");
        assert_eq!(recovered.stem(), "some_page_02", "{output:?}");
    }
}

/// §16.12 item 9 — longest-match, so `_clean_denoised.png` is not read as `_clean.png`
/// with a stem ending in `_denoised`.
#[test]
fn suffixes_are_matched_longest_first() {
    let suffixes = known_suffixes();
    let lengths: Vec<usize> = suffixes.iter().map(|s| s.len()).collect();
    assert!(
        lengths.windows(2).all(|pair| pair[0] >= pair[1]),
        "known_suffixes must be ordered longest-first, got {suffixes:?}"
    );

    let cache = Path::new("/cache");
    let path =
        CachePaths::from_parts(cache, "page", fixed_uuid()).for_output(Output::DenoisedOutput);
    let recovered = CachePaths::from_existing(&path, cache).unwrap();
    assert_eq!(recovered.stem(), "page");
}

#[test]
fn from_existing_rejects_names_without_a_uuid_or_a_known_suffix() {
    let cache = Path::new("/cache");
    assert!(CachePaths::from_existing(Path::new("/cache/page_base.png"), cache).is_err());
    assert!(CachePaths::from_existing(
        Path::new("/cache/00000000-0000-4000-8000-000000000001_page.txt"),
        cache
    )
    .is_err());
}

/// §4.2: a fresh `CachePaths` gets a fresh uuid, so two runs over the same file never
/// clobber each other.
#[test]
fn new_generates_distinct_uuids() {
    let original = Path::new("/inputs/page01.png");
    let first = CachePaths::new(original, Path::new("/cache"));
    let second = CachePaths::new(original, Path::new("/cache"));

    assert_eq!(first.stem(), "page01");
    assert_ne!(first.uuid(), second.uuid());
}

/// §16.12 item 8 — resume finds the entry an earlier run left.
#[test]
fn discover_finds_the_entry_for_an_image() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    let paths = CachePaths::from_parts(cache, "page01", fixed_uuid());
    std::fs::write(paths.for_output(Output::RawJson), b"{}").unwrap();
    std::fs::write(paths.for_output(Output::BaseImage), b"x").unwrap();
    // A different image's entry must not be picked up.
    std::fs::write(
        CachePaths::from_parts(cache, "page02", Uuid::new_v4()).for_output(Output::RawJson),
        b"{}",
    )
    .unwrap();

    let found = CachePaths::discover(cache, Path::new("/inputs/page01.png"))
        .unwrap()
        .expect("an entry for page01");
    assert_eq!(found.uuid(), fixed_uuid());
    assert_eq!(found.stem(), "page01");
}

/// §16.12 item 8 — deterministic tie-break: the lexicographically smallest uuid wins,
/// never mtime or readdir order (§5.7).
#[test]
fn discover_is_deterministic_with_several_candidates() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    let low = Uuid::parse_str("00000000-0000-4000-8000-00000000000a").unwrap();
    let high = Uuid::parse_str("ffffffff-0000-4000-8000-00000000000b").unwrap();
    for uuid in [high, low] {
        std::fs::write(
            CachePaths::from_parts(cache, "page01", uuid).for_output(Output::RawJson),
            b"{}",
        )
        .unwrap();
    }

    for _ in 0..5 {
        let found = CachePaths::discover(cache, Path::new("page01.png"))
            .unwrap()
            .unwrap();
        assert_eq!(found.uuid(), low);
    }
}

#[test]
fn discover_returns_none_when_nothing_matches() {
    let dir = tempfile::tempdir().unwrap();
    assert!(CachePaths::discover(dir.path(), Path::new("absent.png"))
        .unwrap()
        .is_none());
    assert!(
        CachePaths::discover(&dir.path().join("no-such-dir"), Path::new("absent.png"))
            .unwrap()
            .is_none()
    );
}
