//! P8b regression checks for OCR wiring and stale no-engine diagnostics.

use std::path::Path;

fn rust_sources(root: &Path, sources: &mut Vec<String>) {
    for entry in std::fs::read_dir(root).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(std::fs::read_to_string(path).unwrap());
        }
    }
}

#[test]
fn cli_sources_have_no_stale_no_engine_warnings() {
    let mut sources = Vec::new();
    rust_sources(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src").as_path(),
        &mut sources,
    );

    for stale in [
        "v1 ships no OCR engine",
        "OCR-based box discarding is inactive",
        "the OCR report will be empty",
    ] {
        assert!(
            !sources.iter().any(|source| source.contains(stale)),
            "stale OCR diagnostic remains in a pc-cli source file: {stale}"
        );
    }
}
