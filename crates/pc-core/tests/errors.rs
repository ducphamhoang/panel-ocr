//! C1 tests — spec §2.9 `StageError`.

use pc_core::StageError;
use std::path::PathBuf;

#[test]
// spec §2.9: the Display strings are user-facing (they land in the §5.5 summary
// table), so their exact form is part of the contract
fn error_display_strings() {
    let io = StageError::Io {
        path: PathBuf::from("/tmp/a.png"),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "nope"),
    };
    assert_eq!(io.to_string(), "io error at /tmp/a.png: nope");

    assert_eq!(
        StageError::UnsupportedFormat("jp2".into()).to_string(),
        "unsupported image format: jp2"
    );
    assert_eq!(
        StageError::Model("missing".into()).to_string(),
        "model error: missing"
    );
    assert_eq!(
        StageError::Inference("panicked: boom".into()).to_string(),
        "inference failed: panicked: boom"
    );
    assert_eq!(
        StageError::InvalidInput("bad".into()).to_string(),
        "invalid stage input: bad"
    );
    assert_eq!(
        StageError::UnmaterializedHandle.to_string(),
        "image handle is not materialized on disk"
    );
    assert_eq!(
        StageError::Empty("no boxes".into()).to_string(),
        "stage produced no usable output: no boxes"
    );
}

#[test]
// spec §2.9: Io and Decode expose their cause via `source()`, so a nested cause
// chain survives into the log line
fn io_error_exposes_source() {
    use std::error::Error as _;
    let io = StageError::Io {
        path: PathBuf::from("/tmp/a.png"),
        source: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
    };
    assert!(io.source().is_some());
}

#[test]
// spec §2.9: serde_json errors convert via #[from], so `?` works on JSON I/O
// inside a stage without hand-written mapping
fn serde_error_converts_via_from() {
    let json_err = serde_json::from_str::<pc_core::Rect>("{").unwrap_err();
    let converted: StageError = json_err.into();
    assert!(matches!(converted, StageError::Serde(_)));
    assert!(converted.to_string().starts_with("serialization error: "));
}
