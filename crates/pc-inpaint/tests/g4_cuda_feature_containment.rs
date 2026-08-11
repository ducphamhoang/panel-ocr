//! GPU-4 — manifest-coherence for the inpainting CUDA feature chain. `pc-inpaint/cuda`
//! must imply exactly `{onnx, pc-ort/cuda, pc-core/cuda}` (as a SET, so an accidental
//! fourth member fails too), `pc-inpaint/onnx` must contain `pc-ort/onnx`, and `pc-cli`'s
//! `cuda` feature must forward `pc-inpaint/cuda` **and** still forward `pc-detect/cuda`
//! and `pc-ocr/cuda` (the anti-regression halves — adding inpainting must not silently
//! drop the detector or the OCR backend). Same `toml_edit` walk pattern as `pc-detect`'s
//! `g2_cuda_feature_containment.rs` and `pc-ocr`'s `g3_cuda_feature_containment.rs`.
//!
//! The workspace-wide "no crate's `default` reaches `cuda`" sweep is deliberately NOT
//! duplicated here — `pc-detect`'s
//! `no_crate_reaches_the_cuda_feature_from_its_default_feature_set` already walks every
//! `crates/*/Cargo.toml` (including `pc-inpaint`) and `xtask/Cargo.toml` via its
//! `every_crate_manifest` helper, exactly as it does for `pc-ocr`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use toml_edit::{DocumentMut, Item, Value};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("pc-inpaint must be two levels below the workspace root")
        .to_path_buf()
}

fn read_manifest(path: &Path) -> DocumentMut {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .parse()
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn entry_features(item: &Item) -> Vec<String> {
    let values = item
        .as_value()
        .and_then(Value::as_array)
        .or_else(|| {
            item.as_table()
                .and_then(|table| table.get("features"))
                .and_then(Item::as_value)
                .and_then(Value::as_array)
        })
        .or_else(|| {
            item.as_value()
                .and_then(Value::as_inline_table)
                .and_then(|table| table.get("features"))
                .and_then(Value::as_array)
        });
    values
        .into_iter()
        .flat_map(|array| array.iter())
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

/// §16.47 item 5's forwardings, extended by GPU-4: `pc-inpaint/cuda` implies **exactly**
/// `onnx`, `pc-ort/cuda`, and `pc-core/cuda` (direct, not transitive — same shape as
/// `pc-detect/cuda` and `pc-ocr/cuda`); `pc-inpaint/onnx` must still carry the shared
/// `pc-ort/onnx` boundary; `pc-cli/cuda` implies `onnx`, `pc-detect/cuda`, `pc-ocr/cuda`,
/// **and** `pc-inpaint/cuda`. If either chain is broken, a `cuda`-feature build of the
/// inpainter links `ort` without the CUDA distribution or without the onnx backend, or
/// `pc-cli/cuda` silently stops carrying the detector or the OCR backend.
#[test]
fn the_cuda_features_pull_in_inpaint_ort_and_core_cuda_and_cli_still_forwards_the_rest() {
    let root = workspace_root();

    let inpaint = read_manifest(&root.join("crates/pc-inpaint/Cargo.toml"));
    let inpaint_features = inpaint
        .get("features")
        .and_then(Item::as_table)
        .expect("pc-inpaint must have a features table");
    let inpaint_cuda: BTreeSet<String> = inpaint_features
        .get("cuda")
        .expect("pc-inpaint must declare cuda")
        .get_implied_features()
        .into_iter()
        .collect();
    assert_eq!(
        inpaint_cuda,
        BTreeSet::from([
            "onnx".to_owned(),
            "pc-ort/cuda".to_owned(),
            "pc-core/cuda".to_owned()
        ]),
        "pc-inpaint/cuda must imply EXACTLY onnx, pc-ort/cuda, pc-core/cuda"
    );
    let inpaint_onnx: BTreeSet<String> = inpaint_features
        .get("onnx")
        .expect("pc-inpaint must declare onnx")
        .get_implied_features()
        .into_iter()
        .collect();
    assert!(
        inpaint_onnx.contains("pc-ort/onnx"),
        "pc-inpaint/onnx must contain pc-ort/onnx; got {inpaint_onnx:?}"
    );

    let cli = read_manifest(&root.join("crates/pc-cli/Cargo.toml"));
    let cli_features = cli
        .get("features")
        .and_then(Item::as_table)
        .expect("pc-cli must have a features table");
    let cli_cuda: BTreeSet<String> = cli_features
        .get("cuda")
        .expect("pc-cli must declare cuda")
        .get_implied_features()
        .into_iter()
        .collect();
    for implied in ["onnx", "pc-detect/cuda", "pc-ocr/cuda", "pc-inpaint/cuda"] {
        assert!(
            cli_cuda.contains(implied),
            "pc-cli/cuda must imply {implied}; got {cli_cuda:?}"
        );
    }
}

trait ImpliedFeatures {
    fn get_implied_features(&self) -> Vec<String>;
}

impl ImpliedFeatures for Item {
    fn get_implied_features(&self) -> Vec<String> {
        entry_features(self)
    }
}
