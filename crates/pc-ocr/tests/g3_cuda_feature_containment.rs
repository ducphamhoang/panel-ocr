//! GPU-3 — manifest-coherence for the OCR CUDA feature chain. `pc-ocr/cuda` must imply
//! `onnx`, `pc-ort/cuda`, and `pc-core/cuda` directly (not transitively), and `pc-cli`'s
//! `cuda` feature must forward `pc-ocr/cuda` **and** still forward `pc-detect/cuda` (the
//! second half is the anti-regression check — adding OCR must not silently drop the
//! detector). Same `toml_edit` walk pattern as `pc-detect`'s
//! `g2_cuda_feature_containment.rs`, whose helper idiom is reused here rather than
//! duplicated blindly.
//!
//! The workspace-wide "no crate's `default` reaches `cuda`" sweep is deliberately NOT
//! duplicated here — `pc-detect`'s `no_crate_reaches_the_cuda_feature_from_its_default_feature_set`
//! already walks every `crates/*/Cargo.toml` (including `pc-ocr`) and `xtask/Cargo.toml`
//! via its `every_crate_manifest` helper.

use std::path::{Path, PathBuf};
use toml_edit::{DocumentMut, Item, Value};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("pc-ocr must be two levels below the workspace root")
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

/// §16.47 item 5's forwardings, extended by GPU-3: `pc-ocr/cuda` implies `onnx`,
/// `pc-ort/cuda`, and `pc-core/cuda` (direct, not transitive — same shape as
/// `pc-detect/cuda`); `pc-cli/cuda` implies `onnx`, `pc-detect/cuda`, **and**
/// `pc-ocr/cuda`. If either chain is broken, a `cuda`-feature build of OCR links `ort`
/// without the CUDA distribution or without the onnx backend, or `pc-cli/cuda`
/// silently stops carrying the detector.
#[test]
fn the_cuda_features_pull_in_ocr_ort_and_core_cuda_and_cli_still_forwards_the_detector() {
    let root = workspace_root();

    let ocr = read_manifest(&root.join("crates/pc-ocr/Cargo.toml"));
    let ocr_features = ocr
        .get("features")
        .and_then(Item::as_table)
        .expect("pc-ocr must have a features table");
    let ocr_cuda = ocr_features
        .get("cuda")
        .expect("pc-ocr must declare cuda")
        .get_implied_features();
    for implied in ["onnx", "pc-ort/cuda", "pc-core/cuda"] {
        assert!(
            ocr_cuda.contains(&implied.to_owned()),
            "pc-ocr/cuda must imply {implied}; got {ocr_cuda:?}"
        );
    }

    let cli = read_manifest(&root.join("crates/pc-cli/Cargo.toml"));
    let cli_features = cli
        .get("features")
        .and_then(Item::as_table)
        .expect("pc-cli must have a features table");
    let cli_cuda = cli_features
        .get("cuda")
        .expect("pc-cli must declare cuda")
        .get_implied_features();
    for implied in ["onnx", "pc-detect/cuda", "pc-ocr/cuda"] {
        assert!(
            cli_cuda.contains(&implied.to_owned()),
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
