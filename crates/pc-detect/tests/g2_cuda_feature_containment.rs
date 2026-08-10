//! GPU-2 (§16.47 item 7, row 2) — the workspace's manifest-coherence invariant: no
//! crate's `default` feature set may reach the `cuda` feature (it is opt-in, §16.22 item
//! 5(a)), and `pc-detect`'s / `pc-cli`'s `cuda` feature must pull in exactly the
//! `onnx`/`ort` linkage the detector needs. Same `toml_edit` walk pattern as the sibling
//! `perf2_feature_containment.rs`, whose helper idiom is reused here rather than
//! duplicated blindly.

use std::collections::{BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use toml_edit::{DocumentMut, Item, Value};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("pc-detect must be two levels below the workspace root")
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

fn every_crate_manifest(root: &Path) -> Vec<PathBuf> {
    let mut manifests = Vec::new();
    for entry in std::fs::read_dir(root.join("crates"))
        .unwrap_or_else(|error| panic!("read crates dir: {error}"))
    {
        let entry = entry.expect("read crates entry");
        let path = entry.path().join("Cargo.toml");
        if path.is_file() {
            manifests.push(path);
        }
    }
    // `xtask` is a workspace member but lives outside `crates/` — the sibling
    // `perf2_feature_containment.rs` already reads it explicitly for the same reason.
    // Its own `cuda` feature (§16.47 items 6, 9) is exactly as load-bearing as the three
    // crates' below, so the containment sweep must cover it too.
    manifests.push(root.join("xtask/Cargo.toml"));
    manifests.sort();
    manifests
}

/// The transitive closure of a feature entry's implied features. Cross-crate
/// (`name/feature`) edges are recorded in the returned set as terminal leaves but never
/// recursed into — each crate's manifest is walked independently, and the containment
/// assertion needs "everything one `cargo build` of this crate would enable", which
/// includes a cross-crate name even though this walk can't see inside it.
///
/// **Fixed 2026-08-11 (independent review, GPU-2 G2-A/B): the previous version only
/// recorded an implied name when it happened to also be a local key** (`if
/// features.contains_key(&implied) { queue.push_back(implied) }`, with nothing recording
/// the name otherwise) — so `default = ["pc-detect/cuda"]` was invisible to this walk,
/// the exact §16.22 item 5(a) violation this test exists to catch. `reached` now records
/// every implied name unconditionally; only recursion (`queue`) is gated on being a
/// local key, via a separate `expanded` set so re-visiting a local feature doesn't skip
/// its own expansion.
fn transitive_closure(features: &toml_edit::Table, seed: &str) -> BTreeSet<String> {
    let mut queue = VecDeque::from([seed.to_owned()]);
    let mut expanded = BTreeSet::new();
    let mut reached = BTreeSet::new();
    reached.insert(seed.to_owned());
    while let Some(feature) = queue.pop_front() {
        if !expanded.insert(feature.clone()) {
            continue;
        }
        if let Some(item) = features.get(&feature) {
            for implied in entry_features(item) {
                reached.insert(implied.clone());
                if features.contains_key(&implied) {
                    queue.push_back(implied);
                }
            }
        }
    }
    reached
}

fn feature_is_cuda_like(feature: &str) -> bool {
    feature == "cuda" || feature.ends_with("/cuda")
}

/// §16.22 item 5(a) "Opt-in only … Never auto-detected": CUDA must never be reachable
/// from any crate's `default` feature set, or the opt-in tier silently becomes default.
///
/// Anti-vacuity: the assertion that `pc-core`/`pc-detect`/`pc-cli` each declare a `cuda`
/// feature is hard-coded here (not derived from the walk), so the test cannot pass by
/// finding zero `cuda` features anywhere.
#[test]
fn no_crate_reaches_the_cuda_feature_from_its_default_feature_set() {
    let root = workspace_root();

    for manifest_path in every_crate_manifest(&root) {
        let manifest = read_manifest(&manifest_path);
        // A crate without a `[features]` table has no default set at all, so it
        // trivially reaches nothing — do not panic on it (most stage crates declare
        // no features of their own).
        let Some(features) = manifest.get("features").and_then(Item::as_table) else {
            continue;
        };
        let reached = transitive_closure(features, "default");
        let cuda_reached = reached.iter().any(|feature| feature_is_cuda_like(feature));
        assert!(
            !cuda_reached,
            "{}: the `default` feature set reaches `cuda` (reached {reached:?})",
            manifest_path.display()
        );
    }

    // Anti-vacuity: these crates must declare `cuda` at all, so the walk above had
    // something to look for in each of the crates/binary that own a CUDA-capable stage
    // or gate a cuda-tier test (xtask, per §16.47 items 6/9).
    for (name, path) in [
        ("pc-core", "crates/pc-core/Cargo.toml"),
        ("pc-detect", "crates/pc-detect/Cargo.toml"),
        ("pc-cli", "crates/pc-cli/Cargo.toml"),
        ("xtask", "xtask/Cargo.toml"),
    ] {
        let manifest = read_manifest(&root.join(path));
        let features = manifest
            .get("features")
            .and_then(Item::as_table)
            .expect("{name} must have a features table");
        assert!(
            features.contains_key("cuda"),
            "{name} does not declare a `cuda` feature"
        );
    }
}

/// §16.47 item 5's forwardings, pinned: `pc-detect/cuda` implies `onnx`, `ort/cuda`, and
/// `pc-core/cuda`; `pc-cli/cuda` implies `onnx` and `pc-detect/cuda`. If either chain is
/// broken, a `cuda`-feature build of the detector links `ort` without the CUDA
/// distribution or without the onnx backend.
#[test]
fn the_cuda_feature_pulls_in_both_the_onnx_backend_and_orts_cuda_linkage() {
    let root = workspace_root();

    let detect = read_manifest(&root.join("crates/pc-detect/Cargo.toml"));
    let detect_features = detect
        .get("features")
        .and_then(Item::as_table)
        .expect("pc-detect must have a features table");
    let detect_cuda = detect_features
        .get("cuda")
        .expect("pc-detect must declare cuda")
        .get_implied_features();
    let detect_reached = detect_features
        .get("cuda")
        .map(|item| {
            let mut closure = transitive_closure(detect_features, "cuda");
            closure.extend(entry_features(item));
            closure
        })
        .expect("pc-detect must declare cuda");
    for implied in ["onnx", "ort/cuda", "pc-core/cuda"] {
        assert!(
            detect_cuda.contains(&implied.to_owned()),
            "pc-detect/cuda must imply {implied}; got {detect_cuda:?}"
        );
        assert!(
            detect_reached.contains(implied),
            "pc-detect/cuda must reach {implied}; got {detect_reached:?}"
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
    let cli_reached = cli_features
        .get("cuda")
        .map(|item| {
            let mut closure = transitive_closure(cli_features, "cuda");
            closure.extend(entry_features(item));
            closure
        })
        .expect("pc-cli must declare cuda");
    for implied in ["onnx", "pc-detect/cuda"] {
        assert!(
            cli_cuda.contains(&implied.to_owned()),
            "pc-cli/cuda must imply {implied}; got {cli_cuda:?}"
        );
        assert!(
            cli_reached.contains(implied),
            "pc-cli/cuda must reach {implied}; got {cli_reached:?}"
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
