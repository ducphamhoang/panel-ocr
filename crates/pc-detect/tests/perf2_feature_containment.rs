//! Production-manifest feature containment for the detector's maintainer-only tuning knob.

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

fn dependency_item<'a>(manifest: &'a DocumentMut, name: &str) -> Option<&'a Item> {
    manifest
        .get("dependencies")
        .and_then(Item::as_table)
        .and_then(|table| table.get(name))
        .or_else(|| {
            manifest
                .get("build-dependencies")
                .and_then(Item::as_table)
                .and_then(|table| table.get(name))
        })
}

fn table_field<'a>(item: &'a Item, field: &str) -> Option<&'a Item> {
    item.as_table().and_then(|table| table.get(field))
}

fn inline_field<'a>(item: &'a Item, field: &str) -> Option<&'a Value> {
    item.as_value()
        .and_then(Value::as_inline_table)
        .and_then(|table| table.get(field))
}

fn string_field(item: &Item, field: &str) -> Option<String> {
    table_field(item, field)
        .and_then(Item::as_value)
        .and_then(Value::as_str)
        .or_else(|| inline_field(item, field).and_then(Value::as_str))
        .map(str::to_owned)
}

fn feature_names(item: &Item) -> Vec<String> {
    let values = item.as_value().and_then(Value::as_array).or_else(|| {
        table_field(item, "features")
            .and_then(Item::as_value)
            .and_then(Value::as_array)
            .or_else(|| inline_field(item, "features").and_then(Value::as_array))
    });
    values
        .into_iter()
        .flat_map(|array| array.iter())
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn normal_and_build_dependencies(manifest: &DocumentMut) -> Vec<(&str, &Item)> {
    ["dependencies", "build-dependencies"]
        .into_iter()
        .flat_map(|section| {
            manifest
                .get(section)
                .and_then(Item::as_table)
                .into_iter()
                .flat_map(|table| table.iter())
        })
        .collect()
}

fn forwards_bench_tuning(manifest: &DocumentMut) -> bool {
    manifest
        .get("features")
        .and_then(Item::as_table)
        .into_iter()
        .flat_map(|table| table.iter())
        .any(|(_, item)| {
            item.as_value()
                .and_then(Value::as_array)
                .into_iter()
                .flat_map(|array| array.iter())
                .filter_map(Value::as_str)
                .any(|feature| feature == "pc-detect/bench-tuning")
        })
}

fn default_reaches_bench_tuning(manifest: &DocumentMut) -> bool {
    let features = manifest
        .get("features")
        .and_then(Item::as_table)
        .expect("pc-detect must have a features table");
    let default = features
        .get("default")
        .map(feature_names)
        .unwrap_or_default();
    let mut queue = VecDeque::from(default);
    let mut visited = BTreeSet::new();

    while let Some(feature) = queue.pop_front() {
        if !visited.insert(feature.clone()) {
            continue;
        }
        if feature == "bench-tuning" {
            return true;
        }
        if let Some(item) = features.get(&feature) {
            for implied in feature_names(item) {
                if features.contains_key(&implied) {
                    queue.push_back(implied);
                }
            }
        }
    }

    false
}

fn reachable_manifests(root: &Path) -> (Vec<PathBuf>, Vec<String>) {
    let start = root.join("crates/pc-cli/Cargo.toml");
    let mut queue = VecDeque::from([start]);
    let mut visited = BTreeSet::new();
    let mut violations = Vec::new();

    while let Some(manifest_path) = queue.pop_front() {
        let manifest_path = manifest_path
            .canonicalize()
            .unwrap_or_else(|error| panic!("canonicalize {}: {error}", manifest_path.display()));
        if !visited.insert(manifest_path.clone()) {
            continue;
        }
        let manifest = read_manifest(&manifest_path);
        if forwards_bench_tuning(&manifest) {
            violations.push(format!(
                "{} forwards pc-detect/bench-tuning from its feature table",
                manifest_path.display()
            ));
        }
        for (name, item) in normal_and_build_dependencies(&manifest) {
            let package = string_field(item, "package").unwrap_or_else(|| name.to_owned());
            if package == "pc-detect" && feature_names(item).iter().any(|f| f == "bench-tuning") {
                violations.push(format!(
                    "{} declares pc-detect/bench-tuning in a normal/build dependency",
                    manifest_path.display()
                ));
            }
            if let Some(path) = string_field(item, "path") {
                let dependency_manifest = manifest_path
                    .parent()
                    .expect("manifest has a parent")
                    .join(path)
                    .join("Cargo.toml");
                if dependency_manifest.is_file() {
                    queue.push_back(dependency_manifest);
                }
            }
        }
    }

    (visited.into_iter().collect(), violations)
}

/// Covers only the production/release manifest graph's normal + build dependency edges. It
/// deliberately does not and cannot cover `cargo test --workspace`'s own build: pc-detect's
/// self dev-dependency enables `testkit` + `bench-tuning` for these perf2_* tests, and resolver-v2
/// feature unification legitimately carries that feature into the workspace test build.
#[test]
fn production_release_manifest_graph_normal_and_build_edges_exclude_bench_tuning() {
    let root = workspace_root();
    let cli = read_manifest(&root.join("crates/pc-cli/Cargo.toml"));
    let detect = dependency_item(&cli, "pc-detect").expect("pc-cli must depend on pc-detect");
    assert_eq!(
        feature_names(detect),
        vec!["testkit"],
        "pc-cli's direct pc-detect dependency features must be exactly [\"testkit\"]"
    );

    let detect_manifest = read_manifest(&root.join("crates/pc-detect/Cargo.toml"));
    assert!(
        !default_reaches_bench_tuning(&detect_manifest),
        "pc-detect's default feature set reaches bench-tuning"
    );

    let (reachable, violations) = reachable_manifests(&root);
    assert!(
        reachable
            .iter()
            .any(|path| path.ends_with("pc-cli/Cargo.toml")),
        "the containment walk must start at pc-cli"
    );
    assert!(
        violations.is_empty(),
        "production/release manifest graph enables bench-tuning: {violations:?}"
    );
}

/// Anti-vacuity control for the production/release containment gate: pc-detect must still
/// declare `bench-tuning`, and xtask must still enable `pc-detect/bench-tuning` for maintainer
/// tooling. This does not inspect dev-dependencies in the production graph.
#[test]
fn bench_tuning_still_exists_and_is_enabled_by_xtask() {
    let root = workspace_root();
    let detect = read_manifest(&root.join("crates/pc-detect/Cargo.toml"));
    let features = detect
        .get("features")
        .and_then(Item::as_table)
        .expect("pc-detect must have a features table");
    assert!(
        features.contains_key("bench-tuning"),
        "pc-detect's bench-tuning feature disappeared"
    );

    let xtask = read_manifest(&root.join("xtask/Cargo.toml"));
    let xtask_features = xtask
        .get("features")
        .and_then(Item::as_table)
        .expect("xtask must have a features table");
    let enables_bench_tuning = xtask_features.iter().any(|(_, item)| {
        item.as_value()
            .and_then(Value::as_array)
            .into_iter()
            .flat_map(|array| array.iter())
            .filter_map(Value::as_str)
            .any(|feature| feature == "pc-detect/bench-tuning")
    });
    assert!(
        enables_bench_tuning,
        "xtask no longer enables pc-detect/bench-tuning"
    );
}
