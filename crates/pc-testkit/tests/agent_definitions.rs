//! Gates the `.claude/agents/` frontmatter consumed by the agent harness.
//!
//! These checks intentionally use plain string operations instead of a YAML parser.  The
//! quotable-value rule is stricter than YAML: a false rejection is repaired by quoting a value,
//! while a false acceptance recreates the historical silent loss of a role prohibition.

use pc_testkit::paths;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const EXPECTED_AGENT_NAMES: &[&str] = &[
    "architect",
    "fable-adjudicator",
    "fresh-reader",
    "rust-engineer",
];
const READ_ONLY_AGENTS: &[&str] = &["fresh-reader", "architect", "fable-adjudicator"];
const REQUIRED_KEYS: &[&str] = &["name", "description", "tools", "model"];
const ALLOWED_MODELS: &[&str] = &["opus", "sonnet", "haiku", "fable"];
const MUTATING_TOOLS: &[&str] = &["Edit", "Write", "NotebookEdit"];

fn agent_paths() -> Vec<PathBuf> {
    let directory = paths::workspace_root().join(".claude/agents");
    let mut files = fs::read_dir(&directory)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to read an entry in {}: {error}",
                        directory.display()
                    )
                })
                .path()
        })
        .filter(|path| {
            path.is_file() && path.extension().is_some_and(|extension| extension == "md")
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn file_text(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn frontmatter_lines(path: &Path) -> Vec<(usize, String)> {
    let text = file_text(path);
    assert!(
        text.starts_with("---\n"),
        "{} must start with `---\\n`",
        path.display()
    );

    let lines = text.lines().collect::<Vec<_>>();
    let closing = lines
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, line)| **line == "---")
        .map(|(index, _)| index)
        .unwrap_or_else(|| panic!("{} has no closing `---` line", path.display()));

    lines[1..closing]
        .iter()
        .enumerate()
        .map(|(index, line)| (index + 2, (*line).to_owned()))
        .collect()
}

fn frontmatter_fields(path: &Path) -> BTreeMap<String, String> {
    frontmatter_lines(path)
        .into_iter()
        .filter_map(|(line_number, line)| {
            line.split_once(':').map(|(key, value)| {
                let key = key.trim().to_owned();
                let value = unquote(value.trim()).to_owned();
                assert!(
                    !key.is_empty(),
                    "{} line {line_number} has an empty frontmatter key",
                    path.display()
                );
                (key, value)
            })
        })
        .collect()
}

fn unquote(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(value)
}

fn field<'a>(fields: &'a BTreeMap<String, String>, path: &Path, key: &str) -> &'a str {
    fields
        .get(key)
        .map(String::as_str)
        .unwrap_or_else(|| panic!("{} is missing required key `{key}`", path.display()))
}

#[test]
fn agent_directory_contains_exact_expected_set() {
    let paths = agent_paths();
    assert!(
        !paths.is_empty(),
        "`.claude/agents/` must contain agent definitions"
    );

    let found = paths
        .iter()
        .map(|path| {
            path.file_stem()
                .unwrap_or_else(|| panic!("agent path has no filename: {}", path.display()))
                .to_string_lossy()
                .into_owned()
        })
        .collect::<BTreeSet<_>>();
    let expected = EXPECTED_AGENT_NAMES
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>();

    assert_eq!(
        found, expected,
        "unexpected `.claude/agents/` definition set"
    );
}

#[test]
fn agent_frontmatter_has_opening_and_closing_delimiters() {
    for path in agent_paths() {
        let _ = frontmatter_lines(&path);
    }
}

#[test]
fn frontmatter_values_are_safely_quotable_or_quoted() {
    for path in agent_paths() {
        for (line_number, line) in frontmatter_lines(&path) {
            let Some((_, raw_value)) = line.split_once(':') else {
                continue;
            };
            let value = raw_value.trim();
            let wrapped_in_double_quotes =
                value.starts_with('"') && value.ends_with('"') && value.len() >= 2;
            if wrapped_in_double_quotes {
                continue;
            }

            if let Some(offending) = value.find(": ").map(|index| &value[index..index + 2]) {
                panic!(
                    "{} line {line_number}: frontmatter value contains offending substring `{offending}`",
                    path.display()
                );
            }
            if let Some(offending) = value.chars().next().filter(|character| {
                matches!(
                    character,
                    '#' | '&' | '*' | '!' | '|' | '>' | '%' | '@' | '`' | '[' | '{' | ','
                )
            }) {
                panic!(
                    "{} line {line_number}: frontmatter value starts with offending character `{offending}`",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn every_agent_has_required_frontmatter_keys() {
    for path in agent_paths() {
        let fields = frontmatter_fields(&path);
        for key in REQUIRED_KEYS {
            assert!(
                fields.contains_key(*key),
                "{} is missing required key `{key}`",
                path.display()
            );
        }
    }
}

#[test]
fn agent_names_match_filename_stems() {
    for path in agent_paths() {
        let fields = frontmatter_fields(&path);
        let stem = path
            .file_stem()
            .unwrap_or_else(|| panic!("agent path has no filename: {}", path.display()))
            .to_string_lossy();
        assert_eq!(
            field(&fields, &path, "name"),
            stem,
            "name mismatch in {path:?}"
        );
    }
}

#[test]
fn pinned_read_only_agents_have_no_mutating_tools() {
    for agent in READ_ONLY_AGENTS {
        let path = paths::workspace_root()
            .join(".claude/agents")
            .join(format!("{agent}.md"));
        let fields = frontmatter_fields(&path);
        let tools = field(&fields, &path, "tools")
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>();
        // Compare complete comma-separated tokens so a similarly named tool cannot satisfy or
        // evade the prohibition through a substring match on the whole `tools` value.
        assert!(
            MUTATING_TOOLS
                .iter()
                .all(|mutating| !tools.contains(mutating)),
            "{agent} must not have Edit, Write, or NotebookEdit tools; parsed tools: {tools:?}"
        );
    }
}

#[test]
fn rust_engineer_has_edit_and_write_tools() {
    // Positive control: without this assertion, the read-only rule could pass by finding no
    // relevant tools anywhere, even if rust-engineer had accidentally lost its write access.
    let path = paths::workspace_root().join(".claude/agents/rust-engineer.md");
    let fields = frontmatter_fields(&path);
    let tools = field(&fields, &path, "tools")
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    assert!(
        tools.contains(&"Edit"),
        "rust-engineer must have the Edit tool: {tools:?}"
    );
    assert!(
        tools.contains(&"Write"),
        "rust-engineer must have the Write tool: {tools:?}"
    );
}

#[test]
fn agent_models_are_allowed_and_pinned_by_role() {
    for path in agent_paths() {
        let fields = frontmatter_fields(&path);
        let name = field(&fields, &path, "name");
        let model = field(&fields, &path, "model");
        assert!(
            ALLOWED_MODELS.contains(&model),
            "{path:?} has unsupported model `{model}`; allowed models: {ALLOWED_MODELS:?}"
        );
        let expected = if name == "fable-adjudicator" {
            "fable"
        } else {
            "opus"
        };
        assert_eq!(model, expected, "{name} has the wrong pinned model");
    }
}
