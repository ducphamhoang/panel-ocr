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

fn reject_disallowed_control_characters(value: &str, subject: &str) -> Result<(), String> {
    if let Some((byte_offset, character)) = value.char_indices().find(|(_, character)| {
        (*character < '\u{20}' && *character != '\t') || *character == '\u{7f}'
    }) {
        return Err(format!(
            "{subject} contains disallowed control character U+{:04X} at byte offset {byte_offset}",
            character as u32
        ));
    }
    Ok(())
}

fn quotable_or_quoted(value: &str) -> Result<(), String> {
    reject_disallowed_control_characters(value, "frontmatter value")?;

    if value.starts_with('"') {
        if !value.ends_with('"') || value.len() < 2 {
            return Err("quoted value never closed".to_owned());
        }

        let interior = &value[1..value.len() - 1];
        // This is intentionally stricter than YAML: descriptions need no interior quotes, so a
        // rephrase is cheap, while accepting malformed quoting can silently disable a role
        // prohibition. That makes over-rejecting the sound direction for this file class.
        if interior.contains('"') {
            return Err("quoted value has an interior quote".to_owned());
        }
        if interior.contains('\\') {
            return Err("quoted value has an interior backslash".to_owned());
        }
        return Ok(());
    }

    if let Some(offending) = value.find(": ").map(|index| &value[index..index + 2]) {
        return Err(format!(
            "frontmatter value contains offending substring `{offending}`"
        ));
    }
    if let Some(offending) = value.chars().next().filter(|character| {
        matches!(
            character,
            '#' | '&' | '*' | '!' | '|' | '>' | '%' | '@' | '`' | '[' | '{' | ','
        )
    }) {
        return Err(format!(
            "frontmatter value starts with offending character `{offending}`"
        ));
    }
    Ok(())
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
            let Some((raw_key, raw_value)) = line.split_once(':') else {
                continue;
            };
            let key = raw_key.trim();
            if let Err(rule) = reject_disallowed_control_characters(key, "frontmatter key") {
                panic!("{} line {line_number}: {rule}", path.display());
            }
            let value = raw_value.trim();
            if let Err(rule) = quotable_or_quoted(value) {
                panic!("{} line {line_number}: {rule}", path.display());
            }
        }
    }
}

#[test]
fn quotable_or_quoted_accepts_safe_values_and_rejects_unsafe_values() {
    // The file-driven test can only fail by corrupting a real definition, so the pure rule needs direct coverage.
    let cases = [
        ("\"a normal quoted description.\"", true),
        ("plain-no-colon-value", true),
        ("'Read, Grep, Glob, Bash'", true),
        ("opus", true),
        ("\"quoted \u{01} value\"", false),
        ("\"quoted \u{1b} value\"", false),
        // This unquoted row proves the control-character check is branch-independent.
        ("plain\u{1b}value", false),
        ("\"quoted\tvalue\"", true),
        ("plain\tvalue", true),
        ("\"he said \"hi\" and left\"", false),
        ("\"abc\" trailing junk", false),
        ("\"unterminated", false),
        ("'Read-only: produces a design'", false),
        ("# comment-looking", false),
        ("\"has \\ backslash\"", false),
    ];

    for (value, expected_acceptance) in cases {
        assert_eq!(
            quotable_or_quoted(value).is_ok(),
            expected_acceptance,
            "unexpected validation result for {value:?}"
        );
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
