//! Gates the `.claude/agents/` frontmatter consumed by the agent harness.
//!
use pc_testkit::paths;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use yaml_rust2::{Yaml, YamlLoader};

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

fn frontmatter_block(path: &Path) -> String {
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

    lines[1..closing].join("\n")
}

// What failed before was a hand-rolled rule standing in for YAML, with an accept-set nobody
// could enumerate. What this is: a real parser doing the YAML judgement, plus a single-sentence
// constraint covering one axis where two real parsers measurably disagree and the spec sides with
// the stricter one. Its accept-set is one sentence long and testable. The harness's own loader is
// neither of these parsers, so where two real implementations disagree, the safe direction is the
// stricter one — rejecting costs an author nothing, since no legitimate agent definition contains
// an ESC byte.
//
// This gate asserts “parses under yaml-rust2 0.11.0, plus printable-characters”, which is a much
// closer proxy for the harness's loader than the old rule but is still a proxy — and now with
// measured evidence that two real parsers differ, so the residual risk is concrete rather than
// theoretical.
// The additive constraint is one sentence: reject any character below 0x20 except TAB, LF, and
// CR, and reject DEL (0x7F).
fn frontmatter_gate(document: &str) -> Result<Yaml, String> {
    if let Some((byte_offset, character)) = document.char_indices().find(|(_, character)| {
        (*character < '\u{20}' && !matches!(character, '\t' | '\n' | '\r'))
            || *character == '\u{7f}'
    }) {
        return Err(format!(
            "frontmatter contains non-printable character U+{:04X} at byte offset {byte_offset}",
            character as u32
        ));
    }

    let documents = YamlLoader::load_from_str(document).map_err(|error| error.to_string())?;
    documents
        .into_iter()
        .next()
        .ok_or_else(|| "frontmatter has an empty YAML document".to_owned())
}

fn parsed_frontmatter(path: &Path) -> Yaml {
    frontmatter_gate(&frontmatter_block(path))
        .unwrap_or_else(|error| panic!("{} has invalid YAML frontmatter: {error}", path.display()))
}

fn frontmatter_fields(path: &Path) -> BTreeMap<String, String> {
    let Yaml::Hash(fields) = parsed_frontmatter(path) else {
        panic!("{} frontmatter must be a YAML mapping", path.display());
    };
    fields
        .into_iter()
        .map(|(key, value)| {
            let Yaml::String(key) = key else {
                panic!("{} has a non-string frontmatter key", path.display());
            };
            let Yaml::String(value) = value else {
                panic!(
                    "{} has a non-string value for frontmatter key `{key}`",
                    path.display()
                );
            };
            (key, value)
        })
        .collect()
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
fn frontmatter_blocks_pass_the_acceptance_gate() {
    for path in agent_paths() {
        let _ = parsed_frontmatter(&path);
    }
}

#[test]
// Drives `frontmatter_gate`, NOT `yaml-rust2` alone — the distinction is load-bearing, so the name
// says "acceptance gate" rather than naming the parser. Two rows below (the quoted and unquoted
// ESC cases) are rejected by the printable-characters rule while yaml-rust2 accepts them, so a name
// crediting the parser would teach the next reader something measurably false.
fn the_acceptance_gate_accepts_and_rejects_the_recorded_frontmatter_cases() {
    let cases = [
        // The bare `: ` makes this plain scalar invalid YAML.
        (
            "name: sample\ndescription: Read-only: produces a design\ntools: Read, Grep, Glob, Bash\nmodel: opus\n",
            false,
        ),
        // An unescaped interior quote makes this double-quoted scalar invalid YAML.
        (
            "name: sample\ndescription: \"he said \"hi\" and left\"\ntools: Read, Grep, Glob, Bash\nmodel: opus\n",
            false,
        ),
        // Content after a closing quote is trailing junk, not part of the scalar.
        (
            "name: sample\ndescription: \"abc\" trailing junk\ntools: Read, Grep, Glob, Bash\nmodel: opus\n",
            false,
        ),
        // A double-quoted scalar that never closes must be rejected by the parser.
        (
            "name: sample\ndescription: \"unterminated\ntools: Read, Grep, Glob, Bash\nmodel: opus\n",
            false,
        ),
        // yaml-rust2 alone accepts this, but the printable-characters rule rejects ESC.
        (
            "name: sample\ndescription: \"quoted \u{1b} value\"\ntools: Read, Grep, Glob, Bash\nmodel: opus\n",
            false,
        ),
        // This unquoted ESC is rejected by the printable-characters rule, not the parser;
        // yaml-rust2 alone accepts it.
        (
            "name: sample\ndescription: unquoted \u{1b} value\ntools: Read, Grep, Glob, Bash\nmodel: opus\n",
            false,
        ),
        // A normal quoted scalar is valid YAML.
        (
            "name: sample\ndescription: \"a normal quoted description.\"\ntools: Read, Grep, Glob, Bash\nmodel: opus\n",
            true,
        ),
        // A plain scalar with no colon is valid YAML.
        (
            "name: sample\ndescription: plain-no-colon-value\ntools: Read, Grep, Glob, Bash\nmodel: opus\n",
            true,
        ),
        // The harness stores tools as a comma-separated string, not a YAML sequence.
        (
            "name: sample\ndescription: sample description\ntools: Read, Grep, Glob, Bash\nmodel: sonnet\n",
            true,
        ),
        // An unquoted model name is a valid plain scalar.
        (
            "name: sample\ndescription: sample description\ntools: Read\nmodel: opus\n",
            true,
        ),
        // TAB is legal inside a YAML quoted scalar and must not be rejected as C0 control data.
        (
            "name: sample\ndescription: \"quoted\tvalue\"\ntools: Read, Grep, Glob, Bash\nmodel: opus\n",
            true,
        ),
    ];

    for (document, expected_acceptance) in cases {
        let parsed = frontmatter_gate(document).is_ok();
        assert_eq!(
            parsed, expected_acceptance,
            "unexpected parser result for {document:?}"
        );
    }

    // An unquoted literal TAB is deliberately unpinned: PyYAML rejects it while yaml-rust2
    // accepts it, and which parser the harness follows in this case is undecided.
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
