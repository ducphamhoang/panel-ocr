//! spec §16.33 items 3, 5 and 9 — the ratified path literals and the ratified scope statements must
//! be readable, verbatim, at the two documents that claim to record them.
//!
//! **Why a documentation gate is worth a test file here.** §16.33 item 2 measured what happens
//! without one: the supported-platform claim drifted across eight occurrences in five files, and the
//! document two of those sites cited as the authority (`docs/ARCHITECTURE_DECISIONS.md`) had no such
//! section at all. Nobody noticed, because nothing could. Path defaults are the same class of claim
//! — spread across a resolver, a spec clause and a decisions register, with no mechanism keeping the
//! three in agreement.
//!
//! **What this gate checks, and what it deliberately does not.** It checks that each ratified
//! resolution-table **row** — the whole row, mapping included, not the bare directory literal —
//! appears, exactly once and in the ratified order, inside the table span belonging to the lead-in
//! label that names its root. Comparison is on whitespace-normalized text, so a Markdown reformat
//! that pads a cell or realigns a column is not a false failure. It does NOT check that the resolver
//! produces those paths — that is `crates/pc-cli/tests/x1_platform_paths.rs`,
//! which asserts behaviour against the same literals typed out independently there. The two files
//! together are what pin code and documentation to each other; neither alone does.
//!
//! Every literal below is typed out here. None is read from either document and then compared with
//! itself, which is cookbook rules 7 and 13's vacuous gate.

use pc_testkit::paths;

/// The heading opening the ratification's own section, and the anchor for the span scoping below.
const RATIFICATION_HEADING: &str = "## 16.33 ";

/// One of §16.33 item 5's two resolution tables: the lead-in sentence that names **which root** the
/// table resolves, plus its three platform rows in the ratified order.
///
/// **Why the lead-in is part of the pinned artifact.** A second mutation probe on 2026-08-04 swapped
/// only the two lead-in sentences in the spec, leaving all six row lines untouched. That inverts
/// cache↔config on all three platforms — including reverting E1 towards upstream's measured
/// `%APPDATA%`-for-both behaviour — and the predecessor version of this file, which searched for six
/// free-floating row strings anywhere in §16.33, still passed 6/6. A row string on its own does not
/// know which table it belongs to; only the label above it says.
struct ResolutionTable {
    /// The lead-in sentence, matched exactly once inside §16.33 and used as the span's start anchor.
    lead_in: &'static str,
    /// A short name for failure messages.
    root: &'static str,
    /// The three platform rows, in the order the spec lists them. Order is asserted, not just
    /// membership: item 5's tables are read top-to-bottom as "in order, first match wins" per row,
    /// and a reordered table is a reviewable change to a ratified artifact either way.
    rows: &'static [&'static str],
}

/// Every row of §16.33 item 5's two resolution tables, typed out here as the **complete row text**
/// rather than as the bare directory literals it contains, and grouped under the lead-in label that
/// binds it to a root.
///
/// **Why the whole row and not the literal — this was measured, not reasoned.** The first version of
/// this file gated the ten bare literals with `section.contains(literal)`. A mutation probe on
/// 2026-08-04 swapped the two Windows cells (cache↔config, i.e. reverting E1 towards upstream's
/// measured `%APPDATA%`-for-both behaviour) and the file still passed 5/5: the swapped-out literal
/// `%LOCALAPPDATA%\panel-ocr` still occurs elsewhere in §16.33's prose (item 13's `ends_with`
/// cross-check), so the bare-substring search was satisfied by an unrelated occurrence. A bare
/// literal proves the *word* is somewhere in the section; only the row proves the *mapping*, and only
/// the row-inside-its-labeled-span proves *which* mapping.
/// Cardinality-is-not-identity (cookbook rule's set-vs-count point) applied to prose.
///
/// Leading indentation and interior cell padding are deliberately excluded — the rows sit inside a
/// numbered item today and their indent and column alignment are Markdown details, not ratified
/// facts. Comparison runs over `collapse_whitespace`d text for that reason.
const RATIFIED_RESOLUTION_TABLES: &[ResolutionTable] = &[
    ResolutionTable {
        lead_in: "Cache root, in order, first match wins:",
        root: "cache root",
        rows: &[
            // Linux and macOS are today's behaviour, unchanged by v1.1.
            "| Linux | `$XDG_CACHE_HOME/panel-ocr` → `$HOME/.cache/panel-ocr` → `./.panel-ocr-cache` |",
            "| macOS | `$XDG_CACHE_HOME/panel-ocr` → `$HOME/Library/Caches/panel-ocr` → `./.panel-ocr-cache` |",
            // The E1 split, and the DEVIATION(19) from upstream's `%APPDATA%` for both roots. This
            // is the row both mutation probes above targeted.
            r"| Windows | `$XDG_CACHE_HOME/panel-ocr` → `%LOCALAPPDATA%\panel-ocr` → `./.panel-ocr-cache` |",
        ],
    },
    ResolutionTable {
        lead_in: "Config root, in order, first match wins:",
        root: "config root",
        rows: &[
            "| Linux | `$XDG_CONFIG_HOME/panel-ocr` → `$HOME/.config/panel-ocr` → `./.panel-ocr` |",
            "| macOS | `$XDG_CONFIG_HOME/panel-ocr` → `$HOME/Library/Application Support/panel-ocr` → `./.panel-ocr` |",
            r"| Windows | `$XDG_CONFIG_HOME/panel-ocr` → `%APPDATA%\panel-ocr` → `./.panel-ocr` |",
        ],
    },
];

/// Two tables, three rows each: 3 platforms x 2 roots.
const EXPECTED_RESOLUTION_TABLE_COUNT: usize = 2;
const EXPECTED_ROWS_PER_TABLE: usize = 3;
const EXPECTED_RESOLUTION_ROW_COUNT: usize = 6;

/// Every ratified row, flattened across both tables, for the cross-checks that do not care which
/// table a row came from.
fn all_ratified_rows() -> Vec<&'static str> {
    RATIFIED_RESOLUTION_TABLES
        .iter()
        .flat_map(|table| table.rows.iter().copied())
        .collect()
}

/// Every default-root literal §16.33 item 5's two tables prescribe, across all three platforms and
/// both roots, plus the two last-resort relative fallbacks. Ten: 6 platform-specific roots, 2 XDG
/// roots shared by all platforms, 2 fallbacks.
///
/// These are NOT searched for in the spec — that is what the row list above does, and doing it with
/// bare literals is the defect the row list exists to fix. They are cross-checked against the row
/// list instead, so a typo in either constant is caught.
const RATIFIED_PATH_LITERALS: &[&str] = &[
    // XDG, honoured first on every platform (item 4, E3).
    "$XDG_CACHE_HOME/panel-ocr",
    "$XDG_CONFIG_HOME/panel-ocr",
    // Linux (unchanged by v1.1).
    "$HOME/.cache/panel-ocr",
    "$HOME/.config/panel-ocr",
    // macOS (unchanged by v1.1).
    "$HOME/Library/Caches/panel-ocr",
    "$HOME/Library/Application Support/panel-ocr",
    // Windows — the E1 split, and the DEVIATION(19) from upstream's `%APPDATA%` for both.
    r"%LOCALAPPDATA%\panel-ocr",
    r"%APPDATA%\panel-ocr",
    // Last resort, all platforms.
    "./.panel-ocr-cache",
    "./.panel-ocr",
];

/// §16.24 item 1(f)'s literal-constant pattern: hard-coded, so this file cannot pass by checking an
/// empty list of literals.
const EXPECTED_PATH_LITERAL_COUNT: usize = 10;

/// Ratified scope statements that must be readable verbatim in BOTH documents, because each is the
/// kind of narrow claim that gets widened when paraphrased. Quoting rather than paraphrasing is this
/// project's transcription rule, and these are the phrases whose widening would matter most.
const RATIFIED_SCOPE_PHRASES: &[(&str, &str)] = &[
    (
        "maintainer-local, Linux and macOS only",
        "§16.33 item 9 (E4) — xtask's Python-subprocess paths. Widening this to \
         'xtask is untested on Windows' would lose that its NON-Python tests ARE required there.",
    ),
    (
        "Windows GPU CI runner",
        "§16.33 item 11 — the DirectML re-grounding rests partly on there being no runner to \
         verify it on. Dropping the phrase leaves the re-grounding resting on item 2's carve-out \
         alone, which is a narrower argument than the one ratified.",
    ),
    (
        "core.autocrlf=true",
        "§16.33 item 8 — the concrete mechanism. Without it the .gitattributes rows read as \
         housekeeping and get 'tidied'.",
    ),
];

const EXPECTED_SCOPE_PHRASE_COUNT: usize = 3;

fn read(relative: &str) -> String {
    let path = paths::workspace_root().join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read `{}`: {error}", path.display()))
}

/// The text of §16.33 alone. Scoping the search to the ratification's own span is the point: a
/// literal that happens to appear in some unrelated section would otherwise satisfy the gate, and
/// moving §16.33 would silently stop protecting it (cookbook rule 12 — the gate must point at the
/// artifact carrying the risk).
fn ratification_section() -> String {
    let spec = read("docs/PIPELINE_SPEC_V1.md");
    let start = spec
        .find(RATIFICATION_HEADING)
        .expect("docs/PIPELINE_SPEC_V1.md must contain a `## 16.33 ` section");
    let after = &spec[start + RATIFICATION_HEADING.len()..];
    let end = after
        .match_indices("\n## ")
        .next()
        .map_or(spec.len(), |(offset, _)| {
            start + RATIFICATION_HEADING.len() + offset
        });
    spec[start..end].to_owned()
}

/// Collapse every run of whitespace to a single space, so cell padding and column alignment — both
/// Markdown formatting, neither a ratified fact — cannot turn this gate red. Mirrors
/// `crates/pc-testkit/tests/platform_claim_sites.rs`'s `flatten`.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// True for a Markdown table alignment separator (`|---|---|`, `| :--- | ---: |`, ...), which is
/// formatting and is therefore skipped rather than pinned.
fn is_separator_row(line: &str) -> bool {
    line.starts_with('|')
        && line
            .chars()
            .all(|c| matches!(c, '|' | '-' | ':' | ' ' | '\t'))
        && line.contains('-')
}

/// The data rows of the Markdown table that immediately follows `lead_in` inside §16.33, normalized,
/// with the header and alignment-separator rows dropped.
///
/// The span is anchored at the lead-in label and bounded by the end of the table: only blank lines may
/// separate the label from its table, so a deleted table cannot be silently satisfied by some later
/// table further down the section.
fn table_rows_under(section: &str, lead_in: &str) -> Vec<String> {
    let occurrences = section.matches(lead_in).count();
    assert_eq!(
        occurrences,
        1,
        "§16.33 must contain the lead-in label {lead_in:?} exactly once — found {occurrences}. The \
         label is what binds the table below it to a root; without it the rows are unattributed."
    );
    let after = &section[section.find(lead_in).expect("checked above") + lead_in.len()..];

    let mut lines = after.lines().map(str::trim).peekable();
    // The remainder of the label's own line, then any blank lines, and nothing else.
    while lines.peek().is_some_and(|line| line.is_empty()) {
        lines.next();
    }

    let table: Vec<&str> = lines
        .take_while(|line| line.starts_with('|'))
        .collect::<Vec<_>>();
    assert!(
        table.len() >= 2,
        "the first non-blank content after {lead_in:?} in §16.33 is not a Markdown table (found \
         {} table line(s)). The label and its table must stay adjacent, or this gate stops \
         attributing rows to roots.",
        table.len()
    );

    let header = collapse_whitespace(table[0]).to_lowercase();
    assert!(
        header.contains("platform") && header.contains("order"),
        "the table under {lead_in:?} must keep a header naming `platform` and `order` — `order` is \
         load-bearing, it is what makes the row a first-match-wins sequence. Found: {header:?}"
    );

    table[1..]
        .iter()
        .filter(|line| !is_separator_row(line))
        .map(|line| collapse_whitespace(line))
        .collect()
}

#[test]
// Anti-vacuity for both literal lists.
fn the_ratified_literal_counts_are_pinned() {
    assert_eq!(
        RATIFIED_RESOLUTION_TABLES.len(),
        EXPECTED_RESOLUTION_TABLE_COUNT,
        "§16.33 item 5 has {EXPECTED_RESOLUTION_TABLE_COUNT} resolution tables (cache root, config \
         root)"
    );
    for table in RATIFIED_RESOLUTION_TABLES {
        assert_eq!(
            table.rows.len(),
            EXPECTED_ROWS_PER_TABLE,
            "the {} table covers {EXPECTED_ROWS_PER_TABLE} platforms",
            table.root
        );
    }
    assert_eq!(
        all_ratified_rows().len(),
        EXPECTED_RESOLUTION_ROW_COUNT,
        "§16.33 item 5's two tables have {EXPECTED_RESOLUTION_ROW_COUNT} rows (3 platforms x 2 \
         roots)"
    );
    assert_eq!(
        RATIFIED_PATH_LITERALS.len(),
        EXPECTED_PATH_LITERAL_COUNT,
        "§16.33 item 5's two tables prescribe {EXPECTED_PATH_LITERAL_COUNT} default-root literals"
    );
    assert_eq!(
        RATIFIED_SCOPE_PHRASES.len(),
        EXPECTED_SCOPE_PHRASE_COUNT,
        "{EXPECTED_SCOPE_PHRASE_COUNT} ratified scope phrases are gated"
    );
}

#[test]
// spec §16.33 items 3, 4, 5 — each resolution table, located by its own lead-in label, must hold
// exactly its three ratified rows in the ratified order. Row-to-root attribution is the point: this
// asserts the rows found UNDER "Cache root, in order, first match wins:" are the cache rows, which a
// search for free-floating row strings anywhere in §16.33 cannot assert.
//
// What must break for this to fail, verified by running each of these on 2026-08-04:
//   - swap the two lead-in sentences only, leaving all six row lines untouched (inverts cache↔config
//     on all three platforms, reverting E1): FAILS on both tables. The predecessor version of this
//     test PASSED that mutation 6/6.
//   - swap the two Windows cells between the cache and config tables (the E1 revert): FAILS.
//   - change any single cell of any row: FAILS.
//   - reorder rows within a table, or duplicate one: FAILS.
//   - move §16.33 so the span search lands elsewhere, or separate a label from its table: FAILS.
//
// What deliberately does NOT fail it: padding a cell or realigning a column, since comparison runs
// over whitespace-collapsed text.
fn each_ratified_resolution_table_holds_exactly_its_rows_in_order_under_its_own_lead_in_label() {
    let section = ratification_section();
    for table in RATIFIED_RESOLUTION_TABLES {
        let found = table_rows_under(&section, table.lead_in);
        let expected: Vec<String> = table
            .rows
            .iter()
            .map(|row| collapse_whitespace(row))
            .collect();
        assert_eq!(
            found, expected,
            "the table under {:?} in §16.33 is not the ratified {} table. A difference means a cell \
             was edited, a row reordered or duplicated, or the two lead-in labels were swapped so \
             each table is now attributed to the wrong root — note that the individual directory \
             literals may still all be present in the section, which is why this gate matches whole \
             rows inside a labeled span.",
            table.lead_in, table.root
        );
    }
}

#[test]
// spec §16.33 items 3, 4, 5 — additionally, no ratified row may appear a second time anywhere else in
// §16.33. The span test above pins each table's contents; this one catches a stray second copy of a
// row outside the two spans, which would give a future editor two places to disagree.
fn no_ratified_resolution_row_appears_twice_in_the_ratification_section() {
    let section = ratification_section();
    let normalized: Vec<String> = section.lines().map(collapse_whitespace).collect();
    let mut wrong = Vec::new();
    for row in all_ratified_rows() {
        let wanted = collapse_whitespace(row);
        let occurrences = normalized.iter().filter(|line| **line == wanted).count();
        if occurrences != 1 {
            wrong.push(format!("{occurrences}x (expected 1): {row}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {EXPECTED_RESOLUTION_ROW_COUNT} ratified resolution-table row(s) do not appear \
         exactly once as a line inside §16.33's own section:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
}

#[test]
// Keeps the two hard-coded constants in this file honest about each other: every literal in
// RATIFIED_PATH_LITERALS must occur in at least one row of RATIFIED_RESOLUTION_TABLES, and every row
// must be covered by at least one literal. This is a self-consistency check on this file's own
// constants — it asserts nothing about the spec, and is named so it cannot be mistaken for doing so.
fn this_files_two_hardcoded_literal_lists_agree_with_each_other() {
    let rows = all_ratified_rows();

    let mut orphan_literals = Vec::new();
    for literal in RATIFIED_PATH_LITERALS {
        if !rows.iter().any(|row| row.contains(literal)) {
            orphan_literals.push(*literal);
        }
    }
    assert!(
        orphan_literals.is_empty(),
        "literal(s) in RATIFIED_PATH_LITERALS appear in no row of RATIFIED_RESOLUTION_TABLES, so \
         one of the two constants has a typo: {orphan_literals:?}"
    );

    let mut uncovered_rows = Vec::new();
    for row in &rows {
        if !RATIFIED_PATH_LITERALS
            .iter()
            .any(|literal| row.contains(literal))
        {
            uncovered_rows.push(*row);
        }
    }
    assert!(
        uncovered_rows.is_empty(),
        "row(s) of RATIFIED_RESOLUTION_TABLES contain none of RATIFIED_PATH_LITERALS: \
         {uncovered_rows:?}"
    );
}

#[test]
// spec §16.33 item 3 — the two Windows literals must also be readable at the decisions register,
// which is the document `docs/PIPELINE_SPEC_V1.md` line 6 and `.github/workflows/ci.yml` both cite
// as the platform authority. Item 2 measured what a citation with no target costs.
//
// Only the two Windows rows are required here, not all ten: the decisions register records the
// v1.1 change, and it is not a second copy of the spec's full table. Asserting all ten would force
// duplication nobody ratified.
fn the_windows_root_literals_appear_verbatim_in_the_decisions_register() {
    let decisions = read("docs/ARCHITECTURE_DECISIONS.md");
    for literal in [r"%LOCALAPPDATA%\panel-ocr", r"%APPDATA%\panel-ocr"] {
        assert!(
            decisions.contains(literal),
            "docs/ARCHITECTURE_DECISIONS.md does not name `{literal}` verbatim, so its \
             `Platform support` section does not actually record the E1 split its citers rely on \
             it for"
        );
    }
}

#[test]
// spec §16.33 items 8, 9, 11 — each ratified scope phrase must be readable in BOTH documents.
// Cookbook rule 1's dominant defect class applied to prose rather than to a test name: a narrow
// claim restated in wider terms is how every over-generalisation in the F1 sequence happened, and
// the defence is to keep the narrow wording itself present and greppable.
fn every_ratified_scope_phrase_appears_verbatim_in_both_documents() {
    let documents = [
        ("docs/PIPELINE_SPEC_V1.md §16.33", ratification_section()),
        (
            "docs/ARCHITECTURE_DECISIONS.md",
            read("docs/ARCHITECTURE_DECISIONS.md"),
        ),
    ];
    let mut absent = Vec::new();
    for (label, text) in &documents {
        for (phrase, why) in RATIFIED_SCOPE_PHRASES {
            if !text.contains(phrase) {
                absent.push(format!("{label}: {phrase:?} — {why}"));
            }
        }
    }
    assert!(
        absent.is_empty(),
        "{} ratified scope phrase(s) are missing:\n  {}",
        absent.len(),
        absent.join("\n  ")
    );
}

#[test]
// The DEVIATION register must actually carry the entry §16.33 item 3 says it registers, and the
// entry must name both the number and the site. §14's own closing line requires a
// `// DEVIATION(n): ...` comment at the implementation site; that comment is owed by the next
// pipeline step, so this asserts the register half only, and says so rather than implying more.
fn the_windows_root_deviation_is_registered_in_section_14() {
    let spec = read("docs/PIPELINE_SPEC_V1.md");
    let start = spec
        .find("## 14. Deliberate deviations from upstream")
        .expect("§14 must exist");
    let end = spec[start..]
        .find("\n## 15.")
        .map_or(spec.len(), |offset| start + offset);
    let section = &spec[start..end];

    assert!(
        section.contains("19. **Windows cache root is `%LOCALAPPDATA%`"),
        "§14 must carry item 19 registering the Windows cache-root divergence from upstream \
         (§16.33 item 3)"
    );
    assert!(
        section.contains("DEVIATION(19)"),
        "§14 item 19 must name the `DEVIATION(19)` marker its implementation site owes"
    );
    assert!(
        section.contains("crates/pc-cli/src/paths.rs"),
        "§14 item 19 must name the implementation site that owes the DEVIATION(19) comment, so a \
         future parity investigation can find it"
    );
}
