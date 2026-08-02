//! §16.24 item 11's four doc-shape gates for `docs/DETECTOR_ORACLE.md` — TESTS, not checklist
//! prose. Landing this alongside the F1 atomic recording commit, so these run live rather than
//! in the dormant-but-verified form §16.29 item 3 permits for an early skeleton.

use std::collections::BTreeSet;
use std::path::PathBuf;

fn doc_text() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/DETECTOR_ORACLE.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

/// Pulls every `` `verdict` `` token out of the doc's own markdown tables, keyed by which
/// section it appeared under, by walking lines rather than a full markdown parser — the doc's
/// table shape is authored and controlled by this same commit, so a line-oriented walk is exact
/// for it and does not need a general-purpose dependency.
struct DocRow {
    field: String,
    verdict: String,
}

fn parse_rows_in_section(text: &str, heading: &str) -> Vec<DocRow> {
    let mut rows = Vec::new();
    let mut in_section = false;
    for line in text.lines() {
        if line.starts_with("### ") || line.starts_with("## ") {
            in_section = line.trim_start_matches('#').trim() == heading;
            continue;
        }
        if !in_section {
            continue;
        }
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = line
            .trim_matches('|')
            .split('|')
            .map(|cell| cell.trim())
            .collect();
        if cells.len() < 2 {
            continue;
        }
        // Skip the header row and the `---|---` separator row.
        if cells[0] == "Field" || cells[0].chars().all(|c| c == '-') {
            continue;
        }
        // The field cell is a backtick-delimited code span, sometimes followed by trailing
        // prose (e.g. "`blocks` (container)") — take only the code span itself.
        let field = match cells[0].split('`').nth(1) {
            Some(code_span) => code_span.to_owned(),
            None => continue,
        };
        let verdict = cells[1].trim_matches('`').to_owned();
        if field.is_empty() || verdict.is_empty() || verdict == "\u{2014}" {
            // "—" (em dash): a documented container row with no scalar verdict of its own
            // (`blocks`, decomposed into `DetectedBlock`'s own rows elsewhere).
            continue;
        }
        rows.push(DocRow { field, verdict });
    }
    rows
}

fn json_keys(path: &str) -> BTreeSet<String> {
    object_keys(&read_json(path), path)
}

fn read_json(path: &str) -> serde_json::Value {
    let full = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(path);
    let bytes = std::fs::read(&full)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", full.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", full.display()))
}

fn object_keys(value: &serde_json::Value, context: &str) -> BTreeSet<String> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("{context} is not a JSON object"))
        .keys()
        .cloned()
        .collect()
}

/// Keys of the first element of a top-level JSON array file (e.g. `detector_blocks.json`).
fn json_array_element_keys(path: &str) -> BTreeSet<String> {
    let value = read_json(path);
    let array = value
        .as_array()
        .unwrap_or_else(|| panic!("{path} is not a JSON array at its top level"));
    let first = array
        .first()
        .unwrap_or_else(|| panic!("{path} is an empty array; no element to read keys from"));
    object_keys(first, &format!("{path}'s first element"))
}

/// Keys of the first element of a named array field inside a top-level JSON object file
/// (e.g. `#raw.json`'s `blocks` field).
fn json_object_field_array_element_keys(path: &str, field: &str) -> BTreeSet<String> {
    let value = read_json(path);
    let array = value
        .get(field)
        .unwrap_or_else(|| panic!("{path} has no top-level field `{field}`"))
        .as_array()
        .unwrap_or_else(|| panic!("{path}'s `{field}` field is not a JSON array"));
    let first = array.first().unwrap_or_else(|| {
        panic!("{path}'s `{field}` array is empty; no element to read keys from")
    });
    object_keys(first, &format!("{path}'s `{field}`[0]"))
}

const RAW_JSON: &str =
    "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01#raw.json";
const DETECTOR_BLOCKS_JSON: &str =
    "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_detector_blocks.json";

/// §16.24 item 11, gate 1: "set equality between the doc table's field column and the
/// serialized key set, **both directions**, duplicates rejected."
///
/// `PageDataRaw`'s own row set excludes `blocks` (the container is decomposed into
/// `DetectedBlock`'s own rows in a separate table, per the doc's own text) — so the top-level
/// key-set check drops `blocks` from both sides rather than requiring a table row for a
/// container that has no scalar verdict of its own.
#[test]
fn field_column_matches_serialized_keys_both_directions() {
    let text = doc_text();

    let page_data_raw_rows = parse_rows_in_section(
        &text,
        "`PageDataRaw` (upstream `#raw.json` / ours `#raw.json`, top-level keys)",
    );
    let detected_block_rows =
        parse_rows_in_section(&text, "`DetectedBlock` (per-element fields of `blocks`)");
    let raw_block_rows = parse_rows_in_section(
        &text,
        "`RawBlock` (`detector_blocks.json`, the pre-coverage-filter artifact `OursSide` reads)",
    );

    assert!(
        !page_data_raw_rows.is_empty(),
        "no PageDataRaw rows parsed out of the doc"
    );
    assert!(
        !detected_block_rows.is_empty(),
        "no DetectedBlock rows parsed out of the doc"
    );
    assert!(
        !raw_block_rows.is_empty(),
        "no RawBlock rows parsed out of the doc"
    );

    let check_group = |label: &str, rows: &[DocRow], actual_keys: BTreeSet<String>| {
        let mut doc_fields: Vec<String> = rows.iter().map(|row| row.field.clone()).collect();
        doc_fields.sort();
        let duplicates: Vec<&String> = doc_fields
            .windows(2)
            .filter(|pair| pair[0] == pair[1])
            .map(|pair| &pair[0])
            .collect();
        assert!(
            duplicates.is_empty(),
            "{label}: duplicate field row(s) in the doc: {duplicates:?}"
        );
        let doc_field_set: BTreeSet<String> = doc_fields.into_iter().collect();
        let missing_from_doc: Vec<&String> = actual_keys.difference(&doc_field_set).collect();
        let missing_from_artifact: Vec<&String> = doc_field_set.difference(&actual_keys).collect();
        assert!(
            missing_from_doc.is_empty() && missing_from_artifact.is_empty(),
            "{label}: field-column/key-set mismatch; in artifact but not doc: {missing_from_doc:?}; in doc but not artifact: {missing_from_artifact:?}"
        );
    };

    let mut raw_json_keys = json_keys(RAW_JSON);
    raw_json_keys.remove("blocks");
    check_group("PageDataRaw", &page_data_raw_rows, raw_json_keys);
    check_group(
        "DetectedBlock",
        &detected_block_rows,
        json_object_field_array_element_keys(RAW_JSON, "blocks"),
    );
    check_group(
        "RawBlock",
        &raw_block_rows,
        json_array_element_keys(DETECTOR_BLOCKS_JSON),
    );
}

/// §16.24 item 11, gate 2: every `EXPLAINED-§x.y` anchor present in the spec.
#[test]
fn every_explained_anchor_resolves_in_the_spec() {
    let doc = doc_text();
    let spec_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/PIPELINE_SPEC_V1.md");
    let spec = std::fs::read_to_string(&spec_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", spec_path.display()));

    let needle = "EXPLAINED-\u{a7}";
    let mut anchors = Vec::new();
    let mut search_from = 0usize;
    while let Some(offset) = doc[search_from..].find(needle) {
        let anchor_start = search_from + offset;
        let tail = &doc[anchor_start + needle.len()..];
        let tail_len = tail
            .char_indices()
            .find(|(_, c)| !(c.is_ascii_digit() || *c == '.'))
            .map(|(index, _)| index)
            .unwrap_or(tail.len());
        let dotted = &tail[..tail_len];
        // Only a real `§N.M` citation (two non-empty all-digit groups) is a genuine anchor; the
        // doc's own generic template sentence names the bucket as "EXPLAINED-§14.x" with a
        // literal `x`, which trims to "14." here and must not be checked as a citation.
        let is_real_citation = match dotted.split_once('.') {
            Some((section, item)) => {
                !section.is_empty()
                    && !item.is_empty()
                    && section.chars().all(|c| c.is_ascii_digit())
                    && item.chars().all(|c| c.is_ascii_digit())
            }
            None => false,
        };
        if is_real_citation {
            anchors.push(format!("{needle}{dotted}"));
        }
        search_from = anchor_start + needle.len();
    }

    assert!(
        !anchors.is_empty(),
        "expected at least one EXPLAINED-§x.y row in the doc (raw_mask, mask_coverage)"
    );

    for anchor in &anchors {
        // "EXPLAINED-§14.12" -> the spec's own citable form is the bare dotted anchor "§14.12"
        // (see e.g. docs/PIPELINE_SPEC_V1.md's §14 item 17 text, which cites "§14.12" this way).
        let dotted = anchor.trim_start_matches("EXPLAINED-");
        assert!(
            spec.contains(dotted),
            "anchor `{anchor}` (looked for `{dotted}`) does not resolve in docs/PIPELINE_SPEC_V1.md"
        );
    }
}

/// §16.24 item 11, gate 3: no `OPEN` row once the fixture is present. The fixture already
/// exists (this is the atomic commit landing it), so this runs live, not dormant.
#[test]
fn no_open_row_once_the_fixture_is_present() {
    let text = doc_text();
    let sections = [
        "`PageDataRaw` (upstream `#raw.json` / ours `#raw.json`, top-level keys)",
        "`DetectedBlock` (per-element fields of `blocks`)",
        "`RawBlock` (`detector_blocks.json`, the pre-coverage-filter artifact `OursSide` reads)",
        "`PageData` (§2.5, preprocessor output — out of scope for a *detector* oracle)",
    ];
    for section in sections {
        let rows = parse_rows_in_section(&text, section);
        assert!(!rows.is_empty(), "section `{section}` parsed no rows");
        for row in rows {
            assert_ne!(
                row.verdict, "OPEN",
                "row `{}` in section `{section}` is OPEN; every row must close before the fixture commits",
                row.field
            );
        }
    }
}

/// §16.24 item 11, gate 4 / §16.20 item 3(f): three complete, role-distinct, non-self
/// signatures.
#[test]
fn three_signatures_are_recorded_role_distinct_and_non_self() {
    let text = doc_text();
    let heading = "## Signatures (§16.20 item 3(f))";
    let start = text
        .find(heading)
        .unwrap_or_else(|| panic!("doc has no `{heading}` section"));
    let section = &text[start..];

    let mut signers = Vec::new();
    for line in section.lines() {
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = line
            .trim_matches('|')
            .split('|')
            .map(|cell| cell.trim())
            .collect();
        if cells.len() < 2 {
            continue;
        }
        if cells[0] == "#" || cells[0].chars().all(|c| c == '-') {
            continue;
        }
        signers.push(cells[1].to_owned());
    }

    assert_eq!(
        signers.len(),
        3,
        "expected exactly 3 signature rows, found {}: {signers:?}",
        signers.len()
    );

    // Read the producing agent's own name off the "Producing agent" row rather than hard-coding
    // a marker string, so this check stays data-driven if that row's wording ever changes.
    let producing_agent_line = text
        .lines()
        .find(|line| {
            line.trim_start_matches('|')
                .trim_start()
                .starts_with("Producing agent")
        })
        .unwrap_or_else(|| panic!("doc has no `Producing agent` row for this test to read"));
    let producing_agent = producing_agent_line
        .trim_matches('|')
        .split('|')
        .nth(1)
        .unwrap_or_else(|| panic!("`Producing agent` row has no second cell"))
        .trim();
    // The first word of the producing-agent cell (e.g. "Claude") is the self-reference token:
    // a signer whose cell contains that word, case-insensitively, anywhere, is self-review
    // regardless of how the rest of the cell is phrased — narrower substring matches (like the
    // literal "orchestrator session" marker this test used to check) miss a signer who is the
    // same agent under different wording.
    let self_reference_token = producing_agent
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("`Producing agent` cell `{producing_agent}` is empty"))
        .to_ascii_lowercase();
    for signer in &signers {
        assert!(
            !signer.to_ascii_lowercase().contains(&self_reference_token),
            "signer `{signer}` names the producing agent (`{producing_agent}`) — §16.13 item 4 forbids self-review"
        );
    }

    let distinct: BTreeSet<&String> = signers.iter().collect();
    assert_eq!(
        distinct.len(),
        3,
        "all 3 signers must be role-distinct; got {signers:?}"
    );

    // §16.24 item 6: all three signatures land IN the atomic commit, not after it — so this
    // gate must fail, not merely warn, while any signature is still pending. A test that passed
    // with a pending row would let an under-signed doc through the exact gate meant to stop it.
    let pending = signers
        .iter()
        .filter(|signer| signer.contains("pending"))
        .count();
    assert_eq!(
        pending, 0,
        "{pending} signature row(s) still pending; all 3 must be recorded before this fixture commits (§16.24 item 6)"
    );
}
