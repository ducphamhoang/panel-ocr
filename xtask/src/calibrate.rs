//! `cargo xtask calibrate-goldens` — spec §7.3, task F2.
//!
//! F2 **measures and records**; it does not gate. Its output, `docs/GOLDEN_CALIBRATION.md`,
//! is the artifact §7.3 requires to exist *before* the §11.7(B)12 parity test may be
//! unignored. Where a measurement's input has not been recorded yet, the row is written
//! as BLOCKED with the reason — a missing number is information, not something to
//! silently omit.

use crate::paths;
use crate::record::{self, INTER_AREA_REFERENCE, INTER_AREA_TARGET, NLM_BUBBLES};
use anyhow::{anyhow, Context, Result};
use image::{DynamicImage, GrayImage, RgbImage};
use pc_core::PageDataRaw;
use pc_detect::oracle::{self, Derivation, Expectations, ExpectedPair, IdentityBranch, Mechanism};
use pc_detect::RawBlock;
use pc_testkit::golden::{GoldenReport, GoldenThresholds};
use pc_testkit::metrics;
use std::fmt::Write as _;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Section {
    title: String,
    body: String,
}

fn render(sections: &[Section]) -> String {
    let mut rendered = String::new();
    for (index, section) in sections.iter().enumerate() {
        let _ = writeln!(rendered, "## {}. {}\n", index + 1, section.title);
        rendered.push_str(section.body.trim_end());
        rendered.push_str("\n\n");
    }
    rendered
}

pub fn run(out: Option<&std::path::Path>) -> Result<()> {
    let body = render_document()?;

    let target = out
        .map(|path| path.to_path_buf())
        .unwrap_or_else(|| paths::docs_root().join("GOLDEN_CALIBRATION.md"));
    std::fs::write(&target, &body).with_context(|| format!("writing {}", target.display()))?;
    println!(
        "\nwrote {} ({} bytes)",
        paths::display_relative(&target),
        body.len()
    );
    Ok(())
}

fn render_document() -> Result<String> {
    let statuses: Vec<_> = TRACKED_TESTS.iter().map(tracked_status).collect();
    let (nlm, nlm_ok) = section_nlm()?;
    let (area, area_ok) = section_inter_area()?;
    let detector_section = match detector_box_counts() {
        Ok(facts) => section_detector_box_counts(&facts),
        Err(error) => section_detector_box_counts_blocked(&error),
    };
    let sections = vec![
        nlm,
        area,
        section_detector_status(&statuses),
        // TODO(F2-2): demo_bubbles masking calibration report, §10.7(B)15
        section_demo_bubbles_status(),
        detector_section,
        section_find_edges(),
        write_verdict(nlm_ok, area_ok, &statuses),
    ];
    let mut document = String::new();
    write_header(&mut document);
    document.push_str(&render(&sections));
    Ok(document)
}

fn write_header(body: &mut String) {
    body.push_str(
        "# Golden calibration (spec §7.3, task F2)\n\
         \n\
         **Generated in full by `cargo xtask calibrate-goldens` — do not hand-edit.** Every\n\
         number here is measured at generation time; re-run the command to refresh it.\n\
         Recording provenance (tool versions, parameters, output hashes) is inlined below\n\
         from each fixture directory's `PROVENANCE.json`.\n\
         \n\
         §7.3's rule, restated: these numbers must be *measured before a tolerance is\n\
         frozen*, never fitted afterwards. If a measurement misses a specified tolerance,\n\
         the discrepancy goes back to the two architects jointly (CLAUDE.md) — it is never\n\
         resolved by loosening the threshold here.\n\n",
    );
}

// ------------------------------------------------------- §11.7(B)12 NLM parity

fn section_nlm() -> Result<(Section, bool)> {
    section_nlm_under(&paths::recorded_root(), &paths::upstream_root())
}

fn section_nlm_under(recorded_root: &Path, upstream_root: &Path) -> Result<(Section, bool)> {
    let mut body = String::new();
    body.push_str(
        "Input: `tests/fixtures/upstream/demo_bubbles/<name>_bubble_raw.png` as luma8.\n\
         Reference: `tests/fixtures/recorded/nlm/<name>_h10_t7_s21.png`, produced by\n\
         `cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)`\n\
         (§16.10 item 19). Ours: `pc_denoise::nlm::denoise` with `NlmParams::defaults()`.\n\
         Specified tolerances (`GoldenThresholds::nlm_parity()`): SSIM ≥ 0.98,\n\
         mean |Δ| ≤ 1.0, max Δ ≤ 8.\n\n",
    );

    let thresholds = GoldenThresholds::nlm_parity();
    let mut rows = Vec::new();
    let mut all_met = true;
    let mut any_measured = false;
    let mut blocked = false;

    for name in NLM_BUBBLES {
        let reference_path = recorded_root
            .join("nlm")
            .join(record::nlm_reference_name(name));
        if !reference_path.is_file() {
            body.push_str(&format!(
                "- `{name}`: **BLOCKED** — `{}` not recorded. Run `cargo xtask record-fixtures --only nlm`.\n",
                paths::display_relative(&reference_path)
            ));
            all_met = false;
            blocked = true;
            continue;
        }
        let source = upstream_root
            .join("demo_bubbles")
            .join(format!("{name}_bubble_raw.png"));
        let input = match image::open(&source) {
            Ok(image) => image.to_luma8(),
            Err(error) => {
                append_blocked_image_row(&mut body, name, "source", &source, &error);
                all_met = false;
                blocked = true;
                continue;
            }
        };
        let ours = match pc_denoise::nlm::denoise(
            &DynamicImage::ImageLuma8(input),
            pc_denoise::nlm::NlmParams::defaults(),
        ) {
            DynamicImage::ImageLuma8(gray) => gray,
            other => other.to_luma8(),
        };
        let reference: GrayImage = match image::open(&reference_path) {
            Ok(image) => image.to_luma8(),
            Err(error) => {
                append_blocked_image_row(&mut body, name, "reference", &reference_path, &error);
                all_met = false;
                blocked = true;
                continue;
            }
        };
        if reference.dimensions() != ours.dimensions() {
            append_blocked_dimension_row(
                &mut body,
                name,
                "reference",
                reference.dimensions(),
                "denoised output",
                ours.dimensions(),
            );
            all_met = false;
            blocked = true;
            continue;
        }

        let report = GoldenReport::compare_gray(name, &reference, &ours);
        let unmet = thresholds.unmet(&report);
        all_met &= unmet.is_empty();
        any_measured = true;
        rows.push((report, unmet));
    }

    if !rows.is_empty() {
        if blocked {
            body.push('\n');
        }
        body.push_str(&GoldenReport::markdown_header());
        body.push('\n');
        for (report, _) in &rows {
            body.push_str(&report.to_markdown_row());
            body.push('\n');
        }
        body.push('\n');
        for (report, unmet) in &rows {
            if unmet.is_empty() {
                let _ = writeln!(
                    body,
                    "- `{}`: **all specified tolerances met.**",
                    report.name
                );
            } else {
                let _ = writeln!(
                    body,
                    "- `{}`: **UNMET** — {}",
                    report.name,
                    unmet.join("; ")
                );
            }
        }
        body.push('\n');
    }
    if blocked && rows.is_empty() {
        body.push('\n');
    }
    provenance(&mut body, &recorded_root.join("nlm/PROVENANCE.json"));

    let _ = writeln!(
        &mut body,
        "**§7.3 gate status:** {}\n",
        if any_measured && all_met {
            "SATISFIED — `n1_nlm.rs::b12_recorded_opencv_parity` may be (and is) unignored."
        } else {
            "NOT satisfied — the §11.7(B)12 test must stay `#[ignore]`d."
        }
    );
    Ok((
        Section {
            title: "NLM vs. `cv2.fastNlMeansDenoising` — §11.7(B)12 (frozen gate)".into(),
            body,
        },
        all_met && any_measured,
    ))
}

fn append_blocked_image_row(
    body: &mut String,
    name: &str,
    role: &str,
    path: &Path,
    error: &impl std::fmt::Display,
) {
    let _ = writeln!(
        body,
        "- `{name}`: **BLOCKED** — could not decode {role} `{}`: `{error}`.",
        paths::display_relative(path)
    );
}

fn append_blocked_dimension_row(
    body: &mut String,
    name: &str,
    role: &str,
    actual: (u32, u32),
    expected_role: &str,
    expected: (u32, u32),
) {
    let _ = writeln!(
        body,
        "- `{name}`: **BLOCKED** — {role} dimensions {}×{} do not match {expected_role} dimensions {}×{}.",
        actual.0, actual.1, expected.0, expected.1
    );
}

// ---------------------------------------------------- §8.7(A)2 INTER_AREA gate

fn section_inter_area() -> Result<(Section, bool)> {
    section_inter_area_under(&paths::recorded_root(), &paths::upstream_root())
}

fn section_inter_area_under(recorded_root: &Path, upstream_root: &Path) -> Result<(Section, bool)> {
    let mut body = String::new();
    body.push_str(&format!(
        "Downscale of `long_strip.jpg` to {}×{}. Specified tolerances: mean |Δ| ≤ 1.0 and\n\
         max per-channel Δ ≤ 2.\n\n\
         §16.13 item 5: the reference was recorded from a **lossless PNG re-encode of the\n\
         Rust-side JPEG decode**, so the gate measures resize arithmetic and not\n\
         libjpeg-turbo-vs-`image` decoder differences. The second row below is the\n\
         diagnostic that quantifies how large that decoder difference is; it is recorded\n\
         for visibility and is **not** a gate.\n\n",
        INTER_AREA_TARGET.0, INTER_AREA_TARGET.1
    ));

    let reference_path = recorded_root.join(INTER_AREA_REFERENCE);
    let source_path = upstream_root.join("long_strip.jpg");
    if !reference_path.is_file() {
        body.push_str(&format!(
            "**BLOCKED** — `{}` not recorded. Run `cargo xtask record-fixtures --only inter-area`.\n\n",
            paths::display_relative(&reference_path)
        ));
        provenance(&mut body, &recorded_root.join("inter_area/PROVENANCE.json"));
        return Ok((
            Section {
                title: "`resize_area` vs. `cv2.INTER_AREA` — §8.7(A)2 (frozen gate)".into(),
                body,
            },
            false,
        ));
    }

    let source: RgbImage = match image::open(&source_path) {
        Ok(image) => image.to_rgb8(),
        Err(error) => {
            append_blocked_image_row(&mut body, "long_strip", "source", &source_path, &error);
            body.push('\n');
            provenance(&mut body, &recorded_root.join("inter_area/PROVENANCE.json"));
            return Ok((
                Section {
                    title: "`resize_area` vs. `cv2.INTER_AREA` — §8.7(A)2 (frozen gate)".into(),
                    body,
                },
                false,
            ));
        }
    };
    let ours = pc_detect::resize::resize_area(&source, INTER_AREA_TARGET.0, INTER_AREA_TARGET.1);
    let reference: RgbImage = match image::open(&reference_path) {
        Ok(image) => image.to_rgb8(),
        Err(error) => {
            append_blocked_image_row(
                &mut body,
                "long_strip",
                "reference",
                &reference_path,
                &error,
            );
            body.push('\n');
            provenance(&mut body, &recorded_root.join("inter_area/PROVENANCE.json"));
            return Ok((
                Section {
                    title: "`resize_area` vs. `cv2.INTER_AREA` — §8.7(A)2 (frozen gate)".into(),
                    body,
                },
                false,
            ));
        }
    };
    if reference.dimensions() != ours.dimensions() {
        append_blocked_dimension_row(
            &mut body,
            "long_strip",
            "reference",
            reference.dimensions(),
            "resized output",
            ours.dimensions(),
        );
        body.push('\n');
        provenance(&mut body, &recorded_root.join("inter_area/PROVENANCE.json"));
        return Ok((
            Section {
                title: "`resize_area` vs. `cv2.INTER_AREA` — §8.7(A)2 (frozen gate)".into(),
                body,
            },
            false,
        ));
    }

    body.push_str(
        "| Comparison | Dimensions | Mean abs diff | Max per-channel Δ | SSIM (as gray) |\n",
    );
    body.push_str("|---|---:|---:|---:|---:|\n");
    let mean = metrics::mean_abs_diff_rgb(&reference, &ours);
    let max = metrics::max_delta_rgb(&reference, &ours);
    let ssim = metrics::ssim_rgb_as_gray(&reference, &ours);
    let _ = writeln!(
        body,
        "| ours vs. cv2 INTER_AREA (gate) | {}×{} | {mean:.6} | {max} | {ssim:.6} |",
        reference.width(),
        reference.height()
    );

    // The diagnostic image itself is scratch (§16.13 item 6); its metrics were computed
    // at record time and live in PROVENANCE.json.
    // Read through the shared canonical schema (§16.17 item 2 / §16.24 item 1), NOT by
    // hand-indexing a `serde_json::Value`. The pre-migration shape nested these metrics under a
    // `jpeg_decode_diagnostic` key, the canonical shape puts them in the group's `diagnostics`
    // map, and `Value`'s index returns `Null` for a missing key — so the old lookup combined with
    // the `is_null()` skip below silently DROPPED this row from a generated document rather than
    // failing. §16.24 item 19(c)'s reader enumeration missed this consumer entirely.
    //
    // FOUR states, distinguished, each with a TRUTHFUL message. Only absence is silent, because
    // `calibrate-goldens` is meant to run in a checkout where nothing has been recorded. The first
    // version of this fix collapsed the other three: it used `if let Ok(text) = read_to_string`,
    // which hid an unreadable-but-present file as if it were absent, and `.ok()`, which reported a
    // PARSE failure with the message "carries no `diagnostics.metrics_vs_reference`" — a false
    // message, cookbook rule 1's corollary, and the same defect as §16.24 item 20(b)'s
    // `NonRelativePath`-for-a-misplaced-path. Distinguishing them costs four lines.
    let provenance_path = recorded_root.join("inter_area/PROVENANCE.json");
    if provenance_path.is_file() {
        let metrics = std::fs::read_to_string(&provenance_path)
            .map_err(|error| format!("exists but could not be read: {error}"))
            .and_then(|text| {
                serde_json::from_str::<pc_testkit::provenance::GroupProvenance>(&text).map_err(
                    |error| {
                        format!("does not parse as canonical provenance (§16.17 item 2): {error}")
                    },
                )
            })
            .and_then(|parsed| {
                parsed
                    .diagnostics
                    .get("metrics_vs_reference")
                    .cloned()
                    .ok_or_else(|| {
                        "parses, but carries no `diagnostics.metrics_vs_reference`".to_owned()
                    })
            });
        match metrics {
            Ok(metrics) => {
                let _ = writeln!(
                    body,
                    "| cv2-from-JPEG vs. cv2-from-PNG (decoder diagnostic, non-gating) | {}×{} | {:.6} | {} | {:.6} |",
                    INTER_AREA_TARGET.0,
                    INTER_AREA_TARGET.1,
                    metrics["mean_abs_diff"].as_f64().unwrap_or(f64::NAN),
                    metrics["max_delta"].as_u64().unwrap_or(0),
                    metrics["ssim_as_gray"].as_f64().unwrap_or(f64::NAN),
                );
            }
            Err(why) => {
                let _ = writeln!(
                    body,
                    "| cv2-from-JPEG vs. cv2-from-PNG (decoder diagnostic, non-gating) | — | **UNREADABLE** | — | — |\n\n\
                     > `{}` {why}. Re-record with \
                     `cargo xtask record-fixtures --only inter-area --force`. This row is reported \
                     rather than omitted deliberately: silently dropping it is how a provenance \
                     shape change went unnoticed (§16.24 item 19).",
                    paths::display_relative(&provenance_path)
                );
            }
        }
    }
    body.push('\n');

    let met = mean <= 1.0 && max <= 2;
    let _ = writeln!(
        body,
        "**§8.7(A)2 status:** {}\n",
        if met {
            "tolerances met — `d2_resize.rs::a2_inter_area_matches_the_recorded_opencv_reference` \
             may be (and is) unignored."
        } else {
            "**UNMET** — the test must stay `#[ignore]`d and the shortfall goes back to the \
             two architects jointly (§7.3)."
        }
    );
    provenance(&mut body, &recorded_root.join("inter_area/PROVENANCE.json"));
    Ok((
        Section {
            title: "`resize_area` vs. `cv2.INTER_AREA` — §8.7(A)2 (frozen gate)".into(),
            body,
        },
        met,
    ))
}

// ------------------------------------------------------- detector test status

#[derive(Debug, Clone, Copy)]
struct TrackedTest {
    spec: &'static str,
    subject: &'static str,
    test_file: &'static str,
    test_fn: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TrackedStatus {
    Live,
    Ignored { reason: String },
    Unknown { why: String },
}

const TRACKED_TESTS: &[TrackedTest] = &[
    TrackedTest {
        spec: "§8.7(A)6",
        subject: "Hand-written detect determinism + committed-raw equality",
        test_file: "crates/pc-detect/tests/d7_run.rs",
        test_fn: "a6_pending_recorded_page_equality_and_determinism",
    },
    TrackedTest {
        spec: "§8.7(B)9",
        subject: "Recorded-page box-count/coordinate regression lock",
        test_file: "crates/pc-detect/tests/d7_run.rs",
        test_fn: "b9_pending_recorded_page_regression_lock",
    },
    TrackedTest {
        spec: "§9.7(B)11",
        subject: "Hand-written preprocess tier arithmetic",
        test_file: "crates/pc-preprocess/tests/p5_run.rs",
        test_fn: "b11_pending_recorded_page_tier_arithmetic",
    },
    TrackedTest {
        spec: "§11.7(B)13",
        subject: "End-to-end denoise golden PNGs (_noise_mask.png, _clean_denoised.png)",
        test_file: "crates/pc-denoise/tests/n4_run.rs",
        test_fn: "b13_pending_recorded_page_end_to_end_golden",
    },
];

fn tracked_status(test: &TrackedTest) -> TrackedStatus {
    tracked_status_under(&paths::workspace_root(), test)
}

fn tracked_status_under(root: &std::path::Path, test: &TrackedTest) -> TrackedStatus {
    let path = root.join(test.test_file);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            return TrackedStatus::Unknown {
                why: format!(
                    "could not read `{}`: {error}",
                    paths::display_relative(&path)
                ),
            }
        }
    };
    let needle = format!("fn {}(", test.test_fn);
    let lines: Vec<&str> = text.lines().collect();
    let matches: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| line.contains(&needle).then_some(index))
        .collect();
    let [function_line] = matches.as_slice() else {
        return TrackedStatus::Unknown {
            why: format!(
                "expected exactly one `{needle}` in `{}`; found {}",
                paths::display_relative(&path),
                matches.len()
            ),
        };
    };

    let mut index = *function_line;
    while index > 0 {
        let line = lines[index - 1].trim_start();
        if line.starts_with("#[") || line.starts_with("//") {
            if line.starts_with("#[ignore") {
                return TrackedStatus::Ignored {
                    reason: ignore_reason(line),
                };
            }
            index -= 1;
        } else {
            break;
        }
    }
    TrackedStatus::Live
}

fn ignore_reason(attribute: &str) -> String {
    let Some(equals) = attribute.find('=') else {
        return attribute.to_owned();
    };
    let value = &attribute[equals + 1..];
    let Some(start) = value.find('"') else {
        return attribute.to_owned();
    };
    let remainder = &value[start + 1..];
    let Some(end) = remainder.find('"') else {
        return attribute.to_owned();
    };
    remainder[..end].to_owned()
}

fn tracked_status_label(status: &TrackedStatus) -> String {
    match status {
        TrackedStatus::Live => "LIVE".into(),
        TrackedStatus::Ignored { reason } => format!("IGNORED — {reason}"),
        TrackedStatus::Unknown { why } => format!("UNKNOWN — {why}"),
    }
}

fn section_detector_status(statuses: &[TrackedStatus]) -> Section {
    let mut body = String::from(
        "Measured from the test attributes in the current workspace tree; `UNKNOWN` is never treated as live or ignored.\n\n\
         | Test | Spec | Status |\n|---|---|---|\n",
    );
    for (test, status) in TRACKED_TESTS.iter().zip(statuses) {
        let _ = writeln!(
            body,
            "| {} (`{}`) | {} | {} |",
            test.subject,
            test.test_fn,
            test.spec,
            tracked_status_label(status)
        );
    }
    body.push('\n');
    for (test, status) in TRACKED_TESTS.iter().zip(statuses) {
        if let TrackedStatus::Ignored { reason } = status {
            let _ = writeln!(
                body,
                "`{}` (`{}`) is ignored with measured reason: `{reason}`.",
                test.subject, test.test_fn
            );
        }
    }
    body.push_str(&format!(
        "\nAmong the {} tracked tests, §11.7(B)13's denoise golden requires special handling: deriving its golden references by running our own denoiser would violate the \"no self-oracle\" rule (cookbook rule 7); its derivation/signing decision needs the joint architects, not a unilateral fix.\n\n",
        TRACKED_TESTS.len()
    ));
    // This is intentionally the verbatim §16.24 item 12 quote. Its word "four" is spec text,
    // not a derived count; the dynamic count belongs in the preceding sentence.
    body.push_str(
        "Those four tests un-ignore only against the signed maintainer page, never against `demo_bubbles`. (§16.24 item 12)\n\n",
    );
    Section {
        title: "Detector-dependent test status".into(),
        body,
    }
}

const DEMO_BUBBLES_BLOCKED_REASON: &str =
    "Scratch-only demo_bubbles masking calibration report (§10.7(B)15) is not implemented; this remaining scope within F2 will be added in a subsequent F2 sub-task/PR per the ratified §16.24 item 12 approach: record to scratch, commit nothing.";

fn section_demo_bubbles_status() -> Section {
    Section {
        title: "Reserved for demo_bubbles masking calibration".into(),
        body: format!(
            "| Measurement | Spec | Status | Reason |\n|---|---|---|---|\n| demo_bubbles masking calibration report | §10.7(B)15 | BLOCKED | {DEMO_BUBBLES_BLOCKED_REASON} |\n"
        ),
    }
}

fn section_find_edges() -> Section {
    Section {
        title: "PIL `FIND_EDGES` cross-check — §10.3 step 2 / §16.9 item 21".into(),
        body: "Not a fixture: `pc_mask::border` consumes no recorded file. `cargo xtask\nrecord-fixtures --only find-edges` runs real `PIL.ImageFilter.FIND_EDGES` over all\n512 distinct 3×3 masks plus 500 random masks and asserts agreement with §10.3\nstep 2's closed form, and that a fully-set 3×3 mask yields **8** edges. That is the\nempirical confirmation of §16.9 item 21's hand proof; a disagreement is escalated,\nnever patched. See the run log for the current result.\n".into(),
    }
}

fn write_verdict(nlm_ok: bool, area_ok: bool, statuses: &[TrackedStatus]) -> Section {
    let mut body = String::from("| Gate | Spec | Status |\n|---|---|---|\n");
    let _ = writeln!(
        body,
        "| NLM parity | §11.7(B)12 | {} |",
        if nlm_ok { "MET" } else { "NOT MET / BLOCKED" }
    );
    let _ = writeln!(
        body,
        "| INTER_AREA parity | §8.7(A)2 | {} |",
        if area_ok { "MET" } else { "NOT MET / BLOCKED" }
    );
    let _ = writeln!(
        body,
        "| demo_bubbles masking calibration report | §10.7(B)15 | BLOCKED — {DEMO_BUBBLES_BLOCKED_REASON} |"
    );
    for (test, status) in TRACKED_TESTS.iter().zip(statuses) {
        let _ = writeln!(
            body,
            "| {} | {} | {} |",
            test.subject,
            test.spec,
            tracked_status_label(status)
        );
    }
    Section {
        title: "Verdict".into(),
        body,
    }
}

// ------------------------------------------------------- §15.1 box counts

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoxCountFacts {
    stem: String,
    ours_total: usize,
    upstream_total: usize,
    pairs_compared: usize,
    gating_rows: usize,
    unmatched_ours: Vec<(usize, &'static str)>,
    unmatched_upstream: Vec<(usize, &'static str)>,
}

const DETECTOR_STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";

// Deliberate duplicate transcription of the frozen test's signed pairing rather than an import
// from `crates/pc-detect/tests/f1_real_page_gate.rs::signed_expectations()`. `oracle::compare`
// verifies CoverageFilteredUpstream strongly by reconstructing and comparing the cited rect;
// ClassDuplicateOf is weaker, checking only that its cited upstream index is in range and paired,
// not that the citation is semantically the correct duplicate.
fn authored_expectations() -> Expectations {
    Expectations {
        pairs: vec![
            ExpectedPair {
                ours: 0,
                upstream: 3,
                branch: IdentityBranch::LineInformed,
                derivation: Some(Derivation::YoloSynthesizedCorners),
            },
            ExpectedPair {
                ours: 1,
                upstream: 0,
                branch: IdentityBranch::LineInformed,
                derivation: Some(Derivation::YoloUnioned),
            },
            ExpectedPair {
                ours: 2,
                upstream: 2,
                branch: IdentityBranch::LineInformed,
                derivation: Some(Derivation::YoloSynthesizedCorners),
            },
        ],
        unmatched_ours: vec![oracle::UnmatchedEntry {
            index: 3,
            mechanism: Mechanism::CoverageFilteredUpstream {
                pre_filter_index: 4,
            },
        }],
        unmatched_upstream: vec![oracle::UnmatchedEntry {
            index: 1,
            mechanism: Mechanism::ClassDuplicateOf { upstream_index: 2 },
        }],
        totals: oracle::Totals {
            pairs: 3,
            ours_total: 4,
            upstream_total: 4,
        },
    }
}

fn validate_authored_mechanisms(expectations: &Expectations) -> Result<()> {
    for entry in expectations
        .unmatched_ours
        .iter()
        .chain(expectations.unmatched_upstream.iter())
    {
        if !matches!(
            entry.mechanism,
            Mechanism::ClassDuplicateOf { .. } | Mechanism::CoverageFilteredUpstream { .. }
        ) {
            return Err(anyhow!(
                "xtask authored unsupported oracle mechanism {}",
                entry.mechanism.variant_name()
            ));
        }
    }
    Ok(())
}

fn detector_box_counts() -> Result<BoxCountFacts> {
    detector_box_counts_under(&pc_testkit::paths::recorded_root())
}

fn detector_box_counts_under(recorded_root: &Path) -> Result<BoxCountFacts> {
    let ours_path = recorded_root.join(format!("detector/{DETECTOR_STEM}_detector_blocks.json"));
    let ours: Vec<RawBlock> = serde_json::from_slice(
        &std::fs::read(&ours_path)
            .with_context(|| format!("reading {}", paths::display_relative(&ours_path)))?,
    )
    .with_context(|| {
        format!(
            "parsing {} as Vec<RawBlock>",
            paths::display_relative(&ours_path)
        )
    })?;

    let raw_path = recorded_root.join(format!("detector/{DETECTOR_STEM}#raw.json"));
    let page: PageDataRaw = serde_json::from_slice(
        &std::fs::read(&raw_path)
            .with_context(|| format!("reading {}", paths::display_relative(&raw_path)))?,
    )
    .with_context(|| {
        format!(
            "parsing {} as PageDataRaw",
            paths::display_relative(&raw_path)
        )
    })?;

    let upstream_path =
        recorded_root.join(format!("detector/{DETECTOR_STEM}_upstream_oracle.json"));
    let upstream: oracle::UpstreamOracle = serde_json::from_slice(
        &std::fs::read(&upstream_path)
            .with_context(|| format!("reading {}", paths::display_relative(&upstream_path)))?,
    )
    .with_context(|| {
        format!(
            "parsing {} as UpstreamOracle",
            paths::display_relative(&upstream_path)
        )
    })?;

    let expectations = authored_expectations();
    validate_authored_mechanisms(&expectations)?;
    let report = oracle::compare(
        oracle::OursSide {
            blocks: &ours,
            scale: page.scale,
            image_size: page.image_size,
        },
        &upstream,
        &expectations,
    );

    let unmatched_ours = expectations
        .unmatched_ours
        .iter()
        .map(|entry| (entry.index, entry.mechanism.variant_name()))
        .collect();
    let unmatched_upstream = expectations
        .unmatched_upstream
        .iter()
        .map(|entry| (entry.index, entry.mechanism.variant_name()))
        .collect();
    Ok(BoxCountFacts {
        stem: DETECTOR_STEM.into(),
        ours_total: ours.len(),
        upstream_total: upstream.blocks.len(),
        pairs_compared: report.pairs_compared,
        gating_rows: report.gating().len(),
        unmatched_ours,
        unmatched_upstream,
    })
}

fn section_detector_box_counts(facts: &BoxCountFacts) -> Section {
    let mut body = String::new();
    let _ = writeln!(body, "Recorded page: `{}`.\n", facts.stem);
    let _ = writeln!(
        body,
        "| Ours total | Upstream total | Pairs compared | Gating rows |\n|---:|---:|---:|---:|\n| {} | {} | {} | {} |\n",
        facts.ours_total, facts.upstream_total, facts.pairs_compared, facts.gating_rows
    );
    body.push_str("| Side | Unmatched index | Mechanism |\n|---|---:|---|\n");
    for (index, mechanism) in &facts.unmatched_ours {
        let _ = writeln!(body, "| ours | {index} | `{mechanism}` |");
    }
    for (index, mechanism) in &facts.unmatched_upstream {
        let _ = writeln!(body, "| upstream | {index} | `{mechanism}` |");
        if *mechanism == "ClassDuplicateOf" {
            let _ = writeln!(
                body,
                "\nThe unmatched upstream `{mechanism}` entry at index {index} is the expected, accepted consequence of §15.1's ratified class-agnostic NMS decision and deviation §14.13; it is not an anomaly."
            );
        }
    }
    body.push('\n');
    provenance(
        &mut body,
        &paths::recorded_root().join("detector/PROVENANCE.json"),
    );
    Section {
        title: "Detector box-count comparison — §15.1".into(),
        body,
    }
}

fn section_detector_box_counts_blocked(error: &anyhow::Error) -> Section {
    let mut body = format!(
        "| Measurement | Spec | Status | Reason |\n|---|---|---|---|\n| detector box-count comparison | §15.1 | **BLOCKED** | could not be measured: `{error:#}`. The committed detector fixture set is incomplete or unreadable; restore or re-record it before rerunning `cargo xtask calibrate-goldens`. |\n"
    );
    body.push('\n');
    provenance(
        &mut body,
        &paths::recorded_root().join("detector/PROVENANCE.json"),
    );
    Section {
        title: "Detector box-count comparison — §15.1".into(),
        body,
    }
}

fn provenance(body: &mut String, path: &std::path::Path) {
    // Same discipline as the decoder-diagnostic row above: absence is silent (nothing recorded
    // yet), but a file that EXISTS and cannot be read is reported into the document rather than
    // dropped. The previous `let Ok(..) else { return }` made those two indistinguishable, so a
    // corrupt or unreadable provenance file removed its own audit trail from the generated doc —
    // which is the failure mode §16.24 item 19 is about.
    if !path.is_file() {
        return;
    }
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            let _ = writeln!(
                body,
                "> **Recording provenance `{}` could not be read: {error}.** Reported rather than \
                 omitted: a missing provenance block would otherwise be indistinguishable from a \
                 group that was never recorded.\n",
                paths::display_relative(path)
            );
            return;
        }
    };
    let _ = writeln!(
        body,
        "<details><summary>Recording provenance (`{}`)</summary>\n\n```json\n{}\n```\n\n</details>\n",
        paths::display_relative(path),
        text.trim_end()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_detect::oracle::{self, Mechanism, UnmatchedEntry};

    #[test]
    fn tracked_status_reads_the_real_tree() {
        let statuses: Vec<_> = TRACKED_TESTS.iter().map(tracked_status).collect();
        assert!(matches!(statuses[0], TrackedStatus::Live));
        assert!(matches!(statuses[1], TrackedStatus::Live));
        assert!(matches!(statuses[2], TrackedStatus::Live));
        match &statuses[3] {
            TrackedStatus::Ignored { reason } => assert!(reason.contains("golden PNGs")),
            other => panic!("expected the denoise golden test to be ignored, got {other:?}"),
        }
    }

    #[test]
    fn tracked_status_reports_unknown_for_a_missing_fn() {
        let missing_fn = TrackedTest {
            spec: "§test",
            subject: "missing function",
            test_file: "crates/pc-detect/tests/d7_run.rs",
            test_fn: "this_function_does_not_exist",
        };
        let missing_file = TrackedTest {
            spec: "§test",
            subject: "missing file",
            test_file: "crates/pc-detect/tests/this_file_does_not_exist.rs",
            test_fn: "f",
        };
        match tracked_status(&missing_fn) {
            TrackedStatus::Unknown { why } => {
                assert!(why.contains("crates/pc-detect/tests/d7_run.rs"));
                assert!(!why.contains(&paths::workspace_root().display().to_string()));
            }
            other => panic!("expected unknown status, got {other:?}"),
        }
        assert!(matches!(
            tracked_status(&missing_file),
            TrackedStatus::Unknown { .. }
        ));
    }

    #[test]
    fn tracked_status_sees_an_ignore_above_intervening_attributes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("fixture.rs");
        let ignored = "#[ignore = \"x\"]\n#[test]\n/// documented\nfn f() {}\n";
        std::fs::write(&path, ignored).expect("write ignored fixture");
        let tracked = TrackedTest {
            spec: "§test",
            subject: "fixture",
            test_file: "fixture.rs",
            test_fn: "f",
        };
        assert!(matches!(
            tracked_status_under(dir.path(), &tracked),
            TrackedStatus::Ignored { .. }
        ));

        std::fs::write(&path, "#[test]\n/// documented\nfn f() {}\n").expect("write live fixture");
        assert!(matches!(
            tracked_status_under(dir.path(), &tracked),
            TrackedStatus::Live
        ));
    }

    #[test]
    fn the_document_headings_are_numbered_in_order() {
        let document = render_document().expect("render document");
        let headings: Vec<_> = document
            .lines()
            .filter(|line| line.starts_with("## "))
            .collect();
        assert_eq!(
            headings,
            vec![
                "## 1. NLM vs. `cv2.fastNlMeansDenoising` — §11.7(B)12 (frozen gate)",
                "## 2. `resize_area` vs. `cv2.INTER_AREA` — §8.7(A)2 (frozen gate)",
                "## 3. Detector-dependent test status",
                "## 4. Reserved for demo_bubbles masking calibration",
                "## 5. Detector box-count comparison — §15.1",
                "## 6. PIL `FIND_EDGES` cross-check — §10.3 step 2 / §16.9 item 21",
                "## 7. Verdict",
            ]
        );
    }

    #[test]
    fn the_generated_document_no_longer_claims_stale_d1_d4_blocking() {
        let document = render_document().expect("render document");
        assert!(!document.contains("BLOCKED on D1+D4"));
        assert!(document.contains(
            "Those four tests un-ignore only against the signed maintainer page, never against `demo_bubbles`."
        ));
        assert!(document.contains("(§16.24 item 12)"));
    }

    #[test]
    fn the_report_does_not_claim_the_demo_bubbles_report_needs_a_maintainer_manga_page() {
        let document = render_document().expect("render document");
        assert!(!document.contains("license-clean full manga pages"));
    }

    #[test]
    fn the_committed_document_matches_render_document() {
        let document = render_document().expect("render document");
        assert_eq!(document, include_str!("../../docs/GOLDEN_CALIBRATION.md"));
    }

    #[test]
    fn the_demo_bubbles_report_is_explicitly_blocked_in_section_four_and_the_verdict() {
        let document = render_document().expect("render document");
        assert!(document.contains(&format!(
            "| demo_bubbles masking calibration report | §10.7(B)15 | BLOCKED | {DEMO_BUBBLES_BLOCKED_REASON} |"
        )));
        assert!(document.contains(&format!(
            "| demo_bubbles masking calibration report | §10.7(B)15 | BLOCKED — {DEMO_BUBBLES_BLOCKED_REASON} |"
        )));
        assert!(document.contains("remaining scope within F2"));
        assert!(!document.contains("separate F2 follow-up task"));
    }

    #[test]
    fn the_detector_provenance_inlines_json_content_not_just_its_path() {
        let document = render_document().expect("render document");
        assert!(document.contains("tests/fixtures/recorded/detector/PROVENANCE.json"));
        assert!(
            document.contains("c6cb214360132a1b52b09a5a084735930ee9f757c560067a810fa37c1cb6d686")
        );
    }

    #[test]
    fn the_status_rule_is_emitted_even_when_every_tracked_test_is_live() {
        let statuses = vec![TrackedStatus::Live; TRACKED_TESTS.len()];
        let section = section_detector_status(&statuses);
        assert!(section.body.contains(
            "Those four tests un-ignore only against the signed maintainer page, never against `demo_bubbles`."
        ));
        assert!(section.body.contains("(§16.24 item 12)"));
        assert!(section.body.contains("§11.7(B)13's denoise golden"));
        assert!(section
            .body
            .contains(&format!("{} tracked tests", TRACKED_TESTS.len())));
        assert!(!section.body.contains("all four tracked tests"));
    }

    #[test]
    fn the_status_section_prints_each_measured_ignore_reason() {
        let statuses = vec![
            TrackedStatus::Ignored {
                reason: "measured reason one".into(),
            },
            TrackedStatus::Ignored {
                reason: "measured reason two".into(),
            },
            TrackedStatus::Live,
            TrackedStatus::Live,
        ];
        let section = section_detector_status(&statuses);
        assert!(section.body.contains("measured reason one"));
        assert!(section.body.contains("measured reason two"));
    }

    #[test]
    fn section_nlm_blocks_a_corrupt_reference_without_aborting() {
        let (recorded_root, upstream_root) = copy_nlm_fixtures();
        std::fs::write(
            recorded_root.path().join("nlm/ray_h10_t7_s21.png"),
            b"not a PNG",
        )
        .expect("corrupt synthetic NLM reference");

        let (section, _) = section_nlm_under(recorded_root.path(), upstream_root.path())
            .expect("corrupt NLM reference must become a blocked row");
        assert!(section.body.contains("`ray`: **BLOCKED**"));
        assert!(section.body.contains("could not decode reference"));
        let blocked_end = section
            .body
            .find("could not decode reference")
            .expect("blocked reference detail");
        assert!(section.body[blocked_end..].contains("\n\n| Name |"));
    }

    #[test]
    fn section_nlm_blocks_a_reference_dimension_mismatch_without_aborting() {
        let (recorded_root, upstream_root) = copy_nlm_fixtures();
        GrayImage::new(8, 8)
            .save(recorded_root.path().join("nlm/ray_h10_t7_s21.png"))
            .expect("write wrong-sized NLM reference");

        let (section, _) = section_nlm_under(recorded_root.path(), upstream_root.path())
            .expect("wrong-sized NLM reference must become a blocked row");
        assert!(section.body.contains(
            "`ray`: **BLOCKED** — reference dimensions 8×8 do not match denoised output dimensions 256×329."
        ));
    }

    #[test]
    fn section_nlm_blocks_a_missing_source_without_aborting() {
        let (recorded_root, upstream_root) = copy_nlm_fixtures();
        std::fs::remove_file(upstream_root.path().join("demo_bubbles/ray_bubble_raw.png"))
            .expect("remove synthetic NLM source");

        let (section, _) = section_nlm_under(recorded_root.path(), upstream_root.path())
            .expect("missing NLM source must become a blocked row");
        assert!(section.body.contains("`ray`: **BLOCKED**"));
        assert!(section.body.contains("could not decode source"));
    }

    #[test]
    fn section_inter_area_blocks_a_corrupt_reference_without_aborting() {
        let (recorded_root, upstream_root) = copy_inter_area_fixtures();
        std::fs::write(
            recorded_root
                .path()
                .join("inter_area/long_strip_inter_area_500x4000.png"),
            b"not a PNG",
        )
        .expect("corrupt synthetic INTER_AREA reference");

        let (section, _) = section_inter_area_under(recorded_root.path(), upstream_root.path())
            .expect("corrupt INTER_AREA reference must become a blocked row");
        assert!(section.body.contains("**BLOCKED**"));
        assert!(section.body.contains("could not decode reference"));
        let blocked_end = section
            .body
            .find("could not decode reference")
            .expect("blocked reference detail");
        assert!(section.body[blocked_end..].contains("\n\n<details><summary>Recording provenance"));
    }

    #[test]
    fn section_inter_area_blocks_a_reference_dimension_mismatch_without_aborting() {
        let (recorded_root, upstream_root) = copy_inter_area_fixtures();
        RgbImage::new(8, 8)
            .save(
                recorded_root
                    .path()
                    .join("inter_area/long_strip_inter_area_500x4000.png"),
            )
            .expect("write wrong-sized INTER_AREA reference");

        let (section, _) = section_inter_area_under(recorded_root.path(), upstream_root.path())
            .expect("wrong-sized INTER_AREA reference must become a blocked row");
        assert!(section.body.contains(
            "**BLOCKED** — reference dimensions 8×8 do not match resized output dimensions 500×4000."
        ));
        let blocked_end = section
            .body
            .find("reference dimensions 8×8")
            .expect("blocked dimension detail");
        assert!(section.body[blocked_end..].contains("\n\n<details>"));
    }

    #[test]
    fn section_inter_area_blocks_a_missing_source_without_aborting() {
        let (recorded_root, upstream_root) = copy_inter_area_fixtures();
        std::fs::remove_file(upstream_root.path().join("long_strip.jpg"))
            .expect("remove synthetic INTER_AREA source");

        let (section, _) = section_inter_area_under(recorded_root.path(), upstream_root.path())
            .expect("missing INTER_AREA source must become a blocked row");
        assert!(section.body.contains("**BLOCKED**"));
        assert!(section.body.contains("could not decode source"));
        let blocked_end = section
            .body
            .find("could not decode source")
            .expect("blocked source detail");
        assert!(section.body[blocked_end..].contains("\n\n<details>"));
    }

    #[test]
    fn section_nlm_blocks_all_references_with_spacing_before_provenance() {
        let (recorded_root, upstream_root) = copy_nlm_fixtures();
        for name in NLM_BUBBLES {
            std::fs::write(
                recorded_root
                    .path()
                    .join("nlm")
                    .join(record::nlm_reference_name(name)),
                b"not a PNG",
            )
            .expect("corrupt synthetic NLM reference");
        }

        let (section, _) = section_nlm_under(recorded_root.path(), upstream_root.path())
            .expect("all blocked NLM references must remain renderable");
        assert!(section.body.contains("`nightmare`: **BLOCKED**"));
        assert!(section.body.contains("`ray`: **BLOCKED**"));
        let blocked_end = section
            .body
            .rfind("could not decode reference")
            .expect("last blocked reference detail");
        assert!(section.body[blocked_end..].contains("\n\n<details>"));
    }

    #[test]
    fn the_blocked_detector_reason_keeps_error_detail_without_workspace_paths() {
        let recorded_root = tempfile::tempdir_in(paths::workspace_root()).expect("temp dir");
        let error = detector_box_counts_under(recorded_root.path()).expect_err("missing fixture");
        let section = section_detector_box_counts_blocked(&error);
        assert!(section.body.contains("No such file or directory"));
        assert!(!section
            .body
            .contains(&paths::workspace_root().display().to_string()));
    }

    #[test]
    fn a_blocked_detector_section_still_inlines_detector_provenance() {
        let error = anyhow!("synthetic detector fixture failure");
        let section = section_detector_box_counts_blocked(&error);
        assert!(section
            .body
            .contains("tests/fixtures/recorded/detector/PROVENANCE.json"));
        assert!(section
            .body
            .contains("c6cb214360132a1b52b09a5a084735930ee9f757c560067a810fa37c1cb6d686"));
    }

    #[test]
    fn authored_expectations_have_not_changed_since_manual_frozen_test_verification() {
        // The frozen signed_expectations() lives in an integration-test binary and cannot be
        // imported here. These literals are maintained from manual field-by-field verification
        // against that frozen source; oracle::compare is the runtime check.
        let expectations = authored_expectations();
        assert_eq!(expectations.pairs.len(), 3);
        assert_eq!(expectations.pairs[0].ours, 0);
        assert_eq!(expectations.pairs[0].upstream, 3);
        assert_eq!(expectations.pairs[0].branch, IdentityBranch::LineInformed);
        assert_eq!(
            expectations.pairs[0].derivation,
            Some(Derivation::YoloSynthesizedCorners)
        );
        assert_eq!(expectations.pairs[1].ours, 1);
        assert_eq!(expectations.pairs[1].upstream, 0);
        assert_eq!(expectations.pairs[1].branch, IdentityBranch::LineInformed);
        assert_eq!(
            expectations.pairs[1].derivation,
            Some(Derivation::YoloUnioned)
        );
        assert_eq!(expectations.pairs[2].ours, 2);
        assert_eq!(expectations.pairs[2].upstream, 2);
        assert_eq!(expectations.pairs[2].branch, IdentityBranch::LineInformed);
        assert_eq!(
            expectations.pairs[2].derivation,
            Some(Derivation::YoloSynthesizedCorners)
        );
        assert!(matches!(
            expectations.unmatched_ours.as_slice(),
            [oracle::UnmatchedEntry {
                index: 3,
                mechanism: Mechanism::CoverageFilteredUpstream {
                    pre_filter_index: 4
                }
            }]
        ));
        assert!(matches!(
            expectations.unmatched_upstream.as_slice(),
            [oracle::UnmatchedEntry {
                index: 1,
                mechanism: Mechanism::ClassDuplicateOf { upstream_index: 2 }
            }]
        ));
        assert_eq!(expectations.totals.pairs, 3);
        assert_eq!(expectations.totals.ours_total, 4);
        assert_eq!(expectations.totals.upstream_total, 4);
    }

    #[test]
    fn the_recorded_page_box_counts_match_the_signed_hand_derivation() {
        let facts = detector_box_counts().expect("detector fixtures compare");
        assert_eq!(
            facts,
            BoxCountFacts {
                stem: "ja_Pepper-and-Carrot_by-David-Revoy_E01P01".into(),
                ours_total: 4,
                upstream_total: 4,
                pairs_compared: 3,
                gating_rows: 0,
                unmatched_ours: vec![(3, "CoverageFilteredUpstream")],
                unmatched_upstream: vec![(1, "ClassDuplicateOf")],
            }
        );
    }

    #[test]
    fn every_unpaired_box_on_both_sides_carries_a_named_mechanism() {
        let facts = detector_box_counts().expect("detector fixtures compare");
        assert_eq!(
            facts.unmatched_ours.len(),
            facts.ours_total - facts.pairs_compared
        );
        assert_eq!(
            facts.unmatched_upstream.len(),
            facts.upstream_total - facts.pairs_compared
        );
        assert_eq!(facts.unmatched_ours, vec![(3, "CoverageFilteredUpstream")]);
        assert_eq!(facts.unmatched_upstream, vec![(1, "ClassDuplicateOf")]);
    }

    #[test]
    fn section_inter_area_uses_the_supplied_recorded_root_for_diagnostics() {
        let (recorded_root, upstream_root) = copy_inter_area_fixtures();
        let provenance_path = recorded_root.path().join("inter_area/PROVENANCE.json");
        let provenance =
            std::fs::read_to_string(paths::recorded_root().join("inter_area/PROVENANCE.json"))
                .expect("read real INTER_AREA provenance");
        let provenance = provenance
            .replace("0.003438", "12.345678")
            .replace("\"max_delta\": 1", "\"max_delta\": 99")
            .replace("0.9999985218837129", "0.123456");
        std::fs::write(provenance_path, provenance).expect("write synthetic provenance");

        let (section, _) = section_inter_area_under(recorded_root.path(), upstream_root.path())
            .expect("synthetic provenance must remain readable");
        assert!(section.body.contains(
            "| cv2-from-JPEG vs. cv2-from-PNG (decoder diagnostic, non-gating) | 500×4000 | 12.345678 | 99 | 0.123456 |"
        ));
    }

    #[test]
    fn the_rendered_section_prints_the_facts_it_was_given_and_not_a_transcription() {
        let facts = BoxCountFacts {
            stem: "synthetic".into(),
            ours_total: 11,
            upstream_total: 7,
            pairs_compared: 5,
            gating_rows: 2,
            unmatched_ours: vec![(9, "ClassDuplicateOf")],
            unmatched_upstream: vec![(6, "CoverageFilteredUpstream")],
        };
        let section = section_detector_box_counts(&facts);
        assert!(section.body.contains("Recorded page: `synthetic`."));
        assert!(section.body.contains(
            "| Ours total | Upstream total | Pairs compared | Gating rows |\n|---:|---:|---:|---:|\n| 11 | 7 | 5 | 2 |"
        ));
        assert!(section.body.contains("| ours | 9 | `ClassDuplicateOf` |"));
        assert!(section
            .body
            .contains("| upstream | 6 | `CoverageFilteredUpstream` |"));
    }

    #[test]
    fn the_box_count_section_explains_the_ratified_class_duplicate_deviation() {
        let facts = BoxCountFacts {
            stem: "synthetic".into(),
            ours_total: 1,
            upstream_total: 2,
            pairs_compared: 1,
            gating_rows: 0,
            unmatched_ours: Vec::new(),
            unmatched_upstream: vec![(1, "ClassDuplicateOf")],
        };
        let section = section_detector_box_counts(&facts);
        assert!(section
            .body
            .contains("§15.1's ratified class-agnostic NMS decision"));
        assert!(section.body.contains("deviation §14.13"));
        assert!(section.body.contains("expected, accepted consequence"));
        assert!(section
            .body
            .contains("\n\n<details><summary>Recording provenance"));
    }

    #[test]
    fn a_missing_detector_fixture_returns_a_blocked_section_instead_of_panicking() {
        let dir = tempfile::tempdir().expect("temp dir");
        let error = detector_box_counts_under(dir.path()).expect_err("fixture must be missing");
        let section = section_detector_box_counts_blocked(&error);
        assert!(section.body.contains("**BLOCKED**"));
        assert!(section.body.contains("detector box-count comparison"));
        assert!(section.body.contains("could not be measured"));
    }

    #[test]
    fn xtask_authors_only_citation_verified_mechanisms() {
        let expectations = authored_expectations();
        validate_authored_mechanisms(&expectations).expect("authored mechanisms are verified");

        let error = validate_authored_mechanisms(&oracle::Expectations {
            pairs: Vec::new(),
            unmatched_ours: vec![UnmatchedEntry {
                index: 0,
                mechanism: Mechanism::DocumentedSplitMerge {
                    register_entry: "§synthetic",
                },
            }],
            unmatched_upstream: Vec::new(),
            totals: oracle::Totals {
                pairs: 0,
                ours_total: 0,
                upstream_total: 0,
            },
        })
        .expect_err("no-op mechanisms must be rejected");
        assert!(error.to_string().contains("DocumentedSplitMerge"));
    }

    fn copy_nlm_fixtures() -> (tempfile::TempDir, tempfile::TempDir) {
        let recorded_root = tempfile::tempdir().expect("recorded temp dir");
        let upstream_root = tempfile::tempdir().expect("upstream temp dir");
        std::fs::create_dir(recorded_root.path().join("nlm")).expect("NLM recorded dir");
        std::fs::create_dir(upstream_root.path().join("demo_bubbles"))
            .expect("demo bubbles upstream dir");
        std::fs::copy(
            paths::recorded_root().join("nlm/PROVENANCE.json"),
            recorded_root.path().join("nlm/PROVENANCE.json"),
        )
        .expect("copy NLM provenance");
        for name in NLM_BUBBLES {
            let reference_name = record::nlm_reference_name(name);
            std::fs::copy(
                paths::recorded_root().join("nlm").join(&reference_name),
                recorded_root.path().join("nlm").join(reference_name),
            )
            .expect("copy NLM reference");
            std::fs::copy(
                paths::upstream_root()
                    .join("demo_bubbles")
                    .join(format!("{name}_bubble_raw.png")),
                upstream_root
                    .path()
                    .join("demo_bubbles")
                    .join(format!("{name}_bubble_raw.png")),
            )
            .expect("copy NLM source");
        }
        (recorded_root, upstream_root)
    }

    fn copy_inter_area_fixtures() -> (tempfile::TempDir, tempfile::TempDir) {
        let recorded_root = tempfile::tempdir().expect("recorded temp dir");
        let upstream_root = tempfile::tempdir().expect("upstream temp dir");
        std::fs::create_dir(recorded_root.path().join("inter_area"))
            .expect("INTER_AREA recorded dir");
        std::fs::copy(
            paths::recorded_root().join("inter_area/PROVENANCE.json"),
            recorded_root.path().join("inter_area/PROVENANCE.json"),
        )
        .expect("copy INTER_AREA provenance");
        std::fs::copy(
            paths::recorded_root().join(INTER_AREA_REFERENCE),
            recorded_root.path().join(INTER_AREA_REFERENCE),
        )
        .expect("copy INTER_AREA reference");
        std::fs::copy(
            paths::upstream_root().join("long_strip.jpg"),
            upstream_root.path().join("long_strip.jpg"),
        )
        .expect("copy INTER_AREA source");
        (recorded_root, upstream_root)
    }
}
