//! `cargo xtask calibrate-goldens` — spec §7.3, task F2.
//!
//! F2 **measures and records**; it does not gate. Its output, `docs/GOLDEN_CALIBRATION.md`,
//! is the artifact §7.3 requires to exist *before* the §11.7(B)12 parity test may be
//! unignored. Where a measurement's input has not been recorded yet, the row is written
//! as BLOCKED with the reason — a missing number is information, not something to
//! silently omit.

use crate::paths;
use crate::record::{self, INTER_AREA_REFERENCE, INTER_AREA_TARGET, NLM_BUBBLES};
use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, RgbImage};
use pc_testkit::golden::{GoldenReport, GoldenThresholds};
use pc_testkit::metrics;
use std::fmt::Write as _;

pub fn run(out: Option<&std::path::Path>) -> Result<()> {
    let mut body = String::new();
    write_header(&mut body);

    let nlm_ok = section_nlm(&mut body)?;
    let area_ok = section_inter_area(&mut body)?;
    section_blocked(&mut body);
    write_verdict(&mut body, nlm_ok, area_ok);

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

fn section_nlm(body: &mut String) -> Result<bool> {
    body.push_str("## 1. NLM vs. `cv2.fastNlMeansDenoising` — §11.7(B)12 (frozen gate)\n\n");
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

    for name in NLM_BUBBLES {
        let reference_path = paths::recorded_root()
            .join("nlm")
            .join(record::nlm_reference_name(name));
        if !reference_path.is_file() {
            body.push_str(&format!(
                "- `{name}`: **BLOCKED** — `{}` not recorded. Run `cargo xtask record-fixtures --only nlm`.\n",
                paths::display_relative(&reference_path)
            ));
            all_met = false;
            continue;
        }
        let source = paths::upstream_root()
            .join("demo_bubbles")
            .join(format!("{name}_bubble_raw.png"));
        let input = image::open(&source)
            .with_context(|| format!("decoding {}", source.display()))?
            .to_luma8();
        let ours = match pc_denoise::nlm::denoise(
            &DynamicImage::ImageLuma8(input),
            pc_denoise::nlm::NlmParams::defaults(),
        ) {
            DynamicImage::ImageLuma8(gray) => gray,
            other => other.to_luma8(),
        };
        let reference: GrayImage = image::open(&reference_path)
            .with_context(|| format!("decoding {}", reference_path.display()))?
            .to_luma8();

        let report = GoldenReport::compare_gray(name, &reference, &ours);
        let unmet = thresholds.unmet(&report);
        all_met &= unmet.is_empty();
        any_measured = true;
        rows.push((report, unmet));
    }

    if !rows.is_empty() {
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
        provenance(body, &paths::recorded_root().join("nlm/PROVENANCE.json"));
    }

    let _ = writeln!(
        body,
        "**§7.3 gate status:** {}\n",
        if any_measured && all_met {
            "SATISFIED — `n1_nlm.rs::b12_recorded_opencv_parity` may be (and is) unignored."
        } else {
            "NOT satisfied — the §11.7(B)12 test must stay `#[ignore]`d."
        }
    );
    Ok(all_met && any_measured)
}

// ---------------------------------------------------- §8.7(A)2 INTER_AREA gate

fn section_inter_area(body: &mut String) -> Result<bool> {
    body.push_str("## 2. `resize_area` vs. `cv2.INTER_AREA` — §8.7(A)2 (frozen gate)\n\n");
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

    let reference_path = paths::recorded_root().join(INTER_AREA_REFERENCE);
    let source_path = paths::upstream_root().join("long_strip.jpg");
    if !reference_path.is_file() {
        body.push_str(&format!(
            "**BLOCKED** — `{}` not recorded. Run `cargo xtask record-fixtures --only inter-area`.\n\n",
            paths::display_relative(&reference_path)
        ));
        return Ok(false);
    }

    let source: RgbImage = image::open(&source_path)
        .with_context(|| format!("decoding {}", source_path.display()))?
        .to_rgb8();
    let ours = pc_detect::resize::resize_area(&source, INTER_AREA_TARGET.0, INTER_AREA_TARGET.1);
    let reference: RgbImage = image::open(&reference_path)
        .with_context(|| format!("decoding {}", reference_path.display()))?
        .to_rgb8();

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
    // A missing file stays a silent skip: `calibrate-goldens` is meant to run in a checkout where
    // nothing has been recorded. A file that is PRESENT but does not carry the metrics is a
    // different case and says so in the output, because that is the state the old code hid.
    let provenance_path = paths::recorded_root().join("inter_area/PROVENANCE.json");
    if let Ok(text) = std::fs::read_to_string(&provenance_path) {
        match serde_json::from_str::<pc_testkit::provenance::GroupProvenance>(&text)
            .ok()
            .and_then(|parsed| parsed.diagnostics.get("metrics_vs_reference").cloned())
        {
            Some(metrics) => {
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
            None => {
                let _ = writeln!(
                    body,
                    "| cv2-from-JPEG vs. cv2-from-PNG (decoder diagnostic, non-gating) | — | **UNREADABLE** | — | — |\n\n\
                     > `{}` exists but carries no `diagnostics.metrics_vs_reference`. Re-record with \
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
    provenance(
        body,
        &paths::recorded_root().join("inter_area/PROVENANCE.json"),
    );
    Ok(met)
}

// ------------------------------------------------------------ blocked sections

fn section_blocked(body: &mut String) {
    body.push_str("## 3. Measurements blocked on the detector recording (F1, detector group)\n\n");
    body.push_str(
        "None of the following can be measured until spec §8.5 tasks **D1** (`pc-models`) and\n\
         **D4** (`pc-detect/src/onnx.rs`) exist and the recording is run on a machine holding\n\
         `comictextdetector.pt.onnx`. `cargo xtask record-fixtures --only detector` prints the\n\
         same explanation with the exact artifact list.\n\n\
         | Measurement | Spec | Blocked test |\n\
         |---|---|---|\n\
         | demo_bubbles masking calibration report (IoU, exact %, max Δ, SSIM per fixture) | §10.7(B)15 | *(non-gating report; no test)* |\n\
         | Hand-written detect determinism + committed-raw equality | §8.7(A)6 / §16.20 item 1(b) | `pc-detect d7_run.rs::a6_pending_recorded_page_equality_and_determinism` |\n\
         | Recorded-page box-count/coordinate regression lock | §8.7(B)9 | `pc-detect d7_run.rs::b9_pending_recorded_page_regression_lock` |\n\
         | Hand-written preprocess tier arithmetic | §9.7(B)11 / §16.20 item 1(a) | `pc-preprocess p5_run.rs::b11_pending_recorded_page_tier_arithmetic` |\n\
         | End-to-end denoise golden PNGs (`_noise_mask.png`, `_clean_denoised.png`) | §11.7(B)13 / §16.10 item 20 | `pc-denoise n4_run.rs::b13_pending_recorded_page_end_to_end_golden` |\n\n\
         §10.7(B)15 additionally needs one or two license-clean full manga pages (≤ 400 KB\n\
         each) supplied by the maintainer (§7.2), which no automated step can source.\n\n",
    );

    body.push_str("## 4. PIL `FIND_EDGES` cross-check — §10.3 step 2 / §16.9 item 21\n\n");
    body.push_str(
        "Not a fixture: `pc_mask::border` consumes no recorded file. `cargo xtask\n\
         record-fixtures --only find-edges` runs real `PIL.ImageFilter.FIND_EDGES` over all\n\
         512 distinct 3×3 masks plus 500 random masks and asserts agreement with §10.3\n\
         step 2's closed form, and that a fully-set 3×3 mask yields **8** edges. That is the\n\
         empirical confirmation of §16.9 item 21's hand proof; a disagreement is escalated,\n\
         never patched. See the run log for the current result.\n\n",
    );
}

fn write_verdict(body: &mut String, nlm_ok: bool, area_ok: bool) {
    body.push_str("## 5. Verdict\n\n");
    let _ = writeln!(
        body,
        "| Gate | Spec | Status |\n\
         |---|---|---|\n\
         | NLM parity | §11.7(B)12 | {} |\n\
         | INTER_AREA parity | §8.7(A)2 | {} |\n\
         | Detector-dependent goldens | §8.7(A)6, §8.7(B)9, §9.7(B)11, §10.7(B)15, §11.7(B)13 | BLOCKED on D1+D4 |\n",
        if nlm_ok { "MET" } else { "NOT MET / BLOCKED" },
        if area_ok { "MET" } else { "NOT MET / BLOCKED" },
    );
}

fn provenance(body: &mut String, path: &std::path::Path) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let _ = writeln!(
        body,
        "<details><summary>Recording provenance (`{}`)</summary>\n\n```json\n{}\n```\n\n</details>\n",
        paths::display_relative(path),
        text.trim_end()
    );
}
