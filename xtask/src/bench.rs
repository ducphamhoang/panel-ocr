//! `cargo xtask bench-detector` — non-gating CPU-EP tuning measurements.

use pc_detect::onnx::SessionTuning;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;
#[cfg(feature = "onnx")]
use std::{path::PathBuf, process::Command, time::Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurationSummary {
    pub kept: usize,
    pub median: Duration,
    pub min: Duration,
    pub max: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchError {
    TooFewSamples { kept: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub label: String,
    pub tuning: SessionTuning,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OracleArm {
    Skipped {
        reason: String,
    },
    Measured {
        milliseconds: f64,
        reps: usize,
        warmup: usize,
    },
}

pub fn summarise(durations: &[Duration], warmup_count: usize) -> DurationSummary {
    try_summarise(durations, warmup_count).expect("benchmark needs at least three kept samples")
}

pub fn try_summarise(
    durations: &[Duration],
    warmup_count: usize,
) -> Result<DurationSummary, BenchError> {
    let kept_durations = durations.get(warmup_count..).unwrap_or_default();
    if kept_durations.len() < 3 {
        return Err(BenchError::TooFewSamples {
            kept: kept_durations.len(),
        });
    }
    let mut sorted = kept_durations.to_vec();
    sorted.sort_unstable();
    let median = if sorted.len() % 2 == 0 {
        let left = sorted[sorted.len() / 2 - 1];
        let right = sorted[sorted.len() / 2];
        Duration::from_nanos((left.as_nanos() + right.as_nanos()) as u64 / 2)
    } else {
        sorted[sorted.len() / 2]
    };
    Ok(DurationSummary {
        kept: sorted.len(),
        median,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
    })
}

pub fn builtin_variants() -> Vec<Variant> {
    let no_flush = SessionTuning {
        flush_denormals: false,
        ..SessionTuning::default()
    };
    let parallel = SessionTuning {
        parallel_execution: true,
        ..SessionTuning::default()
    };
    let spinning = SessionTuning {
        intra_op_spinning: Some(false),
        ..SessionTuning::default()
    };
    let dynamic = SessionTuning {
        dynamic_block_base: Some(128),
        ..SessionTuning::default()
    };
    vec![
        Variant {
            label: "pinned".into(),
            tuning: SessionTuning::default(),
        },
        Variant {
            label: "flush_denormals=off".into(),
            tuning: no_flush,
        },
        Variant {
            label: "parallel_execution=on".into(),
            tuning: parallel,
        },
        Variant {
            label: "intra_op_spinning=off".into(),
            tuning: spinning,
        },
        Variant {
            label: "dynamic_block_base=128".into(),
            tuning: dynamic,
        },
    ]
}

fn render_report(label: &str, reason: &str) -> String {
    let mut report = format!("# Detector benchmark (non-gating)\n\n## {label}\n\n");
    let _ = writeln!(report, "Oracle: **SKIPPED** — {reason}");
    report
}

pub const ORACLE_FORWARD_SCRIPT: &str = r#"
import hashlib, statistics, sys, time, cv2, numpy as np
model_path, tensor_path, reps, warmup = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tensor = np.fromfile(tensor_path, dtype=np.float32).reshape((1, 3, 1024, 1024))
tensor_digest = hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
net = cv2.dnn.readNet(model_path)
output_names = net.getUnconnectedOutLayersNames()
durations = []
for index in range(warmup + reps):
    net.setInput(tensor)
    started = time.perf_counter()
    net.forward(output_names)
    elapsed = time.perf_counter() - started
    if index >= warmup:
        durations.append(elapsed * 1000.0)
print(f"ORACLE_RESULT {statistics.median(durations):.6f} {tensor_digest}")
"#;

#[cfg(not(feature = "onnx"))]
pub fn run(
    _detector: Option<&std::path::Path>,
    _reps: usize,
    _warmup: usize,
    _variants: &str,
    _out: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    println!("SKIPPED: xtask was built without the ONNX feature");
    Ok(())
}

#[cfg(feature = "onnx")]
pub fn run(
    detector: Option<&std::path::Path>,
    reps: usize,
    warmup: usize,
    variants: &str,
    out: Option<&std::path::Path>,
) -> anyhow::Result<()> {
    use anyhow::{bail, Context};

    if reps < 3 {
        bail!("--reps must be at least 3 so median/min/max are meaningful");
    }
    let model = detector
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("PANEL_OCR_ONNX_MODEL").map(PathBuf::from));
    let Some(model) = model.filter(|path| path.is_file()) else {
        println!("SKIPPED: detector model is missing; provide --detector onnx:<path> or PANEL_OCR_ONNX_MODEL");
        return Ok(());
    };

    let variants = selected_variants(variants)?;
    let scratch = crate::paths::scratch_dir().context("creating benchmark scratch directory")?;
    let mut rows = Vec::new();
    let mut baseline: Option<Vec<f32>> = None;
    for variant in variants {
        let raw_out = scratch.join(format!(
            "bench-{}-{}.f32",
            std::process::id(),
            variant.label.replace(['=', ','], "_")
        ));
        let output = Command::new(std::env::current_exe().context("locating xtask executable")?)
            .args([
                "bench-detector-child",
                "--detector",
                &model.display().to_string(),
                "--label",
                &variant.label,
                "--reps",
                &reps.to_string(),
                "--warmup",
                &warmup.to_string(),
                "--raw-out",
                &raw_out.display().to_string(),
            ])
            .output()
            .with_context(|| format!("spawning fresh child for {}", variant.label))?;
        if !output.status.success() {
            bail!(
                "benchmark child for {} failed: {}",
                variant.label,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout
            .lines()
            .find(|line| line.starts_with("BENCH_RESULT "))
            .context("benchmark child did not emit BENCH_RESULT")?;
        let fields: Vec<_> = line.split_whitespace().collect();
        let digest = fields
            .get(5)
            .context("benchmark child result lacks digest")?
            .to_string();
        let summary = DurationSummary {
            kept: fields[1].parse().context("parse kept")?,
            median: Duration::from_secs_f64(
                fields[2].parse::<f64>().context("parse median")? / 1000.0,
            ),
            min: Duration::from_secs_f64(fields[3].parse::<f64>().context("parse min")? / 1000.0),
            max: Duration::from_secs_f64(fields[4].parse::<f64>().context("parse max")? / 1000.0),
        };
        let values = read_f32_file(&raw_out)?;
        let (diff_count, max_abs) = if let Some(reference) = &baseline {
            diff_stats(reference, &values)
        } else {
            baseline = Some(values);
            (0, 0.0)
        };
        rows.push((variant.label, summary, digest, diff_count, max_abs));
        let _ = std::fs::remove_file(raw_out);
    }

    let oracle = match crate::env::PythonTooling::discover(None)? {
        Some(tooling) => measure_oracle(&tooling.interpreter, &model, &scratch, reps, warmup)?,
        None => OracleArm::Skipped {
            reason: "cv2/PIL/numpy tooling is unavailable".into(),
        },
    };
    let oracle_report = render_oracle_report(&oracle, &rows);
    let mut sections = vec![Section {
        title: "ONNX CPU detector candidates".into(),
        body: render_rows(&rows),
    }];
    sections.push(Section {
        title: "cv2.dnn oracle".into(),
        body: oracle_report,
    });
    let document = render_sections(&sections);
    if let Some(path) = out {
        std::fs::write(path, &document).with_context(|| format!("writing {}", path.display()))?;
        println!(
            "wrote {} ({} bytes)",
            crate::paths::display_relative(path),
            document.len()
        );
    } else {
        print!("{document}");
    }
    Ok(())
}

#[cfg(feature = "onnx")]
pub fn run_child(
    model: &Path,
    label: &str,
    reps: usize,
    warmup: usize,
    raw_out: &Path,
) -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    let variant = builtin_variants()
        .into_iter()
        .find(|variant| variant.label == label)
        .with_context(|| format!("unknown benchmark variant {label}"))?;
    let page =
        pc_testkit::paths::recorded("detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg");
    let decoded = image::open(&page)
        .context("decoding recorded detector page")?
        .to_rgb8();
    let (width, height, _) =
        pc_detect::calculate_new_size_and_scale(decoded.width(), decoded.height(), 1000, 4000);
    let base = pc_detect::resize_area(&decoded, width, height);
    let detector =
        pc_detect::onnx::OnnxDetector::from_path_with_tuning(model, 0, 0, &variant.tuning)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let mut durations = Vec::with_capacity(warmup + reps);
    let mut last_values = Vec::new();
    for _ in 0..warmup + reps {
        let start = Instant::now();
        let (_, values, _) = detector
            .detect_raw_tensors(&base)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        durations.push(start.elapsed());
        last_values = values;
    }
    let summary = summarise(&durations, warmup);
    let flat: Vec<f32> = last_values.into_iter().flatten().collect();
    if flat.is_empty() {
        bail!("detector returned no raw values");
    }
    write_f32_file(raw_out, &flat)?;
    println!(
        "BENCH_RESULT {} {:.3} {:.3} {:.3} {}",
        summary.kept,
        summary.median.as_secs_f64() * 1000.0,
        summary.min.as_secs_f64() * 1000.0,
        summary.max.as_secs_f64() * 1000.0,
        digest_values(&flat),
    );
    Ok(())
}

#[cfg(not(feature = "onnx"))]
pub fn run_child(
    _model: &Path,
    _label: &str,
    _reps: usize,
    _warmup: usize,
    _raw_out: &Path,
) -> anyhow::Result<()> {
    anyhow::bail!("xtask was built without the ONNX feature")
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, PartialEq)]
struct Section {
    title: String,
    body: String,
}

#[cfg(feature = "onnx")]
fn selected_variants(spec: &str) -> anyhow::Result<Vec<Variant>> {
    if spec == "all" {
        return Ok(builtin_variants());
    }
    let all = builtin_variants();
    let selected: Vec<_> = spec
        .split(',')
        .map(str::trim)
        .map(|label| {
            all.iter()
                .find(|variant| variant.label == label)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("unknown detector benchmark variant `{label}`"))
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(selected)
}

fn digest_values(values: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_ne_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn write_f32_file(path: &Path, values: &[f32]) -> anyhow::Result<()> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    std::fs::write(path, bytes).map_err(Into::into)
}

fn read_f32_file(path: &Path) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        anyhow::bail!("raw output file is not f32-aligned");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("4 bytes")))
        .collect())
}

fn diff_stats(left: &[f32], right: &[f32]) -> (usize, f32) {
    if left.len() != right.len() {
        return (left.len().max(right.len()), f32::INFINITY);
    }
    left.iter()
        .zip(right)
        .fold((0, 0.0), |(count, max_abs), (a, b)| {
            let difference = (*a - *b).abs();
            (
                count + usize::from(difference != 0.0),
                max_abs.max(difference),
            )
        })
}

fn render_rows(rows: &[(String, DurationSummary, String, usize, f32)]) -> String {
    let mut body = String::from("| variant | kept | median ms | min ms | max ms | raw sha256 | diff count | maxabs |\n|---|---:|---:|---:|---:|---|---:|---:|\n");
    for (index, (label, summary, digest, diff_count, max_abs)) in rows.iter().enumerate() {
        if index == 0 {
            let _ = writeln!(
                body,
                "| {label} | {} | {:.3} | {:.3} | {:.3} | `{digest}` | — | — |",
                summary.kept,
                summary.median.as_secs_f64() * 1000.0,
                summary.min.as_secs_f64() * 1000.0,
                summary.max.as_secs_f64() * 1000.0
            );
        } else {
            let _ = writeln!(
                body,
                "| {label} | {} | {:.3} | {:.3} | {:.3} | `{digest}` | {diff_count} | {max_abs:.6} |",
                summary.kept,
                summary.median.as_secs_f64() * 1000.0,
                summary.min.as_secs_f64() * 1000.0,
                summary.max.as_secs_f64() * 1000.0
            );
        }
    }
    body
}

#[cfg(feature = "onnx")]
fn measure_oracle(
    interpreter: &Path,
    model: &Path,
    scratch: &Path,
    reps: usize,
    warmup: usize,
) -> anyhow::Result<OracleArm> {
    let page =
        pc_testkit::paths::recorded("detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg");
    let decoded = image::open(&page)?.to_rgb8();
    let (width, height, _) =
        pc_detect::calculate_new_size_and_scale(decoded.width(), decoded.height(), 1000, 4000);
    let base = pc_detect::resize_area(&decoded, width, height);
    let boxed = pc_detect::onnx::letterbox(&base);
    let tensor = pc_detect::onnx::to_nchw(&boxed.image);
    let expected_digest = digest_values(&tensor);
    let tensor_path = scratch.join("bench-oracle-input.f32");
    write_f32_file(&tensor_path, &tensor)?;
    let script_path = scratch.join("bench-oracle-forward.py");
    std::fs::write(&script_path, ORACLE_FORWARD_SCRIPT)?;
    let output = Command::new(interpreter)
        .arg(&script_path)
        .arg(model)
        .arg(&tensor_path)
        .arg(reps.to_string())
        .arg(warmup.to_string())
        .output()?;
    let _ = std::fs::remove_file(&script_path);
    let _ = std::fs::remove_file(&tensor_path);
    if !output.status.success() {
        return Ok(OracleArm::Skipped {
            reason: format!(
                "cv2.dnn forward failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        });
    }
    let milliseconds = parse_oracle_result(&output.stdout, &expected_digest)?;
    Ok(OracleArm::Measured {
        milliseconds,
        reps,
        warmup,
    })
}

fn render_oracle_report(
    oracle: &OracleArm,
    rows: &[(String, DurationSummary, String, usize, f32)],
) -> String {
    match oracle {
        OracleArm::Skipped { reason } => render_report("oracle", reason),
        OracleArm::Measured {
            milliseconds,
            reps,
            warmup,
        } => {
            let control_ms = rows
                .iter()
                .find(|(label, ..)| label == "pinned")
                .map(|(_, summary, ..)| summary.median.as_secs_f64() * 1000.0);
            let Some(control_ms) = control_ms else {
                return "Oracle: **SKIPPED** — no pinned control result was measured; no ratio computed.\n".into();
            };
            format!(
                "Oracle: cv2.dnn median forward() over {reps} reps, {warmup} warmup excluded: {milliseconds:.3} ms; ours vs cv2.dnn ratio: {:.3}\n",
                control_ms / milliseconds,
            )
        }
    }
}

fn parse_oracle_result(stdout: &[u8], expected_digest: &str) -> anyhow::Result<f64> {
    use anyhow::{bail, Context};

    let output_text = String::from_utf8_lossy(stdout);
    let line = output_text
        .lines()
        .find(|line| line.starts_with("ORACLE_RESULT "))
        .context("cv2.dnn oracle did not emit ORACLE_RESULT")?;
    let fields: Vec<_> = line.split_whitespace().collect();
    let milliseconds = fields
        .get(1)
        .context("oracle result lacks median milliseconds")?
        .parse::<f64>()
        .context("parse oracle median milliseconds")?;
    let actual_digest = fields.get(2).context("oracle result lacks input digest")?;
    if actual_digest != &expected_digest {
        bail!(
            "cv2.dnn oracle consumed a different input tensor: wrote sha256 {expected_digest}, Python reported {actual_digest}"
        );
    }
    if !milliseconds.is_finite() || milliseconds < 0.0 {
        bail!("oracle median milliseconds is invalid: {milliseconds}");
    }
    Ok(milliseconds)
}

#[cfg(feature = "onnx")]
fn render_sections(sections: &[Section]) -> String {
    let mut rendered = String::new();
    for (index, section) in sections.iter().enumerate() {
        let _ = writeln!(rendered, "## {}. {}\n", index + 1, section.title);
        rendered.push_str(section.body.trim_end());
        rendered.push_str("\n\n");
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn summarise_drops_the_warmup_and_reports_median_min_max() {
        let durations = [
            Duration::from_millis(900),
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(3),
            Duration::from_millis(4),
        ];
        let summary = summarise(&durations, 1);
        assert_eq!(summary.kept, 4);
        assert_eq!(summary.median, Duration::from_micros(2500));
        assert_eq!(summary.min, Duration::from_millis(1));
        assert_eq!(summary.max, Duration::from_millis(4));
        assert_ne!(summary.median, summarise(&durations, 0).median);
    }

    #[test]
    fn summarise_refuses_fewer_than_three_kept_samples() {
        let durations = [
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(3),
        ];
        assert_eq!(
            try_summarise(&durations, 1),
            Err(BenchError::TooFewSamples { kept: 2 })
        );
        assert!(try_summarise(&durations, 0).is_ok());
    }

    #[test]
    fn the_variant_menu_includes_the_shipped_default_and_no_flush_control() {
        let variants = builtin_variants();
        let labels: BTreeSet<_> = variants
            .iter()
            .map(|variant| variant.label.as_str())
            .collect();
        assert!(labels.contains("pinned"));
        assert!(labels.contains("flush_denormals=off"));
        let pinned = variants
            .iter()
            .find(|variant| variant.label == "pinned")
            .expect("pinned control");
        assert_eq!(pinned.tuning, SessionTuning::default());
        assert!(pinned.tuning.flush_denormals);
        let no_flush = variants
            .iter()
            .find(|variant| variant.label == "flush_denormals=off")
            .expect("no-flush control");
        assert!(!no_flush.tuning.flush_denormals);
        assert_ne!(pinned.tuning, no_flush.tuning);
    }

    #[test]
    fn oracle_arms_render_skip_and_measured_reports() {
        let skipped = render_report("pinned", "cv2 is not installed");
        assert!(skipped.contains("SKIPPED"));
        assert!(skipped.contains("cv2"));
        assert!(!skipped.contains("vs cv2.dnn"));

        let rows = vec![(
            "pinned".into(),
            DurationSummary {
                kept: 3,
                median: Duration::from_millis(2),
                min: Duration::from_millis(1),
                max: Duration::from_millis(3),
            },
            "digest".into(),
            0,
            0.0,
        )];
        let measured = render_oracle_report(
            &OracleArm::Measured {
                milliseconds: 1.0,
                reps: 3,
                warmup: 1,
            },
            &rows,
        );
        assert!(measured.contains("ours vs cv2.dnn ratio: 2.000"));
        assert!(measured.contains("forward()"));
    }

    #[test]
    fn the_first_variant_is_marked_as_baseline_even_when_it_is_not_pinned() {
        let rows = vec![
            (
                "flush_denormals=on".into(),
                DurationSummary {
                    kept: 3,
                    median: Duration::from_millis(2),
                    min: Duration::from_millis(1),
                    max: Duration::from_millis(3),
                },
                "baseline-digest".into(),
                0,
                0.0,
            ),
            (
                "parallel_execution=on".into(),
                DurationSummary {
                    kept: 3,
                    median: Duration::from_millis(2),
                    min: Duration::from_millis(1),
                    max: Duration::from_millis(3),
                },
                "identical-digest".into(),
                0,
                0.0,
            ),
        ];
        let rendered = render_rows(&rows);
        assert!(rendered.contains(
            "| flush_denormals=on | 3 | 2.000 | 1.000 | 3.000 | `baseline-digest` | — | — |"
        ));
        assert!(rendered.contains("| parallel_execution=on | 3 | 2.000 | 1.000 | 3.000 | `identical-digest` | 0 | 0.000000 |"));
    }

    #[test]
    fn the_oracle_result_rejects_a_digest_for_a_different_input_tensor() {
        let output = b"ORACLE_RESULT 12.345 digest-of-a-different-tensor\n";
        let error = parse_oracle_result(output, "digest-of-the-tensor-we-wrote")
            .expect_err("a mismatched tensor digest must fail the oracle arm");
        assert!(error.to_string().contains("different input tensor"));
    }
}
