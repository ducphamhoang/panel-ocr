//! `cargo xtask ocr-device-compare` — GPU-3 G3-D: the OCR CUDA divergence measurement.
//!
//! Part 2 of the G3-D brief. Runs the SAME real manga-ocr encoder+decoder ONNX graphs on
//! the CPU execution provider and the CUDA execution provider (a real GPU with cuDNN on
//! `PATH`), then measures four channels of divergence:
//!
//!   1. encoder hidden states (CPU vs CUDA)
//!   2. decoder logits, **teacher-forced** (every CPU beam-search prefix replayed through
//!      CUDA with the CPU encoder — isolates the decoder's own divergence)
//!   3. end-to-end token-ID sequence and decoded string (CPU free run vs CUDA free run)
//!   4. CUDA self-nondeterminism (two independent CUDA free runs)
//!
//! The output is `docs/OCR_DEVICE_DIVERGENCE.md` — a committed, **non-gating** diagnostic.
//! No number in it is a pass/fail gate (§16.22 item 2's CPU-determinism carve-out applying
//! in the negative direction: CUDA is exempt from determinism, so none of this asserts).
//!
//! **Feature gating:** the producer needs the `cuda` feature (which pulls in `onnx` and
//! real CUDA registration through `pc-ocr/cuda`). The subcommand is registered in every
//! build — a non-`cuda` build refuses with a clear message naming the feature rather than
//! silently no-op'ing.

#[cfg(not(feature = "cuda"))]
use anyhow::{bail, Result};
use std::path::PathBuf;

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    /// Falls back to PANEL_OCR_MANGA_OCR_ENCODER if unset.
    #[arg(long)]
    pub(crate) encoder: Option<PathBuf>,
    /// Falls back to PANEL_OCR_MANGA_OCR_DECODER if unset.
    #[arg(long)]
    pub(crate) decoder: Option<PathBuf>,
    /// Falls back to a committed fixture: `demo_bubbles/handwritten_bubble_raw.png`.
    #[arg(long)]
    pub(crate) crop: Option<PathBuf>,
    #[arg(long, default_value = "docs/OCR_DEVICE_DIVERGENCE.md")]
    pub(crate) out: PathBuf,
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::Args;
    use crate::paths;
    use anyhow::{anyhow, bail, Context, Result};
    use pc_core::device::{resolve, Device, DeviceSupport};
    use pc_ocr::decode::{self, LogitsSource};
    use pc_ocr::device_divergence::{
        compare_tensors, first_divergence_index, min_top2_margin, TensorDivergence,
    };
    use pc_ocr::onnx::{EncoderOutput, MangaOcrSessions};
    use pc_ocr::{post, preprocess, vocab};
    use std::cell::RefCell;
    use std::fmt::Write as _;
    use std::path::{Path, PathBuf};

    /// The `ort` version pin from the workspace manifest (`Cargo.toml`), which is what
    /// the report's environment table means by "ort version pin".
    const ORT_VERSION_PIN: &str = "2.0.0-rc.12";

    /// A [`LogitsSource`] that records every `(prefix, logits)` pair it serves, as a side
    /// effect of running the real beam search it wraps — the teacher-forcing mechanism of
    /// the G3-D design note (isolating the decoder's own divergence from the encoder's).
    struct RecordingLogitsSource<'a> {
        inner: &'a dyn LogitsSource,
        recorded: RefCell<Vec<(Vec<u32>, Vec<f32>)>>,
    }

    impl<'a> RecordingLogitsSource<'a> {
        fn new(inner: &'a dyn LogitsSource) -> Self {
            Self {
                inner,
                recorded: RefCell::new(Vec::new()),
            }
        }

        fn recorded(&self) -> Vec<(Vec<u32>, Vec<f32>)> {
            self.recorded.borrow().clone()
        }
    }

    impl LogitsSource for RecordingLogitsSource<'_> {
        fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, pc_core::StageError> {
            let logits = self.inner.logits(prefix)?;
            self.recorded
                .borrow_mut()
                .push((prefix.to_vec(), logits.clone()));
            Ok(logits)
        }
    }

    /// Resolve a model path: explicit arg, else the named environment variable, else a
    /// hard error naming both.
    fn resolve_model_path(explicit: Option<&Path>, env_name: &str, flag: &str) -> Result<PathBuf> {
        if let Some(path) = explicit {
            return Ok(path.to_path_buf());
        }
        match std::env::var_os(env_name) {
            Some(path) => Ok(PathBuf::from(path)),
            None => bail!(
                "no model path given: pass --{flag} or set the {env_name} environment variable",
            ),
        }
    }

    fn sha256_of(path: &Path) -> String {
        match pc_models::sha256_hex(path) {
            Ok(hex) => hex,
            Err(error) => format!("(unavailable: {error})"),
        }
    }

    fn file_display(path: &Path) -> String {
        // `pc_testkit::paths::workspace_root()` canonicalizes, which on Windows produces
        // the verbatim `\\?\` prefix; strip it so a committed report carries readable
        // paths.
        let rendered = pc_testkit::paths::slash_separated(path);
        rendered
            .strip_prefix("//?/")
            .map(|rest| rest.to_owned())
            .unwrap_or(rendered)
    }

    fn format_delta(value: f32) -> String {
        if value.is_nan() {
            "NaN".to_owned()
        } else {
            format!("{value:.6}")
        }
    }

    /// One full `encode` + free-running `beam_search` on a session, returning the token-ID
    /// sequence and the post-processed decoded string (mirroring `manga.rs`'s `recognize`).
    struct FreeRun {
        ids: Vec<u32>,
        text: String,
    }

    fn free_run(sessions: &MangaOcrSessions, pixel_values: &[f32]) -> Result<FreeRun> {
        let encoder_output = sessions.encode(pixel_values)?;
        struct Source<'a> {
            sessions: &'a MangaOcrSessions,
            encoder_output: &'a EncoderOutput,
        }
        impl LogitsSource for Source<'_> {
            fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, pc_core::StageError> {
                self.sessions.decode_step(prefix, self.encoder_output)
            }
        }
        let source = Source {
            sessions,
            encoder_output: &encoder_output,
        };
        let ids = decode::beam_search(&source, &decode::MANGA_OCR_BEAM_CONFIG)?;
        let text = post::post_process(&vocab::Vocab::embedded().decode_skip_special(&ids));
        Ok(FreeRun { ids, text })
    }

    pub(crate) fn run(args: Args) -> Result<()> {
        // Resolve the model paths (arg → env → hard error naming both).
        let encoder = resolve_model_path(
            args.encoder.as_deref(),
            "PANEL_OCR_MANGA_OCR_ENCODER",
            "encoder",
        )?;
        let decoder = resolve_model_path(
            args.decoder.as_deref(),
            "PANEL_OCR_MANGA_OCR_DECODER",
            "decoder",
        )?;
        let crop = args.crop.unwrap_or_else(|| {
            pc_testkit::paths::upstream("demo_bubbles/handwritten_bubble_raw.png")
        });
        let out = args.out;

        // Load the crop and preprocess once; both sessions consume the same pixels.
        let image =
            image::open(&crop).with_context(|| format!("opening crop {}", file_display(&crop)))?;
        let pixel_values = preprocess::pixel_values(&image);
        println!(
            "crop: {} ({}x{})",
            file_display(&crop),
            image.width(),
            image.height()
        );
        println!("encoder: {}", file_display(&encoder));
        println!("decoder: {}", file_display(&decoder));
        println!("encoder sha256: {}", sha256_of(&encoder));
        println!("decoder sha256: {}", sha256_of(&decoder));

        // Step 2: two independent sessions. A CUDA build failure (e.g. cuDNN missing) is
        // propagated as the real error — never caught and reported as "no CUDA available".
        println!("\nbuilding CPU session…");
        let cpu = MangaOcrSessions::from_paths_for_device(&encoder, &decoder, Device::Cpu)
            .map_err(|error| anyhow!("building the CPU OCR session: {error}"))?;
        println!("building CUDA session…");
        let cuda = MangaOcrSessions::from_paths_for_device(&encoder, &decoder, Device::Cuda)
            .map_err(|error| anyhow!("building the CUDA OCR session: {error}"))?;
        println!("both sessions built\n");

        // Channel 1: encoder divergence. A shape mismatch is a hard error, not a divergence.
        let cpu_enc = cpu
            .encode(&pixel_values)
            .map_err(|error| anyhow!("CPU encoder run: {error}"))?;
        let cuda_enc = cuda
            .encode(&pixel_values)
            .map_err(|error| anyhow!("CUDA encoder run: {error}"))?;
        if cpu_enc.shape != cuda_enc.shape {
            bail!(
                "encoder output shape mismatch between CPU and CUDA: {:?} vs {:?} — the two \
                 builds disagree about the model itself",
                cpu_enc.shape,
                cuda_enc.shape
            );
        }
        let channel1 = compare_tensors(&cpu_enc.hidden_states, &cuda_enc.hidden_states)
            .map_err(|error| anyhow!("channel 1 (encoder): {error}"))?;
        println!(
            "channel 1 — encoder: {} values, {} bitwise-differing, max|Δ| {}, mean|Δ| {}",
            channel1.len,
            channel1.bitwise_differing_count,
            format_delta(channel1.max_abs_delta),
            format_delta(channel1.mean_abs_delta)
        );

        // Channel 2: decoder divergence, teacher-forced. The CPU decode runs for real; the
        // recording is a side effect. Every recorded (prefix, cpu_logits) pair is then
        // replayed through the CUDA decoder with the CPU encoder (`&cpu_enc`, NOT
        // `&cuda_enc`) — this isolates the decoder's own divergence per the brief's design
        // note.
        struct CpuLogitsSource<'a> {
            sessions: &'a MangaOcrSessions,
            encoder_output: &'a EncoderOutput,
        }
        impl LogitsSource for CpuLogitsSource<'_> {
            fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, pc_core::StageError> {
                self.sessions.decode_step(prefix, self.encoder_output)
            }
        }
        let cpu_source = CpuLogitsSource {
            sessions: &cpu,
            encoder_output: &cpu_enc,
        };
        let recording = RecordingLogitsSource::new(&cpu_source);
        let cpu_ids = decode::beam_search(&recording, &decode::MANGA_OCR_BEAM_CONFIG)
            .map_err(|error| anyhow!("CPU beam search: {error}"))?;
        let recorded = recording.recorded();
        println!(
            "channel 2 — teacher-forced: CPU beam search made {} decoder calls",
            recorded.len()
        );

        let mut row_divergences: Vec<TensorDivergence> = Vec::with_capacity(recorded.len());
        let mut cpu_logit_rows: Vec<Vec<f32>> = Vec::with_capacity(recorded.len());
        for (prefix, cpu_logits) in &recorded {
            let cuda_logits = cuda.decode_step(prefix, &cpu_enc).map_err(|error| {
                anyhow!(
                    "CUDA decoder step for prefix of length {}: {error}",
                    prefix.len()
                )
            })?;
            let divergence = compare_tensors(cpu_logits, &cuda_logits)
                .map_err(|error| anyhow!("channel 2 row: {error}"))?;
            row_divergences.push(divergence);
            cpu_logit_rows.push(cpu_logits.clone());
        }
        let row_bitwise_total: usize = row_divergences
            .iter()
            .map(|d| d.bitwise_differing_count)
            .sum();
        let row_values: usize = row_divergences.iter().map(|d| d.len).sum();
        let mismatched_rows = row_divergences
            .iter()
            .filter(|d| d.bitwise_differing_count > 0)
            .count();
        let row_maxes: Vec<f32> = row_divergences.iter().map(|d| d.max_abs_delta).collect();
        let per_row_max_of_maxes = nan_aware_max(&row_maxes);
        // The min top-2 margin is over the real CPU logit rows the decode actually saw —
        // how close beam search's decisions were to going the other way.
        let top2_margin = min_top2_margin(&cpu_logit_rows);
        println!(
            "channel 2 — {} rows, {} values, {} bitwise-differing, {} rows with any bitwise difference",
            recorded.len(),
            row_values,
            row_bitwise_total,
            mismatched_rows
        );
        println!(
            "channel 2 — per-row max-of-maxes {:.6}, min top-2 margin {:.6}",
            per_row_max_of_maxes, top2_margin
        );

        // Channel 3 uses the CPU token IDs from the channel-2 decode as the CPU free run.
        let cpu_text = post::post_process(&vocab::Vocab::embedded().decode_skip_special(&cpu_ids));
        let cpu_free_run = FreeRun {
            ids: cpu_ids,
            text: cpu_text,
        };

        // Channel 3: a fresh, independent CUDA free run.
        println!("\nchannel 3 — CUDA free-running decode (run A)…");
        let cuda_run_a = free_run(&cuda, &pixel_values)?;
        let channel3_first_divergence = first_divergence_index(&cpu_free_run.ids, &cuda_run_a.ids);
        let channel3_strings_equal = cpu_free_run.text == cuda_run_a.text;
        println!("channel 3 — CPU  text {:?}", cpu_free_run.text);
        println!("channel 3 — CUDA text {:?}", cuda_run_a.text);
        println!(
            "channel 3 — first token-ID divergence {:?}, decoded strings equal: {}",
            channel3_first_divergence, channel3_strings_equal
        );

        // Channel 4: a second, fully independent CUDA free run.
        println!("\nchannel 4 — CUDA free-running decode (run B, independent)…");
        let cuda_run_b = free_run(&cuda, &pixel_values)?;
        let channel4_first_divergence = first_divergence_index(&cuda_run_a.ids, &cuda_run_b.ids);
        let channel4_strings_equal = cuda_run_a.text == cuda_run_b.text;
        println!("channel 4 — CUDA text {:?}", cuda_run_b.text);
        println!(
            "channel 4 — first token-ID divergence {:?}, decoded strings equal: {}",
            channel4_first_divergence, channel4_strings_equal
        );

        // Write the report.
        let document = render_report(
            &encoder,
            &decoder,
            &crop,
            &channel1,
            &row_divergences,
            top2_margin,
            &cpu_free_run,
            &cuda_run_a,
            &cuda_run_b,
        )?;
        std::fs::write(&out, &document)
            .with_context(|| format!("writing {}", paths::display_relative(&out)))?;
        println!(
            "\nwrote {} ({} bytes)",
            paths::display_relative(&out),
            document.len()
        );
        Ok(())
    }

    fn nan_aware_max(values: &[f32]) -> f32 {
        let mut maximum = f32::NEG_INFINITY;
        for &value in values {
            if value.is_nan() {
                return f32::NAN;
            }
            maximum = maximum.max(value);
        }
        maximum
    }

    /// The report renderer: a pure function over the measured data, so its shape is
    /// testable without a GPU. Modeled on `docs/MODE_COMPARISON.md`'s shape.
    #[allow(clippy::too_many_arguments)]
    fn render_report(
        encoder: &Path,
        decoder: &Path,
        crop: &Path,
        channel1: &TensorDivergence,
        channel2_rows: &[TensorDivergence],
        channel2_min_top2_margin: f32,
        cpu: &FreeRun,
        cuda_a: &FreeRun,
        cuda_b: &FreeRun,
    ) -> Result<String> {
        let mut body = String::new();

        body.push_str(
            "# OCR device divergence (GPU-3 G3-D)\n\n\
             **Generated in full by `cargo xtask ocr-device-compare` — do not hand-edit.** Every \
             value here is measured at generation time. This is a **non-gating** diagnostic: no \
             number here is a pass/fail gate (§16.22 item 2's CPU-determinism carve-out applying in \
             the negative direction: CUDA is exempt from determinism, so none of this asserts). It \
             records what the SAME real manga-ocr encoder+decoder ONNX graphs produced on the CPU \
             execution provider vs the CUDA execution provider on one machine.\n\n",
        );

        body.push_str("## 1. Method and provenance\n\n");
        let _ = writeln!(
            body,
            "- **Crop:** {} (loaded with `image::open`; preprocessed by `pc_ocr::preprocess::pixel_values`).",
            file_display(crop)
        );
        let _ = writeln!(
            body,
            "- **Encoder:** {} — sha256 `{}`",
            file_display(encoder),
            sha256_of(encoder)
        );
        let _ = writeln!(
            body,
            "- **Decoder:** {} — sha256 `{}`",
            file_display(decoder),
            sha256_of(decoder)
        );
        let _ = writeln!(
            body,
            "- **Sessions:** two independent `MangaOcrSessions` via \
             `from_paths_for_device(.., Device::Cpu)` and `from_paths_for_device(.., Device::Cuda)`."
        );
        let _ = writeln!(
            body,
            "- **Channel 2 mechanism (teacher-forced):** the real CPU beam search runs and records \
             every `(prefix, logits)` pair it serves; each is replayed through the CUDA decoder with \
             the **CPU** encoder (`&cpu_enc`), isolating the decoder's own divergence from the \
             encoder's and from the beam search's feedback loop."
        );
        let _ = writeln!(
            body,
            "- **Channel 3/4 mechanism:** two fully independent free-running `encode` + `beam_search` \
             decodes (one CPU, two CUDA), compared by `first_divergence_index` and decoded-string equality."
        );

        body.push_str("\n## 2. Device\n\n");
        let _ = writeln!(
            body,
            "- **CPU policy:** {}\n- **CUDA policy:** {}",
            device_policy_report(Device::Cpu),
            device_policy_report(Device::Cuda)
        );

        body.push_str("\n## 3. Environment\n\n");
        let _ = writeln!(body, "| Property | Value |");
        let _ = writeln!(body, "|---|---|");
        let _ = writeln!(body, "| `ort` crate version pin | `{}` |", ORT_VERSION_PIN);
        let facts = environment_facts();
        let _ = writeln!(
            body,
            "| CUDA_PATH | {} |",
            facts.cuda_path.as_deref().unwrap_or("(not set)")
        );
        let _ = writeln!(
            body,
            "| CUDNN_PATH | {} |",
            facts.cudnn_path.as_deref().unwrap_or("(not set)")
        );
        let _ = writeln!(
            body,
            "| driver version | {} |",
            facts.driver_version.as_deref().unwrap_or("(not queryable)")
        );
        let _ = writeln!(
            body,
            "| CUDA version | {} |",
            facts.cuda_version.as_deref().unwrap_or("(not queryable)")
        );

        body.push_str("\n## 4. Channels\n\n");
        let _ = writeln!(
            body,
            "| Channel | What is compared | Values | Bitwise-differing | max abs Δ | mean abs Δ | per-row max-of-maxes | min top-2 margin |",
        );
        let _ = writeln!(body, "|---|---|---|---:|---:|---:|---:|---:|");

        let ch2_bitwise: usize = channel2_rows
            .iter()
            .map(|d| d.bitwise_differing_count)
            .sum();
        let ch2_values: usize = channel2_rows.iter().map(|d| d.len).sum();
        let ch2_maxes: Vec<f32> = channel2_rows.iter().map(|d| d.max_abs_delta).collect();
        let ch2_max = nan_aware_max(&ch2_maxes);
        let ch2_means: Vec<f32> = channel2_rows.iter().map(|d| d.mean_abs_delta).collect();
        let ch2_mean = if channel2_rows.is_empty() || ch2_means.iter().any(|value| value.is_nan()) {
            f32::NAN
        } else {
            ch2_means.iter().sum::<f32>() / channel2_rows.len() as f32
        };

        let _ = writeln!(
            body,
            "| 1 | encoder hidden states, CPU vs CUDA | {} | {} | {} | {} | — | — |",
            channel1.len,
            channel1.bitwise_differing_count,
            format_delta(channel1.max_abs_delta),
            format_delta(channel1.mean_abs_delta)
        );
        let _ = writeln!(
            body,
            "| 2 | decoder logits, teacher-forced (CPU encoder) | {} ({} rows) | {} | {} | {} | {} | {} |",
            ch2_values,
            channel2_rows.len(),
            ch2_bitwise,
            format_delta(ch2_max),
            format_delta(ch2_mean),
            format_delta(ch2_max),
            format_delta(channel2_min_top2_margin)
        );
        let _ = writeln!(
            body,
            "| 3 | end-to-end token IDs, CPU run vs CUDA run A | first divergence `{:?}` | strings equal: {} | — | — | — | — |",
            first_divergence_index(&cpu.ids, &cuda_a.ids),
            cpu.text == cuda_a.text
        );
        let _ = writeln!(
            body,
            "| 4 | end-to-end token IDs, CUDA run A vs CUDA run B | first divergence `{:?}` | strings equal: {} | — | — | — | — |",
            first_divergence_index(&cuda_a.ids, &cuda_b.ids),
            cuda_a.text == cuda_b.text
        );

        body.push_str("\n## 5. Decoded text\n\n");
        let _ = writeln!(body, "| Run | token IDs | decoded text |");
        let _ = writeln!(body, "|---|---|---|");
        let _ = writeln!(body, "| CPU | `{:?}` | `{}` |", cpu.ids, cpu.text);
        let _ = writeln!(body, "| CUDA A | `{:?}` | `{}` |", cuda_a.ids, cuda_a.text);
        let _ = writeln!(body, "| CUDA B | `{:?}` | `{}` |", cuda_b.ids, cuda_b.text);

        body.push_str("\n## 6. Verdict\n\n");
        body.push_str(
            "**Non-gating.** No number above is a pass/fail gate. CUDA is exempt from determinism, so \
             none of this asserts. The numbers record what actually happened on this machine at \
             generation time, for the ratification step (G3-E) to cite.\n\n",
        );

        Ok(body)
    }

    fn device_policy_report(device: Device) -> String {
        match resolve(device, DeviceSupport::compiled()) {
            Ok(policy) => policy.report(),
            Err(refusal) => format!("(could not resolve: {})", refusal.message()),
        }
    }

    /// Best-effort environment facts for the report. Never blocks the report: every probe
    /// is `Option`al and failure is swallowed into `None`.
    struct EnvironmentFacts {
        cuda_path: Option<String>,
        cudnn_path: Option<String>,
        driver_version: Option<String>,
        cuda_version: Option<String>,
    }

    fn environment_facts() -> EnvironmentFacts {
        EnvironmentFacts {
            cuda_path: std::env::var_os("CUDA_PATH").map(|v| v.to_string_lossy().into_owned()),
            cudnn_path: cudnn_path_disclosure(),
            driver_version: query_nvidia_smi(),
            cuda_version: query_cuda_version(),
        }
    }

    /// cuDNN disclosure: `CUDNN_PATH` if set, else the first `PATH` entry whose basename
    /// contains `cudnn` — cuDNN is commonly installed by adding its `bin` dir to `PATH`
    /// without setting the variable (as this machine does).
    fn cudnn_path_disclosure() -> Option<String> {
        if let Some(value) = std::env::var_os("CUDNN_PATH") {
            return Some(value.to_string_lossy().into_owned());
        }
        let path = std::env::var_os("PATH")?;
        std::env::split_paths(&path)
            .map(|dir| dir.to_string_lossy().into_owned())
            .find(|dir| dir.to_ascii_lowercase().contains("cudnn"))
    }

    fn query_nvidia_smi() -> Option<String> {
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=driver_version", "--format=csv,noheader"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|line| line.trim().to_owned())
            .filter(|line| !line.is_empty())
    }

    fn query_cuda_version() -> Option<String> {
        std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|text| {
                text.lines()
                    .find(|line| line.contains("release"))
                    .map(|line| line.trim().to_owned())
            })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn sample_free_run(ids: &[u32], text: &str) -> FreeRun {
            FreeRun {
                ids: ids.to_vec(),
                text: text.to_owned(),
            }
        }

        #[test]
        fn resolve_model_path_uses_the_explicit_arg_and_errors_cleanly() {
            assert!(resolve_model_path(
                Some(Path::new("/x")),
                "PANEL_OCR_MANGA_OCR_ENCODER",
                "encoder"
            )
            .is_ok());
            let error = resolve_model_path(None, "PANEL_OCR_MANGA_OCR_ENCODER", "encoder")
                .expect_err("no arg and no env var must fail");
            assert!(error.to_string().contains("PANEL_OCR_MANGA_OCR_ENCODER"));
            assert!(error.to_string().contains("--encoder"));
        }

        #[test]
        fn recording_logits_source_records_every_call() {
            struct Stub;
            impl LogitsSource for Stub {
                fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, pc_core::StageError> {
                    Ok(prefix.iter().map(|&id| id as f32).collect())
                }
            }
            let stub = Stub;
            let recording = RecordingLogitsSource::new(&stub);
            recording.logits(&[1, 2]).expect("stub");
            recording.logits(&[1, 2, 3]).expect("stub");
            let recorded = recording.recorded();
            assert_eq!(recorded.len(), 2);
            assert_eq!(recorded[0].0, vec![1, 2]);
            assert_eq!(recorded[0].1, vec![1.0, 2.0]);
            assert_eq!(recorded[1].0, vec![1, 2, 3]);
        }

        #[test]
        fn render_report_includes_the_decoded_text_table_and_non_gating_verdict() {
            let cpu = sample_free_run(&[2, 15, 20], "ＡＢ");
            let cuda_a = sample_free_run(&[2, 15, 21], "ＡＣ");
            let cuda_b = sample_free_run(&[2, 15, 21], "ＡＣ");
            let channel1 = TensorDivergence {
                len: 3,
                bitwise_differing_count: 0,
                max_abs_delta: 0.0,
                mean_abs_delta: 0.0,
            };
            let rows = vec![TensorDivergence {
                len: 2,
                bitwise_differing_count: 1,
                max_abs_delta: 0.5,
                mean_abs_delta: 0.25,
            }];
            let document = render_report(
                Path::new("encoder.onnx"),
                Path::new("decoder.onnx"),
                Path::new("crop.png"),
                &channel1,
                &rows,
                0.1,
                &cpu,
                &cuda_a,
                &cuda_b,
            )
            .expect("render");
            assert!(document.contains("## 4. Channels"));
            assert!(document.contains("## 5. Decoded text"));
            assert!(document.contains("| CPU | `[2, 15, 20]` | `ＡＢ` |"));
            assert!(document.contains(
                "| 3 | end-to-end token IDs, CPU run vs CUDA run A | first divergence `Some(2)` | strings equal: false |"
            ));
            assert!(document.contains(
                "| 4 | end-to-end token IDs, CUDA run A vs CUDA run B | first divergence `None` | strings equal: true |"
            ));
            assert!(document.contains("**Non-gating.**"));
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) use cuda::run;

#[cfg(not(feature = "cuda"))]
pub(crate) fn run(_args: Args) -> Result<()> {
    bail!(
        "`ocr-device-compare` requires the `cuda` feature: rebuild with \
         `cargo build --features xtask/cuda -p xtask` (this build has no CUDA execution \
         provider linked in, so there is nothing to measure)"
    )
}
