//! `cargo xtask lama-device-compare` — GPU-4 G4-C: the real LaMa CUDA divergence
//! measurement.
//!
//! Runs the SAME real cached LaMa `lama-manga.onnx` graph on the CPU execution provider
//! and the CUDA execution provider (a real GPU with cuDNN on `PATH`), across three
//! independent full `inpaint_page` runs sharing one `PageInput`, then measures four
//! channels of divergence:
//!
//!   1. per-tile decoded RGB and raw f32 output, CPU vs CUDA-A
//!   2. per-tile decoded RGB and raw f32 output, CUDA-A vs CUDA-B (self-nondeterminism)
//!   3. page-level `inpainting`/`clean_inpaint` artifacts, CPU vs CUDA-A (whole artifact
//!      and write-region-restricted)
//!   4. page-level artifacts, CUDA-A vs CUDA-B
//!
//! The whole design rests on there being **no cross-tile feedback loop** in
//! `inpaint_page` — `cover`/`owners` are computed once, before the inference loop, from
//! pure CPU arithmetic. Three independent runs sharing one `PageInput` therefore receive
//! byte-identical tile crops by construction. That premise is verified at runtime with a
//! byte-for-byte tile-identity assertion (and a hard error if it fails) rather than
//! assumed.
//!
//! **Feature gating:** the producer needs the `cuda` feature (which pulls in `onnx` and
//! real CUDA registration through `pc-inpaint/cuda`). The subcommand is registered in
//! every build — a non-`cuda` build refuses with a clear message naming the feature,
//! mirroring `ocr-device-compare`'s exact shape.
//!
//! **Non-gating:** the output is `docs/LAMA_DEVICE_DIVERGENCE.md` — a committed,
//! non-gating diagnostic. No number in it is a pass/fail gate (§16.22 item 2's
//! CPU-determinism carve-out applying in the negative direction: CUDA is exempt from
//! determinism, so none of this asserts).

#[cfg(not(feature = "cuda"))]
use anyhow::{bail, Result};
use std::path::PathBuf;

#[derive(Debug, clap::Args)]
pub(crate) struct Args {
    /// Falls back to PANEL_OCR_LAMA_MODEL, then the managed cache path.
    #[arg(long)]
    pub(crate) model: Option<PathBuf>,
    #[arg(long, default_value = "docs/LAMA_DEVICE_DIVERGENCE.md")]
    pub(crate) out: PathBuf,
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::Args;
    use crate::paths;
    use anyhow::{anyhow, bail, Context, Result};
    use image::{GrayImage, RgbaImage};
    use pc_core::device::{resolve, Device, DeviceSupport};
    use pc_core::ImageHandle;
    use pc_inpaint::device_divergence::ByteDivergence;
    use pc_inpaint::onnx::OnnxInpainter;
    use pc_inpaint::Inpainter;
    use pc_ocr::device_divergence::{compare_tensors, TensorDivergence};
    use std::fmt::Write as _;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    /// The `ort` version pin from the workspace manifest (`Cargo.toml`), which is what
    /// the report's environment table means by "ort version pin".
    const ORT_VERSION_PIN: &str = "2.0.0-rc.12";

    /// The shipped default `InpainterConfig` — the exact radii are disclosed verbatim
    /// in the report (`InpainterConfig::default()`: `min_inpainting_radius: 7,
    /// max_inpainting_radius: 20, inpainting_isolation_radius: 5,
    /// inpainting_fade_radius: 4`).
    fn default_inpainter_config() -> pc_config::InpainterConfig {
        pc_config::InpainterConfig::default()
    }

    /// The forced multi-tile source's `InpainterConfig`: `min_inpainting_radius =
    /// max_inpainting_radius = 300` (both equal, so `padded_region`'s growth clamp is
    /// exact — see `crates/pc-inpaint/src/growth.rs`), with the isolation and fade
    /// radii left at their shipped defaults. This is a real, hard-coded producer
    /// constant, not a CLI flag — a later run cannot silently change the disclosed
    /// method.
    fn source_b_inpainter_config() -> pc_config::InpainterConfig {
        pc_config::InpainterConfig {
            min_inpainting_radius: 300,
            max_inpainting_radius: 300,
            ..pc_config::InpainterConfig::default()
        }
    }

    /// The six demo_bubbles pages that actually inpaint under the shipped default
    /// config (per `docs/MODE_COMPARISON.md` §3; `handwritten` has 0 eligible regions
    /// and is skipped). Each is real content, real detection, real masking, single-tile
    /// (every page is under 512px on both axes, verified).
    const SOURCE_A_PAGES: &[&str] = &["black", "darkrays", "nightmare", "ray", "spikey", "square"];

    /// The committed detector fixture stem for the forced multi-tile source.
    const REPLAY_STEM: &str = "ja_Pepper-and-Carrot_by-David-Revoy_E01P01";

    /// One tile-level record: the tile's RGB bytes, the mask's bits, the raw pre-decode
    /// f32 output (shape + values), and the decoded RGB the trait caller actually
    /// received.
    #[derive(Debug, Clone)]
    struct TileRecord {
        tile_rgb: Vec<u8>,
        mask_bits: Vec<u8>,
        raw_shape: Vec<i64>,
        raw_values: Vec<f32>,
        decoded_rgb: Vec<u8>,
    }

    /// Wraps a real `OnnxInpainter`, forwarding every `inpaint_tile` call through
    /// `run_raw` (so the raw pre-decode f32 output is visible) and recording, per call:
    /// the tile's RGB bytes, the mask's bits, and the raw (shape, values) pair. The
    /// decoded `RgbImage` returned to the trait caller is unchanged from what
    /// `inpaint_tile` would normally produce (via `decode_output_tile`) — this
    /// decorator changes nothing about `inpaint_page`'s real behavior, it only observes
    /// it.
    ///
    /// The recorded buffer is a `Mutex`, not a `RefCell`: [`Inpainter`] requires
    /// `Send + Sync`, and the codebase's own interior-mutability mechanism for a
    /// shareable inpainter is the `Mutex` (see `OnnxInpainter`'s `session` field), so
    /// the same mechanism and the same poison recovery are used here.
    struct RecordingInpainter<'a> {
        inner: &'a OnnxInpainter,
        recorded: Mutex<Vec<TileRecord>>,
    }

    impl<'a> RecordingInpainter<'a> {
        fn new(inner: &'a OnnxInpainter) -> Self {
            Self {
                inner,
                recorded: Mutex::new(Vec::new()),
            }
        }

        fn recorded(&self) -> Vec<TileRecord> {
            self.recorded
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl Inpainter for RecordingInpainter<'_> {
        fn inpaint_tile(
            &self,
            tile: &image::RgbImage,
            mask: &pc_imageops::BinaryMask,
        ) -> Result<image::RgbImage, pc_core::StageError> {
            let (shape, values) = self.inner.run_raw(tile, mask)?;
            let decoded = pc_inpaint::onnx::decode_output_tile(&shape, &values)?;
            self.recorded
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(TileRecord {
                    tile_rgb: tile.as_raw().clone(),
                    mask_bits: mask.as_bits().to_vec(),
                    raw_shape: shape,
                    raw_values: values,
                    decoded_rgb: decoded.as_raw().clone(),
                });
            Ok(decoded)
        }
    }

    /// The three independent runs' per-tile records and page outputs for one `PageInput`.
    struct TripleRun {
        tiles: Vec<Vec<TileRecord>>, // [cpu, cuda_a, cuda_b]
        pages: Vec<pc_inpaint::PageOutput>,
    }

    /// Resolve a model path: explicit arg, else `PANEL_OCR_LAMA_MODEL`, else the managed
    /// cache path (mirroring `ocr_device_compare.rs`'s `resolve_model_path` pattern,
    /// singular model this time since LaMa has one artifact not two).
    fn resolve_model_path(explicit: Option<&Path>) -> Result<PathBuf> {
        if let Some(path) = explicit {
            return Ok(path.to_path_buf());
        }
        match std::env::var_os("PANEL_OCR_LAMA_MODEL") {
            Some(path) => Ok(PathBuf::from(path)),
            None => {
                let cache_root = pc_cli::paths::default_cache_dir();
                let cached = pc_cli::paths::models_dir(&cache_root)
                    .join(pc_models::LAMA_MANGA_INPAINTER.file_name);
                if cached.is_file() {
                    Ok(cached)
                } else {
                    bail!(
                        "no model path given: pass --model, set the PANEL_OCR_LAMA_MODEL \
                         environment variable, or run `panel-ocr models download` to fetch \
                         the managed cache copy (expected at {})",
                        cached.display()
                    )
                }
            }
        }
    }

    /// Resolve the detector model for Source A (real detection): `PANEL_OCR_ONNX_MODEL`,
    /// else the managed cache path — the same arg → env → cache shape as the LaMa model
    /// resolution, mirroring `mode_bench`'s detector fallback. The `cuda` feature pulls
    /// in `pc-detect/onnx` transitively (via `pc-cli/cuda` → `pc-detect/cuda`), so the
    /// real `TextDetector` session is available in this build even though xtask's own
    /// `onnx` flag is not set.
    fn resolve_detector_model() -> Result<PathBuf> {
        if let Some(path) = std::env::var_os("PANEL_OCR_ONNX_MODEL") {
            return Ok(PathBuf::from(path));
        }
        let cache_root = pc_cli::paths::default_cache_dir();
        let cached =
            pc_cli::paths::models_dir(&cache_root).join(pc_models::COMIC_TEXT_DETECTOR.file_name);
        if cached.is_file() {
            Ok(cached)
        } else {
            bail!(
                "no detector model path given: set the PANEL_OCR_ONNX_MODEL environment \
                 variable or run `panel-ocr models download` to fetch the managed cache \
                 copy (expected at {})",
                cached.display()
            )
        }
    }

    fn sha256_of(path: &Path) -> String {
        match pc_models::sha256_hex(path) {
            Ok(hex) => hex,
            Err(error) => format!("(unavailable: {error})"),
        }
    }

    fn file_display(path: &Path) -> String {
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

    /// The real detect → preprocess → mask pipeline for one source image, mirroring
    /// `xtask/src/mode_bench.rs`'s `detect_page`/`measure_cell_inner` functions exactly.
    fn detect_preprocess_mask(
        image: &Path,
        detector: &dyn pc_detect::TextDetector,
    ) -> Result<(pc_core::MaskData, pc_core::PageDataRaw, usize)> {
        let detect_output = pc_detect::run(
            pc_detect::DetectInput {
                schema_version: pc_core::SCHEMA_VERSION,
                source: ImageHandle::from_path(image),
                original_path: image.to_path_buf(),
                target_height_lower: 1000,
                target_height_upper: 4000,
                base_image_dest: None,
                raw_mask_dest: None,
                min_mask_coverage: pc_detect::DEFAULT_MIN_MASK_COVERAGE,
                config: pc_config::TextDetectorConfig::default(),
            },
            detector,
        )
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| "running the detector")?;
        let detected_boxes = detect_output.analytics.blocks_detected;
        // The page's `raw_mask` handle is what `inpaint_page` needs (mirroring
        // `pc-pipeline`'s `single.rs:436-447`), so it is cloned out before the page moves
        // into preprocessing.
        let detect_page = detect_output.page.clone();

        let preprocess_output = pc_preprocess::run(
            pc_preprocess::PreprocessInput {
                schema_version: pc_core::SCHEMA_VERSION,
                page: detect_output.page,
                config: pc_config::PreprocessorConfig::default(),
                performing_ocr: false,
            },
            None,
        )
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| "running preprocessing")?;

        let mask_output = pc_mask::run(pc_mask::MaskInput {
            schema_version: pc_core::SCHEMA_VERSION,
            page: preprocess_output.page,
            original_image: ImageHandle::from_path(image),
            config: pc_config::MaskerConfig::default(),
            extract_text: false,
            debug_outputs: false,
            dests: pc_mask::MaskDests::default(),
        })
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| "running masking")?;

        Ok((mask_output.mask_data, detect_page, detected_boxes))
    }

    /// The decoded inputs every `inpaint_page` run shares: the full-resolution original,
    /// the detect stage's raw mask, and the mask stage's combined mask — each owned, so
    /// a fresh `PageInput` borrowing them can be built once per run (three runs share one
    /// `PageInput`'s data by construction, and `PageInput` itself borrows, so it cannot
    /// be reused by value across the three calls).
    struct SharedInputs {
        original: image::RgbImage,
        raw_mask: GrayImage,
        combined_mask: RgbaImage,
        mask_data: pc_core::MaskData,
    }

    /// Build the shared decoded inputs from a real `MaskData`/`regions` (via
    /// `pc_detect::run` → `pc_preprocess::run` → `pc_mask::run`, mirroring
    /// `xtask/src/mode_bench.rs`'s `detect_page`/`measure_cell_inner`). The `original`
    /// is the full-resolution user image (mirroring `pc-pipeline`'s `single.rs:123`),
    /// the `raw_mask` is the detect stage's own page handle (mirroring
    /// `single.rs:436-447`).
    fn build_shared_inputs(
        image: &Path,
        page: &pc_core::PageDataRaw,
        mask_data: &pc_core::MaskData,
    ) -> Result<SharedInputs> {
        let original = ImageHandle::from_path(image)
            .load()
            .map_err(|error| anyhow!(error.to_string()))?
            .to_rgb8();
        let raw_mask = page
            .raw_mask
            .load()
            .map_err(|error| anyhow!(error.to_string()))?
            .to_luma8();
        let combined_mask = mask_data
            .combined_mask
            .load()
            .map_err(|error| anyhow!(error.to_string()))?
            .to_rgba8();
        Ok(SharedInputs {
            original,
            raw_mask,
            combined_mask,
            mask_data: mask_data.clone(),
        })
    }

    fn page_input<'a>(
        shared: &'a SharedInputs,
        min_mask_thickness: u32,
        config: &'a pc_config::InpainterConfig,
    ) -> pc_inpaint::PageInput<'a> {
        pc_inpaint::PageInput {
            original: &shared.original,
            raw_mask: &shared.raw_mask,
            combined_mask: &shared.combined_mask,
            noise_mask: None,
            regions: &shared.mask_data.regions,
            min_mask_thickness,
            config,
        }
    }

    /// The anti-vacuity assertion this design's whole justification rests on: after all
    /// three runs, assert that every recorded `(tile_rgb, mask_bits)` pair at index `i`
    /// is byte-identical across all three recordings, for every `i`.
    ///
    /// If this assertion ever fails, abort with a loud error rather than reporting
    /// divergence numbers — it would mean the "no cross-tile feedback" premise this
    /// design depends on is false on this input, and the measurement below would be
    /// comparing different inputs, not the model's own divergence.
    fn assert_tile_identity(triple: &TripleRun, page_label: &str) -> Result<()> {
        let [cpu, cuda_a, cuda_b] = &triple.tiles[..] else {
            bail!("internal: exactly three recordings expected");
        };
        let counts = [cpu.len(), cuda_a.len(), cuda_b.len()];
        if counts[0] != counts[1] || counts[0] != counts[2] {
            bail!(
                "tile-identity assertion FAILED for {page_label}: the three runs recorded \
                 different tile counts ({}, {}, {}) — the \"no cross-tile feedback\" premise \
                 this design depends on is false on this input, so the divergence numbers \
                 below would be comparing different inputs, not the model's own divergence",
                counts[0],
                counts[1],
                counts[2]
            );
        }
        for (index, ((cpu_record, cuda_a_record), cuda_b_record)) in
            cpu.iter().zip(cuda_a).zip(cuda_b).enumerate()
        {
            if cpu_record.tile_rgb != cuda_a_record.tile_rgb
                || cpu_record.tile_rgb != cuda_b_record.tile_rgb
                || cpu_record.mask_bits != cuda_a_record.mask_bits
                || cpu_record.mask_bits != cuda_b_record.mask_bits
                || cpu_record.raw_shape != cuda_a_record.raw_shape
                || cpu_record.raw_shape != cuda_b_record.raw_shape
            {
                bail!(
                    "tile-identity assertion FAILED for {page_label} at tile index {index}: \
                     the (tile_rgb, mask_bits, raw_shape) triple differs across the three \
                     independent runs — the \"no cross-tile feedback\" premise this design \
                     depends on is false on this input, so the divergence numbers below would \
                     be comparing different inputs, not the model's own divergence"
                );
            }
        }
        Ok(())
    }

    /// One full page measurement: detect → preprocess → mask → three independent
    /// `inpaint_page` runs (CPU, CUDA-A, CUDA-B), with the tile-identity assertion and
    /// the page-level `tiles_inferred` guard applied.
    fn measure_page(
        image: &Path,
        detector: &dyn pc_detect::TextDetector,
        model: &Path,
        cuda_policy: &pc_core::device::DevicePolicy,
        config: &pc_config::InpainterConfig,
        label: &str,
    ) -> Result<TripleRun> {
        let (mask_data, page, detected_boxes) = detect_preprocess_mask(image, detector)?;
        let min_mask_thickness = pc_config::MaskerConfig::default().min_mask_thickness;
        let shared = build_shared_inputs(image, &page, &mask_data)?;

        println!(
            "{label}: {}x{}, {detected_boxes} detected boxes, {} regions",
            page.image_size.0,
            page.image_size.1,
            mask_data.regions.len()
        );

        let cpu_inpainter =
            OnnxInpainter::from_path_with_policy(model, &pc_core::device::DevicePolicy::cpu())
                .map_err(|error| anyhow!("building the CPU inpainter: {error}"))?;
        let cuda_a_inpainter = OnnxInpainter::from_path_with_policy(model, cuda_policy)
            .map_err(|error| anyhow!("building the CUDA-A inpainter: {error}"))?;
        let cuda_b_inpainter = OnnxInpainter::from_path_with_policy(model, cuda_policy)
            .map_err(|error| anyhow!("building the CUDA-B inpainter: {error}"))?;

        let cpu_recording = RecordingInpainter::new(&cpu_inpainter);
        let cuda_a_recording = RecordingInpainter::new(&cuda_a_inpainter);
        let cuda_b_recording = RecordingInpainter::new(&cuda_b_inpainter);

        let cpu_page = pc_inpaint::inpaint_page(
            page_input(&shared, min_mask_thickness, config),
            &cpu_recording,
        )
        .map_err(|error| anyhow!("CPU inpaint_page run: {error}"))?;
        let cuda_a_page = pc_inpaint::inpaint_page(
            page_input(&shared, min_mask_thickness, config),
            &cuda_a_recording,
        )
        .map_err(|error| anyhow!("CUDA-A inpaint_page run: {error}"))?;
        let cuda_b_page = pc_inpaint::inpaint_page(
            page_input(&shared, min_mask_thickness, config),
            &cuda_b_recording,
        )
        .map_err(|error| anyhow!("CUDA-B inpaint_page run: {error}"))?;

        if cpu_page.tiles_inferred == 0
            || cuda_a_page.tiles_inferred == 0
            || cuda_b_page.tiles_inferred == 0
        {
            bail!(
                "{label} should have inpainted but a run recorded tiles_inferred == 0 \
                 (cpu {}, cuda_a {}, cuda_b {})",
                cpu_page.tiles_inferred,
                cuda_a_page.tiles_inferred,
                cuda_b_page.tiles_inferred
            );
        }

        let triple = TripleRun {
            tiles: vec![
                cpu_recording.recorded(),
                cuda_a_recording.recorded(),
                cuda_b_recording.recorded(),
            ],
            pages: vec![cpu_page, cuda_a_page, cuda_b_page],
        };
        assert_tile_identity(&triple, label)?;
        Ok(triple)
    }

    fn byte_divergence(left: &[u8], right: &[u8]) -> Result<ByteDivergence> {
        pc_inpaint::device_divergence::compare_bytes(left, right).map_err(anyhow::Error::msg)
    }

    fn byte_divergence_where(
        left: &[u8],
        right: &[u8],
        selector: &[bool],
    ) -> Result<ByteDivergence> {
        pc_inpaint::device_divergence::compare_bytes_where(left, right, selector)
            .map_err(anyhow::Error::msg)
    }

    fn merge_bytes(mut acc: ByteDivergence, next: ByteDivergence) -> ByteDivergence {
        acc.len += next.len;
        acc.differing_count += next.differing_count;
        acc.max_abs_delta = acc.max_abs_delta.max(next.max_abs_delta);
        acc
    }

    fn merge_tensors(mut acc: TensorDivergence, next: TensorDivergence) -> TensorDivergence {
        let acc_len = acc.len;
        let next_len = next.len;
        let total = acc_len + next_len;
        let acc_sum = acc.mean_abs_delta * acc_len as f32;
        let next_sum = next.mean_abs_delta * next_len as f32;
        acc.len = total;
        acc.bitwise_differing_count += next.bitwise_differing_count;
        acc.max_abs_delta = acc.max_abs_delta.max(next.max_abs_delta);
        acc.mean_abs_delta = if total == 0 {
            0.0
        } else {
            (acc_sum + next_sum) / total as f32
        };
        acc
    }

    /// A selector over a page artifact's RGBA bytes: `true` exactly where the artifact's
    /// own alpha is non-zero — i.e. the `final_mask > 0` write region, since
    /// `inpainting` is `attach_alpha(inpainted, final_mask)` (confirmed in
    /// `crates/pc-inpaint/src/lib.rs`'s `attach_alpha`). One entry per **byte** (the
    /// artifact is RGBA, so each pixel contributes four), matching
    /// `compare_bytes_where`'s buffer-length selector contract.
    fn alpha_selector(image: &RgbaImage) -> Vec<bool> {
        image
            .as_raw()
            .chunks_exact(4)
            .flat_map(|pixel| {
                let selected = pixel[3] > 0;
                [selected; 4]
            })
            .collect()
    }

    /// Per-page channel rows for Source A (single-tile demo_bubbles pages).
    #[derive(Debug, Clone)]
    struct PageRow {
        page: String,
        tiles: usize,
        tiles_inferred: usize,
        c1_decoded: ByteDivergence,
        c1_raw: TensorDivergence,
        c2_decoded: ByteDivergence,
        c2_raw: TensorDivergence,
        c3_inpainting: ByteDivergence,
        c3_inpainting_where: ByteDivergence,
        c3_clean: ByteDivergence,
        c3_clean_where: ByteDivergence,
        c4_inpainting: ByteDivergence,
        c4_clean: ByteDivergence,
        min_margin: f32,
    }

    /// Per-tile rows for Source B (the forced multi-tile page).
    #[derive(Debug, Clone)]
    struct TileRow {
        index: usize,
        c1_decoded: ByteDivergence,
        c1_raw: TensorDivergence,
        c2_decoded: ByteDivergence,
        c2_raw: TensorDivergence,
    }

    /// Compare the two sides' tiles and page artifacts for one triple run, producing a
    /// `PageRow` and (for the multi-tile source) per-tile rows.
    fn measure_page_row(page: &str, triple: &TripleRun) -> Result<PageRow> {
        let [cpu, cuda_a, cuda_b] = &triple.tiles[..] else {
            bail!("internal: exactly three recordings expected");
        };

        let mut c1_decoded = ByteDivergence {
            len: 0,
            differing_count: 0,
            max_abs_delta: 0,
        };
        let mut c1_raw = TensorDivergence {
            len: 0,
            bitwise_differing_count: 0,
            max_abs_delta: 0.0,
            mean_abs_delta: 0.0,
        };
        let mut c2_decoded = ByteDivergence {
            len: 0,
            differing_count: 0,
            max_abs_delta: 0,
        };
        let mut c2_raw = TensorDivergence {
            len: 0,
            bitwise_differing_count: 0,
            max_abs_delta: 0.0,
            mean_abs_delta: 0.0,
        };
        for ((cpu_record, cuda_a_record), cuda_b_record) in cpu.iter().zip(cuda_a).zip(cuda_b) {
            c1_decoded = merge_bytes(
                c1_decoded,
                byte_divergence(&cpu_record.decoded_rgb, &cuda_a_record.decoded_rgb)?,
            );
            c1_raw = merge_tensors(
                c1_raw,
                compare_tensors(&cpu_record.raw_values, &cuda_a_record.raw_values)
                    .map_err(anyhow::Error::msg)?,
            );
            c2_decoded = merge_bytes(
                c2_decoded,
                byte_divergence(&cuda_a_record.decoded_rgb, &cuda_b_record.decoded_rgb)?,
            );
            c2_raw = merge_tensors(
                c2_raw,
                compare_tensors(&cuda_a_record.raw_values, &cuda_b_record.raw_values)
                    .map_err(anyhow::Error::msg)?,
            );
        }

        let cpu_page = &triple.pages[0];
        let cuda_a_page = &triple.pages[1];
        let cuda_b_page = &triple.pages[2];

        let c3_inpainting = byte_divergence(
            cpu_page.inpainting.as_raw(),
            cuda_a_page.inpainting.as_raw(),
        )?;
        let selector = alpha_selector(&cpu_page.inpainting);
        let c3_inpainting_where = byte_divergence_where(
            cpu_page.inpainting.as_raw(),
            cuda_a_page.inpainting.as_raw(),
            &selector,
        )?;
        let c3_clean = byte_divergence(
            cpu_page.clean_inpaint.as_raw(),
            cuda_a_page.clean_inpaint.as_raw(),
        )?;
        let c3_clean_where = byte_divergence_where(
            cpu_page.clean_inpaint.as_raw(),
            cuda_a_page.clean_inpaint.as_raw(),
            &selector,
        )?;

        let c4_inpainting = byte_divergence(
            cuda_a_page.inpainting.as_raw(),
            cuda_b_page.inpainting.as_raw(),
        )?;
        let c4_clean = byte_divergence(
            cuda_a_page.clean_inpaint.as_raw(),
            cuda_b_page.clean_inpaint.as_raw(),
        )?;

        // The min quantization margin characterizes the CPU run's own raw values — how
        // close the CPU decode's byte rounding was to flipping. It is a CPU-only
        // characterization, not a CPU-vs-CUDA comparison (disclosed up front in the
        // report's method section).
        let min_margin = pc_inpaint::device_divergence::min_quantization_margin(
            &cpu.iter()
                .flat_map(|record| record.raw_values.iter().copied())
                .collect::<Vec<_>>(),
        );

        Ok(PageRow {
            page: page.to_owned(),
            tiles: cpu.len(),
            tiles_inferred: cpu_page.tiles_inferred,
            c1_decoded,
            c1_raw,
            c2_decoded,
            c2_raw,
            c3_inpainting,
            c3_inpainting_where,
            c3_clean,
            c3_clean_where,
            c4_inpainting,
            c4_clean,
            min_margin,
        })
    }

    /// Per-tile channel rows for the forced multi-tile source (Source B).
    fn measure_tile_rows(triple: &TripleRun) -> Result<Vec<TileRow>> {
        let [cpu, cuda_a, cuda_b] = &triple.tiles[..] else {
            bail!("internal: exactly three recordings expected");
        };
        cpu.iter()
            .zip(cuda_a)
            .zip(cuda_b)
            .enumerate()
            .map(|(index, ((cpu_record, cuda_a_record), cuda_b_record))| {
                let c1_decoded =
                    byte_divergence(&cpu_record.decoded_rgb, &cuda_a_record.decoded_rgb)?;
                let c1_raw = compare_tensors(&cpu_record.raw_values, &cuda_a_record.raw_values)
                    .map_err(anyhow::Error::msg)?;
                let c2_decoded =
                    byte_divergence(&cuda_a_record.decoded_rgb, &cuda_b_record.decoded_rgb)?;
                let c2_raw = compare_tensors(&cuda_a_record.raw_values, &cuda_b_record.raw_values)
                    .map_err(anyhow::Error::msg)?;
                Ok(TileRow {
                    index,
                    c1_decoded,
                    c1_raw,
                    c2_decoded,
                    c2_raw,
                })
            })
            .collect()
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

    /// The report renderer: a pure function over the measured data, so its shape is
    /// testable without a GPU. Modeled on `docs/OCR_DEVICE_DIVERGENCE.md`'s shape.
    #[allow(clippy::too_many_arguments)]
    fn render_report(
        model: &Path,
        source_a_rows: &[PageRow],
        source_b_row: &PageRow,
        source_b_tile_rows: &[TileRow],
        tile_identity_asserted: bool,
        forced_multi_tile_tiles_inferred: usize,
        cuda_provider_request_count: usize,
        cpu_policy_report: &str,
        cuda_policy_report: &str,
        source_b_config: &pc_config::InpainterConfig,
    ) -> Result<String> {
        let mut body = String::new();

        body.push_str(
            "# LaMa device divergence (GPU-4 G4-C)\n\n\
             **Generated in full by `cargo xtask lama-device-compare` — do not hand-edit.** Every \
             value here is measured at generation time. This is a **non-gating** diagnostic: no \
             number here is a pass/fail gate (§16.22 item 2's CPU-determinism carve-out applying in \
             the negative direction: CUDA is exempt from determinism, so none of this asserts). It \
             records what the SAME real LaMa `lama-manga.onnx` graph produced on the CPU execution \
             provider vs the CUDA execution provider on one machine, across three independent full \
             `inpaint_page` runs sharing one `PageInput`.\n\n",
        );

        body.push_str("## 1. Method and provenance\n\n");
        let _ = writeln!(
            body,
            "- **Model:** {} — sha256 `{}`",
            file_display(model),
            sha256_of(model)
        );
        let _ = writeln!(
            body,
            "- **Runs:** three independent full `inpaint_page` runs per page, each wrapped in its \
             own fresh `RecordingInpainter` around its own `OnnxInpainter`: \
             `cpu = OnnxInpainter::from_path_with_policy(&model, &DevicePolicy::cpu())`, \
             `cuda_a` and `cuda_b` each from `resolve(Device::Cuda, DeviceSupport::compiled())`."
        );
        let _ = writeln!(
            body,
            "- **No cross-tile feedback loop:** `cover`/`owners` (the tile windows) are computed \
             once, before the inference loop starts, from pure CPU arithmetic over `original`/the \
             masks, with zero dependency on any tile's own inference output — so three independent \
             runs sharing one `PageInput` receive byte-identical tile crops by construction."
        );
        let _ = writeln!(
            body,
            "- **Tile-identity assertion:** after all three runs, every recorded `(tile_rgb, \
             mask_bits)` pair at index `i` is byte-identical across all three recordings, for every \
             `i`. Confirmed at generation time: **{}**. If this ever failed, the producer aborts \
             with a loud error rather than reporting divergence numbers.",
            if tile_identity_asserted {
                "PASSED"
            } else {
                "FAILED (this run aborted before completing)"
            }
        );
        let _ = writeln!(
            body,
            "- **Source A (default-config pages):** the six `tests/fixtures/upstream/demo_bubbles/ \
             *_bubble_raw.png` pages that actually inpaint under the shipped default \
             `InpainterConfig` (`black`, `darkrays`, `nightmare`, `ray`, `spikey`, `square`; \
             `handwritten` has 0 eligible regions and is skipped). Each page is real content, real \
             detection (a real `TextDetector` session), real masking, single-tile (every page is \
             under 512px on both axes, verified)."
        );
        let _ = writeln!(
            body,
            "- **Source A config:** the shipped default `InpainterConfig` verbatim — \
             `min_inpainting_radius: 7, max_inpainting_radius: 20, \
             inpainting_isolation_radius: 5, inpainting_fade_radius: 4`."
        );
        let _ = writeln!(
            body,
            "- **Source B (forced multi-tile):** the committed detector fixture \
             `tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg` \
             (a real, replayable page via `pc_detect::ReplayDetector`), run through the SAME real \
             detect → preprocess → mask pipeline but with a deliberately widened \
             `InpainterConfig`: `min_inpainting_radius = max_inpainting_radius = {}` (both equal, \
             so `padded_region`'s growth clamp is exact — see `crates/pc-inpaint/src/growth.rs`), \
             `inpainting_isolation_radius` and `inpainting_fade_radius` left at their shipped \
             defaults (`{}` and `{}`). This is a real, hard-coded producer constant, not a CLI \
             flag — a later run cannot silently change the disclosed method. Observed \
             `tiles_inferred` this run: **{forced_multi_tile_tiles_inferred}**.",
            source_b_config.min_inpainting_radius,
            source_b_config.inpainting_isolation_radius,
            source_b_config.inpainting_fade_radius,
        );
        let _ = writeln!(
            body,
            "- **CPU-pinned upstream stages:** detect, preprocess and mask run on the CPU in all \
             three runs — the CUDA policy is only ever handed to the inpainter session itself."
        );
        let _ = writeln!(
            body,
            "- **Disclosure stated up front:** the min-quantization-margin column below is \
             **CPU-only** — a characterization of the CPU run's own raw values, measuring how \
             close its byte rounding was to flipping. It is **not** a CPU-vs-CUDA comparison."
        );

        body.push_str("\n## 2. Device\n\n");
        let _ = writeln!(
            body,
            "- **CPU policy:** {}\n- **CUDA policy:** {}",
            cpu_policy_report, cuda_policy_report
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
        let _ = writeln!(
            body,
            "| CUDA provider requests | {} |",
            cuda_provider_request_count
        );

        body.push_str("\n## 4. Channels\n\n");
        body.push_str(
            "Channels 1 and 2 are per-tile: channel 1 compares the CPU run's tile output to \
             CUDA-A's, channel 2 compares CUDA-A's to CUDA-B's (self-nondeterminism, the analogue \
             of GPU-3's channel 4 at the tile level). Channels 3 and 4 are page-level: channel 3 \
             compares the CPU run's `inpainting`/`clean_inpaint` artifacts to CUDA-A's (whole \
             artifact and write-region-restricted to `final_mask > 0`, available as `inpainting`'s \
             own alpha channel), channel 4 the same CUDA-A vs CUDA-B. A whole-artifact number can \
             understate a real difference by orders of magnitude when most of the image is \
             untouched, so the write-region-restricted number is reported alongside it.\n\n",
        );

        body.push_str("### 4.1 Source A — per-page rows (shipped default config)\n\n");
        let _ = writeln!(
            body,
            "| Page | tiles | tiles_inferred | Ch1 decoded diff | Ch1 raw bitwise-diff | Ch1 raw max\\|Δ\\| | Ch2 decoded diff | Ch2 raw bitwise-diff | Ch2 raw max\\|Δ\\| | Ch3 inpainting diff | Ch3 inpainting where diff | Ch3 clean diff | Ch3 clean where diff | Ch4 inpainting diff | Ch4 clean diff | min quant margin |",
        );
        let _ = writeln!(
            body,
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        );

        for row in source_a_rows {
            let _ = writeln!(
                body,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                row.page,
                row.tiles,
                row.tiles_inferred,
                row.c1_decoded.differing_count,
                row.c1_raw.bitwise_differing_count,
                format_delta(row.c1_raw.max_abs_delta),
                row.c2_decoded.differing_count,
                row.c2_raw.bitwise_differing_count,
                format_delta(row.c2_raw.max_abs_delta),
                row.c3_inpainting.differing_count,
                row.c3_inpainting_where.differing_count,
                row.c3_clean.differing_count,
                row.c3_clean_where.differing_count,
                row.c4_inpainting.differing_count,
                row.c4_clean.differing_count,
                format_delta(row.min_margin),
            );
        }

        body.push_str("\n### 4.2 Source B — forced multi-tile page and per-tile rows\n\n");
        let _ = writeln!(
            body,
            "| Page | tiles | tiles_inferred | Ch1 decoded diff | Ch1 raw bitwise-diff | Ch1 raw max\\|Δ\\| | Ch2 decoded diff | Ch2 raw bitwise-diff | Ch2 raw max\\|Δ\\| | Ch3 inpainting diff | Ch3 inpainting where diff | Ch3 clean diff | Ch3 clean where diff | Ch4 inpainting diff | Ch4 clean diff | min quant margin |",
        );
        let _ = writeln!(
            body,
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        );
        let row = source_b_row;
        let _ = writeln!(
            body,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            row.page,
            row.tiles,
            row.tiles_inferred,
            row.c1_decoded.differing_count,
            row.c1_raw.bitwise_differing_count,
            format_delta(row.c1_raw.max_abs_delta),
            row.c2_decoded.differing_count,
            row.c2_raw.bitwise_differing_count,
            format_delta(row.c2_raw.max_abs_delta),
            row.c3_inpainting.differing_count,
            row.c3_inpainting_where.differing_count,
            row.c3_clean.differing_count,
            row.c3_clean_where.differing_count,
            row.c4_inpainting.differing_count,
            row.c4_clean.differing_count,
            format_delta(row.min_margin),
        );

        let _ = writeln!(
            body,
            "\nPer-tile channel 1/2 rows (decoded RGB and raw f32, per recorded tile index):\n"
        );
        let _ = writeln!(
            body,
            "| Tile | Ch1 decoded diff | Ch1 raw bitwise-diff | Ch1 raw max\\|Δ\\| | Ch1 raw mean\\|Δ\\| | Ch2 decoded diff | Ch2 raw bitwise-diff | Ch2 raw max\\|Δ\\| | Ch2 raw mean\\|Δ\\| |",
        );
        let _ = writeln!(body, "|---|---:|---:|---:|---:|---:|---:|---:|---:|");
        for tile in source_b_tile_rows {
            let _ = writeln!(
                body,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                tile.index,
                tile.c1_decoded.differing_count,
                tile.c1_raw.bitwise_differing_count,
                format_delta(tile.c1_raw.max_abs_delta),
                format_delta(tile.c1_raw.mean_abs_delta),
                tile.c2_decoded.differing_count,
                tile.c2_raw.bitwise_differing_count,
                format_delta(tile.c2_raw.max_abs_delta),
                format_delta(tile.c2_raw.mean_abs_delta),
            );
        }

        // Global aggregates: the max-of-maxes and mean-of-means of the raw f32
        // divergence across ALL tiles of both sources, for channel 1 (CPU vs CUDA-A) and
        // channel 2 (CUDA-A vs CUDA-B). These are the numbers the ratification step
        // summarises. `mean-of-means` here is the mean of the per-tile mean-abs-deltas
        // (each tile is one observation), and `max-of-maxes` is the largest per-tile
        // max-abs-delta — both computed over the same per-tile `TensorDivergence`s the
        // rows above report.
        body.push_str("\n### 4.3 Global aggregates (all tiles, both sources)\n\n");
        let _ = writeln!(
            body,
            "| Aggregate | Ch1 raw max-of-maxes | Ch1 raw mean-of-means | Ch2 raw max-of-maxes | Ch2 raw mean-of-means |",
        );
        let _ = writeln!(body, "|---|---:|---:|---:|---:|");

        let mut c1_maxes = Vec::new();
        let mut c1_means = Vec::new();
        let mut c2_maxes = Vec::new();
        let mut c2_means = Vec::new();
        for row in source_a_rows.iter().chain(std::iter::once(source_b_row)) {
            c1_maxes.push(row.c1_raw.max_abs_delta);
            c1_means.push(row.c1_raw.mean_abs_delta);
            c2_maxes.push(row.c2_raw.max_abs_delta);
            c2_means.push(row.c2_raw.mean_abs_delta);
        }
        for tile in source_b_tile_rows {
            c1_maxes.push(tile.c1_raw.max_abs_delta);
            c1_means.push(tile.c1_raw.mean_abs_delta);
            c2_maxes.push(tile.c2_raw.max_abs_delta);
            c2_means.push(tile.c2_raw.mean_abs_delta);
        }
        let max_of = |values: &[f32]| values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean_of = |values: &[f32]| {
            if values.is_empty() || values.iter().any(|value| value.is_nan()) {
                f32::NAN
            } else {
                values.iter().sum::<f32>() / values.len() as f32
            }
        };
        let _ = writeln!(
            body,
            "| global | {} | {} | {} | {} |",
            format_delta(max_of(&c1_maxes)),
            format_delta(mean_of(&c1_means)),
            format_delta(max_of(&c2_maxes)),
            format_delta(mean_of(&c2_means)),
        );

        body.push_str("\n## 5. Verdict\n\n");
        body.push_str(
            "**Non-gating.** No number above is a pass/fail gate. CUDA is exempt from determinism, so \
             none of this asserts. The numbers record what actually happened on this machine at \
             generation time, for the ratification step (G4-D) to cite.\n\n",
        );

        Ok(body)
    }

    pub(crate) fn run(args: Args) -> Result<()> {
        // Resolve the model path (arg → env → managed cache path).
        let model = resolve_model_path(args.model.as_deref())?;
        let out = args.out;

        println!("model: {}", file_display(&model));
        println!("model sha256: {}", sha256_of(&model));

        // The CUDA policy is resolved ONCE, before any session is built. The hard-error
        // guard fires if it carries no provider request — otherwise the CUDA runs could
        // silently be CPU runs in disguise.
        let cuda_policy = resolve(Device::Cuda, DeviceSupport::compiled())
            .map_err(|refusal| anyhow::anyhow!(refusal.message()))?;
        if cuda_policy.provider_requests().is_empty() {
            bail!(
                "the resolved CUDA policy carries no execution-provider request — the CUDA \
                 runs would silently be CPU runs in disguise; refusing to produce a report"
            );
        }
        println!("\ncuda policy: {}", cuda_policy.report());

        // Source A: the six demo_bubbles pages that actually inpaint under the shipped
        // default config. Real detection needs the real ONNX detector session.
        let detector_model = resolve_detector_model()?;
        println!("detector: {}", file_display(&detector_model));
        let detector = pc_detect::onnx::OnnxDetector::from_path(&detector_model)
            .map_err(|error| anyhow!("building the ONNX detector: {error}"))?;
        let default_config = default_inpainter_config();
        let mut source_a_rows = Vec::new();
        for name in SOURCE_A_PAGES {
            let path = pc_testkit::paths::upstream(format!("demo_bubbles/{name}_bubble_raw.png"));
            let triple = measure_page(
                &path,
                &detector,
                &model,
                &cuda_policy,
                &default_config,
                &format!("Source A — {name}"),
            )?;
            let row = measure_page_row(name, &triple)?;
            println!(
                "Source A — {name}: tiles inferred {}, channel 1 decoded differing bytes {}",
                row.tiles_inferred, row.c1_decoded.differing_count
            );
            source_a_rows.push(row);
        }

        // Source B: the forced multi-tile replay page.
        let fixture_dir = paths::recorded_root().join("detector");
        let replay_image = fixture_dir.join(format!("{REPLAY_STEM}.jpg"));
        let replay_detector = pc_detect::ReplayDetector::new(&fixture_dir, REPLAY_STEM);
        let source_b_config = source_b_inpainter_config();
        let source_b_triple = measure_page(
            &replay_image,
            &replay_detector,
            &model,
            &cuda_policy,
            &source_b_config,
            "Source B — forced multi-tile",
        )?;
        // Hard-error guard: the forced-multi-tile source must actually produce >= 2 tiles
        // — a silent single-tile or zero-tile run would produce a clean-looking report that
        // measured nothing about tiling. If it does not, do not adjust the constant — report
        // the actual value and the failure.
        let source_b_tiles_inferred = source_b_triple.pages[0].tiles_inferred;
        if source_b_tiles_inferred < 2 {
            bail!(
                "the forced multi-tile source produced {source_b_tiles_inferred} tiles \
                 (guard: >= 2) — a silent single-tile or zero-tile run would measure \
                 nothing about tiling. Not adjusting the radii; reporting the actual value \
                 and flagging back to the Orchestrator."
            );
        }
        println!("Source B — forced multi-tile: tiles inferred {source_b_tiles_inferred}");
        let source_b_row = measure_page_row("forced-multi-tile", &source_b_triple)?;
        let source_b_tile_rows = measure_tile_rows(&source_b_triple)?;

        // Write the report.
        let document = render_report(
            &model,
            &source_a_rows,
            &source_b_row,
            &source_b_tile_rows,
            true,
            source_b_tiles_inferred,
            cuda_policy.provider_requests().len(),
            &pc_core::device::DevicePolicy::cpu().report(),
            &cuda_policy.report(),
            &source_b_config,
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

    #[cfg(test)]
    mod tests {
        use super::*;

        fn zero_bytes_divergence(len: usize) -> ByteDivergence {
            ByteDivergence {
                len,
                differing_count: 0,
                max_abs_delta: 0,
            }
        }

        fn sample_row(name: &str) -> PageRow {
            PageRow {
                page: name.to_owned(),
                tiles: 1,
                tiles_inferred: 1,
                c1_decoded: zero_bytes_divergence(786432),
                c1_raw: TensorDivergence {
                    len: 786432,
                    bitwise_differing_count: 0,
                    max_abs_delta: 0.0,
                    mean_abs_delta: 0.0,
                },
                c2_decoded: zero_bytes_divergence(786432),
                c2_raw: TensorDivergence {
                    len: 786432,
                    bitwise_differing_count: 0,
                    max_abs_delta: 0.0,
                    mean_abs_delta: 0.0,
                },
                c3_inpainting: zero_bytes_divergence(0),
                c3_inpainting_where: zero_bytes_divergence(0),
                c3_clean: zero_bytes_divergence(0),
                c3_clean_where: zero_bytes_divergence(0),
                c4_inpainting: zero_bytes_divergence(0),
                c4_clean: zero_bytes_divergence(0),
                min_margin: 0.5,
            }
        }

        fn sample_tile_row(index: usize) -> TileRow {
            TileRow {
                index,
                c1_decoded: zero_bytes_divergence(786432),
                c1_raw: TensorDivergence {
                    len: 786432,
                    bitwise_differing_count: 0,
                    max_abs_delta: 0.0,
                    mean_abs_delta: 0.0,
                },
                c2_decoded: zero_bytes_divergence(786432),
                c2_raw: TensorDivergence {
                    len: 786432,
                    bitwise_differing_count: 0,
                    max_abs_delta: 0.0,
                    mean_abs_delta: 0.0,
                },
            }
        }

        fn source_b_config() -> pc_config::InpainterConfig {
            pc_config::InpainterConfig {
                min_inpainting_radius: 300,
                max_inpainting_radius: 300,
                ..pc_config::InpainterConfig::default()
            }
        }

        #[test]
        fn render_report_includes_the_method_device_environment_and_non_gating_verdict() {
            let document = render_report(
                Path::new("lama-manga.onnx"),
                &[sample_row("black"), sample_row("darkrays")],
                &sample_row("forced-multi-tile"),
                &[sample_tile_row(0), sample_tile_row(1)],
                true,
                3,
                1,
                "cpu policy report",
                "cuda policy report",
                &source_b_config(),
            )
            .expect("render");
            assert!(document.contains("## 1. Method and provenance"));
            assert!(document.contains("## 2. Device"));
            assert!(document.contains("## 3. Environment"));
            assert!(document.contains("## 4. Channels"));
            assert!(document.contains("## 5. Verdict"));
            assert!(document.contains("**Non-gating.**"));
            assert!(document.contains("PASSED"));
            assert!(document.contains("cpu policy report"));
            assert!(document.contains("cuda policy report"));
            assert!(document.contains("| CUDA provider requests | 1 |"));
            assert!(document.contains("Observed `tiles_inferred` this run: **3**"));
            assert!(document.contains("min_inpainting_radius = max_inpainting_radius = 300"));
            assert!(document.contains("| black |"));
            assert!(document.contains("| darkrays |"));
            assert!(document.contains("| forced-multi-tile |"));
            assert!(
                document.contains("### 4.2 Source B — forced multi-tile page and per-tile rows")
            );
            assert!(document.contains("### 4.3 Global aggregates (all tiles, both sources)"));
            assert!(document.contains("| 0 |"));
            assert!(document.contains("| 1 |"));
        }

        #[test]
        fn render_report_discloses_failed_assertion_in_method() {
            let document = render_report(
                Path::new("lama-manga.onnx"),
                &[sample_row("black")],
                &sample_row("forced-multi-tile"),
                &[sample_tile_row(0)],
                false,
                3,
                1,
                "cpu",
                "cuda",
                &pc_config::InpainterConfig::default(),
            )
            .expect("render");
            assert!(document.contains("FAILED (this run aborted before completing)"));
        }

        #[test]
        fn render_report_discloses_the_min_margin_column_is_cpu_only() {
            let document = render_report(
                Path::new("lama-manga.onnx"),
                &[sample_row("black")],
                &sample_row("forced-multi-tile"),
                &[sample_tile_row(0)],
                true,
                3,
                1,
                "cpu",
                "cuda",
                &pc_config::InpainterConfig::default(),
            )
            .expect("render");
            assert!(document.contains("CPU-only"));
            assert!(document.contains("CPU-vs-CUDA comparison"));
        }

        #[test]
        fn assert_tile_identity_accepts_identical_recordings() {
            let make_record = |value: u8| TileRecord {
                tile_rgb: vec![value],
                mask_bits: vec![1],
                raw_shape: vec![1, 3, 1, 1],
                raw_values: vec![0.5],
                decoded_rgb: vec![value],
            };
            let triple = TripleRun {
                tiles: vec![
                    vec![make_record(1), make_record(2)],
                    vec![make_record(1), make_record(2)],
                    vec![make_record(1), make_record(2)],
                ],
                pages: vec![],
            };
            assert!(assert_tile_identity(&triple, "test page").is_ok());
        }

        #[test]
        fn assert_tile_identity_rejects_differing_tile_inputs() {
            let make_record = |tile: u8, mask: u8| TileRecord {
                tile_rgb: vec![tile],
                mask_bits: vec![mask],
                raw_shape: vec![1, 3, 1, 1],
                raw_values: vec![0.5],
                decoded_rgb: vec![0],
            };
            let triple = TripleRun {
                tiles: vec![
                    vec![make_record(1, 1)],
                    vec![make_record(1, 1)],
                    vec![make_record(2, 1)],
                ],
                pages: vec![],
            };
            let error = assert_tile_identity(&triple, "test page").expect_err("must fail");
            assert!(error.to_string().contains("tile-identity assertion FAILED"));
        }

        #[test]
        fn merge_bytes_and_tensors_aggregate_like_a_contiguous_buffer() {
            let a = ByteDivergence {
                len: 3,
                differing_count: 1,
                max_abs_delta: 7,
            };
            let b = ByteDivergence {
                len: 2,
                differing_count: 1,
                max_abs_delta: 3,
            };
            let merged = merge_bytes(a, b);
            assert_eq!(merged.len, 5);
            assert_eq!(merged.differing_count, 2);
            assert_eq!(merged.max_abs_delta, 7);

            let ta = TensorDivergence {
                len: 2,
                bitwise_differing_count: 1,
                max_abs_delta: 0.5,
                mean_abs_delta: 0.25,
            };
            let tb = TensorDivergence {
                len: 2,
                bitwise_differing_count: 0,
                max_abs_delta: 0.1,
                mean_abs_delta: 0.05,
            };
            let merged = merge_tensors(ta, tb);
            assert_eq!(merged.len, 4);
            assert_eq!(merged.bitwise_differing_count, 1);
            assert_eq!(merged.max_abs_delta, 0.5);
            assert!((merged.mean_abs_delta - 0.15).abs() < 1e-6);
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) use cuda::run;

#[cfg(not(feature = "cuda"))]
pub(crate) fn run(_args: Args) -> Result<()> {
    bail!(
        "`lama-device-compare` requires the `cuda` feature: rebuild with \
         `cargo build --features xtask/cuda -p xtask` (this build has no CUDA execution \
         provider linked in, so there is nothing to measure)"
    )
}
