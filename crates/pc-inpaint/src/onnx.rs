//! Task **L5** (spec §16.38 item 16(d)): the `ort`-backed [`crate::Inpainter`].
//!
//! *"L5 — the ONNX session behind the `onnx` feature (heavy, its own call). `ort` session
//! construction through `pc_core::device::resolve` (§16.36 item 2, no second policy
//! surface), the lazy `OnceLock` latch of item 8, the run-fatal declaration of item 9,
//! NCHW tensor construction for two separate inputs (item 1(b)), and **runtime**
//! output-shape validation per item 1(c)."*
//!
//! The tensor arithmetic and the output validation live **ungated** in this file, exactly
//! as `pc_detect::onnx` keeps `to_nchw`/`bind_outputs` ungated: they are the
//! parity-critical part, and cookbook rule 6 records the default no-`onnx` tier as the one
//! that actually executes. Only [`OnnxInpainter`] itself needs `ort`.
//!
//! **Fatality, declared and not inferred** (cookbook rule 4, §16.38 item 9(g)):
//!
//! | failure | classification | variant |
//! |---|---|---|
//! | session construction (missing / corrupt / unloadable model, refused device) | **run-fatal** | [`StageError::Model`] |
//! | anything inside `Session::run`, or a bad runtime output shape | **per-image** | [`StageError::Inference`] |
//!
//! Construction takes no image, which is what licences the run-fatal half (§16.38 item
//! 9(b)); item 9(c)–(e) is what chooses it. `inpaint_tile` receives the image, so its
//! failures are per-image.

use crate::TILE;
use image::{Rgb, RgbImage};
use pc_core::StageError;
use pc_imageops::BinaryMask;
use std::path::Path;

/// The `image` input's name in the pinned graph (§16.38 item 1(b)).
pub const IMAGE_INPUT: &str = "image";
/// The `mask` input's name in the pinned graph (§16.38 item 1(b)).
///
/// There is deliberately **no** concatenated 4-channel input: item 1(b) decoded two
/// separate `ValueInfoProto`s, and item 6(a)'s sibling `config.json` declaring
/// `"input_channels": 4` describes the *architecture*, not this graph's interface.
pub const MASK_INPUT: &str = "mask";

/// `image` is `[batch, 3, 512, 512]` (§16.38 item 1(b)).
pub const IMAGE_CHANNELS: usize = 3;
/// `mask` is `[batch, 1, 512, 512]` (§16.38 item 1(b)).
pub const MASK_CHANNELS: usize = 1;

/// The pinned graph is exercised one tile at a time.
///
/// §16.38 item 1(d) measured the `batch` axis as genuinely dynamic but **linear** in wall
/// clock (1.58 / 3.03 / 6.55 s for `b = 1 / 2 / 4`), so batching buys no throughput. It is
/// a named constant rather than a bare `1` only so the tensor shapes below read as shapes.
pub const BATCH: usize = 1;

/// The number of set-mask samples the model is handed per tile call.
const MASK_LEN: usize = MASK_CHANNELS * (TILE as usize) * (TILE as usize);
/// The number of image samples per tile call.
const IMAGE_LEN: usize = IMAGE_CHANNELS * (TILE as usize) * (TILE as usize);

/// The runtime output shape this implementation requires, and the ONLY shape it accepts.
///
/// **This is a hard-coded literal on purpose, and it is not read from the graph.** §16.38
/// item 1(c) decoded the raw protobuf's `output` value-info as
/// `[dim_param "batch", 3, dim_param "batch", dim_param "Sigmoidoutput_dim_3"]` — three of
/// four axes symbolic, with the **height axis reusing the batch symbol** — and states the
/// binding consequence: *"the implementation must validate the **runtime** output shape and
/// must never derive geometry from the declared one."* So [`OnnxInpainter`] never reads
/// `session.outputs()[0]`'s shape, and [`decode_output_tile`] compares against this
/// constant instead.
///
/// **MEASURED at L5, and it narrows item 1(c) rather than contradicting it.** Loaded through
/// `ort` 2.0.0-rc.12, `session.outputs()` reports the output as `("output", f32,
/// [-1, 3, 512, 512])` — ONNX Runtime runs its own shape inference at load and concretises
/// the two spatial symbols, so the *declared* shape an `ort` caller sees is **not** the
/// unusable one item 1(c) read out of the protobuf. That does not license trusting it: item
/// 1(c)'s obligation is on the runtime shape, an inference pass is a property of the runtime
/// rather than of the artifact, and the runtime shape is what the decode actually indexes
/// against. The runtime shape was measured as `[1, 3, 512, 512]` — equal to this constant —
/// which is the resolution item 1(c) left open.
pub const EXPECTED_OUTPUT_SHAPE: [i64; 4] = [BATCH as i64, 3, TILE as i64, TILE as i64];

/// The divisor applied to the `image` input and the multiplier applied to the output.
///
/// **MEASURED, task L5, 2026-08-07**, against the pinned artifact (sha256
/// `50a1abae…b0100`, 207,482,644 bytes) on the CPU execution provider — recorded here
/// because the graph itself declares nothing about it and the alternative was a guess.
/// The measurement is reproduced by
/// `a_zero_mask_still_regenerates_the_tile_but_stays_close_to_the_input_which_pins_the_scaling`
/// in `tests/l5_real_model.rs`, whose mean-absolute-error bound is what would go red if the
/// division (or the matching multiplication) were dropped; the probe is described in the spec
/// transcription this task files, not re-derived here.
pub const SAMPLE_SCALE: f32 = 255.0;

/// Ensure a model path names a regular file before any `ort` call.
///
/// Byte-identical in behaviour to `pc_detect::onnx::ensure_model_file` and
/// `pc_ocr::onnx::ensure_model_file`; duplicated rather than shared because neither of
/// those crates is a dependency of this one and a stage crate does not gain a dependency
/// on a sibling stage for four lines.
pub fn ensure_model_file(path: &Path) -> Result<(), StageError> {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(()),
        Ok(_) | Err(_) => Err(StageError::Model(format!(
            "model path is missing or is not a regular file: {}",
            path.display()
        ))),
    }
}

/// The `image` tensor's payload: planar NCHW float32, each channel divided by
/// [`SAMPLE_SCALE`] (§16.38 item 1(b)).
///
/// `Err` is [`StageError::Inference`], not [`StageError::InvalidInput`]: this is only ever
/// reached from [`crate::Inpainter::inpaint_tile`], whose whole contract is the per-image
/// half of §16.38 item 9(g)'s split.
pub fn image_to_nchw(tile: &RgbImage) -> Result<Vec<f32>, StageError> {
    if tile.dimensions() != (TILE, TILE) {
        return Err(StageError::Inference(format!(
            "the inpainting tile is {:?}; the pinned artifact's `image` input is \
             [batch, 3, {TILE}, {TILE}] with literal {TILE}s (§16.38 item 1(b))",
            tile.dimensions()
        )));
    }
    let plane = (TILE as usize) * (TILE as usize);
    let mut tensor = vec![0.0_f32; IMAGE_LEN];
    for y in 0..TILE {
        for x in 0..TILE {
            let Rgb([r, g, b]) = *tile.get_pixel(x, y);
            let index = (y as usize) * (TILE as usize) + (x as usize);
            tensor[index] = f32::from(r) / SAMPLE_SCALE;
            tensor[plane + index] = f32::from(g) / SAMPLE_SCALE;
            tensor[2 * plane + index] = f32::from(b) / SAMPLE_SCALE;
        }
    }
    Ok(tensor)
}

/// The `mask` tensor's payload: one NCHW plane, `1.0` where the mask is set.
///
/// §16.38 item 1(g): *"Mask convention confirmed: value 1 = fill, value 0 = keep."* A set
/// bit means **inpaint this pixel**, which is also [`crate::Inpainter`]'s documented
/// convention, so this function is deliberately not an inversion point.
pub fn mask_to_nchw(mask: &BinaryMask) -> Result<Vec<f32>, StageError> {
    if mask.dimensions() != (TILE, TILE) {
        return Err(StageError::Inference(format!(
            "the inpainting mask is {:?}; the pinned artifact's `mask` input is \
             [batch, 1, {TILE}, {TILE}] with literal {TILE}s (§16.38 item 1(b))",
            mask.dimensions()
        )));
    }
    let mut tensor = vec![0.0_f32; MASK_LEN];
    for y in 0..TILE {
        for x in 0..TILE {
            if mask.get(x, y) {
                tensor[(y as usize) * (TILE as usize) + (x as usize)] = 1.0;
            }
        }
    }
    Ok(tensor)
}

/// Validate a **runtime** output shape against [`EXPECTED_OUTPUT_SHAPE`] and decode it to
/// an RGB tile.
///
/// This is §16.38 item 1(c)'s obligation discharged: the declared shape is never consulted,
/// so the graph's symbolic height axis cannot mislead the geometry. Every rejection is
/// [`StageError::Inference`] — per §16.38 item 9(g) this code path only runs with an image
/// in hand.
///
/// Samples are multiplied by [`SAMPLE_SCALE`], rounded half-away-from-zero (`f32::round`'s
/// documented rule) and clamped into `0..=255`. Clamping rather than rejecting an
/// out-of-range sample is deliberate: the graph ends in a sigmoid, so `1.0 + 1e-7` is
/// rounding noise and not a defect — the L5 measurement saw a maximum of `0.99999857`.
///
/// A **non-finite** sample is a rejection, not a clamp, and that choice is declared here
/// because the spec is silent on it: `NaN` has no defensible byte value, `f32::clamp` panics
/// on it, and picking black or white silently would put an invented pixel on the page. The
/// classification is the per-image one, so one bad tile costs one page and not the run.
pub fn decode_output_tile(shape: &[i64], values: &[f32]) -> Result<RgbImage, StageError> {
    if shape != EXPECTED_OUTPUT_SHAPE {
        return Err(StageError::Inference(format!(
            "the inpainting model returned runtime output shape {shape:?}, expected \
             {EXPECTED_OUTPUT_SHAPE:?}; the graph's DECLARED shape is unusable (§16.38 \
             item 1(c): three of four axes symbolic, height reusing the batch symbol) and \
             is deliberately not consulted"
        )));
    }
    if values.len() != IMAGE_LEN {
        return Err(StageError::Inference(format!(
            "the inpainting model returned {} samples for shape {shape:?}, expected \
             {IMAGE_LEN}",
            values.len()
        )));
    }

    if let Some(index) = values.iter().position(|sample| !sample.is_finite()) {
        return Err(StageError::Inference(format!(
            "the inpainting model returned a non-finite sample ({}) at flat index {index} \
             of shape {shape:?}; there is no defensible byte value for it, so this tile is \
             refused rather than filled with an invented pixel",
            values[index]
        )));
    }

    let plane = (TILE as usize) * (TILE as usize);
    Ok(RgbImage::from_fn(TILE, TILE, |x, y| {
        let index = (y as usize) * (TILE as usize) + (x as usize);
        Rgb([
            to_byte(values[index]),
            to_byte(values[plane + index]),
            to_byte(values[2 * plane + index]),
        ])
    }))
}

/// Only ever called after [`decode_output_tile`]'s non-finite scan, so `clamp` cannot see a
/// `NaN` here.
fn to_byte(sample: f32) -> u8 {
    (sample * SAMPLE_SCALE).round().clamp(0.0, 255.0) as u8
}

#[cfg(feature = "onnx")]
mod session {
    use super::{
        decode_output_tile, ensure_model_file, image_to_nchw, mask_to_nchw, BATCH, IMAGE_CHANNELS,
        IMAGE_INPUT, MASK_CHANNELS, MASK_INPUT,
    };
    use crate::{Inpainter, TILE};
    use image::RgbImage;
    use ort::{
        session::{builder::GraphOptimizationLevel, Session},
        value::{Outlet, Tensor, TensorElementType, ValueType},
    };
    use pc_core::device::{Device, DeviceSupport};
    use pc_core::StageError;
    use pc_imageops::BinaryMask;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    /// Whether ONNX Runtime's shared environment can be obtained.
    ///
    /// Same probe, and the same caveat, as `pc_detect::onnx::runtime_available` and
    /// `pc_ocr::onnx::runtime_available`: it does not claim a process-global environment
    /// via `ort::init()`, and a missing shared library can still fail at link time before
    /// this is reachable.
    pub fn runtime_available() -> bool {
        ort::environment::current().is_ok()
    }

    /// The `ort`-backed [`Inpainter`] — one session, one tile per call.
    #[derive(Debug)]
    pub struct OnnxInpainter {
        // `ort` rc.12's `Session::run` takes `&mut self`, so a shareable `Inpainter`
        // (`Send + Sync`) needs interior mutability. Same reasoning, same mechanism and
        // the same poison recovery as `pc_ocr::onnx::MangaOcrSessions`.
        session: Mutex<Session>,
        model_path: PathBuf,
    }

    impl OnnxInpainter {
        /// Construct on the CPU execution provider.
        pub fn from_path(model: &Path) -> Result<Self, StageError> {
            Self::from_path_for_device(model, Device::Cpu)
        }

        /// Construct for an explicitly requested device.
        ///
        /// §16.38 item 16(d): *"`ort` session construction through
        /// `pc_core::device::resolve` (§16.36 item 2, **no second policy surface**)"*, and
        /// §16.22 item 5(f): *"the device policy resolver is model-agnostic, so v1.5's
        /// LaMa inpainting reuses it instead of growing a second one"*. This function
        /// therefore has no device knobs of its own.
        ///
        /// Every failure here is [`StageError::Model`], i.e. run-fatal (§16.38 item 9).
        pub fn from_path_for_device(model: &Path, device: Device) -> Result<Self, StageError> {
            pc_core::device::resolve(device, DeviceSupport::compiled())
                .map_err(|refusal| StageError::Model(refusal.message()))?;

            // Before any ort call, so a missing file reports its own path rather than an
            // ORT parse error about it.
            ensure_model_file(model)?;

            let session = build_session(model)?;
            validate_declared_inputs(model, &session)?;
            Ok(Self {
                session: Mutex::new(session),
                model_path: model.to_path_buf(),
            })
        }

        pub fn model_path(&self) -> &Path {
            &self.model_path
        }

        /// The graph's inputs, as `(name, element type, declared shape)`.
        ///
        /// Exposed so a real-weights test can compare the live graph against §16.38 item
        /// 1(b)'s decoded signature without going through inference.
        pub fn declared_inputs(&self) -> Vec<(String, String, Vec<i64>)> {
            let session = self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            session.inputs().iter().map(describe_outlet).collect()
        }

        /// The graph's outputs, in the same form.
        ///
        /// **Nothing in this crate's inference path reads this** — §16.38 item 1(c)
        /// forbids deriving geometry from the declared output shape. It exists only so a
        /// real-weights test can *record* what `ort` reports, which at L5 measured as
        /// `("output", f32, [-1, 3, 512, 512])` and is therefore NOT the fully symbolic
        /// shape item 1(c) decoded from the protobuf; see
        /// [`super::EXPECTED_OUTPUT_SHAPE`] for why that changes nothing.
        pub fn declared_outputs(&self) -> Vec<(String, String, Vec<i64>)> {
            let session = self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            session.outputs().iter().map(describe_outlet).collect()
        }

        /// One inference, returning the **runtime** output shape alongside the samples.
        ///
        /// Split out from [`Inpainter::inpaint_tile`] so a real-weights test can assert
        /// the measured shape and value range directly, rather than inferring them from a
        /// decoded image that has already been clamped into `0..=255`.
        pub fn run_raw(
            &self,
            tile: &RgbImage,
            mask: &BinaryMask,
        ) -> Result<(Vec<i64>, Vec<f32>), StageError> {
            let image_values = image_to_nchw(tile)?;
            let mask_values = mask_to_nchw(mask)?;
            let side = TILE as usize;
            let image_tensor =
                Tensor::<f32>::from_array(([BATCH, IMAGE_CHANNELS, side, side], image_values))
                    .map_err(|error| {
                        StageError::Inference(format!(
                            "failed to create the `{IMAGE_INPUT}` tensor: {error}"
                        ))
                    })?;
            let mask_tensor =
                Tensor::<f32>::from_array(([BATCH, MASK_CHANNELS, side, side], mask_values))
                    .map_err(|error| {
                        StageError::Inference(format!(
                            "failed to create the `{MASK_INPUT}` tensor: {error}"
                        ))
                    })?;

            let mut session = self.session.lock().unwrap_or_else(|poisoned| {
                tracing::warn!(
                    "a previous inpainting inference panicked inside the session lock; \
                     reusing the session"
                );
                poisoned.into_inner()
            });
            // Two SEPARATE named inputs (§16.38 item 1(b)); binding by name rather than by
            // position so a re-export that reorders the graph's inputs cannot silently swap
            // the image and the mask.
            let outputs = session
                .run(ort::inputs![IMAGE_INPUT => image_tensor, MASK_INPUT => mask_tensor])
                .map_err(|error| {
                    StageError::Inference(format!("ONNX inpainting session run failed: {error}"))
                })?;
            if outputs.len() != 1 {
                return Err(StageError::Inference(format!(
                    "the inpainting model returned {} outputs, expected exactly 1",
                    outputs.len()
                )));
            }
            let (_name, output) = outputs
                .iter()
                .next()
                .ok_or_else(|| StageError::Inference("no inpainting output".to_owned()))?;
            let (shape, values) = output.try_extract_tensor::<f32>().map_err(|error| {
                StageError::Inference(format!(
                    "failed to extract the inpainting output as f32: {error}"
                ))
            })?;
            Ok((shape.to_vec(), values.to_vec()))
        }
    }

    impl Inpainter for OnnxInpainter {
        fn inpaint_tile(&self, tile: &RgbImage, mask: &BinaryMask) -> Result<RgbImage, StageError> {
            let (shape, values) = self.run_raw(tile, mask)?;
            decode_output_tile(&shape, &values)
        }
    }

    fn build_session(model: &Path) -> Result<Session, StageError> {
        let mut builder = Session::builder().map_err(|error| {
            StageError::Model(format!("failed to configure {}: {error}", model.display()))
        })?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| {
                StageError::Model(format!("failed to configure {}: {error}", model.display()))
            })?;
        // 0 = "let ONNX Runtime choose", matching `pc_ocr::onnx::build_session`. §16.38 item
        // 1(f) measured ~1.6 s/tile all-core against ~4.7 s single-threaded on the spike
        // machine, so the default is the only defensible one and there is no config key for
        // it in the ratified eight-key `[inpainter]` block (item 13(a)).
        builder = builder.with_intra_threads(0).map_err(|error| {
            StageError::Model(format!("failed to configure {}: {error}", model.display()))
        })?;
        builder = builder.with_inter_threads(0).map_err(|error| {
            StageError::Model(format!("failed to configure {}: {error}", model.display()))
        })?;
        // No execution provider is registered: ONNX Runtime's built-in CPU provider is
        // implicit, which is what `DevicePolicy::cpu()` reports (§16.36).
        builder.commit_from_file(model).map_err(|error| {
            StageError::Model(format!("failed to load {}: {error}", model.display()))
        })
    }

    /// Check the two **input** signatures against §16.38 item 1(b)'s decoded values.
    ///
    /// The inputs are checked at construction because they are declared with literal
    /// `dim_value`s — item 1(b) decoded `0a 03 08 80 04` — so a mismatch means the artifact
    /// is not the pinned one and nothing this stage does will work. That makes it a
    /// construction failure, i.e. run-fatal [`StageError::Model`] (§16.38 item 9).
    ///
    /// The **output** is deliberately checked only for arity and element type here. Its
    /// declared shape is the unusable one of item 1(c) and is validated at run time
    /// instead, by `decode_output_tile`.
    fn validate_declared_inputs(model: &Path, session: &Session) -> Result<(), StageError> {
        let inputs: Vec<(String, String, Vec<i64>)> =
            session.inputs().iter().map(describe_outlet).collect();
        if inputs.len() != 2 {
            return Err(StageError::Model(format!(
                "{} has {} inputs, expected exactly 2 (`{IMAGE_INPUT}` and `{MASK_INPUT}`, \
                 which are SEPARATE tensors and not a concatenated 4-channel one — §16.38 \
                 item 1(b)); {}",
                model.display(),
                inputs.len(),
                describe(&inputs)
            )));
        }

        for (name, channels) in [
            (IMAGE_INPUT, IMAGE_CHANNELS as i64),
            (MASK_INPUT, MASK_CHANNELS as i64),
        ] {
            let found = inputs.iter().find(|(input, _, _)| input == name);
            let Some((_, element_type, shape)) = found else {
                return Err(StageError::Model(format!(
                    "{} has no input named `{name}` (§16.38 item 1(b)); {}",
                    model.display(),
                    describe(&inputs)
                )));
            };
            if element_type != "f32" {
                return Err(StageError::Model(format!(
                    "{}'s `{name}` input has element type {element_type}, expected f32 \
                     (§16.38 item 1(b): `elem_type = 1`); {}",
                    model.display(),
                    describe(&inputs)
                )));
            }
            // The batch axis is genuinely dynamic (item 1(d) ran b = 2 and b = 4), so it is
            // the only axis not pinned here.
            let spatial_ok = shape.len() == 4
                && shape[1] == channels
                && shape[2] == i64::from(TILE)
                && shape[3] == i64::from(TILE);
            if !spatial_ok {
                return Err(StageError::Model(format!(
                    "{}'s `{name}` input has declared shape {shape:?}, expected \
                     [batch, {channels}, {TILE}, {TILE}] (§16.38 item 1(b) decoded the \
                     H/W axes as literal {TILE}s, and item 1(d) measured three off-size \
                     inputs being rejected); {}",
                    model.display(),
                    describe(&inputs)
                )));
            }
        }

        let outputs: Vec<(String, String, Vec<i64>)> =
            session.outputs().iter().map(describe_outlet).collect();
        if outputs.len() != 1 {
            return Err(StageError::Model(format!(
                "{} has {} outputs, expected exactly 1; {}",
                model.display(),
                outputs.len(),
                describe(&outputs)
            )));
        }
        if outputs[0].1 != "f32" {
            return Err(StageError::Model(format!(
                "{}'s output has element type {}, expected f32; {}",
                model.display(),
                outputs[0].1,
                describe(&outputs)
            )));
        }
        // Its declared SHAPE is not checked, on purpose. §16.38 item 1(c).
        Ok(())
    }

    fn describe_outlet(outlet: &Outlet) -> (String, String, Vec<i64>) {
        let (element_type, shape) = match outlet.dtype() {
            ValueType::Tensor { ty, shape, .. } => (
                match ty {
                    TensorElementType::Float32 => "f32".to_owned(),
                    other => format!("{other:?}"),
                },
                shape.to_vec(),
            ),
            other => (format!("{other:?}"), Vec::new()),
        };
        (outlet.name().to_owned(), element_type, shape)
    }

    fn describe(tensors: &[(String, String, Vec<i64>)]) -> String {
        use std::fmt::Write as _;
        let mut rendered = String::from("tensors:");
        for (index, (name, element_type, shape)) in tensors.iter().enumerate() {
            let _ = write!(rendered, " [{index}] {name} {element_type} {shape:?}");
        }
        rendered
    }
}

#[cfg(feature = "onnx")]
pub use session::{runtime_available, OnnxInpainter};
