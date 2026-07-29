//! Pure ONNX preprocessing and output decoding (spec §8.3, steps 3 and 5).
//!
//! The runtime-backed [`OnnxDetector`] belongs behind the `onnx` feature. These helpers
//! are deliberately ungated so their parity-critical arithmetic can be tested without
//! an ONNX Runtime installation.

use crate::{
    detector::{RawBlock, RawDetection},
    mask,
    resize::round_half_away,
    yolo::{self, LetterboxGeometry},
};
use image::{GrayImage, Rgb, RgbImage};
use pc_core::StageError;
use std::{fmt::Write as _, path::Path};

#[cfg(feature = "onnx")]
use crate::detector::TextDetector;
#[cfg(feature = "onnx")]
use std::{path::PathBuf, sync::Mutex};

/// Network input side length (spec §8.3 step 3).
pub const NET_SIZE: u32 = 1024;
/// Letterbox stride named by the upstream call. It is inert because `auto = false`.
pub const STRIDE: u32 = 64;
/// Right/bottom letterbox fill value (spec §8.3 step 3).
pub const PAD_VALUE: u8 = 114;
/// CPU ONNX Runtime intra-op thread count (spec §8.3 step 3).
pub const INTRA_THREADS: usize = 1;

/// An RGB network input and the geometry needed to map its outputs back to the base image.
#[derive(Debug, Clone)]
pub struct Letterboxed {
    pub image: RgbImage,
    pub geometry: LetterboxGeometry,
}

/// Letterbox an RGB image to the fixed square network input.
#[allow(clippy::assertions_on_constants, clippy::manual_is_multiple_of)]
pub fn letterbox(image: &RgbImage) -> Letterboxed {
    debug_assert!(NET_SIZE % STRIDE == 0);
    let (width, height) = image.dimensions();

    let (new_width, new_height) = if width == 0 || height == 0 {
        (1, 1)
    } else {
        let ratio =
            (f64::from(NET_SIZE) / f64::from(height)).min(f64::from(NET_SIZE) / f64::from(width));

        // DEVIATION(14): upstream would pass a zero-sized dimension to cv2.resize for
        // sufficiently small inputs. The lower clamp keeps the resize valid; without it,
        // dw or dh could reach 1024 and make mask::crop_letterbox return InvalidInput.
        let new_width =
            round_half_away(f64::from(width) * ratio).clamp(1, i64::from(NET_SIZE)) as u32;
        let new_height =
            round_half_away(f64::from(height) * ratio).clamp(1, i64::from(NET_SIZE)) as u32;
        (new_width, new_height)
    };

    let resized = resize_bilinear_rgb(image, new_width, new_height);
    let dw = NET_SIZE - new_width;
    let dh = NET_SIZE - new_height;
    let mut boxed = RgbImage::from_pixel(NET_SIZE, NET_SIZE, Rgb([PAD_VALUE; 3]));
    for y in 0..new_height {
        for x in 0..new_width {
            boxed.put_pixel(x, y, *resized.get_pixel(x, y));
        }
    }

    Letterboxed {
        image: boxed,
        geometry: LetterboxGeometry {
            net_size: NET_SIZE,
            dw: dw as f32,
            dh: dh as f32,
            image_size: (width, height),
        },
    }
}

/// RGB twin of [`mask::resize_bilinear`], using the pinned half-pixel convention.
pub fn resize_bilinear_rgb(image: &RgbImage, new_width: u32, new_height: u32) -> RgbImage {
    if image.dimensions() == (new_width, new_height) {
        return image.clone();
    }
    if new_width == 0 || new_height == 0 || image.width() == 0 || image.height() == 0 {
        return RgbImage::new(new_width, new_height);
    }

    let scale_x = image.width() as f32 / new_width as f32;
    let scale_y = image.height() as f32 / new_height as f32;
    RgbImage::from_fn(new_width, new_height, |x, y| {
        let source_x =
            ((x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, image.width().saturating_sub(1) as f32);
        let source_y =
            ((y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, image.height().saturating_sub(1) as f32);
        let x0 = source_x.floor() as u32;
        let y0 = source_y.floor() as u32;
        let x1 = (x0 + 1).min(image.width() - 1);
        let y1 = (y0 + 1).min(image.height() - 1);
        let weight_x = source_x - x0 as f32;
        let weight_y = source_y - y0 as f32;
        let top_left = image.get_pixel(x0, y0).0;
        let top_right = image.get_pixel(x1, y0).0;
        let bottom_left = image.get_pixel(x0, y1).0;
        let bottom_right = image.get_pixel(x1, y1).0;

        Rgb([
            interpolate_channel(
                top_left[0],
                top_right[0],
                bottom_left[0],
                bottom_right[0],
                weight_x,
                weight_y,
            ),
            interpolate_channel(
                top_left[1],
                top_right[1],
                bottom_left[1],
                bottom_right[1],
                weight_x,
                weight_y,
            ),
            interpolate_channel(
                top_left[2],
                top_right[2],
                bottom_left[2],
                bottom_right[2],
                weight_x,
                weight_y,
            ),
        ])
    })
}

fn interpolate_channel(
    top_left: u8,
    top_right: u8,
    bottom_left: u8,
    bottom_right: u8,
    weight_x: f32,
    weight_y: f32,
) -> u8 {
    let top = f32::from(top_left) * (1.0 - weight_x) + f32::from(top_right) * weight_x;
    let bottom = f32::from(bottom_left) * (1.0 - weight_x) + f32::from(bottom_right) * weight_x;
    (top * (1.0 - weight_y) + bottom * weight_y)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Convert RGB HWC bytes to planar NCHW floats, normalized by exactly 255.0.
pub fn to_nchw(image: &RgbImage) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let plane_len = width as usize * height as usize;
    let mut tensor = vec![0.0; 3 * plane_len];
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y).0;
            let index = y as usize * width as usize + x as usize;
            tensor[index] = f32::from(pixel[0]) / 255.0;
            tensor[plane_len + index] = f32::from(pixel[1]) / 255.0;
            tensor[2 * plane_len + index] = f32::from(pixel[2]) / 255.0;
        }
    }
    tensor
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputMeta {
    pub name: String,
    pub shape: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputBinding {
    pub blks: usize,
    pub mask: usize,
    pub lines_map: usize,
}

impl OutputBinding {
    pub const BY_INDEX: Self = Self {
        blks: 0,
        mask: 1,
        lines_map: 2,
    };
}

/// Validate the three indexed model outputs and apply the mask/lines-map order guard.
pub fn bind_outputs(outputs: &[OutputMeta]) -> Result<OutputBinding, StageError> {
    if outputs.len() != 3 {
        return Err(StageError::Model(format!(
            "expected exactly 3 model outputs, got {}",
            outputs.len()
        )));
    }

    let actual_columns = outputs[0].shape.last().copied().unwrap_or(-1);
    if actual_columns != yolo::ROW_STRIDE as i64 {
        return Err(StageError::Model(format!(
            "blk output has {} columns, expected {}; {}",
            actual_columns,
            yolo::ROW_STRIDE,
            describe_outputs(outputs)
        )));
    }

    let first_channels = concrete_positive_channel(&outputs[1]);
    let second_channels = concrete_positive_channel(&outputs[2]);
    match (first_channels, second_channels) {
        (Some(1), Some(2)) => Ok(OutputBinding::BY_INDEX),
        (Some(2), Some(1)) => Ok(OutputBinding {
            blks: 0,
            mask: 2,
            lines_map: 1,
        }),
        (Some(first), Some(second)) => Err(StageError::Model(format!(
            "mask/lines outputs have unsupported channel pair ({first}, {second}); expected (1, 2) or (2, 1); {}",
            describe_outputs(outputs)
        ))),
        // Dynamic dimensions are inconclusive: retain the documented index binding.
        _ => Ok(OutputBinding::BY_INDEX),
    }
}

fn concrete_positive_channel(output: &OutputMeta) -> Option<i64> {
    output
        .shape
        .get(1)
        .copied()
        .filter(|channels| *channels > 0)
}

/// Render output names and shapes for diagnostics, including failed bindings.
pub fn describe_outputs(outputs: &[OutputMeta]) -> String {
    let mut rendered = String::from("outputs:");
    for (index, output) in outputs.iter().enumerate() {
        let _ = write!(rendered, " [{index}] {} {:?}", output.name, output.shape);
    }
    rendered
}

/// Decode the network mask and return it at the original base-image dimensions.
pub fn decode_mask(values: &[f32], geometry: &LetterboxGeometry) -> Result<GrayImage, StageError> {
    let expected_len = u64::from(geometry.net_size)
        .checked_mul(u64::from(geometry.net_size))
        .and_then(|length| usize::try_from(length).ok())
        .ok_or_else(|| StageError::InvalidInput("mask dimensions are too large".into()))?;
    if values.len() != expected_len {
        return Err(StageError::InvalidInput(format!(
            "mask has {} samples, expected {expected_len}",
            values.len()
        )));
    }
    if !geometry.dw.is_finite()
        || !geometry.dh.is_finite()
        || geometry.dw < 0.0
        || geometry.dh < 0.0
    {
        return Err(StageError::InvalidInput(
            "letterbox padding must be finite and non-negative".into(),
        ));
    }

    let processed = mask::postprocess_mask(values, geometry.net_size, geometry.net_size)?;
    let cropped = mask::crop_letterbox(&processed, geometry.dw as u32, geometry.dh as u32)?;
    Ok(mask::resize_bilinear(
        &cropped,
        geometry.image_size.0,
        geometry.image_size.1,
    ))
}

/// Validate and decode flattened YOLO block rows using the frozen D5 implementation.
pub fn decode_blocks(
    rows: &[f32],
    columns: usize,
    geometry: &LetterboxGeometry,
) -> Result<Vec<RawBlock>, StageError> {
    if columns != yolo::ROW_STRIDE {
        return Err(StageError::InvalidInput(format!(
            "YOLO block tensor has {} columns, expected {}",
            columns,
            yolo::ROW_STRIDE
        )));
    }
    if !rows.len().is_multiple_of(columns) {
        return Err(StageError::InvalidInput(format!(
            "YOLO block tensor has {} values, not a whole number of rows of {} columns",
            rows.len(),
            columns
        )));
    }
    Ok(yolo::postprocess(rows, geometry))
}

/// Ensure a model path names a regular file before any runtime initialization.
pub fn ensure_model_file(path: &Path) -> Result<(), StageError> {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(()),
        Ok(_) | Err(_) => Err(StageError::Model(format!(
            "model path is missing or is not a regular file: {}",
            path.display()
        ))),
    }
}

#[cfg(feature = "onnx")]
use ort::{
    session::{builder::GraphOptimizationLevel, Session, SessionOutputs},
    value::{Outlet, Tensor},
};

#[cfg(feature = "onnx")]
#[derive(Debug)]
pub struct OnnxDetector {
    model_path: PathBuf,
    // DEVIATION(15): `concurrent_models` would be honoured at provider construction;
    // v1 shares one Mutex-guarded session, and a configured value > 1 is warned-and-ignored.
    session: Mutex<Session>,
    outputs: Vec<OutputMeta>,
    output_names: Vec<String>,
    output_shapes: Vec<Vec<i64>>,
}

#[cfg(feature = "onnx")]
impl OnnxDetector {
    pub fn from_path(model: &Path) -> Result<Self, StageError> {
        // This pre-flight must precede every ort call so path errors remain actionable.
        ensure_model_file(model)?;

        let mut builder = Session::builder().map_err(|error| {
            StageError::Model(format!("failed to configure {}: {error}", model.display()))
        })?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| {
                StageError::Model(format!("failed to configure {}: {error}", model.display()))
            })?;
        builder = builder.with_intra_threads(INTRA_THREADS).map_err(|error| {
            StageError::Model(format!("failed to configure {}: {error}", model.display()))
        })?;
        builder = builder.with_inter_threads(INTRA_THREADS).map_err(|error| {
            StageError::Model(format!("failed to configure {}: {error}", model.display()))
        })?;
        // No execution provider is registered: ONNX Runtime's default CPU provider is
        // the CPU-only v1 decision.
        let session = builder.commit_from_file(model).map_err(|error| {
            StageError::Model(format!("failed to load {}: {error}", model.display()))
        })?;

        let inputs = session.inputs().len();
        if inputs != 1 {
            return Err(StageError::Model(format!(
                "model has {inputs} inputs, expected exactly 1"
            )));
        }

        let outputs = session
            .outputs()
            .iter()
            .map(outlet_meta)
            .collect::<Result<Vec<_>, _>>()?;
        tracing::debug!(
            model = %model.display(),
            outputs = %describe_outputs(&outputs),
            "ONNX model outputs"
        );
        if outputs.len() != 3 {
            return Err(StageError::Model(format!(
                "model has {} outputs, expected exactly 3; {}",
                outputs.len(),
                describe_outputs(&outputs)
            )));
        }
        let binding = bind_outputs(&outputs)?;
        validate_output_shapes(&outputs, binding)?;

        let output_names = outputs.iter().map(|output| output.name.clone()).collect();
        let output_shapes = outputs.iter().map(|output| output.shape.clone()).collect();
        let detector = Self {
            model_path: model.to_path_buf(),
            session: Mutex::new(session),
            outputs,
            output_names,
            output_shapes,
        };
        Ok(detector)
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    pub fn output_shapes(&self) -> &[Vec<i64>] {
        &self.output_shapes
    }

    pub fn outputs(&self) -> &[OutputMeta] {
        &self.outputs
    }
}

#[cfg(feature = "onnx")]
impl TextDetector for OnnxDetector {
    fn detect(&self, image: &RgbImage) -> Result<RawDetection, StageError> {
        let boxed = letterbox(image);
        let input = Tensor::<f32>::from_array((
            [1usize, 3, NET_SIZE as usize, NET_SIZE as usize],
            to_nchw(&boxed.image),
        ))
        .map_err(|error| {
            StageError::InvalidInput(format!("failed to create input tensor: {error}"))
        })?;

        let (metas, values) = {
            // Keep the guard alive through extraction because `SessionOutputs` borrows the
            // session; end this scope immediately after owned output data is extracted so
            // all pure validation and decoding runs outside the session's critical section.
            // The recovered poison guard is sound for ort rc.12: `Session` is only
            // `{ inner: Arc<SharedSessionInner>, inputs: Vec<Outlet>, outputs: Vec<Outlet> }`
            // (`session/mod.rs:114-118`), `Session::run` (`mod.rs:212`) immediately calls
            // `run_inner` (`mod.rs:272-273`), and that function only reads those fields.
            // A panic can leak un-Released `OrtValue`s (`mod.rs:329` onward), but cannot
            // leave Rust-side session state half-updated; the only Rust callbacks are the
            // `extern "system"` logging callbacks (`logging.rs:108/139`), whose unwinds
            // abort rather than poison this mutex.
            let mut session = self.session.lock().unwrap_or_else(|poisoned| {
                tracing::warn!(
                    model = %self.model_path.display(),
                    "a previous ONNX inference panicked inside the session lock; reusing the \
                     session (ort rc.12 `Session::run` mutates no Rust-side session state)"
                );
                poisoned.into_inner()
            });
            let runtime_outputs = session.run(ort::inputs![input]).map_err(|error| {
                StageError::Inference(format!("ONNX session run failed: {error}"))
            })?;
            extract_outputs(&runtime_outputs)?
        };

        decode_outputs(&metas, &values, &boxed.geometry)
    }
}

#[cfg(feature = "onnx")]
fn extract_outputs(
    runtime_outputs: &SessionOutputs<'_>,
) -> Result<(Vec<OutputMeta>, Vec<Vec<f32>>), StageError> {
    let mut metas = Vec::with_capacity(runtime_outputs.len());
    let mut values = Vec::with_capacity(runtime_outputs.len());
    for (name, output) in runtime_outputs.iter() {
        let (shape, tensor_values) = output.try_extract_tensor::<f32>().map_err(|error| {
            StageError::Inference(format!(
                "failed to extract ONNX output `{name}` as f32: {error}"
            ))
        })?;
        let expected_len = shape_len(shape).map_err(StageError::InvalidInput)?;
        if tensor_values.len() != expected_len {
            return Err(StageError::InvalidInput(format!(
                "ONNX output `{name}` has {} values for shape {shape:?}, expected {expected_len}",
                tensor_values.len()
            )));
        }
        metas.push(OutputMeta {
            name: name.to_string(),
            shape: shape.to_vec(),
        });
        values.push(tensor_values.to_vec());
    }

    Ok((metas, values))
}

/// Decode extracted ONNX outputs after the session mutex has been released.
pub fn decode_outputs(
    metas: &[OutputMeta],
    values: &[Vec<f32>],
    geometry: &LetterboxGeometry,
) -> Result<RawDetection, StageError> {
    if values.len() != metas.len() {
        return Err(StageError::InvalidInput(format!(
            "ONNX output metadata/value count mismatch: {} metas, {} values",
            metas.len(),
            values.len()
        )));
    }

    let binding = bind_outputs(metas)?;
    for (name, index) in [
        ("blocks", binding.blks),
        ("mask", binding.mask),
        ("lines_map", binding.lines_map),
    ] {
        if index >= values.len() {
            return Err(StageError::InvalidInput(format!(
                "{name} output binding index {index} is out of range for {} values",
                values.len()
            )));
        }
    }
    validate_output_shapes(metas, binding)?;
    let blocks = decode_blocks(
        &values[binding.blks],
        metas[binding.blks]
            .shape
            .last()
            .and_then(|columns| usize::try_from(*columns).ok())
            .unwrap_or(0),
        geometry,
    )?;
    let mask = decode_mask(&values[binding.mask], geometry)?;
    // `lines_map` is bound for the output-order swap guard, but DBNet line polygons
    // are deliberately deferred in v1 (§8.3).
    let _lines_map = binding.lines_map;

    Ok(RawDetection { blocks, mask })
}

#[cfg(feature = "onnx")]
pub fn runtime_available() -> bool {
    // This probes the usable ort environment without claiming a process-global one via
    // `ort::init()`. A missing shared library can still fail at link time, before this
    // runtime probe is reachable.
    ort::environment::current().is_ok()
}

#[cfg(feature = "onnx")]
fn outlet_meta(outlet: &Outlet) -> Result<OutputMeta, StageError> {
    let shape = outlet.dtype().tensor_shape().ok_or_else(|| {
        StageError::Model(format!(
            "output `{}` has no tensor shape, expected a tensor output",
            outlet.name()
        ))
    })?;
    Ok(OutputMeta {
        name: outlet.name().to_string(),
        shape: shape.to_vec(),
    })
}

#[cfg(feature = "onnx")]
fn shape_len(shape: &[i64]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |length, dimension| {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| format!("tensor shape has invalid dimension in {shape:?}"))?;
        length
            .checked_mul(dimension)
            .ok_or_else(|| format!("tensor shape is too large: {shape:?}"))
    })
}

fn validate_output_shapes(
    outputs: &[OutputMeta],
    binding: OutputBinding,
) -> Result<(), StageError> {
    let blks = &outputs[binding.blks].shape;
    if blks.len() != 3 || blks[0] != 1 || blks[1] <= 0 || blks[2] != yolo::ROW_STRIDE as i64 {
        return Err(StageError::Model(format!(
            "blk output shape {blks:?}, expected [1, N, {}]; observed output `{}` bound from index {}; {}",
            yolo::ROW_STRIDE,
            outputs[binding.blks].name,
            binding.blks,
            describe_outputs(outputs)
        )));
    }

    let mask = &outputs[binding.mask].shape;
    let expected_mask = [1, 1, i64::from(NET_SIZE), i64::from(NET_SIZE)];
    if mask.as_slice() != expected_mask {
        return Err(StageError::Model(format!(
            "mask output shape {mask:?}, expected {expected_mask:?}; observed output `{}` bound from index {}; {}",
            outputs[binding.mask].name,
            binding.mask,
            describe_outputs(outputs)
        )));
    }

    let lines = &outputs[binding.lines_map].shape;
    let expected_lines = [1, 2, i64::from(NET_SIZE), i64::from(NET_SIZE)];
    if lines.as_slice() != expected_lines {
        return Err(StageError::Model(format!(
            "lines_map output shape {lines:?}, expected {expected_lines:?}; observed output `{}` bound from index {}; {}",
            outputs[binding.lines_map].name,
            binding.lines_map,
            describe_outputs(outputs)
        )));
    }
    Ok(())
}

#[cfg(test)]
mod decode_tests {
    use super::{decode_outputs, LetterboxGeometry, OutputMeta, NET_SIZE};
    use crate::yolo;
    use pc_core::StageError;

    #[test]
    fn decode_outputs_decodes_synthetic_tensors_and_rejects_mismatched_values() {
        let metas = vec![
            OutputMeta {
                name: "blk".into(),
                shape: vec![1, 1, yolo::ROW_STRIDE as i64],
            },
            OutputMeta {
                name: "seg".into(),
                shape: vec![1, 1, i64::from(NET_SIZE), i64::from(NET_SIZE)],
            },
            OutputMeta {
                name: "det".into(),
                shape: vec![1, 2, i64::from(NET_SIZE), i64::from(NET_SIZE)],
            },
        ];
        let mut block_values = vec![0.0; yolo::ROW_STRIDE];
        block_values[0] = 2.0;
        block_values[1] = 2.0;
        block_values[2] = 1.0;
        block_values[3] = 1.0;
        block_values[4] = 0.9;
        block_values[5] = 0.9;
        let values = vec![
            block_values,
            vec![0.0; NET_SIZE as usize * NET_SIZE as usize],
            vec![],
        ];
        let geometry = LetterboxGeometry {
            net_size: NET_SIZE,
            dw: 0.0,
            dh: 0.0,
            image_size: (4, 4),
        };

        let decoded = decode_outputs(&metas, &values, &geometry).expect("synthetic outputs decode");
        assert_eq!(decoded.blocks.len(), 1);
        assert_eq!(decoded.mask.dimensions(), (4, 4));

        let error = decode_outputs(&metas, &values[..2], &geometry)
            .expect_err("metadata/value mismatch must be an InvalidInput error");
        assert!(matches!(error, StageError::InvalidInput(_)));
    }
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::{validate_output_shapes, OutputBinding, OutputMeta, NET_SIZE};
    use crate::yolo;

    #[test]
    fn validate_output_shapes_reports_bound_output_name_and_full_picture() {
        let outputs = vec![
            OutputMeta {
                name: "det".into(),
                shape: vec![1, 10, yolo::ROW_STRIDE as i64],
            },
            OutputMeta {
                name: "seg".into(),
                shape: vec![1, 2, i64::from(NET_SIZE), i64::from(NET_SIZE)],
            },
            OutputMeta {
                name: "mask".into(),
                shape: vec![1, 2, i64::from(NET_SIZE), i64::from(NET_SIZE)],
            },
        ];

        let error = validate_output_shapes(&outputs, OutputBinding::BY_INDEX)
            .expect_err("the swapped segmentation output should fail mask validation");
        let message = error.to_string();
        assert!(message.contains("observed output `seg` bound from index 1"));
        assert!(message.contains("mask output shape [1, 2, 1024, 1024]"));
        assert!(message.contains(
            "outputs: [0] det [1, 10, 7] [1] seg [1, 2, 1024, 1024] [2] mask [1, 2, 1024, 1024]"
        ));
    }
}
