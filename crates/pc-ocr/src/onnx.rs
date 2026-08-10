//! ONNX session construction for the manga-ocr backend (task P7c).
//!
//! This module owns manga-ocr ONNX session construction and inference.

#[cfg(feature = "onnx")]
use pc_core::device::{Device, DeviceSupport};
use pc_core::StageError;
use std::path::Path;

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
use std::sync::Mutex;

#[cfg(feature = "onnx")]
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::{Outlet, Tensor, TensorElementType, ValueType},
};

#[cfg(feature = "onnx")]
/// Whether ONNX Runtime's shared environment can be obtained.
pub fn runtime_available() -> bool {
    // This probes the usable ort environment without claiming a process-global one via
    // `ort::init()`. A missing shared library can still fail at link time, before this
    // runtime probe is reachable.
    ort::environment::current().is_ok()
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<i64>,
    pub symbols: Vec<String>,
    pub element_type: String,
}

#[cfg(feature = "onnx")]
#[derive(Debug)]
pub struct EncoderOutput {
    pub hidden_states: Vec<f32>,
    pub shape: Vec<i64>,
}

#[cfg(feature = "onnx")]
#[derive(Debug)]
pub struct MangaOcrSessions {
    // DEVIATION(15): ort rc.12's session execution takes `&mut self`; `encode`/`decode_step`
    // below use these Mutex guards to preserve a shareable OCR backend.
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    encoder_inputs: Vec<TensorMeta>,
    encoder_outputs: Vec<TensorMeta>,
    decoder_inputs: Vec<TensorMeta>,
    decoder_outputs: Vec<TensorMeta>,
}

#[cfg(feature = "onnx")]
impl MangaOcrSessions {
    pub fn from_paths(encoder: &Path, decoder: &Path) -> Result<Self, StageError> {
        Self::from_paths_for_device(encoder, decoder, Device::Cpu)
    }

    pub fn from_paths_for_device(
        encoder: &Path,
        decoder: &Path,
        device: Device,
    ) -> Result<Self, StageError> {
        let policy = pc_core::device::resolve(device, DeviceSupport::compiled())
            .map_err(|refusal| StageError::Model(refusal.message()))?;
        // §16.47 item 4: the OCR stage has no ratified CUDA path, so even a build that
        // CAN register CUDA refuses here — AFTER resolve, so the GPU-1 NotCompiledIn
        // refusal (and its frozen message) keeps precedence in a non-cuda build.
        pc_core::device::ensure_stage_supports(&policy, pc_core::device::Stage::Ocr)
            .map_err(|refusal| StageError::Model(refusal.message()))?;

        // Both pre-flights must happen before any ort call. In particular, this preserves the
        // encoder-specific error when both paths are absent and keeps garbage decoder bytes from
        // obscuring a missing encoder.
        ensure_model_file(encoder)?;
        ensure_model_file(decoder)?;

        let encoder_session = build_session(encoder)?;
        let decoder_session = build_session(decoder)?;
        let encoder_inputs = collect_metas(encoder_session.inputs())?;
        let encoder_outputs = collect_metas(encoder_session.outputs())?;
        let decoder_inputs = collect_metas(decoder_session.inputs())?;
        let decoder_outputs = collect_metas(decoder_session.outputs())?;

        Ok(Self {
            encoder: Mutex::new(encoder_session),
            decoder: Mutex::new(decoder_session),
            encoder_inputs,
            encoder_outputs,
            decoder_inputs,
            decoder_outputs,
        })
    }

    pub fn encoder_inputs(&self) -> &[TensorMeta] {
        &self.encoder_inputs
    }

    pub fn encoder_outputs(&self) -> &[TensorMeta] {
        &self.encoder_outputs
    }

    pub fn decoder_inputs(&self) -> &[TensorMeta] {
        &self.decoder_inputs
    }

    pub fn decoder_outputs(&self) -> &[TensorMeta] {
        &self.decoder_outputs
    }

    /// Run one encoder forward pass and retain the runtime output shape for the decoder.
    pub fn encode(&self, pixel_values: &[f32]) -> Result<EncoderOutput, StageError> {
        let input = Tensor::<f32>::from_array((
            [
                1_usize,
                3,
                crate::preprocess::INPUT_SIZE as usize,
                crate::preprocess::INPUT_SIZE as usize,
            ],
            pixel_values.to_vec(),
        ))
        .map_err(|error| {
            StageError::Inference(format!("failed to create encoder input tensor: {error}"))
        })?;

        let (shape, hidden_states) = {
            let mut session = self.encoder.lock().unwrap_or_else(|poisoned| {
                tracing::warn!(
                    "a previous ONNX encoder inference panicked inside the session lock; reusing the session"
                );
                poisoned.into_inner()
            });
            let runtime_outputs =
                session
                    .run(ort::inputs!["pixel_values" => input])
                    .map_err(|error| {
                        StageError::Inference(format!("ONNX encoder session run failed: {error}"))
                    })?;
            let output = runtime_outputs.get("last_hidden_state").ok_or_else(|| {
                StageError::Inference(
                    "ONNX encoder output `last_hidden_state` was not returned".into(),
                )
            })?;
            let (shape, values) = output.try_extract_tensor::<f32>().map_err(|error| {
                StageError::Inference(format!(
                    "failed to extract ONNX encoder output `last_hidden_state` as f32: {error}"
                ))
            })?;
            (shape.to_vec(), values.to_vec())
        };

        validate_values_for_shape(&shape, hidden_states.len(), "encoder output")?;
        if shape.len() != 3 {
            return Err(StageError::Inference(format!(
                "ONNX encoder output `last_hidden_state` has shape {shape:?}, expected [batch, seq_len, hidden]"
            )));
        }

        Ok(EncoderOutput {
            hidden_states,
            shape,
        })
    }

    /// Run one decoder forward pass and return the logits for its final sequence position.
    pub fn decode_step(
        &self,
        input_ids: &[u32],
        encoder: &EncoderOutput,
    ) -> Result<Vec<f32>, StageError> {
        if input_ids.is_empty() {
            return Err(StageError::Inference(
                "decoder input_ids must contain at least one token".into(),
            ));
        }
        if encoder.shape.len() != 3 {
            return Err(StageError::Inference(format!(
                "encoder hidden-state shape is {:?}, expected [batch, seq_len, hidden]",
                encoder.shape
            )));
        }
        validate_values_for_shape(
            &encoder.shape,
            encoder.hidden_states.len(),
            "encoder hidden states",
        )?;
        let encoder_shape = encoder
            .shape
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension).map_err(|_| {
                    StageError::Inference(format!(
                        "encoder hidden-state shape has invalid dimension: {:?}",
                        encoder.shape
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ids_tensor = Tensor::<i64>::from_array((
            [1_usize, input_ids.len()],
            input_ids.iter().map(|&id| id as i64).collect::<Vec<i64>>(),
        ))
        .map_err(|error| {
            StageError::Inference(format!("failed to create decoder input tensor: {error}"))
        })?;
        let hidden_tensor =
            Tensor::<f32>::from_array((encoder_shape, encoder.hidden_states.clone())).map_err(
                |error| {
                    StageError::Inference(format!(
                        "failed to create decoder encoder-hidden-state tensor: {error}"
                    ))
                },
            )?;

        let (shape, values) = {
            let mut session = self.decoder.lock().unwrap_or_else(|poisoned| {
                tracing::warn!(
                    "a previous ONNX decoder inference panicked inside the session lock; reusing the session"
                );
                poisoned.into_inner()
            });
            let runtime_outputs = session
                .run(ort::inputs!["input_ids" => ids_tensor, "encoder_hidden_states" => hidden_tensor])
                .map_err(|error| {
                    StageError::Inference(format!("ONNX decoder session run failed: {error}"))
                })?;
            let output = runtime_outputs.get("logits").ok_or_else(|| {
                StageError::Inference("ONNX decoder output `logits` was not returned".into())
            })?;
            let (shape, values) = output.try_extract_tensor::<f32>().map_err(|error| {
                StageError::Inference(format!(
                    "failed to extract ONNX decoder output `logits` as f32: {error}"
                ))
            })?;
            (shape.to_vec(), values.to_vec())
        };

        if shape.len() != 3 || shape[0] != 1 {
            return Err(StageError::Inference(format!(
                "ONNX decoder output `logits` has shape {shape:?}, expected [1, seq_len, vocab]"
            )));
        }
        let seq_len = usize::try_from(shape[1]).map_err(|_| {
            StageError::Inference(format!(
                "ONNX decoder output `logits` has invalid sequence dimension: {shape:?}"
            ))
        })?;
        let vocab = usize::try_from(shape[2]).map_err(|_| {
            StageError::Inference(format!(
                "ONNX decoder output `logits` has invalid vocabulary dimension: {shape:?}"
            ))
        })?;
        if seq_len != input_ids.len() || vocab == 0 {
            return Err(StageError::Inference(format!(
                "ONNX decoder output `logits` has shape {shape:?}, expected [1, {}, vocab > 0]",
                input_ids.len()
            )));
        }
        validate_values_for_shape(&shape, values.len(), "decoder logits")?;
        let start = (seq_len - 1) * vocab;
        Ok(values[start..start + vocab].to_vec())
    }
}

#[cfg(feature = "onnx")]
fn validate_values_for_shape(
    shape: &[i64],
    values_len: usize,
    tensor_name: &str,
) -> Result<(), StageError> {
    let expected_len = shape.iter().try_fold(1usize, |length, &dimension| {
        let dimension = usize::try_from(dimension).map_err(|_| {
            StageError::Inference(format!("{tensor_name} has invalid shape {shape:?}"))
        })?;
        if dimension == 0 {
            return Err(StageError::Inference(format!(
                "{tensor_name} has invalid shape {shape:?}"
            )));
        }
        length.checked_mul(dimension).ok_or_else(|| {
            StageError::Inference(format!("{tensor_name} shape is too large: {shape:?}"))
        })
    })?;
    if values_len != expected_len {
        return Err(StageError::Inference(format!(
            "{tensor_name} has {values_len} values for shape {shape:?}, expected {expected_len}"
        )));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn build_session(path: &Path) -> Result<Session, StageError> {
    let mut builder = Session::builder().map_err(|error| {
        StageError::Model(format!("failed to configure {}: {error}", path.display()))
    })?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|error| {
            StageError::Model(format!("failed to configure {}: {error}", path.display()))
        })?;
    builder = builder.with_intra_threads(0).map_err(|error| {
        StageError::Model(format!("failed to configure {}: {error}", path.display()))
    })?;
    builder = builder.with_inter_threads(0).map_err(|error| {
        StageError::Model(format!("failed to configure {}: {error}", path.display()))
    })?;
    // No execution provider is registered: ONNX Runtime's default CPU provider is used.
    builder
        .commit_from_file(path)
        .map_err(|error| StageError::Model(format!("failed to load {}: {error}", path.display())))
}

#[cfg(feature = "onnx")]
fn collect_metas(outlets: &[Outlet]) -> Result<Vec<TensorMeta>, StageError> {
    outlets.iter().map(tensor_meta).collect()
}

#[cfg(feature = "onnx")]
fn tensor_meta(outlet: &Outlet) -> Result<TensorMeta, StageError> {
    let ValueType::Tensor {
        ty,
        shape,
        dimension_symbols,
    } = outlet.dtype()
    else {
        return Err(StageError::Model(format!(
            "tensor `{}` has no tensor value type",
            outlet.name()
        )));
    };

    let shape = shape.to_vec();
    let symbols = shape
        .iter()
        .enumerate()
        .map(|(index, dimension)| {
            if *dimension == -1 {
                dimension_symbols[index].clone()
            } else {
                String::new()
            }
        })
        .collect();
    let element_type = match ty {
        TensorElementType::Float32 => "f32".to_owned(),
        TensorElementType::Int64 => "i64".to_owned(),
        _ => {
            return Err(StageError::Model(format!(
                "tensor `{}` has unsupported element type {ty:?}; expected f32 or i64",
                outlet.name()
            )));
        }
    };

    Ok(TensorMeta {
        name: outlet.name().to_owned(),
        shape,
        symbols,
        element_type,
    })
}

/// Render tensor names, element types, raw shapes, and symbolic dimensions for diagnostics.
#[cfg(feature = "onnx")]
#[allow(dead_code)]
pub fn describe(metas: &[TensorMeta]) -> String {
    use std::fmt::Write as _;

    let mut rendered = String::from("tensors:");
    for (index, meta) in metas.iter().enumerate() {
        let _ = write!(
            rendered,
            " [{index}] {} {} shape={:?} symbols={:?}",
            meta.name, meta.element_type, meta.shape, meta.symbols
        );
    }
    rendered
}
