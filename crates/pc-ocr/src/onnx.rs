//! ONNX session construction for the manga-ocr backend (task P7c).
//!
//! This module deliberately stops at session construction and graph I/O metadata. Inference,
//! decoding, and `OcrEngine` wiring belong to later P7 tasks.

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
    value::{Outlet, TensorElementType, ValueType},
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
pub struct MangaOcrSessions {
    // DEVIATION(15): ort rc.12's session execution takes `&mut self`; later inference code will
    // therefore use these Mutex guards to preserve a shareable OCR backend.
    #[allow(dead_code)]
    encoder: Mutex<Session>,
    #[allow(dead_code)]
    decoder: Mutex<Session>,
    encoder_inputs: Vec<TensorMeta>,
    encoder_outputs: Vec<TensorMeta>,
    decoder_inputs: Vec<TensorMeta>,
    decoder_outputs: Vec<TensorMeta>,
}

#[cfg(feature = "onnx")]
impl MangaOcrSessions {
    pub fn from_paths(encoder: &Path, decoder: &Path) -> Result<Self, StageError> {
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
