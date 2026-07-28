//! The recorded ONNX graph signature used by task F3 (spec §16.16).

use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ELEMENT_TYPE_F32: &str = "f32";
pub const COMIC_TEXT_DETECTOR_SIGNATURE: &str = "model_signature/comictextdetector.signature.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSignature {
    pub schema_version: u32,
    pub model_file_name: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub opset: i64,
    pub inputs: Vec<TensorSignature>,
    pub outputs: Vec<TensorSignature>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSignature {
    pub name: String,
    pub element_type: String,
    pub shape: Vec<i64>,
}

impl TensorSignature {
    pub fn is_f32(&self) -> bool {
        self.element_type == ELEMENT_TYPE_F32
    }
}

impl ModelSignature {
    pub fn input(&self, name: &str) -> &TensorSignature {
        self.inputs
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| {
                let recorded = self
                    .inputs
                    .iter()
                    .map(|tensor| tensor.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                panic!("unknown input `{name}`; recorded input names: [{recorded}]")
            })
    }

    pub fn output(&self, name: &str) -> &TensorSignature {
        self.outputs
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| {
                let recorded = self
                    .outputs
                    .iter()
                    .map(|tensor| tensor.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                panic!("unknown output `{name}`; recorded output names: [{recorded}]")
            })
    }
}

pub fn load_model_signature(rel: impl AsRef<Path>) -> ModelSignature {
    let path = crate::paths::recorded(rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "failed to read recorded model signature `{}`: {error}",
            path.display()
        )
    });
    serde_json::from_slice(&bytes).unwrap_or_else(|error| {
        panic!(
            "failed to deserialize recorded model signature `{}`: {error}",
            path.display()
        )
    })
}

pub fn comic_text_detector_signature() -> ModelSignature {
    load_model_signature(COMIC_TEXT_DETECTOR_SIGNATURE)
}
