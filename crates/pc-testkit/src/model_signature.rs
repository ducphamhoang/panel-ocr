//! The recorded ONNX graph signature used by task F3 (spec §16.16).

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

pub const ELEMENT_TYPE_F32: &str = "f32";
pub const ELEMENT_TYPE_I64: &str = "i64";
pub const COMIC_TEXT_DETECTOR_SIGNATURE: &str = "model_signature/comictextdetector.signature.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dim {
    Fixed(i64),
    Symbolic(String),
}

impl Dim {
    pub fn fixed(&self) -> Option<i64> {
        match self {
            Self::Fixed(value) => Some(*value),
            Self::Symbolic(_) => None,
        }
    }

    pub fn symbol(&self) -> Option<&str> {
        match self {
            Self::Symbolic(symbol) => Some(symbol),
            Self::Fixed(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSig<D> {
    pub name: String,
    pub element_type: String,
    pub shape: Vec<D>,
}

impl<D> TensorSig<D> {
    pub fn is_f32(&self) -> bool {
        self.element_type == ELEMENT_TYPE_F32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSig<D> {
    pub schema_version: u32,
    pub model_file_name: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub opset: i64,
    pub inputs: Vec<TensorSig<D>>,
    pub outputs: Vec<TensorSig<D>>,
}

impl<D> ModelSig<D> {
    pub fn input(&self, name: &str) -> &TensorSig<D> {
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

    pub fn output(&self, name: &str) -> &TensorSig<D> {
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

pub type TensorSignature = TensorSig<i64>;
pub type ModelSignature = ModelSig<i64>;
pub type SymbolicTensorSignature = TensorSig<Dim>;
pub type SymbolicModelSignature = ModelSig<Dim>;

pub trait SignatureInput<D> {}

impl SignatureInput<i64> for &Path {}
impl SignatureInput<i64> for &PathBuf {}
impl SignatureInput<i64> for PathBuf {}
impl SignatureInput<i64> for &str {}
impl SignatureInput<i64> for String {}

pub struct TypedSignaturePath<D, P> {
    path: P,
    dimension: PhantomData<D>,
}

impl<D, P> TypedSignaturePath<D, P> {
    pub fn new(path: P) -> Self {
        Self {
            path,
            dimension: PhantomData,
        }
    }
}

impl<D, P: AsRef<Path>> AsRef<Path> for TypedSignaturePath<D, P> {
    fn as_ref(&self) -> &Path {
        self.path.as_ref()
    }
}

impl<D, P> SignatureInput<D> for TypedSignaturePath<D, P> {}

pub fn load_model_signature<D>(rel: impl AsRef<Path> + SignatureInput<D>) -> ModelSig<D>
where
    D: for<'de> Deserialize<'de>,
{
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
