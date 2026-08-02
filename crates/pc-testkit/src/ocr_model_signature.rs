use crate::model_signature::{
    load_model_signature, Dim, SymbolicModelSignature, TypedSignaturePath,
};

/// A third-party ONNX artifact this repo records a graph signature for. `pc_models` also
/// carries a `ModelSpec` for the same file (registry membership landed as task P8a, spec
/// §16.31 item 2) — this type stays separate because it pins the graph signature, not the
/// download/cache identity.
pub struct OcrModelPin {
    pub file_name: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
}

/// spec §16.30 item 1 (ratified artifact pin), `mayocream/manga-ocr-onnx` @
/// `24b12778d85800835e2ca409236de281b8ab7b9f`.
pub const MANGA_OCR_ENCODER: OcrModelPin = OcrModelPin {
    file_name: "encoder_model.onnx",
    sha256: "15fa8155fe9bc1a7d25d9bb353debaa4def033d0174e907dbd2dd6d995def85f",
    size_bytes: 343_454_249,
};
pub const MANGA_OCR_DECODER: OcrModelPin = OcrModelPin {
    file_name: "decoder_model.onnx",
    sha256: "ef7765261e9d1cdc34d89356986c2bbc2a082897f753a89605ae80fdfa61f5e8",
    size_bytes: 117_480_262,
};

pub const MANGA_OCR_ENCODER_SIGNATURE: &str = "ocr_model_signature/encoder_model.signature.json";
pub const MANGA_OCR_DECODER_SIGNATURE: &str = "ocr_model_signature/decoder_model.signature.json";

pub fn manga_ocr_encoder_signature() -> SymbolicModelSignature {
    load_model_signature::<Dim>(TypedSignaturePath::new(MANGA_OCR_ENCODER_SIGNATURE))
}

pub fn manga_ocr_decoder_signature() -> SymbolicModelSignature {
    load_model_signature::<Dim>(TypedSignaturePath::new(MANGA_OCR_DECODER_SIGNATURE))
}
