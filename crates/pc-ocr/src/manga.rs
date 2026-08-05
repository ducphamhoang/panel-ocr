//! The manga-ocr ONNX engine and language-agnostic factory.

#![cfg(feature = "onnx")]

use crate::{decode, onnx, post, preprocess, vocab, OcrEngine, OcrEngineFactory};
use image::DynamicImage;
use pc_core::{device::Device, Language, StageError};
use std::path::Path;

#[derive(Debug)]
pub struct MangaOcrEngine {
    sessions: onnx::MangaOcrSessions,
}

impl MangaOcrEngine {
    pub fn from_paths(encoder: &Path, decoder: &Path) -> Result<Self, StageError> {
        Self::from_paths_for_device(encoder, decoder, Device::Cpu)
    }

    pub fn from_paths_for_device(
        encoder: &Path,
        decoder: &Path,
        device: Device,
    ) -> Result<Self, StageError> {
        Ok(Self {
            sessions: onnx::MangaOcrSessions::from_paths_for_device(encoder, decoder, device)?,
        })
    }
}

impl OcrEngine for MangaOcrEngine {
    fn languages(&self) -> &[Language] {
        &[Language::Japanese, Language::English]
    }

    fn recognize(&self, crop: &DynamicImage) -> Result<String, StageError> {
        if crop.width() == 0 || crop.height() == 0 {
            return Err(StageError::InvalidInput(
                "OCR crop must have non-zero dimensions".into(),
            ));
        }
        let pixel_values = preprocess::pixel_values(crop);
        let encoder_output = self.sessions.encode(&pixel_values)?;

        struct DecoderLogitsSource<'a> {
            sessions: &'a onnx::MangaOcrSessions,
            encoder_output: &'a onnx::EncoderOutput,
        }

        impl<'a> decode::LogitsSource for DecoderLogitsSource<'a> {
            fn logits(&self, prefix: &[u32]) -> Result<Vec<f32>, StageError> {
                self.sessions.decode_step(prefix, self.encoder_output)
            }
        }

        let source = DecoderLogitsSource {
            sessions: &self.sessions,
            encoder_output: &encoder_output,
        };
        let ids = decode::beam_search(&source, &decode::MANGA_OCR_BEAM_CONFIG)?;
        let text = vocab::Vocab::embedded().decode_skip_special(&ids);
        Ok(post::post_process(&text))
    }
}

pub struct MangaOcrFactory {
    engine: MangaOcrEngine,
}

impl MangaOcrFactory {
    pub fn new(engine: MangaOcrEngine) -> Self {
        Self { engine }
    }
}

impl OcrEngineFactory for MangaOcrFactory {
    fn engine_for(&self, _lang: Option<Language>) -> Option<&dyn OcrEngine> {
        Some(&self.engine)
    }
}
