//! `pc-ocr` — the OCR boundary (spec §9.2).
//!
//! This crate deliberately contains **no stage logic**. It owns exactly the seam that
//! keeps the `candle`/model escape hatch real (§1 rule 4):
//!
//!   * [`OcrEngine`] — recognise text in one cropped box
//!   * [`OcrEngineFactory`] — pick an engine for a (possibly unknown) language
//!
//! `pc-preprocess` depends on this crate for the traits only; the concrete engine is
//! injected as the stage's `Ctx` (§3). The manga-ocr ONNX backend is task **P7**, behind
//! the (declared, unimplemented) `onnx` feature — see spec §16.8 item 1.

pub mod engine;
pub mod post;
pub mod vocab;

#[cfg(any(test, feature = "testkit"))]
pub mod mock;

pub use engine::{OcrEngine, OcrEngineFactory};

#[cfg(any(test, feature = "testkit"))]
pub use mock::{MockOcrEngine, MockOcrFactory};
