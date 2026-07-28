//! spec §7.2 pattern applied to the OCR boundary — the mocked engine the frozen P6
//! tests are written against. Test-only (`cfg(any(test, feature = "testkit"))`), so per
//! `pc-testkit`'s panic policy misuse panics with a diagnostic rather than returning
//! `Result`.
//!
//! This is test infrastructure rather than shipped stage logic: everything else in Stage 2
//! is tested through it, and it is compiled out of a normal build.

use crate::engine::{OcrEngine, OcrEngineFactory};
use image::DynamicImage;
use pc_core::{Language, StageError};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// A programmable OCR engine: a scripted response queue, a fallback response, and an
/// optional hard failure. Interior mutability throughout, because `recognize` takes
/// `&self` (the trait is shared across threads, §4.5).
#[derive(Debug)]
pub struct MockOcrEngine {
    languages: Vec<Language>,
    script: Mutex<VecDeque<String>>,
    fallback: String,
    failing: bool,
    calls: AtomicUsize,
    crops: Mutex<Vec<(u32, u32)>>,
}

impl Default for MockOcrEngine {
    fn default() -> Self {
        Self {
            languages: vec![Language::Japanese, Language::English],
            script: Mutex::new(VecDeque::new()),
            fallback: String::new(),
            failing: false,
            calls: AtomicUsize::new(0),
            crops: Mutex::new(Vec::new()),
        }
    }
}

impl MockOcrEngine {
    /// Returns `""` for every crop — the §9.7(B)11 configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Same text for every call (used once the script, if any, is exhausted).
    pub fn with_response(mut self, text: impl Into<String>) -> Self {
        self.fallback = text.into();
        self
    }

    /// Responses handed out in call order; after the script runs dry the engine falls
    /// back to [`MockOcrEngine::with_response`] (default `""`).
    pub fn with_script<I, S>(self, responses: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        {
            let mut script = self.script.lock().expect("mock script mutex");
            script.extend(responses.into_iter().map(Into::into));
        }
        self
    }

    pub fn with_languages(mut self, languages: Vec<Language>) -> Self {
        self.languages = languages;
        self
    }

    /// Make `recognize` return `StageError::Inference` — exercises §9.3 step 7 /
    /// §14.9's fail-open rule.
    pub fn failing(mut self) -> Self {
        self.failing = true;
        self
    }

    /// Number of `recognize` calls so far.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// Dimensions of every crop handed to `recognize`, in call order. Lets a test prove
    /// the stage cropped the rect it claims to have cropped.
    pub fn crops(&self) -> Vec<(u32, u32)> {
        self.crops.lock().expect("mock crops mutex").clone()
    }
}

impl OcrEngine for MockOcrEngine {
    fn languages(&self) -> &[Language] {
        &self.languages
    }

    fn recognize(&self, crop: &DynamicImage) -> Result<String, StageError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.crops
            .lock()
            .expect("mock crops mutex")
            .push((crop.width(), crop.height()));

        if self.failing {
            return Err(StageError::Inference("mock ocr engine failure".into()));
        }

        let scripted = self.script.lock().expect("mock script mutex").pop_front();
        Ok(scripted.unwrap_or_else(|| self.fallback.clone()))
    }
}

/// Owns one [`MockOcrEngine`] and decides which languages it is offered for.
///
/// `accepted == None` means "every language, including unknown". A non-`None` list is
/// matched exactly, so `Some(vec![None])` models an engine offered *only* for
/// unknown-language boxes — the routing case §16.8 item 4 is about.
#[derive(Debug)]
pub struct MockOcrFactory {
    engine: MockOcrEngine,
    accepted: Option<Vec<Option<Language>>>,
}

impl Default for MockOcrFactory {
    fn default() -> Self {
        Self::new(MockOcrEngine::new())
    }
}

impl MockOcrFactory {
    /// Offers `engine` for every language.
    pub fn new(engine: MockOcrEngine) -> Self {
        Self {
            engine,
            accepted: None,
        }
    }

    /// Offer the engine only for these languages (`None` in the list == unknown).
    pub fn accepting(mut self, languages: Vec<Option<Language>>) -> Self {
        self.accepted = Some(languages);
        self
    }

    /// `engine_for` always returns `None` — no engine handles anything.
    pub fn denying_all(mut self) -> Self {
        self.accepted = Some(Vec::new());
        self
    }

    pub fn engine(&self) -> &MockOcrEngine {
        &self.engine
    }
}

impl OcrEngineFactory for MockOcrFactory {
    fn engine_for(&self, lang: Option<Language>) -> Option<&dyn OcrEngine> {
        match &self.accepted {
            None => Some(&self.engine),
            Some(accepted) => accepted
                .contains(&lang)
                .then_some(&self.engine as &dyn OcrEngine),
        }
    }
}
