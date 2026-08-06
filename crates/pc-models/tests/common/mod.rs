//! Shared fakes for the frozen `pc-models` test suites. FROZEN with the tests.
//!
//! **Rule for this crate's tests: no test performs network I/O.** Everything spec §8.5's
//! D1 row asks for -- progress reporting, sha256 verification, atomic installation,
//! corrupt-file recovery, the offline error message -- is exercised against the in-memory
//! fakes below. The real transport (reqwest, rustls, GitHub's 302 to
//! release-assets.githubusercontent.com, ~90 MB) is covered *only* by the `#[ignore]`d,
//! env-gated smoke test at the bottom of `d1_install.rs`.
//!
//! This mirrors spec §16.13 item 4's principle from the other side: that item forbids a
//! Rust stand-in for a *reference* implementation, because it would make a parity gate
//! circular. Here the fake stands in for a *transport*, which is not a reference for
//! anything -- the digest, the byte stream and the filesystem effects are all real.
#![allow(dead_code)]

use pc_models::{ModelError, ModelFetcher, ModelSpec, ModelStream, ProgressSink, Requirement};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tempfile::TempDir;

/// The payload every offline install test transfers: 28 bytes instead of 90 MB, because
/// what is under test is the orchestration, not the byte count.
pub const PAYLOAD: &[u8] = b"panel-ocr fake model payload";

/// `sha256(PAYLOAD)`, computed once with the system `sha256sum` (cross-checked against
/// `openssl dgst -sha256` and Python's `hashlib`) and checked in as a literal.
/// Deliberately NOT computed by calling `pc_models::sha256_hex` in the test: that would
/// make the digest gate compare the implementation against itself.
pub const PAYLOAD_SHA256: &str = "b29e1f5d4bf038c79ecb1c2a3a7d7e832318443e2d0fdf836953b1173a406728";

/// A stand-in for [`pc_models::COMIC_TEXT_DETECTOR`] with a 28-byte payload. The URL uses
/// the reserved `.invalid` TLD (RFC 6761), so a test that somehow reached a real transport
/// would fail DNS resolution instead of silently hitting the network.
pub const FAKE_SPEC: ModelSpec = ModelSpec {
    name: "fake-detector",
    file_name: "fake-detector.onnx",
    url: "https://example.invalid/fake-detector.onnx",
    sha256: PAYLOAD_SHA256,
    // `Required`, matching the doc comment above: this stands in for
    // `COMIC_TEXT_DETECTOR`. Compile-forced by §16.38 item 19's mandatory `Requirement`
    // field (no `Default` impl); pre-authorised by the D1 tie-break ruling because it
    // changes no assertion in any test.
    requirement: Requirement::Required,
};

/// A fresh temp dir plus an existing `models/` subdirectory inside it, mirroring
/// `pc_cli::paths::models_dir` (`{cache_root}/models`). The `TempDir` must be kept alive
/// by the caller for as long as the directory is used.
pub fn models_dir() -> (TempDir, PathBuf) {
    let root = TempDir::new().expect("temp dir");
    let models = root.path().join("models");
    std::fs::create_dir_all(&models).expect("create the models directory");
    (root, models)
}

/// Where [`FAKE_SPEC`] installs to inside `models`.
pub fn dest_of(models: &Path) -> PathBuf {
    models.join(FAKE_SPEC.file_name)
}

/// Sorted names of the entries directly inside `dir`; `[]` when `dir` does not exist.
pub fn entries(dir: &Path) -> Vec<String> {
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = read_dir
        .map(|entry| {
            entry
                .expect("readable directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

// ---------------------------------------------------------------- the fake transport

/// A [`ModelFetcher`] that serves bytes from memory, and can be scripted to fail.
pub struct MemoryFetcher {
    body: Vec<u8>,
    content_length: Option<u64>,
    chunk: usize,
    fail_after: Option<usize>,
    transport_error: Option<String>,
    /// Taken by the first `fetch`, so it can be moved into the returned reader.
    observer: Mutex<Option<Box<dyn FnMut() + Send>>>,
    urls: Mutex<Vec<String>>,
    calls: AtomicUsize,
}

impl MemoryFetcher {
    pub fn new(body: &[u8]) -> Self {
        Self {
            body: body.to_vec(),
            content_length: Some(body.len() as u64),
            // Small enough that a 28-byte payload takes several reads, so a
            // mid-transfer observation really is mid-transfer.
            chunk: 8,
            fail_after: None,
            transport_error: None,
            observer: Mutex::new(None),
            urls: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
        }
    }

    /// A transport that cannot reach the URL at all: offline, DNS failure, TLS failure,
    /// or a non-success HTTP status.
    pub fn offline(message: &str) -> Self {
        let mut fetcher = Self::new(&[]);
        fetcher.transport_error = Some(message.to_string());
        fetcher
    }

    /// Report this `content_length`. `None` models a server that sent no `Content-Length`.
    pub fn with_content_length(mut self, total: Option<u64>) -> Self {
        self.content_length = total;
        self
    }

    /// Drop the connection once `bytes` bytes have been read.
    pub fn failing_after(mut self, bytes: usize) -> Self {
        self.fail_after = Some(bytes);
        self
    }

    /// Run `observer` after every successful read, i.e. while the transfer is in flight.
    pub fn observing(self, observer: impl FnMut() + Send + 'static) -> Self {
        *self.observer.lock().expect("observer lock") = Some(Box::new(observer));
        self
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub fn urls(&self) -> Vec<String> {
        self.urls.lock().expect("url lock").clone()
    }
}

impl ModelFetcher for MemoryFetcher {
    fn fetch(&self, url: &str) -> Result<ModelStream, ModelError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.urls.lock().expect("url lock").push(url.to_string());

        if let Some(message) = &self.transport_error {
            return Err(ModelError::Transport {
                url: url.to_string(),
                message: message.clone(),
            });
        }

        Ok(ModelStream {
            content_length: self.content_length,
            reader: Box::new(ScriptedReader {
                body: self.body.clone(),
                position: 0,
                chunk: self.chunk,
                fail_after: self.fail_after,
                observer: self.observer.lock().expect("observer lock").take(),
            }),
        })
    }
}

struct ScriptedReader {
    body: Vec<u8>,
    position: usize,
    chunk: usize,
    fail_after: Option<usize>,
    observer: Option<Box<dyn FnMut() + Send>>,
}

impl Read for ScriptedReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if let Some(limit) = self.fail_after {
            if self.position >= limit {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "fake transport dropped the connection",
                ));
            }
        }

        let mut end = self.body.len().min(self.position + self.chunk);
        if let Some(limit) = self.fail_after {
            end = end.min(limit);
        }
        let count = end.saturating_sub(self.position).min(buf.len());
        if count == 0 {
            return Ok(0);
        }
        buf[..count].copy_from_slice(&self.body[self.position..self.position + count]);
        self.position += count;
        if let Some(observer) = self.observer.as_mut() {
            observer();
        }
        Ok(count)
    }
}

// ---------------------------------------------------------------- the progress recorder

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressEvent {
    Start(Option<u64>),
    Advance(u64),
    Finish,
}

#[derive(Debug, Default)]
pub struct RecordingProgress {
    pub events: Vec<ProgressEvent>,
}

impl ProgressSink for RecordingProgress {
    fn start(&mut self, total: Option<u64>) {
        self.events.push(ProgressEvent::Start(total));
    }
    fn advance(&mut self, bytes: u64) {
        self.events.push(ProgressEvent::Advance(bytes));
    }
    fn finish(&mut self) {
        self.events.push(ProgressEvent::Finish);
    }
}

impl RecordingProgress {
    /// The argument of the first `start` call; an outer `None` means `start` never
    /// happened at all.
    pub fn started_with(&self) -> Option<Option<u64>> {
        self.events.iter().find_map(|event| match event {
            ProgressEvent::Start(total) => Some(*total),
            _ => None,
        })
    }
    pub fn starts(&self) -> usize {
        self.events
            .iter()
            .filter(|event| matches!(event, ProgressEvent::Start(_)))
            .count()
    }
    pub fn total_advanced(&self) -> u64 {
        self.events
            .iter()
            .map(|event| match event {
                ProgressEvent::Advance(bytes) => *bytes,
                _ => 0,
            })
            .sum()
    }
    pub fn finishes(&self) -> usize {
        self.events
            .iter()
            .filter(|event| **event == ProgressEvent::Finish)
            .count()
    }
}
