//! Model registry and managed-cache acquisition for panel-ocr.
//!
//! The model cache is deliberately kept out of the stage crates.  This crate owns the
//! filesystem and network policy; callers receive a verified path and stage crates only
//! consume that path.

use pc_core::StageError;
use reqwest::blocking::{Client, Response};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// A model known to the application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    pub name: &'static str,
    pub file_name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
}

/// The detector weights pinned by spec §8.3 step 3.
pub const COMIC_TEXT_DETECTOR: ModelSpec = ModelSpec {
    name: "comic-text-detector",
    file_name: "comictextdetector.pt.onnx",
    url: "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
    sha256: "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f",
};

/// The manga-ocr encoder weights (spec §16.30 item 1, ratified artifact pin).
pub const MANGA_OCR_ENCODER: ModelSpec = ModelSpec {
    name: "manga-ocr-encoder",
    file_name: "encoder_model.onnx",
    url: "https://huggingface.co/mayocream/manga-ocr-onnx/resolve/24b12778d85800835e2ca409236de281b8ab7b9f/encoder_model.onnx",
    sha256: "15fa8155fe9bc1a7d25d9bb353debaa4def033d0174e907dbd2dd6d995def85f",
};

/// The manga-ocr decoder weights (spec §16.30 item 1, ratified artifact pin).
pub const MANGA_OCR_DECODER: ModelSpec = ModelSpec {
    name: "manga-ocr-decoder",
    file_name: "decoder_model.onnx",
    url: "https://huggingface.co/mayocream/manga-ocr-onnx/resolve/24b12778d85800835e2ca409236de281b8ab7b9f/decoder_model.onnx",
    sha256: "ef7765261e9d1cdc34d89356986c2bbc2a082897f753a89605ae80fdfa61f5e8",
};

/// The registry exposed to model-management callers.
pub const ALL: &[&ModelSpec] = &[&COMIC_TEXT_DETECTOR, &MANGA_OCR_ENCODER, &MANGA_OCR_DECODER];

/// The result of querying an override or the managed cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    Override(PathBuf),
    Cached(PathBuf),
    Missing(PathBuf),
}

impl Resolution {
    /// Return the path represented by this resolution.
    pub fn path(&self) -> &Path {
        match self {
            Self::Override(path) | Self::Cached(path) | Self::Missing(path) => path,
        }
    }
}

/// Errors produced while resolving, fetching, or installing a model.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model override path does not exist: {path}")]
    OverrideMissing { path: PathBuf },

    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("failed to fetch model from {url}: {message}")]
    Transport { url: String, message: String },

    #[error("sha256 mismatch for {path}: expected {expected}, actual {actual}")]
    HashMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },

    #[error("model {name} is unavailable: could not obtain {url} at {dest}: {reason}; run `panel-ocr models download` or provide `--model-path FILE`")]
    Unavailable {
        name: String,
        url: String,
        dest: PathBuf,
        reason: String,
    },
}

impl From<ModelError> for StageError {
    fn from(error: ModelError) -> Self {
        Self::Model(error.to_string())
    }
}

/// A response body returned by a model fetcher.
pub struct ModelStream {
    pub content_length: Option<u64>,
    pub reader: Box<dyn Read + Send>,
}

/// Injectable model transport.
pub trait ModelFetcher: Send + Sync {
    fn fetch(&self, url: &str) -> Result<ModelStream, ModelError>;
}

/// The production blocking HTTP transport.
pub struct ReqwestFetcher {
    client: Client,
}

impl ReqwestFetcher {
    /// Construct a client with a connection timeout and a generous whole-request timeout.
    ///
    /// The finite 1800-second request timeout accommodates a ~90 MB download while
    /// guaranteeing termination if the connection stalls.
    pub fn new() -> Self {
        // reqwest blocking's own default is `Timeout(Some(Duration::from_secs(30)))`
        // (see reqwest-0.12.28/src/blocking/client.rs:1498-1503), which covers
        // "connect, read and write" per its doc comment. 30s is far too short for
        // this use case: the model file is 94,669,756 bytes, so a 30s budget
        // demands >3 MB/s sustained throughput.
        //
        // 1800s over 94,669,756 bytes implies a sustained-throughput floor of
        // about 52 KB/s. Any realistic connection clears that easily, so this
        // will not cause spurious failures, but it guarantees the fetcher
        // terminates instead of hanging indefinitely.
        const WHOLE_REQUEST_TIMEOUT_SECS: u64 = 1800;

        let client = Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(WHOLE_REQUEST_TIMEOUT_SECS))
            // `tcp_keepalive` only shortens detection of a peer that is fully
            // gone (dropped connection); it does NOT cover an application-level
            // stall, since a live server holding the socket open while sending
            // nothing will still ACK keepalive probes. The finite `.timeout()`
            // above is what actually bounds that stalled case.
            .tcp_keepalive(Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .expect("the fixed reqwest client configuration must be valid");
        Self { client }
    }
}

impl Default for ReqwestFetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelFetcher for ReqwestFetcher {
    fn fetch(&self, url: &str) -> Result<ModelStream, ModelError> {
        let response = self
            .client
            .get(url)
            .send()
            .map_err(|error| ModelError::Transport {
                url: url.to_owned(),
                message: error.to_string(),
            })?;

        ensure_success(url, response)
    }
}

fn ensure_success(url: &str, response: Response) -> Result<ModelStream, ModelError> {
    let status = response.status();
    if !status.is_success() {
        return Err(ModelError::Transport {
            url: url.to_owned(),
            message: format!("HTTP status {status}"),
        });
    }

    Ok(ModelStream {
        content_length: response.content_length(),
        reader: Box::new(response),
    })
}

/// A progress callback owned by the presentation layer.
pub trait ProgressSink {
    fn start(&mut self, total: Option<u64>);
    fn advance(&mut self, bytes: u64);
    fn finish(&mut self);
}

/// A progress sink for callers that do not render progress.
#[derive(Debug, Default)]
pub struct NoProgress;

impl ProgressSink for NoProgress {
    fn start(&mut self, _total: Option<u64>) {}

    fn advance(&mut self, _bytes: u64) {}

    fn finish(&mut self) {}
}

/// Query model availability without creating directories or changing files.
pub fn resolve(
    spec: &ModelSpec,
    models_dir: &Path,
    override_path: Option<&Path>,
) -> Result<Resolution, ModelError> {
    if let Some(path) = override_path {
        if !path.is_file() {
            return Err(ModelError::OverrideMissing {
                path: path.to_path_buf(),
            });
        }
        return Ok(Resolution::Override(path.to_path_buf()));
    }

    let cached = models_dir.join(spec.file_name);
    if cached.is_file() {
        Ok(Resolution::Cached(cached))
    } else {
        Ok(Resolution::Missing(cached))
    }
}

/// Return a unique staging path in the destination's directory.
pub fn partial_path(dest: &Path) -> PathBuf {
    static NEXT_PARTIAL_ID: AtomicU64 = AtomicU64::new(0);
    let id = NEXT_PARTIAL_ID.fetch_add(1, Ordering::Relaxed);
    let name = dest
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| std::borrow::Cow::Borrowed("model"));
    dest.with_file_name(format!(".{name}.partial-{}-{id}", std::process::id()))
}

/// Compute a file's lowercase SHA-256 digest.
pub fn sha256_hex(path: &Path) -> Result<String, ModelError> {
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| io_error(path, source))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hex_digest(&hasher.finalize()))
}

/// Verify a file's SHA-256 digest, accepting ASCII case differences in the expectation.
pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<(), ModelError> {
    let actual = sha256_hex(path)?;
    if actual.eq_ignore_ascii_case(expected_hex) {
        Ok(())
    } else {
        Err(ModelError::HashMismatch {
            path: path.to_path_buf(),
            expected: expected_hex.to_owned(),
            actual,
        })
    }
}

/// Download, verify, and atomically publish a managed model file.
pub fn install(
    spec: &ModelSpec,
    models_dir: &Path,
    fetcher: &dyn ModelFetcher,
    progress: &mut dyn ProgressSink,
) -> Result<PathBuf, ModelError> {
    fs::create_dir_all(models_dir).map_err(|source| io_error(models_dir, source))?;
    let dest = models_dir.join(spec.file_name);
    let partial = partial_path(&dest);

    let result = install_inner(spec, &dest, &partial, fetcher, progress);
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result
}

fn install_inner(
    spec: &ModelSpec,
    dest: &Path,
    partial: &Path,
    fetcher: &dyn ModelFetcher,
    progress: &mut dyn ProgressSink,
) -> Result<PathBuf, ModelError> {
    let mut stream = fetcher.fetch(spec.url)?;
    let expected_length = stream.content_length;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(partial)
        .map_err(|source| io_error(partial, source))?;
    let mut hasher = Sha256::new();
    let mut total_read = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];

    progress.start(expected_length);
    let transfer = (|| -> Result<(), ModelError> {
        loop {
            let read = stream
                .reader
                .read(&mut buffer)
                .map_err(|source| ModelError::Transport {
                    url: spec.url.to_owned(),
                    message: format!("failed while transferring response body: {source}"),
                })?;
            if read == 0 {
                break;
            }
            file.write_all(&buffer[..read])
                .map_err(|source| io_error(partial, source))?;
            hasher.update(&buffer[..read]);
            total_read += read as u64;
            progress.advance(read as u64);
        }

        if let Some(expected) = expected_length.filter(|expected| total_read != *expected) {
            return Err(ModelError::Transport {
                url: spec.url.to_owned(),
                message: format!(
                    "failed while transferring response body: {}",
                    io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("response body contained {total_read} bytes, expected {expected}"),
                    )
                ),
            });
        }

        file.sync_all()
            .map_err(|source| io_error(partial, source))?;

        let actual = hex_digest(&hasher.finalize());
        if !actual.eq_ignore_ascii_case(spec.sha256) {
            return Err(ModelError::HashMismatch {
                path: dest.to_path_buf(),
                expected: spec.sha256.to_owned(),
                actual,
            });
        }

        Ok(())
    })();
    progress.finish();
    transfer?;

    drop(file);
    fs::rename(partial, dest).map_err(|source| io_error(dest, source))?;
    Ok(dest.to_path_buf())
}

/// Resolve an override or ensure that a verified managed-cache file is available.
pub fn ensure_available(
    spec: &ModelSpec,
    models_dir: &Path,
    override_path: Option<&Path>,
    fetcher: &dyn ModelFetcher,
    progress: &mut dyn ProgressSink,
) -> Result<PathBuf, ModelError> {
    match resolve(spec, models_dir, override_path)? {
        Resolution::Override(path) => Ok(path),
        Resolution::Missing(dest) => match install(spec, models_dir, fetcher, progress) {
            Ok(path) => Ok(path),
            Err(error @ ModelError::Transport { .. }) => Err(unavailable(spec, &dest, error)),
            Err(error) => Err(error),
        },
        Resolution::Cached(path) => match verify_sha256(&path, spec.sha256) {
            Ok(()) => Ok(path),
            Err(error @ ModelError::HashMismatch { .. }) => {
                let (expected, actual) = match &error {
                    ModelError::HashMismatch {
                        expected, actual, ..
                    } => (expected.clone(), actual.clone()),
                    _ => unreachable!("the match arm guarantees a hash mismatch"),
                };
                tracing::warn!(
                    path = %path.display(),
                    expected = %expected,
                    actual = %actual,
                    "managed model digest mismatch; attempting a fresh download"
                );
                match install(spec, models_dir, fetcher, progress) {
                    Ok(installed) => Ok(installed),
                    Err(replacement @ ModelError::Transport { .. })
                    | Err(replacement @ ModelError::HashMismatch { .. }) => {
                        let reason = format!(
                            "replacement failed after cached digest error ({error}): {replacement}"
                        );
                        Err(unavailable_with_reason(spec, &path, reason))
                    }
                    Err(replacement) => Err(replacement),
                }
            }
            Err(_) => match install(spec, models_dir, fetcher, progress) {
                Ok(installed) => Ok(installed),
                Err(replacement @ ModelError::Transport { .. }) => {
                    Err(unavailable(spec, &path, replacement))
                }
                Err(replacement) => Err(replacement),
            },
        },
    }
}

fn unavailable(spec: &ModelSpec, dest: &Path, error: ModelError) -> ModelError {
    unavailable_with_reason(spec, dest, error.to_string())
}

fn unavailable_with_reason(spec: &ModelSpec, dest: &Path, reason: String) -> ModelError {
    ModelError::Unavailable {
        name: spec.name.to_owned(),
        url: spec.url.to_owned(),
        dest: dest.to_path_buf(),
        reason,
    }
}

fn io_error(path: &Path, source: io::Error) -> ModelError {
    ModelError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn hex_digest(digest: &[u8]) -> String {
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}
