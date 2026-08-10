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

/// Whether a registry entry is needed by a default run.
///
/// Ratified by spec §16.38 item 19 (decision D1). Optionality is a **mandatory field** on
/// [`ModelSpec`] rather than a derived `OPTIONAL` slice or an `is_optional()` predicate:
/// both of those were explicitly rejected, because a field with no `Default` impl forces
/// every construction site to state the answer, so a new model cannot silently inherit
/// "required" and grow every user's `models download`.
///
/// Deliberately **no** `#[derive(Default)]` and no `impl Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requirement {
    /// Fetched by `models download` with no flags. Absence is a `models verify` failure.
    Required,
    /// Fetched only with `models download --include-optional`. Absence is reported as a
    /// distinct, non-failing status; **corruption is still a failure** — optionality
    /// licenses absence, never corruption.
    Optional,
}

/// A model known to the application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    pub name: &'static str,
    pub file_name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    /// §16.38 item 19. Mandatory, and stated at every construction site.
    pub requirement: Requirement,
}

/// The detector weights pinned by spec §8.3 step 3.
pub const COMIC_TEXT_DETECTOR: ModelSpec = ModelSpec {
    name: "comic-text-detector",
    file_name: "comictextdetector.pt.onnx",
    url: "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
    sha256: "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f",
    requirement: Requirement::Required,
};

/// The manga-ocr encoder weights (spec §16.30 item 1, ratified artifact pin).
pub const MANGA_OCR_ENCODER: ModelSpec = ModelSpec {
    name: "manga-ocr-encoder",
    file_name: "encoder_model.onnx",
    url: "https://huggingface.co/mayocream/manga-ocr-onnx/resolve/24b12778d85800835e2ca409236de281b8ab7b9f/encoder_model.onnx",
    sha256: "15fa8155fe9bc1a7d25d9bb353debaa4def033d0174e907dbd2dd6d995def85f",
    requirement: Requirement::Required,
};

/// The manga-ocr decoder weights (spec §16.30 item 1, ratified artifact pin).
pub const MANGA_OCR_DECODER: ModelSpec = ModelSpec {
    name: "manga-ocr-decoder",
    file_name: "decoder_model.onnx",
    url: "https://huggingface.co/mayocream/manga-ocr-onnx/resolve/24b12778d85800835e2ca409236de281b8ab7b9f/decoder_model.onnx",
    sha256: "ef7765261e9d1cdc34d89356986c2bbc2a082897f753a89605ae80fdfa61f5e8",
    requirement: Requirement::Required,
};

/// The LaMa inpainting weights (spec §16.38 item 1(a), the artifact pinned by the L0
/// spike; substitution for upstream's TorchScript checkpoint registered as
/// `DEVIATION(25)` at §14).
///
/// DEVIATION(25): upstream downloads `anime-manga-big-lama.pt`
/// (`pcleaner/model_downloader.py:21-22`) and loads it through `simple_lama_inpainting`;
/// this port has no TorchScript loader, so it loads the ONNX export from
/// `mayocream/koharu` at revision `15439cba09df388c51de6e47c6020bc31edab41f`. The
/// provenance chain is a *string match* on the republisher's declared
/// `source_checkpoint`, not numerical equivalence at any tolerance (§16.38 item 6(a)).
///
/// **`Requirement::Required` as of §16.46 item 11(b).** It was `Optional`, on the ground
/// that *"`inpainting_enabled` defaults to `false` … §16.38 item 19 forbids growing every
/// user's `models download` by that much for a feature that defaults off"* — and §16.46
/// item 1(b) turns inpainting on by default, which retires that ground entirely.
///
/// **What the promotion buys, stated as the failure it prevents.** With the flag on and this
/// artifact absent, the first page carrying an eligible region aborts the whole batch: the
/// provider declares its construction failures run-fatal, so §16.38 item 9(f)'s outcome is
/// *"**zero** pages are exported and the exit code is 1"*. Item 9(f) accepted that on the
/// explicit ground that *"the blast radius is confined to users who explicitly set the flag,
/// since it defaults to false"* — a sentence §16.46 item 11(a) supersedes. Leaving this
/// `Optional` would therefore have made a run-fatal abort the DEFAULT first-run experience
/// for anyone who had not passed `--include-optional`, while `models download` reported
/// success and `models verify` reported OK.
///
/// **The cost, not minimised:** every user's `models download` grows by 207,482,644 bytes.
/// Two alternatives were put to the maintainer and declined — leaving it optional and
/// accepting the abort, and degrading to a flat fill with a WARN (which contradicts §16.38
/// item 9's *"The remedy is provisioning, not a runtime fallback"* and would need its own
/// ratification). §16.46 item 11(b) records the choice.
pub const LAMA_MANGA_INPAINTER: ModelSpec = ModelSpec {
    name: "lama-manga-inpainter",
    file_name: "lama-manga.onnx",
    url: "https://huggingface.co/mayocream/koharu/resolve/15439cba09df388c51de6e47c6020bc31edab41f/lama-manga.onnx",
    sha256: "50a1abae0d73bd46d08eae36c8590cd59ad09029494c9698702b050ef00b0100",
    requirement: Requirement::Required,
};

/// The registry exposed to model-management callers.
///
/// §16.38 item 19(a): this stays the **complete** registry — it is not shrunk to a
/// required-only list. Callers scope their own iteration through [`selected`] and
/// [`skipped`]; `models path` deliberately iterates the whole thing.
pub const ALL: &[&ModelSpec] = &[
    &COMIC_TEXT_DETECTOR,
    &MANGA_OCR_ENCODER,
    &MANGA_OCR_DECODER,
    &LAMA_MANGA_INPAINTER,
];

/// Is `spec` in scope for `models download` / `models verify` at this flag setting?
///
/// The single predicate behind both [`selected`] and [`skipped`], so the two are a
/// partition of [`ALL`] by construction rather than by two hand-written filters that
/// could drift apart.
fn is_selected(spec: &ModelSpec, include_optional: bool) -> bool {
    include_optional || spec.requirement == Requirement::Required
}

/// The registry entries `models download` fetches and `models verify` reports
/// (§16.38 item 19(b)). Without `--include-optional`, exactly the `Required` ones.
pub fn selected(include_optional: bool) -> Vec<&'static ModelSpec> {
    ALL.iter()
        .copied()
        .filter(|spec| is_selected(spec, include_optional))
        .collect()
}

/// The registry entries left out at this flag setting. `models download` must **name**
/// these rather than skip them silently (§16.38 item 19(c)).
pub fn skipped(include_optional: bool) -> Vec<&'static ModelSpec> {
    ALL.iter()
        .copied()
        .filter(|spec| !is_selected(spec, include_optional))
        .collect()
}

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

/// One model's `models verify` outcome (spec §13.1, ratified by §16.38 item 19(d)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyStatus {
    /// Present, right size, right digest.
    Ok,
    /// A [`Requirement::Required`] model is absent from the managed cache.
    Missing,
    /// A [`Requirement::Optional`] model is absent from the managed cache. **Not** a
    /// failure, and its label deliberately does not contain the substring `MISSING`, so a
    /// script grepping the old word cannot confuse the two states.
    NotInstalled,
    SizeMismatch {
        actual: u64,
        expected: u64,
    },
    HashMismatch {
        actual: String,
        expected: String,
    },
    /// Verification could not be carried out at all (an I/O error on the cached file).
    Error(String),
}

impl VerifyStatus {
    /// The status for a model absent from the managed cache — the only place a
    /// [`Requirement`] changes a verification outcome.
    pub fn absent(requirement: Requirement) -> Self {
        match requirement {
            Requirement::Required => Self::Missing,
            Requirement::Optional => Self::NotInstalled,
        }
    }

    /// The status column `models verify` prints.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Missing => "MISSING",
            Self::NotInstalled => "NOT INSTALLED (optional)",
            Self::SizeMismatch { .. } => "SIZE MISMATCH",
            Self::HashMismatch { .. } => "HASH MISMATCH",
            Self::Error(_) => "ERROR",
        }
    }

    /// Whether this outcome clears `all_ok` and makes `models verify` exit `EXIT_FATAL`.
    ///
    /// `NotInstalled` is the only non-`Ok` status that is not a failure. Note what this
    /// signature does *not* take: a [`Requirement`]. Every corruption status is a failure
    /// with no optionality in scope, which is how §16.38 item 19(d)'s "optionality
    /// licenses absence, never corruption" is enforced structurally rather than by a
    /// branch someone could later write the other way.
    pub fn is_failure(&self) -> bool {
        !matches!(self, Self::Ok | Self::NotInstalled)
    }
}

/// One model's verification, with the path it was looked for at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verification {
    pub path: PathBuf,
    pub status: VerifyStatus,
}

/// Verify one model against the managed cache.
///
/// Filesystem-only and **network-free** by construction — it takes no [`ModelFetcher`] —
/// which is what makes `models verify`'s whole policy unit-testable under §16.18 item 3's
/// prohibition on test-reachable network I/O. It creates no directory and modifies
/// nothing.
///
/// `expected_size` is the published byte length when one is pinned, `None` otherwise; it
/// arrives as a parameter because the size table is `pc-cli`'s presentation-layer data
/// (see `pc_cli::models::expected_size`), not registry policy.
pub fn verify(spec: &ModelSpec, models_dir: &Path, expected_size: Option<u64>) -> Verification {
    let path = match resolve(spec, models_dir, None) {
        Ok(Resolution::Cached(path)) => path,
        Ok(Resolution::Missing(path)) => {
            return Verification {
                path,
                status: VerifyStatus::absent(spec.requirement),
            }
        }
        Ok(Resolution::Override(_)) => {
            unreachable!("managed model verification never supplies an override")
        }
        Err(error) => {
            return Verification {
                path: models_dir.join(spec.file_name),
                status: VerifyStatus::Error(error.to_string()),
            }
        }
    };

    if let Some(expected) = expected_size {
        match fs::metadata(&path) {
            Ok(metadata) if metadata.len() != expected => {
                return Verification {
                    status: VerifyStatus::SizeMismatch {
                        actual: metadata.len(),
                        expected,
                    },
                    path,
                }
            }
            Ok(_) => {}
            Err(source) => {
                let status = VerifyStatus::Error(io_error(&path, source).to_string());
                return Verification { path, status };
            }
        }
    }

    let status = match verify_sha256(&path, spec.sha256) {
        Ok(()) => VerifyStatus::Ok,
        Err(ModelError::HashMismatch {
            actual, expected, ..
        }) => VerifyStatus::HashMismatch { actual, expected },
        Err(error) => VerifyStatus::Error(error.to_string()),
    };
    Verification { path, status }
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
