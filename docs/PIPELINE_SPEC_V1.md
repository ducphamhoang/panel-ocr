# panel-ocr v1 Pipeline Architecture Spec

**Status:** Draft from Technical Architecture (Opus) — pending Senior Rust Engineer
co-review (test code + disagreement check) per CLAUDE.md's Plan phase.
**Scope:** v1 milestone only — CLI + detect → preprocess → mask → denoise → export
**Fixed constraints:** everything in `docs/ARCHITECTURE_DECISIONS.md` (GPL-3, full-Rust/no-Python-runtime, `ort` 2.0.0-rc.12 pinned, CPU-only EP, TOML config, stage-based crates, fresh `clap` CLI, Linux + macOS, GUI/inpaint/PSD deferred)

---

## 0. Reading guide

- §1–§7 are **cross-cutting**: workspace layout, shared types, stage contract, data flow, error policy, config, fixtures.
- §8–§12 are the **five stage specs** (detect, preprocess, mask, denoise, export), each with: crate, contract, algorithm, dependencies, task breakdown, fixtures, acceptance criteria, out-of-scope.
- §13 is the **consolidated task table** (simple/heavy, order, dependencies) — this is what the TDD loop is driven from.
- §14 lists **deliberate deviations from upstream** and §15 the **decisions needing reviewer sign-off** before tests are frozen.

Terminology: I use "upstream" for PanelCleaner (Python). Line references are to the clone at
`/tmp/claude-0/-home-user-panel-ocr/d16688e0-2c58-5f05-b3ef-834afb4679ed/scratchpad/PanelCleaner`.

---

## 1. Workspace layout

Single cargo workspace at repo root. `edition = "2021"`, MSRV pinned in `rust-toolchain.toml`.

```
panel-ocr/
├── Cargo.toml                # [workspace], shared [workspace.dependencies], resolver = "2"
├── rust-toolchain.toml
├── crates/
│   ├── pc-core/              # shared types: Rect, PageData, MaskData, ImageHandle, Step/Output, errors
│   ├── pc-config/            # TOML profile + config, defaults, validation, legacy-INI importer (v1.5)
│   ├── pc-imageops/          # pure image algorithms shared by stages (no stage logic)
│   ├── pc-models/            # model file resolution: download, sha256 verify, cache dir
│   ├── pc-detect/            # STAGE 1  (+ trait TextDetector, ONNX backend, mock/replay backends)
│   ├── pc-ocr/               # trait OcrEngine, manga-ocr ONNX backend, mock engine
│   ├── pc-preprocess/        # STAGE 2
│   ├── pc-mask/              # STAGE 3
│   ├── pc-denoise/           # STAGE 4
│   ├── pc-export/            # STAGE 5
│   ├── pc-pipeline/          # orchestrator: chaining, checkpoints, rayon batching, error policy
│   ├── pc-cli/               # bin `panel-ocr` (clap)
│   └── pc-testkit/           # dev-dependency only: fixture paths, image metrics (SSIM/IoU/Δ), golden helpers
├── tests/fixtures/
│   ├── upstream/             # vendored PanelCleaner fixtures + ATTRIBUTION.md (GPL-3)
│   └── recorded/             # recorded real-model outputs (the "mocked ML boundary")
└── xtask/                    # maintainer-only: record-fixtures, calibrate-goldens, export-onnx notes
```

Rules Codex must not violate:

1. **Stage crates never touch the filesystem for path *resolution*.** They receive `ImageHandle`s and typed structs; `pc-pipeline` owns all path construction and persistence. (Stage crates may *read* an `ImageHandle` that carries a path, and may *write* an output only when handed an explicit destination `PathBuf`.)
2. **No stage crate depends on another stage crate.** All shared vocabulary lives in `pc-core`; all shared pixel math in `pc-imageops`.
3. **`pc-cli` contains no algorithm code** — argument parsing, config loading, and calls into `pc-pipeline` only.
4. **Every ML call goes through a trait** (`TextDetector`, `OcrEngine`) so the `candle` escape hatch stays real (decision #2).

Crate dependency graph (acyclic):

```
pc-core   ← pc-config, pc-imageops, pc-models, all stages, pc-pipeline, pc-cli
pc-imageops ← pc-detect, pc-mask, pc-denoise, pc-export
pc-models ← pc-detect, pc-ocr
pc-detect, pc-ocr, pc-preprocess, pc-mask, pc-denoise, pc-export ← pc-pipeline ← pc-cli
pc-testkit ← (dev-dependencies of every crate above)
```

Third-party crates (pinned in `[workspace.dependencies]`): `serde`/`serde_json`, `toml_edit`, `image`, `imageproc`, `ndarray`, `rayon`, `clap` (derive), `thiserror`, `anyhow` (binary only), `tracing` + `tracing-subscriber`, `ort` (`=2.0.0-rc.12`, feature-gated), `reqwest` (rustls), `sha2`, `hf-hub`, `indicatif`, `regex`, `csv`. Dev: `insta` (**restricted use, §15.10** — only for the two named regression locks in §8.7(B)9 and §9.7(B)11; every numeric/algorithmic gate elsewhere uses hand-written assertions), `approx`.

---

## 2. Shared types (`pc-core`)

All types `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]` unless noted. JSON is `snake_case`. Every persisted struct carries `schema_version: u32` (start at `1`) as its first field so future format changes are detectable.

### 2.1 Geometry

Upstream `Box` (`structures.py:25-145`) → `Rect`. Coordinates are **inclusive-exclusive in use** (upstream feeds them to `PIL.crop`, which treats `x2/y2` as exclusive) but upstream's `pad()` clamps `x2` to `width`, and `__contains__` uses `<=` on both ends. We preserve upstream's arithmetic exactly and document the semantics as: **`x1,y1` inclusive; `x2,y2` exclusive for cropping/rasterising; inclusive for `contains()`** (matching upstream's mixed convention — deviating here would change box merging behaviour).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rect { pub x1: i32, pub y1: i32, pub x2: i32, pub y2: i32 }

impl Rect {
    pub fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Self;
    pub fn width(&self)  -> i32;                    // x2 - x1
    pub fn height(&self) -> i32;                    // y2 - y1
    pub fn area(&self)   -> i64;                    // width * height as i64 (upstream: int, can be large)
    pub fn center(&self) -> (i32, i32);             // ((x1+x2)/2, (y1+y2)/2), floor-div like Python //
    pub fn contains(&self, p: (i32, i32)) -> bool;  // x1 <= x <= x2 && y1 <= y <= y2
    pub fn merge(&self, other: &Rect) -> Rect;      // bounding union
    pub fn overlaps(&self, other: &Rect, threshold_percent: f64) -> bool;
    pub fn overlaps_center(&self, other: &Rect) -> bool;
    pub fn pad(&self, amount: i32, canvas: (u32, u32)) -> Rect;
    pub fn right_pad(&self, amount: i32, canvas: (u32, u32)) -> Rect;
    pub fn scale(&self, factor: f64) -> Rect;       // per-coordinate, truncating toward zero (Python int())
    pub fn translate(&self, dx: i32, dy: i32) -> Rect;
    pub fn is_empty(&self) -> bool;                 // width <= 0 || height <= 0
    pub fn to_crop(&self, canvas: (u32,u32)) -> Option<(u32,u32,u32,u32)>; // clamped (x,y,w,h) for image crate
}
```

Exact semantics Codex must reproduce (from `structures.py`):

- `overlaps`: `x_ov = max(0, min(x2,o.x2) - max(x1,o.x1))`, `y_ov` likewise, `inter = x_ov*y_ov`; `smaller = min(area, o.area)`, **and if that is 0 use 1**; return `inter as f64 / smaller as f64 > threshold_percent / 100.0` (strictly greater).
- `overlaps_center`: `other.contains(self.center()) || self.contains(other.center())`.
- `pad`: `x1' = max(x1-a, 0)`, `y1' = max(y1-a, 0)`, `x2' = min(x2+a, canvas.0)`, `y2' = min(y2+a, canvas.1)`.
- `right_pad`: only `x2' = min(x2+a, canvas.0)`.
- `scale`: `(x as f64 * factor) as i32` per coordinate (truncation, **not** rounding — matches Python `int()`).
- `center`: Python floor division on non-negative sums; use `(x1 + x2).div_euclid(2)`.

### 2.2 Language

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language { Japanese, English }   // upstream jpn / eng; "unknown" == Option::None

pub const RTL_BOX_ORDER_LANGUAGES: &[Language] = &[Language::Japanese];
```

Upstream's RTL set (`ocr/supported_languages.py:153`) also contains chi_sim/chi_tra/ara/fas/heb; those language codes are unreachable in v1 because the detector only emits `eng`/`ja`/`unknown` and Tesseract is deferred (§15.4). Keep the constant as a slice so adding codes later is additive.

### 2.3 `ImageHandle` — how images cross stage boundaries

Solves "in-memory single run vs disk-checkpointed batch run" without two sets of contracts:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageHandle {
    /// Where the image lives (or will live) on disk. `None` only in pure in-memory runs.
    pub path: Option<PathBuf>,
    #[serde(skip)]
    cached: Option<Arc<DynamicImage>>,
}

impl ImageHandle {
    pub fn from_path(p: impl Into<PathBuf>) -> Self;
    pub fn from_memory(img: DynamicImage) -> Self;                     // path: None
    pub fn with_both(p: impl Into<PathBuf>, img: DynamicImage) -> Self;
    /// Returns the cached image if present, else decodes from `path`.
    pub fn load(&self) -> Result<Arc<DynamicImage>, CoreError>;
    /// Decode-free size query (uses cache, else image header).
    pub fn dimensions(&self) -> Result<(u32, u32), CoreError>;
    pub fn is_materialized(&self) -> bool;                              // path exists on disk
}
```

Invariant: after `serde` round-trip, `cached` is `None`; a handle with `path: None` that round-trips is an error (`CoreError::UnmaterializedHandle`) — the pipeline must materialize before checkpointing. This invariant is directly testable.

### 2.4 `PageDataRaw` — detector output (upstream `#raw.json`)

```rust
pub struct PageDataRaw {
    pub schema_version: u32,
    pub original_path: PathBuf,        // the user's input file
    pub base_image: ImageHandle,       // scaled RGB copy written as PNG (upstream `_base.png`)
    pub raw_mask: ImageHandle,         // refined text mask, 8-bit grayscale (upstream `_raw_mask.png`)
    pub scale: f64,                    // base_image size / original size (<= 1.0)
    pub image_size: (u32, u32),        // size of base_image
    pub blocks: Vec<DetectedBlock>,
}

pub struct DetectedBlock {
    pub rect: Rect,                    // in base_image coordinates
    pub language: Option<Language>,    // None == upstream "unknown"
    pub confidence: f32,               // YOLO objectness*class score, rounded to 3 dp (upstream parity)
    pub mask_coverage: f32,            // mean(raw_mask within rect)/255 — kept for analytics/debug
}
```

`scale` semantics (upstream `PageData.scale`, `ctd_interface.resize_to_target`): `scale = new_height / original_height`; `scale == 1.0` when no resize happened. Note `masker.py:104` compares `scale != 1`, and `denoiser` derives `scale_up_factor = original.width / mask.width`; we keep `scale` and recompute the up-factor from actual sizes (more robust; see §11.3).

### 2.5 `PageData` — preprocessor output (upstream `#clean.json`)

Upstream keeps four parallel lists plus a parallel `box_language` list, with an implicit index-pairing between `merged_extended_boxes` and `reference_boxes` (`masker.py:63-65` zips them). We make that pairing a type:

```rust
pub struct PageData {
    pub schema_version: u32,
    pub original_path: PathBuf,
    pub base_image: ImageHandle,
    pub raw_mask: ImageHandle,
    pub scale: f64,
    pub image_size: (u32, u32),
    pub page_language: Option<Language>,
    /// Tight boxes: detector boxes, filtered, merged-on-center, padded slightly. Reading-order sorted.
    pub text_boxes: Vec<TextBox>,
    /// Extended boxes: one per tight box, padded a lot. Used to raster the "box mask".
    pub extended_boxes: Vec<Rect>,
    /// Masking regions: overlapping extended boxes merged, each paired with its grown reference box.
    pub masking_regions: Vec<MaskingRegion>,
}

pub struct TextBox { pub rect: Rect, pub language: Option<Language> }

pub struct MaskingRegion {
    pub masking: Rect,    // upstream merged_extended_boxes[i]
    pub reference: Rect,  // upstream reference_boxes[i]  (masking padded by box_reference_padding)
}
```

Invariants (assert in debug, test explicitly):
`extended_boxes.len() == text_boxes.len()`; every `MaskingRegion` satisfies `reference` contains `masking`; all rects lie within `image_size`.

### 2.6 `MaskData` — masker output (upstream `#mask_data.json`)

```rust
pub struct MaskData {
    pub schema_version: u32,
    pub original_path: PathBuf,
    pub base_image: ImageHandle,
    pub combined_mask: ImageHandle,     // RGBA
    pub scale: f64,
    pub regions: Vec<MaskRegionStats>,  // one per attempted masking region (includes failures)
}

pub struct MaskRegionStats {
    pub rect: Rect,                     // the masking box (upstream MaskFittingResults.mask_box)
    pub std_deviation: f64,             // border std dev of the *chosen* candidate
    pub failed: bool,                   // std_deviation > masker.mask_max_standard_deviation
    pub thickness: Option<u32>,         // None when the box mask was chosen
}
```

Regions whose precise mask was blank (detector false positive) are **absent** from this list — upstream drops those `None` fitments before building `MaskData` (`masker.py:68`, `save_denoising_data`).

### 2.7 Analytics

```rust
pub struct OcrAnalytic {
    pub path: PathBuf,                                 // original image path
    pub num_boxes: usize,                              // boxes considered before removal
    pub box_areas_ocred: Vec<i64>,
    pub box_areas_removed: Vec<i64>,
    pub removed: Vec<RemovedBox>,                      // (text, box in ORIGINAL image coords)
}
pub struct RemovedBox { pub text: String, pub rect: Rect }

pub struct MaskFittingAnalytic {
    pub path: PathBuf, pub fit_found: bool, pub candidate_index: usize,
    pub std_deviation: f64, pub thickness: Option<u32>,
}
pub struct DenoiseAnalytic { pub path: PathBuf, pub std_deviations: Vec<f64>, pub boxes_denoised: usize }
pub struct DetectAnalytic { pub path: PathBuf, pub blocks_detected: usize, pub blocks_kept: usize }
```

### 2.8 Step / Output enums

Mirror upstream `output_structures.py` minus inpainting:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Step { Detect = 1, Preprocess, Mask, Denoise, Export }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Output {
    // Detect
    BaseImage, RawMask, RawJson,
    // Preprocess
    CleanJson,
    // Mask
    BoxMask, CutMask, FinalMask, MaskOverlay, IsolatedText, MaskedOutput, MaskDataJson,
    // Denoise
    DenoiseMask, DenoisedOutput,
}

impl Output { pub fn step(self) -> Step; pub fn cache_suffix(self) -> &'static str; }
```

`cache_suffix` values are the upstream file suffixes verbatim (`_base.png`, `_raw_mask.png`, `#raw.json`, `#clean.json`, `_box_mask.png`, `_cut_mask.png`, `_combined_mask.png`, `_with_masks.png`, `_text.png`, `_clean.png`, `#mask_data.json`, `_noise_mask.png`, `_clean_denoised.png`). Keeping them identical means a user can diff our cache against upstream's during parity work — cheap and useful.

### 2.9 Errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum StageError {
    #[error("io error at {path}: {source}")] Io { path: PathBuf, #[source] source: std::io::Error },
    #[error("failed to decode image {path}: {source}")] Decode { path: PathBuf, #[source] source: image::ImageError },
    #[error("unsupported image format: {0}")] UnsupportedFormat(String),
    #[error("model error: {0}")] Model(String),
    #[error("inference failed: {0}")] Inference(String),
    #[error("invalid stage input: {0}")] InvalidInput(String),
    #[error("serialization error: {0}")] Serde(#[from] serde_json::Error),
    #[error("image handle is not materialized on disk")] UnmaterializedHandle,
    #[error("stage produced no usable output: {0}")] Empty(String),
}
```

`StageError` is **never** used for "this box was noise" or "this mask didn't fit" — those are normal outcomes recorded in analytics. Only conditions that make the *image* unprocessable are errors.

---

## 3. The stage contract

Per the decisions doc, each stage crate exposes one pure function. Refinement: **model handles are not serializable**, so stages that need a model take it as a second, non-serde parameter. This keeps `StageInput` fully `serde`-derivable (the requirement the decisions doc actually cares about) while letting an `ort` session live outside the data contract.

```rust
// pc-core
pub trait Stage {
    type Input:  Serialize + DeserializeOwned;
    type Output: Serialize + DeserializeOwned;
    type Ctx<'a>;                       // () for stages with no external resources
    const STEP: Step;
    fn run(input: Self::Input, ctx: Self::Ctx<'_>) -> Result<Self::Output, StageError>;
}
```

Concrete per stage:

| Stage | Crate | `Ctx<'a>` | Free function |
|---|---|---|---|
| Detect | `pc-detect` | `&'a dyn TextDetector` | `pc_detect::run(input, detector)` |
| Preprocess | `pc-preprocess` | `Option<&'a dyn OcrEngineFactory>` | `pc_preprocess::run(input, ocr)` |
| Mask | `pc-mask` | `()` | `pc_mask::run(input)` |
| Denoise | `pc-denoise` | `()` | `pc_denoise::run(input)` |
| Export | `pc-export` | `()` | `pc_export::run(input)` |

Every stage `run` is **pure with respect to global state**: no reading of env/config files, no logging side effects beyond `tracing`, no writing to paths not present in its input. It is *not* pure with respect to disk (it may read `ImageHandle` paths and write to destination paths given in its input) — this is deliberate, so large images don't have to be held in memory for a whole batch.

Config structs are passed **by value inside the `Input`** (they are `serde`-derivable), so a checkpointed JSON records exactly which settings produced it. This is what makes v2's staleness detection possible later, and costs nothing now.

---

## 4. Cross-stage data flow

### 4.1 Two modes, one code path

`pc-pipeline` drives everything. It has exactly one run function per pipeline, parameterized by a mode:

```rust
pub enum Checkpointing {
    /// Persist every intermediate to the cache dir; enables --skip-*, --resume, --cache-masks.
    Disk,
    /// Keep intermediates in memory; write only the requested final exports.
    /// Debug/overlay outputs are unavailable in this mode.
    Memory,
}
```

Default: `Disk` (matches upstream; needed for `--skip-*` and for the export stage's discovery step). `Memory` is selected automatically when the run is a single image, no debug outputs are requested, and `--no-cache` is passed. **Both modes execute the identical stage functions** — the only difference is whether `pc-pipeline` calls `materialize()` on the outputs between stages.

### 4.2 Cache paths

```rust
pub struct CachePaths { cache_dir: PathBuf, uuid: Uuid, stem: String }
impl CachePaths {
    pub fn new(original: &Path, cache_dir: &Path) -> Self;                 // fresh uuid
    pub fn from_existing(path_with_uuid: &Path, cache_dir: &Path) -> Result<Self, StageError>;
    pub fn for_output(&self, out: Output) -> PathBuf;   // {cache}/{uuid}_{stem}{suffix}
}
```

Format `{uuid}_{stem}{suffix}` mirrors upstream `OutputPathGenerator` (`output_structures.py:269`), including uuid-as-clobber-protection and uuid recovery via `stem.split('_')[0]`. Export mode drops the uuid (`export_mode=True` upstream) — modelled as `ExportPaths` (separate type, no uuid).

### 4.3 Concrete chaining (single image, `Disk` mode)

```
                       user input file  ──────────────────────────────┐
                                                                      │ (kept for export
 [split?]  pc_imageops::calculate_best_splits + split_image           │  metadata + scale-up)
      │  (only if general.split_long_strips and the image qualifies)   │
      ▼                                                               │
 DetectInput { source: ImageHandle(input or segment), target_height_*, base_png_dest,
               raw_mask_dest, original_path }
      │  pc_detect::run(input, &detector)
      ▼
 DetectOutput { page: PageDataRaw, analytics }
      │  pipeline: write page → {uuid}_{stem}#raw.json ; base_image/raw_mask already written
      ▼
 PreprocessInput { page: PageDataRaw, config: PreprocessorConfig, performing_ocr: false }
      │  pc_preprocess::run(input, ocr_factory)
      ▼
 PreprocessOutput { page: PageData, ocr_analytic }
      │  pipeline: write page → #clean.json
      ▼
 MaskInput { page: PageData, original_image: ImageHandle, config: MaskerConfig,
             extract_text, debug_outputs, dests: MaskDests { combined_mask, clean, text, box_mask, cut_mask, overlay } }
      │  pc_mask::run(input)
      ▼
 MaskOutput { mask_data: MaskData, cleaned: ImageHandle, combined_mask: ImageHandle,
              text_layer: Option<ImageHandle>, analytics: Vec<MaskFittingAnalytic> }
      │  pipeline: write mask_data → #mask_data.json
      ▼
 DenoiseInput { mask_data: MaskData, masked_image: ImageHandle /* _clean.png */,
                config: DenoiserConfig, dests: DenoiseDests { noise_mask, clean_denoised } }
      │  pc_denoise::run(input)
      ▼
 DenoiseOutput { denoised: ImageHandle, noise_mask: ImageHandle, analytics }
      │  pipeline: (if split) stitch segment outputs back together
      ▼
 ExportInput { original_path, export_path, output_dir, outputs, sources: ExportSources, general: GeneralConfig,
               denoising_enabled }
      │  pc_export::run(input)
      ▼
 ExportOutput { files_written: Vec<PathBuf> }
```

**Field provenance table** (this is the "dependencies on other stages" answer, in one place):

| Consumer field | Producer |
|---|---|
| `PreprocessInput.page` | `DetectOutput.page` (or `#raw.json` on disk when `--skip-text-detection`) |
| `MaskInput.page` | `PreprocessOutput.page` (or `#clean.json`) |
| `MaskInput.page.base_image` / `.raw_mask` | written by detect |
| `MaskInput.original_image` | the user's input file (needed when `scale != 1`) |
| `DenoiseInput.mask_data` | `MaskOutput.mask_data` (or `#mask_data.json`) |
| `DenoiseInput.masked_image` | `MaskOutput.cleaned` (`_clean.png`) |
| `DenoiseInput.mask_data.combined_mask` | `MaskOutput.combined_mask` (`_combined_mask.png`) |
| `ExportInput.sources` | `MaskOutput` + `DenoiseOutput` handles (or discovered from cache, §12.3) |

### 4.4 Resume / skip

`--skip-text-detection`, `--skip-preprocess`, `--skip-mask`, `--skip-denoise` each mean "the artifacts for this step already exist in the cache; load them instead of computing". The pipeline resolves the starting step, then for each image loads the JSON checkpoint for `Step::prev()` via `CachePaths::from_existing`. Loading a checkpoint whose `schema_version` is unknown, or whose referenced `ImageHandle` paths are missing, is a **per-image error** (skip that image, continue) — not a fatal one.

### 4.5 Batch parallelism

`rayon` `par_iter` over images at the *whole-pipeline* granularity (not per stage), with a semaphore-free bound: `min(config.max_threads_or_cpus, images.len())`. Rationale: upstream parallelizes per-stage with a process pool because Python needs it; per-image parallelism in Rust gives better cache locality and makes per-image error isolation trivial. The detect stage is the exception: a single `ort` session is shared across threads (`ort::Session` is `Send + Sync` for CPU EP; sessions are cheap to share and expensive to duplicate), with `text_detector.concurrent_models` controlling how many sessions are created (default 1, shared).

---

## 5. Error-handling philosophy

**Rule: one bad image never kills a batch.**

1. **Per-image isolation.** `pc-pipeline` processes each image inside a boundary that produces:
   ```rust
   pub enum ImageOutcome {
       Completed { original: PathBuf, files_written: Vec<PathBuf>, analytics: ImageAnalytics },
       Skipped   { original: PathBuf, reason: SkipReason },   // e.g. unsupported format, no text found
       Failed    { original: PathBuf, step: Step, error: StageError },
   }
   ```
   On `Failed`, later stages for that image are **not** attempted; the error is logged at `ERROR` with the image path and step, and the run continues with the next image.
2. **Panics are contained.** Each per-image unit is wrapped in `catch_unwind`; a panic becomes `Failed { error: StageError::Inference("panicked: ...") }`. Justification: `ort`/FFI and third-party pixel code can panic on malformed input, and one malformed page must not abort a 500-page batch.
3. **Fatal (abort) conditions** — these exit before/independently of per-image work: unreadable/invalid config, no input images found, model file missing or hash mismatch (and download unavailable), cache dir not creatable, output dir not writable.
4. **`--fail-fast`** flips (1) to abort-on-first-`Failed`, for CI and debugging.
5. **Exit codes:** `0` all images completed (skips allowed); `2` at least one image failed but others succeeded; `1` fatal condition. A summary table is always printed listing failed/skipped images with reasons.
6. **Sub-image conditions are never errors:** blank precise mask for a box (detector noise) → `WARN`, box dropped; mask fit exceeded `mask_max_standard_deviation` → recorded `failed: true`, box left uncleaned, image still exported; zero boxes on a page → `Skipped { NoTextDetected }` (upstream produces an export with an empty mask; we follow upstream and still export the untouched image, but mark the outcome so the summary is honest — the exported file is byte-equivalent to a copy of the input for `masked_output`).
7. **Determinism requirement:** identical inputs + identical config must produce identical outputs, including ordering of boxes and of analytics entries, regardless of thread count. This forbids the `set`-iteration nondeterminism upstream has in `resolve_overlaps` (§9.3, §14.2).

---

## 6. Config (`pc-config`)

TOML via `toml_edit`, one `Profile` per file, plus an app-level `Config`. v1 sections and defaults — **exactly** upstream's values (`config.py`), minus `[inpainter]` (v1.5) and minus GUI/post-action keys (v2):

```toml
[general]
preferred_file_type          = ""        # empty = keep original suffix
preferred_mask_file_type     = ".png"
input_height_lower_target    = 1000
input_height_upper_target    = 4000
split_long_strips            = true
preferred_split_height       = 2000
split_tolerance_margin       = 500
long_strip_aspect_ratio      = 0.33
merge_after_split            = true
max_threads                  = 0         # 0 = all cores
always_cache_masks           = false

[text_detector]
model_path                   = ""        # empty = use managed cache
concurrent_models            = 1

[preprocessor]
box_min_size                 = 400       # 20*20
suspicious_box_min_size      = 40000     # 200*200
box_overlap_threshold        = 20.0
ocr_enabled                  = true
ocr_language                 = "detect_box"   # detect_box | detect_page | jpn | eng
reading_order                = "auto"         # auto | manga | comic
ocr_max_size                 = 3000      # 30*100
ocr_blacklist_pattern        = "[～．ー！？０-９~.!?0-9-]*"
ocr_strict_language          = false
box_padding_initial          = 2
box_right_padding_initial    = 3
box_padding_extended         = 5
box_right_padding_extended   = 5
box_reference_padding        = 20

[masker]
mask_growth_step_pixels      = 2
mask_growth_steps            = 11
min_mask_thickness            = 4
allow_colored_masks           = true
off_white_max_threshold       = 240
mask_max_standard_deviation   = 15.0
mask_improvement_threshold    = 0.1
mask_selection_fast           = false
debug_mask_color              = [108, 30, 240, 127]

[denoiser]
denoising_enabled             = true
noise_min_standard_deviation  = 0.25
noise_outline_size            = 5
noise_fade_radius             = 1
colored_images                = false
filter_strength               = 10
color_filter_strength         = 10
template_window_size          = 7
search_window_size            = 21
```

Validation (fail config load, not per-image): `mask_growth_step_pixels >= 1`, `mask_growth_steps >= 1`, `min_mask_thickness >= 0`, `0.0 <= mask_improvement_threshold < 1.0`, `mask_max_standard_deviation > 0`, `0 <= box_overlap_threshold <= 100`, `template_window_size` and `search_window_size` odd and `>= 3`, `off_white_max_threshold` in `0..=255`, `ocr_blacklist_pattern` compiles as a regex, `long_strip_aspect_ratio > 0`.

Missing keys take defaults; unknown keys produce a `WARN` and are **preserved** on round-trip (that is why `toml_edit`). `panel-ocr profile new/show/edit/validate` subcommands wrap this.

---

## 7. Fixtures and the mocked ML boundary

### 7.1 Vendored upstream fixtures (task **F0**, prerequisite for everything)

Copy into `tests/fixtures/upstream/` with a `ATTRIBUTION.md` (source repo, commit, GPL-3 notice):

```
demo_bubbles/{black,darkrays,handwritten,nightmare,ray,spikey,square}_bubble_{raw,clean}.png
long_strip.jpg
ocr_output/good_detected_text.csv
ocr_output/good_detected_text.txt
```

Verified properties (measured, use as test constants): all `demo_bubbles` are 8-bit **grayscale** PNGs; `raw`/`clean` pairs are the same size —
`black 202×319`, `darkrays 208×320`, `handwritten 72×132`, `nightmare 219×343`, `ray 256×329`, `spikey 354×354`, `square 144×270`. `long_strip.jpg` is `1000×8000` RGB, progressive JPEG, 300 dpi.

### 7.2 Recorded fixtures — how CI mocks ML (task **F1**, maintainer-run)

CI must not run models (confirmed scope), but the masking golden tests need real detector output. Solution: **record once, replay forever.**

`cargo xtask record-fixtures` (maintainer machine, real models present) writes to `tests/fixtures/recorded/`:

- For each `demo_bubbles/*_raw.png`: `<name>_base.png`, `<name>_raw_mask.png`, `<name>#raw.json` (a `PageDataRaw` JSON).
- For one or two full manga pages the maintainer supplies (kept small, ≤ 400 KB each, license-clean): the same triple.
- The recorded JSON's `ImageHandle` paths are stored **relative to the fixtures root** and rebased on load by `pc-testkit`.

`pc-detect` ships `ReplayDetector` (reads a recorded `_raw_mask.png` + block list keyed by input file stem) and `MockDetector` (programmable: fixed blocks + a synthetic mask), both behind `#[cfg(any(test, feature = "testkit"))]` plus a `--detector=replay:<dir>` hidden CLI flag so end-to-end CLI tests run with zero model files.

The one-time recording is a **checked-in artifact**, reviewed like code. This is what makes the demo_bubbles goldens usable in CI.

### 7.3 Golden calibration gate (ordering requirement)

Upstream's `*_clean.png` images were produced by a specific PanelCleaner version/profile we cannot fully verify. Per CLAUDE.md, tests are frozen once written — so tolerance numbers must be *measured before freezing*, not guessed after. Therefore:

> **Task F2 (must complete before the Stage-3 golden test is written):** implement `cargo xtask calibrate-goldens`, run it against the reference implementation available at that moment, and record measured deltas in `docs/GOLDEN_CALIBRATION.md`. The tolerances proposed in §10.7 are the *specified* values; if calibration shows they cannot be met for a specific fixture, the discrepancy goes back to the two architects jointly (per CLAUDE.md) **before** the test is frozen — never silently loosened afterwards.

To keep the project unblocked regardless, each stage's acceptance criteria are split into:
- **(A) Primary gates** — synthetic, exact, implementation-independent. These are the real correctness contract; they cannot be blocked by third-party fixture uncertainty.
- **(B) Parity gates** — the upstream goldens with tolerances. These prove we actually match PanelCleaner.

---

## 8. STAGE 1 — Text detection (`pc-detect`)

### 8.1 Crate and location

`crates/pc-detect`. Features: `onnx` (default; pulls `ort`), `testkit` (mock/replay backends). Module layout:

```
src/lib.rs            // Stage impl + run()
src/detector.rs       // trait TextDetector, RawDetection
src/onnx.rs           // #[cfg(feature="onnx")] ort session, letterbox, tensor plumbing
src/yolo.rs           // NMS + decode + class→language
src/mask.rs           // U-Net mask postprocess + refinement
src/resize.rs         // calculate_new_size_and_scale + area resize
src/mock.rs           // #[cfg(feature="testkit")] MockDetector, ReplayDetector
```

### 8.2 Contract

```rust
pub struct DetectInput {
    pub schema_version: u32,
    pub source: ImageHandle,            // the input file, or a strip segment
    pub original_path: PathBuf,         // path recorded in PageDataRaw (segment path when split)
    pub target_height_lower: u32,       // general.input_height_lower_target
    pub target_height_upper: u32,       // general.input_height_upper_target
    pub base_image_dest: Option<PathBuf>,  // where to write the scaled PNG (None in Memory mode)
    pub raw_mask_dest: Option<PathBuf>,
    pub min_mask_coverage: f32,         // 0.1, fixed constant in v1 (see 8.3 step 6)
}

pub struct DetectOutput { pub page: PageDataRaw, pub analytics: DetectAnalytic }

pub trait TextDetector: Send + Sync {
    /// `image`: RGB8, already scaled to the target height range.
    fn detect(&self, image: &RgbImage) -> Result<RawDetection, StageError>;
}

pub struct RawDetection {
    /// Boxes in `image` coordinates, confidence-filtered and NMS'd, in detector order.
    pub blocks: Vec<RawBlock>,
    /// U-Net text-probability mask, 8-bit, same size as `image`.
    pub mask: GrayImage,
}
pub struct RawBlock { pub rect: Rect, pub class_index: u8, pub confidence: f32 }
```

### 8.3 Algorithm

**1 — Load and normalise.** Decode `source` (supported: png, jpg/jpeg, webp, tiff/tif, bmp/dib, jp2, ppm — upstream's `SUPPORTED_IMG_TYPES`). Convert to RGB8. Multi-page TIFFs are rejected with `Skipped { UnsupportedFormat }` (upstream also rejects them).

**2 — Resize to target height.** Port `ctd_interface.calculate_new_size_and_scale` exactly:

```
if lower <= 0 || upper <= 0 || height <= upper        -> (width, height, 1.0)
else if lower >= upper                                -> scale = upper/height; new_h = upper;
                                                          new_w = round(width*scale)
else:
   inv_lower = height/lower;  inv_upper = height/upper
   n = ceil(inv_upper)
   if n <= inv_lower  -> scale = 1/n; new_h = round(height*scale); new_w = round(width*scale)
   else:
      max_h = round(height/inv_upper); min_h = round(height/inv_lower)
      new_h = floor(max_h/4)*4;  if new_h < min_h { new_h = max_h }
      scale = new_h/height;  new_w = round(width*scale)
```

Deviation: upstream's `lower >= upper` branch sets `new_height = height_target_lower` while computing `scale` from `upper` — inconsistent when `lower > upper`. We use `upper` for both (identical when `lower == upper`, which is the only sane config). Documented in §14.1. `round` = round-half-away-from-zero (Python `round()` is banker's rounding; the inputs here make the difference unobservable except at exact `.5`, so we specify half-away-from-zero and test the boundary explicitly).

Resample with an **area-average (box) filter** matching `cv2.INTER_AREA` for downscaling: output pixel = mean of the source rectangle `[x*sx, (x+1)*sx) × [y*sy, (y+1)*sy)` with fractional edge weights, computed in f32, rounded half-away-from-zero to u8. Do **not** use `image::imageops::resize(Lanczos3/Triangle)` — they differ visibly. (`image`'s `FilterType::Triangle` is not INTER_AREA; a dedicated ~40-line implementation is required.)

Write the scaled RGB image to `base_image_dest` as PNG (compression default) when present.

**3 — Inference.** `TextDetector::detect`. ONNX backend specifics (`src/onnx.rs`):

- Model: upstream's `comictextdetector.pt.onnx`
  (`https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx`,
  sha256 `1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f`) — resolved/downloaded/verified by `pc-models`.
- Preprocess (port of `inference.preprocess_img` + `letterbox`): letterbox to `1024×1024` with `stride=64, auto=false`, i.e. scale by `r = min(1024/h, 1024/w)` (no upscaling beyond `r=1`... upstream `letterbox` default `scaleup=True`; keep upstream behaviour: allow upscale), resize with bilinear, pad **right/bottom** with `(114,114,114)` to reach the padded size; record `(dw, dh)` = total padding in x/y. Channel order: RGB, NCHW, `f32 / 255.0`.
- Outputs (3): `blks` `[1, N, 5 + n_classes]` (n_classes = 3), `mask` `[1, 1, H, W]`, `lines_map` `[1, 2, H, W]`. If the second output has 2 channels and the third has 1, swap them (upstream guards for this: `inference.py:181-185`). Bind by index, but log the actual output names/shapes once at `DEBUG` so a model swap is diagnosable.
- Execution provider: CPU only, `intra_threads = 1` (parallelism is at the image level), session created once and shared.

**4 — YOLO postprocess** (`src/yolo.rs`, port of `postprocess_yolo` + yolov5 `non_max_suppression`):

- Candidate filter: `objectness (col 4) > 0.4` (`conf_thresh`).
- Score per class: `objectness * class_prob`; take the best class → `class_index`, `confidence`; then **drop candidates whose best score is `<= 0.4`** (upstream `yolov5_utils.py:241`, strict `>`) — this is a second, separate filter from the objectness gate above and is easy to miss. Upstream also caps survivors at `max_det = 300` after NMS — reproduce the cap.
- Box decode: columns 0..4 are `cx, cy, w, h` → `x1 = cx - w/2`, `y1 = cy - h/2`, `x2 = cx + w/2`, `y2 = cy + h/2`.
- NMS: greedy, IoU threshold `0.35`, sorted by descending score, **class-agnostic** — **decided** (§15.1): verified upstream runs *per-class* NMS (`inference.py:115` + `yolov5_utils.py:261`, `agnostic=False`), which can emit duplicate overlapping boxes for one balloon detected under two language classes. v1 deliberately deviates to class-agnostic NMS to remove the duplicate at the source; the surviving box's language is used as-is. This is documented as deviation §14.13. Box-count divergence from upstream is an accepted cost — our detector regression tests (§8.7(B)9) lock against our own recorded fixtures, not upstream's exact box count.
- Rescale to base-image coords: `resize_ratio = (im_w / (1024 - dw), im_h / (1024 - dh))`; multiply x-coords by `.0`, y-coords by `.1`; truncate to i32 (`astype(np.int32)`).
- `confidence` rounded to 3 decimals (upstream `np.round(..., 3)`), for stable golden JSON.
- `class_index → language`: `0 => Some(English)`, `1 => Some(Japanese)`, `2 => None`; any other index → `None` + `WARN`.

**5 — Mask postprocess + refinement** (`src/mask.rs`):

- Raw U-Net output → `squeeze` → multiply by 255 → clamp → u8 (`postprocess_mask` with `thresh=None`).
- Crop off the letterbox padding: `mask[0..H-dh, 0..W-dw]`, then resize to base-image size with **bilinear** (upstream uses `cv2.INTER_LINEAR` here).
- **Refinement (v1 = "simple" mode, ported from koharu's `refine_segmentation_mask`)**:
  1. For each detected block, `expanded = rect.pad(16, image_size)` (upstream `expand_textwindow(expand_r=16)`).
  2. Rasterize the union of expanded rects into an in-bounds mask.
  3. `base[p] = 255` iff `in_bounds[p] != 0 && raw_mask[p] > 60`.
  4. Dilate `base` with an L1 (diamond) structuring element of radius 3.
  5. Clip the dilated result back to `in_bounds` (zero outside).
  If `blocks` is empty, the refined mask is all-zero.
- The refined mask is what `PageDataRaw.raw_mask` points at (upstream also stores the refined one: `ctd_interface.py:182`).
- **Out of scope for v1:** upstream's `refine_mask` (top-k colour masklists + per-channel Otsu candidates + connected-component XOR merge + hole filling, `comic_text_detector/utils/textmask.py:18-210`) and `refine_undetected_mask`. Rationale: the masker's own growth-and-score loop is what determines final quality; a coarser precise mask degrades fit granularity but cannot break the pipeline, and the XOR-merge algorithm is the single riskiest numerical port in the whole project. Keep the door open with `enum MaskRefineMode { Simple, Annotation }` in `TextDetectorConfig` where only `Simple` is implemented in v1 (`Annotation` → `StageError::InvalidInput`). See §15.2.

**6 — False-positive filter.** For each block, `mask_coverage = mean(refined_mask over rect) / 255.0`; drop blocks with `mask_coverage < 0.1`. Provenance: upstream applies this `mask_score_thresh = 0.1` test in `group_output` to blocks that received no DBNet text lines (`textblock.py:485-490`); since v1 emits no line polygons (below), *every* block is line-less and the filter applies to all of them — which is exactly the code path upstream takes when the line map yields nothing.

**7 — Emit** `PageDataRaw` (surviving blocks in NMS order) + `DetectAnalytic`.

**Explicitly out of scope for stage 1 in v1:** DBNet line-polygon extraction (`SegDetectorRepresenter`), line→block assignment, block splitting at distance gaps, `examine_textblk` orientation/font-size estimation, scattered-line synthesis, the `_raw_boxes.png` debug visualization (needs font rendering), GPU EPs, multi-model concurrency > shared session, torch `.pt` loading.

### 8.4 Dependencies on other stages

None (first stage). Consumes: user input file + `[general]`/`[text_detector]` config.

### 8.5 Task breakdown

| ID | Task | Kind |
|---|---|---|
| **D1** | `pc-models`: model path resolution, download with progress, sha256 verify, atomic rename, `--model-path` override, offline error message | simple |
| **D2** | `resize.rs`: `calculate_new_size_and_scale` + INTER_AREA box-filter resize | **heavy** (numerical) |
| **D3** | `trait TextDetector`, `RawDetection`, `DetectInput/Output`, `MockDetector`, `ReplayDetector` | simple |
| **D4** | `onnx.rs`: `ort` session setup, letterbox preprocess, output binding/swap-guard | **heavy** (model adapter) |
| **D5** | `yolo.rs`: conf filter, xywh→xyxy, class-agnostic NMS, rescale, class→language | **heavy** (numerical) |
| **D6** | `mask.rs`: mask squeeze/scale/crop/bilinear-resize + Simple refinement (threshold/expand/dilate/clip) | **heavy** |
| **D7** | `lib.rs`: `run()` wiring, mask-coverage filter, `PageDataRaw` assembly, analytics | simple |
| **D8** | `pc-imageops::split`: `calculate_best_splits` + `split_image` + `stitch_images` | **heavy** (numerical) |
| **D9** | Strip-split orchestration in `pc-pipeline`: segment cache dir, `splits.json` manifest, merge-at-export bookkeeping | simple |

Batching: `{D1, D3}` one call; `{D7, D9}` one call; `D2`, `D4`, `D5`, `D6`, `D8` isolated.

### 8.6 Fixtures

- `long_strip.jpg` → **D8** (splitting) and **D2** (resize: `8000 → 4000`, `scale = 0.5`).
- `demo_bubbles/*_raw.png` → **D2** (no-resize path: all heights ≤ 4000 → `scale == 1.0`, dimensions unchanged, pixels unchanged).
- `tests/fixtures/recorded/*` → **D7** (replay end-to-end, deterministic `#raw.json`).
- Synthetic: a 1024×1024 letterbox round-trip fixture for **D4** (exercise `dw/dh` when aspect ≠ 1), a synthetic probability mask + boxes for **D6**, a synthetic candidate array for **D5**.

### 8.7 Acceptance criteria

**(A) Primary**

1. `calculate_new_size_and_scale` matches this table exactly (derived from the algorithm; `w=1000` throughout):

   | h | lower | upper | → (w, h, scale) | branch |
   |---|---|---|---|---|
   | 3000 | 1000 | 4000 | (1000, 3000, 1.0) | no-op (`h <= upper`) |
   | 8000 | 1000 | 4000 | (500, 4000, 0.5) | integer inverse scale n=2 |
   | 5000 | 1000 | 4000 | (500, 2500, 0.5) | integer inverse scale n=2 |
   | 4300 | 2000 | 2100 | (488, 2100, 2100/4300) | multiple-of-4 branch |
   | 4500 | 4000 | 4000 | (889, 4000, 4000/4500) | exact-size branch |
   | 8000 | 0 | 4000 | (1000, 8000, 1.0) | disabled |
   | 8000 | 1000 | 0 | (1000, 8000, 1.0) | disabled |

2. INTER_AREA resize of a synthetic 4×4 → 2×2 image equals the exact 2×2 block means (bit-exact, hand-computed); a 1000×8000 → 500×4000 resize of `long_strip.jpg` has mean absolute difference ≤ 1.0 vs a recorded `cv2.INTER_AREA` reference (recorded once via xtask) and max per-channel delta ≤ 2. Justification: identical mathematics, differing only in float accumulation order and rounding of the `.5` case.
3. NMS on a hand-built candidate set (three overlapping boxes with known IoUs straddling 0.35, one below objectness `0.4`, one with objectness `> 0.4` but best class-score `<= 0.4` — must also be dropped) yields exactly the expected surviving boxes and classes.
4. Refinement on a synthetic 64×64 probability mask with one box: pixels at value 61 survive, at 60 do not (strict `>`); no output pixel lies outside `rect.pad(16)`; a lone pixel at value 200 **outside** every expanded box is zeroed.
5. Coverage filter: a block whose refined-mask mean is 25/255 (≈0.098) is dropped; 26/255 (≈0.102) is kept.
6. `run()` with `ReplayDetector` over a recorded fixture produces a `PageDataRaw` whose JSON is byte-identical across 10 runs and across 1 vs 8 rayon threads. **`insta` snapshot, decided (§15.10) — allowed here under mandatory safeguards:** (a) the first accepted snapshot must be reviewed against a hand-traced expected value, with reviewer/date/method recorded in `docs/GOLDEN_CALIBRATION.md`, before commit; (b) the snapshot is subject to the frozen-tests rule exactly like a hand-written test — `cargo insta accept` is forbidden in CI, and any change requires joint-architect sign-off (state this as a comment at the `assert_json_snapshot!` call site); (c) no other snapshot tests may be added without the same sign-off.
7. `split_image` then `stitch_images` on `long_strip.jpg` reproduces the source **bit-exactly** (as raw pixels, before re-encode).

**(B) Parity**

8. `calculate_best_splits(long_strip.jpg, preferred=2000, tolerance=500, split=true, max_ar=0.33)` returns exactly **3** split rows; split *i* ∈ `[2000*i − 500, 2000*i + 500)` for i∈{1,2,3}; and for each returned row `y`, its squared-horizontal-difference score is ≤ the 10th percentile of the scores in its search range (i.e. it lands in the low-change band, which is the algorithm's actual contract, and is robust to tie-breaking differences). Aspect-ratio gate: the same call with `max_ar = 0.1` returns `[]` (since 1000/8000 = 0.125 > 0.1).
9. On the recorded page fixture, box count and each box's coordinates match the recorded `#raw.json` exactly (this is a regression lock, not a Python-parity claim).

---

## 9. STAGE 2 — Preprocessing (`pc-preprocess`)

### 9.1 Crate and location

`crates/pc-preprocess`. Modules: `lib.rs` (stage + `run`), `filter.rs` (size/language filtering), `merge.rs` (overlap resolution), `order.rs` (reading order), `ocr_filter.rs` (OCR-based discard). `pc-ocr` is a dependency only for the `OcrEngine` trait; the concrete engine is injected.

### 9.2 Contract

```rust
pub struct PreprocessInput {
    pub schema_version: u32,
    pub page: PageDataRaw,
    pub config: PreprocessorConfig,
    /// true for `panel-ocr ocr` runs: boxes the OCR engine can't handle are discarded outright.
    pub performing_ocr: bool,
}

pub struct PreprocessOutput {
    pub page: PageData,
    pub ocr_analytic: Option<OcrAnalytic>,     // Some iff an OCR engine was supplied
}

// pc-ocr
pub trait OcrEngine: Send + Sync {
    fn languages(&self) -> &[Language];
    fn recognize(&self, crop: &DynamicImage) -> Result<String, StageError>;
}
pub trait OcrEngineFactory: Send + Sync {
    /// `None` language = unknown; the factory picks a best-effort engine or returns None.
    fn engine_for(&self, lang: Option<Language>) -> Option<&dyn OcrEngine>;
}
```

### 9.3 Algorithm (exact order — port of `preprocessor.prep_json_file`)

**1 — Language assignment.**
- If `ocr_language` is `detect_box` or `detect_page`: keep each block's detected language (`Some(Japanese)`/`Some(English)`/`None`).
- Otherwise: overwrite **every** block's language with the configured language.

**2 — Block filtering** (in detector order, keeping order):
- If `performing_ocr && language.is_none() && ocr_strict_language` → drop.
- If `rect.area() < box_min_size` → drop.
- If `language.is_none() && rect.area() < suspicious_box_min_size` → drop.

**3 — Page language.** `page_language = mode(languages.filter(is_some))`, ties broken by **first occurrence** (Python `Counter.most_common(1)` is insertion-order-stable). If none, `None`. If `ocr_language == detect_page`, set every box's language to `page_language`.

**4 — Merge mutually-centered boxes** (`resolve_total_overlaps`): FIFO queue of tight rects in order; pop front as `b`; find all remaining `q` with `b.overlaps_center(q)`; merge each into `b` (removing them from the queue); push `b` to results. Repeat. **Language of a merged box** = the language of the popped (earliest) box; upstream mutates `boxes` without touching `box_language`, silently desynchronizing the parallel lists when a merge happens. Our `TextBox` pairing makes that impossible, so we specify: earliest box's language wins. See §14.3.

**5 — Initial padding.** `rect = rect.pad(box_padding_initial, image_size)` then `rect = rect.right_pad(box_right_padding_initial, image_size)` for every tight box.

**6 — Reading-order sort.** `x_factor = -0.4`, `y_factor = 1.0`; if `reading_order == Comic` **or** (`reading_order == Auto` and `page_language` ∉ `RTL_BOX_ORDER_LANGUAGES`) then `x_factor = +0.4`. Sort tight boxes ascending by `x_factor * x1 as f64 + y_factor * y1 as f64`. Sort must be **stable** (Rust's `sort_by` is; Python's `sorted` is) so equal keys keep detector order.

**7 — OCR discard pass** (only when an `OcrEngineFactory` is supplied and `ocr_enabled`):
For each box (in order) with `rect.area() < ocr_max_size`:
- crop `base_image` to `rect`;
- `text = factory.engine_for(box.language)?.recognize(&crop)?`;
- record `rect.area()` in `box_areas_ocred`;
- if `Regex::new(ocr_blacklist_pattern)` **fully matches** `text` with `(?s)` (DOTALL) → drop the box, and record `RemovedBox { text, rect: rect.scale(1.0 / page.scale) }` (original-image coordinates) plus `rect.area()` in `box_areas_removed`.
`num_boxes` in the analytic = number of boxes *before* removal (upstream records `len(boxes)` after the `None`-punching loop, which is the pre-removal length — reproduce that value). If no box is small enough to be a candidate, emit `OcrAnalytic { num_boxes: boxes.len(), .. empty vectors }`.
Engine errors: a single `recognize` failure is logged at `WARN` and the box is **kept** (fail-open — never delete content because OCR broke). Deviation from upstream, which would propagate the exception; justified by §5's isolation principle.

**8 — Extended boxes.** `extended = tight.map(rect)` then `pad(box_padding_extended)` then `right_pad(box_right_padding_extended)`.

**9 — Merge extended → masking regions** (`resolve_overlaps(threshold = box_overlap_threshold)`): FIFO queue **in index order** (upstream uses a `set`, hence nondeterministic — we fix this, §14.2); pop front as `b`; find all remaining `q` with `b.overlaps(q, threshold)`; merge and remove; push. Note: upstream re-tests only against the *remaining* queue (no re-scan after merging), so a chain `A~B~C` where `A` and `C` don't overlap merges A+B, and C separately if it no longer overlaps the merged box. Reproduce this exactly — one pass, no transitive re-scan.

**10 — Reference boxes.** For each merged rect: `reference = merged.pad(box_reference_padding, image_size)`. Emit `MaskingRegion { masking: merged, reference }` preserving order.

**11 — Emit** `PageData` + analytics.

**Explicitly out of scope for stage 2 in v1:** the `_boxes.png` / `_boxes_final.png` visualizations (font rendering), OCR review/edit workflow (`OCRResult`, `OCRStatus`), Tesseract engine, `ocr_use_tesseract`/`ocr_engine` selection beyond "manga-ocr or mock" (§15.4), split-analytic merging (`merge_ocr_analytics`) — that lives in the pipeline for `ocr` mode and is v1 only in the simple "offset y by cumulative segment heights" form (D9).

### 9.4 Dependencies on other stages

From `DetectOutput.page`: `base_image` (for OCR crops), `raw_mask` (passed through untouched), `scale` (for original-coordinate analytics), `image_size` (padding clamps), `blocks` (rects + languages), `original_path`.

### 9.5 Task breakdown

| ID | Task | Kind |
|---|---|---|
| **P1** | `PreprocessInput/Output`, `PageData`/`TextBox`/`MaskingRegion` types + serde + invariant asserts | simple |
| **P2** | `filter.rs`: language assignment/override, size + suspicious-size filters, page-language mode | simple |
| **P3** | `merge.rs`: `resolve_total_overlaps` and `resolve_overlaps` (deterministic FIFO, single pass) | simple |
| **P4** | `order.rs`: reading-order key + stable sort, auto/manga/comic resolution | simple |
| **P5** | Padding tiers + `MaskingRegion` construction + `run()` wiring | simple |
| **P6** | `pc-ocr`: `trait OcrEngine`/`OcrEngineFactory`, `MockOcrEngine` (scripted responses), blacklist-regex filter logic + analytics | simple |
| **P7** | `pc-ocr`: manga-ocr ONNX backend (encoder/decoder sessions, greedy decode loop, vocab, preprocessing) — port from koharu `manga_ocr` | **heavy** |

Batching: `{P1, P2, P3, P4, P5}` one sequential call (all pure geometry/boilerplate on the same data structure); `{P6}` one call; `P7` isolated.

### 9.6 Fixtures

- **Synthetic rects** for P2–P5: hand-built `PageDataRaw` values with known areas/overlaps. These are the primary tests — geometry needs no images.
- `ocr_output/good_detected_text.csv` / `.txt` → these define the **OCR report format** (2 boxes for `img1.jpg` with the first at `(100,100)-(300,200)`, area 20 000; `page1.jpg` 2 lines, `page2.jpg` 3 lines). They are **not** filter-behaviour fixtures — the filter is a regex over engine output. Assigned to **E4** (§12) as format goldens, and reused in P6 only as the source of the "area == 20_000" arithmetic check. **Decided (§15.5), confirmed:** `run_ocr` — the code path that actually produces this format — sets `ocr_blacklist_pattern = ".*"` and `ocr_max_size = 10**10` (`main.py:866-868`), making the filter inert there; the fixtures cannot be filter-behavior tests.
- `demo_bubbles/handwritten_bubble_raw.png` (72×132, area 9 504 < `ocr_max_size` 3 000? no — 9 504 > 3 000) — use `demo_bubbles/handwritten_bubble_raw.png` cropped to a 40×70 region as a small-box OCR candidate crop for P6's `MockOcrEngine` plumbing test (the crop content is irrelevant; the mock returns scripted text).
- Recorded page fixture → P5 end-to-end determinism.

### 9.7 Acceptance criteria

**(A) Primary**

1. Given blocks with areas `399, 400, 39_999(lang=None), 40_000(lang=None)` and `box_min_size=400`, `suspicious_box_min_size=40_000`: exactly `400` (with a language) and `40_000` survive. (`<` is strict in both filters.)
2. `overlaps` boundary: two boxes with `intersection / min_area` exactly `0.20` and `threshold = 20.0` do **not** merge (strict `>`); `0.2001` does. Zero-area box does not panic (divisor forced to 1).
3. `resolve_total_overlaps`: three boxes where A and B contain each other's centers and C is disjoint → 2 boxes, A∪B first (input order preserved), and the merged box's language equals A's.
4. `resolve_overlaps` single-pass semantics: for A(0,0,100,100), B(90,0,190,100), C(180,0,280,100) with a threshold such that A~B and B~C but not A~C, the output is exactly `[A∪B, C]` — **not** `[A∪B∪C]`.
5. Reading order: for boxes at `(0,0)`, `(500,0)`, `(0,500)`: `Manga`/`Auto`+Japanese order is `(500,0), (0,0), (0,500)`; `Comic`/`Auto`+English order is `(0,0), (500,0), (0,500)`. Two boxes with equal keys retain detector order.
6. Padding clamps: a box touching the right edge, `right_pad(5)` on a 100-wide image yields `x2 == 100`, never 105.
7. Invariants hold for every generated `PageData`: `extended_boxes.len() == text_boxes.len()`; every `reference` ⊇ `masking`; every rect within `image_size`.
8. Blacklist filter: with `ocr_blacklist_pattern = "[～．ー！？０-９~.!?0-9-]*"`, engine output `""`, `"..."`, `"！？"`, `"123"` all → dropped; `"こんにちは"`, `"A1"`, `"1 2"` (contains a space) → kept. Empty-string match confirms `*` semantics with full-match.
9. Analytics: `RemovedBox.rect` is in original-image coordinates — with `scale = 0.5` a box at `(10,10,30,30)` is recorded as `(20,20,60,60)`.
10. Determinism: `run()` on a fixed input produces byte-identical `PageData` JSON across 100 iterations (guards against any accidental `HashSet` ordering).

**(B) Parity**

11. On the recorded page fixture with `MockOcrEngine` returning `""` for every crop, the resulting `PageData` box tiers are locked as an `insta` JSON snapshot. **Decided (§15.10) — same three safeguards as §8.7(B)9 apply**: hand-traced review documented in `docs/GOLDEN_CALIBRATION.md` before commit; `cargo insta accept` forbidden in CI, changes need joint-architect sign-off; no further snapshot tests without the same process.

---

## 10. STAGE 3 — Masking (`pc-mask`)

**This is the heart of the project.** It is where PanelCleaner differs from every other cleaner: rather than inpainting, it grows the AI mask outward and scores the *uniformity of the pixels the mask's own border sits on*, picking the largest mask whose border is most uniform, then fills that mask with the border's median colour. If nothing is uniform enough, the box is left alone.

### 10.1 Crate and location

`crates/pc-mask`. Modules: `lib.rs` (stage + `run`), `boxmask.rs`, `grow.rs` (kernels + iterative dilation), `border.rs` (edge extraction + border std dev + median colours), `fit.rs` (`pick_best_mask`), `combine.rs` (composition, cleaned image, text layer). Shared primitives (`BinaryMask`, kernels, composition) live in `pc-imageops`; `pc-mask` owns the *fitting policy*.

### 10.2 Contract

```rust
pub struct MaskInput {
    pub schema_version: u32,
    pub page: PageData,
    /// The user's original (unscaled) image — needed to produce the cleaned output when scale != 1.
    pub original_image: ImageHandle,
    pub config: MaskerConfig,
    pub extract_text: bool,
    pub debug_outputs: bool,          // --cache-masks
    pub dests: MaskDests,
}

pub struct MaskDests {
    pub combined_mask: Option<PathBuf>,   // _combined_mask.png  (RGBA)
    pub cleaned: Option<PathBuf>,         // _clean.png
    pub text_layer: Option<PathBuf>,      // _text.png           (iff extract_text)
    pub box_mask: Option<PathBuf>,        // _box_mask.png       (iff debug_outputs)
    pub cut_mask: Option<PathBuf>,        // _cut_mask.png       (iff debug_outputs)
    pub mask_overlay: Option<PathBuf>,    // _with_masks.png     (iff debug_outputs)
}

pub struct MaskOutput {
    pub mask_data: MaskData,
    pub combined_mask: ImageHandle,       // RGBA, base_image size
    pub cleaned: ImageHandle,             // original size
    pub text_layer: Option<ImageHandle>,
    pub analytics: Vec<MaskFittingAnalytic>,
}

// pc-imageops
pub struct BinaryMask { w: u32, h: u32, bits: Vec<u8> /* 1 byte per px, 0 or 1 */ }
```

`BinaryMask` uses one byte per pixel (not bit-packed): simpler, and dilation is memory-bandwidth bound anyway. It carries `and`, `or`, `crop_into(rect, target_size, offset)`, `is_blank`, `bbox`, `to_gray`, `from_gray_threshold`.

### 10.3 Algorithm

**Step 0 — Load.** `base = page.base_image.load()` (RGB or L as stored), `precise = BinaryMask::from_gray_threshold(page.raw_mask.load(), 0)` — upstream does `mask.convert("1", dither=NONE)`, which for an 8-bit grayscale image thresholds at **> 127**. ⚠️ Precision point: PIL's `L → 1` conversion without dither uses `value > 127`. Our refined mask is strictly 0 or 255, so both `> 0` and `> 127` agree; specify `> 127` for upstream parity and to be safe with hand-authored fixtures.

**Step 1 — Box mask.** Rasterize `page.extended_boxes` into a `BinaryMask` of `image_size`: for each rect set all pixels in `[x1..=x2] × [y1..=y2]` to 1 — **inclusive on both ends**, because upstream uses `ImageDraw.rectangle`, which is inclusive of `x2,y2`. (This one-pixel difference from crop semantics is real; test it.) Write as PNG if `dests.box_mask`.

**Step 2 — Cut mask.** `cut = precise AND box_mask`. Write if `dests.cut_mask`.

**Step 3 — Per-region fit.** For each `MaskingRegion { masking, reference }` **in order**, run `fit_region` (below). It returns `Option<Fitment>`; `None` means "the precise mask was blank here → detector noise → drop this region entirely" (no analytics entry, no `MaskData` entry).

`fit_region(base, cut, box_mask, masking, reference, config) -> Option<Fitment>`:

1. `x_offset = masking.x1 - reference.x1`, `y_offset = masking.y1 - reference.y1` (both ≥ 0; `reference` ⊇ `masking`).
2. `base_crop = base.crop(reference)` — this is the *analysis canvas*; all candidate masks live in its coordinate frame.
3. `precise_cut = cut.crop(masking)` pasted at `(x_offset, y_offset)` into a zeroed mask of `base_crop`'s size.
4. **If `precise_cut` is blank → return `None`** (upstream logs a warning naming the image and box; do the same at `WARN`).
5. `box_candidate = box_mask.crop(masking)` pasted the same way. Its `thickness` is `None` (it isn't grown from the precise mask).
6. **Generate growth candidates** (`grow.rs`, port of `make_mask_steps_convolution`):
   - `pad = max(min_mask_thickness, mask_growth_step_pixels) * 2`.
   - Create a **replicate-padded** buffer of `precise_cut` with `pad` pixels on all four sides (`np.pad(mode="edge")`).
   - Candidate 0: dilate the padded buffer **in place** with `kernel(min_mask_thickness)`; yield the center crop; `thickness = min_mask_thickness`.
   - Candidates 1..`mask_growth_steps-1`: dilate the same padded buffer in place with `kernel(mask_growth_step_pixels)` each time; yield the center crop; `thickness = min_mask_thickness + i * mask_growth_step_pixels`.
   - Total candidates from growth: exactly `mask_growth_steps` (11 by default).
   - Why in-place on the padded buffer: upstream reuses `padded_mask` across iterations, so border-replication effects accumulate; dilating the padded buffer reproduces that exactly. (Upstream's `convolve2d` + `> 0` threshold **is** binary dilation for a non-negative kernel, so binary dilation is an exact — and vastly faster — reformulation, not an approximation. This equivalence must be stated in a code comment.)
   - `kernel(thickness)`: `diameter = thickness*2 + 1`.
     - If `diameter <= 5`: a full `diameter × diameter` square of 1s with the **four corners set to 0**.
     - Else: OpenCV `MORPH_ELLIPSE` of size `(diameter, diameter)`, reproduced exactly as:
       ```
       r = c = thickness
       for i in 0..diameter:
           dy = i - r
           dx = round_to_nearest( (c as f64) * sqrt(((r*r - dy*dy) as f64) * (1.0/(r*r) as f64)) )
           j1 = max(c - dx, 0); j2 = min(c + dx + 1, diameter)
           row i is 1 on [j1, j2), else 0
       ```
       (This is `cv::getStructuringElement`'s code path verbatim; `saturate_cast<int>` from `double` rounds to nearest.) For `thickness == 0` the kernel is the single center pixel and dilation is identity.
7. **Candidate ordering** (this determines everything downstream):
   - `mask_selection_fast == true`: `[box_candidate, growth_0, growth_1, ...]`, and **break out of scoring** as soon as a candidate's deviation is exactly `0.0`.
   - `mask_selection_fast == false` (default): `[growth_0, ..., growth_n-1, box_candidate]`.
8. **Score each candidate** with `border_std_deviation(base_crop, candidate, off_white_max_threshold, allow_colored_masks)` → `(std_dev: f64, median_color: [u8;3])`. If any candidate yields an **empty edge set**, the whole region is abandoned: **return `None`** (upstream's `BlankMaskError` propagates out of `pick_best_mask` as `None`).
9. **Select** (upstream `image_ops.py:678-686`) — iterate candidates in order, tracking `best`:
   ```
   accept candidate i  iff  i == 0
                        or  dev_i <= best_dev * (1.0 - mask_improvement_threshold)
   ```
   On accept, update `best_dev`, `best_color`, `best_mask`, `best_thickness`. Note the consequences, which are the *intent* of the algorithm and must not be "fixed": a larger mask only wins if it improves the border uniformity by ≥ 10 %; and `best_dev * (1 - t)` with a `best_dev` of `0.0` means nothing can ever beat a perfect mask.
10. **Failure threshold.** If `best_dev > mask_max_standard_deviation` → return `Fitment { mask: None, .. }` (analytics + `MaskData` entry still produced, `failed: true`). Otherwise `Fitment { mask: Some(best_mask), .. }`.
11. `Fitment` fields: `mask: Option<BinaryMask>`, `median_color: [u8;3]`, `coords: (i32,i32) = (reference.x1, reference.y1)`, `std_deviation: f64`, `candidate_index: usize` (index of the chosen candidate in the candidate list), `thickness: Option<u32>`, `masking_rect: Rect`.

**`border_std_deviation`** (`border.rs`, port of `image_ops.py:467-522`):

1. If `!allow_colored_masks`: convert `base_crop` to 8-bit luma **using PIL's `L` conversion coefficients**: `round(0.299*R + 0.587*G + 0.114*B)` — PIL uses the ITU-R 601-2 luma transform with integer truncation (`L = (R*299 + G*587 + B*114) / 1000`, truncating). Specify: `((r as u32 * 299 + g as u32 * 587 + b as u32 * 114) / 1000) as u8`. (The `image` crate's `to_luma8` uses different coefficients — **do not** use it here.)
2. **Edge extraction.** Upstream applies PIL's `FIND_EDGES` (3×3 kernel `[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]`) to the mode-`1` mask and then selects pixels where the result is truthy. With values in `{0,255}`, and PIL leaving the outermost 1-pixel ring **unfiltered (copied)**, this reduces exactly to:
   ```
   edge(p) = mask[p] == 1  &&  ( p is on the 1-pixel image border
                                 || any of the 8 neighbours of p is 0 )
   ```
   (Proof: for a set pixel with `k` set neighbours the response is `255*(8-k)`, nonzero iff `k < 8`; for a clear pixel the response is ≤ 0 and clamps to 0; border pixels are copied through unchanged.) Implement this definition directly — do **not** implement a generic convolution.
3. `border_pixels` = the `base_crop` values at edge positions, gathered in **row-major order** (matters for the median-heuristic's tie behaviour).
4. If `border_pixels.is_empty()` → `Err(BlankMask)`.
5. **Grayscale path** (`!allow_colored_masks`, or a grayscale base):
   - `std` = **population** standard deviation (`ddof = 0`, i.e. divide by `n`) of the u8 values as `f64`.
   - `median` = NumPy-style median: sort; odd `n` → middle element; even `n` → `(a + b) / 2.0`; then truncate toward zero to `i32` (Python `int()`), then `median_color = [m, m, m]`.
6. **Colour path**:
   - `mean_color` = per-channel `f64` mean.
   - `distances[i] = ||color_i − mean_color||₂`.
   - `std` = **sample** standard deviation of `distances` (`ddof = 1`, divide by `n − 1`). If `n == 1`, NumPy yields `NaN` with a warning — specify: if `n < 2`, `std = 0.0` (a single border pixel is trivially uniform). Deviation, §14.4.
   - `median_color`: **heuristic first** — if any exact RGB triple occurs in **more than** `n/2` of the samples, use it. Otherwise the **geometric median** via Weiszfeld: start at the mean; each iteration compute distances, drop zero-distance points, weights `= (1/d) / Σ(1/d)`, new estimate `= Σ w_i * p_i`; stop when `||new − old|| < 1e-5` or after 500 iterations; if all points coincide with the current estimate, return it. Truncate each channel toward zero to `u8`.
7. **Off-white snap**: if `min(median_color) > off_white_max_threshold` → `median_color = [255,255,255]`.
8. Return `(std, median_color)`.

**Precision rule — decided (§15.9): `f64` only, no exceptions.** Every std-dev/median/Weiszfeld computation in `border.rs` and `fit.rs` (this section and §10.3 step 3's candidate selection) must use `f64`; `f32` must not appear in either module. Verified upstream: `color_std` explicitly casts to `np.float64` (`image_ops.py:456`), and the candidate-selection comparison (§10.3 step 3.9, `dev_i <= best_dev * (1 - threshold)`) is pure `f64` upstream (`image_ops.py:681-683`). This comparison determines which mask candidate is painted, so a borderline `f32` rounding flip would change visible output, not just an internal number. Enforce via clippy lint or explicit review at PR time.

**Step 4 — Combine** (`combine.rs`):

- `combined_mask` = RGBA image of `image_size`, all `(0,0,0,0)`. For each `Fitment` with `mask: Some(m)` **in region order**: build an RGBA layer where `m[p] == 1` → `(median_color.r, .g, .b, 255)` else `(0,0,0,0)`, and **alpha-composite** it onto `combined_mask` at `coords`. (`alpha_composite`, not `paste`: overlapping opaque layers — the later one wins; identical result to `paste` here since alpha is 0 or 255, but keep the operation named correctly.) Write to `dests.combined_mask` as PNG.
- `cleaned`: if `page.scale != 1.0`, load `original_image` and resize `combined_mask` to the original size with **nearest-neighbour**; else clone `base`. Then composite `combined_mask` over it using the mask's own alpha. Write to `dests.cleaned`. **Output mode — decided (§15.3):** always composite internally in RGB; at write time, if the base/original image is `L` and every `median_color` used is achromatic (`r == g == b`), convert to `L` before writing, otherwise write `RGB`. This is export-equivalent to upstream (whose own base image is always 3-channel internally, `cv2.IMREAD_COLOR`; grayscale is restored only at export) rather than upstream-identical, but is lossless for the achromatic case that always holds on grayscale pages (verified against the `black_bubble` golden: fill is exactly `(0,0,0)`).
- `text_layer` (iff `extract_text`): transparent RGBA canvas of the original size; the mask resized to it with nearest-neighbour if needed; paste the **original** image through the mask (keeping only what the mask covers, i.e. the text). Write to `dests.text_layer`.
- `mask_overlay` (iff `debug_outputs`): `base` with `combined_mask` recoloured to `masker.debug_mask_color` and composited. This is the only debug visualization in v1 (no text/font rendering).

**Step 5 — Emit** `MaskData` (one `MaskRegionStats` per `Fitment`, including failures, in region order) and `Vec<MaskFittingAnalytic>`.

**Explicitly out of scope for stage 3 in v1:** any inpainting fallback for failed masks (v1.5); `_mask_fitments.png` (per-candidate colour-cycled visualization) and `_std_devs.png` (σ/thickness text annotations) — both need font rendering, deferred to v1.5 with the GUI work; PSD/layered output; per-candidate mask caching in `MaskOutput` (memory blowup — `Fitment.debug_masks` upstream keeps all 12 candidates per box; we keep only the chosen one).

### 10.4 Dependencies on other stages

From `PreprocessOutput.page`: `masking_regions` (the fit loop's iteration space), `extended_boxes` (box-mask raster), `base_image`, `raw_mask`, `scale`, `image_size`, `original_path`. From the pipeline: `original_image` handle, `MaskerConfig`.

### 10.5 Task breakdown

| ID | Task | Kind |
|---|---|---|
| **M1** | `pc-imageops::BinaryMask` (+ and/or/crop-into-offset/blank/bbox/to_gray/from_gray_threshold) and box-mask rasterization (inclusive rect) | simple |
| **M2** | `grow.rs`: `kernel(thickness)` (small-square-with-cut-corners + exact OpenCV ellipse) and iterative in-place dilation on a replicate-padded buffer, producing the candidate sequence | **heavy** |
| **M3** | `border.rs`: PIL-parity luma conversion, edge extraction, grayscale std/median, colour std, heuristic median, Weiszfeld geometric median, off-white snap | **heavy** |
| **M4** | `fit.rs`: `fit_region` — offsets, crops, blank detection, candidate ordering (fast/slow), scoring loop with early break, improvement-threshold selection, failure threshold, `Fitment` | **heavy** |
| **M5** | `combine.rs`: RGBA layer build, alpha composition, cleaned-image production (incl. scale != 1 + output-mode rule), text-layer extraction | simple |
| **M6** | `lib.rs`: `run()` wiring, `MaskData`/analytics emission, debug artifact writes, rayon over regions | simple |

Batching: `{M1}` may batch with `{M5, M6}` only after M2–M4 land (they depend on the candidate/fitment types); recommended order M1 → M2 → M3 → M4 → {M5, M6}. M2, M3, M4 each isolated.

Parallelism note: regions within a page may be fitted with `rayon` (`par_iter().map(fit_region)` then collect **in order**) — but only after the sequential version passes, and the ordered collect is mandatory for determinism.

### 10.6 Fixtures

- `demo_bubbles/*_raw.png` + `*_clean.png` → the **calibration fixtures** (7 pairs; see the non-gating status of §10.7(B) above), driven by `tests/fixtures/recorded/<name>_raw_mask.png` + `<name>#raw.json` (§7.2) so no model runs in CI. Each fixture stresses a different case: `square` (trivial), `handwritten` (thin strokes, small canvas), `black` (dark bubble → non-white median colour → exercises the off-white snap *not* firing, and is the frozen gate in item 16), `ray`/`darkrays` (radial lines crossing the border → high border σ → exercises the improvement threshold), `spikey` (non-convex balloon → exercises growth clipping), `nightmare` (highest-σ stress case). Measured diffs show all 7 pairs have substantial changed-pixel counts (e.g. `nightmare` 16,873/75,117 px) — do not assume any demo fixture exercises the "leave it untouched, `failed: true`" path; that path is covered instead by the synthetic primary gate §10.7(A)9.
- Synthetic images for M1–M4 primary gates (below).

### 10.7 Acceptance criteria

**(A) Primary — exact, synthetic**

1. **Kernels.** `kernel(1)` = 3×3 ones with corners zeroed (cross-ish, 5 set). `kernel(2)` = 5×5 ones with corners zeroed (21 set). `kernel(3)` = the exact 7×7 OpenCV ellipse:
   rows (as set-pixel ranges over j∈0..7): `dy=-3 → dx=0 → [3,4)`; `dy=-2 → dx=round(sqrt(5))=2 → [1,6)`; `dy=-1 → dx=round(sqrt(8))=3 → [0,7)`; `dy=0 → dx=3 → [0,7)`; mirrored below. Total set = 1+5+7+7+7+5+1 = 33. Assert the full matrix.
2. **Dilation.** A single set pixel at the center of a 21×21 mask, dilated with `kernel(4)` (9×9 ellipse), equals the kernel footprint translated to the center — bit-exact. Two successive dilations with `kernel(2)` equal one dilation with the Minkowski sum of the two kernels (validated against a hand-computed 9×9 reference), confirming the in-place accumulation semantics.
3. **Candidate count and thicknesses.** With `min_mask_thickness=4`, `mask_growth_step_pixels=2`, `mask_growth_steps=11`: growth candidates have thicknesses `[4,6,8,...,24]` (11 values), plus the box candidate with `thickness == None`; total 12 candidates. Ordering is box-first iff `mask_selection_fast`.
4. **Edge extraction.** For a 5×5 mask with a 3×3 set block at the center: exactly the 8 boundary pixels of the block are edges (the block's center is not). For a 3×3 mask fully set: **all 9** pixels are edges (all on the 1-px image border). For an empty mask: zero edges → `BlankMask`.
5. **Border std dev, grayscale, hand-computed.** A 5×5 base with a known value pattern and a 3×3 centered mask: assert `std` to within `1e-12` of the hand-computed population std of the 8 border-adjacent base values, and `median` equal to the hand-computed truncated median. Include an even-count case (8 values) to pin the `(a+b)/2` then truncate rule (e.g. values whose two middles are 100 and 101 → median 100, **not** 101).
6. **Off-white snap.** median `(241,241,241)` with `off_white_max_threshold = 240` → `(255,255,255)`; `(240,255,255)` → unchanged (uses `min`).
7. **Colour std path.** Border pixels `[(0,0,0) ×3, (255,255,255) ×1]`: heuristic median fires (3 > 4/2) → `(0,0,0)`; `std` = sample std of the four distances-to-mean, hand-computed. Border pixels `[(0,0,0), (255,255,255)]` (2 samples, no majority) → Weiszfeld runs; assert the result is within 1 of the true geometric median `(127,127,127)` region and that it terminates in < 500 iterations.
8. **Selection policy.** Candidate deviations `[10.0, 9.5, 8.0, 8.0]` with `mask_improvement_threshold = 0.1`: index 0 accepted (dev 10); index 1 needs `≤ 9.0` → rejected; index 2 needs `≤ 9.0` → accepted (dev 8); index 3 needs `≤ 7.2` → rejected. Chosen index = 2. With `[0.0, 0.0]`: index 1 needs `≤ 0.0` → **accepted** (`<=` is inclusive) → chosen index 1. This edge case must be locked, since it decides whether a perfect fit prefers the largest mask.
9. **Failure threshold.** All deviations `[20.0, 18.0]` with `mask_max_standard_deviation = 15.0` → `Fitment.mask == None`, `failed == true`, and `MaskData` still contains the region with `std_deviation == 18.0` and the chosen `thickness`.
10. **Blank precise mask.** A region whose `cut` crop is empty → `fit_region` returns `None`; the region is absent from both `MaskData.regions` and `analytics`; a `WARN` is emitted; `run()` succeeds.
11. **Fast mode early break.** With `mask_selection_fast = true` and a box candidate scoring exactly `0.0`, the scoring loop evaluates exactly **one** candidate (assert via an instrumented counter or by constructing a growth candidate that would panic if scored).
12. **Composition.** Two overlapping fitments with different median colours composite deterministically (later region wins in the overlap); `combined_mask` alpha is exactly 0 or 255 everywhere; `cleaned` differs from `base` **only** inside the union of chosen masks (assert pixel-set equality).
13. **Scale != 1.** With `scale = 0.5`, `cleaned` has the original image's dimensions and the mask edges land on even pixel boundaries (nearest-neighbour 2× upscale of a binary mask contains only 2×2 blocks).
14. **Determinism.** Byte-identical `_combined_mask.png` and `#mask_data.json` across 20 runs and across 1 vs 8 threads.

**(B) Parity — the demo_bubbles goldens**

> **Decided (Fable, §15.2): this is a non-gating calibration report, not a frozen test.** v1 ships `MaskRefineMode::Simple` (a dilated U-Net mask), not upstream's full `refine_mask` pixel-accurate refinement — the two will produce different precise-mask footprints, so exact-parity tolerances against `*_clean.png` are not a valid frozen gate. The `demo_bubbles` fixtures are `media/` README assets of unverified producing version/profile and are never used by upstream's own test suite, which reinforces that they are unsuitable as a hard pass/fail lock. The **frozen** test is item 15 below, run by `cargo xtask calibrate-goldens` (task F2) and recorded in `docs/GOLDEN_CALIBRATION.md` for visibility — it must run and its numbers must be recorded, but it does not gate CI. Item 16 remains a frozen gate (it is implementation-independent of refinement mode).

15. *(Calibration report, non-gating)* For each of the 7 `<name>_raw.png` / `<name>_clean.png` pairs, run the full masking stage with the recorded detector output and the **default profile**, then compare `cleaned` against `<name>_clean.png` and record, per fixture, in `docs/GOLDEN_CALIBRATION.md`:
    - **Shape metric:** let `G = { p : raw[p] != clean[p] }` (the pixels upstream changed) and `O = { p : raw[p] != ours[p] }`. Record `IoU(G, O)` and whether `O ⊆ dilate(G, 2px)`.
    - **Value metric:** record `%` of pixels exactly equal, the max per-pixel delta among non-equal pixels, and global SSIM (8×8 windows, grayscale).
    - Target reference values (not gates): `IoU ≥ 0.99`, `≥ 99.5 %` exact, remainder within `±2`, `SSIM ≥ 0.995` — if measured numbers fall short, that is expected evidence of the Simple-vs-full-refinement difference, not a build failure, and should be written up rather than chased.

    **Tolerance rationale (for the recorded reference values above).** Every operation in this stage is either exact integer work (rect rasterization, binary dilation, edge extraction, alpha composition of 0/255 alpha) or an integer-valued reduction (median → `u8`); the only genuine float sensitivities are the Weiszfeld geometric median (±1 per channel after truncation) and the std-dev `<=` comparison (a sub-`1e-12` difference could flip a borderline candidate). Those sensitivities motivate the specific numbers above, but since the precise-mask *input* itself differs by refinement mode, hitting them is a bonus signal, not a requirement.

16. *(Frozen gate)* `black_bubble` specifically: the chosen `median_color` must be dark (`max channel ≤ 40`) and the off-white snap must **not** fire — this catches a class of bug where a naive implementation always fills white. This gate is independent of mask-refinement mode (it only checks the selected fill color), so it stays frozen and gating.

---

## 11. STAGE 4 — Denoising (`pc-denoise`)

### 11.1 Crate and location

`crates/pc-denoise`. Modules: `lib.rs` (stage + `run`), `noise_mask.rs` (grow/fade/composite). The NLM implementation itself lives in `pc-imageops::nlm` (it is a general-purpose image algorithm, and putting it there keeps it independently benchmarkable).

### 11.2 Contract

```rust
pub struct DenoiseInput {
    pub schema_version: u32,
    pub mask_data: MaskData,
    pub original_image: ImageHandle,     // = mask_data.original_path, resolved
    pub masked_image: ImageHandle,       // stage 3 `_clean.png`; used only for the 1-bit shortcut
    pub config: DenoiserConfig,
    pub dests: DenoiseDests,
}
pub struct DenoiseDests { pub noise_mask: Option<PathBuf>, pub denoised: Option<PathBuf> }

pub struct DenoiseOutput {
    pub denoised: ImageHandle,           // RGB(A)/L, original size
    pub noise_mask: ImageHandle,         // RGBA, original size (blank if nothing denoised)
    pub analytics: DenoiseAnalytic,
}
```

### 11.3 Algorithm (port of `denoiser.denoise_page` + `image_ops.generate_noise_mask`)

**1 — 1-bit shortcut.** If the original image's mode is 1-bit (`image` reports `L1`/`Gray(1)`), denoising is pointless: copy `masked_image` to `dests.denoised`, emit a fully transparent noise mask of the original size, and return `DenoiseAnalytic { std_deviations: vec![], boxes_denoised: 0 }`.

**2 — Base canvas.** `cleaned = original.to_rgb8()`; `mask = mask_data.combined_mask.load()` (RGBA). If sizes differ, `scale_up = cleaned.width as f64 / mask.width as f64` and resize `mask` to `cleaned`'s size with **nearest-neighbour**; else `scale_up = 1.0`. Composite `mask` over `cleaned` (this reproduces stage 3's clean output at full resolution rather than trusting `_clean.png`, exactly as upstream does).

**3 — Select regions.** `boxes_to_denoise = mask_data.regions.filter(|r| !r.failed && r.std_deviation > config.noise_min_standard_deviation)` — strictly greater; order preserved. Intuition: a *perfect* fit (σ ≈ 0) means the surrounding area was uniform, so there is no JPEG noise ring to hide; a *failed* fit means nothing was painted, so there is nothing to blend.

**4 — Per region:**
1. `rect = r.rect.scale(scale_up)`.
2. `image_cutout = cleaned.crop(rect)`; `mask_cutout = mask.crop(rect)`.
3. **Binary mask extraction from the RGBA cutout: use the alpha channel** (`alpha > 0`). ⚠️ Upstream converts the RGBA cutout with `.convert("L")`, which computes the *luma of the RGB channels and discards alpha* — for a black-filled mask (e.g. `black_bubble`) the luma is 0 and the noise mask silently vanishes. That is an upstream bug, not a design intent (the intent, per the docstring, is "grow the mask and fade its edges"). v1 uses alpha. Documented in §14.5, flagged in §15.6.
4. `grown = dilate(mask_cutout_binary, kernel(config.noise_outline_size))` using the **same** `kernel()` from `pc-imageops` (`noise_outline_size = 5` → 11×11 ellipse). `size == 0` → identity.
5. `faded = gaussian_blur(grown.to_gray(), config.noise_fade_radius)` — an 8-bit `L` image. Gaussian parity: PIL's `GaussianBlur(radius=r)` is a **three-pass box-blur approximation**, not a true Gaussian. Specify a true separable Gaussian with `sigma = radius` truncated at `3*sigma` (with `radius = 1` the difference is a few levels on a soft edge, well inside the stage tolerance). Documented in §14.6.
6. `denoised_cutout = pc_imageops::nlm::denoise(image_cutout, params)`.
7. Attach `faded` as the alpha channel of `denoised_cutout` → an RGBA layer.
8. Record `(layer, (rect.x1, rect.y1))`.

**5 — Combine.** RGBA canvas of `cleaned`'s size; alpha-composite each layer at its coords → `noise_mask`. If there were no regions, `noise_mask` is a fully transparent RGBA image of the original size. Then composite `noise_mask` over `cleaned` → `denoised`. Write both.

**6 — Analytics.** `std_deviations` = the σ of **all** `mask_data.regions` (not just the denoised ones — upstream reports all, because the CLI histogram shows them against the `noise_min_standard_deviation` cutoff); `boxes_denoised` = number of regions processed.

**NLM specification** (`pc-imageops::nlm`, replacing `cv2.fastNlMeansDenoising`):

```rust
pub struct NlmParams { pub h: f32, pub template_window: u32, pub search_window: u32 }
pub fn denoise(img: &DynamicImage, p: NlmParams) -> DynamicImage;
```

Algorithm (joint-channel classic NLM):
- Let `C` = channel count (1 for `L`, 3 for `RGB`), `t = template_window / 2`, `s = search_window / 2` (both windows must be odd; validated in config).
- Border handling: `BORDER_REFLECT_101` (i.e. `abcd | cba`), matching OpenCV's default in `fastNlMeansDenoising`.
- For each output pixel `p`:
  - For each `q` in the `search_window × search_window` neighbourhood of `p`:
    `d(p,q) = ( Σ_{o ∈ template} Σ_{c<C} (I_c(p+o) − I_c(q+o))² ) / (C · template_window²)`
    `w(p,q) = exp( −max(d(p,q) − 2σ²_est, 0) / h² )` with **`σ²_est = 0`** in v1 (OpenCV's `fastNlMeansDenoising` does not subtract a noise-variance term; specify `w = exp(−d/h²)`).
  - `out_c(p) = Σ_q w(p,q)·I_c(q) / Σ_q w(p,q)`; the self-term (`q == p`, `d = 0`, `w = 1`) is included.
  - Round half-away-from-zero, clamp to `0..=255`.
- Accumulate in `f32`; parallelize over output rows with `rayon`.
- Complexity is `O(W·H·s²·t²)`; a 200×300 crop with defaults is ~2.6 G multiply-adds worst case. Required optimization (mandatory, not optional): incremental integral-image update of `d` along the search axis (OpenCV's approach), or at minimum a precomputed per-row squared-difference table. Benchmark gate: a 256×256 grayscale crop with defaults must denoise in **< 400 ms** single-threaded on CI hardware.

**Deliberate v1 simplification — decided (§15.7):** `colored_images = true` in v1 maps to the **joint-channel RGB** path above (single `h = filter_strength`). The true `cv2.fastNlMeansDenoisingColored` — convert BGR→CIELAB, denoise `L` with `h` and `a,b` with `hColor`, convert back — is deferred to v1.5, at which point `color_filter_strength` becomes meaningful. The default profile has `colored_images = false` (verified `config.py:700`), so v1's default path is unaffected. **`pc-config` must emit a one-time `WARN`** when a loaded profile has `colored_images = true`, stating that v1 uses a joint-channel approximation and `color_filter_strength` is ignored until v1.5. Documented in §14.7.

**Explicitly out of scope for stage 4 in v1:** the Lab-space coloured NLM variant; `color_filter_strength`; any denoising outside the mask regions (this stage only ever touches pixels inside the grown+faded per-region masks); inpainting.

### 11.4 Dependencies on other stages

From `MaskOutput.mask_data`: `regions` (rects + σ + failed flags), `combined_mask` handle, `scale`, `original_path`. From `MaskOutput.cleaned`: the `_clean.png` handle (1-bit shortcut only). From the pipeline: the original image handle, `DenoiserConfig`.

### 11.5 Task breakdown

| ID | Task | Kind |
|---|---|---|
| **N1** | `pc-imageops::nlm`: joint-channel NLM with reflect-101 borders, incremental distance updates, rayon rows | **heavy** |
| **N2** | `pc-imageops`: separable Gaussian blur (`sigma = radius`, truncate 3σ) | simple |
| **N3** | `noise_mask.rs`: region selection, scale-up, crop, alpha-mask extraction, grow, fade, alpha attach, composite | simple |
| **N4** | `lib.rs`: `run()` wiring, 1-bit shortcut, blank-noise-mask path, analytics | simple |

Batching: `{N2, N3, N4}` one sequential call; `N1` isolated.

### 11.6 Fixtures

- `demo_bubbles/nightmare_bubble_raw.png` and `ray_bubble_raw.png` (noisy, high-σ) → N1 quality checks and N3's end-to-end region selection.
- `tests/fixtures/recorded/nlm/<name>_h10_t7_s21.png` — one-time recorded `cv2.fastNlMeansDenoising` reference outputs for two crops (recorded by `xtask record-fixtures`; the recording script is committed so the provenance is auditable) → the N1 parity gate.
- Synthetic: a constant-value image, a step-edge image, and a Gaussian-noise-on-flat image for the primary gates.
- `long_strip.jpg` → **not** used here.

### 11.7 Acceptance criteria

**(A) Primary**

1. **Constant image invariance.** NLM of a uniform image (any value, `L` or `RGB`) returns that exact image, bit-exact (all weights 1, mean = value).
2. **Noise reduction.** For a flat mid-gray image with additive Gaussian noise (σ = 12, fixed seed), the denoised output's standard deviation is **< 40 %** of the input's, and its mean is within `0.5` of the input's mean.
3. **Edge preservation.** For a sharp vertical step edge (0 | 255), the denoised output keeps the two plateaus within `±3` of 0 and 255 at ≥ 4 px from the edge, and the transition width does not exceed the template window.
4. **Border handling.** Denoising a 3×3 constant image with `search_window = 21` (larger than the image) does not panic and returns the constant (validates reflect-101 index clamping).
5. **Determinism / thread-invariance.** Bit-identical output across 1 and 8 rayon threads, 10 runs.
6. **Region selection.** With `noise_min_standard_deviation = 0.25` and regions `[σ=0.25 ok, σ=0.26 ok, σ=20 failed]`: exactly one region (σ=0.26) is denoised. Strictly-greater and the `failed` exclusion are both locked.
7. **1-bit shortcut.** A 1-bit original yields `denoised` byte-identical to `masked_image`, a fully transparent noise mask of the correct size, and empty analytics — with **zero** NLM invocations (instrumented counter).
8. **Scope containment.** For a synthetic case with one region, `denoised` differs from the stage-3 composite **only** within `region.rect.pad(noise_outline_size + 3*noise_fade_radius)`. Nothing outside is touched.
9. **Alpha-derived mask.** With a `combined_mask` whose fill colour is `(0,0,0,255)` (black bubble), the produced noise mask is **non-empty** — the regression test for the upstream luma bug (§14.5).
10. **Blank path.** Zero qualifying regions → a fully transparent `noise_mask` written, `denoised` equal to the stage-3 composite, `boxes_denoised == 0`.
11. **Performance.** The § 11.3 benchmark gate (< 400 ms for 256×256 grayscale, defaults, single-threaded), as a `#[test]` with a generous 3× margin so CI noise doesn't flake it.

**(B) Parity**

12. Against the recorded OpenCV references (same `h = 10`, `template = 7`, `search = 21`): **SSIM ≥ 0.98**, **mean absolute difference ≤ 1.0**, **max per-channel difference ≤ 8**.
    **Tolerance justification.** OpenCV quantizes its weights into a fixed-point lookup table indexed by a *bucketed* distance (`almostDist2Weight`, with `WEIGHT_THRESHOLD = 0.001` truncating far weights to zero) and accumulates in integers; we accumulate in `f32` with exact `exp`. That alone produces small systematic differences concentrated on high-gradient pixels, where a handful of near-threshold weights get dropped by OpenCV but kept by us. `max Δ ≤ 8` (≈3 % of range) bounds those isolated pixels; `mean |Δ| ≤ 1.0` proves there is no systematic bias; `SSIM ≥ 0.98` proves no structural error (a wrong search radius or a border bug would blow through all three). This directly instantiates decisions-doc resolution #4 (perceptual, not bit-exact).
13. End-to-end on a recorded page: `_noise_mask.png` and `_clean_denoised.png` are locked as size/mode/alpha-histogram assertions plus an SSIM ≥ 0.99 comparison against a committed snapshot of our own first accepted output (regression lock, refreshed only by explicit architect decision).

---

## 12. STAGE 5 — Export (`pc-export`)

### 12.1 Crate and location

`crates/pc-export`. Modules: `lib.rs` (stage + `run`), `formats.rs` (suffix→format, save options, metadata), `discover.rs` (which outputs exist / precedence), `ocr_report.rs` (CSV/TXT writers).

### 12.2 Contract

```rust
pub struct ExportInput {
    pub schema_version: u32,
    pub original_path: PathBuf,        // for metadata (dpi, mode) and naming
    pub export_path: PathBuf,          // logical output identity (differs from original for merged strips)
    pub output_dir: PathBuf,           // absolute => used as-is; relative => relative to export_path.parent()
    pub outputs: Vec<Output>,          // requested + available (already resolved by precedence)
    pub sources: ExportSources,
    pub preferred_file_type: Option<String>,       // None => keep original suffix
    pub preferred_mask_file_type: String,          // default ".png"
}

pub struct ExportSources {
    pub masked: Option<ImageHandle>,        // _clean.png
    pub denoised: Option<ImageHandle>,      // _clean_denoised.png
    pub final_mask: Option<ImageHandle>,    // _combined_mask.png
    pub denoise_mask: Option<ImageHandle>,  // _noise_mask.png
    pub isolated_text: Option<ImageHandle>, // _text.png
}

pub struct ExportOutput { pub files_written: Vec<PathBuf> }
```

### 12.3 Algorithm (port of `image_export.copy_to_output` + `discover_viable_outputs`)

**1 — Resolve destinations.**
- If `output_dir.is_absolute()`: `base = output_dir`, else `base = export_path.parent() / output_dir`.
- `mkdir -p base`.
- Cleaned output: `base / {stem}_clean{suffix}` where `suffix = preferred_file_type.unwrap_or(original_path.extension())`.
- Mask output: `base / {stem}_mask{preferred_mask_file_type}`.
- Text output: `base / {stem}_text{preferred_mask_file_type}`.
(These reproduce upstream's `OutputPathGenerator(export_mode=True)` suffixes: `_clean`, `_mask`, `_text`.)

**2 — Precedence.** Exactly one *cleaned* image and at most one *mask* image are exported, choosing the highest available stage:
`cleaned: denoised > masked`; `mask: denoise_mask > final_mask`; `text: isolated_text` (independent).
`--save-only-{cleaned,mask,text}` narrow the categories. `denoising_enabled == false` (or `--skip-denoise`) removes the denoise candidates.
In `Disk` mode with `--skip-*` flags, availability is discovered by probing cache paths (upstream globs `*#clean.json`; we instead iterate the pipeline's own per-image `CachePaths`, which is deterministic and avoids re-parsing JSON — same outcome, less I/O).

**3 — Cleaned image.** Load the chosen source, convert to the **original image's colour mode** (upstream: `image.convert(original.mode)` — so a grayscale input stays grayscale, a palette input is re-paletted), save with the format-specific options below, carrying over the original's `dpi` when present.

**4 — Mask image.**
- `final_mask` only: load `_combined_mask.png`, resize to the original image's size with **nearest-neighbour**, save.
- `denoise_mask`: load `_combined_mask.png`, resize to original size with **nearest-neighbour**, convert to RGBA, alpha-composite `_noise_mask.png` (also RGBA) over it, save. ⚠️ Upstream uses **bilinear** for the mask upscale in this one branch and nearest everywhere else (`image_export.py:221` vs `:205`/`:244`) — an inconsistency that softens exported mask edges. v1 uses nearest-neighbour uniformly. §14.8, flagged §15.8.

**5 — Text image.** Copy/re-encode `_text.png` to the text destination (RGBA preserved; forcing a non-alpha `preferred_mask_file_type` such as `.jpg` must warn and flatten onto white rather than fail).

**6 — Format options** (`formats.rs`, from `save_optimized`):

| suffix | format | options |
|---|---|---|
| `.png` | PNG | max compression (level 9 equivalent), no interlace |
| `.jpg`/`.jpeg` | JPEG | quality 95, progressive |
| `.webp` | WebP | lossless for masks, quality 95 for images |
| `.tif`/`.tiff` | TIFF | LZW compression (or the original's compression if the original was TIFF with a known one) |
| `.bmp`/`.dib` | BMP | — |
| `.jp2` | JPEG2000 | — (if unsupported by `image`, reject at config-validation time with a clear message) |
| `.ppm` | PPM | — |

`dpi` is written when the original had it and the target format supports it. An unknown suffix is a **config validation error**, not a runtime one.

**7 — OCR report writer** (`ocr_report.rs`, for `panel-ocr ocr`): given `Vec<OcrAnalytic>`, emit either
- **CSV**: header `filename,startx,starty,endx,endy,text`, one row per box, coordinates in original-image space, text quoted per RFC 4180; or
- **TXT**: `"{filename}: \n"` then one line per box's text, blank line between files.
Both formats are defined by the vendored fixtures (§12.6).

**8 — Emit** `files_written`.

**Explicitly out of scope for stage 5 in v1:** PSD / layered export (`LayeredExport`, `export_to_psd`, `bundle_psd`) — v1.5; inpainted outputs; post-action hooks/custom commands; the `merge_cached_images` stitch-all debug variant (only the outputs actually needed for merged-strip export are stitched, per D9).

### 12.4 Dependencies on other stages

`ExportSources` handles come from `MaskOutput` (masked, final_mask, isolated_text) and `DenoiseOutput` (denoised, denoise_mask); `original_path`/`export_path` from the pipeline (they diverge only for merged strip segments); `GeneralConfig` for format preferences.

### 12.5 Task breakdown

| ID | Task | Kind |
|---|---|---|
| **E1** | `formats.rs`: suffix→format map, per-format save options, mode conversion, dpi carry-over | simple |
| **E2** | `discover.rs`: availability probing + precedence resolution + `--save-only-*` narrowing | simple |
| **E3** | `lib.rs`: destination resolution (absolute vs relative `output_dir`), mask scaling/compositing, `run()` wiring | simple |
| **E4** | `ocr_report.rs`: CSV + TXT writers | simple |
| **E5** | Merged-strip export in `pc-pipeline`: read `splits.json`, stitch per-output images in segment order, re-point `export_path` | simple |

Batching: `{E1, E2, E3}` one sequential call; `{E4, E5}` one sequential call.

### 12.6 Fixtures

- `ocr_output/good_detected_text.csv` → **E4** golden: writing an `OcrAnalytic` for `img1.jpg` with boxes `(100,100,300,200)`/text `some text perhaps` and `(534,275,592,414)`/text `or nothing at all` must reproduce this file **byte-for-byte** (modulo a trailing newline, which the fixture lacks — assert with the trailing newline trimmed).
- `ocr_output/good_detected_text.txt` → **E4** golden: two files (`page1.jpg` 2 lines, `page2.jpg` 3 lines) reproduce the fixture byte-for-byte (again trailing-newline-normalized). Note the fixture's exact `"{name}: "` (colon + space + newline) header form.
- `long_strip.jpg` → **E5**: split into 4 segments, run a mock pipeline, stitch, and assert the exported image has the original 1000×8000 dimensions.
- `demo_bubbles/square_bubble_clean.png` → **E1/E3**: PNG→PNG round-trip must be pixel-identical; PNG→JPEG must decode within tolerance; grayscale mode must be preserved.

### 12.7 Acceptance criteria

**(A) Primary**

1. PNG→PNG export is **pixel-identical** to the source (lossless), and the output mode equals the original's mode (grayscale in → grayscale out).
2. PNG→JPEG export decodes to within **max per-channel Δ ≤ 6** and SSIM ≥ 0.99 of the source at quality 95. (Justification: q95 4:2:0/4:4:4 JPEG on flat manga art is near-lossless; ≤ 6 catches an accidental quality or colour-space regression while tolerating normal DCT error.)
3. `dpi` from `long_strip.jpg` (300×300) survives a JPEG→JPEG export.
4. Destination resolution: absolute `output_dir` → `output_dir/{stem}_clean.png`; relative `cleaned` → `{input_parent}/cleaned/{stem}_clean.png`. Parent directories are created.
5. Precedence: with both `masked` and `denoised` present, exactly one cleaned file is written and it is the denoised one; with `--skip-denoise`, the masked one. `files_written` lists exactly the files that exist on disk afterwards.
6. `--save-only-mask` writes no cleaned and no text file.
7. Mask export is upscaled to the original size with nearest-neighbour: every pixel of the exported mask belongs to one of the source mask's colours (no interpolated intermediate values) — the regression test for §14.8.
8. Text export to a non-alpha format warns and flattens onto white rather than erroring.
9. Unknown `preferred_file_type` (e.g. `.xyz`) fails **config validation**, with the error naming the supported list.

**(B) Parity**

10. The two OCR-report byte-for-byte gates in §12.6.
11. `long_strip.jpg` split → stitch → export yields exactly 1000×8000, and (with a pass-through mock pipeline) is pixel-identical to the input's decoded pixels.

---

## 13. Consolidated task table

Order is the recommended implementation order; "Dep" lists blocking task IDs. Kind per CLAUDE.md: **heavy** = own isolated Codex call; **simple** = batchable.

| # | ID | Crate | Task | Kind | Dep |
|---|---|---|---|---|---|
| 0 | **F0** | — | Vendor upstream fixtures + `ATTRIBUTION.md`; workspace skeleton, CI (fmt/clippy/test, Linux+macOS) | simple | — |
| 1 | **C1** | pc-core | `Rect` + geometry ops + serde; `Language`; `Step`/`Output` + `cache_suffix`; `StageError`; `trait Stage` | simple | F0 |
| 2 | **C2** | pc-core | `ImageHandle` (+ materialization invariant); `PageDataRaw`, `PageData`, `TextBox`, `MaskingRegion`, `MaskData`, analytics types | simple | C1 |
| 3 | **C3** | pc-config | TOML profile/config load+save via `toml_edit`, all v1 defaults, validation, unknown-key preservation | simple | C1 |
| 4 | **C4** | pc-testkit | Fixture path resolution, image loading helpers, SSIM / IoU / max-Δ / mean-Δ metrics, golden-compare macros | simple | F0 |
| 5 | **D2** | pc-detect | `calculate_new_size_and_scale` + INTER_AREA resize | **heavy** | C1 |
| 6 | **D8** | pc-imageops | `calculate_best_splits` + `split_image` + `stitch_images` | **heavy** | C1 |
| 7 | **M1** | pc-imageops | `BinaryMask` + box-mask rasterization (inclusive rects) | simple | C1 |
| 8 | **M2** | pc-mask | Growth kernels (cut-corner square + exact OpenCV ellipse) + iterative padded dilation | **heavy** | M1 |
| 9 | **M3** | pc-mask | Edge extraction + border std dev (gray + colour, heuristic + geometric median, off-white snap) | **heavy** | M1 |
| 10 | **M4** | pc-mask | `fit_region`: crops/offsets, candidate ordering, scoring loop, improvement + failure thresholds | **heavy** | M2, M3 |
| 11 | **M5/M6** | pc-mask | Composition, cleaned image, text layer, `run()` wiring, `MaskData`/analytics, debug writes | simple | M4, C2 |
| 12 | **P1–P5** | pc-preprocess | Types; filters; overlap resolution; reading order; padding tiers; `run()` | simple | C2 |
| 13 | **P6** | pc-ocr | `trait OcrEngine`/`Factory`, `MockOcrEngine`, blacklist filter + analytics | simple | P1 |
| 14 | **N2/N3/N4** | pc-denoise | Gaussian blur; noise-mask build; `run()` + 1-bit shortcut + analytics | simple | M6 |
| 15 | **N1** | pc-imageops | NLM denoise (joint-channel, reflect-101, incremental, rayon) | **heavy** | C1 |
| 16 | **E1–E3** | pc-export | Formats/options/metadata; discovery + precedence; destinations + mask composite; `run()` | simple | C2 |
| 17 | **E4** | pc-export | OCR report CSV/TXT writers | simple | C2 |
| 18 | **D1** | pc-models | Model resolution: download, sha256, atomic install, offline errors | simple | C1 |
| 19 | **D3** | pc-detect | `trait TextDetector`, `RawDetection`, `MockDetector`, `ReplayDetector` | simple | C2 |
| 20 | **D4** | pc-detect | `ort` session + letterbox preprocess + output binding | **heavy** | D1, D3 |
| 21 | **D5** | pc-detect | YOLO postprocess: conf filter, decode, class-agnostic NMS, rescale, language | **heavy** | D3 |
| 22 | **D6** | pc-detect | Mask postprocess + Simple refinement | **heavy** | D3, M1 |
| 23 | **D7** | pc-detect | `run()` wiring + coverage filter + `PageDataRaw` + analytics | simple | D4, D5, D6, D2 |
| 24 | **G1** | pc-pipeline | `CachePaths`, checkpoint read/write, stage chaining, `Checkpointing` modes | simple | all stages |
| 25 | **G2** | pc-pipeline | Batch runner: rayon, per-image isolation + `catch_unwind`, `ImageOutcome`, summary, `--fail-fast` | simple | G1 |
| 26 | **D9/E5** | pc-pipeline | Strip-split orchestration + `splits.json` + merged-strip export stitching | simple | G1, D8, E3 |
| 27 | **X1** | pc-cli | `clap` surface, config discovery, logging/verbosity, progress bars, analytics printout, exit codes | simple | G2 |
| 28 | **F1** | xtask | `record-fixtures` (real models → `tests/fixtures/recorded/`, incl. OpenCV NLM references) | simple | D4, D6 |
| 29 | **F2** | xtask | `calibrate-goldens` + `docs/GOLDEN_CALIBRATION.md` (**must precede freezing the §11.7(B) test**; §10.7(B) is a non-gating report per §15.2 but still must be run and recorded). Also records: the detector's upstream-vs-ours box-count comparison on the recorded page fixture (§15.1 verification), and the demo_bubbles masking calibration report (§15.2). | simple | F1, M6, N4 |
| 30 | **P7** | pc-ocr | manga-ocr ONNX backend (encoder/decoder, greedy decode, vocab) | **heavy** | P6, D1 |

**Suggested Codex call batches:**
`[F0, C1, C2]` · `[C3]` · `[C4]` · `[D2]` · `[D8]` · `[M1]` · `[M2]` · `[M3]` · `[M4]` · `[M5, M6]` · `[P1..P5]` · `[P6]` · `[N1]` · `[N2, N3, N4]` · `[E1, E2, E3]` · `[E4]` · `[D1, D3]` · `[D4]` · `[D5]` · `[D6]` · `[D7]` · `[G1, G2]` · `[D9, E5]` · `[X1]` · `[F1]` · `[F2]` · `[P7]`

Critical path note: **M2 → M3 → M4** is the highest-risk sequence in the project and should be scheduled early with generous escalation budget (Fable consult after 5 iterations, per CLAUDE.md). Note that M1–M6, P1–P6, N1–N4 and E1–E4 are all implementable and fully testable **before** any ONNX work exists, using synthetic and recorded fixtures — so the model adapter is deliberately late on the critical path.

### 13.1 CLI surface (X1)

```
panel-ocr clean  <PATHS>...   [--output-dir DIR] [--profile NAME|--profile-path FILE]
                              [--skip-text-detection|--skip-preprocess|--skip-mask|--skip-denoise]
                              [--save-only-cleaned|--save-only-mask|--save-only-text] [--extract-text]
                              [--cache-masks] [--keep-cache] [--no-cache] [--fail-fast]
                              [--threads N] [--model-path FILE] [--hide-analytics] [-v...]
panel-ocr ocr    <PATHS>...   [--format csv|txt] [--output FILE] [--profile ...]
panel-ocr profile  new|show|list|validate|edit
panel-ocr cache    show|clear [--models] [--images]
panel-ocr models   download|verify|path
```

Fresh, idiomatic design per decision #6 — no docopt compatibility. Verbosity maps to `tracing` levels; `--hide-analytics` suppresses the per-stage summary tables.

---

## 14. Deliberate deviations from upstream (all intentional, all documented)

1. **`calculate_new_size_and_scale`, `lower >= upper` branch** — upstream sets `new_height = lower` while computing `scale` from `upper`; we use `upper` for both. Identical when `lower == upper`.
2. **`resolve_overlaps` determinism** — upstream iterates a Python `set`, making the merge order (and therefore the merged boxes) nondeterministic between runs. We use index order (FIFO). Required by §5.7.
3. **Box/language desynchronization** — upstream's `resolve_total_overlaps` mutates `boxes` without touching `box_language`, so after a center-merge the parallel lists disagree. Our `TextBox` pairing prevents this; the earliest box's language wins.
4. **Colour `std` with a single border pixel** — upstream's `np.std(..., ddof=1)` yields `NaN`; we yield `0.0`.
5. **Noise-mask channel — decided (§15.6), confirmed as a genuine upstream bug.** Mechanism verified: `denoiser.py:62` passes an RGBA `_combined_mask.png` crop into `generate_noise_mask`, which calls `grow_mask` (`image_ops.py`); `grow_mask`'s own docstring declares a mode-`"1"` (binary) input contract, but its first act on the RGBA input, `mask.convert("L")`, takes RGB luma and discards alpha — so a black fill `(0,0,0,255)` yields luma 0 and the mask vanishes regardless of full alpha coverage. A mask's "is this pixel covered" signal must not depend on the brightness of its fill color, so this is a bug, not intent. v1 uses the **alpha** channel; the `black_bubble` denoise output diverges from upstream as an accepted consequence (correct over identical).
6. **Gaussian blur** — PIL's `GaussianBlur` is a 3-pass box approximation; we use a true separable Gaussian with `sigma = radius`.
7. **Coloured NLM — decided (§15.7), confirmed.** v1 uses joint-channel NLM for RGB; the Lab-split variant with `color_filter_strength` is v1.5. Verified upstream's own default is `colored_images = false` (`config.py:700`), so v1's default path is unaffected. `pc-config` emits a one-time `WARN` when a loaded profile sets `colored_images = true`, stating v1's approximation and that `color_filter_strength` is ignored until v1.5 — an opt-in setting must not silently behave differently.
8. **Mask upscale filter on export — decided (§15.8), confirmed.** Verified all 5 relevant upstream sites: `image_export.py:205` (final_mask) NEAREST, `:221` (denoise_mask) **BILINEAR**, `:244` (inpainted_mask) NEAREST, plus `masker.py:107` and `denoiser.py:86` both NEAREST — line 221 is the sole outlier across 5 sites on the same hard-edged fill-mask artifact. v1 uses nearest-neighbour uniformly, matching upstream's dominant, evident intent.
9. **OCR engine failure is fail-open** — a `recognize` error keeps the box rather than propagating.
10. **Per-image error isolation** — upstream lets exceptions from a worker abort the pool; we isolate (§5).
11. **Per-image parallelism** instead of per-stage process pools.
12. **Detector mask refinement** — v1 ships the "Simple" refinement, not upstream's `refine_mask`/`refine_undetected_mask` (§15.2).
13. **Class-agnostic NMS** — upstream runs per-class NMS (`agnostic=False`, `inference.py:115` + `yolov5_utils.py:261`), which can emit duplicate boxes for one balloon across language classes; we run class-agnostic NMS instead (see §8.3 step 4, §15.1). `yolo.rs` must carry a `// DEVIATION(13): ...` comment at the NMS call site.

Each of these must appear as a `// DEVIATION(n): ...` comment at the implementation site referencing this section, so a future parity investigation finds them immediately.

---

## 15. Decisions needing reviewer sign-off before tests are frozen

These are places where I made a call that a Senior Rust Engineer should confirm or overturn *now*, since tests become immutable afterwards.

All 10 items below were reviewed and decided by Fable (Senior Rust Engineer advisor), against direct verification of the upstream Python source. Decisions are binding; see each item for the evidence.

1. **Class-agnostic NMS** (§8.3 step 4) — **DECIDED: class-agnostic.** Verified upstream runs per-class NMS (`inference.py:115`, `yolov5_utils.py:261`, `agnostic=False`). Class-agnostic is the correct v1 choice specifically because it removes the duplicate-box problem at the source rather than relying on upstream's buggy box/language-desync merge (§14.3) to clean it up after the fact. Box-count divergence from upstream is accepted; see deviation §14.13.
2. **`MaskRefineMode::Simple` for v1** (§8.3 step 5) — **DECIDED: ship Simple, do not port upstream's full `refine_mask`/`refine_undetected_mask`.** Verified the full algorithm (`textmask.py:18-214`): top-k grey/Otsu masks + XOR-minimizing merge + hole filling — real work, correctly flagged as the riskiest port in the project. Deciding evidence: the `demo_bubbles` fixtures live in `media/` (README demo assets) and are **never used by upstream's own test suite** — their exact producing version/profile is unverifiable. Porting the riskiest algorithm in the project to chase parity with fixtures of unknown provenance is a bad trade. **Consequence (not a contingency — the plan of record):** §10.7(B) item 15's upstream-image comparison is downgraded from a frozen gate to a **non-gating calibration report** (see §10.7(B) rewrite below); the frozen masking test is a regression lock against our own recorded-fixture pipeline output, calibrated by F2. `Annotation` mode remains the v1.5 door for a full refinement port.
3. **Cleaned-image colour mode when a colour median meets a grayscale page** (§10.3 step 4) — **DECIDED: confirmed as specified**, with an upstream-accuracy correction. Upstream's base image is always 3-channel (`cv2.IMREAD_COLOR`, `ctd_interface.py:198`); its own `_clean.png` is RGB at `scale == 1` and is only restored to the original mode at export via `convert(original.mode)`. Our stage-level rule ("L if all medians achromatic, else RGB") is *export-equivalent*, not upstream-identical: since border-color computation is per-channel-symmetric, a grayscale page's medians are always exactly achromatic (r==g==b), and RGB→L is lossless in that case. Verified empirically against the `black_bubble` golden: its fill is exactly `(0,0,0)`, fully achromatic — confirming the premise that a colored fill would break this rule is false for all 7 demo_bubbles fixtures (all mode `L`). Implementation: composite in RGB internally, convert to `L` at write time when the rule says `L` — single code path, no branch duplication.
4. **Tesseract deferred to v1.5** — **DECIDED: confirmed.** Verified `config.py:375` (`ocr_use_tesseract: bool = False` default) and `ocr/ocr.py:65-66` (when disabled, the factory returns `MangaOcr()` for every language). v1's manga-ocr-only OCR exactly matches upstream's own default-profile behavior; no correctness gap.
5. **Fixture mapping correction** — **DECIDED: confirmed.** The CSV/TXT fixtures define the export-stage OCR report format (E4), not filter behavior. Clincher: `run_ocr` (the code path that produces this format) explicitly sets `ocr_blacklist_pattern = ".*"` and `ocr_max_size = 10**10` (`main.py:866-868`) — the filter is inert in that code path, so the fixtures cannot be filter-behavior tests.
6. **Alpha-vs-luma noise mask** (§14.5) — **DECIDED: confirmed as a genuine upstream bug; use alpha.** Verified the exact mechanism: `denoiser.py:62` passes an RGBA `_combined_mask.png` crop into `generate_noise_mask`, which calls `grow_mask` (`image_ops.py`); `grow_mask`'s own docstring declares a mode-`"1"` (binary) input contract, but it receives RGBA and its first act, `mask.convert("L")`, takes RGB luma and silently discards alpha. A black fill `(0,0,0,255)` yields luma 0, so the mask vanishes regardless of full alpha coverage — this cannot be intentional; a mask's "is this pixel covered" signal must not depend on the *brightness* of its fill color. v1 uses alpha; the `black_bubble` denoise divergence from upstream is accepted (correct over identical).
7. **Joint-channel RGB NLM for `colored_images = true`** (§14.7) — **DECIDED: deferral confirmed**, plus one addition. Verified `config.py:700`: `colored_images: bool = False` default, so v1's default path is unaffected. Addition: `pc-config` must emit a one-time `WARN` when a loaded profile sets `colored_images = true`, stating that v1 uses a joint-channel approximation and `color_filter_strength` is ignored until v1.5 — an opt-in setting silently behaving differently is not acceptable without a logged notice.
8. **Nearest-neighbour mask upscale on export** (§14.8) — **DECIDED: confirmed, nearest uniformly.** Verified all five relevant upstream call sites: `image_export.py:205` (final_mask) NEAREST, `:221` (denoise_mask) **BILINEAR**, `:244` (inpainted_mask) NEAREST, plus `masker.py:107` and `denoiser.py:86` both NEAREST. Line 221 is the sole outlier across five sites operating on the same hard-edged fill-mask artifact; bilinear there manufactures interpolated colors that exist nowhere in the actual mask. Normalizing to nearest matches upstream's dominant, evident intent.
9. **`f64` vs `f32` for border statistics** — **DECIDED: `f64` confirmed, no exceptions.** Verified `color_std` explicitly casts to `np.float64` (`image_ops.py:456`), and the candidate-selection comparison `mask_deviation <= lowest_border_deviation * (1 - threshold)` (`image_ops.py:681-683`) is pure f64 upstream. This comparison picks which mask candidate paints the page; a borderline `f32` rounding flip changes the visible output, not just an internal number. Rule for implementation: no `f32` appears anywhere in `border.rs`/`fit.rs` — enforce via clippy lint or explicit code review at PR time.
10. **`insta` snapshot usage** — **DECIDED: allowed, narrowly, with three mandatory safeguards.** Scope: only the two specified regression locks (§8.7(B)9, §9.7(B)11); every numeric/algorithmic gate in §10.7(A) and all of M2–M4/border/fit/kernels must use hand-written assertions, no exceptions. Safeguards, binding on implementation: **(a)** the first accepted snapshot for each of the two sites must be reviewed against a hand-traced expected value, with reviewer/date/method recorded in `docs/GOLDEN_CALIBRATION.md`, before commit; **(b)** snapshot files are subject to the frozen-tests rule exactly like any other test — `cargo insta accept` is forbidden in CI, and any snapshot change requires the same joint-architect sign-off as a hand-written test edit (state this both in CI config and as a comment at each `assert_json_snapshot!` call site); **(c)** no snapshot tests may be added anywhere else without going through this same §15-style sign-off process first.

---

## 16. Summary of what v1 is NOT

Global out-of-scope list, so Codex has one place to check before building anything speculative:

- **Inpainting** (LaMa, `InpainterConfig`, `_inpainting.png`, `_clean_inpaint.png`, any inpainting fallback for failed masks) — v1.5.
- **PSD / layered export** (`LayeredExport`, per-image and bulk PSD, `koharu-psd` port) — v1.5.
- **GUI** (`egui`, staleness-aware recompute, OCR review window, `Output`-driven image viewer) — v2.
- **GPU execution providers** (CUDA/CoreML/DirectML), `.pt`/torch loading, multi-device dispatch — v2.
- **Windows support** — v2 (Linux + macOS only).
- **Legacy INI config import** — v1.5 (`pc-config` is TOML-only in v1).
- **Tesseract / non-Japanese OCR engines**, OCR review/edit workflows, OCR result *parsers* (import) — v1.5+.
- **Font-rendering-dependent debug visualizations**: `_raw_boxes.png`, `_boxes.png`, `_boxes_final.png`, `_mask_fitments.png`, `_std_devs.png` — v1.5. (v1 debug artifacts are limited to `_box_mask.png`, `_cut_mask.png`, `_with_masks.png`, none of which need text.)
- **DBNet line polygons**, line-based block splitting/merging, orientation/font-size estimation — v1.5.
- **Upstream `refine_mask`/`refine_undetected_mask`** (top-k colour + Otsu + XOR merge) — v1.5 as `MaskRefineMode::Annotation`.
- **Lab-space coloured NLM** and `color_filter_strength` — v1.5.
- **Post-action hooks**, memory watcher / OOM warnings, translations/i18n, `stitch_all` debug merging — v1.5+.
