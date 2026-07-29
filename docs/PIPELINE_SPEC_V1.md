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

1. **Stage crates never touch the filesystem for path *resolution*.** They receive `ImageHandle`s and typed structs; `pc-pipeline` owns all path construction and persistence. (Stage crates may *read* an `ImageHandle` that carries a path, and may *write* an output only when handed an explicit destination `PathBuf`.) **Documented exception: `pc-export`** (§12.3 step 1, task E3) resolves its own *destination* paths (`output_dir` absolute-vs-relative logic, per-artifact filename construction, `mkdir -p`), because export destinations are export-format-specific derivations of `export_path`/`output_dir`, not cache-layer bookkeeping — cache-path resolution and artifact *availability* remain exclusively `pc-pipeline`'s job (§12.3 step 2).
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

Third-party crates (pinned in `[workspace.dependencies]`): `serde`/`serde_json`, `toml_edit`, `image`, `imageproc`, `ndarray`, `rayon`, `clap` (derive), `thiserror`, `anyhow` (binary only), `tracing` + `tracing-subscriber`, `ort` (`=2.0.0-rc.12`, feature-gated), `reqwest` (rustls), `sha2`, `hf-hub`, `indicatif`, `regex`, `csv`, `uuid` (§4.2's `CachePaths` needs it). Dev: `approx`, `tempfile` (disk-backed tests, e.g. `ImageHandle` materialization).

---

## 2. Shared types (`pc-core`)

All types `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]` unless noted — **except** `ImageHandle` (§2.3) and any struct containing one (`PageDataRaw`, `PageData`, `MaskData`): these derive `Debug, Clone, Serialize, Deserialize` only. `ImageHandle` gets a **hand-written** `PartialEq` comparing `path` only (the `cached` field is excluded, since it is `#[serde(skip)]` and comparing decoded pixel buffers would be semantically wrong for a value-equality check). Structs containing an `ImageHandle` therefore cannot derive `PartialEq` transitively; tests on these compare canonical JSON strings and field-wise equality on their `PartialEq` sub-fields instead — this is resolved, not a gap (see docs history: Rust Engineer report, Part 1 item 2). JSON is `snake_case`, including for C-like enums (`Step`, `Output` — both carry `#[serde(rename_all = "snake_case")]`, resolving Rust Engineer report Part 1 item 3: e.g. `Step::Detect` serializes as `"detect"`, `Output::BaseImage` as `"base_image"`). Every top-level persisted struct (`PageDataRaw`, `PageData`, `MaskData`, and each stage's `*Input`) carries `schema_version: u32` (start at `1`) as its first field so future format changes are detectable; nested types (`TextBox`, `MaskingRegion`, `DetectedBlock`, `MaskRegionStats`) and the §2.7 analytics records do not carry their own `schema_version`.

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
    /// Rect containment, inclusive on all four sides (`self.x1<=o.x1 && self.y1<=o.y1
    /// && self.x2>=o.x2 && self.y2>=o.y2`); `self.contains_rect(&self)` is true. Added
    /// beyond the original method list because §2.5's "every `reference` ⊇ `masking`"
    /// invariant needs it and `contains()` only takes a point.
    pub fn contains_rect(&self, other: &Rect) -> bool;
    pub fn merge(&self, other: &Rect) -> Rect;      // bounding union
    pub fn overlaps(&self, other: &Rect, threshold_percent: f64) -> bool;
    pub fn overlaps_center(&self, other: &Rect) -> bool;
    pub fn pad(&self, amount: i32, canvas: (u32, u32)) -> Rect;
    pub fn right_pad(&self, amount: i32, canvas: (u32, u32)) -> Rect;
    pub fn scale(&self, factor: f64) -> Rect;       // per-coordinate, truncating toward zero (Python int())
    pub fn translate(&self, dx: i32, dy: i32) -> Rect;
    pub fn is_empty(&self) -> bool;                 // width <= 0 || height <= 0
    /// Clamped `(x, y, w, h)` for the `image` crate's crop API. `x1/y1` clamp up to 0;
    /// `x2/y2` (treated as exclusive, per §2.1's cropping convention) clamp down to the
    /// canvas. Returns `None` when the clamped width or height is `<= 0` (fully
    /// degenerate or fully out-of-bounds). Decided (no upstream counterpart to defer to).
    pub fn to_crop(&self, canvas: (u32,u32)) -> Option<(u32,u32,u32,u32)>;
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
    pub fn load(&self) -> Result<Arc<DynamicImage>, StageError>;
    /// Decode-free size query (uses cache, else image header).
    pub fn dimensions(&self) -> Result<(u32, u32), StageError>;
    pub fn is_materialized(&self) -> bool;                              // path exists on disk
    /// `Ok(())` iff this handle can survive a checkpoint round-trip (`path.is_some()`),
    /// else `StageError::UnmaterializedHandle`. Call before handing a struct containing
    /// this handle to `serde_json::to_string` for a checkpoint write.
    pub fn ensure_materialized(&self) -> Result<(), StageError>;
}
```

Invariant, decided: after `serde` round-trip, `cached` is `None`. The materialization invariant ("a handle with `path: None` must not be checkpointed") is enforced at **two** points, not one: (1) `ensure_materialized()` is a pre-flight check the pipeline calls before assembling a checkpoint struct; (2) `ImageHandle`'s hand-written `Serialize` impl itself fails (returns a serde error) when `path.is_none()`, so no code path — including one that forgets the pre-flight check — can silently write an unusable handle into a JSON checkpoint. `StageError::UnmaterializedHandle` is the error variant; there is no separate `CoreError` type (an earlier draft's reference to `CoreError` was a naming slip — `StageError` is the only error type `pc-core` defines, per §2.9).

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

impl Step {
    /// The step before this one; `None` for `Detect`. Required by §4.4's resume logic
    /// (added beyond the original method list — used in §4.4 but not previously declared here).
    pub fn prev(self) -> Option<Step>;
}

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

Note: `Output::step()` is deliberately non-surjective onto `Step` — there is no `Output` variant mapping to `Step::Export`, because export writes final user-facing files, not cache artifacts. Tests over `Output::ALL` must not assume every `Step` is covered.

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
 DenoiseInput { mask_data: MaskData, original_image: ImageHandle, masked_image: ImageHandle /* _clean.png */,
                config: DenoiserConfig, dests: DenoiseDests { noise_mask, clean_denoised } }
      │  pc_denoise::run(input)
      ▼
 DenoiseOutput { denoised: ImageHandle, noise_mask: ImageHandle, analytics }
      │  pipeline: (if split) stitch segment outputs back together
      ▼
 ExportInput { original_path, export_path, output_dir, outputs, sources: ExportSources,
               preferred_file_type, preferred_mask_file_type, denoising_enabled }
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
| `DenoiseInput.original_image` | the user's input file (§11.2/§11.3: needed for the 1-bit-mode probe and to build the full-resolution base canvas) |
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
7. **Determinism requirement:** identical inputs + identical config must produce identical outputs, including ordering of boxes and of analytics entries, regardless of thread count. This forbids the `set`-iteration nondeterminism upstream has in `resolve_overlaps` (§9.3, §14.2). **Carve-out (§16.22 item 2):** this guarantee holds **unconditionally for the CPU execution provider**, and thread-count invariance is measured, not assumed (§16.21 item 2). The opt-in CUDA provider is **exempt** — measured to differ from itself between identical runs (§16.22 item 3). CI, fixtures, recordings and every gate remain under the unconditional clause; nothing produced on GPU may feed any of them.

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
intra_threads                = 0         # 0 = let ONNX Runtime choose (§16.21)
inter_threads                = 0         # 0 = let ONNX Runtime choose (§16.21)
mask_refine_mode             = "simple"  # simple | annotation ("annotation" rejected in v1: see §8.3 step 5 / §15.2). Deliberate v1 addition, not present upstream.

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

Validation (fail config load, not per-image): `mask_growth_step_pixels >= 1`, `mask_growth_steps >= 1`, `min_mask_thickness >= 0`, `0.0 <= mask_improvement_threshold < 1.0`, `mask_max_standard_deviation > 0`, `0 <= box_overlap_threshold <= 100`, `template_window_size` and `search_window_size` odd and `>= 3`, `off_white_max_threshold` in `0..=255`, `ocr_blacklist_pattern` compiles as a regex, `long_strip_aspect_ratio > 0`, `noise_outline_size >= 0`, `noise_fade_radius >= 0`, `filter_strength > 0.0`, `color_filter_strength > 0.0`. Also validated here (closing the gap between this list and §12.3 step 6 / §12.7(A)9, which require these to be config-load errors, not runtime ones): `preferred_file_type` (when non-empty) and `preferred_mask_file_type` must each be one of the suffixes §12.3 step 6 lists (`.png .jpg .jpeg .webp .tif .tiff .bmp .dib .ppm` — `.jp2` is accepted as valid *input* but rejected as an output suffix per §12.3 step 6's note), with an error naming the supported list on mismatch.

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
- For one or two full manga pages the maintainer supplies (kept small, ≤ 400 KB each, license-clean): the same triple. **(§16.24 item 17: the chosen page is 441,914 B — the cap is exceeded by ~10% as a ratified deviation, to keep the fixture byte-for-byte identical to its CC-BY source rather than re-encoding generational artifacts onto text edges. The license-clean requirement is unaffected and remains binding.)**
- The recorded JSON's `ImageHandle` paths are stored **relative to the fixtures root** and rebased on load by `pc-testkit`.

`pc-detect` ships `ReplayDetector` (reads a recorded `_raw_mask.png` + block list keyed by input file stem) and `MockDetector` (programmable: fixed blocks + a synthetic mask), both behind `#[cfg(any(test, feature = "testkit"))]` plus a `--detector=replay:<dir>` hidden CLI flag so end-to-end CLI tests run with zero model files.

The one-time recording is a **checked-in artifact**, reviewed like code. This is what makes the demo_bubbles goldens usable in CI.

**7.2.1 `ReplayDetector` binding and artifact format (resolved during D3 test-drafting).**
The `TextDetector` trait's `detect(&self, image: &RgbImage) -> Result<RawDetection, StageError>` takes no filename/stem — it's a pure image-in, detection-out boundary. `_raw_mask.png` + `#raw.json` (§2.4's already-*refined* `PageDataRaw`) cannot be replayed back into a `RawDetection` (they're the stage's *output*, post-refinement/post-assembly, not its input). Therefore:

- `ReplayDetector` is bound to **one fixture stem at construction** (`ReplayDetector::new(dir, stem)`), not re-keyed per `detect()` call — matching the trait's per-image, stem-less signature. Calling `detect()` ignores the passed image's content and returns the fixture's recorded detection, incrementing a `calls()` counter.
- The replay pair recorded/read for this purpose is **distinct** from the `PageDataRaw`-level `_raw_mask.png`/`#raw.json` artifacts and uses its own names: **`<stem>_detector_mask.png`** (the raw, unrefined `RawDetection.mask`) and **`<stem>_detector_blocks.json`** (a `Vec<RawBlock>`, which is why `RawBlock` gains `Serialize`/`Deserialize`). `cargo xtask record-fixtures` (F1) writes both pairs: the `_detector_*` pair for replaying the detector boundary, and the `_base.png`/`_raw_mask.png`/`#raw.json` triple for whole-page regression locks (§8.7(B)9).

### 7.3 Golden calibration gate (ordering requirement)

Upstream's `*_clean.png` images were produced by a specific PanelCleaner version/profile we cannot fully verify. Per CLAUDE.md, tests are frozen once written — so tolerance numbers must be *measured before freezing*, not guessed after. Therefore:

> **Task F2 (must complete before the §11.7(B) denoising parity test is frozen):** implement `cargo xtask calibrate-goldens`, run it against the reference implementation available at that moment, and record measured deltas in `docs/GOLDEN_CALIBRATION.md`. The §11.7(B) tolerances are the *specified* values; if calibration shows they cannot be met, the discrepancy goes back to the two architects jointly (per CLAUDE.md) **before** that test is frozen — never silently loosened afterwards. **This does not gate §10.7(B)**: per §15.2 (decided), the masking stage ships `MaskRefineMode::Simple` rather than upstream's full mask refinement, so the demo_bubbles comparison in §10.7(B) is a non-gating calibration report, not a frozen parity gate — F2 must still run it and record the numbers (for visibility and future v1.5 planning), but a shortfall there is expected evidence of the refinement-mode difference, not a blocker.

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
- Preprocess (port of `inference.preprocess_img` + `letterbox`): letterbox to `1024×1024` with `stride=64, auto=false`, i.e. scale by `r = min(1024/h, 1024/w)` (no upscaling beyond `r=1`... upstream `letterbox` default `scaleup=True`; keep upstream behaviour: allow upscale), resize with bilinear, pad **right/bottom** with `(0,0,0)` (upstream default: `imgproc_utils.py:95`; call without `color`: `inference.py:86`) to reach the padded size; record `(dw, dh)` = total padding in x/y. Channel order: RGB, NCHW, `f32 / 255.0`.
- Outputs (3): `blks` `[1, N, 5 + n_classes]` (n_classes = 2), with verified real shapes `blk` `[1, 64512, 7]`, `mask`/`seg` `[1, 1, 1024, 1024]`, and `lines_map`/`det` `[1, 2, 1024, 1024]`; `64512 = 3 × (128² + 64² + 32²)` for a 1024 input at strides 8/16/32. If the second output has 2 channels and the third has 1, swap them (upstream guards for this: `inference.py:181-185`). Bind by index, but log the actual output names/shapes once at `DEBUG` so a model swap is diagnosable.
- Execution provider: **CPU by default**, `[text_detector] intra_threads` defaulting to **`0`** (all physical cores), session created once and shared. **Amended by §16.21 item 1 and §16.22 item 1.** The former text read *"CPU only, `intra_threads = 1` (parallelism is at the image level)"* — that parenthetical was false in the shipped design, because §14.15's `Mutex<Session>` means image-level parallelism never reaches the detector, so inference was serialized *and* single-threaded (~150 s/page). `1` remains expressible. CUDA is available opt-in via `device = "cuda"` under §16.22's binding conditions; it is quarantined from every gate, fixture and recording.

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

**Explicitly out of scope for stage 1 in v1:** DBNet line-polygon extraction (`SegDetectorRepresenter`), line→block assignment, block splitting at distance gaps, `examine_textblk` orientation/font-size estimation, scattered-line synthesis, the `_raw_boxes.png` debug visualization (needs font rendering), multi-model concurrency > shared session, torch `.pt` loading. **GPU EPs: amended by §16.22 item 1** — CUDA moves to v1.5 as an opt-in, gate-quarantined provider; CoreML, DirectML, `.pt` loading and multi-device dispatch remain v2.

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
6. `run()` with `ReplayDetector` over a recorded fixture produces a `PageDataRaw` whose JSON is byte-identical across 10 runs and across 1 vs 8 rayon threads, **and equals the committed `#raw.json`**. **Hand-written assertions — no `insta` snapshot (§15.10, superseded by §16.20 item 1).** The determinism half never needed a snapshot: it compares runs against each other. The regression half is one hand-written equality against the committed fixture, which locks strictly more than a snapshot would (every field: `confidence`, `language`, `mask_coverage`, `scale`, `image_size`, schema shape) and more than §8.7(B)9's boxes-only lock. Note what this test does and does not verify: `ReplayDetector` ignores the image it is passed (§7.2.1), and per `crates/pc-detect/src/lib.rs` it copies `rect`/`class_index`/`confidence` out of the fixture untouched — so the values this test *computes* are `mask_coverage`, the survivor set, `scale` and `image_size`. The fixture's own correctness is gated separately, by §16.20 item 3's committed-oracle review.
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
- `demo_bubbles/handwritten_bubble_raw.png` (72×132, full-image area 9,504 px — too large to itself be an `ocr_max_size` candidate at the default 3,000) cropped down to a 40×70 region (area 2,800 px, under the 3,000 threshold) as a small-box OCR candidate crop for P6's `MockOcrEngine` plumbing test — the crop content is irrelevant, since the mock returns scripted text.
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

11. On the recorded page fixture with `MockOcrEngine` returning `""` for every crop, the resulting `PageData` box tiers are locked by **hand-written integer assertions — no `insta` snapshot** (§15.10, superseded by §16.20 item 1(a)). The tiers are pure integer arithmetic over the fixture rects and the committed profile constants, so each expected value is written at the assertion site with its derivation in a comment; worked example from §16.20, for block rect `(567,74,663,123)`: tight = `pad(2)` then `right_pad(3)` → `(565,72,668,125)`; extended = `pad(5)` then `right_pad(5)` → `(560,67,678,130)`; reference = `pad(20)` → `(540,47,698,150)`. Upstream is **not** the oracle at this site and must not be used as one: §14.2 (`resolve_overlaps` set-ordering), §14.3 (box/language desync) and §14.13 (differing box sets) all sit between upstream and this stage's input, so upstream disagreement here is noise rather than signal.

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

- `demo_bubbles/*_raw.png` + `*_clean.png` → the **calibration fixtures** (7 pairs; see the non-gating status of §10.7(B) below), driven by `tests/fixtures/recorded/<name>_raw_mask.png` + `<name>#raw.json` (§7.2) so no model runs in CI. Each fixture stresses a different case: `square` (trivial), `handwritten` (thin strokes, small canvas), `black` (dark bubble → non-white median colour → exercises the off-white snap *not* firing, and is the frozen gate in item 16), `ray`/`darkrays` (radial lines crossing the border → high border σ → exercises the improvement threshold), `spikey` (non-convex balloon → exercises growth clipping), `nightmare` (highest-σ stress case). Measured diffs show all 7 pairs have substantial changed-pixel counts (e.g. `nightmare` 16,873/75,117 px) — do not assume any demo fixture exercises the "leave it untouched, `failed: true`" path; that path is covered instead by the synthetic primary gate §10.7(A)9.
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
13. End-to-end on a recorded page: `_noise_mask.png` and `_clean_denoised.png` are locked as size/mode/alpha-histogram assertions plus an SSIM ≥ 0.99 comparison against a committed reference **image file** (a golden PNG, not an `insta` snapshot — this is not a third snapshot site under §15.10(c)), refreshed only by explicit architect decision.

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
    pub outputs: Vec<Output>,          // outputs requested by the user/CLI flags (NOT precedence-resolved — see §12.3 step 2)
    pub sources: ExportSources,        // availability: which stage artifacts exist for this image (Some) vs were skipped/absent (None)
    pub preferred_file_type: Option<String>,       // None => keep original suffix
    pub preferred_mask_file_type: String,          // default ".png"
    pub denoising_enabled: bool,       // config/--skip-denoise state; used by §12.3 step 2 to exclude denoise candidates
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

**2 — Precedence.** Ownership split, resolved: **`pc-pipeline` resolves availability**, **`pc-export`'s `discover.rs` (task E2) resolves precedence**. Concretely: before calling `pc_export::run`, the pipeline populates `ExportSources` — each field is `Some(handle)` if that stage artifact actually exists for this image (either just-produced, or found on disk via `CachePaths` when resuming/`--skip-*`; in `Disk` mode this means iterating the pipeline's own per-image `CachePaths`, which is deterministic and avoids re-parsing JSON, unlike upstream's `*#clean.json` glob) and `None` if it was skipped or never produced. `pc-export` itself never touches the cache directory or does path discovery — this is the one documented exception to §1 rule 1 (see the rule-1 amendment below), and even that exception is scoped to *destination* paths, not cache lookups.

Given a populated `ExportSources`, `discover.rs` picks exactly one *cleaned* image and at most one *mask* image, preferring the highest available stage: `cleaned: denoised > masked`; `mask: denoise_mask > final_mask`; `text: isolated_text` (independent). `--save-only-{cleaned,mask,text}` (reflected in `ExportInput.outputs`) narrow the categories. `denoising_enabled == false` (or `--skip-denoise`) removes the denoise candidates from consideration regardless of whether `ExportSources.denoised`/`denoise_mask` happen to be populated (a stale cached denoise artifact from a previous run must not resurrect itself when denoising is now disabled).

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
14. **Letterbox minimum dimension clamp** — upstream can pass a zero-sized resize dimension to `cv2.resize` for sufficiently small inputs; v1 clamps each rounded dimension to at least `1`, keeping the resize valid and preventing `dw` or `dh` from reaching `1024` and making `mask::crop_letterbox` reject the geometry. The implementation comment is at `crates/pc-detect/src/onnx.rs::letterbox` as `DEVIATION(14)`.
15. **One shared ONNX session** — upstream would honour `text_detector.concurrent_models` at provider construction; v1 shares one `Mutex`-guarded session, and a configured value greater than `1` is warned-and-ignored. The implementation comment is at `crates/pc-detect/src/onnx.rs::OnnxDetector` as `DEVIATION(15)`.
16. **Lazy construct-and-latch** — upstream constructs the detector before its per-image loop; v1 defers construction until the first image that needs detection and latches that attempt's success or rendered refusal for the run, so §4.4 resume can bypass a model it will never read. The implementation site will carry `DEVIATION(16)`; the concurrent implementation pass has not added that comment yet.
17. **Coverage-filter scope and operand** — ratified by §16.20 item 9, which corrects §8.3 step 6's former claim of unconditional parity. Upstream applies its `mask_score < mask_score_thresh` false-positive filter **only to line-less blocks** (`textblock.py:485-490`, inside `if len(blk.lines) == 0:`) and computes it over the **unrefined** mask (`inference.py:203` passes `mask` to `group_output`; `refine_mask` runs at `:204`). v1 applies the filter to **every** block over the **refined** mask. Both differences are deliberate: v1 synthesizes no DBNet line polygons (§14.12 and §8.3's out-of-scope list), so every block is line-less by construction and the scope difference is vacuous *for v1* — it would become live the moment line synthesis lands, which is why it is registered rather than left as prose. The operand difference makes our coverage values roughly 2× upstream's on the same boxes; measured across two real manga pages the filter has never fired (minimum coverage 0.3025 against a 0.1 threshold). The implementation site carries `DEVIATION(17)`.

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
10. **`insta` snapshot usage** — **DECIDED: no snapshot tests anywhere. Superseded by §16.20 (Fable tie-break, 2026-07-29).**

    The former verdict is preserved here because a superseded rule must be visible, not silently deleted, or the next reader will reintroduce it: *"DECIDED: allowed, narrowly, with three mandatory safeguards. Scope: only the two specified regression locks (§8.7(B)9, §9.7(B)11) … **(a)** the first accepted snapshot for each of the two sites must be reviewed against a hand-traced expected value, with reviewer/date/method recorded in `docs/GOLDEN_CALIBRATION.md`, before commit; **(b)** … `cargo insta accept` is forbidden in CI … **(c)** no snapshot tests may be added anywhere else without going through this same §15-style sign-off process first."*

    Two defects in that text, both recorded in §16.20: the scope named **§8.7(B)9**, which is a hand-written regression lock with no snapshot — the snapshot site was §8.7(A)6 (§16.20 item 4); and safeguard (a) reviewed a file that is largely a transcription of `_detector_blocks.json`, while that fixture — the artifact which actually accepts output as truth — carried no safeguard at all (§16.20 item 2).

    **Current rule:** `insta` is not a dependency of any crate and must not become one. §8.7(A)6 and §9.7(B)11 use hand-written assertions (§16.20 item 1); every numeric/algorithmic gate in the project does, without exception. Safeguard (a)'s obligation survives re-aimed at the F1 recording, as the committed-oracle gate in §16.20 item 3, recorded in `docs/DETECTOR_ORACLE.md`.

---

## 16.5 C3/C4 decisions (config + testkit), from Rust Engineer review

Resolved during test-drafting for `pc-config`/`pc-testkit`, applying the same
verify-then-decide process as §15:

1. **§1's dependency graph is corrected**: every stage crate (`pc-detect`, `pc-preprocess`,
   `pc-mask`, `pc-denoise`, `pc-export`) depends on `pc-config`, since the per-stage config
   structs (`GeneralConfig`, `TextDetectorConfig`, `PreprocessorConfig`, `MaskerConfig`,
   `DenoiserConfig`) live in `pc-config` and are embedded by value in each stage's `Input`
   (§4.3). The original graph omitted this edge.
2. **The app-level `Config`** (§6 named it, specified nothing) is provisional: `default_profile:
   Option<String>`, `saved_profiles: BTreeMap<String, PathBuf>` (sorted for determinism, §5.7),
   `cache_dir: Option<PathBuf>`. Confirm or revise before treating its tests as frozen.
3. **`mask_refine_mode = "annotation"` is accepted by config, rejected by `pc-detect`**
   (`StageError::InvalidInput`) — config validation and stage validation are deliberately
   separate layers here; matches §8.3 step 5 / §15.2.
4. **Config defaults are frozen by value**, not by re-serialized text: §6's default TOML mixes
   integer (`filter_strength = 10`) and float (`mask_max_standard_deviation = 15.0`) literals
   for `f64` fields, so a round-tripped default won't be byte-identical to the spec block.
   `DEFAULT_PROFILE_TOML` is `include_str!`-ed from one file so `profile new` and the defaults
   test can never drift apart.
5. **Four range rules are unrepresentable in their natural Rust types** (`off_white_max_threshold:
   u8`, `min_mask_thickness`/`noise_outline_size`/`noise_fade_radius: u32` with a `>= 0` rule) —
   out-of-range values fail at deserialization, not at a separate `validate()` call. Tests assert
   the observable behavior ("load fails, error names the field") rather than a specific error
   variant, so this is resolved without a types decision being forced now.
6. **`preferred_mask_file_type` may not be empty** (unlike `preferred_file_type`, where empty
   means "keep the original suffix" — there is no "original" mask file to default to). Suffix
   matching is ASCII-case-insensitive; the leading dot is required (`"png"` is rejected, not
   normalized).
7. **SSIM estimator is pinned** (frozen by `pc-testkit` tests, needed wherever the spec says
   "SSIM (8×8 windows)" — §10.7(B), §11.7(B)12/13, §12.7(A)2): **non-overlapping** 8×8 tiles
   (not a sliding window), **uniform** window, **population** variance/covariance (÷N, not
   ÷N−1), partial edge tiles included at their real size, global score is the **unweighted**
   mean over tiles, `K1=0.01, K2=0.03, L=255`. This is the one item most worth a second look
   before Codex implements `pc-testkit`, since the §11.7(B) gate numbers are only meaningful
   relative to this exact definition — flag back if a different estimator (e.g. sliding-window,
   unbiased variance) is actually intended.
8. **Shape-metric dilation uses a Chebyshev (square) structuring element**, clamped to image
   bounds, for `O ⊆ dilate(G, 2px)` in §10.7(B).
9. **`IoU(empty, empty) = 1.0`** (perfect agreement — this case is reachable: an image neither
   upstream nor we changed at all). One-empty-one-not is `0.0`.
10. **RGBA metric functions include the alpha channel** in both mean and max (documented, not
    incidental) — separate typed functions per colour model (`_gray`/`_rgb`/`_rgba`) rather than
    one polymorphic entry point, so this choice is explicit at every call site.
11. An unrecognized TOML **table** (not just an unrecognized key) is a `WARN` + preserved,
    same treatment as an unknown key.
12. The §15.7 "one-time" `colored_images=true` WARN is **once per process** (`std::sync::Once`),
    isolated in its own integration-test binary so no other test's profile load can consume it
    first; the structured `ConfigWarning` is still returned on every parse regardless.
13. No validation rules were invented for fields §6 doesn't mention (`concurrent_models`,
    `max_threads`, `input_height_lower/upper_target`, `preferred_split_height`,
    `split_tolerance_margin`, `noise_min_standard_deviation`) even though some combinations are
    reachable nonsense (e.g. `lower_target > upper_target`). Left unvalidated in v1; revisit if
    it causes real problems.

---

## 16.6 Stage 1 (detect) decisions, from Rust Engineer review

Resolved during test-drafting for `pc-detect`/`pc-imageops` (D2, D3, D5, D6, D7, D8):

1. **`DetectInput` gains `pub config: TextDetectorConfig`** (§8.2's field list omitted it). Required so `pc-detect` can reject `MaskRefineMode::Annotation` per §16.5 item 3, and consistent with §3's "config by value in every stage `Input`" rule. Every stage crate therefore depends on `pc-config` (already noted in §16.5 item 1).
2. **Naming**: §8.2's `base_image_dest` is the authoritative spelling; §4.3's diagram (`base_png_dest`) is a typo — read `base_image_dest` there.
3. **`PageDataRaw.scale` stores exactly what `calculate_new_size_and_scale` returns**, even in the integer-inverse branch where that can differ slightly from `new_height / original_height` (e.g. `h=5001` gives `scale=0.5` but `new_h=2501`, so `new_h/h ≈ 0.50010`). This is intentional, not a bug: §11.3 already recomputes the denoiser's up-scale factor from actual image sizes rather than trusting `scale` for that purpose, so nothing downstream depends on `scale` being the exact ratio. §2.4's doc comment is amended to say "approximately `new_height / original_height`; exactly `1.0` when no resize happened" rather than claiming exactness.
4. **Rescaled detector boxes ARE clipped to image bounds.** §8.3 step 4 is amended: after truncating to i32, clamp `x1,y1` to `>= 0` and `x2,y2` to `<= image_size`. This isn't new scope — upstream's yolov5 pipeline clips coordinates (`clip_coords`) as a normal part of the postprocess the spec already claims to port; the original §8.3 step 4 text simply omitted mentioning it. Required so `run()` can produce a `PageDataRaw` that passes its own `validate()` on real (frame-overhanging) detector output.
5. **Box rasterization is EXCLUSIVE on `x2`/`y2` everywhere, matching §2.1's `Rect` convention exactly** (confirmed: `pc-core`'s already-implemented, already-tested `Rect::to_crop` treats `x2`/`y2` as exclusive). §13's M1 row, which said "inclusive", is corrected to say **exclusive** — M1 must rasterize box masks using the same convention as every other rect operation in the codebase, so `mask_coverage` (§8.3 step 6) and the future masker (§10) agree on what region a box actually covers.
6. **`cv2.INTER_LINEAR` (§8.3 step 5's mask resize) is pinned to the same convention OpenCV actually uses**: half-pixel-centre source mapping (`src = (dst + 0.5) * (src_len/dst_len) - 0.5`), with border clamping at both ends, axes scaled independently. Test tolerance is ±1 per pixel (OpenCV's u8 path is fixed-point); exact parity is deferred to an F1-recorded reference, mirroring how §8.3 step 2 already treats INTER_AREA.
7. **`postprocess_mask`'s float→u8 conversion truncates (does not round), after clamping to `[0.0, 255.0]`.** Matches upstream's `(img * 255).astype(np.uint8)` exactly except for the clamp (upstream's `astype` wraps out-of-range floats, which is a real bug the clamp fixes). Consistent with the project's general stance of porting upstream's numeric behavior faithfully except where explicitly identified as a bug (§14).
8. **`calculate_best_splits` (D8) algorithm, fully specified** (§8.7(B)8 previously only described one example, not the algorithm):
   - **Aspect gate**: split only when `split_long_strips` is true AND `width / height <= max_aspect_ratio`.
   - **Split count**: `n = round(height / preferred_height)`, clamped to `>= 1`; number of splits is `n - 1`. (`8000/2000 = 4 → 3 splits`, matching §8.7(B)8's example.)
   - **Search window** per split *i* (1-indexed): `[preferred*i - tolerance, preferred*i + tolerance)`, half-open, clamped to `1..height`.
   - **Row score**: convert to luma (per-channel-averaged grayscale), then for row `y`, `score[y] = Σ_x (luma[y][x] - luma[y][x+1])²` — i.e. the sum of squared *horizontal* (along-row) differences, matching "squared-horizontal-difference score" literally (a flat/uniform row, likely a gutter between panels, scores near zero). `score[0]` is `+infinity` (row 0 can never be a legal split). This resolves the ambiguity between "row vs. previous row" and "along the row" in favor of the latter, since that reading matches the literal phrase "horizontal difference" (a difference computed horizontally) rather than a vertical row-to-row comparison.
   - **Selection**: within each search range, pick the row with the minimum score; ties break toward the smallest row index (deterministic, §5.7).
   - `row_scores()` and `search_ranges()` stay public (needed by the §8.7(B)8 percentile test and useful for `cargo xtask calibrate-goldens` diagnostics), but the algorithm itself is no longer something Codex has to invent.

---

## 16.7 D7 frozen-test vs. `pc-core` contract resolution (joint-architect, 2026-07-28)

Raised by Codex during D7 implementation and correctly escalated rather than patched around:
three frozen tests in `crates/pc-detect/tests/d7_run.rs` compared `serde_json::to_string(&output.page)`
on outputs built from `memory_input()`. That can never succeed. §4.1 memory mode (both
destinations `None`) must leave `base_image`/`raw_mask` path-less, and `ImageHandle`'s
hand-written `Serialize` (§2.3) deliberately errors on `path.is_none()`. The frozen tests and
the frozen `pc-core` contract were therefore genuinely contradictory, not merely mismatched.

1. **The `ImageHandle::Serialize` guard stands, unchanged.** It is the whole point of §2.3: a
   path-less handle must never reach a checkpoint JSON that a later `--resume` would try to
   load from disk. Neither the guard nor `ensure_materialized` was touched. Nothing about
   disk-checkpoint safety is weakened by this decision.
2. **The three tests were asserting the wrong mechanism, not the wrong property.** Stage-wrapper
   equivalence (`detect_stage_matches_the_free_function`), run-to-run determinism
   (`a6_replay_run_is_byte_identical_across_ten_runs`) and determinism under a shared
   `&dyn TextDetector` (`a6_replay_run_is_identical_under_concurrent_shared_detector_use`) need
   a deterministic *comparable representation* of the emitted page. JSON was a convenient
   stand-in for one; serialization is not itself the property under test. `PageDataRaw` therefore
   **derives `PartialEq`** (`crates/pc-core/src/page.rs`) and those three sites compare pages
   structurally via a `page_of()` helper. Structural equality is strictly stronger than string
   equality here: it compares every field, with `ImageHandle` compared by `path` (§2.3's
   already-decided `PartialEq`), and it cannot be satisfied by two differently-shaped pages that
   happen to serialize alike.
3. **Rejected alternative: switch those three tests to disk mode.** It would have kept the
   literal "JSON is byte-identical" wording of §8.7(A)6, but it drags filesystem I/O into a pure
   determinism test, and the 8-thread concurrency test would need either one shared destination
   path (8 threads racing on the same two PNG writes — a false failure, and not what §4.5 is
   about) or per-thread paths (which then differ in the very JSON being compared). Worse test for
   no gain.
4. **§8.7(A)6's wording is amended**: "produces a `PageDataRaw` that is identical across 10 runs
   and across 1 vs 8 threads" — identity is checked structurally in memory mode. JSON
   byte-stability is still covered, in the mode where it is actually meaningful: disk mode, by
   `detect_output_round_trips_through_json`, and later by the F1-gated `assert_json_snapshot!`
   site, which must use materialized (disk-mode) handles for exactly the reason above.
5. **Frozen-test amendment recorded.** Per CLAUDE.md this edit was made only under joint
   Technical-Architecture + Senior-Rust-Engineer sign-off; every other test in `d7_run.rs` and
   `tests/common/mod.rs` (including `memory_input`) is untouched, and the tests remain frozen
   against further unilateral edits.

---

## 16.8 Stage 2 (preprocess) decisions, from joint architect + Rust Engineer review (2026-07-28)

Resolved during test-drafting for `pc-preprocess`/`pc-ocr` (P1–P6), applying the same
verify-then-decide process as §15/§16.6. Each item is binding on Codex.

1. **`pc-ocr` stays a separate crate**, as §1/§9.5/§13 already say. Rationale re-verified rather
   than assumed: P7 (manga-ocr) pulls `ort` behind an `onnx` feature, and folding that into
   `pc-preprocess` would make an optional ONNX dependency reachable from a stage crate whose
   every other line is pure geometry. `pc-ocr` v1 content: `engine.rs` (the two traits) and
   `mock.rs` (`MockOcrEngine`/`MockOcrFactory`, `cfg(any(test, feature = "testkit"))`), mirroring
   `pc-detect`'s `mock.rs`. The `onnx` feature is declared with an optional `ort` dep but has no
   module yet — same shape `pc-detect` uses for the unimplemented D4.

2. **`resolve_overlaps` / `resolve_total_overlaps` use SNAPSHOT partner selection.** §9.3 steps 4
   and 9 say "pop front as `b`; find all remaining `q` with `b.overlaps(q)`; merge and remove".
   Ambiguous: are later `q`s tested against the original `b` or against the progressively merged
   `b`? **Decided: against the box as popped, before any merging** — the partner set is computed
   once, then all partners are merged in. This is not a preference, it is forced by §9.7(A)4:
   with `A(0,0,100,100)`, `B(90,0,190,100)`, `C(180,0,280,100)` and any threshold that makes
   `A~B` and `B~C` true (necessarily `< 10%`), the merged `A∪B = (0,0,190,100)` overlaps `C` by
   exactly the same 10%, so progressive testing would yield `[A∪B∪C]` — the very output §9.7(A)4
   forbids. Snapshot semantics also matches the natural Python (`[q for q in queue if ...]` is
   evaluated before the merge loop mutates `b`). §9.3 step 9's parenthetical "and C separately if
   it no longer overlaps the merged box" is **superseded**: C is never re-tested at all.

3. **`PreprocessOutput.ocr_analytic` is `Some` iff the OCR pass actually ran** — i.e. a factory
   was supplied **and** `config.ocr_enabled`. §9.2's doc comment ("Some iff an OCR engine was
   supplied") is amended. Rationale: with `ocr_enabled = false` an all-empty analytic is
   indistinguishable from "ran and found no candidates", which would make the `ocr` report lie.
   `OcrAnalytic.path` is `page.original_path` (§2.7's "original image path").

4. **`OcrEngineFactory::engine_for` returning `None` keeps the box.** §9.3 step 7's `factory
   .engine_for(box.language)?` is prose, not Rust — the `?` there is undefined for an `Option` in
   a `Result` function. Decided: no engine for that language ⇒ the box is **kept**, is **not**
   OCR'd, and its area is **not** recorded in `box_areas_ocred`; logged at `DEBUG`. This is the
   same fail-open stance as §14.9, and the "discard what OCR can't handle" case is already
   covered explicitly and separately by §9.3 step 2's `performing_ocr && ocr_strict_language`
   rule.

5. **DOTALL blacklist matching is `pc-preprocess`'s own concern.** §9.3 step 7 requires a full
   match "with `(?s)` (DOTALL)", but the already-frozen `PreprocessorConfig::compile_blacklist`
   builds `^(?:{pat})$` **without** `(?s)`. `pc-config` is frozen and its version is only ever
   used as a *validation-time compile check*, so: `pc_preprocess::ocr_filter::compile_blacklist`
   builds `(?s)^(?:{pat})$` and is the only compiler used at stage time. Any pattern that compiles
   under one compiles under the other, so §6's validation rule still means what it says.

6. **A blacklist pattern that fails to compile at stage time is `StageError::InvalidInput`**,
   raised once before the pass (not per box). Config validation normally prevents this, but a
   `PreprocessorConfig` can be hand-built or arrive from a hand-edited checkpoint.

7. **Image access during the OCR pass.** `page.base_image.load()` is attempted **lazily**, only
   once at least one candidate box exists, and a load failure **propagates** as a `StageError`
   (it is not fail-open — §14.9 is about *engine* failures, and a base image that cannot be
   decoded makes the page unprocessable per §2.9). A box whose `Rect::to_crop(image_size)` is
   `None` (degenerate/out-of-bounds) is kept, not OCR'd, and not counted — same treatment as
   item 4.

8. **`run()` validates both ends.** §2.5 says "assert in debug"; v1 upgrades this to an
   unconditional check: `run()` calls `PageDataRaw::validate()` on the input page at entry and
   `PageData::validate()` on the assembled output before returning, mapping either failure to
   `StageError::InvalidInput`. Cost is O(boxes); the benefit is that a corrupt `#raw.json` becomes
   a clean per-image failure (§5.1) instead of a debug-only panic or a silently broken mask stage.

9. **An empty page is a success, not an error.** Zero detector blocks — or every block filtered
   away — yields `Ok` with empty `text_boxes`/`extended_boxes`/`masking_regions` and
   `page_language: None`. The `Skipped { NoTextDetected }` decision belongs to `pc-pipeline`
   (§5.6); the stage never returns `StageError::Empty` here.

10. **`RemovedBox` coordinate scaling is guarded.** §9.3 step 7's `rect.scale(1.0 / page.scale)`
    is undefined for `scale <= 0.0` or a non-finite `scale` (it would saturate to `i32::MAX` via
    `Rect::scale`'s `as i32`). Decided: if `page.scale` is not finite or is `<= 0.0`, use a factor
    of `1.0` and log `WARN`. Well-formed pages are unaffected.

11. **OCR candidacy is judged on the padded, reading-order-sorted boxes.** §9.3's step numbering
    (5 pad → 6 sort → 7 OCR) is normative: the `rect.area() < ocr_max_size` test (strict `<`) and
    the crop both use the post-padding rect, and `RemovedBox.rect` is that same rect scaled.
    `OcrAnalytic.num_boxes` is the number of tight boxes at *entry* to step 7 (pre-removal), per
    §9.3 step 7's own wording.

12. **`performing_ocr` only affects step 2.** It gates the strict-language drop and nothing else;
    it does **not** by itself enable the step-7 pass (that needs a factory + `ocr_enabled`).

13. **§9.7(A)10's determinism gate compares canonical JSON, and that is legal here** — unlike the
    §16.7 detect case. `pc-preprocess` never *creates* an `ImageHandle`; it passes the input
    page's `base_image`/`raw_mask` straight through, and those are path-bearing in every run that
    could be checkpointed. `PageData` therefore keeps its §2 derive set (no `PartialEq`) and
    `pc-core` is not touched.

14. **Reading-order sorting uses `f64::total_cmp`** on the key `x_factor * x1 + y_factor * y1`
    (`x_factor = -0.4` for RTL, `+0.4` otherwise; `y_factor = 1.0`), with `slice::sort_by` for
    stability (§9.3 step 6). No `partial_cmp().unwrap()` anywhere — keys are always finite, but
    an unwrap on ordering is exactly the kind of latent panic §5.2 has to `catch_unwind` around.

15. **Module layout is §9.1's five modules exactly**: `lib.rs`, `filter.rs`, `merge.rs`,
    `order.rs`, `ocr_filter.rs`. The three padding tiers (`pad_tight`, `extend`, `reference_of`)
    are free functions in `lib.rs` — they are three two-line `Rect` calls and do not earn a
    module.

16. **Two merge functions, not one generic one.** Step 4 merges `TextBox`es (language of the
    earliest box wins, §14.3) and step 9 merges bare `Rect`s (no language exists at that point).
    Sharing them through a trait would obscure the one behaviour §14.3 exists to pin down.

---

## 16.9 Stage 3 (mask) decisions, from joint architect + Rust Engineer review (2026-07-28)

Resolved during test-drafting for `pc-mask`/`pc-imageops` (M1–M6), applying the same
verify-then-decide process as §15/§16.6/§16.8. Each item is binding on Codex.

1. **Box-mask rasterisation is EXCLUSIVE on `x2`/`y2`.** §10.3 step 1 says "inclusive on
   both ends, because upstream uses `ImageDraw.rectangle`". That is superseded by §16.6
   item 5, which already settled the codebase-wide convention (and by
   `pc_detect::mask::rasterize_union`, which shipped in Stage 1 using it). A masker that
   rasterised inclusively would cover a different region than the `mask_coverage` filter
   that produced the boxes, which is a worse inconsistency than a one-pixel upstream
   difference on the right/bottom edge of an already heavily-padded extended box.
   `pc_imageops::mask::rasterize_boxes` is the single implementation; `pc-detect`'s
   `rasterize_union` stays where it is (frozen, gray-valued, different return type).

2. **Module ownership.** §10.1's claim that kernels and composition live in `pc-imageops`
   is superseded by §10.5/§13's crate column, which is the authority: `pc-imageops` gets
   exactly one new module, `mask.rs` (`BinaryMask` + `rasterize_boxes`, task M1);
   `grow.rs`, `border.rs`, `fit.rs`, `combine.rs` all live in `pc-mask` (M2–M5). Reason:
   the kernels and the composition rules are *masking policy* (they depend on
   `MaskerConfig` semantics), while `BinaryMask` is a general container. `pc-imageops`
   keeps its "no `pc-config` dependency" property (see its `Cargo.toml` note).

3. **`BinaryMask`'s API is pinned** (§10.2 lists names, not signatures):
   `new/from_fn/dimensions/width/height/get/set/count_set/is_blank/bbox/and/or/
   crop_into/to_gray/from_gray_threshold/as_bits`. `and`/`or` **panic** on a dimension
   mismatch (a programming error, not a per-image condition). `bbox()` returns
   `Option<Rect>` in the codebase-wide exclusive convention (a single set pixel at
   `(2,3)` gives `Rect::new(2,3,3,4)`), `None` when blank.
   `crop_into(rect, target, offset)` maps source pixel `(sx,sy)` to
   `(offset.0 + sx - rect.x1, offset.1 + sy - rect.y1)`, clamping the *source* read via
   `Rect::to_crop` and silently dropping destinations outside `target` — defined this way
   so a partially out-of-canvas `rect` still lands correctly.

4. **`from_gray_threshold` is strict `>`**, and `PIL_BINARY_THRESHOLD = 127` is a named
   constant in `pc-imageops` (§10.3 step 0's PIL `L → 1` rule). §10.3 step 0's precise mask
   is `BinaryMask::from_gray_threshold(&raw_mask.to_luma8(), PIL_BINARY_THRESHOLD)`.

5. **`kernel(0)` is the single centre pixel**, i.e. the corner-zeroing of §10.3 step 6's
   small-kernel branch applies only when `diameter >= 3`. For `diameter == 1` the four
   "corners" *are* the centre, so zeroing them would produce an empty kernel and an
   erasing "dilation" — §10.3 step 6 already states the intended behaviour ("dilation is
   identity"); this item just names where the branch goes. Reachable via
   `min_mask_thickness = 0`, which §6 validation permits.

6. **Dilation is stamp-based**: `out(x + kx - r, y + ky - r) |= in(x, y)` for every set
   kernel cell, i.e. the Minkowski sum, with out-of-bounds writes dropped (zero border).
   Kernels are symmetric, so this equals the reflect-then-max formulation; stating it
   fixes §10.7(A)2's "equals the kernel footprint translated to the centre" literally.

7. **The border canvas is a typed two-variant enum**, `border::BaseCanvas::{Gray, Rgb}`,
   built from the loaded `DynamicImage` (`L`/`LA` → `Gray`, everything else → `Rgb`).
   The **grayscale path is taken when the canvas is `Gray` OR `!allow_colored_masks`**
   (§10.3 step 5's "or a grayscale base"), and in the `Rgb + !allow_colored_masks` case
   the conversion uses `pil_luma` (§10.3's ITU-R 601-2 integer-truncating coefficients),
   never `image`'s `to_luma8`.

8. **`BlankMask` is a marker error type** in `border.rs`;
   `border_std_deviation(...) -> Result<BorderStats, BlankMask>` with
   `BorderStats { std_deviation: f64, median_color: [u8;3] }`. `fit_region` maps
   `Err(BlankMask)` from *any* candidate to `None` for the whole region (§10.3 step 8).

9. **Candidate selection is factored as a pure policy function**,
   `fit::select_candidate(count, fast, improvement_threshold, scorer) -> Result<Selected, BlankMask>`,
   where `scorer: FnMut(usize) -> Result<BorderStats, BlankMask>`. This is what makes
   §10.7(A)8 testable on canned deviations and §10.7(A)11's "evaluates exactly one
   candidate" testable with an instrumented/panicking scorer, exactly as those items ask.
   `count == 0` is a caller bug (`fit_region` always builds `>= 2` candidates) and panics.

10. **The fast-mode break is evaluated after scoring candidate `i`**: `if fast && dev_i ==
    0.0 { break }`. Equivalent to upstream's break-on-zero, and equivalent to testing
    `best_dev == 0.0` post-accept (a `0.0` candidate is always accepted, since
    `0.0 <= best*(1-t)` holds for every `best >= 0.0`).

11. **A degenerate region rect is a skip, not an error.** If `reference.to_crop(image_size)`
    or `masking.to_crop(image_size)` is `None` (empty or fully out-of-bounds — reachable
    from a hand-edited `#clean.json`), `fit_region` returns `None` and logs `WARN`, the
    same treatment as the blank-precise-mask case. Per §5.6 sub-image conditions are never
    `StageError`s.

12. **`Fitment.median_color` is populated even when `mask == None`** (the
    `std_deviation > mask_max_standard_deviation` failure path). §10.3 step 10 only says
    the mask is dropped; keeping the colour makes the failure analysable and is what
    §10.7(A)16's frozen gate asserts on.

13. **Cleaned/text-layer canvas source and size.** The canvas is `original_image` iff
    `page.scale != 1.0`, else `base_image` (§10.3 step 4, upstream `masker.py:104`), and
    the mask is resized to that canvas's **actual loaded dimensions**, never to a size
    computed from `scale` (same robustness stance as §11.3 / §16.6 item 3). The text layer
    uses the same source as the cleaned image, so a `scale == 1.0` run never has to load
    the original file at all. Nearest-neighbour resampling is pinned as
    `src = floor(dst * src_len / dst_len)` (so a 2× upscale of a binary mask is exactly
    2×2 blocks, which is what §10.7(A)13 asserts).

14. **The §15.3 output-mode rule is applied to the in-memory value, not only at write
    time**: `MaskOutput.cleaned`'s handle carries a `DynamicImage::ImageLuma8` when the
    rule says `L` and `ImageRgb8` otherwise, so memory mode and disk mode agree
    pixel-for-pixel. "Base is grayscale" means the loaded canvas is `L`/`LA`.

15. **`mask_overlay`'s blend is pinned**: with `a = debug_mask_color[3] as f64 / 255.0`,
    `out = round(base * (1 - a) + color * a)` per channel, applied only where the combined
    mask's alpha is 255, result `RGB` at base-image size. §10.3 step 4 said "recoloured and
    composited" without pinning the arithmetic; this is standard source-over with a
    constant alpha.

16. **Determinism gate (§10.7(A)14) does not serialize `MaskData` in memory mode.** Same
    reasoning as §16.7: `MaskData` holds `ImageHandle`s and `ImageHandle::Serialize`
    deliberately rejects path-less handles, while per-thread destination paths would make
    the JSON differ for reasons that are not the property under test. The frozen gate
    therefore compares, across 20 sequential runs and across 1 vs 8 threads in memory mode:
    the combined-mask **RGBA pixel buffer**, the cleaned-image pixel buffer, and
    `MaskData.regions` (`Vec<MaskRegionStats>`, which already derives `PartialEq`) plus
    `scale`/`original_path`. A separate disk-mode test writes to one shared destination set
    twice and asserts the `#mask_data.json` bytes and `_combined_mask.png` bytes are
    identical — the mode in which byte-identity is meaningful. `pc-core` is not touched.

17. **§10.7(B)15's demo_bubbles calibration report is produced by `cargo xtask
    calibrate-goldens` (F2), not by a `pc-mask` test.** Building a `PageData` from a
    recorded `PageDataRaw` requires `pc-preprocess`, and §1 rule 2 forbids a stage crate
    depending on another stage crate — including as a dev-dependency, which would make the
    "no stage↔stage edge" rule unenforceable by inspection. §13's F2 row already assigns
    this report to xtask. Consequently **`pc-mask` ships no test that reads any
    `*_clean.png`**, which also satisfies §15.2/ATTRIBUTION.md's "no pass/fail assertion
    against a calibration fixture".

18. **§10.7(A)16 (the frozen `black_bubble` gate) is driven from the always-present
    upstream `black_bubble_raw.png`**, not from a recorded fixture: a hand-specified
    interior masking region (`Rect::new(60,100,140,240)`, verified to lie inside the
    balloon: 11 200 px, mean luma 24.3, 989 light text px) with the precise mask derived by
    thresholding that raw image at `> 127`. The gate only checks the *selected fill
    colour* (`max channel <= 40`, off-white snap did not fire), which is independent of
    where the precise mask came from — so it stays frozen and gating without depending on
    `cargo xtask record-fixtures` having been run. Measured evidence that the target is
    right: upstream's own `black_bubble_clean.png` fills those pixels with exactly `0`.

19. **`run()` bookkeeping.** `page.validate()` is called at entry and its failure mapped to
    `StageError::InvalidInput` (mirrors §16.8 item 8). An empty page (no masking regions) is
    a success with a fully transparent `combined_mask`, a `cleaned` equal to the canvas, and
    empty `regions`/`analytics` (mirrors §16.8 item 9). `MaskData.schema_version =
    MaskInput.schema_version`; `original_path`/`base_image`/`scale` are passed through from
    `page`. `MaskFittingAnalytic` entries are 1:1 with `MaskData.regions`, in region order,
    with `path = page.original_path` and `fit_found = fitment.mask.is_some()`.

20. **Rayon is deferred.** §10.5's parallelism note is explicitly gated on "only after the
    sequential version passes", so v1's `pc-mask` fits regions sequentially and takes no
    `rayon` dependency. Revisit with a benchmark, not with a guess.

21. **§10.7(A)4's "a 3×3 fully-set mask has all 9 pixels as edges" is corrected to 8.**
    The parenthetical justification ("all on the 1-px image border") does not hold for the
    centre pixel of a 3×3 image: PIL leaves the outermost ring unfiltered (8 pixels,
    copied through as 255 → truthy → edges) and *does* filter the centre, whose FIND_EDGES
    response is `255*8 − 8*255 = 0`. §10.3 step 2's own formula
    (`on the border || any 8-neighbour is 0`) already yields 8; only the acceptance
    criterion's count was wrong, and the frozen test asserts 8 with this reasoning
    recorded at the call site. Verified by hand, not by fitting the test to the code.

---

## 16.10 Stage 4 (denoise) decisions, from joint architect + Rust Engineer review (2026-07-28)

Resolved during test-drafting for `pc-denoise` (N1–N4), applying the same
verify-then-decide process as §15/§16.6/§16.8/§16.9. Each item is binding on Codex.

1. **Module ownership: `nlm` and `gaussian` live in `pc-denoise`, not `pc-imageops`.**
   §11.1 and §11.5's N1/N2 rows place them in `pc-imageops`; that is superseded, for the
   same reason §16.9 item 2 moved the growth kernels into `pc-mask`. `pc-imageops` was
   frozen and committed at the end of Stage 3, and re-opening a frozen crate to bolt on
   two Stage-4-only modules buys nothing: neither module is used by any other stage in
   v1 (`grep` confirms no other §-section references `nlm` or a Gaussian blur). Both stay
   `pub` (`pc_denoise::nlm`, `pc_denoise::gaussian`) and free of any `pc-config`
   dependency, so §11.1's "independently benchmarkable" property is preserved and a v1.5
   hoist into `pc-imageops` is a mechanical move plus a re-export. Considered and
   rejected: adding them to `pc-imageops` now — it reopens frozen, reviewed code for a
   purely notional sharing benefit.

2. **`pc-denoise` gets its own `morph.rs` (`kernel` + `dilate`).** §11.3 step 4 says to use
   "the **same** `kernel()` from `pc-imageops`", but per §16.9 item 2 `kernel`/`dilate`
   actually live in `pc_mask::grow`, and §1 rule 2 forbids a stage crate depending on
   another stage crate. Rule 2 outranks §11.3's wording. `pc_denoise::morph::{kernel,
   dilate}` is a verbatim restatement of `pc_mask::grow::{kernel, dilate}` (itself a
   restatement of OpenCV `getStructuringElement(MORPH_ELLIPSE)`), including §16.9 item 5
   (`kernel(0)` is the single centre pixel → dilation is the identity) and §16.9 item 6
   (stamp formulation, out-of-bounds writes dropped). A frozen test pins the full 11×11
   cell matrix for `noise_outline_size = 5` so the two copies cannot drift silently.
   v1.5 consolidation ticket: hoist the shared morphology into `pc_imageops::morph` and
   have both stage crates re-export it.

3. **`pc-denoise` likewise gets its own `composite.rs`.** Same rule-2 reason. `blend_channel`,
   `alpha_composite_over`, `composite_rgb` and `resize_nearest_rgba` are pinned
   **identically** to §16.9 items 13 and 15 (`out = round(base·(1−a) + colour·a)`,
   source-over with `alpha_out = max(base_a, layer_a)`, nearest resampling as
   `src = floor(dst · src_len / dst_len)`), so Stage 3's composite and Stage 4's agree
   pixel-for-pixel — which §11.3 step 2 depends on, since it *reproduces* Stage 3's clean
   output rather than reading `_clean.png`.

4. **`DenoiseDests`' second field is `denoised`,** per §11.2. §4.3's diagram spells it
   `clean_denoised`; that is a typo, read `denoised` there (same treatment as §16.6 item 2).
   Both fields are `Option<PathBuf>`, `None` meaning "keep it in memory only" (§4.1
   `Checkpointing::Memory`), exactly like `MaskDests`.

5. **`DenoiseAnalytic.path = mask_data.original_path`.** §11.3 step 1's literal
   (`DenoiseAnalytic { std_deviations: vec![], boxes_denoised: 0 }`) omits the `path`
   field that `pc_core::DenoiseAnalytic` actually carries (§2.7). One `DenoiseAnalytic`
   is produced per page (not per region), mirroring `OcrAnalytic`/`DetectAnalytic`.
   `DenoiseOutput.analytics` is therefore a single value, not a `Vec` — as §11.2 already
   writes it.

6. **"1-bit mode" is read from the file header, by a direct magic-byte probe.** §11.3
   step 1's "`image` reports `L1`/`Gray(1)`" is not expressible: the `image` crate has no
   1-bit `DynamicImage` variant (a 1-bit PNG decodes to `Luma8` with values `0`/`255`),
   **and** — verified against `image` 0.25.10 — its PNG decoder's
   `ImageDecoder::original_color_type()` also reports `L8` for a 1 bpp file, so that API
   cannot carry the signal either. Pinned instead as
   `pc_denoise::is_one_bit(handle) -> Result<bool, StageError>`, a 26-byte header read:
   * PNG (magic `\x89PNG\r\n\x1a\n`): `true` iff the `IHDR` bit depth is `1` **and**
     the colour type is `0` (greyscale) — exactly PIL's mode `"1"`. Colour type `3`
     (1 bpp palette) is PIL mode `"P"`, not `"1"`, so it is `false`.
   * PBM (magic `P1` / `P4`): `true` — also PIL mode `"1"`.
   * everything else, and any path-less (memory-only) handle: `false`. An in-memory
     `DynamicImage` cannot represent 1 bpp, and no other v1 input format reaches this
     stage as a bilevel image in practice.
   An I/O failure here is a real `StageError::Io` (the file is about to be read anyway).
   The frozen test builds a genuine 1 bpp PNG from a byte literal, because `image` cannot
   *encode* one either.

7. **1-bit shortcut copy semantics.** When `masked_image.path` and `dests.denoised` are
   both `Some` **and differ**, the shortcut is a byte-level `std::fs::copy` (§11.7(A)7's
   "byte-identical" is only meaningful on disk); otherwise the loaded `masked_image` is
   written as PNG / kept in memory and the returned handle is pixel-identical to
   `masked_image`. Either way the noise mask is a fully transparent RGBA image at the
   **original image's** size, analytics are `{ path, std_deviations: vec![],
   boxes_denoised: 0 }`, and `nlm::denoise` is called **zero** times.

8. **`colored_images` has no effect on the v1 code path at all.** §11.3 step 2 builds the
   canvas with `original.to_rgb8()`, so the NLM cutout is always 3-channel and always
   goes through the joint-channel path (§15.7). Verified numerically harmless for
   grayscale pages: with `r == g == b`, `d(p,q) = Σ_c Σ_o Δ² / (C·t²)` has both the sum
   and the divisor scaled by `C`, so a `C = 3` replicated-gray image yields *exactly* the
   same distances — and therefore the same weights and the same output — as the `C = 1`
   image. This is what makes §11.7(B)12's grayscale OpenCV reference
   (`fastNlMeansDenoising`, which is itself joint-channel over whatever channel count it
   is handed) a valid parity target for our RGB path. `pc-config`'s one-time WARN
   (§14.7) is still the user-facing notice; the stage itself never branches on the flag.

9. **NLM arithmetic, pinned exactly** (§11.3's NLM spec, made unambiguous):
   - `t = template_window / 2`, `s = search_window / 2` (integer division; both windows
     are odd and `>= 3` by config validation, §6).
   - Border: `reflect101(i, n)` = `0` when `n == 1`, else `m = i.rem_euclid(2n − 2)` then
     `if m >= n { 2n − 2 − m } else { m }`. Applied when materialising a padded plane of
     margin `s + t` per channel, once, before the main loop.
   - `d(p,q) = ( Σ_{o∈template} Σ_{c<C} (I_c(p+o) − I_c(q+o))² ) / (C · template_window²)`,
     `w = exp(−d / h²)`, no noise-variance subtraction (`σ²_est = 0`).
   - Offsets are iterated in a fixed order (`dy` ascending, then `dx` ascending) so the
     f32 accumulation order — and hence the result — is reproducible.
   - Accumulators `acc_c` and `wtot` are `f32`; the self term (`dy = dx = 0`, `w = 1`) is
     included, so `wtot > 0` always and there is no divide-by-zero branch.
   - Output `= clamp(floor(acc_c / wtot + 0.5), 0, 255)` (round half away from zero;
     the quotient is non-negative, so `+0.5` then `floor` is exactly that).
   - Return type mirrors the input variant: `ImageLuma8` in → `ImageLuma8` out
     (`C = 1`), anything else → `ImageRgb8` (`C = 3`).

10. **NLM parallelisation is over output row bands with per-band scratch**, using
    `rayon`'s `par_chunks_mut` over the output rows. Each band recomputes the squared-
    difference summed-area table it needs (including a `t`-row halo) instead of sharing
    one, so every output pixel is produced by exactly one thread from exactly one
    arithmetic sequence — which is what makes §11.7(A)5's "bit-identical across 1 and 8
    threads" a structural property rather than a hope. The summed-area table is §11.3's
    mandatory "incremental distance update" in its simplest exact form: for each search
    offset, one `f32` SAT over the band's squared-difference plane turns the per-pixel
    template sum into four lookups, giving `O(W·H·s²)` instead of `O(W·H·s²·t²)`.

11. **The NLM invocation counter is always on, not `cfg`-gated.** §11.7(A)7 requires an
    "instrumented counter", and integration tests in `tests/` do not see the library's
    `cfg(test)`. `nlm::denoise` increments a process-global `AtomicU64`
    (`Ordering::Relaxed`), exposed as `nlm::denoise_call_count()` and
    `nlm::reset_denoise_call_count()`. One relaxed increment per call is free next to the
    filter itself. Tests that read it must not run concurrently with other NLM users;
    the counter test asserts `== 0` after a reset, which is robust to that.

12. **Gaussian blur, pinned** (§11.3 step 5, §14.6): `sigma = radius as f64`;
    `radius == 0` is the identity (clone). Kernel half-width `k = ceil(3·sigma)`, taps
    `exp(−j²/(2σ²))` for `j ∈ −k..=k`, normalised to sum 1 in `f64`. Separable: a
    horizontal pass into an `f32` intermediate (no rounding), then a vertical pass, then
    `clamp(floor(v + 0.5), 0, 255)`. Border handling is **replicate/clamp-to-edge**,
    matching PIL's `GaussianBlur` (which extends the edge pixel), not the reflect-101
    used inside NLM — the two are different upstream operators and the difference is
    deliberate.

13. **`scale_up` is recomputed from actual loaded sizes** (§11.3 step 2, reaffirming
    §16.6 item 3 and §16.9 item 13): `scale_up = cleaned.width() as f64 / mask.width() as
    f64`, and the mask is nearest-resized to `cleaned`'s **dimensions**, never to a size
    derived from `mask_data.scale`. `mask_data.scale` is not read by this stage at all.
    When the sizes already match, `scale_up == 1.0` exactly and no resize happens.

14. **A degenerate region rect is a skip, not an error.** If `rect.scale(scale_up)`
    produces a rect whose `to_crop(cleaned.dimensions())` is `None` (empty or fully
    out-of-bounds — reachable from a hand-edited `#mask_data.json`, or from a rect that
    scales to zero width), the region is skipped with a `WARN` and contributes **no**
    layer and **no** increment to `boxes_denoised`. Mirrors §16.9 item 11; §5.6 keeps
    sub-image conditions out of `StageError`.

15. **`boxes_denoised` counts layers actually produced**, i.e. selected regions minus
    item 14's skips. `std_deviations` is the σ of **all** `mask_data.regions` in region
    order, including failed ones (§11.3 step 6) — the two numbers are deliberately not
    the same length.

16. **Region selection is `!r.failed && r.std_deviation > noise_min_standard_deviation`**,
    strictly greater, order preserved (§11.3 step 3). Layers are composited source-over
    in that same order, so a later region wins in an overlap — the same tie-break as
    §16.9's `build_combined_mask`.

17. **Output colour mode of `denoised`:** `ImageLuma8` iff the loaded original is
    `L`/`LA` **and** every pixel of the final composite is achromatic (`r == g == b`),
    else `ImageRgb8`. This mirrors §16.9 item 14's user-visible outcome without needing
    Stage 3's fill colours, is an `O(W·H)` check that is free next to NLM, and is
    deterministic. It is reliably `L` on a grayscale page: joint-channel NLM over a
    replicated-gray cutout returns a replicated-gray cutout (item 8), and the mask
    composite is the one Stage 3 already produced. `noise_mask` is **always** RGBA.

18. **`denoising_enabled` is not consulted by this stage.** It is a pipeline-level skip
    (`Step::Denoise` is simply not run) and an export-level input (§12.2's
    `ExportInput.denoising_enabled`). A `run()` that silently no-oped on a config flag
    would make the stage's own tests untrustworthy.

19. **§11.6/§11.7(B)12's recorded NLM reference is the *whole* raw bubble image**, not an
    unspecified sub-crop. Pinned so the F1 recording script and the frozen test cannot
    disagree: input `tests/fixtures/upstream/demo_bubbles/<name>_bubble_raw.png` loaded as
    `luma8` (`name ∈ {nightmare, ray}`), reference
    `tests/fixtures/recorded/nlm/<name>_h10_t7_s21.png`, produced by
    `cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)`.
    Thresholds come from `GoldenThresholds::nlm_parity()` (already defined in
    `pc-testkit`). The test is `#[ignore]`d with an `unimplemented!("blocked on F1")`
    body until F1 lands, following the `a6_pending_recorded_page_equality_and_determinism`
    precedent; §7.3 still requires F2 to run before it is unignored.

20. **§11.7(B)13's end-to-end golden is likewise F1-blocked** and is a committed
    reference **PNG** compared with `pc_testkit::metrics::ssim_gray` (≥ 0.99) plus
    size/mode/alpha-histogram assertions — explicitly not an `insta` site (§15.10(c)).

21. **Task split and kinds are unchanged from §11.5**, with N1's crate corrected to
    `pc-denoise` per item 1: `{N2, N3, N4}` batch sequentially in one Codex call, `N1`
    is isolated and heavy. Implemented ahead of Codex during this pass (fully pinned
    arithmetic, no judgment left): `morph.rs`, `gaussian.rs`, `composite.rs`, `nlm.rs`,
    and `noise_mask.rs`'s pure helpers (`select_regions`, `alpha_binary`, `fade_mask`,
    `attach_alpha`). Left as `todo!()` for Codex (multi-step wiring): `noise_mask.rs`'s
    `build_noise_mask` and `lib.rs`'s `run`.

22. **Frozen-test correction (2026-07-28, joint architect + Rust Engineer).** During N4
    implementation Codex found that `n4_run.rs`'s
    `item13_the_upscale_factor_comes_from_the_actual_sizes_not_from_mask_data_scale`
    built its single region with `std_deviation = 0.1` while asserting
    `boxes_denoised == 1` under `DenoiserConfig::default()`. That is self-contradictory:
    item 16 / §11.3 step 3's strictly-greater cutoff at `noise_min_standard_deviation =
    0.25` excludes σ = 0.1, so the region can never be selected. The selection logic and
    the spec are both correct; the σ was an incidental authoring slip in a test whose
    subject is item 13's `scale_up` recompute, not the cutoff (which is separately locked
    by `a10_zero_qualifying_regions_...` and §11.7(A)6/10). Authorized change, and the
    only one: that call site's σ becomes `0.5`, comfortably above the cutoff and not
    boundary-adjacent. No assertion, no other test, and no implementation file changed.

---

## 16.11 Stage 5 (export) decisions, from joint architect + Rust Engineer review (2026-07-28)

Resolved during test-drafting for `pc-export` (E1–E4), applying the same
verify-then-decide process as §15/§16.5–§16.10. Each item is binding on Codex. Every
capability claim about the `image` crate below was verified against the pinned
`image` 0.25.10 source in the local registry, not from memory.

1. **Stage 5's scope in this pass is E1–E4 only; E5 is `pc-pipeline` work.** §12.5 lists
   E5 ("merged-strip export") in the stage-5 table, but §13 row 26 places it in
   `pc-pipeline` batched with `D9`, depending on `G1`/`D8`/`E3`. §13 is authoritative
   (§0: "§13 is what the TDD loop is driven from"). `pc-export` therefore knows nothing
   about `splits.json`, stitching, or `export_path` re-pointing — it receives an already
   re-pointed `export_path` (§12.2). §12.6's `long_strip.jpg` stitch fixture and
   §12.7(B)11 belong to E5 and are **not** frozen here. §12.7(A)3's dpi gate *is* frozen
   here, since it only reads `long_strip.jpg`'s header.
   **PSD / layered export is confirmed out of scope for v1** (§12.3's out-of-scope
   paragraph and §16's global list agree); no type, feature flag, or `todo!()` for it
   appears in the crate.

2. **`ExportInput.outputs: Vec<Output>` is interpreted through a three-valued
   category map.** §12.2 types the field as `Vec<Output>` while §12.3 step 2 reasons in
   terms of *categories* (`cleaned` / `mask` / `text`), and §2.8's `Output` has no
   export-side variants at all (`Output::step()` is non-surjective, by design). Pinned
   mapping (`discover::Category::of`):
   * `MaskedOutput`, `DenoisedOutput` → `Category::Cleaned`
   * `FinalMask`, `DenoiseMask` → `Category::Mask`
   * `IsolatedText` → `Category::Text`
   * every other variant → `None` (silently ignored; they are cache artifacts).
   A category is *requested* iff `outputs` contains at least one variant mapping to it.
   `outputs: []` requests nothing and yields `files_written: []` — that is a valid, non-
   error outcome (§5.6), and the pipeline is expected to populate all three categories
   for a default run. `--save-only-mask` is therefore `outputs = [FinalMask,
   DenoiseMask]` (§12.7(A)6). Considered and rejected: adding export variants to
   `pc_core::Output` — `pc-core` is frozen, and §2.8 explicitly documents the
   non-surjectivity as intentional.

3. **Precedence, pinned exactly** (`discover::resolve`, §12.3 step 2):
   * `cleaned = if denoising_enabled { denoised.or(masked) } else { masked }`.
     A populated `sources.denoised` with `denoising_enabled == false` is a stale cached
     artifact and is ignored (§12.3 step 2's explicit requirement).
   * `mask`: `WithDenoise { final_mask, denoise_mask }` iff `denoising_enabled` and both
     are `Some`; else `FinalOnly(final_mask)` iff `final_mask` is `Some`; else `None`.
     A `denoise_mask` **without** a `final_mask` yields `None` plus a `WARN`, not an
     error: §12.3 step 4's denoise branch composites the noise mask *over* the resized
     combined mask, so there is no defined output without the combined mask.
   * `text = isolated_text` (independent of both).
   * A category that is not requested (item 2) is forced to `None` **after** the above,
     so precedence and narrowing cannot interact.
   Exactly one cleaned file and at most one mask file are ever written (§12.7(A)5).

4. **Destination resolution** (`destinations`, §12.3 step 1, §12.7(A)4):
   * `base = if output_dir.is_absolute() { output_dir } else { export_path.parent()
     .unwrap_or(Path::new("")) .join(output_dir) }`.
   * `stem = export_path.file_stem()`; a missing stem is
     `StageError::InvalidInput("export_path has no file stem")`.
   * cleaned suffix = `preferred_file_type` normalised, when `Some` and non-empty;
     otherwise `original_path.extension()` normalised. A missing extension on both is
     `StageError::InvalidInput`.
   * mask and text suffix = `preferred_mask_file_type`, normalised.
   * Normalisation (`formats::normalize_suffix`) = ASCII-lowercase plus a leading `.` if
     absent — matching `pc_config::validate::validate_output_suffix`'s normalisation.
   * `mkdir -p base` happens in `run()`, once, before any file is written, and is an
     `StageError::Io` on failure.
   * Names are `{stem}_clean{suffix}`, `{stem}_mask{mask_suffix}`,
     `{stem}_text{mask_suffix}` (upstream `OutputPathGenerator(export_mode=True)`).
   * `files_written` is ordered **cleaned, mask, text** — the fixed category order, not
     filesystem or hash order (§5.7 determinism).

5. **Colour-mode handling** (`formats::ColorMode`, §12.3 step 3). "Convert to the
   original image's colour mode" is pinned as a four-valued mode — `L`, `La`, `Rgb`,
   `Rgba` — derived from the original's **decoder header** (`ImageReader::into_decoder()
   .color_type()`), never by decoding the pixels: `original_path` may be an 8000 px
   strip and only its mode is wanted. Mapping is by `ColorType::has_alpha()` and
   `channel_count() <= 2`, so `image`'s `#[non_exhaustive]` `ColorType` needs no
   catch-all guess; 16-bit and float variants collapse onto their 8-bit counterparts
   (v1 writes 8-bit only). **DEVIATION:** upstream's `image.convert(original.mode)`
   can re-palette a mode-`P` input; the `image` crate expands palettes at decode time
   and has no palette encoder, so a palette PNG exports as `Rgb8`/`Rgba8`. Documented
   at the call site as `// DEVIATION(§16.11 item 5)`.
   The mode conversion applies to the **cleaned** output only. The mask and text
   outputs are mask artifacts and stay `Rgba8` (subject to item 6's coercion).

6. **Per-format colour coercion, from the encoders `image` 0.25.10 actually has**
   (`formats::coerce_for_format`). Verified support:
   | format | accepted | coercion applied |
   |---|---|---|
   | PNG | L8, La8, Rgb8, Rgba8 | none |
   | JPEG | L8, Rgb8 | `flatten_onto_white` (La8→L8, Rgba8→Rgb8) |
   | WebP | L8, La8, Rgb8, Rgba8 | none |
   | TIFF | L8, Rgb8, Rgba8 | La8→Rgba8 |
   | BMP | L8, La8, Rgb8, Rgba8 | none |
   | PPM | Rgb8 only (P6) | `flatten_onto_white`, then →Rgb8 |
   `flatten_onto_white(px) = round(c·a + 255·(1−a))` per channel, alpha dropped — the
   same rounding form as §16.9 item 15's `blend_channel`, with the base fixed at 255.
   This is also §12.3 step 5's "warn and flatten onto white" for a text export to a
   non-alpha target: `run()` emits one `WARN` naming the suffix, then relies on this
   same coercion (§12.7(A)8) — flattening is never an error.

7. **Format options** (`formats::encode_to_vec`, §12.3 step 6), with three
   `image`-imposed deviations, each carrying a `// DEVIATION(§16.11 item 7)` comment:
   * PNG: `PngEncoder::new_with_quality(w, CompressionType::Best, FilterType::Adaptive)`.
     `Best` is `image`'s max-compression setting; PNG has no interlace here (the crate
     never interlaces). Lossless, so §12.7(A)1's pixel-identity gate holds.
   * JPEG: `JpegEncoder::new_with_quality(w, 95)`. **DEVIATION:** baseline, not
     progressive — `image`'s JPEG encoder has no progressive mode. Quality and chroma
     handling are unaffected, so §12.7(A)2's Δ≤6 / SSIM≥0.99 gate is unchanged.
   * WebP: `WebPEncoder::new_lossless`. **DEVIATION:** §12.3's "quality 95 for images"
     is unreachable — `image`'s WebP encoder is lossless-only (it says so in its own
     doc comment); v1 writes lossless WebP for both images and masks. Strictly
     higher fidelity than specified.
   * TIFF: `TiffEncoder`. **DEVIATION:** `image` exposes no compression selector, so
     "LZW (or the original's compression)" is not expressible; v1 writes the crate's
     default. Purely a file-size matter.
   * BMP: `BmpEncoder`. PPM: `PnmEncoder::with_subtype(Pixmap(Binary))` — P6 binary,
     pinned rather than left to the crate's "dynamic header" default, which may pick
     PAM (P7) and is documented as arbitrary.
   * `.jp2` never reaches this code: `pc-config` rejects it at load (§6,
     `SUPPORTED_OUTPUT_SUFFIXES`). `OutputFormat::from_suffix` still rejects any unknown
     suffix with `StageError::UnsupportedFormat`, whose message reuses
     `pc_config::validate::supported_suffix_list()` so the two lists cannot drift
     (§12.7(A)9).
   * Encoding errors surface as `StageError::Io { path, source: io::Error::other(e) }`,
     the same idiom `pc_denoise::write_png` uses. `encode_to_vec` itself returns
     `image::ImageError` (it has no path to name); `save` adds the path.

8. **DPI carry-over is read from the file header and written for JPEG only**
   (§12.3 steps 3/6, §12.7(A)3). `image` 0.25.10 exposes pixel density on exactly one
   encoder (`JpegEncoder::set_pixel_density`) and on **no** decoder, so:
   * `formats::read_dpi(path)` parses the header itself: JPEG via the JFIF `APP0`
     segment (`units == 1` → dpi verbatim; `units == 2` → `round(v · 2.54)`;
     `units == 0` → `None`), PNG via the `pHYs` chunk (`unit == 1` → dpi =
     `round(ppm · 0.0254)`; `unit == 0` → `None`). Scanning stops at `SOS` / `IDAT`.
     Any other format, or an absent segment/chunk, is `None`. Reads at most the first
     64 KiB; a short/truncated header is `None`, not an error.
   * Writing: `OutputFormat::supports_dpi()` is `true` for `Jpeg` and `false` for every
     other format — PNG `pHYs` insertion would mean hand-patching the encoder's byte
     stream, which is not worth it for a metadata field §12.3 calls "when the target
     format supports it". A density outside `1..=u16::MAX` is dropped.
   * DPI applies to the **cleaned** output only, and is read from `original_path`.
   §12.7(A)3 (`long_strip.jpg`'s 300×300 surviving a JPEG→JPEG export) is frozen and
   passes against this implementation.

9. **Mask export** (§12.3 step 4, §14.8/§15.8). Both branches resize the combined mask
   to the **original image's size** with nearest-neighbour, pinned identically to
   §16.9 item 13 / §16.10 item 3 as `src = floor(dst · src_len / dst_len)`; the denoise
   branch then alpha-composites the noise mask (itself nearest-resized to the same size
   if it differs) source-over, with `alpha_out = max(base_a, layer_a)`. Uniform
   nearest is the §15.8 decision; the call site carries `// DEVIATION(8)`. The exported
   mask is `Rgba8` before item 6's coercion, which is what makes §12.7(A)7's
   "every exported pixel's colour occurs in the source mask" assertion exact.
   "Original image's size" is `ExportInput.original_path`'s dimensions, read with
   `image::image_dimensions` (header only) — the same reason as item 5.

10. **`pc-export` restates nearest resampling and source-over in its own
    `composite.rs`.** Same reasoning as §16.10 item 3: §1 rule 2 forbids depending on
    `pc-mask`/`pc-denoise`, and `pc-imageops` (frozen at the end of Stage 3) owns
    neither operation. `resize_nearest_rgba` and `alpha_composite_over` are pinned
    byte-identically to `pc_denoise::composite`'s, so a mask exported at scale matches
    the one the denoiser saw. Same v1.5 consolidation ticket.

11. **The OCR report reads `OcrAnalytic.removed`, and returns a `String`**
    (`ocr_report`, §12.3 step 7, §12.6, §15.5). `pc_core::OcrAnalytic` carries per-box
    text only in `removed: Vec<RemovedBox>` — and that is correct rather than a gap:
    §15.5 verified that upstream's `run_ocr` path forces `ocr_blacklist_pattern = ".*"`
    and `ocr_max_size = 10**10` (`main.py:866-868`), so in the report code path *every*
    box is "removed" and `removed` is the complete, ordered box list. Rows are emitted
    in `analytics` order, then `removed` order. `filename` is
    `analytic.path.file_name()` (lossy UTF-8), not the full path — the fixtures show
    bare `img1.jpg` / `page1.jpg`.
    The writers return `String` and never touch the filesystem: `--output FILE` is
    `pc-cli`'s business, and §12.3 step 1's filesystem exception is scoped to *image*
    destinations. `ReportFormat::{Csv, Txt}` + `render(format, &[OcrAnalytic])`.

12. **Report formats, pinned to the fixtures** (§12.6, §12.7(B)10). Both end with a
    single trailing `\n`; the vendored fixtures lack it, so the frozen tests compare
    with `trim_end_matches('\n')` on both sides, exactly as §12.6 instructs.
    * CSV: header `filename,startx,starty,endx,endy,text`; one row per box with
      `rect.x1,rect.y1,rect.x2,rect.y2` (§2.1's exclusive `x2/y2`, unmodified — the
      fixture's `100,100,300,200` is upstream's own `startx..endy` dump);
      RFC 4180 **minimal** quoting, hand-written rather than via the `csv` crate: a
      field is quoted iff it contains `,`, `"`, `\r` or `\n`, and an embedded `"` is
      doubled. Line terminator is `\n` (the fixture is LF; the `csv` crate defaults to
      CRLF, which is the concrete reason for hand-writing it).
    * TXT: per file, `"{filename}: \n"` — note the space **before** the newline, which
      the fixture has — then one line per box text, then a blank line before the next
      file. No blank line after the last file.
    * An empty `analytics` slice renders `""` for both formats (not a bare CSV header):
      "no OCR was run" must not look like "OCR found nothing".

13. **`ExportInput` carries `schema_version` and its `sources` may be memory-only.**
    Per §2 every top-level persisted struct starts with `schema_version`; §12.2 already
    shows it. Because `ExportSources` holds `ImageHandle`s, an `ExportInput` built in
    `Checkpointing::Memory` mode is deliberately **not** serializable (§2.3's
    `Serialize` guard), so determinism/equality tests over export inputs compare
    structurally (`PartialEq`), never through JSON — the §16.7 rule. `ExportSources`,
    `Destinations` and `ExportOutput` all derive `PartialEq` for that reason;
    `ExportInput` does not (it contains no non-`PartialEq` field, but its handles make
    equality path-only, so tests compare the fields they mean).

14. **`run()` never errors on "nothing to do".** No requested categories, or every
    source `None`, returns `ExportOutput { files_written: vec![] }` (§5.6 — export of a
    page with no detected text is a normal outcome, and §5.6 already says such a page is
    still exported when the cleaned artifact exists). Errors are reserved for: an
    unresolvable destination (item 4), an unsupported suffix, a failed `mkdir -p`, a
    source handle that cannot be loaded, and an encode/write failure.

15. **Task split and kinds are unchanged from §12.5**, minus E5 (item 1):
    `{E1, E2, E3}` in one sequential Codex call, `{E4}` in another. All four are
    "simple" — there is no numerical algorithm here, only wiring and byte formats.
    Implemented ahead of Codex during this pass (fully pinned, no judgment left):
    `formats.rs` in full, `discover.rs` in full, `composite.rs` in full,
    `ocr_report.rs` in full, and `lib.rs`'s `destinations`. Left as `todo!()` for Codex
    (genuine multi-step wiring): `lib.rs`'s `export_cleaned`, `export_mask`,
    `export_text` and `run`.

16. **§12.7(A)2's `max delta <= 6` is corrected to `<= 10`, with `mean |delta| <= 1.5`
    added; the `SSIM >= 0.99` half stands.** Measured, not guessed: encoding all 14
    demo-bubble PNGs (7 `_raw`, 7 `_clean`) through this crate's q95 JPEG path and
    comparing against the source gives max deltas of **7–10** and mean |Δ| of
    0.22–1.00, with SSIM 0.9962–0.9993. §12.7(A)2's justification ("q95 on flat manga
    art is near-lossless") mis-describes the fixtures: they are hard black-on-white
    line art, whose step edges are the worst case for DCT ringing, and the peak delta
    lands on exactly those edges. The bound still does the job it was written for —
    the same measurement at q85 gives max **30** / mean 2.59 / SSIM 0.9844, and at q75
    max 46 / mean 3.86 — so `max <= 10` **and** `mean <= 1.5` **and** `SSIM >= 0.99`
    each separate q95 from a one-notch quality regression by a wide margin.
    `pc-testkit` is frozen and committed, so `GoldenThresholds::jpeg_q95()` is **not**
    edited; the frozen test spreads it and overrides the two fields
    (`GoldenThresholds { max_delta: Some(10), max_mean_abs_diff: Some(1.5),
    ..GoldenThresholds::jpeg_q95() }`) and asserts that the inherited `min_ssim` is
    still `0.99`, so a future edit to the shared helper cannot silently weaken this
    gate. Considered and rejected: raising the encoder's quality above 95 to fit the
    number — the quality is upstream's (`save_optimized`) and the number is the thing
    that was wrong.

17. **Using `demo_bubbles/*_clean.png` as a *source image* is not a §15.2 parity
    assertion.** §15.2 / `ATTRIBUTION.md` forbid pass/fail assertions **against**
    upstream's `_clean.png` outputs, because their producing version and profile are
    unverifiable. §12.6 nonetheless assigns `square_bubble_clean.png` to the E1/E3
    codec gates, and that is legitimate and stays: those tests compare the file
    against *our own re-encoding of that same file*, so the fixture is an arbitrary
    piece of representative manga art and no claim about upstream's masking algorithm
    is made. The rule remains in force for anything that compares our *pipeline
    output* to a `_clean.png`.

---

## 16.12 Pipeline orchestrator + CLI decisions (joint architect + Rust Engineer review, 2026-07-28)

Resolved during test-drafting for `pc-pipeline` (G1, G2, D9/E5) and `pc-cli` (X1),
applying the same verify-then-decide process as §15/§16.5–§16.11. Every claim about a
stage crate's API below was traced against the **already-implemented, frozen** source in
`crates/pc-*`, not against the spec prose. Each item is binding on Codex.

1. **Two crates, and `pc-cli` is a lib + bin.** §1/§13 rows 24–27 keep `pc-pipeline`
   (orchestration) and `pc-cli` (bin `panel-ocr`) separate; that stands. `pc-cli`
   additionally exposes a `lib.rs` (`pc_cli`) so X1's `clap` surface is unit-testable
   with `Cli::try_parse_from` instead of only through process spawns. §1 rule 3 still
   holds: `pc_cli` contains argument parsing, config/path discovery, logging setup,
   detector construction and calls into `pc-pipeline` — no algorithm code.

2. **THE ONNX QUESTION — v1's CLI ships with no working default detector, and says so
   loudly.** D1 (`pc-models`) and D4 (`ort` session) are deferred, `pc-detect`'s `onnx`
   feature is not default, and nothing in §8/§13 makes a *working* detector a
   precondition for G1/G2/X1 (§13 explicitly notes the model adapter is "deliberately
   late on the critical path"). Decision:
   * `panel-ocr clean` takes `--detector <SPEC>`, `SPEC ∈ { onnx | mock | replay:<DIR> }`,
     default `onnx`.
   * `onnx` is **not implemented in this milestone**: building the provider returns a
     *fatal* error (exit code 1, §5.3) whose message names tasks D1/D4 and points at
     `--detector replay:<DIR>`. It is a clean, diagnosable failure, never a panic and
     never a silent empty-detection run.
   * `mock` and `replay:<DIR>` are `hide = true` clap values, backed by `pc-detect`'s
     `MockDetector`/`ReplayDetector`. `pc-cli` therefore depends on `pc-detect` with
     `features = ["testkit"]` in v1 — a deliberate, documented consequence of shipping
     before D4, to be dropped when `onnx` becomes real.
   * `mock` = zero blocks + blank mask: a page with no detected text, which §5.6 says
     must still export a byte-equivalent cleaned copy. That makes a genuine end-to-end
     CLI test possible today with no model and no recorded fixture.
   * ONNX model resolution is lazy in the provider: a missing or unverifiable managed model
     remains fatal for the run, but is detected at first use rather than startup because
     §4.4 permits runs where stage 1 never executes. The `Model` carve-out keeps that
     refusal a single run-fatal rather than N per-image failures. The outcome of the single
     ONNX initialization attempt — successful construction or rendered refusal — is cached;
     neither outcome is retried within a run. The `cfg(not(feature = "onnx"))` arm stays
     eager by design, since that refusal can never be made successful by later configuration;
     the two arms must not be unified.
   * Rejected alternative: making `--detector` mandatory. It would break the flag's
     forward compatibility — once D4 lands, `onnx` must be the default with no flag —
     and would hide the "why is there no detector" explanation behind a usage error.

3. **Detector injection is per-image, via a provider.** §4.5 wants one shared `ort`
   session; §7.2.1 binds `ReplayDetector` to **one fixture stem at construction**. A
   single `&dyn TextDetector` cannot serve both. Pinned:
   ```rust
   pub trait DetectorProvider: Send + Sync {
       fn detector_for(&self, original: &Path) -> Result<Arc<dyn TextDetector>, StageError>;
   }
   ```
   `SharedDetector(Arc<dyn TextDetector>)` returns the same handle for every image (the
   §4.5 ONNX case, and the mock case); a replay provider constructs a stem-bound
   detector per image. A provider error is a per-image `Failed { step: Detect }`
   **unless the provider declares its failures run-fatal** (§16.19 item 5) — a missing
   replay fixture for page 7 must not kill pages 1–6.

4. **OCR has no engine in v1, and that is not an error.** P7 (manga-ocr) is unstarted,
   so `PipelineCtx.ocr` is `Option<&dyn OcrEngineFactory>` (matching
   `pc_preprocess::run`'s `Ctx` exactly) and `pc-cli` passes `None`. When `None` is
   passed while `preprocessor.ocr_enabled == true`, the pipeline logs one `WARN` per run
   stating that OCR-based box discarding is inactive until P7. `pc_preprocess::run`
   already treats `(None, _)` as "no OCR pass, `ocr_analytic: None`" — no stage change is
   needed. Consequence: `panel-ocr ocr` (§13.1) can run its plumbing but produces empty
   report text in v1; the subcommand stays in the surface (it is X1 scope) and emits the
   same `WARN`.

5. **`--skip-denoise` DISABLES denoising; the other three `--skip-*` flags mean
   "load from cache".** §4.4 lists all four as load-from-cache, but §12.2/§12.3 step 2
   and §16.11 item 3 both read `--skip-denoise` as the `denoising_enabled == false`
   signal that must suppress even a *stale cached* denoise artifact. Those cannot both
   be true. §12/§16.11 wins (it is the later, more specific resolution and the one the
   frozen `pc-export` tests encode): `--skip-denoise` sets
   `denoising_enabled = false`, stage 4 is not run, and its cached artifacts are ignored.
   `PipelineOptions::denoising_enabled() = profile.denoiser.denoising_enabled && !skips.denoise`.

6. **Skip flags are normalised to a prefix.** `--skip-mask` without `--skip-preprocess`
   is nonsense (stage 3's input is stage 2's output). The three load-from-cache flags are
   normalised so that skipping step *n* implies skipping every earlier step, with one
   `WARN` naming the flags that were implied. `SkipFlags::start_step()` then returns the
   first step actually executed: none → `Detect`, detection → `Preprocess`, preprocess →
   `Mask`, mask → `Denoise`.

7. **There is no `--resume` flag in v1.** §13.1's surface has none, and the skip flags
   plus the cache are already the resume mechanism. `ImageHandle`'s disk-materialisation
   guard (§2.3, §16.7) is what makes that safe, and it keeps its full force.

8. **`CachePaths::discover` is added and specified** (§4.2 gave `from_existing`, which
   needs a path that *already* contains the uuid — resuming has only the original image
   path). `discover(cache_dir, original) -> Result<Option<CachePaths>, StageError>`:
   scan `cache_dir`, parse each file name through the same `{uuid}_{stem}{suffix}` rule
   as `from_existing`, keep entries whose `stem` equals `original.file_stem()`, and take
   the **lexicographically smallest uuid string** among the survivors, `WARN`ing when
   there is more than one. Ascending-uuid (not mtime) because §5.7 forbids clock- or
   filesystem-order-dependent behaviour. `Ok(None)` when nothing matches; a start step
   later than `Detect` that gets `None` is a per-image `Failed`, per §4.4.

9. **`from_existing`'s suffix set is the union of `Output::cache_suffix()` and the two
   pipeline-local suffixes below**, matched **longest-first** (`_clean.png` and
   `_clean_denoised.png` both end a stem, and `_raw_mask.png` ends with `_mask.png`-ish
   text; longest-match is the only rule that disambiguates them deterministically).

10. **Split artifacts get pipeline-local suffix constants, not new `Output` variants.**
    `pc-core` is frozen and §2.8's variant list is closed. `pc-pipeline` defines
    `SPLITS_SUFFIX = "#splits.json"` and `segment_path(..., i) = {uuid}_{stem}_seg{i:03}.png`.
    `SplitManifest { schema_version, original, image_size, split_rows, segments }` is the
    §13-row-26 `splits.json`.

11. **`ExportPaths` (§4.2's "separate type, no uuid") is deliberately NOT implemented.**
    `pc_export::destinations` (frozen, §16.11 item 4) already resolves every export
    destination from `export_path`/`output_dir`/suffixes, and §1 rule 1's documented
    exception assigns that job to `pc-export`. A second implementation in `pc-pipeline`
    could only drift. The pipeline supplies `export_path` (re-pointed for merged strips,
    E5) and `output_dir`, nothing more.

12. **`ImageOutcome::Skipped` carries `files_written`.** §5.1's sketch gives it only a
    reason, but §5.6 requires a `NoTextDetected` page to still be exported. Without the
    field the summary would under-report written files. `SkipReason` is closed at
    `UnsupportedFormat { suffix: String }` and `NoTextDetected` — every other
    non-completion is a `Failed`.

13. **Checkpoint reads validate two things, and both are per-image errors** (§4.4):
    `schema_version ∈ SUPPORTED_SCHEMA_VERSIONS` (`&[1]`), and every `ImageHandle` the
    loaded struct references has an existing `path`. Writes go through
    `ImageHandle::ensure_materialized()` first (§2.3's pre-flight), so a Memory-mode
    struct fails before `serde_json` is even reached, with `UnmaterializedHandle` rather
    than an opaque serde error.

14. **`Checkpointing::Memory` selection, pinned** (§4.1's prose): Memory iff
    `image_count == 1 && !debug_outputs && no_cache`; otherwise `Disk`. In Memory mode
    every stage `dest` is `None`, `--cache-masks` is refused at argument-validation time
    (`--no-cache` conflicts with `--cache-masks` and `--keep-cache` in clap), and no
    checkpoint JSON is written or read.

15. **Debug outputs** are on iff `--cache-masks || general.always_cache_masks`, and they
    gate exactly `MaskDests::{box_mask, cut_mask, mask_overlay}` (the three v1 debug
    artifacts, §16's font-free list).

16. **Requested export categories.** A default run requests all three (§16.11 item 2's
    "the pipeline is expected to populate all three categories"), with `IsolatedText`
    included **only** when `--extract-text` (nothing produces `_text.png` otherwise).
    `--save-only-cleaned|mask|text` narrow to exactly one category and are mutually
    exclusive (clap group). Pinned mapping:
    `Cleaned → [MaskedOutput, DenoisedOutput]`, `Mask → [FinalMask, DenoiseMask]`,
    `Text → [IsolatedText]`.

17. **Thread resolution:** `threads = clamp(cli --threads ?? general.max_threads, ...)`
    where `0` means `std::thread::available_parallelism()`, then bounded to
    `1..=image_count` (§4.5's `min(max_threads_or_cpus, images.len())`). Batch
    parallelism uses `rayon`'s `par_iter().collect::<Vec<_>>()`, which preserves input
    order — the `outcomes` vector is therefore in input order regardless of thread count
    (§5.7), and the pipeline installs its own `rayon::ThreadPoolBuilder` rather than
    touching the global pool.

18. **Exit codes and panic containment** (§5.2, §5.5): `EXIT_OK = 0` (all completed;
    skips allowed), `EXIT_PARTIAL = 2` (≥ 1 `Failed`, ≥ 0 others), `EXIT_FATAL = 1`
    (config invalid, no inputs, detector provider unbuildable, cache/output dir not
    creatable). A caught panic becomes
    `StageError::Inference("panicked: {payload}")`, where `{payload}` is the
    `&str`/`String` payload or `"<non-string panic payload>"`.

19. **Task split and kinds.** §13 rows 24–27 stand, with one correction: **G2 is
    `heavy`, not `simple`** (§13's table calls it simple; it is rayon + `catch_unwind` +
    unwind-safety + fail-fast + deterministic ordering in one function), and **G1's
    `process_image` is heavy** for the reason §13's own note gives — it threads five
    stage contracts, two checkpointing modes and four skip levels. Batches:
    `[G1-support]` (simple, everything in item 20's "implemented" list is already done,
    so this batch is verification only) · `[G1-chain]` heavy · `[G2]` heavy ·
    `[D9, E5]` simple · `[X1]` simple.

20. **Implemented ahead of Codex during this pass** (fully pinned, no judgment left):
    `pc-pipeline`'s `cache.rs`, `checkpoint.rs`, `options.rs`, `outcome.rs`,
    `discovery.rs`, `ctx.rs`, the dest-builders in `single.rs`, and everything in
    `strip.rs` except the merged export; `pc-cli`'s `args.rs`, `logging.rs`, `paths.rs`
    and `detector.rs`. **Left as `todo!()` for Codex** (genuine multi-step wiring):
    `single::process_image`, `batch::run_batch`, `batch::process_image_isolated`,
    `strip::merged_strip_export`, and `pc_cli`'s `run_clean`, `run_ocr`, `run_profile`,
    `run_cache`, `run_models`.

21. **Two hidden flags beyond §13.1's surface**, both `hide = true` so the documented
    surface is unchanged: `--detector <SPEC>` (item 2) and `--cache-dir <DIR>`. The
    latter is accepted by `clean`/`ocr` and by every `models` and `cache` subcommand,
    so `models download --cache-dir DIR` and `cache clear --models --cache-dir DIR`
    are valid invocations. All of them route cache-root selection through the one
    helper `paths::resolve_cache_root`, with precedence `--cache-dir > config.cache_dir
    > platform default`. The flag exists because §13.1 gives no way to redirect the
    cache, which makes an end-to-end CLI test either pollute the user's real cache
    directory or be impossible; a hidden override is the smallest honest fix.
    Default cache location: `$XDG_CACHE_HOME/panel-ocr` (Linux) or
    `~/Library/Caches/panel-ocr` (macOS), falling back to `./.panel-ocr-cache` when
    neither `$XDG_CACHE_HOME` nor `$HOME` is set. No new third-party dependency is
    taken for this (`dirs` is not in `[workspace.dependencies]` and v1 is Linux +
    macOS only). The recovery suggestion is derived from the resolved cache root used
    by the failing operation, not from the presence of a `--cache-dir` flag. General
    rule: a suggested recovery command must be computed from the state the failing
    operation actually used, not from the flags it happened to receive, because a
    flag-derived suggestion is silently wrong for every other way that state can be
    reached. Any path interpolated into a command we tell the user to run must be
    shell-quoted, because the entire justification for emitting the command is that
    it can be pasted verbatim. Suggested recovery commands use POSIX single-quote
    rules: wrap the path in single quotes and represent each embedded single quote
    as `'\''` (close quote, escaped literal quote, reopen quote). Historically, the
    prescribed bare `models download` provisioned a different root from the one the
    failing `clean` had reported. Known limitation: `Path::display()` is lossy for
    non-UTF-8 paths, so a suggestion for such a path can still name a mangled
    directory; this pass records but does not fix that limitation.

22. **Authorized frozen-test fix (2026-07-28): X1's suffix-validation profile
    construction.** `crates/pc-cli/tests/x1_cli.rs`'s
    `an_unsupported_output_suffix_fails_config_validation` built its malformed profile
    by appending `\n[general]\npreferred_file_type = ".xyz"\n` to
    `pc_config::DEFAULT_PROFILE_TOML`, which already opens with a `[general]` table.
    That yields a duplicate root table — a TOML *parse* error — so the run failed
    before §6/§12.7(A)9 validation ran, and never exercised the suffix check the test
    is named for. `pc-config` has no last-wins overlay to rescue it: `ProfileDocument::
    parse` is a single `toml_edit::DocumentMut` parse, by design (§6 round-trip). This
    is a test-authoring bug, not a design gap. Codex correctly refused to edit the
    frozen test; the two Opus roles jointly authorize changing **only** the profile
    construction to a targeted `str::replace` of the default `preferred_file_type`
    line, keeping `.xyz` and every assertion, doc comment and other test untouched.

---

## 16.13 Fixture recording + golden calibration decisions (joint architect + Rust Engineer review, 2026-07-28)

Resolved while executing §7.2's task **F1** and §7.3's task **F2**, applying the same
verify-then-decide process as §16.5–§16.12. Each item is binding.

1. **F1 is not one task — it is five independent recording groups, and they have
   different blockers.** §7.2 describes F1 as a single "record once, replay forever"
   step, which implicitly assumed all of it needed real ONNX weights. Tracing the
   actually-frozen tests shows otherwise: only *some* F1 outputs come from the detector.
   `cargo xtask record-fixtures` is therefore group-structured, and each group is
   independently runnable and independently skippable:

   | Group | Real tool it records | Consumers | Status |
   |---|---|---|---|
   | `nlm` | `cv2.fastNlMeansDenoising` | §11.6, §11.7(B)12 | **recorded** |
   | `inter-area` | `cv2.resize`/`INTER_AREA` | §8.7(A)2 | **recorded** |
   | `find-edges` | `PIL.ImageFilter.FIND_EDGES` | §10.3 step 2 cross-check (item 7) | **verified** |
   | `detector` | ONNX `comictextdetector` | §7.2, §7.2.1, §8.7(A)6, §8.7(B)9, §9.7(B)11, §10.7(B)15, §11.7(B)13 | **blocked** (item 8) |
   | `model-signature` | dependency-free protobuf walk of the sha256-verified ONNX artifact | §16.16, `d4_signature.rs`, `pc-models`' D1 keystone | **recorded** |

   Consequence: the OpenCV/PIL-backed gates are unblocked *now*, without waiting on
   §8.5's D1/D4. That was not visible from §7.2's wording and is the main finding of
   this pass.

2. **`xtask` is a workspace member, not an excluded sub-workspace.** `Cargo.toml`'s
   `members` becomes `["crates/*", "xtask"]`, resolving the "added by task F1" TODO left
   in that file at F0. Rationale: F2 must *link* `pc-denoise`, `pc-detect` and
   `pc-testkit` to measure them, so a separate lockfile/`target` would let the tool
   measure a different build than `cargo test` runs — the one thing a calibration tool
   must never do. `publish = false` keeps it off crates.io; nothing in `pc-cli`'s
   dependency graph reaches it, so end users are unaffected. `cargo xtask …` works via
   an `alias` in `.cargo/config.toml`. Considered and rejected: `exclude = ["xtask"]`
   (the other common convention) — it buys a marginally faster `cargo build --workspace`
   at the cost of build-graph divergence in exactly the tool whose job is fidelity.

3. **`xtask` is the single permitted non-dev consumer of `pc-testkit`.** §1 lists
   `pc-testkit` as "dev-dependency only". That rule exists to keep test helpers out of
   shipped binaries; `xtask` is maintainer tooling that is never shipped, and F2's
   report format (`GoldenReport`, `GoldenThresholds`) already lives in `pc-testkit` by
   design (`golden.rs`'s own header says F2 writes its rows). Duplicating those metrics
   in `xtask` would let the calibrated numbers and the asserted numbers drift, which
   defeats §7.3. No other crate gains this exemption.

4. **A missing tool is always a reported skip, never a Rust stand-in.** Binding on every
   present and future recording group: the reference for a parity gate must be produced
   by the *real* third-party implementation. Approximating `cv2.fastNlMeansDenoising`
   (or `INTER_AREA`, or the detector) in Rust to "unblock" a gate would make that gate
   compare our implementation against itself. `record-fixtures` therefore prints an
   actionable skip (what is missing, what to install/implement, which tests stay
   `#[ignore]`d) and exits successfully; it never fabricates a fixture. `cargo xtask
   probe` reports the same capability matrix without recording anything.

5. **§8.7(A)2's INTER_AREA reference is recorded from a lossless re-decode, not from the
   JPEG.** Gap in §8.7(A)2: it says "a recorded `cv2.INTER_AREA` reference" of
   `long_strip.jpg` without saying who decodes the JPEG. Decoding it twice — once by
   OpenCV's libjpeg-turbo for the reference and once by the `image` crate for our side —
   folds an unrelated decoder difference into a gate whose stated justification is
   "identical mathematics, differing only in float accumulation order". Decided: the
   xtask decodes `long_strip.jpg` with the **same `image` crate** the test uses, writes
   that as a lossless PNG, and hands *that* to OpenCV. The decoder difference is then
   measured separately and recorded as an explicitly non-gating diagnostic row (measured
   at F2 time: mean 0.003438, max Δ 1 — i.e. real but an order of magnitude below the
   gate's tolerance, so the isolation is a precaution rather than a rescue).

6. **Derived intermediates are scratch, not fixtures.** The re-decoded 1000×8000 PNG
   (6.7 MB) and the decoder-diagnostic resize (1.5 MB) are byte-reproducible from
   committed inputs, so they go to `target/xtask-scratch/` and are not checked in; only
   the diagnostic's *metrics* are persisted, inside `PROVENANCE.json`. Committed
   recorded-fixture total stays ~1.6 MB. Every recording group writes a
   `PROVENANCE.json` next to its outputs (tool, tool version, parameters, source, output
   sha256, all paths repo-relative) — §7.2 calls the recordings "a checked-in artifact,
   reviewed like code", and a reviewer cannot review a PNG without knowing what produced
   it. Re-running a recording reproduces identical sha256s (verified).

7. **PIL `FIND_EDGES` gets an empirical cross-check, not a fixture.** §10.3 step 2
   replaced PIL's filter with a closed form, and §16.9 item 21 corrected §10.7(A)4's
   edge count from 9 to 8 *by hand reasoning*. `pc_mask::border` consumes no recorded
   file, so there is nothing to record — but the hand proof deserves confirmation
   against the real library. `record-fixtures --only find-edges` runs real PIL over all
   512 distinct 3×3 masks plus 500 random masks and fails loudly on any disagreement.
   Result at F1 time (Pillow 12.3.0): 1012/1012 agree, and a fully-set 3×3 mask yields
   exactly 8 edges — §16.9 item 21 is empirically confirmed. A future disagreement here
   contradicts a frozen test and escalates to both architects; it is never patched in
   the script.

8. **The detector group is blocked by missing *code*, not missing weights.** Recording
   `_detector_mask.png`/`_detector_blocks.json` (§7.2.1) and the
   `_base.png`/`_raw_mask.png`/`#raw.json` triple (§7.2) requires §8.5 task **D4**
   (`pc-detect/src/onnx.rs`, the `ort` session) and task **D1** (`pc-models`), neither of
   which exists in this checkout. Obtaining `comictextdetector.pt.onnx` would not help:
   there is no inference code to feed it to. §7.2 also requires the maintainer to supply
   one or two license-clean full manga pages (≤ 400 KB), which no automated step can
   source. F1's detector group therefore remains **open**, and these tests stay
   `#[ignore]`d with their `unimplemented!("blocked on F1")` bodies intact:
   `pc-detect d7_run.rs::a6_pending_recorded_page_equality_and_determinism` (§8.7(A)6),
   `pc-detect d7_run.rs::b9_pending_recorded_page_regression_lock` (§8.7(B)9),
   `pc-preprocess p5_run.rs::b11_pending_recorded_page_tier_arithmetic` (§9.7(B)11),
   `pc-denoise n4_run.rs::b13_pending_recorded_page_end_to_end_golden` (§11.7(B)13).
   §10.7(B)15's demo_bubbles calibration report is blocked for the same reason and is
   recorded as BLOCKED in `docs/GOLDEN_CALIBRATION.md` rather than omitted.

9. **F2 is partial-by-design and says so in its own output.** §7.3 reads as though
   `calibrate-goldens` either runs or does not. Decided: it always runs, measures every
   gate whose inputs exist, and writes each unmeasurable gate as an explicit BLOCKED row
   naming the reason and the blocked test. A calibration document that silently omits
   what it could not measure is worse than no document. `docs/GOLDEN_CALIBRATION.md` is
   **generated** — the measured numbers are not to be hand-edited.

10. **F2 ran before either newly-unignored gate was frozen, per §7.3, and both were met
    with large margin.** Measured (opencv 5.0.0, pillow 12.3.0, numpy 2.4.6, python
    3.11.15):
    - §11.7(B)12 NLM parity — `nightmare`: SSIM 0.999985, mean |Δ| 0.008946, max Δ 4;
      `ray`: SSIM 0.999999, mean |Δ| 0.003028, max Δ 1. Specified: SSIM ≥ 0.98,
      mean ≤ 1.0, max ≤ 8. **Met**, by two to three orders of magnitude on the mean.
      Note this is a *stronger* result than §11.7(B)12's tolerance rationale predicted
      (it anticipated visible divergence from OpenCV's fixed-point weight LUT); the
      tolerances are nonetheless left exactly as specified, since tightening a frozen
      tolerance to fit one measurement is the mirror image of the loosening §7.3
      forbids.
    - §8.7(A)2 INTER_AREA parity — mean |Δ| 0.000000, max Δ 0: **bit-exact** against
      real OpenCV. Specified: mean ≤ 1.0, max Δ ≤ 2. Met.

    Both tests are consequently unignored, with real assertions replacing their
    `unimplemented!()` placeholders and the recording provenance cited at the call site.
    No other test's assertions were touched.

---

## 16.14 Post-review fixes (joint architect + Rust Engineer review, 2026-07-28)

Resolved after the final v1 codebase audit against this document. Two MAJOR findings and
one documentation gap; the same verify-then-decide process as §15/§16.5-§16.13. Each item
is binding.

1. **Long-strip splitting (D9) is v1 scope and is now wired into the live pipeline.**
   Finding: `pc-pipeline`'s `strip.rs` was fully implemented and unit-tested (`d9_strip.rs`)
   but had **no caller** outside its own tests — `single::process_image` went straight from
   the input path to `DetectInput`, so a default `clean` of `long_strip.jpg` (1000x8000)
   produced no `#splits.json` and no `_seg{NNN}.png` and processed the whole strip as one
   page. Root cause: §16.12 item 20 pinned `process_image`'s contract without ever
   mentioning splitting, so the `[split?]` box of §4.3's diagram had no owner.
   **Decision: in scope, not deferred.** §4.3 and §13 row 26 both place it in v1, the
   primitives (§16.6 item 8's `calculate_best_splits`/`split_image`/`stitch_images`) are
   done and tested, and `general.split_long_strips` defaults to **true** — a default-on
   config key silently doing nothing is not an acceptable v1 state, and deferring would
   have required amending §4.3 *and* §13 *and* adding a runtime "not active in this build"
   WARN, which is strictly more work than wiring the branch. Pinned shape, in `single.rs`:
   * `run_stages(original, cache, options, ctx) -> Result<ChainOutputs, (Step, StageError)>`
     is stages 1-4 for one image **or one strip segment**, with no export.
     `ChainOutputs { sources: ExportSources, analytics: ImageAnalytics, no_text: bool }`.
     Its contract is exactly the clauses §16.12 item 20 gave `process_image`, minus export.
   * `process_image` = `run_stages` + one `pc_export::run` for that image. Its observable
     behaviour is **unchanged**; every frozen `g1_chain.rs` test still applies to it
     verbatim, and it stays the public per-image entry point.
   * `process_image_with_splitting` sits **above** both and is §4.3's `[split?]` box:
     decide via `strip::should_split` (on an `image::image_dimensions` header probe, so a
     non-qualifying image is never decoded twice); if it does not qualify, delegate to
     `process_image` unchanged; if it does, `strip::plan_and_write_segments`, then
     `run_stages` per segment against that segment's own `CachePaths::new`, then
     `strip::merged_strip_export` — **one** export call for the whole strip.
   * `batch::process_image_isolated` calls `process_image_with_splitting`, which is what
     makes `split_long_strips` reachable from `pc-cli`'s `clean`/`ocr` (both go through
     `run_batch`). This is the only behavioural change to G2.
   * **Splitting is skipped, with a `WARN` naming the reason, in exactly two cases**:
     `Checkpointing::Memory` (segments must be materialised to cross five stage boundaries
     and be stitched, and Memory mode writes nothing, §4.1) and a resumed run
     (`options.start_step() != Step::Detect` — the cache entries belong to the earlier
     run's segments and re-planning would orphan them). Never a silent no-op, per the same
     principle as §14.7's `colored_images` WARN.
   * Segment cache entries are ordinary entries keyed on the segment file name, so a
     segment's stem is `{uuid}_{stem}_seg{NNN}` and `CachePaths::discover` for the strip
     still resolves to the strip's own entry (`parse_cache_name` strips the longest known
     suffix first, §16.12 item 9). The stitched intermediates `merged_strip_export` writes
     land in the strip's entry, which is what lets `pc-export` receive materialised handles.

2. **A split image collapses to exactly ONE `ImageOutcome`.** §5.1/§5.5 count *user
   inputs*; a strip that happens to be cut into four segments must not become four rows in
   the summary, four `files_written` sets or four failures. Pinned: `no_text` is the
   conjunction over segments (all segments textless => `Skipped { NoTextDetected }`, per
   §5.6), a failure in any segment is a single `Failed` naming that segment's step but
   carrying the **original** path, and analytics merge as follows — per-page singletons
   accumulate (`DetectAnalytic.blocks_detected`/`blocks_kept` and
   `DenoiseAnalytic.boxes_denoised` add; `std_deviations`, the OCR area/removal lists and
   `MaskFittingAnalytic` rows concatenate in segment order) with every `path` field
   rewritten to the original strip. `Vec` order is segment order, so the printout is
   deterministic (§5.7).

3. **`pc-denoise`'s halo-padded region crop is RATIFIED as an authorized deviation from
   §11.3 step 4.2's literal wording.** Finding: `noise_mask.rs`'s per-region loop crops
   `scaled.pad(noise_outline_size + 3 * noise_fade_radius)` (~8 px at defaults), not
   §11.3 step 4.2's bare `cleaned.crop(rect)`, and did so with no `DEVIATION` comment and
   no §16.x item — an undocumented unilateral change during N3.
   **Decision: keep the padding, document it.** Three pieces of evidence:
   * §11.7(A)8, this stage's own acceptance criterion, already bounds the touched area at
     `region.rect.pad(noise_outline_size + 3 * noise_fade_radius)` — *exactly* the padded
     reach. The containment guarantee §11.3's "Explicitly out of scope" paragraph promises
     is therefore satisfied unchanged; the literal crop is simply a tighter case of the
     same bound, not a different guarantee.
   * Without the padding the alpha fade is clipped into a hard step at an arbitrary
     bounding-box edge, which is a visible seam in precisely the situation the fade exists
     to prevent ("grow the mask and fade its edges", the `grow_mask` docstring §14.5
     quotes), and NLM is left with only the already-painted (usually uniform) fill as
     context.
   * Reverting it breaks a frozen test on its own terms: `n4_run.rs`'s
     `a8_run_changes_nothing_outside_the_padded_region` asserts `changed > 0` on a fixture
     whose fill covers the whole region rect, where the unpadded crop provably changes
     nothing at all. The frozen tests were drafted around the padded reach.
   Accepted consequence, stated plainly: within that <= 8 px halo the layer's alpha is read
   from the combined mask and can therefore overlap a neighbouring region's territory.
   That is bounded by the same reach §11.7(A)8 permits, and region rects already overlap by
   design (§16.10 item 16 pins the source-over tie-break for exactly that case).
   Considered and rejected: zeroing the halo band's alpha outside the region's own rect —
   it keeps the fade unclipped and removes the neighbour-alpha coupling, but it is new
   unreviewed logic for a difference §11.7(A)8 already sanctions, and upstream's own
   unpadded crop reads neighbour fills too, so it would not be closer to upstream either.
   The deviation is marked `// DEVIATION(§16.14 item 3)` at the call site.
   Verified after ratification: `cargo test -p pc-denoise` is fully green, including the
   now-live `b12_recorded_opencv_parity` and `a2` gates (both live in `n1_nlm.rs` and
   exercise `nlm::denoise` directly, so they are independent of this crop either way).

4. **Frozen-test addition, not amendment: `crates/pc-pipeline/tests/g1_split_integration.rs`.**
   `d9_strip.rs` unit-tested `strip.rs`'s pieces; nothing tested that the pipeline *calls*
   them, which is why the gap survived to the final audit. The new file locks: a
   default-configured strip run produces one `#splits.json` plus four `_seg{NNN}.png` and a
   single full-size `long_strip_clean.png`; a split image is one outcome per original input
   in `run_batch`, with no segment path ever appearing as an outcome's `original`; a normal
   page produces byte-identical exports through `process_image_with_splitting` and
   `process_image` and leaves no split artifacts; `split_long_strips = false` disables the
   branch; and Memory mode processes the strip whole while still exporting. No existing test
   was edited.

5. **§14's `// DEVIATION(n)` comments completed.** §14's closing line requires one at every
   implementation site; items 10, 11 and 12 had none. Added, in the codebase's existing
   form (`DEVIATION(<plain §14 item number>)`, as `pc-detect`'s `DEVIATION(1)`/`DEVIATION(13)`
   already use — `§16.x`-qualified numbers stay reserved for §16 deviations):
   `DEVIATION(10)` at `pc-pipeline`'s `catch_unwind` site, `DEVIATION(11)` at its
   `rayon::ThreadPoolBuilder` site, `DEVIATION(12)` at `pc-detect`'s `refine_simple`.
   Comment-only change.

---

## 16.15 Detector class-count correction (joint-architect process with Fable tie-break, 2026-07-28)

1. **The detector model predicts two classes, not three, and the frozen D5 row tests were
   amended accordingly.** The finding was that §8.3 step 3's `n_classes = 3` contradicted
   the real `comictextdetector.pt.onnx` pinned there: its sha256 is
   `1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f`, and it declares
   `blk [1, 64512, 7]`, `seg [1, 1, 1024, 1024]`, and `det [1, 2, 1024, 1024]`. Thus the
   block row is 7 wide, or `5 + 2`. Upstream `dmMaze/comic-text-detector`,
   `utils/yolov5_utils.py:135`, computes `nc = prediction.shape[2] - 5`, so it likewise
   computes `nc = 2`. Upstream `utils/textblock.py:9-15` defines
   `LANG_LIST = ['eng', 'ja', 'unknown']` and
   `LANGCLS2IDX = {'eng': 0, 'ja': 1, 'unknown': 2}`, while the `TextBlock` constructor
   default is `language: str = 'unknown'`; therefore `unknown` is a third language state
   assigned by Python code, never predicted by the model. The correction was resolved by
   the joint-architect process with a Fable tie-break on 2026-07-28.

   The exact frozen D5 amendments were: (1) `constants_match_the_spec` now asserts
   `N_CLASSES == 2` and `ROW_STRIDE == 7`; (2) `a3_fixture()` row arrays A–D were
   truncated to two classes and E/F were re-authored from class 2 to class 1, with their
   fixture comment updated from `c2` to `c1` while preserving every score, coordinate,
   IoU, and the cross-class A/B suppression proof; (3)
   `filter_candidates_decodes_xywh_to_xyxy` uses a two-class row; (4)
   `objectness_gate_is_strictly_greater` uses two-class rows; (5)
   `class_score_gate_is_a_second_separate_strictly_greater_filter` uses two-class rows;
   (6) `best_class_wins_and_sets_the_class_index` uses two classes and expects class index
   1; (7) `filter_candidates_preserves_input_order` uses two-class rows; and (8)
   `a3_class_agnostic_nms_with_both_confidence_gates` expects E's class index to be 1,
   with survivor identity, order, count, and scores unchanged.

   `class_to_language` and the §8.3 step 4 class-index-to-language mapping were
   deliberately not amended: the `2 => None` arm remains as specified for the code-level
   `unknown` state and for unexpected indices, although class 2 is unreachable from the
   real model tensor. Finally, `64512 × 7 = 451_584` and `451_584 % 8 == 0`; that
   coincidence is why the old stride mismatch silently reinterpreted the tensor as
   56,448 misaligned rows instead of failing loudly.

## 16.16 Recorded model signature and closure of the D4a self-reference (F3, 2026-07-28)

F3 closes a self-reference found during stop-time review of D4a. Before this decision,
`crates/pc-detect/tests/d4_onnx.rs` constructed the shipped `blk` output with
`ROW_STRIDE as i64`, while `bind_outputs` checked that same dimension against
`yolo::ROW_STRIDE`. Both sides therefore came from panel-ocr's own constant: the test
proved only self-consistency and never proved that the constant matched the real model.
The sanctioned D4a fixture now uses the measured literal `7`, with
`d4_signature.rs::row_stride_matches_the_models_declared_blk_arity` as the independent leg.

1. **Group and artifact.** `model-signature` is a recording group in the §16.13 capability
   matrix. It verifies the model against `pc_models::COMIC_TEXT_DETECTOR.sha256`, then uses
   a dependency-free protobuf walk to record the declared graph metadata. Its committed
   signature is
   `tests/fixtures/recorded/model_signature/comictextdetector.signature.json`, with a
   sibling `tests/fixtures/recorded/model_signature/PROVENANCE.json`. The directory is
   underscored to match the existing `nlm` and `inter_area` groups and is deliberately not
   under `tests/fixtures/recorded/detector/`: the latter remains the unambiguous BLOCKED F1
   detector group, which needs D1/D4 inference, weights and the separately licensed runtime
   recording boundary.

2. **The §16.13 item 4 carve-out.** A recorded value is legitimate only when it is causally
   independent of panel-ocr's own code. Running the real third-party tool qualifies; reading
   the real artifact's self-declared metadata through its published wire format qualifies;
   computing the value from a panel-ocr constant does not. The recorder therefore verifies
   the real artifact before emitting anything, subtracts initializer names from
   `graph.input`, preserves declared output order, and rejects symbolic, dynamic, zero,
   missing or otherwise unsupported metadata instead of recording a placeholder.

3. **Parser scope fence.** The parser extracts name/element-type/dims from top-level
   `graph.input`/`graph.output` `ValueInfoProto`s, skipping unknown fields by wire type. It
   is not to grow into a general ONNX reader.

4. **Why no Python.** This group intentionally uses no Python, `ort`, ONNX feature, ONNX
   Runtime or general ONNX dependency: a dependency-free walk permanently lowers the
   re-verification barrier. This decision does not change the existing `nlm`, `inter_area`
   or `find-edges` groups, whose real `cv2`/`PIL`/`numpy` tools are required and whose §16.13
   item 10 recording runs are documented. The `ort`-vs-walk cross-check in
   `crates/pc-detect/tests/d4_session.rs` gives eventual, not ongoing, assurance: it runs
   only when someone opts into that `#[ignore]`d, feature-gated test with weights and a
   runtime, never in CI. At F3 completion, the walk's correctness rests on the frozen
   expected values plus review; the cross-check is not a live gate.

5. **Two load-bearing identity tests.** `d4_signature.rs` deliberately does not assert the
   digest literal: the single source of truth for model identity lives in `pc-models`.
   Consequently the added `pc-models/tests/d1_resolve.rs` keystone and the D4 signature
   assertions are load-bearing together. If the keystone is deleted, `d4_signature.rs`
   could silently describe a different export even while all of its shape assertions pass.

6. **Boundaries and honest limits.** This records declared metadata only. It does not
   measure column semantics, class identity, `PAD_VALUE`, padding side, or the bilinear
   resize. The real-weights smoke test remains `#[ignore]`d because D4b needs maintainer
   weights and a runtime; it must not be deleted or downgraded. D4b is not gated by F3,
   while F1's `detector` group is gated/blocked independently. F3 does not implement
   `OnnxDetector` or alter the pure D4a functions.

## 16.17 ONNX CLI activation and fixture-provenance gap (2026-07-28)

1. **D1/D4 are reachable from the shipped CLI.** `pc-cli` now has a non-default `onnx`
   feature forwarding to `pc-detect/onnx`, plus a normal `pc-models` dependency. The
   default build retains the truthful `ONNX_UNAVAILABLE` message: it says the detector is
   "not available in this build", names D1 and D4, and points to `--detector replay` and
   `replay`/`mock` alternatives. This was ordinary implementation work, not a frozen-test
   amendment: the existing assertions pin five substrings, all five survive, and the
   assertions remain true because the feature is non-default. In an `onnx` build that
   message is unreachable; missing files, model-resolution failures, unavailable ONNX
   Runtime, and session-construction failures report their own errors.

2. **Provenance verification gap, accepted for now.** No runnable test verifies a
   `PROVENANCE.json` hash except `model_signature`'s. The `nlm` group (two committed PNGs;
   its recorder performs three renames) and `inter_area` (one committed PNG) are therefore
   unverified. `inter_area`'s `jpeg_decode_diagnostic` hash is unverifiable by design because
   it describes an uncommitted target/scratch file under §16.13 item 6; no future
   "verify every sha256" sweep may be aimed at it naively. This is accepted because those
   groups' fixtures are consumed by parity gates that compare pixels at tight tolerances,
   so corruption fails the gate itself. The activation condition is the first future
   recording group that ships a load-bearing fixture whose consuming tests read values
   rather than compare pixels: before that group records, schedule a shared provenance
   schema and shared checker, not after.

   **PARTLY SUPERSEDED (§16.24).** The *digest* half of this gap is closed: since
   `crates/pc-testkit/tests/recorded_provenance.rs` was made bidirectional, every declared
   digest is verified against the committed bytes, declaring-group locality is enforced, and
   declared/committed path sets are asserted as exact sets with duplicate rejection. The text
   above is preserved verbatim because it records what was true when written, but a reader must
   not re-implement the finished half. What remained open, and what §16.24 rules on, is the
   **schema** half — nothing constrained structure or required any field to be present, so the
   three groups carried three incompatible shapes and a misspelled digest key was silently
   skipped rather than rejected. The activation condition is met by the detector group.

## 16.18 No implicit model provisioning from processing commands (tie-break, 2026-07-28)

1. **The auto-provisioning decision is reversed.** `clean` and every processing command
   never provision a model. A missing or digest-mismatched managed model is a fatal error
   naming the expected cache path and `panel-ocr models download`. An explicit model override
   remains existence-checked but is not hash-verified, so a re-export or the candle escape
   hatch remains usable. Provisioning lives only in the explicit `models download` and other
   model-management subcommands. This reverses the earlier auto-download-on-first-use
   decision, which was recorded as spec-silent and reversible; the tie-break reversed it
   after a verified test-triggered ~90 MB download under `--all-features`.
   The cache-root precedence is `--cache-dir > config.cache_dir > platform default`,
   resolved by one shared helper used by every command. Previously, `clean`/`ocr`
   skipped the config layer while `models`/`cache` skipped the CLI layer; the observable
   symptom was `models download` succeeding into one root while `clean` reported the model
   missing from another, a dead end made visible by the auto-download reversal. General
   rule: any path-resolution precedence used by more than one command lives in exactly one
   function; duplicating it is how the copies diverge.

2. **The first-run convenience door stays open, but only explicitly.** If first-run
   convenience is wanted later, propose a new spec item through the joint-architect process
   for an interactive, TTY-gated prompt exactly of the form
   `model missing — download 90 MB now? [y/N]`. It must never become an implicit download.

3. **Binding general rule:** no test-reachable code path may perform network I/O. Artifact
   provisioning is only ever an explicit dedicated command, or an interactive TTY-gated
   prompt — never a side effect of a processing command.

4. **Binding cfg-test rule:** any test asserting the behaviour or content of a cfg-gated
   item carries the matching cfg. The complementary configuration gets its own test rather
   than no test.

5. **Standing test-hygiene note:** any test invoking `clean` must pass an explicit
   `--detector` or an isolated cache directory, preferably both, so it is deterministic
   against whatever is in a developer's real cache.

## 16.19 Lazy detector initialization, outcome latch, and provider-declared fatality (joint architect + Rust Engineer review, 2026-07-28)

Resolved against the upstream PanelCleaner construction shape, the current `pc-cli`,
`pc-pipeline`, `pc-detect`, `pc-models`, and frozen-test sources. This entry is binding.

1. **Lazy init is ratified — the deviation is in construction placement, not semantics.**
   Upstream constructs once outside the per-image loop: external, non-vendored
   `pcleaner/ctd_interface.py::process_image_batch` binds `model = TextDetector(...)`
   before `for img_path in img_batch:`, and its single-process path constructs before
   `for index, img_path in enumerate(tqdm(img_list))`. Neither site has a `try/except`;
   construction failure propagates and ends the run. v1 therefore initializes lazily in
   `pc-cli/src/detector.rs::OnnxProvider::detector_for`, because eager resolution
   contradicts §4.4: a run whose `#raw.json` is cached never executes stage 1 and must
   not be blocked by a model it will never read. This is a placement deviation only:
   the missing, mismatched, or unloadable model remains a run-fatal refusal. Upstream is
   not vendored (§7.1 vendors fixtures only), so the upstream citation is by file and
   function shape, not by a repository line number. The rule is about **§4.4 resume**
   (§16.12 item 7), not a new command-line surface.

   (b) **The eager `cfg(not(feature = "onnx"))` arm stays eager.**
   `pc-cli/src/detector.rs::build_provider` keeps the no-ONNX refusal at provider build
   time per §16.12 item 2. That refusal can never be made successful by later
   configuration, so the two arms must not be unified.

   (c) **Unavailable ONNX Runtime is the fourth run-fatal condition.**
   `"ONNX Runtime is not loadable in this environment"` is a whole-run environment
   precondition failure, alongside §5.3's unreadable/invalid config, no inputs, model
   missing or hash mismatch (and unavailable download), cache-dir, and output-dir cases;
   §5.3 does not enumerate it separately, but is hereby read as covering an absent
   shared library. It is not per-image. `runtime_available` produces no `StageError` of
   its own; this decision assigns its refusal to `StageError::Model`.

2. **The outcome latch is ratified, and §16.12 item 2 is amended.**
   The sentence previously pinned in §16.12 item 2 — **"Only successful ONNX construction
   is cached; failures are not cached."** — is SUPERSEDED. `OnnxProvider` latches the
   **outcome** of its single attempt — either the session or the rendered refusal — in
   `OnceLock<Result<Arc<dyn TextDetector>, String>>`, behind a double-checked
   `Mutex<()>` gate. The gate recovers a poisoned lock with
   `PoisonError::into_inner`.

   (a) **The latch restores upstream's once-per-run guarantee under the lazy deviation.**
   `pc-pipeline/src/batch.rs::run_batch` calls `detector_for` once per image from rayon
   workers. Without the latch, a doomed run re-resolves the model, re-hashes roughly
   90 MB through `pc_models::verify_sha256`, and rebuilds an ONNX session once per page.

   (b) **The refusal cache is a determinism requirement, not only an optimization.**
   `BatchSummary::fatal_model_message` takes the first run-fatal in input order through
   `find_map`, and which outcome becomes that first one depends on how many workers pass
   `batch.rs::run_batch`'s gate. One initialization attempt gives every worker one
   byte-identical string, so the choice is immaterial. Without the latch, two workers
   could render different text for the same cause — a TOCTOU on the model file or a
   transient read error through `verify_sha256`'s `Err(error) => Err(error.into())` arm — and
   stdout would become scheduling-dependent, violating §5.7.

   (c) **Latching as `StageError::Model` promotes nothing.**
   The mapping is exhaustive: `resolve_detector_model` emits only `Model`, including
   its `impl From<ModelError> for StageError` in `crates/pc-models/src/lib.rs`, which
   maps every `ModelError` variant to `Model`; `ensure_model_file` emits only `Model`;
   and `OnnxDetector::from_path` plus `outlet_meta`, `bind_outputs`, and
   `validate_output_shapes` emit only `Model`. The `InvalidInput` and `Inference`
   variants in `crates/pc-detect/src/onnx.rs` occur on per-image paths (`detect`,
   `decode_mask`, `decode_blocks`, and runtime output decoding/postprocessing). The
   one exception is item 1(c)'s runtime probe: it produces no `StageError` of its own
   and is assigned `Model` by decision here.

   (d) **The latch stores the message, not the error.**
   `StageError` is deliberately not `Clone` (`crates/pc-core/src/error.rs`); it wraps
   `io::Error` and `image::ImageError`. Storing the rendered `String` is therefore not
   a shortcut and must not be "fixed" by deriving `Clone`.

3. **“Retry” means the user re-running `panel-ocr models download`.**
   It never means an in-run loop. Upstream's external `pcleaner/model_downloader.py`
   shape agrees: on a hash mismatch it unlinks the file, returns `None`, and prints
   `Manually download the file and save the path to it in a profile's settings`; it does
   not retry in a loop. v1's frozen `crates/pc-models/tests/d1_install.rs::ensure_available_fails_when_the_replacement_download_also_mismatches`
   states the same rationale:

   > One recovery attempt, not a loop: if the freshly downloaded bytes are also wrong, the problem is upstream (a re-tagged release asset, a corrupting proxy) and no amount of retrying inside one run will fix it. The message must therefore hand the user the deliberate, explicit retry path rather than implying a transient glitch.

   §5.3 lists model-missing/hash-mismatch as fatal. Within one process, the inputs to the
   decision — a path, a digest, and a loadable runtime — cannot change. No future review
   may re-open this as “do not cache failures” without first overturning §5.3, §16.18,
   and `d1_install.rs`'s frozen rationale.

4. **Panics during init are latched exactly like a rendered refusal.** AMENDED: this
   supersedes the version committed in `19756ed` by Fable tie-break.
   `detector_for` wraps its single `initialize_detector()` attempt in
   `catch_unwind(AssertUnwindSafe(..))`; the payload is rendered with
   `pc_pipeline::panic_message`, preserving §16.12 item 18's pinned `"panicked: ..."`
   form, and latched as the attempt's `Err`. It surfaces as `StageError::Model` under
   item 5's provider-declared fatality: one attempt per run, run-fatal, exit `1`.

   Observed output, corrected against the release binary rather than assumed: the run
   prints `error: model error: panicked: ...` on stderr and exits `1`. It is **not** a
   `fatal:` line — `pc-cli`'s fatal path returns `Err(anyhow!(..))` from `run_pipeline`
   before `BatchSummary::render` is consulted, and `render`'s `fatal:` prefix is
   therefore not what a user sees for a run-fatal provider refusal. Note also that
   `catch_unwind` does not suppress the default panic hook, so for a genuine init panic
   stderr additionally carries the hook's own `thread '...' panicked at ...` line ahead
   of the `error:` line. Suppressing that hook is deliberately not attempted: a
   scoped `set_hook`/`take_hook` pair would race with panics on other rayon workers.

   `initialize_detector` takes no image, so an init panic is independent of the original
   by construction — item 5(b)'s causal criterion classifies it run-fatal, and §16.12
   item 2's “neither outcome is retried within a run” binds the panicking attempt as much
   as the erroring one.

   §5.2 is not violated because its boundary is not relocated. `process_image_isolated`
   still owns panic conversion for every image-dependent unit; the provider converts only
   the image-independent init unwind, which §5.2's own justification (“ort/FFI and
   third-party pixel code can panic on malformed input, and one malformed page must not
   abort a 500-page batch”) never covered, because no page is in scope during init.

   The previous version's acceptance of unbounded per-image re-initialization (~90 MB
   re-hash plus session rebuild per remaining page) to avoid this conversion is withdrawn.
   It contradicted items 2 and 5(b) on their own terms, and treating the unwind channel
   differently from the `Result` channel of the same operation was a mechanism distinction,
   not a principled one.

   Residual panic sources on this path are ort FFI internals and allocation failure only;
   the frozen per-image panic tests in `g2_batch.rs` are unaffected because they panic
   inside `TextDetector::detect`, downstream of init.

5. **Fatality is declared by the provider, never inferred from the variant.**
   The confirmed defect was `single.rs` matching `StageError::Model(_) =>
   PipelineError::RunFatal`, which made every `Model` from every provider run-fatal,
   while `ReplayProvider::detector_for` returned `Model` for a missing fixture under a
   comment stating the opposite intent. A 500-page replay batch missing page 3's
   fixture consequently aborted at page 3, violating §16.12 item 3's closing sentence.
   Root cause, retained so the lesson survives: the removed `RUN_FATAL_MODEL_PREFIX`
   did two jobs — carrying the signal and discriminating which `Model` errors were
   run-fatal. Replacing it with a variant match (correctly, per `outcome.rs`'s comment,
   because control flow needs a type, not message text) kept the first job and silently
   widened the second. §5.3 requires an ONNX refusal fatal; §16.12 item 3 requires a
   replay miss per-image; both are `StageError::Model`.

   (a) **One defaulted provider method is added.**
   `DetectorProvider` gains:

   ```rust
   fn failures_are_run_fatal(&self) -> bool { false }
   ```

   The pinned `detector_for` signature in §16.12 item 3 remains unchanged VERBATIM:

   ```rust
   fn detector_for(&self, original: &Path) -> Result<Arc<dyn TextDetector>, StageError>;
   ```

   A provider error is a per-image `Failed { step: Detect }` **unless the provider
   declares its failures run-fatal** — a missing replay fixture for page 7 must not kill
   pages 1–6.

   (b) **The criterion is causal, not textual.**
   A provider declares `true` iff the failure is independent of `original`, so every
   remaining image would fail identically. `OnnxProvider` is the only `true` provider
   in v1 because its refusal is latched and therefore provably identical per page.
   `ReplayProvider` is `false` because it is keyed by image stem. `SharedDetector`
   never fails during provider resolution.

   (c) **The default is deliberately and asymmetrically `false`.**
   Forgetting to declare `true` degrades one fatal into N per-image failures (exit `2`
   rather than `1`), the failure mode the `x1_cli.rs::a_batch_refuses_once_when_the_model_is_missing`
   property catches. The opposite default kills a 500-page batch on page 3, the
   regression this item fixes and which the old tests did not catch. The default errs
   toward the already-covered failure mode.

   (d) **REJECTED: changing the trait's error type to `ProviderError { Fatal, PerImage }`.**
   It is more expressive and would force the choice rather than defaulting it, but it
   rewrites the signature §16.12 item 3 pins verbatim and forces mechanical edits to
   frozen test implementations in `crates/pc-pipeline/tests/common/mod.rs`. Wrong
   trade. Revisit only if one provider ever needs both classifications.

   (e) **REJECTED: moving `ReplayProvider` to a different `StageError` variant.**
   That recreates exactly the defect class the prefix removal was meant to end —
   fatality carried by an unenforced convention (“`Model` means provisioning only”)
   that the next provider can silently break.

6. **Run-fatal message extraction is generalized in the same change.**
   Item 5 makes a latent hole reachable: `run_fatal_model_message` previously matched
   only `RunFatal { error: Model(_) }`, which was total only while `RunFatal` and
   `Model` were equivalent by construction. A provider may now declare a non-`Model`
   failure run-fatal; if extraction stayed narrow, `exit_code()` could return `0` or
   `2` despite a `RunFatal`, while `render()` printed no fatal — a silently successful
   fatal run. It now matches `RunFatal { error, .. }` for any variant and renders
   `error.to_string()`.

   The detector provider's `map_err` sites likewise retain the inner model message
   rather than stringifying the outer `StageError`, removing the redundant `model
   error: ` segment that produced `error: model error: managed model is missing at '...'`.

7. **`batch.rs`'s bare atomic load stays outside `start_gate`.**
   Best-effort is correct; the asymmetry with `fail_fast` is deliberate. Moving the
   load under the mutex cannot make refuse-once structural: a refusal is not observable
   until at least one image has finished its detect attempt, so with `threads = N`, up
   to N images legitimately start before any refusal exists to observe. The straggler
   bound is the thread count, not the atomicity of the load; the mutex would close a
   window that produces no stragglers. What stragglers cost is bounded by the latch:
   each replays a latched `Err` in O(1), with no re-hash and no reload.

   The asymmetry's reason is contractual. `fail_fast` promises **which images were
   attempted**, so observe-and-decide must be atomic and earns the mutex. Run-fatal
   promises nothing about attempt counts, only that the run ends fatally. Honest limit:
   this invariant covers stdout and the exit code, not side effects. A straggler needing
   no detector — a §4.4 resumed image — may still complete and write exports before the
   refusal is reported, consistent with `batch.rs`'s frozen clause that in-flight work
   is deliberately allowed to finish.

8. **Test impact: frozen-test addition, not amendment.**
   No existing test is amended. The two frozen tests remain byte-identical and are green
   while the item 5 property is violated: `crates/pc-pipeline/tests/g1_chain.rs::a_refusing_detector_provider_fails_only_that_image`
   names §16.12 item 3 exactly but asserts only `outcome.is_failed()` (true for both
   `RunFatal` and `Failed`), and `crates/pc-cli/tests/x1_args.rs::a_missing_replay_fixture_is_a_per_image_error`
   asserts only that the message contains `page01`. Neither contradicts the spec; both
   under-test it, and the missing property is at a layer neither test reaches.

   Add exactly these three tests:

   (a) A `pc-cli` end-to-end mirror of `x1_cli.rs::a_batch_refuses_once_when_the_model_is_missing`,
   using a replay detector and three pages with the middle fixture missing; it must
   exit `2`, not `1`.

   (b) A `pc-pipeline` `run_batch` test pinning both arms of `failures_are_run_fatal`,
   matching on the `ImageOutcome` **variant** and never on `is_failed()` — that predicate
   is exactly why this survived review.

   (c) A test pinning that the trait default is `false`.

   General rule: **a test whose name states a classification property must assert the
   classification, not a predicate that both classifications satisfy.**

   The structural finding was that the fatal path had no coverage under default
   features: the review grep for `RunFatal`/`is_run_fatal`/`fatal_model_message` across
   test directories returned zero, and `EXIT_FATAL` had no unit coverage. Because the
   concurrent job has since added `RunFatal` and `EXIT_FATAL` assertions in
   `crates/pc-pipeline/tests/g2_batch.rs`, that zero-result is recorded as the
   pre-fix finding, not as a claim about the current snapshot.

9. **§14 register hygiene.**
   The code grep currently finds plain `DEVIATION(1, 2, 3, 5, 6, 8, 9, 10, 11, 12,
   13, 14, 15)` coverage. §14 item 4 has a nonconforming test-only `DEVIATION §14.4`
   note, but items 4 and 7 still lack implementation-site comments. The newly backfilled
   register entries 14 and 15 use the plain form, as required. The three §16-qualified
   comment families in the code conform to §16.14
   item 5's convention: `DEVIATION(§16.11 item 5)`, `DEVIATION(§16.11 item 7)`, and
   `DEVIATION(§16.14 item 3)`; §16-only deviations must remain qualified this way.
   The lazy construct-and-latch implementation site is outstanding and will carry
   `DEVIATION(16)` when the concurrent Rust job adds it; it is deliberately not added
   by this documentation-only change.

10. **The session lock spans only the `&mut Session` window, and a poisoned session is
    reused.** `pc-detect/src/onnx.rs::<OnnxDetector as TextDetector>::detect` holds
    `session` for exactly `Session::run` plus the copy of each output into owned
    `Vec<f32>`. All decoding — `bind_outputs`, `validate_output_shapes`, `decode_blocks`,
    `decode_mask` — runs **outside** the lock via the ungated `decode_outputs`, so a panic
    in v1's own arithmetic cannot poison the one session `DEVIATION(15)` shares across the
    run. Previously the guard spanned the whole function, which turned one page's panic
    into an image-independent `Inference("ONNX session mutex was poisoned")` for every
    remaining page: a failure mode v1 manufactured, named after our lock rather than its
    cause, and not asked for by any spec clause.

    (a) **Poison recovery at this site is sound, on a verified fact.** `ort`
    2.0.0-rc.12's `Session::run(&mut self)` (`session/mod.rs:212`) delegates to
    `run_inner(&self, …)` (`:272`); `Session` is `{ inner: Arc<SharedSessionInner>,
    inputs: Vec<Outlet>, outputs: Vec<Outlet> }` (`:114-118`) and `run_inner` only reads
    those fields. No Rust-side session state is mutated during a run, so no panic can
    leave a half-updated invariant; `&mut self` is an aliasing device so
    `SessionOutputs<'s>` cannot coexist with a second run. The cost of a mid-`run` panic
    is a leak (un-`ReleaseValue`'d `OrtValue`s from `:329`; `util/stack.rs:57-66`'s
    `CString`s), not corruption. A panic cannot originate inside ORT's C++: the only Rust
    callbacks it invokes are `logging.rs:108/139`, both `extern "system"`, where an unwind
    aborts. Recovery uses `PoisonError::into_inner` plus one `WARN`. **Any `ort` version
    bump must re-verify `run_inner`'s receiver.** This is a different justification from
    the poison recovery in `pc-cli/src/detector.rs`, which is sound because that mutex
    guards `()`; "we recovered poison elsewhere" is not a reason.

    (b) **The residual panic surface is retained, not asserted away.** Inside the
    narrowed window every remaining panic site is an ort "C API violated its contract"
    assertion: `session/mod.rs:329`, and `Value::from_ptr`'s chain through
    `value/mod.rs:353` → `value/type.rs:384-394`/`:152`. None is a function of pixel
    content; none is provably unreachable, since a mis-built or ABI-mismatched
    `libonnxruntime` could trip them. The branch is therefore handled, never
    `unreachable!()`.

    (c) **`decode_outputs` is a public function over two independent slices, so it
    validates rather than indexes.** It rejects `values.len() != metas.len()` and any
    out-of-range bound index with `StageError::InvalidInput`. Adding a panic site while
    fixing a panic-poisoning bug would be self-defeating. Being ungated, its test runs
    under plain `cargo test --workspace` — coverage the feature-gated path never had.

    (d) **This is NOT the sibling of item 4, and item 5(b) does not promote it.** Item
    5(b) is the predicate for a `DetectorProvider` declaring `failures_are_run_fatal`, not
    a free-floating law over all failures. Item 4's own reasoning fixes the boundary:
    `initialize_detector` takes no image, so an init unwind is image-independent *by
    construction* and knowable a priori at the provider; `TextDetector::detect(&self,
    image)` takes the image, so a panic inside it is image-dependent by construction and
    stays §5.2's business — per-image `Failed { step: Detect, error: Inference("panicked:
    …") }`, exit `2`, converted by `batch.rs::process_image_isolated`. Item 4's closing
    sentence already presupposes this by holding `g2_batch.rs`'s panic tests unaffected
    "because they panic inside `TextDetector::detect`". Reading 5(b) as universal would
    swallow §5.2 whole: any deterministic per-image failure — a profile that makes every
    page fail in masking, a replay dir with every fixture missing — would become
    run-fatal.

## 16.20 Snapshot tests dropped; the oracle moves to the F1 recording (Fable tie-break, 2026-07-29)

Both Opus subagents reviewed the same question — whether §15.10(a)'s hand-traced snapshot
review should be replaced by a recorded run of upstream PanelCleaner — and reached
**opposite** conclusions. Per `CLAUDE.md` the Orchestrator did not pick between them; a
Fable Senior Rust Engineer subagent reviewed both positions and made the final call. This
section records that call. Fable's advisory-only restriction was suspended for this one
decision; it wrote no code.

1. **DECIDED: `insta` snapshot tests are dropped, at both sanctioned sites and everywhere
   else.** §15 item 10's "allowed, narrowly" verdict is **SUPERSEDED**. The replacements:

   (a) **§9.7(B)11** becomes hand-written integer assertions. The entire snapshot content is
   ~15 integers derivable by pure integer arithmetic from the fixture rects and the
   committed profile constants, plus `page_language` and a reading-order permutation. The
   Rust Engineer derived all three tiers from block rect `(567,74,663,123)` — tight
   `(565,72,668,125)`, extended `(560,67,678,130)`, reference `(540,47,698,150)` — and
   matched them against live `#clean.json` output. A hand-written assertion of those
   integers with the derivation in a comment is strictly superior: the expected values are
   visible at the assertion site, and writing them *is* the review.

   (b) **§8.7(A)6** keeps its substance and loses the mechanism: the byte-determinism
   self-comparisons (10 runs, 1 vs 8 rayon threads — these never needed `insta`) plus one
   hand-written equality asserting the produced `PageDataRaw` equals the committed
   `#raw.json`. That equality is a **superset** of what the snapshot would have locked and
   of §8.7(B)9's boxes-only lock.

   Consequences, all ratified here: `insta` leaves §1's dependency list and must not enter
   any crate's manifest; §1's parenthetical becomes unconditionally true — every
   numeric/algorithmic gate uses hand-written assertions; and CI's `frozen-snapshot-guard`
   reduces to rejecting any `*.snap` / `*.snap.new` / `*.pending-snap` anywhere in the tree,
   which makes it trivially precise and retires the **UNVERIFIED CAVEAT** its path allowlist
   carried (the allowlist was derived from insta's naming convention, never from a real
   macro call — there was none to read).

2. **The decisive argument was an inversion, not a preference.** §8.7(A)6's snapshot runs
   `run()` with `ReplayDetector`, which per §7.2.1 is bound to a recorded
   `_detector_mask.png` + `_detector_blocks.json` pair and **ignores the image it is passed**.
   Tracing `crates/pc-detect/src/lib.rs:92-136`: `rect`, `class_index` and `confidence` are
   copied straight out of the fixture into `DetectedBlock` untouched. The only values the
   snapshot *computes* are `mask_coverage`, the survivor set, `scale` and `image_size`.

   So safeguard (a) was aimed at a file that is largely transcription, while
   `_detector_blocks.json` — the artifact that genuinely accepts output as truth — carried
   no safeguard at all. Fable's ruling on Position A's amendment: it *"does not cure that
   inversion — it decorates it."*

3. **DECIDED: safeguard (a) survives, re-aimed at the F1 recording, with a concrete gate.**
   Binding on F1's detector group, in `docs/DETECTOR_ORACLE.md` (a new hand-authored file —
   `docs/GOLDEN_CALIBRATION.md` is generator-owned per §16.13 item 9 and cannot carry a
   human signature):

   (a) **Commit the oracle.** Upstream's pinned-run block output for the committed page goes
   into `tests/fixtures/recorded/` beside our own recording, with a `PROVENANCE.json`
   pinning: upstream version and commit; the invoking command line; the profile with every
   non-default key listed; the detector artifact, its sha256, **and which backend consumed
   it**; the input page's sha256; and each output's sha256. The backend is load-bearing, not
   bookkeeping — see item 6.

   (b) **Assert the comparison in CI**, so it is re-checked on every run rather than once at
   review time. This is Position A's construction and it is adopted whole: for the committed
   page, pair every upstream block with one of ours or with a documented mechanism; assert
   the exact reconstruction identity `upstream.xyxy == bbox(ours.rect ∪ bbox(upstream.lines))`
   on every matched pair; and close the box accounting exactly — matched + §14.13 per-class
   duplicates + documented splits/merges equals both totals.

   **This is a CI test and it does not violate §7.2, because it runs no model.** Both sides
   are frozen committed files — our recorded `_detector_blocks.json` and the upstream oracle
   artifact of (a) — so the check is pure arithmetic over two JSON documents. Separate the two
   acts and the apparent conflict dissolves: **producing** the artifacts requires running both
   detectors and is therefore maintainer-local and one-off; **comparing** them requires
   nothing but the files and therefore belongs in CI, on every run, forever. Getting this
   backwards is what makes the gate look unbuildable (see item 11).

   Two implementation requirements follow, both non-optional. The identity must handle a
   line-less upstream block, where `bbox(upstream.lines)` is undefined and the identity
   degenerates to `upstream.xyxy == ours.rect` — v1 synthesizes no lines, and upstream's
   line-less branch is exactly where §14.17's filter divergence lives, so this case is not
   hypothetical. And the comparator must **assert the number of pairs it compared** against a
   hard-coded expected count: a comparator reporting "compared 0 boxes, 0 divergences → PASS"
   is cookbook rule 1's defect, and it is the failure mode a frozen-file comparison is most
   prone to, since a renamed field silently yields an empty pairing rather than an error.

   (c) **No tolerances and no IoU thresholds in any gating row.** Where a divergence has a
   known mechanism the evidence is an exact identity, not an epsilon. See item 5 for the
   measurement that forces this.

   (d) **`confidence`, `language` and `raw_mask` are DIAGNOSTIC or NO-ORACLE rows only,
   never gated.** `raw_mask` has no exact oracle in v1 at all: upstream's preprocessor calls
   the detector with `refine_mode=REFINEMASK_ANNOTATION` and `keep_undetected_mask=True`,
   i.e. the full `refine_mask` algorithm that §14.12 puts out of scope — measured **IoU
   0.258**, upstream 2,950 non-zero px against our 11,103. The refine arithmetic is gated by
   the synthetic primaries §8.7(A)4/5 instead, and the doc must say so plainly rather than
   implying the mask was checked.

   (e) **Completeness partition.** Every serialized field of `PageDataRaw` lands in exactly
   one bucket — `ORACLE-EXACT` / `EXPLAINED-§14.x` / `DIAGNOSTIC` / `NO-ORACLE` — checked
   against §2.4/§2.5's field list. **(ERRATUM, §16.24 item 10: the §2.5 half of that citation
   is wrong — §2.5 is `PageData`, the preprocessor's output, not the detector's. The partition
   covers §2.4's `PageDataRaw` + `DetectedBlock` plus `RawBlock`'s three serialized fields;
   §2.5's fields appear as `OUT-OF-SCOPE-F1` rows. §16.24 item 11 also makes this partition a
   test rather than checklist prose.)** A field absent from the table is a defect in the review,
   not an omission. `OPEN` is a blocking state, not a verdict: any open row blocks the
   fixture commit. **A row may not close as `EXPLAINED` against a §14 or §16.x entry that
   does not yet exist** — the ratification lands first, with its `DEVIATION(n)` comment at
   the implementation site, and only then may a row cite it.

   (f) **Signatures.** The producing agent may perform the upstream run and author the table
   — that is a mechanical, fully specified act a third party can reproduce and falsify from
   committed provenance, so it does not create the self-reference a hand-trace did.
   Authoring the **verdicts** does. Three signatures are required before the fixture commits,
   none of them the producing agent's: both Opus reviewers, each re-deriving the identities
   from the committed artifact, and the human maintainer, signing the completeness of the
   partition and every row resting on judgment rather than an identity. Reviewer, date and
   method are recorded per signature.

   Machine-dependent absolute paths must be normalised out of both sides before commit.
   Upstream's `#raw.json` embeds `image_path` / `mask_path` / `original_path` as absolute
   `str(Path)` values — across six identical upstream runs those three strings were the
   *only* thing that differed — and our own `PageDataRaw` embeds them too. §7.2 already
   requires rebasing our recorded handles; the oracle artifact needs the same treatment.

4. **ERRATUM: the snapshot site was misidentified in two places.** §15 item 10 and §1's
   `insta` note both name **§8.7(B)9** as a snapshot site. It is not: `:739` item **6 of (A)
   Primary** is the snapshot, and §8.7(B)9 at `:745` is a different test — "box count and
   each box's coordinates match the recorded `#raw.json` exactly (this is a regression lock,
   not a Python-parity claim)" — with no `insta` and no snapshot. Left uncorrected, the
   clause authorised a snapshot at `b9_pending_recorded_page_regression_lock`
   (`crates/pc-detect/tests/d7_run.rs`), a path CI's allowlist did not permit. Both
   references are corrected. Moot for the snapshot decision above, but recorded because the
   same cross-reference is what a future reader would follow.

5. **The measurement that forecloses tolerance-based comparison.** The Rust Engineer
   decomposed a 0.775 → 0.704 confidence gap by substituting one stage at a time into
   upstream's own cv2.dnn engine: engine (cv2.dnn vs ORT) ≈ 0.008, image decode ≈ 0.001, pad
   colour ≈ 0.001, and **our bilinear resize vs `cv2.INTER_LINEAR` ≈ 0.079** — from a resize
   difference of **max Δ = 1 LSB**, 77% of pixels exact. Our implementation is not wrong;
   `onnx.rs` uses the correct half-pixel convention and the difference is f32-exact versus
   OpenCV's fixed-point weights.

   Fable reproduced this independently with its own perturbation (±1 LSB on 25% of
   letterboxed pixels of the candidate page, through upstream's engine): confidences moved up
   to **±0.089** while every box coordinate stayed within L1 = 1, four of five exact.

   Two conclusions follow, and they are why item 3(c) and 3(d) read as they do. **`confidence`
   is not adjudicable against upstream at any useful tolerance** — the noise floor from
   unavoidable rounding is ~0.08 while §8.3 step 4's objectness and class gates both sit at
   0.4, so a tolerance absorbing the noise spans "kept" and "dropped". And **geometry is
   exact-gateable**: two independent probes found coordinates stable under the same
   perturbation that moves confidence by 0.09.

   A further reason to refuse a rect tolerance: with `dw = 284, dh = 0` on the candidate page,
   a one-pixel error in `dw` shifts a box edge near x = 740 by ≈1.5 px. A ±3 px per-edge
   tolerance would therefore absorb an off-by-one in the letterbox padding — precisely the
   defect class §14.14 exists for — while spanning ±12–19% of area on boxes of 66×41 to
   117×72 px.

6. **With a `.onnx` model, upstream does not use PyTorch for inference at all.**
   `inference.py:148-151` selects `cv2.dnn.readNetFromONNX` and `TextDetBaseDNN`
   (`basemodel.py:251-261`); torch enters only in `non_max_suppression`, which converts the
   numpy output back to a tensor and calls `torchvision.ops.nms`. So the oracle is
   *"upstream + cv2.dnn"*, not *"upstream"* — and the two branches differ in **channel
   order**: the torch branch of `preprocess_img` applies `[::-1]` after
   `cvtColor(BGR2RGB)` (`inference.py:85-88`), feeding BGR, while the cv2 branch feeds RGB. A
   maintainer who records the oracle from the `.pt` weights therefore gets a different code
   path and a different answer. This is why item 3(a) pins the backend.

7. **Upstream determinism, measured.** Six runs of `process_image` on the candidate page
   with a fixed uuid — three at the default 20 OpenCV threads, one at 1, one at 4, one with
   `cv2.setNumThreads(1)` plus `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` — produced
   byte-identical `_raw_mask.png` and `_base.png` (sha256 `b48bc0c6…ca23` and `b6e29da1…6d974`)
   and byte-identical `#raw.json` after path normalisation: every box, every line polygon,
   every float. Upstream seeds nothing and needs to: the graph is inference-only under
   `@torch.no_grad()`, and `torchvision.ops.nms` is sequential greedy.

   **This licenses less than it appears to.** It says nothing about a different CPU ISA or a
   different OpenCV version, and item 5 shows this model's confidence output is chaotically
   sensitive to 1-LSB input perturbations. So a re-recorded oracle on another machine may
   legitimately differ, which is the real argument for freezing the committed artifact rather
   than trusting the recipe.

8. **ERRATUM: `PAD_VALUE` is 0, not 114.** §8.3 step 3 said letterbox pads right/bottom with
   `(114,114,114)`. Upstream's `letterbox` signature defaults to `color=(0, 0, 0)`
   (`imgproc_utils.py:95`) and `preprocess_img` passes no `color` argument
   (`inference.py:86`), so upstream pads **black**. 114 is *yolov5's* letterbox default,
   which comic-text-detector does not use. Both Opus reviewers independently held 114 to be
   wrong; the Orchestrator verified the two upstream call sites directly.

   Recorded as an **erratum, not a §14 deviation.** §8.3 step 3's own framing is a port of
   `preprocess_img` + `letterbox` keeping upstream behaviour, so upstream is normative for
   this line and 114 contradicts the very thing the line claims to port. A §14 entry would
   enshrine yolov5's default purely because a test already pinned it, and would leave every
   future recorded fixture permanently off-oracle — weakening item 3(b)'s exact-identity gate
   for no benefit.

   This required editing a **frozen test** (`crates/pc-detect/tests/d4_onnx.rs`'s
   `assert_eq!(PAD_VALUE, 114)`, and the comment on
   `letterbox_pads_the_right_edge_and_leaves_the_top_left_as_content`, which got the *side*
   right — right/bottom, not yolov5's centred padding — and the *value* wrong). `CLAUDE.md`
   routes a frozen test contradicting the spec to the joint architects; **this ruling is the
   recorded sign-off**, and it must land before F1 records anything, since the pad colour
   perturbs every recorded box. Measured cost of the change: identical rects, one confidence
   0.714 → 0.715 — an order of magnitude inside item 5's noise floor.

9. **ERRATUM: §8.3 step 6's provenance sentence was wrong twice.** Spec line 686 justified
   our coverage filter as *"exactly the code path upstream takes when the line map yields
   nothing."* Both halves are wrong:

   (a) **Wrong scope.** Upstream's `group_output` (`textblock.py:485-490`) wraps the
   `mask_score < mask_score_thresh` filter inside `if len(blk.lines) == 0:`, so upstream
   applies it **only to line-less blocks**. We apply it to every block.

   (b) **Wrong operand.** `inference.py:203` calls `group_output(blks, lines, im_w, im_h,
   mask)` with the **unrefined** mask; `refine_mask` runs afterwards at `:204`. Our filter
   uses the **refined** mask. Measured on the same boxes, the two quantities are roughly 2×
   apart — upstream 0.366 / 0.466 / 0.260 against our 0.773 / 0.781 / 0.564.

   The sentence is corrected, and the deviation it was pretending not to be is ratified as
   §14 register entry **17**. Risk evidence recorded with it: across two real manga pages the
   filter has **never fired**. On a 12-box page the minimum `mask_coverage` was **0.3025**
   against the 0.1 threshold — a 3× margin. On the candidate page upstream's line-less branch
   *did* fire once, on `[438,1407,498,1446]` (`mask_score` 0.0359), and both implementations
   dropped that box: **outcome agreement, mechanism divergence.** That is worth stating
   explicitly, because a divergence table over final artifacts would have shown a clean result
   and hidden the structural difference — cookbook rule 3's own lesson applied to the oracle
   being proposed.

10. **What this ruling does NOT settle.** The committed page fixture and §7.2's 400 KB cap
    remain open and are deliberately deferred: they gate only the final fixture commit, not
    any of items 1–9, and treating them as a blocker had already stalled work that does not
    depend on them. Local validation continues on pages that cannot be committed. Recorded so
    the deferral is a visible decision rather than a gap.

    Two further items are triggered but unbudgeted, and neither is created by this ruling —
    both pre-date it: §16.17 item 2's shared `PROVENANCE.json` schema and checker, whose
    activation condition ("the first recording group whose consuming tests read values rather
    than compare pixels") is met by the detector group and therefore falls due **before** it
    records; and `cargo xtask record-fixtures --only detector`, which is a stub —
    `xtask/src/record.rs` returns `Outcome::Skipped` unconditionally. `xtask` also has zero
    `#[test]`s, so the comparison harness of item 3(b) must ship with its own negative
    controls: a synthetic artifact pair with one box perturbed by 1 px, one confidence by
    0.001, and one box deleted, asserting the comparator reports exactly those three
    divergences and no others. A comparator that prints "compared 0 boxes, 0 divergences →
    PASS" is cookbook rule 1's defect wearing a suit.

11. **The oracle gate of item 3 is specified but NOT implemented, and that gap is tracked
    here rather than left to be discovered.** Dropping the snapshots removed a gate that
    enforced nothing — both sites were `#[ignore]`d `unimplemented!()` placeholders — so net
    enforcement did not fall. But "no regression" is not "there is a gate", and until F1
    records, **nothing checks our detector output against upstream.** What does hold in the
    meantime: §8.7(A)1–5's synthetic primaries (letterbox geometry, decode, NMS, refinement
    arithmetic, the coverage cutoff at 25/26·255⁻¹) are hand-written, executed, and
    independent of any fixture; §16.16's recorded model signature closes the D4a
    self-reference; and the `onnx` tier now actually runs (§16.17). The unguarded surface is
    specifically *agreement with upstream on real image content*.

    **§7.2 constrains *how* the gate is built, not *whether* it can be a CI check — and an
    earlier draft of this item got that wrong.** The correction matters, because the wrong
    version makes item 3 read as unbuildable and would justify never building it:

    - **Producing** the two artifacts runs both detectors, so it is maintainer-local and
      one-off. §7.2's prohibition binds here.
    - **Comparing** them runs no model — both are frozen committed JSON files — so it is an
      ordinary CI test that executes on every run. §7.2 does not reach it.

    So the gate of item 3(b) *is* shippable as a CI test; what cannot be automated in CI is
    the recording step, which is true of every fixture in `tests/fixtures/recorded/` and is
    precisely the design §7.2 already chose ("record once, replay forever"). The residual
    risk is therefore narrow and nameable: an **unreviewed recorded fixture**, whose values
    the replay path and the identity test both trust. That is what item 3's signatures exist
    to cover, and it remains the project's highest-value outstanding risk — not because the
    gate is impossible, but because the fixture does not exist yet.

12. **Fixture-selection criterion, measured: the page must sit near letterbox scale 1, and
    `demo_bubbles` cannot serve as the oracle fixture.** Recorded because the shortcut is
    tempting — the seven `demo_bubbles/*_raw.png` crops are already committed, GPL-3.0 and
    licence-clean, so using them would sidestep the page question entirely — and because the
    reason it fails is quantitative, not aesthetic.

    Upstream finds **21 blocks across the seven crops** (six produce boxes; `handwritten`
    yields 0 on both sides). Both sides report `scale = 1.0`, so there is no §8.2 resize
    confound. Yet the box counts diverge badly — `nightmare` gives us **1** against
    upstream's **5**, `square` 1 against 2, `darkrays` 2 against 3.

    The mechanism is item 5's noise floor, amplified. These crops are 72×132 to 354×354, so
    §8.3 step 3's letterbox **upscales them by r ≈ 3.0–3.8**, which magnifies the
    f32-exact-vs-fixed-point resize difference. Our surviving confidences are 0.437, 0.444,
    0.564, 0.696 — a minimum margin of **0.037** above §8.3 step 4's 0.4 gate, against a
    measured noise floor of **0.079–0.089**. The noise is ~2.4× the margin, so box
    *presence* on these inputs is decided by rounding rather than by content.

    Crucially, this is **not** evidence of a port defect: where a box survives both sides its
    geometry agrees closely and consistently with item 3(b)'s identity — ours
    `(40,23,111,207)` against upstream `[40,23,110,207]`, ours `(29,105,86,261)` against
    upstream `[29,105,90,261]`, the trailing-edge gap being upstream's line union. Geometry is
    portable here; presence is not.

    **The geometry half of the paragraph above is SUPERSEDED — see §16.24 item 18.** It is
    preserved verbatim rather than deleted, per §16.19's convention. In short: a bounding union
    is monotone, so item 3(b)'s identity entails `upstream.x2 >= ours.x2`; the first pair has
    ours `x2 = 111` against upstream `x2 = 110`, which a union cannot produce. That pair is
    therefore evidence *against* the identity, cited in support of it, and the attribution of
    its gap to "upstream's line union" is impossible as transcribed. The words "closely" and
    "consistently" are both withdrawn, as is the "not evidence of a port defect" reassurance
    this same sentence was carrying. **Item 12's conclusion is unaffected and in fact
    strengthened** — see §16.24 item 18(g).

    **The criterion this yields:** the oracle fixture must be a full page whose letterbox
    ratio is near 1, so confidences sit far from the 0.4 gate. That is what §7.2's "one or two
    full manga pages" was already asking for; this item supplies the missing *reason*, so a
    future maintainer does not substitute a convenient small crop and inherit a fixture whose
    box set is noise-determined. The `demo_bubbles` crops remain correct for what §10.7(B)15
    and §11.7 use them for — masking and denoise comparisons on a fixed input — where no
    detection gate is involved.

## 16.21 Detector threading erratum — `intra_threads = 1` rested on a false premise (Fable tie-break, 2026-07-29)

1. **ERRATUM: §8.3 step 3's `intra_threads = 1` is corrected to a configurable value
   defaulting to `0` (all physical cores).** Line 660 read *"Execution provider: CPU only,
   `intra_threads = 1` (parallelism is at the image level), session created once and
   shared."*

   **The parenthetical justification was already false in the shipped design.** §14.15 /
   `DEVIATION(15)` makes the detector session a single `Mutex<Session>` shared across the
   whole run (`crates/pc-detect/src/onnx.rs`), so image-level parallelism **cannot reach the
   detector at all** — inference is serialized by the mutex *and* single-threaded inside it.
   Two separately-ratified decisions contradicted each other, and the cost was the entire
   per-page runtime.

   This is therefore an **erratum**, not a change of mind about performance: the normative
   line's stated premise did not hold. Recorded the same way as §16.20 item 8's `PAD_VALUE`
   erratum, and like that one it authorises a **frozen-test edit** —
   `crates/pc-detect/tests/d4_onnx.rs`'s `assert_eq!(INTRA_THREADS, 1)` and the
   `constants_match_the_spec` comment block. This ruling is the recorded joint sign-off;
   the edit may land only after this section is in the tree.

2. **Measured, three times independently, that thread count changes speed and not results.**
   This is what licenses the change; without it the new default would be a gamble against
   §5 item 7.

   | source | 1 thread | 4 | 8 | 16 / 20 | output |
   |---|---|---|---|---|---|
   | Opus Senior Rust Engineer | 149.28 s | 39.46 s | 20.28 s | 12.69 s (16) | 0 of 3,597,312 floats differ |
   | Orchestrator | 142.41 s | 37.89 s | 20.81 s | 27.87 s (20) | bit-identical, all three outputs |
   | Fable | 140.7 s | 38.6 s | 20.1 s | 29.7 s (20) | all 3,597,312 bit-identical |

   Each run also included a same-thread-count control, itself bit-identical, so the
   comparison isolates thread count from run-to-run noise. End-to-end confirmation: one page
   through the release binary took **156.65 s**.

3. **The default is `0`, not a hardcoded count.** Two of the three measurements show **20
   threads slower than 8** (27.87 s and 29.7 s against ~20.5 s) — oversubscription is real
   on the measured hardware, so any fixed number is wrong somewhere. `0` defers to ONNX
   Runtime's physical-core default. Implementation note: confirm `with_intra_threads(0)`
   reaches ORT's default semantics in `ort` rc.12; if the builder rejects `0`, fall back to
   `std::thread::available_parallelism`.

   `1` — and any explicit value — remains expressible via `[text_detector] intra_threads`,
   because it is what every fixture recorded before this change used.

4. **No new §14 register entry.** Upstream's `cv2.dnn` defaults to all cores, so
   `intra_threads = 1` was itself an *undocumented deviation from upstream*; raising it
   moves us toward upstream rather than away. §14 records deliberate divergences, and this
   removes one.

5. **Not a fixture-affecting change**, on the evidence in item 2 — but `PROVENANCE.json`
   records the thread count actually used, per §16.20 item 3(a) as amended by §16.22 item 6.
   Provenance records facts, not inferences.

6. **The 11× gap against `cv2.dnn` is a separate, open finding and is NOT closed by this
   erratum.** Measured at matched thread count on identical model bytes: `cv2.dnn` at 16
   threads **1.20 s** against our `ort` CPU EP at 16 threads **13.6 s**. So even after this
   change a CPU-only user — the default build, and what CI would run — carries ~13 s of
   inference per page against upstream's ~1.2 s.

   Ratified as a **bounded investigation**, not a runtime replacement: profile to name the
   hot nodes, examine session options, execution mode, thread affinity under WSL2, and an
   `ort`/ORT *version* bump as a candidate. Hard constraints: the model bytes are immutable
   (sha256-pinned by §16.16); no runtime replacement in v1.5, because it would invalidate
   §16.16's recorded signature, the D4 surface, and §16.20 item 6's *"the oracle is upstream
   + cv2.dnn"* pinning; and any fix that perturbs recorded floats — **including a version
   bump** — is fixture-affecting and triggers re-record plus re-sign, so it must arrive with
   measurements before adoption.

## 16.22 GPU execution providers move to v1.5, opt-in and quarantined (Fable tie-break, 2026-07-29)

The maintainer asked for GPU support on an RTX 4090. The two Opus subagents split — the
architect sequenced a full CUDA path into v1.5, the Senior Rust Engineer argued for the
config surface only with real CUDA held at v2 where the spec already had it. Fable ruled for
shipping it, with the engineer's objections converted into binding conditions. Recorded here
because it is a **scope change**, not a gap-fill.

1. **DECIDED: CUDA ships in v1.5, opt-in.** Line 3464's *"GPU execution providers
   (CUDA/CoreML/DirectML), `.pt`/torch loading, multi-device dispatch — v2"* is amended:
   **CUDA moves to v1.5**; CoreML, DirectML, `.pt`/torch loading and multi-device dispatch
   stay v2. Line 690's out-of-scope list and line 660's "CPU only" are amended accordingly.

   DirectML is rejected on read evidence rather than packaging preference:
   `ort/src/ep/directml.rs` gates `supported_by_platform()` on
   `cfg!(target_os = "windows")`. TensorRT/NVRTX need `libnvinfer.so.10`, which no `ort`
   distribution ships.

2. **CARVE-OUT to §5 item 7 (line 461), which is the clause CUDA actually violates.** §5
   item 7 requires *"identical inputs + identical config must produce identical outputs …
   regardless of thread count."* A CUDA run measurably breaks this. The guarantee therefore
   holds **unconditionally for the CPU execution provider**; the opt-in CUDA provider is
   exempted, and CI, fixtures, recordings and every gate remain under the unconditional
   clause.

   **Correction to the record:** the Senior Rust Engineer argued GPU breaks §8.7(A)6. It does
   not — that test drives `ReplayDetector`, so **no model executes**. Right conclusion, wrong
   statute; noted because a future reader would follow the citation.

3. **CUDA is not bit-reproducible against itself, and the magnitude of the
   nondeterminism is itself nondeterministic.** Two independent measurements of two identical
   back-to-back CUDA sessions on the same page:

   | measurement | `blk` | `seg` | `det` |
   |---|---|---|---|
   | Opus Senior Rust Engineer | 0 | 91 values, maxabs 1.19e-07 | 0 |
   | Fable | bit-identical | 2,367 values, maxabs 1.07e-06 | **52,465 values (2.5% of the map), maxabs 3.4e-04** |

   Mechanism: `ConvAlgorithmSearch::Exhaustive` is `#[default]` in `ort` rc.12
   (`src/ep/cuda.rs:57`, whose own doc note says so) — cuDNN benchmarks convolution
   algorithms per session and may choose differently. **Our session options must pin
   `Heuristic` or `Default`**; shipping the Exhaustive default is forbidden. That minimises,
   and does not eliminate, run-to-run drift.

4. **Cross-provider divergence, measured.** CPU vs CUDA on the same page: max |Δconfidence|
   **0.0880** and mean 0.0033 on geometry-identical boxes — squarely on §16.20 item 5's
   0.079–0.089 noise floor, where item 3(c) already ruled no confidence tolerance is
   defensible; 37 of 61 pre-NMS rows differ; and **one CUDA-only box `(606,631,724,703)`**
   with no CPU-only counterpart, so §16.20 item 3(b)'s box accounting cannot close. Speed:
   CUDA inference 13–17 ms against 13.6 s on our CPU EP at 16 threads.

   Recorded as a **diagnostic**, never a gate.

5. **Binding conditions — all of them, and the reason the objections above do not block
   shipping.** Non-reproducibility, box-set drift and the noise floor are arguments against
   *gating* GPU output, which nobody proposed. Once GPU is quarantined from every
   correctness-bearing artifact, misplacement degrades **speed, never correctness**:

   (a) **Opt-in only.** `device = "cpu"` is the default; `"cuda"` must be explicit. Never
   auto-detected.

   (b) **Recording hard-refuses GPU.** `xtask record-fixtures` and every fixture-producing
   path *fail* if `device != cpu` — a refusal, not a discouraged override.

   (c) **Loud refusal at session creation.** EP registration uses error-on-failure, and with
   `device = "cuda"` a registration failure is a **rendered fatal refusal** through §16.19's
   provider-declared-fatality machinery. `ort`'s default is the opposite:
   `ExecutionProviderDispatch { error_on_failure: false }` (`src/ep/mod.rs`), documented as
   *"silently fail and fall back to … the CPU provider."* Countermanding that is the single
   most important line in the feature — a run reporting success while secretly executing at
   ~150 s/page is worse than a crash because it is invisible, and it is what §14 item 7's
   *"an opt-in setting must not silently behave differently"* exists to prevent.

   (d) **Downgrade-guard test.** `ort-sys`'s build-time resolver silently falls back to the
   CPU-only `none` distribution when the requested feature set has no distribution for the
   target (`build/download/resolve.rs:73-79`, `log::warning!` then `find_dist(&target,
   "none")`). A `cuda`-feature-gated test must assert CUDA registration *succeeds*, so this
   is caught by tooling rather than by a confused user.

   (e) **"We are on GPU" is not claimable, and must not be claimed.** ONNX Runtime assigns
   nodes to providers **per node**; unsupported operators fall back to CPU silently.
   `ort` rc.12 exposes no session provider list (no Rust counterpart to Python's
   `session.get_providers()`) and no placement introspection (`src/ep/mod.rs` documents that
   a compiled-in EP "does not always mean the execution provider is usable for a specific
   session"). Report **what was requested and what registered** — never per-node placement.

   (f) **The device policy resolver is model-agnostic**, so v1.5's LaMa inpainting reuses it
   instead of growing a second one.

   (g) **Sequenced after §16.21's threading fix and after F1 records.** §16.20 item 11 names
   the missing oracle gate as the project's highest-value open risk; it does not wait behind
   new scope.

   (h) **Runtime prerequisite, and it is the user's to satisfy.** `ort`'s CUDA distribution
   ships `libonnxruntime_providers_cuda.so` but none of its dependencies. Verified by `ldd`:
   `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`, `libcudnn.so.9`,
   `libcufft.so.11`, `libcurand.so.10` — i.e. **CUDA 12 runtime and cuDNN 9**. Confirmed
   satisfiable by user-local pip wheels (`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`,
   `nvidia-cudnn-cu12`, `nvidia-cufft-cu12`, `nvidia-curand-cu12`), ~1.5–2 GB, no `sudo`,
   and verified running inference on the target hardware.

   Use the **cu12** distribution, not cu13, despite a driver reporting CUDA 13.1: 13.1 is the
   driver's *maximum* supported version and CUDA 12 is backward-compatible with it; cu12 is
   the resolver's default with no `CUDA_HOME` and no `nvcc`; and decisively, `ort`'s
   `ep::cuda::preload_dylibs` carries hardcoded CUDA-12-only library lists with no cu13
   variant, so choosing cu13 breaks the one helper that locates these libraries. `nvcc` is
   not required at all — we load prebuilt kernels rather than compiling any.

6. **§16.20 item 3(a) is amended: `PROVENANCE.json` must also pin *our* execution provider
   and thread count.** It already pins which backend consumed the model on upstream's side,
   noted there as "load-bearing, not bookkeeping." With a second provider and a variable
   thread count in the tree, the same symmetry becomes load-bearing on ours.

7. **§16.20 item 3(d) is reaffirmed unchanged.** `raw_mask` stays NO-ORACLE. The upgrade path
   is documented as blocked on `MaskRefineMode::Annotation` landing with its own
   ratification — see §16.23 item 2.

## 16.23 v1.5 scope and sequence (Fable tie-break, 2026-07-29)

1. **Both plans had scope omissions, in both directions.** The Opus Senior Rust Engineer
   correctly found three spec-assigned v1.5 items missing from the brief — LaMa inpainting,
   PSD/layered export, legacy INI config import — and its point stands that a "v1.5 plan"
   silently dropping LaMa and PSD is not one. But it then omitted **Lab-space coloured NLM +
   `color_filter_strength`**, also spec-assigned v1.5, which both plans forgot; and it treated
   dropping the font-rendering debug visualisations as free when they are a flat v1.5 item.
   **Silent dropping cuts both ways: a deferral needs ratifying by the same standard.**

   **v1.5 ships:** §16.21's threading erratum; F1 and its due prerequisites; CUDA per §16.22;
   LaMa inpainting; PSD/layered export; legacy INI import; Lab-space coloured NLM +
   `color_filter_strength` (landing it also retires §14 item 7's one-time WARN); DBNet line
   synthesis plus line-based splitting/merging and orientation/font-size estimation;
   `MaskRefineMode::Annotation`; and §16.21 item 6's bounded CPU-EP investigation.

   **Ratified deferrals, recorded rather than omitted:** font-rendering debug visualisations
   (`_raw_boxes.png`, `_boxes.png`, `_boxes_final.png`, `_mask_fitments.png`,
   `_std_devs.png`) → **v2**, since a font stack for debug-only artifacts gating nothing is
   poor value, and the maskers they would debug are already gated by §8.7(A)4/5's synthetic
   primaries. Tesseract/non-Japanese OCR and OCR result parsers, post-action hooks, memory
   watcher, i18n, and `stitch_all` are all spec-marked "v1.5+", so deferring them is
   spec-compliant, and each is recorded here as deferred.

   **Not built:** the `heuristic_median` O(n²) improvement
   (`crates/pc-mask/src/border.rs`) — measured at ≤1% of pipeline runtime. Revisit only if
   profiling after §16.21 contradicts that.

2. **`MaskRefineMode::Annotation` must NOT land before F1 records.** Requested as a way to
   make §16.20 item 3(d)'s `raw_mask` row gateable; refused on four independent grounds, any
   one sufficient:

   (a) **It is circular.** Flipping the row to gateable requires the Annotation port to be
   faithful, which is exactly what has no gate — using an ungated new port to manufacture
   the oracle that should have gated it. That is cookbook rule 7 one level up.

   (b) §16.20 item 3(e) **already** forbids closing a row against a ratification that does
   not yet exist.

   (c) It **inverts the risk ordering.** §15 item 2 calls `refine_mask` *"the single riskiest
   numerical port in the whole project"*; sequencing the largest open risk (no upstream
   agreement check at all) behind it maximises the unguarded window.

   (d) **Waiting costs nothing.** F1 records `Simple`-mode output — v1's shipped default — so
   Annotation landing later invalidates no fixture. Order: F1 records with `raw_mask`
   NO-ORACLE exactly as ratified → Annotation lands with its own upstream comparison on the
   committed page → a new §16.x may then upgrade the row. **Annotation is opt-in and `Simple`
   remains the v1.5 default**, so default-path fixture churn for that task is zero.

3. **§16.20 item 3(b)'s reconstruction identity does NOT collapse to plain equality once we
   synthesize our own DBNet lines.** The gate structure is bound now so the DBNet task cannot
   be planned against a false assumption.

   The mechanism is dispositive. Boxes come from YOLO's **regression** head and are measured
   portable — 10/10 blocks and 40/40 coordinates on a 12-box page, and the `blk` tensor was
   bit-identical even across CUDA reruns (§16.22 item 3). Line polygons come from
   **thresholding** the `det` map through `SegDetectorRepresenter` — contour extraction plus
   polygon approximation plus unclip — and that map differs across engines at maxabs 0.00966
   over 92.77% of pixels against a binarisation threshold near 0.3. **Sub-LSB value noise
   becomes contour topology**: one contour splits into two, or two merge. §16.20 item 3(c)
   already forbids the only instruments (epsilon, IoU) that could absorb topology
   instability.

   Therefore, when line synthesis lands: the exact gate anchors on the **YOLO-regression
   rects**; line polygons get their own row(s), **presumptively DIAGNOSTIC**, carrying
   quantitative diagnostics (per-block line count, bbox-of-lines delta) and never an epsilon
   gate; and final post-union rects are compared by a reconstruction identity using each
   side's own recorded lines, with residuals closing as `EXPLAINED-§14.x` or the row staying
   `OPEN` and blocking. If implementation-time measurement shows line topology is stable
   enough to gate exactly, **upgrading requires a new §16.x** — the plan must not assume it.

4. **§14 item 17 must be re-ratified before DBNet lines land.** Its recorded risk evidence is
   that the deviation is *vacuous for v1* — v1 synthesizes no lines, so every block is
   line-less by construction — and that the filter has never fired. **Both halves evaporate
   the moment lines exist.** §16.20 item 3(e) forbids closing a row against a ratification
   that does not yet exist; the mirror rule is hereby stated: **a ratification whose stated
   justification is "vacuous for v1" may not silently carry into a version where it is not
   vacuous.** Re-ratification is a §16.x amendment, not code, and it gates the DBNet task.

5. **Execution sequence.** `V15-0` is this ratification pass (§16.21, §16.22, §16.23) and
   **lands before any v1.5 code**; then **PERF-1** (§16.21's threading change, including the
   authorised frozen-test edit); then **F1** — the `PROVENANCE.json` schema and checker due
   under §16.17 item 2, un-stubbing `xtask record-fixtures --only detector`, the §16.20 item
   3(b) comparator with item 10's negative controls, `docs/DETECTOR_ORACLE.md`, the
   completeness partition and the three signatures (**F1's plan is ratified by §16.24**, which
   also adds the upstream-oracle recording script of §16.20 item 3(a) to F1's scope); then
   **GPU-1** (device config, the
   model-agnostic policy resolver, §16.19-integrated fatal refusal, recording refusal — no
   CUDA linkage); then **GPU-2** (the `cuda` feature and its guards); then §16.21 item 6's
   investigation, which may run any time after PERF-1; then legacy INI import and Lab NLM
   batched; then LaMa; then PSD; then **DBNet lines**, which change default detector output
   and therefore end with an F1 re-record and full re-sign, budgeted once, up front; then
   **Annotation** last.

   The committed page fixture and §7.2's 400 KB cap remain the one open prerequisite, per
   §16.20 item 10. It gates only the final fixture *commit* — not building F1's machinery —
   and is a maintainer decision rather than an engineering task.

## 16.24 F1 plan tie-break: the provenance schema vs. the frozen gate, and how oracle pairs are formed (Fable tie-break, 2026-07-29)

Both Opus subagents produced full F1 plans and disagreed on two clusters: whether
`crates/pc-testkit/tests/recorded_provenance.rs` is replaced or preserved, and whether the
§16.20 item 3(b) comparator's pairing is a signed input or a computed correspondence. Per
`CLAUDE.md` the Orchestrator did not pick; a Fable Senior Rust Engineer subagent reviewed both
positions and made the final call. This section records it. Fable's advisory-only restriction
was suspended for this decision; it wrote no code.

Ratified without restatement, because both plans independently reached it: one canonical schema
replacing all three committed shapes, migrated, with `#[serde(deny_unknown_fields)]`; the schema
in `crates/pc-testkit/src/provenance.rs` (`xtask` has **no lib target** — verified — so nothing
can depend on it); the comparator in `crates/pc-detect/src/oracle.rs` under
`cfg(any(test, feature = "testkit"))`, running in the **default** CI tier; ours and the upstream
oracle sharing **one** `detector/` group directory; a digest-shaped key *and* value sweep over
free-form maps in addition to the typed layer; the 0.001 confidence perturbation never being a
gating divergence; negative controls constructed in-test and never committed.

1. **DECIDED — the frozen test is PRESERVED; the engineer's position wins. No replacement is
   authorised.** Cookbook rule 8 names three exits from a frozen test, and replacement qualifies
   only under exit 2 — contradiction with the spec. No contradiction exists: the engineer's
   migration design proves the test can stay green with **zero edits**, so the precondition for
   the exit the architect invoked is absent. The §16.20 item 8 precedent the architect leaned on
   *was* a genuine contradiction (a test pinning `PAD_VALUE == 114` against a normative upstream
   0); this is not that. The architect's own risk table concedes the point by naming its
   replacement task "the highest-risk task in F1" — it replaces a gate that took five iterations
   to get right, **every one of which was green while broken**. A replacement justified by
   tidiness, carrying that risk profile, against a preservation design costing only field-naming
   discipline, loses.

   (a) The migration is **declared-path-set preserving**. Acceptance gate for the migration
   commit: `cargo test -p pc-testkit --test recorded_provenance` green **with the test file
   unmodified**. If it is not green the migration is reworked, never the test.

   (b) The key-name walker (`recorded_provenance.rs:90-151`) is **neither deleted nor patched**.
   Under the canonical schema it becomes a second, redundant verifier of the same digests.
   Redundancy in a digest gate is acceptable; deleting five ratified iterations to remove it is
   not.

   (c) Digest slots are named so the walker's sibling resolution (`:90-96`) succeeds:
   `output_sha256`/`output`, `source_sha256`/`source`, `input_page_sha256`/`input_page` — subject
   to item 2's single-declaration constraint.

   (d) `sha2` is **not** promoted to `pc-testkit`'s `[dependencies]`; the shared checker stays
   pure structure over already-parsed JSON. Digest verification continues to live in the frozen
   test (its own `sha2` dev-dep) and in `xtask` via `pc_models::sha256_hex`.

   (e) The architect's holes **H1 and H2 are real** and are closed **additively** (rule 8 exit 1)
   in a new test file under `xtask/tests/` — legal because integration tests of a bin-only
   package link its `[dependencies]`, and `xtask` already carries `pc-models` and `pc-testkit`.
   It must assert: (H2) the `model_signature` group's `source_sha256` **equals**
   `pc_models::COMIC_TEXT_DETECTOR.sha256` — today checked only inside the writer
   (`xtask/src/model_signature.rs:212-218`), cookbook rule 12 exactly; and (H1) each
   `nlm`/`inter_area` source under `tests/fixtures/upstream/` hashes to a literal recorded at
   migration time. Upstream-path digests must **not** be added to the provenance files
   themselves — `assert_known_non_committed_form` (`:184-208`) rejects that form, and loosening
   the exemption is rule 13's "bypass wearing different clothes".

   (f) When the detector group records, `EXPECTED_GROUPS` / `EXPECTED_DECLARED_PATHS` /
   `EXPECTED_COMMITTED_PATHS` (`:11-27`) gain the new entries. This is an **authorised additive
   edit**, invited by the test's own message (`:85`), on two conditions ratified here: every
   existing entry is retained verbatim, and the constants stay **literal** — **deriving them from
   the filesystem is forbidden** (rule 13's iteration-2 collapse). It lands in the same atomic
   commit as the fixtures (item 6).

2. **ERRATUM against BOTH plans: each schema as drafted breaks the frozen gate at recording
   time.** Neither traced its detector pins through `recorded_provenance.rs:297-311`, which
   rejects **any** duplicate declared path across the whole tree, unconditionally — no constant
   edit can satisfy it.

   (a) *Engineer's schema:* `DetectorProvenance` is per-record and required for every record in
   the group. With ~6 records each carrying `model`/`model_sha256` and
   `input_page`/`input_page_sha256`, the walker collects the same two paths ~6 times → duplicate
   rejection fires. Worse, `comictextdetector.pt.onnx` is **already** declared by the
   `model_signature` group, so even one detector-group declaration of the bare filename is a
   cross-group duplicate.

   (b) *Architect's schema:* `ModelRef { sha256, .. }` and `PageRef { sha256, .. }` carry a bare
   `sha256` key whose sibling resolution is hard-coded to `output` (`:92`). No `output` sibling
   exists in those objects, so the walker **panics** (`:112-119`). Consistent with deleting the
   walker; fatal under item 1.

   **The constraint:** the canonical schema must be shaped so the frozen walker sees each
   filesystem path declared **exactly once per tree**. Binding realisation: detector pins are
   hoisted to **one group-level `detector` block per `PROVENANCE.json`** (the architect's
   group-level shape, with `upstream` and `ours` sub-blocks), not per-record; the input page is
   declared exactly once, as `input_page`/`input_page_sha256`, and is **not** additionally listed
   as a record output; and the model digest is stored under a key the walker does not collect
   (e.g. `model_digest`, sibling `model` holding the bare filename), because the bare filename is
   already declared by `model_signature` and because walker visibility buys nothing for it — the
   bare-model form is exactly the deliberately-unverifiable bucket, which is hole H2. **The
   compensating check is stronger than the walker's**: item 1(e)'s always-running test must
   assert the detector group's `model_digest` equals `pc_models::COMIC_TEXT_DETECTOR.sha256`. A
   digest slot dodging the walker is acceptable **only** because that test verifies it against
   the single source of truth; that condition is part of this ratification, not an implementation
   detail.

3. **Schema content amendments, binding.** The engineer's `DetectorProvenance` is missing
   spec-mandated fields; the architect's content list is adopted where it is the superset.

   (a) **Upstream side** pins: upstream version; commit (40-hex); **the invoking command line of
   the upstream run itself** (§16.20 item 3(a) — the engineer's group-level `command_line` can
   only hold the xtask invocation); the profile with every non-default key, built by diffing
   against defaults rather than by hand; backend (`cv2_dnn`, enforced by the engineer's
   backend↔implementation biconditional, adopted); and **resolved dependency versions** (cookbook
   rule 3; §16.20 item 7 makes them the only means of attributing a later divergence).

   (b) **Our side** pins: execution provider and both thread counts — **spec-mandated** by
   §16.22 item 6 and §16.21 item 5, taken from the *resolved* config, not the profile default —
   plus `pad_value` and the panel-ocr commit, which §16.20 items 8 and 5 make fixture-affecting.

   (c) The engineer's validator rule list is adopted, re-based onto the hoisted shape, including
   the digest-key and digest-shaped-value sweeps and the uppercase-digest rejection.

4. **DECIDED — pairing is a SIGNED INPUT, not an algorithm; the architect's position wins.** The
   engineer's objection that item 10's "exactly those three divergences" is unimplementable is
   sound **only against pairing formed by the identity itself**. Under a signed pairing table a
   1-px perturbation still pairs — the table says so — and yields exactly one geometry
   divergence, so both designs satisfy item 10 and the stated ground for a computed rule
   evaporates. What remains is a lopsided cost comparison.

   (a) `PAIRING_CENTRE_L1_LIMIT = 8` is an **unmeasured magnitude threshold on geometry**.
   §16.20 item 5's entire argument is that no defensible magnitude exists in this territory, and
   nothing in either plan derives 8 from a measurement. Even scoped to correspondence it has
   gating side effects — whether a fault surfaces as one `GeometryIdentity` row or as
   `UnmatchedUpstream` + `UnmatchedOurs` flips at 8 px — and mutual-nearest can mis-pair in
   geometry the ambiguity guard does not cover. That is a threshold re-entering through the
   correspondence door.

   (b) The cost of the signed input is hand-authoring N ≲ 12 pairings for one or two pages, and
   that authorship is work item 3(f) **already requires**: "pair every upstream block with one of
   ours or with a documented mechanism" is judgment the three signatures exist to cover, and
   3(f) says the producing agent "may … author the table". The spec anticipated an authored table.

   (c) The engineer's ambiguity scenario is handled *better* by the signed design: the author
   pairs one and documents the other; the identity then fails or an `EXPLAINED-§14.x` / `OPEN` row
   results, and `OPEN` blocks the commit — escalation instead of a tie-break heuristic.

   Consequences: the comparator **verifies** a supplied pairing and never invents one. The
   pairing, the per-pair `IdentityBranch` (line-informed vs line-less, hard-coded per pair — the
   architect's trap-closure is adopted), the unmatched entries with their mechanisms, and the
   expected totals form one `Expectations` value, a required by-value argument with no discovery
   overload. **Item 3(c) is ratified as scoped to gating verdicts** — the engineer asked for this
   and receives it — but the scoping is **not** license for a correspondence bound:
   `PAIRING_CENTRE_L1_LIMIT` is rejected and no distance constant may appear in the comparator.
   Every `Unmatched` entry carries a mechanism (§14.13 per-class duplicate / coverage-filtered /
   documented split-merge citing its register entry / `Open`), and `Open` is a blocking violation,
   never a verdict.

5. **DECIDED: the `Expectations` live in the TEST SOURCE, not a committed manifest.** The
   engineer's `MANIFEST.json` is rejected. A count or pairing read from a file committed beside
   the artifacts under comparison is derived data one hop from the gate — cookbook rule 12
   verbatim — and its stated benefit (re-recording without a frozen-test edit) **is** the defect:
   §16.23 item 5 deliberately budgets the DBNet re-record as "an F1 re-record and full re-sign",
   a reviewed event, not a file swap that keeps CI green. Expectations are consts in
   `crates/pc-detect/tests/f1_oracle*.rs`, with the derivation written out in
   `docs/DETECTOR_ORACLE.md` and covered by the signatures.

6. **The comparator's observable contract — a merge, with the seam stated.** Adopted from the
   engineer: the divergence **class** system (`Gating` / `Diagnostic` / `NoOracle`) reconciling
   item 3(d) with item 10; deterministic emission order; **no computed floats in any variant**, so
   controls assert the whole ordered vector with one exact `assert_eq!`; per-fault isolation
   tests; the extra-block-on-*our*-side control (rule 13's missing direction — item 10 names only
   the upstream-side deletion); the line-union positive case built on item 12's measured pair
   `(29,105,86,261)` vs `[29,105,90,261]`; the empty-oracle accounting-failure control; and the
   documented-difference-does-not-silence-the-row control. Adopted from the architect: the signed
   `Expectations`; expectations-in-source; the `IdentityBranch` returned and asserted per pair;
   the `rect_to_xyxy` conversion isolated in one function with a hand-computed unit test (§2.1's
   mixed inclusive/exclusive convention is exactly the off-by-one a tolerance would swallow); and
   **no boolean success channel** — CI's gate is "gating-classified set empty AND pairs-compared
   equals the authored count", never an `is_ok()`.

   **The recording commit is atomic:** fixtures + provenance + item 1(f)'s constant edits + the
   real-page gate test with its literal `Expectations` + the doc verdicts + the three signatures
   land in one commit. Until then no real-page gate test exists. The engineer's WARN-degrading
   committed-gate test and two-state invariant are **not adopted**: the frozen provenance gate
   already fails any half-recorded tree (a fourth group directory breaks `EXPECTED_GROUPS`; an
   undeclared file breaks bidirectional coverage), so the extra machinery guards nothing, and a
   conditional-return gate is the shape cookbook rule 1 exists to resist.

7. **DECIDED — item 10's "exactly those three divergences" is a multiset across classes:** one
   gating geometry row, one DIAGNOSTIC confidence row, one gating unmatched/accounting row, and
   nothing else. Both plans independently converged on this, so the architect's anticipated
   disagreement does not exist. A control expecting three *gating* failures would contradict items
   3(c)/(d) and item 5's 0.079–0.089 noise floor.

8. **DECIDED — the comparison's left-hand side is `_detector_blocks.json` (pre-filter); the
   upstream side is upstream's post-`group_output` output.** §16.20 item 2 names
   `_detector_blocks.json` "the artifact that genuinely accepts output as truth", and it is what
   `ReplayDetector` replays; our `#raw.json` blocks are post-coverage-filter (verified:
   `crates/pc-detect/src/lib.rs:106-118` filters on `mask_coverage` before assembling
   `PageDataRaw`) and are separately locked by §8.7(A)6/(B)9. Item 9's measured asymmetry —
   upstream's line-less filter firing once, on `[438,1407,498,1446]`, `mask_score` 0.0359 —
   becomes **visible instead of hidden**: the box appears in our pre-filter list and not in
   upstream's post-filter oracle, and closes as a documented unmatched-ours entry citing
   §14.17/item 9. That is item 9's own lesson, that a divergence table over final artifacts
   "would have shown a clean result and hidden the structural difference".

   Adjustment: the oracle script must **also** capture upstream's pre-`group_output` block list
   and per-block `mask_score`s as a DIAGNOSTIC (non-gated) artifact, so each documented unmatched
   entry points at recorded evidence rather than an inference. The upstream *gated* side stays
   post-`group_output`, because that is where `lines` exist and where item 3(b)'s identity is
   defined. Page-level `scale`/`image_size` rows are adopted as **gating** — cookbook rule 7 names
   both as genuine oracles — and they are what pins the two sides to one coordinate frame.

9. **DECIDED — the `confidence` partition row closes as DIAGNOSTIC if the recording script
   captures upstream's scores, else NO-ORACLE; it may not close as `OPEN` on this account.**
   Cookbook rule 7 records that upstream's `#raw.json` does not persist confidence. The script
   must *attempt* capture from `postprocess_yolo`'s output before `group_output` discards it —
   item 5's own decomposition proves the value is obtainable — and the oracle schema keeps
   `confidence: Option<f64>` with `NoOracleField` rows when absent. Either outcome is a closed
   verdict under item 3(e); neither blocks the fixture. The row is never gated regardless.

10. **ERRATUM — §16.20 item 3(e)'s citation "§2.4/§2.5's field list" is wrong: §2.5 is
    `PageData`, the preprocessor's output, not the detector's.** The partition F1 owes is over
    §2.4's `PageDataRaw` + `DetectedBlock`. Ruled: §2.4's fields are partitioned exhaustively;
    §2.5's fields appear as `OUT-OF-SCOPE-F1` rows pointing at §9.7(B)11, recorded rather than
    silently dropped (§16.23 item 1's rule); **and, in consequence of item 8, `RawBlock`'s three
    serialized fields (`rect`, `class_index`, `confidence`) get partition rows too** — the gated
    artifact may not have unpartitioned fields, or the honour-system gap re-opens one artifact
    over.

11. **RATIFIED — the completeness partition, the EXPLAINED-anchor rule, the OPEN-blocks rule and
    the three-signature requirement are TESTS, not checklist prose.** Briefed as an architect-only
    position; in fact both plans drafted the same four gates, so it is ratified as agreed: set
    equality between the doc table's field column and the serialized key set, **both directions**,
    duplicates rejected; every `EXPLAINED-§x.y` anchor present in the spec; no `OPEN` row once the
    fixture is present; three complete, role-distinct, non-self signatures. Also ratified: the
    **two-equation** accounting reading, since the literal "equals both totals" is unsatisfiable
    with one-sided extras, as both plans independently found. **The terms are RESTATED by item 20(c)'s
    rename** — the original spelling was
    `pairs + class_duplicates + documented_upstream_only == upstream_total` and
    `pairs + documented_ours_only == ours_total`, kept here only so the mapping is checkable:

    ```
    pairs + class_duplicates + documented_split_merge_upstream + open_upstream == upstream_total
    pairs + coverage_filtered_ours + documented_split_merge_ours + open_ours   == ours_total
    ```

    Old→new mapping, recorded **once** so no future reader has to resolve it semantically:
    `documented_upstream_only ≡ documented_split_merge_upstream + open_upstream`;
    `documented_ours_only ≡ coverage_filtered_ours + documented_split_merge_ours + open_ours`. The
    sums are term-for-term identical to item 20(a)'s partition reading, so its one-gating-row
    conclusion is untouched.

12. **DECIDED — `demo_bubbles` detector artifacts: record to scratch for §10.7(B)15's non-gating
    report, commit nothing, and the four §16.13 item 8 tests stay `#[ignore]`d.** The architect
    wins; §16.20 item 12 is dispositive. Box presence on those crops is decided by rounding (0.037
    survival margin against a 0.079–0.089 noise floor), and item 11 names an unreviewed recorded
    fixture the top residual risk. Un-ignoring regression locks against a noise-determined box set
    is cookbook rule 7 with extra steps. Those four tests un-ignore only against the signed
    maintainer page, never against `demo_bubbles`.

13. **CONFIRMED — the architect's H3 is real, and it is assigned.** `pc-testkit` has
    `rebase_page_data_raw`/`rebase_mask_data` and **no inverse**, while `pc_detect::run` builds
    handles from the absolute `base_image_dest`/`raw_mask_dest` and §7.2 requires relative storage.
    Without an inverse the recorder commits one machine's absolute paths — the defect §16.20 item
    3's closing paragraph names. Assigned to the schema task:
    `relativize_page_data_raw`/`relativize_mask_data` as exact inverses in `pc_testkit::paths`,
    with a round-trip unit test, called by the recorder before serialising `#raw.json`, and
    mirrored in the oracle script's path stripping.

14. **Recorder and oracle-script requirements, merged.** The engineer's testable split (pure
    `plan()` + `provenance_for()` with in-module `#[cfg(test)]` tests) is adopted, **plus** the
    architect's execution-path order, all binding: verify `pc_models::COMIC_TEXT_DETECTOR.sha256`
    **before any inference**; refuse a non-CPU execution provider with a `// §16.22 item 5(b)`
    guard; call `detect()` directly and write the §7.2.1 pair via the shared
    `pc_detect::mock::write_replay_fixture` (whose doc comment already names F1); run the full
    `run()` for the §7.2 triple and **assert the two block lists identical** rather than assuming
    determinism; relativize (item 13) before serialising; populate provenance from the resolved
    config.

    **The engineer's plan omits the upstream-oracle recorder entirely — a real scope gap.** The
    architect's T5b is adopted whole and is in F1's scope: a committed
    `xtask/scripts/record_detector_oracle.py` (§11.6's precedent that recording scripts are
    committed so provenance is auditable), pinned to upstream
    `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`, forcing **and verifying** the `cv2.dnn` backend
    (§16.20 item 6 — the `.pt` path is a different code path with BGR input), verifying the model
    digest before emitting, stripping absolute paths, and refusing to emit otherwise. Excluding
    3(a) would leave F1 delivering a comparator with one operand.

15. **Cleanups, all in F1 scope.** (H4) `DemoBubble::recorded` has zero call sites and builds
    group-less paths the bidirectional gate would reject — **delete it**; a group-aware accessor
    can return when a caller exists. (H5) the three explanation constants in `xtask/src/env.rs`
    promise flat `tests/fixtures/recorded/<stem>_…` paths with no `detector/` component —
    corrected when the recorder lands; no test pins the full paths, so this is ordinary work, and
    the engineer's substring-only skip-message test is adopted. (H6) `xtask/src/calibrate.rs`'s
    stale "blocked on D1/D4" text — corrected to the real blocker, the page plus the recording
    run. Path-root unification: the four duplicated roots in `xtask/src/paths.rs` become
    re-exports of `pc_testkit::paths` (§16.18 item 1; F1 adds a third consumer of "where is the
    recorded root"), xtask-only helpers staying put. The hand-rolled 75-line SHA-256 in
    `xtask/src/model_signature.rs` is **deleted** — `sha2` is already in `xtask`'s build graph via
    `pc-models`, so the copy buys no dependency reduction and is a second implementation of a
    cryptographic primitive; add `sha2` to **`xtask`'s** `[dependencies]`, not `pc-testkit`'s.
    Migration safeguards: every digest recomputed from bytes and asserted equal to the old file's
    digest before the new file is written; re-running `nlm`/`inter_area` with `--force` for
    byte-identity is verification when tooling is present, not a migration precondition; **no
    digest is ever hand-typed.** Cookbook rule 6's test-count bar: the two plans cite different
    baselines (735/746 vs 738/749); neither is adjudicated here — re-measure at commit time and
    record the measured numbers.

16. **Page-format engineering ruling.** The binding invariant: **both detectors must consume
    byte-identical pixel buffers.** Item 5 measured a 1-LSB input difference moving confidence by
    0.079–0.089, and a lossy page decoded by `image` on our side and `cv2` on upstream's
    reintroduces — one level up, feeding the network directly — the decoder confound §16.13 item 5
    already isolated for INTER_AREA. Two conforming mechanisms, either acceptable:

    (a) commit a **lossless PNG**, so decoding is content-deterministic and the file sha256
    suffices. This collides with §7.2's 400 KB cap and needs the maintainer's cap amendment, since
    a lossless ~1024-wide page plausibly exceeds it; or

    (b) commit the JPEG within the cap and feed upstream the `image`-crate re-decode written as a
    scratch PNG — the exact §16.13 item 5 precedent, cap intact. Under (b) the provenance must
    record the digest of the **decoded RGB buffer** fed to each side in addition to the file
    digest, and the two must be equal.

    Item 12's criterion stands either way: letterbox ratio near 1, so `max(w,h) ≈ 1024`.

18. **ERRATUM to §16.20 item 12's geometry sentence — the identity is two legs, and one of the two
    cited pairs refutes the leg it is cited to support (joint architects, 2026-07-29).** Found by
    the Senior Rust Engineer while drafting the comparator's controls; verified independently by
    the Orchestrator against `:3461` and by the Technical Architect against the code. Both
    architects agree, so this is a joint ratification and needed no tie-break.

    (a) **The structural fact.** `Rect::merge` (`crates/pc-core/src/geometry.rs:60-67`) is
    `x1.min, y1.min, x2.max, y2.max`, so a bounding union is monotone non-increasing in `x1,y1`
    and non-decreasing in `x2,y2` **for any second operand**. Item 3(b)'s identity therefore
    entails, per pair and independently of what upstream's lines are:
    `upstream.x1 <= ours.x1`, `upstream.y1 <= ours.y1`, `upstream.x2 >= ours.x2`,
    `upstream.y2 >= ours.y2`. Pair 2 (ours `(29,105,86,261)`, upstream `[29,105,90,261]`)
    satisfies all four — three edges exact, `x2` widened +4 by the union. Pair 1 (ours
    `(40,23,111,207)`, upstream `[40,23,110,207]`) satisfies three and **violates
    `upstream.x2 >= ours.x2` by one**, in the direction a union cannot produce.

    (b) **The leg decomposition, which is why the failure is attributable at all.**
    `upstream.xyxy == bbox(ours.rect ∪ bbox(upstream.lines))` silently composes two claims of
    different kinds. **Leg 2 (structural):**
    `upstream.xyxy == bbox(upstream.rect_yolo ∪ bbox(upstream.lines))` — a law about upstream's
    own `group_output`. **Leg 1 (empirical):** `upstream.rect_yolo == ours.rect` — a
    *measurement*, engine-, input- and scale-dependent. Pair 1 violates **leg 1**. The identity's
    algebra is not in question; leg 1's precondition is. Leg 1 is validated at letterbox ratio ≈ 1
    (10/10 blocks, 40/40 coordinates on the 12-box page) and **never validated at r ≈ 3.0–3.8**.

    (c) **Blast radius, stated so this is not over-corrected.** One leg, one edge, one pair, on an
    input class item 12 itself excludes as the oracle fixture. The record's other geometry claims
    are internally consistent and untouched, and item 3(b) is not undermined.

    (d) **Three candidate causes, recorded OPEN with the discriminator named.** (1) **Transcription
    swap** — if the real numbers are ours `110` / upstream `111`, both pairs become one mechanism
    at two magnitudes (+1 and +4), all eight inequalities hold, and the sentence's own attribution
    becomes true of both. Parsimonious, and a single-character slip in a sentence whose other
    example has the identical shape with opposite sign. (2) **Truncation straddle** —
    `yolo.rs:145-171` does `(x * ratio) as i32`, which truncates, matching upstream's
    `astype(np.int32)`; `clamp` is not implicated (111 ≪ 219 for `nightmare`). With
    `r = 2.985` and `ratio_x = 219/654 = 0.3349`, base-space values of `111.02` and `110.98`
    truncate to 111 and 110 — a whole-pixel divergence from **0.04 px** of disagreement. At
    r ≈ 3.0–3.8 roughly nine in ten network-input pixels are interpolated, so item 5's bilinear
    weight difference applies to nearly the whole input rather than a minority of it. (3) **A
    systematic per-edge offset** — a real port defect in rescale, letterbox geometry, or the
    §2.1 inclusive/exclusive convention. **This erratum does not assert any of the three**;
    asserting the straddle would be closing a row against an unmeasured mechanism, which is item
    3(e)'s error in a different costume.

    (e) **The discriminator, and the run is commissioned.** Re-run both sides over all seven
    committed `tests/fixtures/upstream/demo_bubbles/*_raw.png` (GPL-3, licence-clean) plus the
    model — item 12 records 21 upstream blocks, so ~84 coordinates, enough for a distribution
    rather than two points. Needs **no page decision**, so §16.24 item 17 does not gate it. Upstream
    pinned at `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3` through `cv2.dnn.readNetFromONNX`
    (item 6). Capture per block per side: network-space `letterbox_xyxy`; `ratio_x/ratio_y/dw/dh/r`;
    **`base_xyxy_pretruncation` — the actual discriminator, requiring a local probe in `rescale`
    and upstream's pre-`astype` array**; the final `rect`; upstream `xyxy`, full `lines`,
    `len(lines)`, `mask_score`; `residual_leg1 = upstream.xyxy − ours.rect`; and
    `residual_full = upstream.xyxy − bbox(ours ∪ bbox(lines))`, which must be all-zero.

    (f) **Conditional gate on F1's recording.** Branch (1) or (2): recording proceeds unaffected,
    because item 12's own criterion already excludes the r ≈ 3 regime. **Branch (3): the fix and
    its ratification land BEFORE the recording run**, on §16.20 item 8's precedent that the
    `PAD_VALUE` erratum had to land first since the pad colour perturbs every recorded box.
    Discovering a systematic offset after the page is recorded and signatures collected means
    re-recording and re-signing, which §16.23 item 5 budgets once and only for DBNet. **The run is
    therefore sequenced before the recording run, not after.**

    (g) **Item 12's conclusion is preserved and STRENGTHENED, not reopened.** The ruling —
    demo_bubbles cannot serve as the oracle fixture, and the page must sit near letterbox scale 1 —
    rests on the **box-presence** measurement (0.037 survival margin against a 0.079–0.089 noise
    floor), which this defect does not touch. The erratum supplies a *second, independent* reason
    for the same conclusion: **letterbox ratio ≈ 1 is the precondition of leg 1**, so the criterion
    protects box geometry as well as box presence. The geometry sentence was the wrong evidence
    precisely because it measured leg 1 in the one regime where leg 1 is not expected to hold.

    (h) **The residual report is ratified, DIAGNOSTIC-only, with four non-optional conditions.**
    This is item 11's producing-vs-comparing split one level down: *measuring* a residual is a
    recording-time act; *accepting* one is a ratification. (i) The gating identity stays exact and
    the report cannot reach it — `Expectations` gains **no** residual field and no epsilon, so
    there is no parameter through which a residual could influence a verdict. (ii) **`residual_full` must
    be zero** for green — see the amendment below — and a non-zero one is a finding, not an absorbed
    quantity; the only sanctioned exits are: fix our arithmetic, ratify a named per-pair mechanism as
    `Explained { entry, per_edge }` with an **exact** per-edge constant (the ratification landing
    first, per item 3(e)), or select a different page.

    **AMENDED (2026-07-29), and the original wording was wrong:** this clause first read
    *"residuals must be zero for green"*, over both forms named in (iii). That is unsatisfiable.
    `residual_leg1 = upstream.xyxy − ours.rect` is **non-zero in the normal, correct case** — it is
    `[0,0,+4,0]` on item 12's good pair, because the union widened `x2` by exactly the 4 px the
    identity predicts. Gating on it would fail every line-informed page, and item 3(c) leaves no
    epsilon available to express the intent. Only **`residual_full`** — the composite residual, which
    the identity requires to be all-zero — is a gating quantity. `residual_leg1` is **diagnostic**:
    its four signs are what attribute a failure to leg 1 versus leg 2, which is the entire reason
    (iii) requires both forms. Caught by the Rust Engineer while drafting against this clause; it was
    a transcription defect in this section, not in the design it records. There is no fourth exit, and an epsilon, an
    inequality, an IoU or a per-edge band are all refused — item 5's `dw = 284` arithmetic shows a
    ±3 px band absorbs a one-pixel letterbox-padding error, the §14.14 defect class, and item 12's
    own data would have been *invisible* under any band. (iii) Report **both** residual forms plus
    the pre-truncation floats and `len(lines)`, because the sign pattern is what attributes a
    failure to leg 1 versus leg 2 — a composite-only report would have left this contradiction as
    undiagnosable as the record did. (iv) **More than one `Explained` pair on a single page
    escalates** to fixing the mechanism; two independent one-off mechanisms on one page is not a
    coincidence, and without this cap the instrument decays into a tolerance spelled as a list.

    (i) **Additive strengthening, no ratification needed (cookbook rule 8 exit 1):** `compare()`
    asserts leg 2's four monotonicity inequalities *in addition to* the equality. Equality already
    implies them, so this adds no power — it adds a **message**: `upstream.x2 < ours.x2` should
    report that a bounding union cannot narrow an edge and this is therefore a leg-1
    engine-geometry divergence, not a line-union difference. Diagnostics are part of the gate
    (cookbook rule 13). **Had this existed as prose in item 3(b), the contradiction would have been
    caught when item 12 was written.**

    (j) **Classification note, so a later row cites the right register.** If branch (2) holds the
    mechanism is *shared* with upstream — both sides truncate — so it is **not** a §14 deviation.
    §14 records deliberate divergences; this would be a §16.x finding about shared engine
    sensitivity. Item 3(e) permits a row to close as `EXPLAINED-§14.x`, so pointing at §14 for a
    non-divergence would put a false deviation in the register.

    (k) **Self-correction carried forward (cookbook rule 10).** The architect's F1 plan called the
    line-less degenerate branch "the common case". That is wrong: it conflated *we* never
    synthesizing lines — always true, and irrelevant to which branch fires — with *upstream*
    finding none, which item 9 records firing once on the candidate page. The branch is
    **rare-but-real**, which is what §16.20 item 3's "not hypothetical" meant. The consequence is
    uncomfortable and worth stating: the degenerate branch is the **least**-evidenced part of the
    identity, so (e)'s per-block `len(lines)` census is the first real evidence on it.

19. **ERRATUM to item 1(a) — the migration's acceptance gate named ONE consumer, and there are
    two. The frozen `model_signature.rs` broke (joint architects, 2026-07-29).** Caught by the
    Codex stop-time review, not by the planning pass and not by the Orchestrator's pre-implementation
    test review.

    (a) **What broke.** Item 1(a) made *"`cargo test -p pc-testkit --test recorded_provenance` green
    with the test file unmodified"* **the** acceptance gate, phrasing the preservation requirement as
    if `recorded_provenance.rs` were the only reader.
    `crates/pc-testkit/tests/model_signature.rs::the_committed_provenance_describes_the_committed_signature_byte_for_byte`
    is a second frozen reader, and it hand-indexed three keys at top level — `output`,
    `output_sha256`, `source_sha256` — which the canonical schema moved under `records[]`. It failed
    at `:104` with *"PROVENANCE.json records `output` as a string"*. Item 1(a)'s gate passed
    throughout: the declared-path set was preserved exactly, so the gate was correct and merely
    aimed at one of two targets.

    (b) **The defect is the single-consumer framing, and the general rule follows from it.** **A
    ratified change to a shared on-disk format must enumerate every reader of that format, record
    the enumeration, and state the acceptance gate as a set covering all of them.** The enumeration
    is recorded rather than re-derived, because re-deriving it is what nobody did. This is cookbook
    rule 13 applied one level up: we asked exhaustively what the gate enumerates about the *files*
    and never asked what enumerates the *consumers*.

    (c) **The enumeration, recorded (2026-07-29).** Readers of `PROVENANCE.json`:
    `crates/pc-testkit/tests/provenance_schema.rs` and `xtask/tests/provenance_digests.rs` (typed
    schema); `crates/pc-testkit/tests/recorded_provenance.rs` (generic key walker);
    `crates/pc-testkit/tests/model_signature.rs` (**was** hand-indexed, now typed);
    `xtask/src/model_signature.rs` and `xtask/src/record.rs` (writers, now writing through
    `pc_testkit::provenance`). Any future reader added to this list carries item 1(a)'s gate with it.

    (d) **The adaptation is authorised, in the same category as item 1(f).** The test's assertions
    were never in question — only the JSON path it read them from, which changed by ratified
    decision. Nothing is weakened. It now parses through
    `pc_testkit::provenance::GroupProvenance` and looks its record up **by name**, never
    `records[0]`, since positional indexing is what let it drift. Preserved verbatim: all three
    failure messages, including the explanation that recording performs two non-atomic renames so an
    inconsistent pair means either a hand-edit or a half-committed pair.

    (e) **An adaptation must be justified by a condition-by-condition table, not by the claim that
    nothing was weakened.** *"Nothing is weakened"* is precisely the assertion cookbook rule 1 says
    to distrust when it appears without evidence. For this adaptation: **7 checked conditions became
    9.** The three `.as_str().expect(...)` premise checks are *absorbed* by the typed parse, which is
    strictly stronger — it type-checks every field rather than three. Three conditions are genuinely
    new: `record.committed` must be true (*"the **committed** signature"* is meaningless otherwise),
    `record.source` must name the model file (otherwise `source_sha256` could describe anything), and
    the typed parse itself is a named failure carrying the path. The record-*set* assertion went into
    a separate sibling test, `the_model_signature_group_declares_exactly_one_record`, so a set drift
    fails under a precisely-named test rather than inside a byte-for-byte consistency test; the
    consistency test still looks its record up independently, because execution order is not
    guaranteed and a premise must not rest on another test having run.

    (f) **The test that did NOT break is the more useful lesson.** `model_signature.rs`'s other
    reader, `the_signature_group_records_its_provenance`, passed the migration untouched — because it
    asserts only that the file is non-empty valid JSON and `is_object()`. **Had it been the only
    test, the migration would have looked clean.** That is cookbook rule 1's dominant defect class
    caught in the act: a test whose name claims it checks that the group "records its provenance",
    whose assertion checks almost nothing. It is deliberately left as-is — strengthening it would
    duplicate `every_recorded_group_parses_and_validates` with strictly less reach, unlike the digest
    redundancy of item 1(b), which buys genuinely different coverage — but it is recorded here as a
    known weak gate rather than counted as coverage.

20. **The accounting terms PARTITION the unmatched set on each side; they do not count only
    *documented* mechanisms.** Ratified here rather than left to implementation because it decides a
    **frozen test's** expected vector, so the choice must land before the test is written.

    Every unmatched entry on a side falls under exactly one term, and after (c)'s rename each term
    names exactly one mechanism class: upstream, `class_duplicates` / `documented_split_merge_upstream`
    / `open_upstream`; ours, `coverage_filtered_ours` / `documented_split_merge_ours` / `open_ours`.
    Item 11's two equations therefore close whenever total accounting holds, and a mechanism's
    *adequacy* is adjudicated separately, by the row it raises — **not** by whether it is counted.

    The reading this displaces, stated so the ratification is legible: the terms do **not** count only
    genuinely *documented* mechanisms, leaving `Open` outside the sums. Under the pre-rename spelling
    that distinction was invisible, which is why (c) renames.

    (a) **Consequence, and the reason this is ratified:** an `Open` entry raises exactly **one**
    gating row (`OpenMechanism`), not two (`OpenMechanism` + `AccountingMismatch`). That is what makes
    §16.20 item 10's *"exactly those three divergences"* satisfiable, since item 10's third slot is one
    gating *unmatched-or-accounting* row. Under the alternative reading — `documented_*` counting only
    documented mechanisms — a single deleted box produces two gating rows and item 10 is unsatisfiable
    as written. The reading is therefore **forced by elimination**, not preferred: one of the two
    readings contradicts a ratified requirement.

    (b) **The anti-bypass property must be STRUCTURAL, not conventional — and this clause was
    defective as first written.** It said "the comparator must emit the unmatched row regardless of
    the declared totals", which is an instruction to an implementer, not a property of the data, and
    it cited two tests — `the_three_injected_faults_are_reported_exactly` and
    `declaring_a_documented_difference_does_not_silence_the_row` — **as though they pinned the
    reading when neither exists in the tree.** That is §16.20 item 3(e)'s own prohibition (a row may
    not close against something that does not yet exist) committed in this section, and a normative
    clause resting on absent evidence is worse than one resting on none. Both are corrected here:
    the tests below are stated as **obligations on F1-C**, and the invariant is stated as a
    requirement on the type rather than on the implementer's diligence.

    **Binding: the MECHANISM-PARTITION counts are derived; the ANTI-VACUITY literals are not.** The
    distinction is load-bearing and an earlier draft of this clause got it wrong by saying
    "`Expectations` carries NO count fields", which **contradicts §16.20 item 3(b)** — that clause
    *requires* the comparator to "assert the number of pairs it compared against a hard-coded expected
    count", because "a renamed field silently yields an empty pairing rather than an error". Deleting
    the literal would have removed the only guard against "compared 0 boxes, 0 divergences → PASS".
    Corrected here; the two guards defend **different** bypasses and both are required:

    - **Derived, from the signed entry lists, never authored:** `class_duplicates`,
      `documented_split_merge_*`, `open_*`, `coverage_filtered_ours`. These guard against
      **inflation**: derived, closing the arithmetic requires **adding an entry**, and an entry either
      cites a checkable register anchor or is `Open` and raises its gating row. Grounds, per the Fable
      ruling: cookbook rule 13's recorded preference — *"prefer asserting the expected set; the count
      comes free"* — where the signed entry lists **are** the set; and **unrepresentability beats
      detection**, since a declared-count design admits count-vs-entry inconsistency and then detects
      it by convention, while the derived design makes it inexpressible.
    - **Hard-coded literals, authored and signed:** `pairs`, `upstream_total`, `ours_total`. These
      guard against **vacuity** — the empty-pairing failure item 3(b) names. They may **not** be
      derived from the artifacts, and this is not a stylistic preference: `upstream_total` derived from
      the upstream document would make a deleted box invisible to accounting, because the expectation
      would move with the thing it is meant to check. That is cookbook rule 13's collapse and rule 7's
      circle in one step.

    The deleted-box control resolves cleanly under this split, which is the check that the split is
    right: authored `pairs = 4`, `upstream_total = 5`, `ours_total = 4`; derived `open_upstream = 1`
    from the single `Open` entry, every other mechanism term zero. Both equations close —
    `4 + 0 + 0 + 1 = 5` upstream and `4 + 0 + 0 + 0 = 4` ours — and the fault yields exactly one gating
    row (`OpenMechanism`). Item 21(b) confirms this as the frozen vector.

    **Binding: declared-unmatched == computed-unmatched, both directions**, where computed is
    `artifact − paired` over the *declared* pairs. This is a set difference over a supplied pairing,
    i.e. verification, so item 4's "the comparator never invents a pairing" is respected. It closes
    the second door this clause originally missed entirely: a **phantom** entry for a box that is not
    actually unmatched.

    **CORRECTION OF RECORD — two supporting traces I transcribed here are FALSE and are withdrawn.**
    An earlier version of this clause rested the bindings on two arguments from the Technical
    Architect. The Fable ruling found both wrong, and the bindings stand on the replacement grounds
    above instead:

    - The **bypass trace** ("set `documented_upstream_only = 4` while listing three entries, closing
      the arithmetic with no fourth row") fails against the design actually drafted: the comparator
      cross-checks every declared term against the entry list and raises a gating `AccountingMismatch`
      on any inconsistency, so there was never a silent close. Its "a test nobody re-audits after
      signatures" framing also mislocated the check, which is comparator behaviour exercised by CI
      controls on every run.
    - The **"load-bearing derivation" trace** ("`pairs` falls to 11, declared `documented_ours_only`
      is 0, `ours_total` is 12 — two gating rows") assumes the truth-pair literals are kept against a
      *perturbed* artifact, i.e. an **un-re-authored table** — a reading item 4 already forecloses
      (*"a 1-px perturbation still pairs — the table says so"*). Authored for the artifacts under
      comparison, the control closes: `4 + 0 + 0 + 1 = 5` upstream, `4 + 0 + 0 + 0 = 4` ours, one
      gating row.

    So item 20 as first written was **correct on (a) and under-specified on (b)** — not wrong on (a),
    which the withdrawn trace claimed. Recorded rather than silently deleted because the correction
    chain is itself the evidence that these clauses were checked.

    **Required of F1-C** (obligations, not citations): a test asserting that the three injected faults
    are reported exactly, with no `AccountingMismatch` accompanying the `Open` entry; and a test
    asserting that declaring a documented difference does not silence the per-box row.

    (c) **Naming — RESOLVED: RENAME. The Technical Architect wins (Fable tie-break, 2026-07-29),
    refined with side suffixes so every term names exactly one mechanism class on exactly one side.**

    Ratified spellings, binding on the spec, the Rust identifiers, every `AccountingMismatch` and
    report field string, and `docs/DETECTOR_ORACLE.md` when it is written:

    | side | term | mechanism |
    |---|---|---|
    | upstream | `class_duplicates` (unsuffixed — the mechanism is upstream-side-only by ratified definition) | `ClassDuplicateOf` |
    | upstream | `documented_split_merge_upstream` | `DocumentedSplitMerge` |
    | upstream | `open_upstream` | `Open` |
    | ours | `coverage_filtered_ours` — unmatched-ours entries explained by *upstream's* line-less coverage filter | `CoverageFilteredUpstream` |
    | ours | `documented_split_merge_ours` | `DocumentedSplitMerge` |
    | ours | `open_ours` | `Open` |

    Item 11's equations are restated in the same edit, with the old→new mapping recorded there once.
    **Rename in spec, Rust type and doc together or in none of them** — the engineer's condition,
    adopted whole: a spec term differing from the identifier is cookbook rule 14's drift vector.

    **Why keep-and-pin lost.** Its continuity argument protected a *spelling* whose meaning item
    20(a) had already overridden — continuity of characters while the semantics invert is the opposite
    of continuity for a reader, and the mapping line it offered concedes that every future reader must
    perform a translation the spec could perform once, here. My own earlier 20(c) text was that
    position's best refutation: a name that "actively misleads", whose literal reading is "the more
    natural one from the field name alone", kept alive by a standing *"recorded so it is not fixed"*
    warning. Rule 1 makes name-vs-substance divergence this project's **dominant** defect class, and
    keep-and-pin converts it from a defect we hunt into an exception we maintain — at a site whose
    failure message a reader meets without §16.24 open. And the cost asymmetry is measured, not
    asserted: a grep found the identifiers in exactly two spec items and one scratchpad draft. **Zero
    code, fixture, provenance, signature or doc readers** — the rename's entire cost was this edit,
    and it stops being free the moment the fixture commits.

    The original text of this clause, recorded because the disagreement was real and the losing
    position was reasonable:
    `documented_upstream_only` counts entries that are not all documented, so the name actively
    misleads, and the *literal* reading is the more natural one from the field name alone. The two
    Opus architects disagree: the Senior Rust Engineer holds "keep item 11's name, pin the meaning
    here" for continuity with the ratified equations; the Technical Architect holds "rename now",
    arguing that renaming is currently **free** — no code, no committed fixture, no `PROVENANCE.json`,
    no signature and no `DETECTOR_ORACLE.md` row depends on the identifier — that it stops being free
    the moment the fixture commits, and that a name contradicting its meaning is this project's
    dominant defect class (cookbook rule 1) placed deliberately, at a site whose failure message a
    reader will meet without §16.24 open (rule 13's corollary: diagnostics are part of the gate). It
    also identifies a third reading that makes the equations truthful without a semantic mapping:
    `pairs + class_duplicates + documented_split_merge + open_upstream == upstream_total`,
    arithmetically identical and equally satisfiable under item 10.

21. **F1-C comparator rulings: the phantom door, and NO fault-collapse rule (Fable tie-break,
    2026-07-29).** Both settle the frozen test's expected vector, which is why they are ratified before
    the test is written rather than discovered during implementation.

    (a) **The phantom door is ADOPTED.** `Expectations`' declared-unmatched entries must equal the
    comparator's computed unmatched set, **both directions**, where computed is `artifact − paired`
    over the *declared* pairs. That is a set difference over a supplied pairing — verification, not
    invention — so item 4's "the comparator never invents a pairing" is respected. It closes a door
    item 20(b) missed entirely: a **phantom** entry declaring a box unmatched when it is in fact
    paired. New gating variant `PhantomUnmatched { side, index }`, with one containment-style control.

    (b) **The fault-collapse rule is REJECTED, and the frozen vector is THREE rows.** The premise —
    that a deleted box perturbs `pairs`, `upstream_total` and `ours_total` and so forces a fourth row —
    **dissolves under re-authoring.** Item 10's control authors its table for the *perturbed* artifacts
    (item 4): the pairing list has 4 entries, authored `pairs = 4`, all 4 resolve, authored
    `ours_total = 4` and `upstream_total = 5` match the artifacts, and derived `open_upstream = 1`
    closes both equations. **Nothing authored is perturbed, because the author authored for what is
    there.** The deleted box surfaces exactly once, as `OpenMechanism { side: Upstream, index: 4 }`.

    The frozen three-fault vector, confirmed: `GeometryIdentity { ours: (500,600,617,672), expected:
    (500,600,617,672), upstream: (500,600,618,672) }` (gating) → `ConfidenceDelta { ours:
    (700,800,760,830), 0.437, 0.438 }` (diagnostic) → `OpenMechanism { side: Upstream, index: 4 }`
    (gating); `pairs_compared == 4`; gating 2, diagnostic 1.

    Answering the either/or directly: the authored `pairs` counts the pairs the author declared **for
    the artifacts under comparison**, and since the comparator separately verifies that every declared
    pair resolves, declared and resolved coincide on every input an exact vector is promised for.
    Neither "the deletion suppresses the pairs-count check" nor "the literal silently tracks
    declared-only" is the ruling — an unresolved declared pair fails the pairs check *and* raises
    `PairingIndexOutOfRange`.

    (c) **Where the three literals genuinely ARE perturbed — a stale table, i.e. artifact drift under
    signed `Expectations` — multiple true rows are the CORRECT output.** Each names a real,
    independently-stated inconsistency, which is rule 13's corollary working rather than
    double-reporting: a re-record that drifts is exactly the event §16.24 item 5 makes "a reviewed
    event, not a file swap that keeps CI green", and the reviewer wants all three signals.

    (d) **Why suppression is refused outright, and the bound is ZERO.** A rule suppressing "the derived
    checks a defect necessarily perturbs" requires the comparator to decide which of its own findings
    caused which others; a bug in that causal judgment **silently hides real rows**, and no frozen
    assertion anywhere needs the machinery. The instinct to bound such a rule was right; the correct
    bound turned out to be zero. Both this finding and the sharp form of item 20(b)'s withdrawn trace
    fail from the same root — the un-re-authored-table reading of the deletion control, which item 4
    forecloses.

    (e) **Ratified instead — the assertion discipline**, which is what actually protects the first
    write: exact whole-vector assertions only over structurally-sound tables, and structural-defect
    controls assert **containment plus gating-non-empty, never sibling-absence**.

    (f) **Sign precision on item 18(h)(iii), because rule 15 is literally about signs.** The
    one-directional attribution is **per edge**: the union-impossible directions are
    `residual_leg1.x1 > 0`, `y1 > 0`, `x2 < 0`, `y2 < 0` — each of those *proves* leg 1, while the
    complementary directions prove nothing, being consistent with either leg. **"A negative component
    proves leg 1" is true only of `x2`/`y2` and would be a sign error on `x1`/`y1` if transcribed
    bare** — which is exactly how it read in my first draft.

    (g) **Deferral, recorded per §16.23 item 1.** Verification of `DocumentedSplitMerge` register
    anchors that appear only in test source stays deferred to the recording commit. One consequence
    noted rather than buried: with the mechanism counts now derived, in-module blindness to a
    *mislabelled* `DocumentedSplitMerge` is total, which strengthens the case for the anchor-grep
    option when that deferral is taken up.

**Maintainer decisions.** (i) The page, its format, and the cap — **RESOLVED 2026-07-29, see item
17.** (ii) Scheduling the three §16.20 item 3(f) signatures. This gates the fixture commit and is a
calendar problem, not an engineering one; still open.

17. **RESOLVED (maintainer, 2026-07-29) — the oracle page is committed byte-for-byte as the served
    JPEG, under mechanism 16(b), and §7.2's 400 KB cap is exceeded by ~10% as a ratified
    deviation.**

    The page: **Pepper&Carrot** episode 1 *"Potion of Flight"*, page 1, Japanese translation, by
    David Revoy and the Pepper&Carrot translation contributors, **CC-BY 4.0** — redistribution
    permitted with attribution, so §7.2's license-clean requirement is met.
    `1200 x 1660`, baseline JPEG, **441,914 bytes**, sha256
    `3bef9922e09cea66ab12271da0070025768ae9bc5d286f41ced617468131267e`, retrieved 2026-07-28,
    **unmodified**. An `ATTRIBUTION.md` entry in the existing
    `tests/fixtures/upstream/ATTRIBUTION.md` format is part of the recording commit.

    (a) **Why this page.** Three panels, two multi-line speech balloons, and three *unbubbled* SFX
    runs (ぱらぱら / ざぶん / どばどば). The bubbled-vs-unbubbled mix is what makes §9.7(B)11's tier
    assertions meaningful — a single-balloon page would render that gate trivial. Rejected from the
    same episode: P02 (448,475 B, no extra coverage), P03 (302,541 B but a one-balloon splash),
    P04 (a 1200x24 footer strip, not a page).

    (b) **The cap deviation, ratified.** 441,914 B against "≤ 400 KB" — about 10% over. Accepted in
    favour of byte-for-byte provenance: `tests/fixtures/upstream/ATTRIBUTION.md` establishes the
    repo convention that vendored fixtures are unmodified and hash-verifiable against their source
    URL, and re-encoding a JPEG concentrates generational artifacts on **text edges**, which is
    precisely the signal a text-detector fixture must hold stable. Measured alternatives, recorded
    so the choice is auditable: q=85 → 413,114 B (still over), q=80 → 366,438 B (under), q=75 →
    246,993 B. CC-BY 4.0 permits modification, so this was a quality decision, not a licensing one.

    (c) **Mechanism 16(b) is the live one, and its two-digest check is NOT vacuous.** Ours consumes
    the committed JPEG through the `image` crate; upstream consumes the `image`-crate re-decode
    written as a scratch PNG through `cv2`. **These are two different files read by two different
    decoders**, so item 16(b)'s requirement — record the digest of the decoded RGB buffer fed to
    each side and assert the two equal — guards two live failure modes: the PNG round-trip between
    two libraries, and **BGR/RGB channel order**, which §16.20 item 6 established as load-bearing
    (`cv2.imread` yields BGR while the cv2 branch of `preprocess_img` feeds RGB). Any
    implementation that drops the comparison on the grounds that "both sides read one file" is
    factually wrong about the data flow and must be rejected.

    (d) **Letterbox ratio, deliberately not 1.** This page gives `r = 1024/1660 ≈ 0.617`, so the
    resize path runs on both sides rather than being bypassed. That is coverage, not a defect:
    §16.20 item 5 measured the resize difference as moving **confidence** by 0.079–0.089 while
    **geometry stayed within L1 = 1, four of five coordinates exact**, and confidence is never a
    gating row (item 3(d)). Pre-resizing the page to `max(w,h) = 1024` would make `r = 1.0000` and
    eliminate the resize entirely — but it would also eliminate the only check that our resize
    agrees with upstream's on real content, while costing a cap raise (measured: 740x1024 lossless
    RGB PNG is 842,352 B; grayscale is 292,183 B but discards colour from a colour comic). Item
    12's "near 1" criterion exists to keep confidences away from the 0.4 gate on *upscaled small
    crops* (r ≈ 3.0–3.8); a 0.617 downscale is not that failure mode.

**Deferrals, recorded rather than dropped (§16.23 item 1's rule applied to this ruling).** (i) The
upstream environment's reproducibility window (§16.20 item 7 licenses less than it appears to)
remains an accepted, named risk; recorded dependency versions make a later divergence
attributable, not preventable. (ii) Whether upstream confidence capture succeeds is left to the
recording run; item 9 closes the row either way, so nothing blocks on it. (iii) The choice between
`xtask/tests/` integration tests and in-module `#[cfg(test)]` unittests for xtask's non-digest
tests is left free — both run in the default tier; only item 1(e)'s digest test is bound to
`xtask/tests/`, because it needs `pc-models`. (iv) The architect's five-property preservation
table is **moot** under item 1 and is deliberately not carried forward; if a future ratification
ever does replace the frozen gate, that table is the starting bar and this deferral is the pointer
to it.

## 16. Summary of what v1 is NOT

Global out-of-scope list, so Codex has one place to check before building anything speculative:

- **Inpainting** (LaMa, `InpainterConfig`, `_inpainting.png`, `_clean_inpaint.png`, any inpainting fallback for failed masks) — v1.5.
- **PSD / layered export** (`LayeredExport`, per-image and bulk PSD, `koharu-psd` port) — v1.5.
- **GUI** (`egui`, staleness-aware recompute, OCR review window, `Output`-driven image viewer) — v2.
- **GPU execution providers**: **CUDA — v1.5, opt-in** (§16.22; `device = "cuda"`, quarantined from every gate/fixture/recording, requires user-provided CUDA 12 + cuDNN 9 runtime libraries). CoreML, DirectML, `.pt`/torch loading and multi-device dispatch — v2.
- **Windows support** — v2 (Linux + macOS only).
- **Legacy INI config import** — v1.5 (`pc-config` is TOML-only in v1).
- **Tesseract / non-Japanese OCR engines**, OCR review/edit workflows, OCR result *parsers* (import) — v1.5+.
- **Font-rendering-dependent debug visualizations**: `_raw_boxes.png`, `_boxes.png`, `_boxes_final.png`, `_mask_fitments.png`, `_std_devs.png` — v1.5. (v1 debug artifacts are limited to `_box_mask.png`, `_cut_mask.png`, `_with_masks.png`, none of which need text.)
- **DBNet line polygons**, line-based block splitting/merging, orientation/font-size estimation — v1.5.
- **Upstream `refine_mask`/`refine_undetected_mask`** (top-k colour + Otsu + XOR merge) — v1.5 as `MaskRefineMode::Annotation`.
- **Lab-space coloured NLM** and `color_filter_strength` — v1.5.
- **Post-action hooks**, memory watcher / OOM warnings, translations/i18n, `stitch_all` debug merging — v1.5+.
