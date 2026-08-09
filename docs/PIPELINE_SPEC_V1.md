# panel-ocr v1 Pipeline Architecture Spec

**Status:** Draft from Technical Architecture (Opus) — pending Senior Rust Engineer
co-review (test code + disagreement check) per CLAUDE.md's Plan phase.
**Scope:** v1 milestone only — CLI + detect → preprocess → mask → denoise → export
**Fixed constraints:** everything in `docs/ARCHITECTURE_DECISIONS.md` (GPL-3, full-Rust/no-Python-runtime, `ort` 2.0.0-rc.12 pinned, CPU-only EP, TOML config, stage-based crates, fresh `clap` CLI, Linux + macOS, GUI/inpaint/PSD deferred). The platform constraint here reads
"Linux + macOS" for v1 only; Windows joins it at v1.1 per §16.33, and
`docs/ARCHITECTURE_DECISIONS.md`'s "Platform support" section is the authority this line forwards
to.

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
**SUPERSEDED in part by §16.38 item 11 — read it before citing this section's `Step` list.** That entry inserts `Step::Inpaint` between `Denoise` and `Export` at v1.5, so `Step::Export as i32` becomes `6` and `Step::Export.prev()` becomes `Some(Step::Inpaint)`. The `Output` variant list, the `cache_suffix` list and the non-surjectivity note below are unchanged.
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

`rayon` `par_iter` over images at the *whole-pipeline* granularity (not per stage), with a semaphore-free bound: `min(config.max_threads_or_cpus, images.len())`. Rationale: upstream parallelizes per-stage with a process pool because Python needs it; per-image parallelism in Rust gives better cache locality and makes per-image error isolation trivial. The detect stage is the exception: one `ort` session is created per run, but it is owned by a single dedicated worker thread (`pc-detect-onnx`, §16.32) and no rayon worker ever touches it directly. What the rayon workers share is the `Arc<dyn TextDetector>`: each sends its input tensor to the worker over a channel and decodes the reply on its own thread. `ort::Session` being `Send + Sync` for the CPU EP is still true and is no longer the reason anything works. `text_detector.concurrent_models` still controls how many sessions are created (default 1, shared), and a value greater than 1 is warned-and-ignored in v1.

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
**SUPERSEDED in part by §16.38 item 13 — read it before citing the "minus `[inpainter]`" clause below.** That entry adds the eight-key `[inpainter]` block at v1.5, with the same values `config.py` declares. Nothing else in this section is changed by it.
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
device                       = "cpu"     # cpu | cuda ("cuda" needs a build with the `cuda`
                                         # feature: §16.22, §16.36). Never auto-detected
                                         # (§16.22 item 5(a)) -- deliberate divergence from
                                         # upstream, which auto-detects. Global, not
                                         # per-model, so v1.5's LaMa inpainting and OCR
                                         # reuse the same resolver (§16.22 item 5(f), §16.36
                                         # item 2). Deliberate v1.1 addition, not present
                                         # upstream.

[text_detector]
model_path                   = ""        # empty = use managed cache
concurrent_models            = 1
intra_threads                = 0         # 0 = let ONNX Runtime choose (§16.21)
inter_threads                = 0         # 0 = let ONNX Runtime choose (§16.21)
mask_refine_mode             = "simple"  # simple | annotation (Annotation is opt-in; Simple remains the default). Deliberate v1 addition, not present upstream.

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
mask_fallback_to_lowest_deviation = true  # §10.3 step 10 / §16.35. Deliberate v1.1
                                          # addition, not present upstream: DEVIATION(21).

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
- Execution provider: **CPU by default**, `[text_detector] intra_threads` defaulting to **`0`** (all physical cores), session created once and shared. **Amended by §16.21 item 1, §16.22 item 1, and §16.36 item 1.** The former text read *"CPU only, `intra_threads = 1` (parallelism is at the image level)"* — that parenthetical was false in the shipped design, because §14.15's `Mutex<Session>` meant image-level parallelism never reached the detector, so inference was serialized and single-threaded (~150 s/page) (the mutex became a dedicated worker thread — §16.32; the serialization is unchanged). `1` remains expressible. CUDA is available opt-in via `[general] device = "cuda"` under §16.22's binding conditions and §16.36's placement ruling (it lives in `[general]`, not here, so OCR and v1.5's LaMa inpainting share one resolver); it is quarantined from every gate, fixture and recording.

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
  1. For each detected block, `expanded = rect.pad(16, image_size)` (upstream `expand_textwindow(expand_r=16)`). **The parenthetical here is SUPERSEDED by §16.37 item 5 — read it before citing this line as upstream parity.** Upstream's `expand_r` is a divisor, yielding 3–5 px on the committed page, not a flat 16, and it clamps to `im_w - 1`/`im_h - 1`. `Simple`'s behaviour is deliberately unchanged (it is a port of koharu, not of PanelCleaner); only the parity claim is withdrawn.
  2. Rasterize the union of expanded rects into an in-bounds mask.
  3. `base[p] = 255` iff `in_bounds[p] != 0 && raw_mask[p] > 60`.
  4. Dilate `base` with an L1 (diamond) structuring element of radius 3.
  5. Clip the dilated result back to `in_bounds` (zero outside).
  If `blocks` is empty, the refined mask is all-zero.
- The refined mask is what `PageDataRaw.raw_mask` points at (upstream also stores the refined one: `ctd_interface.py:182`).
- **Out of scope for v1:** upstream's `refine_mask` (top-k colour masklists + per-channel Otsu candidates + connected-component XOR merge + hole filling, `comic_text_detector/utils/textmask.py:18-210`) and `refine_undetected_mask`. Rationale: the masker's own growth-and-score loop is what determines final quality; a coarser precise mask degrades fit granularity but cannot break the pipeline, and the XOR-merge algorithm is the single riskiest numerical port in the whole project. Keep the door open with `enum MaskRefineMode { Simple, Annotation }` in `TextDetectorConfig` where `Annotation` is the opt-in upstream refinement path and `Simple` remains the default. See §15.2.

  **The parenthetical `Annotation` → `StageError::InvalidInput` clause on the line above, and the "out of scope for v1" framing it sits in, are SUPERSEDED by §16.39 — read it before citing this bullet as current behaviour.** A4 ports both functions and wires them; `Annotation` is opt-in and non-default, and the stage no longer refuses it. Everything this bullet says about `Simple` is unchanged. (Own physical line, per cookbook rule 14b.)

**6 — False-positive filter.** For each block, `mask_coverage = mean(refined_mask over rect) / 255.0`; drop blocks with `mask_coverage < 0.1`. Provenance: upstream applies this `mask_score_thresh = 0.1` test in `group_output` to blocks that received no DBNet text lines (`textblock.py:485-490`); since v1 emits no line polygons (below), *every* block is line-less and the filter applies to all of them — which is exactly the code path upstream takes when the line map yields nothing.

**The operand named above — `refined_mask` — is SUPERSEDED by §16.39 for `MaskRefineMode::Annotation` only; read it before citing this step as mode-independent.** Under `Annotation` the filter scores the **unrefined**, letterbox-cropped, resized detector mask, which is upstream's own operand. Under `Simple`, the shipped default, this step is exactly as written above. The 0.1 threshold and the every-block scope are the same in both modes. (Own physical line, per cookbook rule 14b.)

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
| **P7** | `pc-ocr`: manga-ocr ONNX backend (encoder/decoder sessions, greedy decode loop, vocab, preprocessing) — port from koharu `manga_ocr` — **"greedy decode loop" is SUPERSEDED by §16.30 item 4: upstream runs beam search (`num_beams=4`, `length_penalty=2.0`, `early_stopping=true`, `no_repeat_ngram_size=3`, `max_length=300`), measured at 12/12 against greedy's 10/12 on manga-ocr's own 12 published labels. Original wording kept per §16.19's convention. §16.30 item 3 also splits this task into P7 (`pc-ocr` only) plus a mandatory P8 (CLI/pipeline wiring, depends on P7); P8's own row is added by the P7/P8 task-breakdown plan, not by that ratification entry (§16.30 item 3(iii)).** | **heavy** |
| **P8** | `pc-models` registry entries for the manga-ocr weights (§16.30 item 1's pin); `pc-cli`'s eager, image-independent `MangaOcrFactory` construction behind the `onnx` feature (fatal, verbatim-message refusal when the feature is absent, mirroring §16.12 item 2's detector precedent); `PipelineCtx::with_ocr` wiring into `run_clean` (gated on `profile.preprocessor.ocr_enabled`) and `run_ocr` (unconditional — that subcommand's purpose); `run_ocr`'s §15.5 report-path overrides (`ocr_blacklist_pattern = ".*"`, `ocr_max_size = 10^10` — **NARROWED by §16.34 item 4: a third override, `ocr_enabled = true`, is also required**); removal of the two "v1 ships no OCR engine" `WARN`s this task's own existence makes false. Depends on P7 (§16.30 item 3, mandatory for v1.0, not optional follow-up). | simple |

Batching: `{P1, P2, P3, P4, P5}` one sequential call (all pure geometry/boilerplate on the same data structure); `{P6}` one call; `P7` isolated; `P8` one call, after `P7`.

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
10. **Failure threshold. SUPERSEDED by §16.35 — read it before citing this clause.** As originally specified, and still exactly what runs when `mask_fallback_to_lowest_deviation = false`: if `best_dev > mask_max_standard_deviation` → return `Fitment { mask: None, .. }` (analytics + `MaskData` entry still produced, `failed: true`). Otherwise `Fitment { mask: Some(best_mask), .. }`. §16.35 adds a fallback step that runs first when the flag is `true` (the shipped v1.1 default): before giving up, retry with the lowest-`std_deviation` candidate among those already scored in step 9; only if that candidate *also* exceeds `mask_max_standard_deviation` does this step's original `None` outcome apply. The threshold comparison itself — the fail-safe — is unchanged; only which candidate reaches it can differ.
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
**SUPERSEDED in part by §16.38 item 12 — read it before citing `ExportSources`' field list.** That entry adds `inpainted` and `inpainted_mask` at v1.5.
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
**SUPERSEDED in part by §16.38 item 12 — read it before citing step 2's precedence.** That entry puts the inpainted sources above the denoised ones in both the cleaned and the mask precedence at v1.5, and extends the stale-artifact rule to them.
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
| 14 | **N2/N3/N4** | pc-denoise | Gaussian blur; noise-mask build; `run()` + 1-bit shortcut + analytics — **the "Gaussian blur" attribution to `pc-denoise` is SUPERSEDED in part by §16.38 item 20: at v1.5 the module lives in `pc-imageops` and `pc_denoise::gaussian` is a re-export. Original wording kept per §16.19's convention. Only that clause is affected — the noise-mask build, `run()`, the 1-bit shortcut and the analytics are still `pc-denoise`, and `nlm` (row 15) is untouched.** | simple | M6 |
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
| 30 | **P7** | pc-ocr | manga-ocr ONNX backend (encoder/decoder, greedy decode, vocab) — **"greedy decode" is SUPERSEDED by §16.30 item 4: upstream runs beam search (`num_beams=4`, `length_penalty=2.0`, `early_stopping=true`, `no_repeat_ngram_size=3`, `max_length=300`), measured at 12/12 against greedy's 10/12 on manga-ocr's own 12 published labels. Original wording kept per §16.19's convention. §16.30 item 3 also splits this row into P7 (`pc-ocr` backend only) plus a mandatory P8 (CLI/pipeline wiring, depends on P7); P8's own row is added by the P7/P8 task-breakdown plan, not by that ratification entry (§16.30 item 3(iii)).** | **heavy** | P6, D1 |
| 31 | **P8** | pc-cli, pc-pipeline, pc-models | `pc_models` registry entries for the manga-ocr weights (§16.30 item 1); eager `MangaOcrFactory` construction behind the `onnx` feature (fatal refusal when absent, mirroring §16.12 item 2's detector precedent); `PipelineCtx::with_ocr` wiring into `run_clean` (gated on `ocr_enabled`) and `run_ocr` (unconditional); `run_ocr`'s §15.5 report-path overrides; removal of the "v1 ships no OCR engine" `WARN`s. Mandatory for v1.0, not optional (§16.30 item 3). | simple | P7 |

**Suggested Codex call batches:**
`[F0, C1, C2]` · `[C3]` · `[C4]` · `[D2]` · `[D8]` · `[M1]` · `[M2]` · `[M3]` · `[M4]` · `[M5, M6]` · `[P1..P5]` · `[P6]` · `[N1]` · `[N2, N3, N4]` · `[E1, E2, E3]` · `[E4]` · `[D1, D3]` · `[D4]` · `[D5]` · `[D6]` · `[D7]` · `[G1, G2]` · `[D9, E5]` · `[X1]` · `[F1]` · `[F2]` · `[P7]` · `[P8]`

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
**SUPERSEDED in part by §16.38 item 19 — read it before citing the `models` line above.** That entry adds the visible `--include-optional` flag to `models download` and `models verify` at v1.5, and states what each of the three subcommands does with an `Optional` registry entry. `models path` itself is unchanged. Nothing else in this section is changed by it.
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

    **This entry is narrowed by §16.39 — read it before citing it as a statement about the whole tree.** It is not retired: A4 ports both upstream functions and ships them as `MaskRefineMode::Annotation`, but the shipped **default** stays `Simple`, so a default run still diverges from upstream's unconditional refinement. What survives is exactly that default-value divergence; the "we did not port the algorithm" half does not. The primary site comment moves from `refine_simple` to the `#[default] Simple` variant, because the artifact carrying the divergence is now the default value and not the algorithm. (Kept on its own physical line for the reason item 17's note below states.)

13. **Class-agnostic NMS** — upstream runs per-class NMS (`agnostic=False`, `inference.py:115` + `yolov5_utils.py:261`), which can emit duplicate boxes for one balloon across language classes; we run class-agnostic NMS instead (see §8.3 step 4, §15.1). `yolo.rs` must carry a `// DEVIATION(13): ...` comment at the NMS call site.
14. **Letterbox minimum dimension clamp** — upstream can pass a zero-sized resize dimension to `cv2.resize` for sufficiently small inputs; v1 clamps each rounded dimension to at least `1`, keeping the resize valid and preventing `dw` or `dh` from reaching `1024` and making `mask::crop_letterbox` reject the geometry. The implementation comment is at `crates/pc-detect/src/onnx.rs::letterbox` as `DEVIATION(14)`.
15. **One shared ONNX session** — upstream would honour `text_detector.concurrent_models` at provider construction; v1 shares one session for the whole run — owned exclusively by a dedicated `pc-detect-onnx` worker thread (§16.32; formerly a `Mutex<Session>`), so requests are serialised by the worker's request channel rather than by a lock — and a configured value greater than 1 is warned-and-ignored. The implementation comment is at `crates/pc-detect/src/onnx.rs`, on the OnnxDetector doc comment, as `DEVIATION(15)`.
16. **Lazy construct-and-latch** — upstream constructs the detector before its per-image loop; v1 defers construction until the first image that needs detection and latches that attempt's success or rendered refusal for the run, so §4.4 resume can bypass a model it will never read. The implementation site will carry `DEVIATION(16)`; the concurrent implementation pass has not added that comment yet.
17. **Coverage-filter scope and operand** — ratified by §16.20 item 9, which corrects §8.3 step 6's former claim of unconditional parity. Upstream applies its `mask_score < mask_score_thresh` false-positive filter **only to line-less blocks** (`textblock.py:485-490`, inside `if len(blk.lines) == 0:`) and computes it over the **unrefined** mask (`inference.py:203` passes `mask` to `group_output`; `refine_mask` runs at `:204`). v1 applies the filter to **every** block over the **refined** mask. Both differences are deliberate: v1 synthesizes no DBNet line polygons (§14.12 and §8.3's out-of-scope list), so every block is line-less by construction and the scope difference is vacuous *for v1* — it would become live the moment line synthesis lands, which is why it is registered rather than left as prose. The operand difference makes our coverage values roughly 2× upstream's on the same boxes; measured across two real manga pages the filter has never fired (minimum coverage 0.3025 against a 0.1 threshold). The implementation site carries `DEVIATION(17)`.

    **This item's "never fired" measurement is SUPERSEDED by §16.37 item 4 — read it before citing that sentence.** The filter *has* fired, on the committed fixture, under the shipped `Simple` mode, since before either section was written. This item's scope and operand rulings are unaffected; only the parenthetical measurement is withdrawn. (Kept on its own physical line: the line-based scanner in `crates/pc-testkit/tests/spec_supersession.rs` attributes every anchor sharing a line with a verb, so folding this into the paragraph above manufactures phantom claims against the older anchors there — cookbook rule 14b.)

    **This item's OPERAND sentence is narrowed by §16.39 — read it before citing "the refined mask" as unconditional.** *"v1 applies the filter to **every** block over the **refined** mask"* describes `MaskRefineMode::Simple`, which is still exactly what it says and is still the shipped default. Under `MaskRefineMode::Annotation` the filter scores the **unrefined**, letterbox-cropped, resized detector mask instead — which is upstream's own operand, so under that mode the operand divergence this item registers does not exist. The **scope** half (every block, not only line-less ones) and the 0.1 threshold are untouched by that entry, and so is `DEVIATION(17)`'s implementation site. (Own physical line, same reason as the note above.)

18. **Rescaled-box clamp to image bounds — ratified by §16.27 item 9, which re-grounds it.** After truncating rescaled detector coordinates to `i32`, v1 clamps `x1,y1` to `>= 0` and `x2,y2` to `<= image_size`. Upstream does **not** do this: `grep -rn clip_coords` over the pinned checkout `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3` exits 1 (no hit in any file), and the only clamping in the vendored yolov5 helper is IoU arithmetic — `yolov5_utils.py:166` comments `.clamp(0)` and `:169` calls it, both inside `box_iou`'s intersection computation, and they are that file's only two occurrences of `clamp` or `clip`. So nothing upstream bounds these coordinates to the frame at any point — verified by grep, not inferred. v1 clamps because `run()` must produce a `PageDataRaw` that passes its own `validate()`, an invariant that is ours and has no upstream counterpart — which is what makes this a deliberate divergence and not the port it was previously described as. The clamp is unchanged in behaviour by that re-grounding: no relaxation, no `validate()` change, no deletion. The implementation site must carry `DEVIATION(18)` at the clamp in `crates/pc-detect/src/yolo.rs`, that file's doc comment must stop citing `clip_coords`, and so must the comment at `crates/pc-detect/tests/d5_yolo.rs:297`, which repeats the withdrawn citation at a second site. All three are **authorised by this entry** and are follow-up work at the time it lands; §16.27 item 9 enumerates them, with the reason the enumeration is explicit rather than summarised, rather than leaving them implicit.

19. **Windows cache root is `%LOCALAPPDATA%`, not upstream's `%APPDATA%` — ratified by §16.33 item 3 (maintainer, 2026-08-04).** Upstream PanelCleaner, run as the tiebreak oracle, places **both** its cache and its config under `%APPDATA%` on Windows. v1.1 splits them: cache under `%LOCALAPPDATA%\panel-ocr`, config under `%APPDATA%\panel-ocr`. Reason: `%APPDATA%` roams with the user profile on a domain-joined machine, and a regenerable model/image cache — model weights are ~95 MB for the detector alone — must not be copied across the network on every logon. This is a deliberate divergence from measured upstream behaviour, not a port, which is why it is registered rather than left as prose. The implementation site is the Windows branch of the resolver in `crates/pc-cli/src/paths.rs` and must carry `DEVIATION(19)`; that comment **exists** as of 2026-08-04, at `crates/pc-cli/src/paths.rs:90` in `cache_dir`'s Windows arm — the register entry landed with the ratification and the site comment landed with the implementation, the same sequencing `DEVIATION(16)` went through — item 16 above still describes that comment as not yet added, which is stale: it exists at `crates/pc-cli/src/detector.rs:39` as of 2026-08-04. **Line-number correction, 2026-08-06, unrelated to §16.37:** this citation read `:36`, which was **correct when written** on 2026-08-04 in commit `83f7f89` — the comment genuinely sat at line 36 then — and went stale in commit `1f2bd73` (§16.36 G1-C), which inserted lines above it and moved it to `:39`. Verified by `git log -S "detector.rs:36" -- docs/PIPELINE_SPEC_V1.md`, and by reading `crates/pc-cli/src/detector.rs` at `1f2bd73^` (line 36) and at `1f2bd73` (line 39); `grep -rn "DEVIATION(16)" crates/` today returns exactly one hit, at `:39`. **This is staleness, not an original error** — the same distinction item 20's note in the next paragraph draws about its own sentence, and the distinction the first draft of this correction got backwards. Found incidentally during §16.37's review; nothing else in this item changes, and the identical `:36` in §16.33 item 3 is corrected the same way. Correcting item 16's sentence is a separate, pre-existing matter and is not done by this item. Linux and macOS roots are unchanged by this item.

**Item 20 is intentionally left unused here.** `UpstreamBoxOutsideFrame` is untranscribed — the note below states landing it "needs its own ruling" — but the closest existing numeric statement about it counts `Divergence` *enum variants*, not §14 register items: §16.27 item 11 describes it as "not among the 19" known-divergence samples, and landing it later "makes it a 20th variant by its own ratification" (that variant-count language, not a §14 reservation, is what this item avoids colliding with). The items below therefore start at 21, and are numbered 21, 22, 23 and 29. (This sentence read "the two items below are therefore numbered 21 and 22" until §16.37 added item 23, and "21, 22 and 23" until §16.37 item 10 added the entry now numbered 29; each reading was true when written and false the moment a further item landed, which is why the count is stated as a start-point plus a list rather than a total. The list is maintained by hand and nothing gates it — a fifth item lands with its own edit here.) **Items 24-28 are also unused here, and for a different reason: they belong to the sibling `lama-inpaint` branch.** That branch reserved 24-27 in its committed `ee7cb60` (2026-08-06 16:47:04 +0700) after checking this branch's state at the time, and has since claimed 28; this branch minted its own 24 in `70a617c` (2026-08-07 00:23:14 +0700) without re-checking, so the fourth entry below is **renumbered from 24 to 29** to resolve that cross-branch collision. 29 was verified free across every branch: the union of numbers claimed anywhere is 1-19 and 21-28. Only the **identifier** moves — the proposition, grounds, implementation site and severity framing of that entry are unchanged, and the migration is total on this branch — which is why no supersession row is recorded for it. A merge with `lama-inpaint` must still reconcile 24-28 in the other direction; this note only settles this branch's side.

21. **Mask fallback to the lowest-deviation scored candidate — ratified by §16.35 (joint architect + Rust Engineer plan, Fable tie-break on tie-direction and default, 2026-08-05).** When `pc-mask::fit::fit_region`'s greedy candidate selection (§10.3 step 9) lands on a candidate whose border `std_deviation` exceeds `mask_max_standard_deviation` (§10.3 step 10), v1.1 retries with the lowest-`std_deviation` candidate already scored in step 9, and paints it only if that candidate itself passes the same threshold; if none does, behaviour is unchanged (`mask: None`). Upstream has no such retry (verified at pinned commit `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`: `pcleaner/image_ops.py`'s `pick_best_mask` returns `best_mask=None` unconditionally on this branch), so with the shipped default (`mask_fallback_to_lowest_deviation = true`) v1.1 paints some boxes upstream leaves unmasked — a genuine default-output divergence, not just an opt-in knob. `mask_fallback_to_lowest_deviation = false` restores upstream's exact selection behaviour. The implementation site is the rescue-selection branch in `crates/pc-mask/src/fit.rs` and must carry `DEVIATION(21)`.
22. **Device selection is opt-in only, never auto-detected — ratified by §16.36 (joint architect + Rust Engineer plan, Fable tie-break, 2026-08-05); binding condition already registered at §16.22 item 5(a).** Upstream auto-detects: `pcleaner/ctd_interface.py:64` (`device = "cuda" if torch.cuda.is_available() else "cpu"`) and `pcleaner/main.py:440`/`:815` (same for `mps`), verified at the pinned commit. v1 and v1.5 never auto-detect on any platform; `[general] device` defaults to `"cpu"` and `"cuda"` must be set explicitly. The implementation site is `pc_core::device::resolve` (§16.36 item 2) and must carry `DEVIATION(22)`.

23. **`get_topk_color`'s histogram tie order is declared by us, because upstream's is unspecified — ratified by §16.37 (joint architect + Senior Rust Engineer plan, 2026-08-06).** Upstream orders candidate colours with `np.argsort(bins * -1)` (`comic_text_detector/utils/textmask.py:19`, pinned commit `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`), whose default introsort has no contractual ordering among equal keys — and the input carries tie groups of up to 255 equal counts. Measured: switching only to a stable sort moves upstream's own refined mask by **277 differing pixels on E01P02** (251 on, 26 off, net +225) and **27 on E01P03** (all 27 off), while leaving **E01P01 byte-identical**; and CPU dispatch alone (AVX2 vs SSE) changes how many candidate colours the function returns (2 vs 3) **on synthetic input**, while **likewise leaving E01P01 byte-identical**. Both qualifiers are load-bearing and are carried here rather than left in §16.37 item 1: neither hazard has been shown to change a real page's output, and dropping "synthetic input" or the countervailing E01P01 result would widen the claim past its evidence. v1.5 uses a **stable** sort on `(Reverse(count), ascending bin index)`. Same class as item 2 above, whose grounds transfer directly. The implementation site is the histogram/tie-rule function landed by §16.37's task A1 — `top_k_colors` in `crates/pc-detect/src/annotate.rs` — and must carry `DEVIATION(23)`; that comment **exists** as of 2026-08-06, at `crates/pc-detect/src/annotate.rs:266` (the doc comment declaring the rule) and `:283` (the sort site inside `top_k_colors`) — the register entry landed with the ratification and the site comment landed with the implementation, the same sequencing `DEVIATION(16)` went through. **Correction, 2026-08-06:** this sentence read *"that comment does not exist yet"*, which was **correct when written** at R0 — that task was spec-only, no code — and went stale the moment A1 landed; verified by `grep -rn "DEVIATION(23)" crates/pc-detect/src/`, which returns the two **declaration** sites — the doc comment and the sort site, whose live line numbers are given by the second correction at the end of this item — and, since A2, two further **prose** mentions at `crates/pc-detect/src/annotate.rs:558` and `:600`, doc-comment sentences in A2's own functions relating their output ordering to this item's tie rule. Those two declare nothing and are not implementation sites. **No hit total is quoted here, deliberately:** that clause read *"which today returns exactly those two hits"*, which was true when A1 landed and false as soon as A2's diff mentioned this item in prose — a count over a grep for a token that prose is free to discuss goes stale on any edit that discusses it. The obligation this item imposes is discharged by the two declaration sites cited below, not by a hit count. **This is staleness, not an original error**, the same distinction item 19 draws about its own `:36` citation. Item 16's own version of that sentence is a separate, pre-existing matter and is **not** corrected here: as item 19 records, the `DEVIATION(16)` comment does exist, at `crates/pc-cli/src/detector.rs:39` (verified by `grep -rn "DEVIATION(16)" crates/`, which returns exactly that one hit). Item 16 is therefore cited here for the register-before-comment *sequencing pattern*, not as a live description of `DEVIATION(16)`. **Second line-number correction, 2026-08-06, made with item 29 below:** `:266`/`:283` were **correct when written** at A1 (verified: `git show HEAD:crates/pc-detect/src/annotate.rs | grep -n "DEVIATION(23)"` at commit `e3a9d70` returns exactly those two lines) and moved to **`:278`/`:295`** when A2 extended that file's module doc comment by twelve lines above them. The live citation is therefore `crates/pc-detect/src/annotate.rs:278` (doc comment) and `:295` (sort site). Staleness from an insertion above, exactly as item 19 records for its own `:36`; the rule and the comment text are unchanged.

29. **OpenCV's documented *reference* Otsu tie rule is what we port; IPP's undocumented fast-path near-tie behaviour is not — ratified by §16.37 item 10 (independent architect + Senior Rust Engineer takes, Fable tie-break on the provenance sub-point, 2026-08-06).** `cv2.threshold(..., THRESH_OTSU)` is not single-valued: `getThreshVal_Otsu_8u` (`modules/imgproc/src/thresh.cpp`, branch `5.x`) is guarded by `CV_IPP_RUN_FAST(ipp_getThreshVal_Otsu_8u(...))`, so on a wheel built with Intel IPP it short-circuits to `ippiComputeThreshold_Otsu_8u_C1R` and never reaches the published algorithm. **The registered proposition is narrow: v1.5 implements `getThreshVal_Otsu_8u`'s published tie rule — strict `sigma > max_sigma`, so of two candidates with equal between-class variance the lower bin index wins — and does not reproduce IPP's fast-path behaviour near a tie.** This is deliberately **not** registered as "we declare our own Otsu tie rule": the rule is OpenCV's own, read out of OpenCV's own source, and calling a port an invention is the paraphrase-widening defect §16.37 items 4 and 5 correct elsewhere in this register. **Grounds, transferred from item 23's:** that item's ground is §16.37 item 2's *"an introsort port would still be a function of a CPU feature set nothing pins, so there is no upstream ordering to be faithful to"*; here the same sentence holds with the noun swapped — an IPP-faithful port would be a function of a **linked proprietary library (IPP) and its dispatch level**, which nothing pins, so there is no single upstream threshold to be faithful to. **What transfers is the ground, not the mechanism:** item 23 declares a rule NumPy does **not** have, while item 29 ports a rule OpenCV's reference code **does** have, choosing between two implementations of one nominally-specified function. **Severity is not inherited from item 23 either:** item 23 moved a real page (277 px on E01P02), while the committed CC-BY test page is **not exposed** by this one — all 12 per-channel Otsu thresholds across E01P01's four boxes are identical on the reference and IPP paths (§16.37 item 10). The implementation site is `otsu_threshold` in `crates/pc-detect/src/annotate.rs` and must carry `DEVIATION(29)`; that comment **exists** as of 2026-08-06, at `crates/pc-detect/src/annotate.rs:410` (the doc comment declaring the choice) and `:449` (the strict comparison itself) — register entry with the ratification, site comment with the implementation, the same sequencing items 19 and 23 went through. **This entry was registered as item 24 when it landed on 2026-08-06, with its site comments carrying the matching 24-numbered marker, and was renumbered to 29 on 2026-08-07** because the sibling `lama-inpaint` branch had already reserved 24-27 in `ee7cb60`; see the gap note above §14's item 21 for the collision record. The identifier is the only thing that changed.

Each of these must appear as a `// DEVIATION(n): ...` comment at the implementation site referencing this section, so a future parity investigation finds them immediately.

**Not registered here: `UpstreamBoxOutsideFrame`.** §16.27 item 11 ratifies this as a decision
("adopted as a GATING row, 0/38 today") but explicitly leaves it **untranscribed** and states that
landing it "needs its own ruling" and "makes it a 20th variant by its own ratification, not a
consequence of this one." No `Divergence::UpstreamBoxOutsideFrame` variant, logic, or test exists
anywhere in `crates/pc-detect/` — `grep -rn UpstreamBoxOutsideFrame --include=*.rs .` returns zero
hits. A prior edit here registered a §14 entry describing it as live comparator behavior ("is
reported as a named GATING divergence"), which was false — this list is for deviations with a
**ratified** implementation plan, each of which must carry a real `DEVIATION(n)` site comment
once that plan lands (items 16, 19, 21 and 22 register ahead of their site comment landing, the
same sequencing item 19's own account describes; what this note reverts is a claim of *already
live* behavior with no ratification and no implementation, which is different). Reverted.
`docs/COOKBOOK.md`'s "still owed" bullet asking for this registration was itself the source of the
confusion (its parenthetical read as "already adopted" without stating "not yet implemented");
that entry has since been corrected to say so (commit `5b909d1`).

---

## 15. Decisions needing reviewer sign-off before tests are frozen

These are places where I made a call that a Senior Rust Engineer should confirm or overturn *now*, since tests become immutable afterwards.

All 10 items below were reviewed and decided by Fable (Senior Rust Engineer advisor), against direct verification of the upstream Python source. Decisions are binding; see each item for the evidence.

1. **Class-agnostic NMS** (§8.3 step 4) — **DECIDED: class-agnostic.** Verified upstream runs per-class NMS (`inference.py:115`, `yolov5_utils.py:261`, `agnostic=False`). Class-agnostic is the correct v1 choice specifically because it removes the duplicate-box problem at the source rather than relying on upstream's buggy box/language-desync merge (§14.3) to clean it up after the fact. Box-count divergence from upstream is accepted; see deviation §14.13.
2. **`MaskRefineMode::Simple` for v1** (§8.3 step 5) — **DECIDED: ship Simple, do not port upstream's full `refine_mask`/`refine_undetected_mask`.** Verified the full algorithm (`textmask.py:18-214`): top-k grey/Otsu masks + XOR-minimizing merge + hole filling — real work, correctly flagged as the riskiest port in the project. Deciding evidence: the `demo_bubbles` fixtures live in `media/` (README demo assets) and are **never used by upstream's own test suite** — their exact producing version/profile is unverifiable. Porting the riskiest algorithm in the project to chase parity with fixtures of unknown provenance is a bad trade. **Consequence (not a contingency — the plan of record):** §10.7(B) item 15's upstream-image comparison is downgraded from a frozen gate to a **non-gating calibration report** (see §10.7(B) rewrite below); the frozen masking test is a regression lock against our own recorded-fixture pipeline output, calibrated by F2. `Annotation` mode is now available as an opt-in for a full refinement port.

   **The "v1.5 door" sentence at the end of this item is SUPERSEDED by §16.39 — read it before citing this item as a description of the tree.** The door is open: A4 ships `Annotation`, opt-in and non-default. This item's decision — that `Simple` is what v1 ships by default, and that the upstream-image comparison stays a non-gating calibration report — is unchanged, and §16.39 depends on it. (Own physical line, per cookbook rule 14b.)

3. **Cleaned-image colour mode when a colour median meets a grayscale page** (§10.3 step 4) — **DECIDED: confirmed as specified**, with an upstream-accuracy correction. Upstream's base image is always 3-channel (`cv2.IMREAD_COLOR`, `ctd_interface.py:198`); its own `_clean.png` is RGB at `scale == 1` and is only restored to the original mode at export via `convert(original.mode)`. Our stage-level rule ("L if all medians achromatic, else RGB") is *export-equivalent*, not upstream-identical: since border-color computation is per-channel-symmetric, a grayscale page's medians are always exactly achromatic (r==g==b), and RGB→L is lossless in that case. Verified empirically against the `black_bubble` golden: its fill is exactly `(0,0,0)`, fully achromatic — confirming the premise that a colored fill would break this rule is false for all 7 demo_bubbles fixtures (all mode `L`). Implementation: composite in RGB internally, convert to `L` at write time when the rule says `L` — single code path, no branch duplication.
4. **Tesseract deferred to v1.5** — **DECIDED: confirmed.** Verified `config.py:375` (`ocr_use_tesseract: bool = False` default) and `ocr/ocr.py:65-66` (when disabled, the factory returns `MangaOcr()` for every language). v1's manga-ocr-only OCR exactly matches upstream's own default-profile behavior; no correctness gap.
5. **Fixture mapping correction** — **DECIDED: confirmed.** **NARROWED by §16.34 item 3 — read it
   before citing which overrides `run_ocr` applies.** The CSV/TXT fixtures define the export-stage OCR report format (E4), not filter behavior. Clincher: `run_ocr` (the code path that produces this format) explicitly sets `ocr_blacklist_pattern = ".*"` and `ocr_max_size = 10**10` (`main.py:866-868`) — the filter is inert in that code path, so the fixtures cannot be filter-behavior tests. (§16.34 item 3 adds the third override upstream also applies, `ocr_enabled = True`; it does not change this item's conclusion about the blacklist.)
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
3. **`mask_refine_mode = "annotation"` is accepted by config and selects the upstream refinement path**
   — config validation and stage validation are deliberately
   separate layers here; matches §8.3 step 5 / §15.2.

   **The "rejected by `pc-detect`" half of this item is SUPERSEDED by §16.39 — read it before citing it as current behaviour.** A4 wires `Annotation`, so the stage accepts it. The layer split this item ratified is untouched and still correct: config accepts the value, and whether the stage runs it is the stage's business. (Own physical line, per cookbook rule 14b.)

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

1. **`DetectInput` gains `pub config: TextDetectorConfig`** (§8.2's field list omitted it). Required because the stage selects the `MaskRefineMode::Annotation` refinement path per §16.5 item 3, and consistent with §3's "config by value in every stage `Input`" rule. Every stage crate therefore depends on `pc-config` (already noted in §16.5 item 1).
2. **Naming**: §8.2's `base_image_dest` is the authoritative spelling; §4.3's diagram (`base_png_dest`) is a typo — read `base_image_dest` there.
3. **`PageDataRaw.scale` stores exactly what `calculate_new_size_and_scale` returns**, even in the integer-inverse branch where that can differ slightly from `new_height / original_height` (e.g. `h=5001` gives `scale=0.5` but `new_h=2501`, so `new_h/h ≈ 0.50010`). This is intentional, not a bug: §11.3 already recomputes the denoiser's up-scale factor from actual image sizes rather than trusting `scale` for that purpose, so nothing downstream depends on `scale` being the exact ratio. §2.4's doc comment is amended to say "approximately `new_height / original_height`; exactly `1.0` when no resize happened" rather than claiming exactness.
4. **Rescaled detector boxes ARE clipped to image bounds.** §8.3 step 4 is amended: after truncating to i32, clamp `x1,y1` to `>= 0` and `x2,y2` to `<= image_size`. This isn't new scope — upstream's yolov5 pipeline clips coordinates (`clip_coords`) as a normal part of the postprocess the spec already claims to port; the original §8.3 step 4 text simply omitted mentioning it. Required so `run()` can produce a `PageDataRaw` that passes its own `validate()` on real (frame-overhanging) detector output. **(See §16.27 item 9 before citing this item's justification: the `clip_coords` citation in the sentence above is false — no such symbol exists in the pinned upstream checkout, `grep` exits 1 — and it is corrected there. The clamp rule itself is KEPT unchanged and re-grounded as entry 18 of the deliberate-deviations register; only the reason given for it was wrong.)**
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
   **SUPERSEDED in part by §16.38 item 16(a) — read it before citing this item's module placement.** That entry hoists the growth kernels out of the stage crates into `pc_imageops::morph` at v1.5; the "no `pc-config` dependency" property stated above is not changed by it, and `mask.rs` stops being the only new module.
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
   **SUPERSEDED in part by §16.38 item 20 — read it before citing this item's module placement for `gaussian`.** Scoped to `gaussian` and to nothing else: `nlm` stays in `pc-denoise`, this item remains the live placement decision for it, and §16.38 item 20 is explicitly not authority for hoisting it. What that entry cashes in is this item's own closing clause — both modules being `pub` and `pc-config`-free, so the hoist is "a mechanical move plus a re-export" — for one of the two modules it names.
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
   **SUPERSEDED in part by §16.38 item 16(a) — read it before citing the two-copies claim.** After the v1.5 hoist there is one implementation and no second copy that could drift, so the frozen cell-matrix test this item requires stops being a drift check and becomes a value lock.
3. **`pc-denoise` likewise gets its own `composite.rs`.** Same rule-2 reason. `blend_channel`,
   `alpha_composite_over`, `composite_rgb` and `resize_nearest_rgba` are pinned
   **identically** to §16.9 items 13 and 15 (`out = round(base·(1−a) + colour·a)`,
   source-over with `alpha_out = max(base_a, layer_a)`, nearest resampling as
   `src = floor(dst · src_len / dst_len)`), so Stage 3's composite and Stage 4's agree
   pixel-for-pixel — which §11.3 step 2 depends on, since it *reproduces* Stage 3's clean
   output rather than reading `_clean.png`.
   **SUPERSEDED by §16.42 — read it before citing this item's pin as current, and read
   its own scope carefully: this item pins `pc-mask` and `pc-denoise` only.** §16.42
   overturns the "restate rather than hoist" pin *for the functions this item names*
   (`blend_channel`, `resize_nearest_rgba`, `alpha_composite_over`, `composite_rgb`),
   which now hoist into `pc_imageops::composite` — matching the precedent this same
   §16.10 already scheduled for `nlm`/`gaussian` (item 1) and that §16.38 items 16(a)
   and 20 already executed for the morphology and the Gaussian blur. `pc-export`'s
   later, separate copy of three of these four functions is pinned by §16.11 item 10,
   not by this item (see the back-pointer there); `pc-inpaint`'s copy is required by
   §16.38 item 3(h) and its own module doc cites this item's pin **by analogy**, not as
   this item's direct authority, since `pc-inpaint` postdates this item. §16.42 covers
   all of them, but this item's own scope was always `pc-mask`/`pc-denoise` only — do
   not read it as having pinned the other two crates' copies itself. The rule-2
   reasoning this item gives for WHY `pc-denoise` could not just import from `pc-mask`
   is unaffected — that reasoning still forbids a stage crate depending on a sibling
   stage crate, and is exactly what motivates hoisting into the shared, non-stage
   `pc_imageops` crate instead of leaving the copies in place.

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
   **SUPERSEDED in part by §16.38 item 12 — read it before citing this precedence as pinned.** That entry adds the inpainted sources at the top of both the cleaned and the mask precedence at v1.5.
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

    **SUPERSEDED by §16.42 — read it before citing "same v1.5 consolidation ticket" as
    still pending.** The ticket this item named is cashed in: `resize_nearest_rgba` and
    `alpha_composite_over` (this item covers only these two of the three functions
    `pc-export` actually has — `blend_channel` is pinned by this same reasoning but not
    named in this item's text) hoist into `pc_imageops::composite`, along with the
    `pc-mask`/`pc-denoise` copies §16.10 item 3 pinned. `pc-export`'s
    previously-absent `pc-imageops` dependency is added as part of this hoist.

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

4. **(SUPERSEDED IN PART by §16.30 — read it before treating "until P7" as this item's completion
   condition. §16.30 item 3 splits P7 into P7 (the `pc-ocr` backend) plus a mandatory new P8 (the
   CLI/pipeline wiring), and rules that v1.0 is not done until both land. So the promise this item
   makes — that the `WARN` retires and OCR-based box discarding becomes reachable "until P7" — is
   discharged when P8 lands, not when P7 does. §16.30 item 3(ii) also records that the two live
   `WARN` strings still read "task P7" and are corrected at P8, not by that ratification. The
   original wording below is kept unchanged per §16.19's convention; only the task that retires it
   has moved.)**
   **OCR has no engine in v1, and that is not an error.** P7 (manga-ocr) is unstarted,
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

21. **PARTLY SUPERSEDED by §16.33 item 3 — read it before citing this item's cache/config
    default roots.** Exactly two things in this item are replaced there: the platform
    enumeration for the default roots (Windows is added, with `%LOCALAPPDATA%` for cache and
    `%APPDATA%` for config), and the second clause of the no-new-dependency parenthetical
    ("and v1 is Linux + macOS only"). The `--cache-dir > config.cache_dir > platform default`
    precedence, the single `paths::resolve_cache_root` funnel, the hidden-flag rationale, the
    recovery-command-from-resolved-state rule, the POSIX single-quote rule and the non-UTF-8
    `Path::display()` limitation all stand unchanged. §16.33 item 6 adds a PowerShell quoting
    flavor beside the POSIX rule for Windows hosts; it does not alter the POSIX rule.

    **Two hidden flags beyond §13.1's surface**, both `hide = true` so the documented
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
    macOS only — the second half of that parenthetical is superseded by §16.33 item 3,
    which adds the Windows row; `dirs` still stays out of the manifest).
    The recovery suggestion is derived from the resolved cache root used
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
   `rayon::ThreadPoolBuilder` site, `DEVIATION(12)` on the default `Simple` variant of `MaskRefineMode`.
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

   (Re-observed 2026-08-04, after §16.32 moved inference onto a dedicated worker thread: a per-image inference panic now originates on the worker, so the default panic hook's stderr line names `pc-detect-onnx` rather than the calling thread — observed as `thread 'pc-detect-onnx' (<tid>) panicked at ...`. There is still exactly ONE such line: `resume_unwind` on the receiving thread does not re-invoke the panic hook. The pinned rendered string `panicked: <payload>` is byte-for-byte unchanged — the thread name is stderr from Rust's default hook and was never part of that pinned string, so this is an observation, not a change to what this item pins.)

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

10. **The session lock spanned only the `&mut Session` window, and a poisoned session was
    reused.** `pc-detect/src/onnx.rs::<OnnxDetector as TextDetector>::detect` held
    `session` for exactly `Session::run` plus the copy of each output into owned
    `Vec<f32>`. All decoding — `bind_outputs`, `validate_output_shapes`, `decode_blocks`,
    `decode_mask` — ran **outside** the lock via the ungated `decode_outputs`, so a panic
    in v1's own arithmetic could not poison the one session `DEVIATION(15)` shared across the
    run. Previously the guard spanned the whole function, which turned one page's panic
    into an image-independent `Inference("ONNX session mutex was poisoned")` for every
    remaining page: a failure mode v1 manufactured, named after our lock rather than its
    cause, and not asked for by any spec clause.

    (a) **Poison recovery at this site was sound, on a verified fact.** `ort`
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

    (b) **The residual panic surface was retained, not asserted away.** Inside the
    narrowed window every remaining panic site was an ort "C API violated its contract"
    assertion: `session/mod.rs:329`, and `Value::from_ptr`'s chain through
    `value/mod.rs:353` → `value/type.rs:384-394`/`:152`. None was a function of pixel
    content; none was provably unreachable, since a mis-built or ABI-mismatched
    `libonnxruntime` could trip them. The branch was therefore handled, never
    `unreachable!()`.

    (c) **`decode_outputs` was a public function over two independent slices, so it
    validated rather than indexed.** It rejected `values.len() != metas.len()` and any
    out-of-range bound index with `StageError::InvalidInput`. Adding a panic site while
    fixing a panic-poisoning bug would have been self-defeating. Being ungated, its test ran
    under plain `cargo test --workspace` — coverage the feature-gated path had never had.

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

    (e) **Mechanism replaced, invariant retained (2026-08-03; see §16.32).** The `Mutex<Session>` this item narrowed no longer exists: the session is owned by one dedicated worker thread, and preprocessing and decoding run on the calling thread, outside the worker entirely. The property this item ratified holds more strongly than when it was written — a panic in v1's own arithmetic cannot reach the session at all, rather than merely not poisoning it. What is withdrawn is only the lock, and with it the poison-recovery path and the "ONNX session mutex was poisoned" string; (a)'s standing duty to re-verify run_inner's receiver on any ort version bump is unchanged and now also covers the worker's catch_unwind/AssertUnwindSafe argument.

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
   hypothetical. **(ERRATUM, see §16.27 item 7 — read it before citing this clause. "This case"
   names two different branches, and only one of them is real: upstream's line-less *filter*
   branch fired once on the candidate page, while a line-less *output* block is impossible —
   no path through `group_output` appends one, re-read against the pinned checkout and probed
   directly. The degenerate identity branch is therefore a property of the artifact grammar,
   not a reachable upstream shape. The requirement to handle it stands unchanged; its "not
   hypothetical" weighting does not, and §16.27 item 8 corrects the same conflation at its
   second site.)** And the comparator must **assert the number of pairs it compared** against a
   hard-coded expected count: a comparator reporting "compared 0 boxes, 0 divergences → PASS"
   is cookbook rule 1's defect, and it is the failure mode a frozen-file comparison is most
   prone to, since a renamed field silently yields an empty pairing rather than an error.

   (c) **No tolerances and no IoU thresholds in any gating row.** Where a divergence has a
   known mechanism the evidence is an exact identity, not an epsilon. See item 5 for the
   measurement that forces this.

   (d) **`confidence`, `language` and `raw_mask` are DIAGNOSTIC or NO-ORACLE rows only,
   never gated.** **(QUALIFIED by §16.25 item 5 — read it before citing this clause. "Never gated"
   binds the field's *value*, which is what the prohibition is about: item 5 records that it is an
   **epistemic** limit, grounded in a measured 0.079–0.089 noise floor against 0.4 gates, so no
   tolerance exists that does not span "kept" and "dropped". It does **not** bind the field's
   *coverage* — whether the oracle artifact carries the field on some gated blocks and not others.
   Coverage has no noise floor and no epsilon; presence is exactly decidable, so gating it cannot
   import the defect this clause exists to prevent. `InconsistentOracleCoverage { field: "confidence"
   | "language" }` is therefore a **gating** row and does not contradict this clause: it says the
   partition cannot be completed, not that the values disagree. `raw_mask` is unaffected either way —
   it is not read per block, so it has no coverage state, and its NO-ORACLE bucket is settled
   unconditionally here and by §16.24 item 7.)** `raw_mask` has no exact oracle in v1 at all: upstream's preprocessor calls
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
   `DEVIATION(15)` made the detector session a single `Mutex<Session>` shared across the
   whole run (`crates/pc-detect/src/onnx.rs`), so image-level parallelism **could not reach the
   detector at all** — inference was serialized by the mutex *and* single-threaded inside it
   (the lock was replaced by a dedicated worker thread — §16.32; the serialization property this
   item describes is unchanged, only its mechanism.).
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

   **CLOSED, 2026-08-03 — see §16.32.** Root cause: CPU denormal/subnormal float stalls inside the ConvTranspose/Conv kernels — not kernel quality, not threading, and not a runtime version gap (an ort/ONNX Runtime version bump was measured and rejected as a regression, answering this item's version-bump candidate rather than deferring it). Setting ONNX Runtime's `session.set_denormal_as_zero` (`SessionBuilder::with_flush_to_zero()`) closes the gap: measured ~16.7-20s per page down to ~0.5-0.9s, this machine now faster than the recorded cv2.dnn baseline on the same page. Every hard constraint this item imposed held: model bytes untouched, no runtime replaced, the version bump was measured and rejected, and no recorded float moved — output is sha256-identical across intra_threads 0/1/8 and both flag states, and identical to the committed recorded fixture, so nothing here is fixture-affecting and no re-record/re-sign is triggered. The flag ships only bundled with the worker-thread confinement of §16.32 item 2(b) — the bundle, not the flag alone, is what the Fable ruling authorised. No new §14 register entry: §14 records deliberate divergences in what v1 produces, and a change measured to perturb zero output bits creates no such divergence (same shape as item 4's own reasoning for `intra_threads`).

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

   **The DirectML sentence above is RE-GROUNDED by §16.33 item 11 — read it before citing this
   paragraph.** Its argument ("the provider requires Windows") was vacuous only while Windows was
   not a target, and Windows becomes a supported platform at v1.1, so per §16.23 item 4's mirror
   rule it cannot carry forward unqualified. The **verdict is unchanged** — DirectML stays v2 —
   and so is the TensorRT/NVRTX sentence, whose reason is platform-independent. The replacement
   ground is item 2's non-CPU-provider carve-out plus the absence of any Windows GPU CI runner.
   Nothing else in this item changes: CUDA still ships at v1.5 opt-in, and CoreML, `.pt`/torch
   loading and multi-device dispatch stay v2.

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

7. **§16.20 item 3(d) is reaffirmed unchanged.** `raw_mask` stays NO-ORACLE. The upgrade path was documented as blocked on `MaskRefineMode::Annotation`; §16.39 later ratified and landed that opt-in path.

## 16.23 v1.5 scope and sequence (Fable tie-break, 2026-07-29)

1. **Both plans had scope omissions, in both directions.** The Opus Senior Rust Engineer
   correctly found three spec-assigned v1.5 items missing from the brief — LaMa inpainting,
   PSD/layered export, legacy INI config import — and its point stands that a "v1.5 plan"
   silently dropping LaMa and PSD is not one. But it then omitted **Lab-space coloured NLM +
   `color_filter_strength`**, also spec-assigned v1.5, which both plans forgot; and it treated
   dropping the font-rendering debug visualisations as free when they are a flat v1.5 item.
   **Silent dropping cuts both ways: a deferral needs ratifying by the same standard.**

   **The 2026-07-29 plan recorded these as v1.5 scope:** §16.21's threading erratum; F1 and its due prerequisites; CUDA per §16.22;
   LaMa inpainting; PSD/layered export; legacy INI import; Lab-space coloured NLM +
   `color_filter_strength` (landing it also retires §14 item 7's one-time WARN); DBNet line
   synthesis plus line-based splitting/merging and orientation/font-size estimation;
   `MaskRefineMode::Annotation` (subsequently landed under §16.39); and §16.21 item 6's bounded CPU-EP investigation.

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

2. **Historical sequencing (2026-07-29): `MaskRefineMode::Annotation` was held until F1 recorded.** Requested as a way to
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
   remains the default**, so default-path fixture churn for that task is zero. §16.39 subsequently landed the path.

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
   **Annotation** (historical final step; §16.39 subsequently landed it).
   **SUPERSEDED in part by §16.38 item 15 — read it before citing this sequence.** LaMa moves ahead of GPU-2 and ahead of the batched INI import + Lab NLM; PSD, DBNet lines and Annotation keep their places, and Annotation is still last.
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

   **(AMENDED by §16.27 item 3 — read it before treating the mechanism list in the sentence above as
   closed. A fifth alternative, `DbnetScattered`, is inserted there: an upstream block built from
   unassigned DBNet line polygons carries no yolo box, can never be paired, and had no legal
   mechanism under the list as written, so a page carrying one had only `Open` — a blocking
   violation — available for a block that is not a defect. `Open` stays blocking and nothing else in
   this item moves.)**

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
   item 3(d) with item 10 — **`NoOracle` is SUPERSEDED by §16.25, which deletes the class and the
   `NoOracleField` variant it existed for; the surviving classes are `Gating` and `Diagnostic`, and
   what `NoOracle` was carrying moved to `ComparisonReport::coverage` as an artifact property. The
   original three-class wording is preserved here per §16.19's convention; §16.25 items 2 and 4 are
   the reasoning, and item 8 records that deleting it amends this ratified enumeration rather than
   following from it.** The rest of this item stands unchanged: deterministic emission order; **no computed floats in any variant**, so
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

   **PARTLY SUPERSEDED by §16.25. THREE claims above are narrowed or withdrawn — all three, not
   just the identifier.** An earlier version of this marker withdrew only the `NoOracleField` phrase
   and left the two sentences after it standing, which is the same under-marking defect one level
   deeper: the strongest claims were the ones left unqualified.

   (i) *"with `NoOracleField` rows when absent"* — **WITHDRAWN.** The variant is deleted. §16.25 item
   3 records why quoting it here was circular: the identifier occurs in this spec exactly once, right
   here, in CamelCase taken from the draft this section was ruling on, so this was a ruling repeating
   the draft's vocabulary while deciding a different question. Whole-field absence is now
   `OracleCoverage::Absent` on `ComparisonReport::coverage`.

   (ii) *"Either outcome is a closed verdict under item 3(e); neither blocks the fixture."* —
   **NARROWED to the two outcomes this item actually enumerates**, captured → `DIAGNOSTIC` and
   not-captured → `NO-ORACLE`. Neither of *those* blocks, and that still holds. But this item did not
   contemplate a **third** state: capture that succeeds on some gated blocks and fails on others.
   `Partial` is neither outcome, its partition row cannot close, and per §16.25 item 5 it **does
   block** under §16.20 item 3(e). Read as a claim about confidence in general — "nothing about this
   field ever blocks" — the sentence is now false, which is why it is narrowed rather than left to a
   reader's charity.

   (iii) *"The row is never gated regardless."* — **NARROWED: it binds the VALUE row.** The value is
   never gated, for the epistemic reason §16.20 item 3(d) gives (a measured 0.079–0.089 noise floor
   against 0.4 gates, so no tolerance avoids spanning kept-and-dropped). **Coverage is a different
   row on a different subject**, it has no noise floor and no epsilon, and
   `InconsistentOracleCoverage { field: "confidence" }` **is gating**. "Regardless" was doing more
   work than the evidence supports.

   **What survives untouched, and is independently grounded:** the DIAGNOSTIC-else-NO-ORACLE branch
   itself; the obligation on the script to *attempt* capture; `confidence: Option<f64>` in the oracle
   schema (grounded in cookbook rule 7's finding that upstream's `#raw.json` does not persist
   confidence, not in this section); and *"may not close as `OPEN` on this account"* — that clause
   forecloses `OPEN` for whole-document **absence** and, per §16.25 item 5, never for
   **inconsistency**, which is precisely why `Partial` blocks without contradicting it. Original
   wording preserved per §16.19's convention.

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

    **(Pointer, not a claim — read §16.27 item 3(b) before treating the upstream sum above as
    complete. A fourth upstream mechanism term, `dbnet_scattered`, is added there. That entry's marker
    sits at item 20 instead, because item 20(c) states in its own text that it restates these two
    equations; no claim is declared against this item, since the ruling being transcribed names item 4
    and item 20(c) only. Widening it to this item is a separate step, flagged there rather than
    performed.)**

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

    **(Partly superseded by §16.27 — read it before citing 18(b), 18(i) or 18(k). Three sub-items are
    corrected there and the rest of this item stands. 18(b)'s leg-2 law is scoped to upstream blocks
    that have a `rect_yolo` at all, since blocks built from unassigned DBNet line polygons have none
    and the law is undefined for them — §16.27 item 4, which also records that the finding does not
    bite the ratified oracle page itself. 18(i)'s diagnostic message becomes derivation-conditional,
    because a bounding union is not what produced the upstream box on a split or scattered block —
    §16.27 item 6. 18(k)'s "rare-but-real" conflates upstream's line-less *filter* branch, which is
    real, with a line-less *output* block, which no path through `group_output` can emit — §16.27
    item 8. Two things are deliberately NOT touched: 18(h) is not reopened, because §16.27 item 2
    gates a different quantity (`ours.rect - rect_yolo`) and leaves `residual_leg1` diagnostic
    exactly as (h) leaves it; and 18(d) is not corrected at all — §16.27 item 10 records the
    re-verification that found its fixture attribution right as written.)**

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

    (c) **The enumeration, recorded (2026-07-29) — CORRECTED, because the first version of this very
    clause was itself incomplete.** Readers and writers of `PROVENANCE.json`:

    | file | role |
    |---|---|
    | `crates/pc-testkit/tests/provenance_schema.rs` | reader, typed schema |
    | `xtask/tests/provenance_digests.rs` | reader, typed schema |
    | `crates/pc-testkit/tests/recorded_provenance.rs` | reader, generic key walker (frozen) |
    | `crates/pc-testkit/tests/model_signature.rs` | reader — **was** hand-indexed, now typed |
    | **`xtask/src/calibrate.rs`** | reader — **MISSED by the first enumeration**, see below |
    | `xtask/src/model_signature.rs` | writer, now through `pc_testkit::provenance` |
    | `xtask/src/record.rs` | writer, now through `pc_testkit::provenance` |
    | `xtask/src/ocr_model_signature.rs` | reader/writer, now through `pc_testkit::provenance` |

    One-line note for the `xtask/src/ocr_model_signature.rs` row: found 2026-08-04 in §16.32 by
    re-running this item's own prescribed grep untruncated.

    Any future reader added to this table carries item 1(a)'s gate with it.

    **How the first enumeration failed, recorded because the method is the lesson.** I ran exactly the
    grep this item prescribes, piped it through `head -40`, and `xtask/src/calibrate.rs` — five hits —
    fell past the cut. I then wrote *"I have now enumerated all of them and there is no third"* into a
    normative clause. **The grep was right; the truncation was mine; and the claim of completeness was
    made on a truncated result.** Cookbook rule 14 gains this as its own failure mode: an enumeration
    that is filtered, headed, or eyeballed is not an enumeration, and "a one-line grep is the whole
    cost" only holds if you read all of its output.

    **What it cost.** `calibrate.rs` read `parsed["jpeg_decode_diagnostic"]["metrics_vs_reference"]`,
    a shape the migration replaced with the group's `diagnostics` map. `serde_json`'s index returns
    `Null` for a missing key, and the surrounding code skipped the row when null — so
    `cargo xtask calibrate-goldens` would have **silently dropped the decoder-diagnostic row from a
    generated document** rather than failing. Verified by probe: with the old lookup against migrated
    provenance the row count is 0; with the fix, 1. It was latent only because the committed
    `docs/GOLDEN_CALIBRATION.md` predates the migration — the next person to regenerate it would have
    lost the numbers with no error. Fixed to parse through `GroupProvenance`, and a present-but-
    unreadable provenance now writes an **UNREADABLE** row into the document instead of omitting it,
    because silent omission is precisely what hid this.

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

    **(AMENDED by §16.27 item 3(b) — read it before treating this table, or the two equations, as
    complete. One upstream row is added there, `dbnet_scattered` for the `DbnetScattered` mechanism,
    because §16.27 item 3(a) adds a mechanism class and this item's own partition rule allows exactly
    one term per class. The added term is derived from the signed entry list like every other
    mechanism term, so the authored anti-vacuity literals of (b) are untouched. Its unsuffixed
    spelling follows the `class_duplicates` row above, the mechanism having no ours-side
    counterpart.)**

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

    (f) **Sign precision on item 18(h)(iii), because rule 15 is literally about signs.**
    **(NARROWED by §16.27 item 5 — read it before citing the attribution below. It is true of pairs
    whose upstream derivation is `YoloUnioned` and silent otherwise: outside that derivation the
    upstream box is not a bounding union of a yolo box with lines, so the difference this clause reads
    signs from carries no information about leg 1. The recorded counterexample has all four components
    in union-impossible directions with leg 1 exact. The signs themselves are not in question, and
    §16.27 item 2 is what replaces the attribution with an exact row.)** The
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

## 16.25 `NoOracleField` is replaced by per-field coverage (cookbook rule 8 exit 2, joint architects, 2026-07-29)

An **amendment**, not a drafting correction. Both Opus architects concur; no tie-break was needed.
Two of F1-C's drafted tests change their expected vectors, and by rule 8's forcedness test neither
reading was forced — the Senior Rust Engineer's own draft contained both — so this is exit 2 and
required a ratification before implementation.

1. **The defect: `NoOracleField` was unsatisfiable, and this is a proof rather than a judgment.**
   The variant was a per-pair `Divergence` in class `NoOracle`, emitted when an upstream block lacks
   `confidence`/`language`. Two drafted sites expected **two** such rows on a `bare_block` pair;
   five expected **zero** on the same input shape. The Technical Architect closed the diagnosis by
   finding two sites that are **byte-identical in every field** — same `Rect::new(100,200,160,240)`,
   same `class_index: 0`, same `confidence: 0.775`, same `bare_block([100,200,160,240])`, same
   `LineLess` branch, same exact geometry — and expect opposite results. Any function of
   `(ours_block, upstream_block, branch, geometry)` returns the same value at both, so **no
   pair-local predicate can satisfy both.** The unrelated clauses an implementer produced
   (`unmatched_*.is_empty()`, branch, identity-held) were not poor judgment; they were the only
   remaining degrees of freedom. Recorded because "the delegate wrote a curve-fit" and "the
   specification it was given was contradictory" call for different responses.

2. **`NoOracleField` is a category error, and the general rule is worth more than the fix.**
   `divergences` lists findings about the *comparison*; "upstream did not record confidence" is a
   property of one *input document*. Nothing diverges — one side simply has no data — and that is
   what made a vector length depend on block count. The report therefore has **three channels, not
   one**: *findings* (`divergences`), *per-pair measurements* (`residuals`, `branches`), and
   *artifact properties* (`coverage`). The middle channel was already correct precedent —
   `residual_leg1` is non-zero in the healthy case and lives outside `divergences` for exactly the
   reason coverage must. **Discriminator for future variants: does the row's truth depend on the
   relation between two sides, or on one artifact alone?** All 15 remaining variants were walked
   against it and are correctly filed; `EmptyPairing` is the near-miss and stays, because its
   content is "no comparison happened" — a fact about the comparison, O(1), carrying the one bit
   §16.20 item 3(b) requires be raised structurally. `NoOracleField` failed all three tests: it was
   one-sided, O(N), and carried zero bits beyond the first. Its own shape shows it — it keyed an
   upstream-only fact by **our** rect.

3. **ERRATUM: §16.24 item 9's `NoOracleField` phrase cannot be cited as authority for the per-pair
   form, and this was verified.** The identifier occurs in the entire spec **exactly once**, at
   §16.24 item 9, in CamelCase Rust, in a document that otherwise spells the concept `NO-ORACLE` in
   five places, all partition-bucket usages. §16.24 is the same-day tie-break over the two F1 plans:
   the phrase is a ruling repeating the draft's own vocabulary while deciding a *different* question
   (whether the row may close `OPEN`). Citing it would be §16.20 item 3(e)'s prohibition in a
   different costume — closing a design against a ratification that merely quotes the design.
   **What survives untouched:** item 9's other half, "the oracle schema keeps
   `confidence: Option<f64>`", is independently grounded in cookbook rule 7's finding that upstream's
   `#raw.json` does not persist confidence. `OracleBlock.confidence: Option<f32>` stays exactly as
   drafted, so the serde round-trip gate and its full-shape literal are unaffected.

4. **The replacement.** `NoOracleField` and `DivergenceClass::NoOracle` are **deleted**. The report
   gains

   ```rust
   pub enum OracleCoverage { Present, Absent, Partial { with: usize, without: usize } }
   pub struct FieldCoverage { pub confidence: OracleCoverage, pub language: OracleCoverage }
   ```

   Grounds, each on ratified text rather than preference: it is the **computable witness for item 9's
   actual decision procedure**, which is a per-field, per-document branch ("DIAGNOSTIC if the script
   captures upstream's scores, else NO-ORACLE") — N per-pair rows answer that question N times and
   only for paired blocks; a **mandatory struct field cannot be omitted where an optional row can**,
   which is §16.24 item 20(b)'s own ratified preference *"unrepresentability beats detection"* applied
   one level over, and the curve-fit's concrete failure was that `divergences` **silently lacked** the
   rows on any realistic page; and **"row" in §16.20 items 3(d) and 3(e) means a table row in
   `DETECTOR_ORACLE.md`**, one per serialized field, never a per-block emission.

5. **`Partial` is GATING, derived from item 3(e) rather than as an exception to item 3(d).**

   **RE-GROUNDED (joint architects, 2026-07-29). The original derivation below named §16.24 item 9
   as the premise when item 9 is only an *instance* of it, and that was the defect** — not, as first
   suspected, a confidence rule wrongly applied to two fields. The load-bearing ground is **§16.20
   item 3(e) alone**: a partition row must close in exactly one bucket; `DIAGNOSTIC` asserts an
   oracle exists for the field across the compared set and `NO-ORACLE` asserts none exists;
   `Partial` satisfies **neither description of the artifact**, so the row cannot close; `OPEN` is a
   blocking state; the fixture commit blocks. Every step is **field-agnostic** — it never mentions
   capture, scores, or item 9 — and it follows from the partition being over *fields* while a bucket
   is a property of the *whole field*: a field present on some gated blocks and absent on others has
   no bucket, **whatever made it absent**.

   That disposes of the objection that "there is no capture step for `language`". True, and
   irrelevant: it is a claim about the *mechanism* of absence, and item 3(e) does not ask. This is
   §16.25's own move applied one notch further — coverage is a different subject from the field's
   value, and the *reason* for absence is a different subject again from whether the row closes.
   Recorded because the objection is the natural one and it reads as decisive until the premise is
   located. Its true observation survives as a note pointing the other way: if every detected
   upstream block necessarily carries a class index, partial `language` is a **stronger** indictment
   of the recording than partial `confidence`, because no legitimate per-block mechanism explains it.

   **Scope, as a two-condition test a future reader runs on a new field.** The rule applies iff
   **both**: (i) the comparator reads the field **per gated upstream block as an `Option` from the
   oracle artifact**, so coverage is defined at all; **and** (ii) the field carries a §16.20 item
   3(e) **partition row**, so an unresolvable bucket is a blocking state. Today that is exactly
   `confidence` and `language`, which is why `ComparisonReport` carries an `OracleCoverage` for those
   two and no others. Both conditions are load-bearing and each excludes a real candidate:

   - **`raw_mask` fails (i).** It is not in `OracleBlock`, the comparator never reads it, and there
     is no per-block optionality to be partial about. Its bucket is settled unconditionally by item
     3(d) as reaffirmed by §16.24 item 7. Rest the exclusion on **(i)**, not on 3(d) settling the
     bucket — otherwise a future field that 3(d) happens to mention becomes ambiguous.
   - **`base_xyxy_pretruncation` fails (ii).** It *is* a per-block `Option` read from the artifact,
     so (i) alone would sweep it in. It carries no partition row — an upstream-only probe, not a
     serialized field of `PageDataRaw`/`DetectedBlock`/`RawBlock` — and its absence is already
     per-pair visible in `PairResidual`. This is the asymmetry refinement (c) required be stated
     deliberately; the **conjunction** is what makes it derivable rather than remembered.

   §16.24 item 9's clause that the row "may not close as `OPEN` **on this account**" is unaffected
   either way: it forecloses `OPEN` for whole-document *absence*, never for *inconsistency*.

   **The implementation was already correct** (`crates/pc-detect/src/oracle.rs`, the loop gating both
   fields) and does not change; the conclusion did not move, only its grounds. **One control is
   still owed before the recording run**, additive under rule 8 exit 1, because the
   `language`-`Partial` path is currently unexercised — the existing test asserts
   `coverage.language == Present`: a language-partial/confidence-`Present` case asserting exactly one
   gating row with `field: "language"` **and** that the covered block's `LanguageDelta` is still
   emitted (mirroring the confidence half's subset-reporting proof, without which the control proves
   blocking but not reporting); plus a both-partial case asserting **two** rows with the order pinned
   (`confidence` before `language`), since emission order is contractual and an unasserted order is
   an unpinned contract.

   The original derivation, preserved per §16.19's convention: *`Partial` ⟹ "the script captured
   upstream's scores" is neither true nor false of the document ⟹ item 9's two-way branch does not
   resolve ⟹ the `confidence` partition row is `OPEN` ⟹ §16.20 item 3(e) makes it blocking.* The
   gating row is therefore **not a verdict about the field** — it is the report stating that the
   partition cannot be completed.

   Item 9's clause "may not close as `OPEN` **on this account**" does not reach it: that account is
   upstream not persisting confidence. Partial capture is a different account — **our** recording
   script producing an internally inconsistent artifact. Two independent supports: item 3(d)'s
   prohibition is **epistemic**, grounded in item 5's *measured* 0.079–0.089 noise floor against 0.4
   gates, and coverage has no noise floor, no tolerance and no epsilon (presence is exactly
   decidable), so gating it cannot import the defect 3(d) exists to prevent — no green run turns red
   because two engines rounded differently. And coverage is a **precondition** in the same sense
   item 8 gates `scale`/`image_size`: those pin the two sides to one coordinate frame, coverage pins
   them to one *compared set*.

   `InconsistentOracleCoverage { field, with, without, missing }` is routed through `divergences` as a
   gating variant, **not only** onto the report struct, so the CI contract stays one channel:
   `report.gating().is_empty()` AND `pairs_compared == <literal>`, with no second thing a caller must
   remember to check. `missing` carries the indices, not just counts — counts send a reader hunting,
   indices are a fix (rule 13's corollary).

   **SUPERSEDES: §16.20 item 3(d)**

6. **Three binding refinements — conditions of the concurrence, not preferences.**

   (a) **Compute the gating subject over the PAIRED upstream blocks**, reporting whole-artifact
   counts as diagnostic numbers. The defect guarded is "the DIAGNOSTIC comparison silently covers a
   subset", and the compared set *is* the paired set: an unmatched block contributes no
   `ConfidenceDelta` whether or not it carries a score, and it already has its own row and
   `Mechanism`. `pre_filter_blocks` stay excluded — item 8 makes them DIAGNOSTIC and pre-`group_output`.

   (b) **Define the empty case: `with == 0 && without == 0` is `Absent`**, never `Partial` and never
   `Present`. `Present` over an empty set is a vacuous truth of exactly the shape cookbook rule 1
   hunts. Two live test sites reach it today (`blocks: vec![]`), both already gating on
   `EmptyPairing`/`AccountingMismatch`, so nothing rests on the choice — but leaving it undefined
   means the first implementer picks it invisibly.

   (c) **Emission position is contractual:** `InconsistentOracleCoverage` is emitted **immediately
   after the page-level frame rows and before any block row**. It is a document property like
   `scale`, and a caveat that the rows below cover a subset must precede the rows it qualifies. The
   frozen exact vectors depend on this, so it lands in the ratification rather than in the
   implementation. State also why `base_xyxy_pretruncation` gets no coverage field — it is per-pair
   visible in `PairResidual` and carries no partition row — or the asymmetry reads as an oversight.

7. **Consequences, enumerated so nobody re-derives them. THREE tests change, not two.**

   | site | after | strength |
   |---|---|---|
   | `the_line_less_identity_degenerates_to_plain_equality` | exact vector `Vec::new()`; `coverage.{confidence,language} == Absent` | **equal** — same content, unconditional rather than per-block |
   | `confidence_and_language_..._absence_is_no_oracle` | exact coverage equality replaces `no_oracle().len() == 2` | **stronger** — a bare `len()` among exact-vector siblings is rule 13's "cardinality is not identity" |
   | `the_divergence_class_partition_is_total_...` | sample swaps `NoOracleField` for `InconsistentOracleCoverage`; `(7,2,1)` → `(8,2)`, total still 10 | **equal** |

   Plus one new `Partial` test — the `Partial` artifact is structurally sound, so it belongs in the
   exact-whole-vector family per §16.24 item 21(e), not the containment family. **26 → 27.** No change
   claims less about the system and one claims more, so concurrence plus this entry is the whole gate.

   Two additive strengthenings to fold in while these sites are open (rule 8 exit 1, no ratification
   needed): the `Partial` test must **also** assert the *with*-block's `ConfidenceDelta` is still
   emitted — that is what proves the subset is reported rather than suppressed, the exact confusion
   that produced the curve-fit; and assert `coverage == Present` in the existing agreeing/disagreeing
   halves so `Absent` is falsifiable against `Present` within one test rather than only across tests.

   **SUPERSEDES: §16.24 item 6**

8. **§16.24 item 6's three-class list is edited by this entry, and says so.** That item adopts *"the
   divergence class system (`Gating` / `Diagnostic` / `NoOracle`)"* verbatim. Deleting the class is
   therefore not a drafting consequence — it amends a ratified enumeration, superseded in place per
   §16.19's convention with the original wording preserved. The class must **go**, not be kept empty:
   a `no_oracle()` accessor that provably always returns empty is rule 1's decoration with a name
   claiming a capability.

   **SUPERSEDES: §16.24 item 9**

9. **Corrections of record.** Two citations in the defect table as first circulated were wrong, and
   the corrected list is normative here: `:262` (`the_unperturbed_pair_...`) uses `oracle_block()`,
   which sets both optional fields on all five blocks, so it never exercises absence at all — the
   five zero-row sites are `:639`, `:797`, `:1071`, `:1209`, `:1315`. And **`:797` is
   line-INFORMED** with both fields `None`, so the contradiction is not "line-less vs line-informed"
   but "any pair whose upstream block lacks the field" — that site is the sole reason the curve-fit
   needed its `LineLess` clause, and it sharpens the diagnosis to span branches rather than states.

10. **One unverified upstream fact, named rather than assumed.**
    **(AMENDED by §16.27 item 3(c) — read it before citing this item. The check came back the other
    way, so this item's own contingency fires rather than a fresh judgment being made: upstream does
    construct such blocks, measured on two of the three recorded page candidates against the pinned
    checkout. The presupposition below is therefore false, and the obligation in the closing sentence
    becomes "one score per gated yolo-derived block". What is NOT affected: this item's `Partial`
    conclusion stands on its own ground — item 6(a) computes coverage over the paired set, and a block
    with no yolo box is never paired, so `Partial` still cannot arise from this shape.)**
    Whether upstream's `group_output`
    can construct a post-filter `TextBlock` from unassigned line polygons — which would have no yolo
    box and hence legitimately no confidence — was **not** checkable in this environment (no upstream
    checkout) and the spec does not record it. §16.24 item 18(b) already presupposes the answer is
    no: leg 2 is `upstream.xyxy == bbox(upstream.rect_yolo ∪ bbox(upstream.lines))`, undefined for a
    block with no `rect_yolo`. So under the ratified identity every gated upstream block descends
    from a yolo box and `Partial` is unreachable except as a recording defect. **If the check comes
    back the other way, the finding is NOT "Partial should be diagnostic"** — it is that item 18(b)'s
    leg-2 law needs an erratum, which is larger than this entry. Recorded as a recording-script
    obligation: the script asserts one score per gated block.

11. **Recorded, not fixed.** `LanguageDelta.upstream_language: Option<Language>` is `Some` at its only
    emission site, so the `Option` is decoration — rule 1's family. Fixing it would edit two frozen
    exact vectors for zero gate power; left as known. And
    `the_divergence_class_partition_is_total_...` covers **10 of 16** variants while its *name* claims
    totality — rule 1's dominant defect class sitting in the test whose job is to prevent it. Making
    the sample exhaustive is free and additive (rule 8 exit 1) and should be taken while the file is open.

12. **Process note: `docs/COOKBOOK.md` rule 8's category-error paragraph was committed in `31d483e`,
    before either architect ruled.** It asserts this entry's conclusion. Now that both concur the
    record is consistent, but had the redesign been rejected the cookbook would have needed
    correcting — writing a lesson ahead of the ratification it depends on is its own small defect,
    and the paragraph carries an anchor to this section so a future reader sees which came first.

## 16.26 The supersession marker becomes machine-readable (convention, 2026-07-29)

What this spec cites as "§16.19's convention" — the old wording preserved verbatim, with a marker at
the old site — was right and unenforced. It failed three times in the F1 sequence
(§16.20 item 3(d), §16.24 item 6, §16.24 item 9): each time the new entry stated what it
amended, the old site stayed bare, and the old site is the one a future reader lands on.
§16.19 itself left its one target, §16.12 item 2, bare — that row is in the pinned set below.
This entry makes the convention checkable.

1. **Measured first, and the measurement changed the design.** A prose scan over the verbs
   (`SUPERSEDED`, `amended`, `ERRATUM`, `QUALIFIED`, `withdrawn`, `NARROWED`, `RE-GROUNDED`, plus
   `supersedes` and case variants — the list is abbreviated, and a re-measurement with only the
   seven printed verbs will not reproduce 51) finds
   51 occurrences across 91 sections and 343 items, yielding 26 candidate claims after a direction
   rule (a *claim* names an older target; a *back-pointer* names a newer claimer) plus `by`- and
   parenthesised-marker filters. Of the 26, **18** name a target carrying no back-pointer;
   hand-auditing those 18: **~10 real, ~8 false positives.** **Four** structural causes, each a real
   line of this spec:

   - **Quoted anchors.** §16.24 item 10 says *"§16.20 item 3(e)'s citation `"§2.4/§2.5's field
     list"` is wrong"* — the target is 3(e); §2.4/§2.5 are quoted content. Four flags.
   - **Slash-lists defeat a `by` filter.** In *"superseded by §10.5/§13's crate column"*, `by`
     attaches to the list, so §13 survives as a phantom claim.
   - **Citing someone else's amendment.** *"per §16.20 item 3(a) as amended by §16.22 item 6"*,
     sitting in §16.21, which claims nothing.
   - **A verb and an unrelated anchor sharing one physical line.** Line 741 carries a real
     supersession clause naming §15.10 *and*, later on that same line — a single unwrapped line of
     ~1100 characters, so no wrapping is needed for the hazard — an unrelated citation of the
     `ReplayDetector` subsection, so the scan pairs the verb with the second anchor and invents a
     claim on a target the clause never mentions. **Added 2026-07-30, recording a cause found on
     2026-07-29 by the enforcing test rather than by the audit that produced the three above**,
     which is the entry's own subject turned on itself and the reason item 8 exists.

   Every additional heuristic moved flags between the two error columns rather than reducing them.
   **A gate wrong half the time gets allowlisted into uselessness** — cookbook rule 13's bypass by
   another road, and worse than no gate because it looks like coverage.

2. **DECIDED: two layers, because enforcement and discovery are different jobs.**

   **Layer A — enforcement.** An explicit marker, written by the author at the moment they know the
   target:

   ```
   **SUPERSEDES: §X item N**
   ```

   It carries no verb list, no distance heuristic and **zero false positives**, because the author
   declares the anchor instead of a parser inferring it. Layer A is what the gate enforces: for
   every marker, the named target's own span must contain a pointer back to the claiming section.
   Mentions of the marker in spec prose — including this example — use placeholder anchors (`§X`),
   which the parser does not resolve; the syntax is live everywhere in this file, code fences
   included. A gate that could not be discussed inside the document it guards would be its own
   worst clause, and this example carried a real anchor until the step-1a review caught it
   (item 7).

   **Layer B — tripwire.** The prose scan survives, pinned as a **28-row verbatim set** (26 measured
   pre-convention pairs, plus the two rows item 6 records for this entry's own quotations), and its
   only job is to fail when a *new* claim is written in prose rather than with a marker. That
   absorbs today's ambiguity without anyone adjudicating the eight false positives now.

3. **Span delimitation, with its leniencies named rather than pattern-matched.** An item's span
   runs from its `N. ` marker to the next `^N. ` in the same section. **Four** deliberate
   leniencies, enumerated because an unenumerated leniency is how a gate becomes vacuous:

   (a) sub-items are **not** separate spans (a claim on `item 3(d)` resolves to item 3, because
   sub-items have no reliable terminator and the real back-pointers sit inside the parent item);
   (b) `step N` anchors fall back to the **whole section**, because §8.3 has no parseable items;
   (c) a bare `§N` target with no item scopes to the whole section — the only available reading —
   and must still resolve (item 4). This case is not decoration: six or more of the pinned rows
   are bare-section targets, so an implementer hits it immediately.

   (d) **A three-component sub-section anchor resolves to its two-component parent's span, but
   ONLY IF the literal sub-section token appears as a bold heading inside that span** (Fable
   tie-break, 2026-07-29). This mirrors (a)'s sub-item leniency with one deliberate asymmetry —
   **the existence check** — and the asymmetry is the whole point: silently truncating a dotted
   *section* number converts a nonexistent anchor into an existing one, which is exactly the
   bypass item 4 ratifies against. So `§7.2.1` resolves, because
   ``**7.2.1 `ReplayDetector` binding and artifact format…**`` is a real bold heading inside §7.2
   (elided tail: `(resolved during D3 test-drafting).`); and **`§7.2.999` still fails**, because no
   such token exists in §7.2's span.

   Ruled after the enforcing test failed on the pinned row `("8.7", "7.2.1")` and an agent tried
   to pass it by truncating every dotted anchor to two components — which turned green all seven
   assertions then present and would have resolved `§7.2.999` too. That truncation was reverted.
   **Teaching the outline parser a bold-heading section class is REFUSED**: there is exactly one
   such subsection in this file, it has no reliable terminator, and this file carries **32**
   `**N — Title.**` item markers — 7 of them in §8.3, the rest in §9.3, §11.3 and §12.3 — every one
   a near-miss waiting to be misparsed into a span. (The file-wide 32, not §8.3's 7, is the figure
   that bears on a parser-class decision, since the parser would see all of them.)

   All four are bounded and enumerated. Getting this lenient makes the gate vacuous — a pointer
   anywhere in the file would count — and getting it strict produces false failures on
   multi-paragraph items, which is why it is ratified rather than left to the implementation.

4. **Three prohibitions, asserted rather than commented.** A pointer elsewhere in the file does not
   count (the check is span-scoped, proven on synthetic text in both directions); **an unresolvable
   anchor is a FAILURE, not a skip** — the drafting measurement script itself had
   `if target not in spans: continue`, which is exactly the bypass this gate exists to prevent; and
   the count of **parsed claims** is **pinned**, so the gate cannot pass by finding zero markers.
   That count is deliberately *not* a count of marker-literal occurrences: item 2's placeholder
   example is a fourth occurrence that yields no claim, so the two numbers differ by one, and a
   dedicated control asserts the placeholder is what the parser rejects rather than leaving the
   difference to be inferred from the count. (This clause names the literal in words rather than
   writing it, because writing it here would make this line a fourth Layer A site — item 8's hazard
   applied to the very item that pins the count.)

5. **Allowlist is a RATCHET, not a retroactive pass.** The pinned prose pairs are asserted as an
   exact set — additions and removals both fail —
   migrating a claim to the marker form updates the constant in the same commit. The ~10 genuine
   back-pointers still missing require writing normative text into §8.3, §11.1, §13 and others, and
   adjudicating 18 flags first; that is architects' work and a **separate task**, not a side effect
   of drafting a test.

6. **Bootstrapping, stated because the gate refuses to be vacuous.** Layer A starts at zero markers,
   so its pinned marker count (item 4's third prohibition) fails `0 ≠ 3` until three purely additive
   markers of the form given in item 2 land in §16.25 items 5, 7 and 8, targeting §16.20 item 3(d),
   §16.24 item 6 and §16.24 item 9 — the sites whose back-pointers already exist and are green.
   (That sentence is deliberately worded to keep the literal verb off a line carrying a live anchor;
   see item 8.) That failure
   **is the gate working.** Two of item 1's verbatim quotes above are themselves claim-shaped to the
   Layer B scanner — quoted-anchor false positives of exactly the class item 1 names — and are
   pinned in Layer B's constant as `(16.26, 13)` and `(16.26, 16.20 item 3)`, commented as
   quotations. Rewording a real quote to dodge the scanner would be worse than pinning it: item 1's
   whole virtue is that each example is a real line of this spec. The gate lives at
   `crates/pc-testkit/tests/spec_supersession.rs`; a fourth claim raises the pinned count and
   extends the ratified-set constant in the same commit as its marker and back-pointer.
   Consequence: the test and the first markers land in one commit, and this entry is the
   ratification the marker syntax needed before appearing in normative text — required by
   `CLAUDE.md` step 1a, which was added in the same session and whose first effect was to stop this
   from being self-approved.

   **STATUS, 2026-07-30: THIS CONVENTION IS NOW ENFORCED.** The gate lives at
   `crates/pc-testkit/tests/spec_supersession.rs` and landed in the same commit as the first three
   markers, as the paragraph above requires. Every "the gate enforces" sentence above may now be
   read in the present tense.

   What had parked it was the **fourth span rule**, which item 3 reserved for ratification rather
   than leaving to the implementation: the test failed on the pinned row `("8.7", "7.2.1")` because
   `§7.2.1` is a real subsection carried by a **bold-text** heading rather than a `###` heading, and
   so is invisible to the outline parser. Two readings were open; item 3(d) records the tie-break
   that closed them. Its scope, quoted rather than paraphrased: a three-component anchor resolves to
   its two-component parent's span "ONLY IF the literal sub-section token appears as a bold heading
   inside that span". Teaching the outline parser a bold-heading section class was **REFUSED**, and
   `nonexistent_three_component_anchor_does_not_resolve` is the control that keeps `§7.2.999`
   failing.

   **The parked interval is recorded rather than erased, because the alternative is the exact defect
   this section exists to correct.** For one commit this entry asserted an enforcement that did not
   exist — an unenforced convention, the class item 1 hand-audited to **~10 real** instances (18
   flagged, drawn from 26 candidates, drawn in turn from 51 raw verb occurrences; the 51 counts
   verbs, not unenforced conventions, and attaching the largest number in that chain to the
   narrowest noun overstates it roughly fivefold) — and it would have been invisible to a reader who
   trusted the prose. It was caught by a stop-time review, not by the author, which is item 7's
   lesson recurring one commit later: **the transcription of a ratification needs a reader even when
   the ratification was itself reviewed.**

7. **The step-1a review that this entry is the first subject of, recorded because its findings
   changed the entry.** Reviewer: Fable, as fresh reader, 2026-07-29; method: §16.26 compared line
   by line against the engineer's draft, with every number re-derived and every quoted example
   re-checked against the spec line it cites. Six edits resulted, and one was **blocking**: the
   Layer A example at item 2 originally carried a *real* anchor, and since it is the file's only
   such token the gate's own ratification entry would have failed the gate three ways — no
   back-pointer at the named target, a marker count of 4 ≠ 3, and an extra row in the ratified set.
   Two number defects of the historical class were also corrected: "failed five times" (the source
   supports **three** marker failures; the other two of `CLAUDE.md`'s five defects are scope
   over-generalisations, not bare old sites) and the `~10 real / ~8 false` denominator (**18**, the
   candidates whose target carries no back-pointer — not the 26 candidates, and 8/18 ≈ 44% is what
   item 1's "wrong half the time" rests on). The attribution in the preamble was also narrowed:
   §16.19 is *cited* for this convention rather than having established it. Recorded because the
   defects were the exact class step 1a was created to catch, on its first use, in an entry whose
   own subject is unenforced conventions — the cheapest available evidence that the step is not
   ceremony.

8. **The scanner is line-based, so line breaks in this file are semantic — found by walking into it
   while applying item 7's edits.** Layer B pairs a verb with the anchors on the *same line*, so a
   supersession verb and a live `§N` anchor sharing one wrapped line produce a claim, and moving a
   line break can create or destroy one. This is not hypothetical: the replacement text for item 6
   was wrapped with the literal verb beside three live anchors, manufacturing three unpinned phantom
   rows, where the wording it replaced had kept them on separate lines by accident. Two
   consequences. (a) **Editing prose near a supersession verb requires re-running the scan**, which
   is why item 7's review is recorded as a method and not a signature — the check is mechanical and
   nobody's care substitutes for it. (b) The one-sentence rule for authors: *keep a supersession
   verb and a live section anchor off the same physical line unless you intend a claim*, and if you
   do intend one, write the marker from item 2 instead. Recorded because a reader who reformats this
   file — an editor re-wrapping a paragraph, a tool normalising line length — can turn the gate red
   or green without touching a word, and would otherwise have no warning.

9. **The two layers are disjoint by construction, and the disjointness is scoped to the declaration
   rather than to the line.** Both layers read the same physical line (item 8), and Layer A's marker
   literal differs from one of Layer B's verbs only in letter case. That coincidence is not
   load-bearing: before scanning, Layer B **masks each marker's declaration span** — the literal
   through its closing `**` — so a marker's own verb and its own declared anchors are invisible to
   Layer B by construction. Adding an upper-case variant to the verb list therefore cannot turn
   every marker into a phantom prose claim, and a control pins that.

   **The mask is scoped to the declaration, not to the line, and the scoping is the whole point.**
   An earlier form of this rule discarded the entire line. It did make the layers disjoint, but it
   also dropped — **silently** — any genuine prose claim authored beside a marker on one physical
   line, and Layer B exists to be the tripwire for exactly that claim, so a silent drop is its one
   forbidden failure mode. Measured before the change: the whole-line form was **inert** on this
   file (the pinned set is 28 rows either way, because no marker line carries a verb today) and yet
   **reachable** by ordinary editing, since nothing here requires a marker to occupy its line alone.
   A leniency that buys nothing today and hides a claim tomorrow is the vacuity item 3 warns about,
   so the rule is enumerated here rather than left to a comment in the test.

   Its control asserts **both** halves at once — the neighbouring claim is found, *and* the marker's
   own declared target is not emitted as prose — so a later edit cannot delete one guarantee and
   still satisfy a test whose name mentions both. Recorded because the whole-line form reached the
   tree first, was reviewed twice, and was corrected only when a reader probed it with a constructed
   line instead of reasoning about it: **the fix for a gate's blind spot needs a control aimed at
   the blind spot, not an argument that the spot is unreachable.**

## 16.27 Derivation recording, the leg-1 gating row, the `DbnetScattered` mechanism, and six errata (Fable ratification package, 2026-07-29; transcribed 2026-07-30)

Fable was convened as tie-breaker on the F1 sequence after the two Opus subagents disagreed. It
re-ran the probe itself under the pinned checkout and cross-validated against §16.20 item 9's
independently ratified measurement of the same coverage-filter firing. **Fable wrote no code and
edited no files**; implementation goes through the normal pipeline. This section is the
transcription, read under `CLAUDE.md` step 1a before commit. The source record is `docs/RULINGS.md`,
items R1 and R3 through R7 plus its sequencing section; R8 of the same package is already
transcribed as §16.26 item 3(d) and is not restated here.

**Record quality, stated once because it bounds every number below.** `docs/RULINGS.md` is the
Orchestrator's record of what the adjudicator ruled, not a verbatim transcript: only text it carries
in block quotes is captured as the adjudicator's own wording, and *"everything else is the
Orchestrator's paraphrase, including the numbers"*.

**What a `>` block below therefore does and does not attribute.** Every `>` block in this entry is
verbatim from `docs/RULINGS.md`. Only **one** of them — item 1's *"not a competing design; it is a
sentence"* — is verbatim from a `>` block *there*, so it is the only one that carries the
adjudicator's own wording. The rest quote the **Orchestrator's record** exactly rather than
paraphrasing it, which is what keeps a narrow scope from being widened in transit (`CLAUDE.md`'s
"a claim's scope travels with it") — but they are not the adjudicator speaking, and no sentence below
may attribute them to the adjudicator. Where a sentence introducing one of these blocks names a
source, it names `docs/RULINGS.md`; "the ruling" is reserved for prose that is already labelled
paraphrase, where no verbatim claim is being made.

This entry therefore ratifies **rules**, and labels every supporting count as what it is. Where a
count could be re-derived while transcribing it was, against the upstream checkout pinned at
`0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`, and the re-derivation is cited beside it; where it could
not, the count is marked unreconciled and may not be cited as reproducible.

1. **Derivation is recorded, and the schema is what is ratified.** The engineer's position wins; the
   architect's prose qualifier was refused as
   > not a competing design; it is a sentence

   The impossibility holds in its decisive form: a split-derived block is observationally
   `xyxy == bbox(lines)` with `ours ⊄ xyxy`, which is *exactly* what a genuine leg-1 defect looks
   like, so any pair-local predicate exempting one exempts the other. The fragility objection is
   answered by precedent — the oracle already reaches inside upstream by ratified decision three
   times (§16.24 items 8, 9, 18(e)); the oracle is the frozen artifact, not the recipe (§16.20
   item 7); and an upstream bump is already a re-record plus full re-sign event.

   (a) **The recorded fields.** Per upstream block:
   `derivation: YoloUnioned | YoloSynthesizedCorners | YoloSplit | DbnetScattered` — a **CLOSED**
   enum. The recorder fails on any unclassifiable block; **there is no `other` bucket**, and adding a
   fifth value is a ratification, not an implementation choice. Alongside it:
   `rect_yolo: Option<[i32;4]>`; `eng_expanded: bool`; and when `eng_expanded` is set,
   `lines_pre_expand` is **REQUIRED** together with `expand_size`.

   (b) **Binding condition on the recorder.** It MUST run **pristine and instrumented**
   `group_output` on **deep-copied identical inputs** and **hard-fail** unless the two outputs are
   field-by-field identical over `xyxy`, `lines`, `language`, `vertical`, `font_size`.
   `PROVENANCE.json` records that the assertion ran — the record of the check is part of the
   artifact, not a claim in a commit message.

   (c) **One exact epsilon-free law per derivation.** Conditions are tabulated rather than summarised
   as "the derivations are consistent", on §16.24 item 19(e)'s rule that an adaptation is justified by
   a condition-by-condition table and not by a reassuring sentence. Every law below is exact: no
   epsilon, no inequality, no IoU, no per-edge band. Line numbers are
   `pcleaner/comic_text_detector/utils/textblock.py` at the pinned commit.

   | `derivation` | `rect_yolo` | exact law | upstream path that produces it |
   |---|---|---|---|
   | `YoloUnioned` | `Some`, the parent yolo box | `xyxy == bbox(rect_yolo ∪ bbox(lines_pre_expand))` | ≥1 line assigned and no split, so `adjust_bbox(with_bbox=True)` runs (method at `:94-105`, its `True` branch at `:96-100`, called from `:509`) |
   | `YoloSynthesizedCorners` | `Some`, and `rect_yolo == xyxy` | `xyxy == rect_yolo == bbox(lines_pre_expand)` **and** `len(lines) == 1` | line-less block survives the `mask_score` filter, then `xywh2xyxypoly` writes the four corners of its own `xyxy` as one line (`:485-492`) |
   | `YoloSplit` | `Some`, the parent yolo box, and in general `rect_yolo != xyxy` | sub-block `xyxy == bbox(lines_pre_expand)`; **and** the unmatched ours-side parent's `rect == rect_yolo` | `split_textblk` then `adjust_bbox(with_bbox=False)` per sub-block (`:440-443`), which discards the parent box entirely |
   | `DbnetScattered` | `None` | `xyxy == bbox(lines_pre_expand)` | constructed from a single unassigned line (`:474`), then `merge_textlines` → `adjust_bbox(with_bbox=False)` (`:399-412`) |

   `YoloSplit`'s law is the one that turns `Mechanism::DocumentedSplitMerge` from an unfalsifiable
   citation into verified evidence: all three of its conjuncts are checkable against the recorded
   artifact.

   Every one of the four laws was exercised on a synthetic input constructed to drive its branch
   (2026-07-30, pinned checkout), and each held exactly. That is a check on the *reading* of upstream's
   control flow, not evidence about real pages: on the three real candidate pages of (f) only three of
   the four derivations occur at all, and `YoloSplit` occurs on none of them. The recorder's hard-fail
   is what keeps an unexercised law from passing silently.

   (d) **Why the laws are stated over `lines_pre_expand`, measured rather than argued.** The
   English-expansion loop (`:518-532`) rewrites `blk.lines` **after** the last `adjust_bbox` and never
   re-adjusts `xyxy`, so for an `eng`-classified horizontal block the served `lines` no longer satisfy
   the law. Two measurements, both taken 2026-07-30 while transcribing, against the pinned checkout.

   *On the ratified oracle page itself*, through the census of (f): its block 3 is the
   `YoloSynthesizedCorners` case, `xyxy = [674,1397,740,1438]` with served
   `bbox(lines) = [674,1393,740,1442]` — expanded by 4 px on both `y` edges, so
   `xyxy == bbox(lines)` is **false** over the served field and true over `lines_pre_expand`. Its
   blocks 0 and 1 are `eng` and horizontal too, and their served `bbox(lines)` overhangs their `xyxy`
   on `y1` in the **same direction but not the same magnitude** — 2 px each (`74 → 72` and
   `630 → 628`) against block 3's 4 px. So the loop's effect is the page's normal case rather than a
   corner of it, while its size is per-block. Those two per-block figures are read off the recorded
   instrumented-`group_output` output for P01 — the page byte-verified by sha256 against §16.24 item
   17 — under the pinned checkout; an earlier draft said "both show the same outward `y1` shift",
   which is true of the direction and false of the amount, and `lines_pre_expand` is required because
   the direction recurs, not because the offset is a constant one could subtract back out.

   *On a synthetic single-box, no-lines input* driven straight into `group_output` (`cls = 0` → `eng`,
   mask fill 255, box `[50,60,150,200]`), which isolates the mechanism from the network: the block
   comes back with `xyxy = [50,60,150,200]`, `bbox(lines_pre_expand) = [50,60,150,200]` — law holds —
   and post-expansion `bbox(lines) = [50,46,150,214]` with `expand_size = 14`. The identical run with
   `cls = 2` → `unknown` leaves the lines untouched, which pins the cause to the language branch and
   not to something else in the path. That is the whole reason `lines_pre_expand` is required rather
   than convenient.

   (e) **Why (b)'s instrumentation is load-bearing, and not replaceable by re-derivation.** Measured
   on the same synthetic branch runs as (c): `xyxy == bbox(lines_pre_expand)` holds for **three** of the four
   derivations — `YoloSynthesizedCorners`, `YoloSplit` and `DbnetScattered` all satisfy it exactly —
   so the law alone does not classify a block. The discriminator is `rect_yolo`: present and equal to
   `xyxy` (corners), present and unequal (split), or absent (scattered). `rect_yolo` is **not
   recoverable from upstream's output**, because `adjust_bbox` overwrites the field it was stored in;
   it exists only inside the run. A recorder that inferred `derivation` from the artifact would
   therefore be guessing between three shapes, which is the failure (b) exists to prevent.

   (f) **The validating corpus is NOT transcribed; the census that IS reproducible is recorded
   instead.** The source record carries per-derivation validation counts and an eng-expanded count over
   a larger corpus. Its own annotation, added 2026-07-30, withdraws those as reproducible fact: the
   pages that produced them are not named anywhere retrievable, and they cannot be the three recorded
   oracle-page candidates, which yield 14 blocks between them. **Do not cite those totals from this
   entry — this entry does not carry them.**

   What is recorded here is the census that was re-run while transcribing (2026-07-30), because a
   measured small number beats an unattributable large one. Pages: the three Pepper&Carrot candidates
   of §16.24 item 17, P01 byte-verified against item 17's recorded sha256 and P02/P03 against item
   17(a)'s recorded sizes. Model: `comictextdetector.pt.onnx`, sha256
   `1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f`, which is
   `pc_models::COMIC_TEXT_DETECTOR.sha256` character-for-character — so the census ran on this
   project's own pinned weights, not a lookalike. Loaded through
   `cv2.dnn.readNetFromONNX` — the backend §16.20 item 6 makes load-bearing. Derivation tagged by
   instrumenting three module-global functions inside `textblock`, each delegating to the real one and
   only labelling the blocks it returns, so no upstream arithmetic was reimplemented.

   | page | blocks | `YoloUnioned` | `YoloSynthesizedCorners` | `YoloSplit` | `DbnetScattered` |
   |---|---|---|---|---|---|
   | P01 (the ratified oracle page) | 4 | 3 | 1 | 0 | **0** |
   | P02 | 6 | 4 | 0 | 0 | **2** |
   | P03 | 4 | 3 | 0 | 0 | **1** |
   | total | 14 | 10 | 1 | 0 | **3** |

   Read this as what it is: **three pages, not a distribution.** It is enough to settle the two
   questions this package turns on — whether a no-yolo-box block occurs at all (item 3) and whether a
   line-less output block occurs (item 7) — and it is not enough to characterise `YoloSplit`, which
   these pages never produce. A future reader needing a real per-derivation distribution takes it from
   the recording run.

   (g) **NOT authorised here: the frozen struct literals in
   `crates/pc-detect/tests/f1_oracle_comparator.rs`.** The package's sequencing note describes a
   mechanical adaptation of those literals as already authorised. Checked while transcribing
   (2026-07-30): **no ruling in the package grants that authorisation** — the sentence asserts it
   without a source. Editing a frozen test needs a named exit under cookbook rule 8, and the closest
   precedent, §16.24 item 19(d), came paired with (e)'s condition-by-condition table. So the
   adaptation is **DEFERRED** pending a joint architect-plus-engineer or Fable ruling that authorises
   it explicitly and states its conditions. Until that lands the file is not edited, and nothing in
   this entry may be read as licensing it. Recorded rather than dropped, per §16.23 item 1's rule.

   **(SUPERSEDED by §16.28 — read it before treating this deferral as still open. The joint
   architect-plus-engineer pass this item calls for happened, the two disagreed on one mechanism,
   Fable tie-broke it, and §16.28 states the resulting authorization and its exact scope. This
   entry's history — that the sequencing note asserted authorization with no source, and that a
   named ruling was required before editing — stands as written; only its "pending" status is
   resolved.)**

2. **A second, new geometry gate: `ours.rect == rect_yolo` on every paired block, exact and
   epsilon-free.** §16.20 item 3(b) alone is no longer fit as the sole geometry gate — the ruling's
   recorded grounds, carried as paraphrase, are that 65 of 100 edges are unconstrained by it, 8 of 25
   blocks are free on all four edges, and the `spikey` fixture tolerates a `[19,60,23,17]` px error.
   The remedy is a row, not a tolerance.

   (a) **This does not reopen §16.24 item 18(h).** Quoted rather than paraphrased, because the two
   comparisons differ by which operand is subtracted:
   > 18(h) is NOT reopened — its amendment made `upstream.xyxy - ours.rect` *diagnostic* precisely
   > because that is non-zero when healthy; the new row gates `ours.rect - rect_yolo`, which is zero
   > when healthy.

   `residual_leg1` stays exactly as item 18(h) leaves it: diagnostic, and expected non-zero on a
   healthy line-informed pair. The new row is a different quantity on a different pair of operands,
   and it is zero on a healthy pair, which is what makes it gateable without an epsilon.

   (b) **Recording-time contingency, with its exits already ratified.** A truncation straddle can fail
   this row at recording time. The sanctioned exits are item 18(h)(ii)'s and no others;
   classification is item 18(j)'s — a §16.x shared-sensitivity finding, **never** a §14 register
   entry, because both sides truncate; and the evidence is the pre-truncation floats on **both**
   sides, not on ours alone.

   (c) **Structural guard, and it is what makes (a)'s row well-posed.** A pair whose upstream
   derivation is `YoloSplit` or `DbnetScattered` is **itself a gating row**. For a split sub-block
   `rect_yolo` is the parent, so a sub-block paired with one of ours would compare our box against a
   box upstream itself discarded; for a scattered block there is no `rect_yolo` to compare against at
   all. Neither shape may be paired, and the guard says so structurally instead of leaving the
   equality row to produce a confusing failure. The gating variant's Rust name is left to F1-C and is
   not ratified here; its message must name the derivation, since that is the fact a reader needs.

3. **`DbnetScattered` becomes a mechanism, and three ratified clauses move with it.** Without it, a
   page carrying scattered blocks has only `Open` available — a blocking violation — and stretching
   `DocumentedSplitMerge` across the no-line-synthesis deviation would widen a register entry past its
   recorded scope.

   The ground is measured, not inferred, and the measurement is item 1(f)'s census: **P02 carries 2
   `DbnetScattered` blocks and P03 carries 1**, with **0 on P01**. The larger no-legal-mechanism ratio
   the source record cites inherits item 1(f)'s provenance gap and is not restated here as fact —
   **the falsification does not need it**, because one such block is enough to refute a claim that
   there are none, and three were counted directly.

   (a) **The mechanism enters a closed enumeration, which is why this is an amendment and not an
   addition.** §16.24 item 4 ends with a closed list: *"Every `Unmatched` entry carries a mechanism
   (§14.13 per-class duplicate / coverage-filtered / documented split-merge citing its register entry
   / `Open`), and `Open` is a blocking violation, never a verdict."* A fifth alternative is inserted:
   **`DbnetScattered`, citing its register entry**. The `Open`-is-blocking half is untouched.

   **SUPERSEDES: §16.24 item 4**

   The mechanism is upstream-side-only **by construction**: v1 synthesizes no DBNet line polygons, so
   we can never produce a scattered block, so one can never appear on our side and can never be
   paired — item 2(c) makes pairing one a gating row. The mechanism carries the same name as the
   `derivation` value of item 1(a) deliberately; the source record uses one name for both positions.

   **Register anchor, recorded as an open obligation rather than invented.** The source record calls
   this a *"mechanism/register entry"* without naming a §14 number, and the register carries no entry
   *for* the no-line-synthesis deviation: §14.17 states the fact in passing, as a premise of the
   coverage-filter entry, and cites §14.12 for it, whose own text is about mask refinement. Under
   §16.20 item 3(e) a row may not close as
   `EXPLAINED` against an entry that does not yet exist, so until such an entry lands **the anchor for
   a `DbnetScattered` entry is this item**, which does exist and does record the mechanism. If a
   future ratification adds a §14 register entry for the deviation, the anchor moves there in the same
   edit as the spec, the Rust identifier and the doc — item 20(c)'s condition, applied to a citation
   rather than to a name.

   (b) **The accounting partition gains a fourth upstream term.** §16.24 item 20 ratifies that every
   unmatched entry on a side falls under exactly one term and that each term names exactly one
   mechanism class; a new mechanism class therefore forces a new term, or scattered blocks fall into
   `open_upstream` and block a page that has nothing wrong with it. Item 20(c)'s table gains one row:

   | side | term | mechanism |
   |---|---|---|
   | upstream | `dbnet_scattered` (unsuffixed — the mechanism is upstream-side-only, see (a)) | `DbnetScattered` |

   **SUPERSEDES: §16.24 item 20**

   The upstream equation becomes
   `pairs + class_duplicates + documented_split_merge_upstream + dbnet_scattered + open_upstream == upstream_total`;
   the ours-side equation is unchanged. The term is **derived** from the signed entry list, on the
   same side of item 20(b)'s split as every other mechanism term — the authored anti-vacuity literals
   remain `pairs`, `upstream_total`, `ours_total` and gain nothing here.

   **The spelling is an application of a ratified rule, not a fresh ruling.** It is unsuffixed on item
   20(c)'s own precedent for `class_duplicates` (*"unsuffixed — the mechanism is upstream-side-only by
   ratified definition"*), since the mechanism has no ours-side counterpart to distinguish it from. A
   later ratification preferring another spelling changes spec, Rust identifier and doc together or in
   none of them, which is item 20(c)'s own condition. Item 20(c) also owns the restatement of item
   11's two equations; item 11 carries a pointer to this entry for that reason and no claim is
   declared against it here, because the source record names item 4 and item 20(c) only. Extending a
   claim to item 11 is a separate, argued step and is flagged rather than performed.

   (c) **§16.25 item 10's presupposition is falsified, and its own ruling decides what happens next.**
   Item 10 named the fact unverified, presupposed the answer was no, and pre-committed the outcome:
   *"If the check comes back the other way, the finding is NOT 'Partial should be diagnostic'"* — it is
   that item 18(b)'s leg-2 law needs an **erratum**, which is item 10's own word and one of the
   supersession verbs `crates/pc-testkit/tests/spec_supersession.rs` recognises, so it is used here
   rather than a synonym. It came back the other way. What follows is that
   ruling **firing**, not a fresh judgment; neither architect cited it, which is how scope drift
   starts. Item 10's obligation is re-scoped, quoted from the source record (`docs/RULINGS.md`)
   rather than restated — the Orchestrator's wording there, not the adjudicator's:
   > Item 10's obligation is re-scoped to one score per gated yolo-derived block.

   **SUPERSEDES: §16.25 item 10**

   Item 10's conclusion about `Partial` is otherwise **untouched and independently sound**: `Partial`
   does not fire on scattered blocks, because §16.25 item 6(a) computes the coverage subject over the
   **PAIRED** set and a scattered block is never paired. So the re-scope narrows a recording-script
   obligation; it does not reopen `Partial`'s gating status, which rests on §16.20 item 3(e).

4. **Erratum 1 of 6 — item 18(b)'s leg-2 law is a law about `YoloUnioned` blocks, not about every
   upstream block.** Leg 2 is `upstream.xyxy == bbox(upstream.rect_yolo ∪ bbox(upstream.lines))`,
   which is undefined for a block that has no `rect_yolo`. Item 3 above measures such blocks. The
   correction is the scope of the law, not its algebra.

   **SUPERSEDES: §16.24 item 18(b)**

   **The scope of the correction, quoted rather than widened, because it is narrower than it looks:**
   > The 11/49 no-legal-mechanism ground does **not** bite p01 (zero scattered, zero split there); it
   > blocks the *vocabulary* and the *alternative pages* (p02 has 2 scattered, p03 has 1).

   Of that quote, the **per-page halves were re-measured and hold exactly** — item 1(f)'s census reads
   0 / 2 / 1 on P01 / P02 / P03 and zero splits anywhere. The `11/49` ratio is from the corpus item
   1(f) marks unreconciled; it is quoted here because it is the **source record's** own scope
   sentence — `docs/RULINGS.md`'s wording, which is the Orchestrator's and not the adjudicator's —
   and it is not asserted as a fact by this entry. Nothing in the correction depends on it.

   Three consequences follow from that scope and nothing wider. **The ratified oracle page is not
   re-selected**: under corrected recording every P01 block closes exactly — three paired with
   **`residual_full`** at `[0,0,0,0]` over `lines_pre_expand`, one `ClassDuplicateOf`, one
   `CoverageFilteredUpstream` — and re-selection could not dodge the finding anyway, since any page
   with an `eng`-classified horizontal block reaches the expansion loop. **Item 18(f) is not touched**:
   no port defect exists, because the mechanism is upstream's own post-`adjust_bbox` mutation. And the
   vocabulary this entry adds is what the alternative pages need; on P01 it is unexercised.

   Two labels on that P01 tally, per this entry's own rule that every supporting count says what it is.
   **The residual is named, because §16.24 item 18(h) defines two and they carry opposite
   expectations:** `residual_full` is the composite the identity requires all-zero, so `[0,0,0,0]` is a
   claim about **it**; `residual_leg1` is diagnostic and, per item 2(a) above, is *expected* non-zero on
   a healthy line-informed pair, so the same vector would mean the opposite there. **And the 3 / 1 / 1
   split is carried as paraphrase** from the source record, not re-derived while transcribing. What
   item 1(f)'s census does independently confirm is its arithmetic base — P01 yields 4 output blocks, 3
   `YoloUnioned` plus 1 `YoloSynthesizedCorners`, and 0 `DbnetScattered` — and §16.20 item 9 already
   ratifies the single coverage-filter firing that the fifth entry accounts for.

   One correction of the engineer's own count is carried across with it: *"3 of 4 gated blocks escape"*
   overcounts, because P01's block 1 is the §14.13 class-duplicate of block 2 and is never a gated
   pair, so the figure is **2 of 3 paired**. The census puts a measurement behind that: P01's blocks 1
   and 2 come back as served `xyxy` `[607,630,723,703]` classified `eng` and `[607,631,724,705]`
   classified `ja` — the same balloon at two class indices, which is the duplicate that clause names,
   at the indices it names. The two boxes differ on **three of the four edges, and not by a uniform
   amount**: `x1` is identical, `y1` and `x2` differ by 1 px, and `y2` by 2 px. That is arithmetic on
   the two vectors quoted here, so it needs no re-run to check — which is why the earlier draft's "one
   pixel apart on two edges" was catchable by reading alone: it does not hold of the served `xyxy`
   values it was attached to. Either way the conclusion is untouched, because §14.13's class-duplicate
   condition is same-balloon-different-class and not an edge-distance test.

5. **Erratum 2 of 6 — item 21(f)'s per-edge sign attribution is narrowed to `YoloUnioned` pairs.**
   Item 21(f) ratifies that "the union-impossible directions are `residual_leg1.x1 > 0`, `y1 > 0`,
   `x2 < 0`, `y2 < 0` — each of those *proves* leg 1, while the complementary directions prove nothing,
   being consistent with either leg." That is sound where the upstream box **is** a bounding union of a
   yolo box with lines, and silent where it is not. (The quote is carried without this entry's usual
   outer italics on purpose: 21(f) emphasises *proves*, and wrapping the whole sentence in `*"…"*`
   swallows that emphasis, which is what an earlier draft here did.)

   **SUPERSEDES: §16.24 item 21**

   **The scope, quoted:**
   > Its per-edge attribution is true of `YoloUnioned` pairs and silent otherwise.

   The counterexample recorded in the ruling — a paraphrase number, not a re-derived one — is
   nightmare block 4: `residual_leg1 [45,74,-1,-1]`, all four components in union-impossible
   directions, **with leg 1 EXACT**. The mechanism is that `residual_leg1` subtracts `ours.rect` from
   `upstream.xyxy`, and outside `YoloUnioned` those two are not related by a union at all, so their
   difference carries no information about `rect_yolo == ours.rect`. Item 2 above is what closes the
   gap: leg 1 gets its own exact row instead of being attributed from signs.

   This correction is **cookbook rule 15 applied to the clause that cites rule 15** — 21(f)'s own
   opening sentence invokes rule 15 for sign precision, and a supporting example refuted the claim it
   was cited for one level up. Both halves of rule 15's discipline apply here: derive the constraint
   the evidence must satisfy, then check each data point against it.

6. **Erratum 3 of 6 — item 18(i)'s diagnostic message is derivation-conditional.** Item 18(i) requires
   that on `upstream.x2 < ours.x2` the comparator *"report that a bounding union cannot narrow an edge
   and this is therefore a leg-1 engine-geometry divergence, not a line-union difference."* The first
   clause is a theorem about bounding unions and stays; the inference in the second clause holds only
   when the upstream box was produced by a union.

   **SUPERSEDES: §16.24 item 18(i)**

   Reworded: the message asserts a leg-1 engine-geometry divergence **only when the pair's upstream
   derivation is `YoloUnioned`**. For `YoloSplit` and `DbnetScattered` the upstream box is
   `bbox(lines)` and never a union with a yolo box, so a narrowed edge is the expected shape rather
   than a divergence, and the old message would name the wrong leg with full confidence. Item 2(c)'s
   structural guard is the other half of the fix: such a pair raises its own gating row, so in a green
   run the reworded message is reached only where its premise holds. The additive-strengthening status
   of 18(i) is unchanged — it is a message, it adds no gate power, and it needed no ratification to
   exist.

   `docs/COOKBOOK.md` rule 15's final bullet carries the same unconditional sentence and is corrected
   in the same change, because a lesson stated unconditionally is how the sentence got into a
   normative clause in the first place.

7. **Erratum 4 of 6 — §16.20 item 3's "this case is not hypothetical" conflates two different
   branches.** Item 3(b) requires the identity to handle a line-less upstream block, *"where
   `bbox(upstream.lines)` is undefined and the identity degenerates to `upstream.xyxy == ours.rect` —
   v1 synthesizes no lines, and upstream's line-less branch is exactly where §14.17's filter
   divergence lives, so this case is not hypothetical."* Two distinct things are named "the line-less
   branch" there. Upstream's line-less **filter** branch is real and fired once, on
   `[438,1407,498,1446]` with `mask_score` 0.0359 — the figures §16.20 item 9 already ratifies, not
   new ones. A line-less **output** block is impossible.

   **SUPERSEDES: §16.20 item 3**

   **The control-flow fact, re-read against the pin rather than assumed.** `group_output` lives at
   `pcleaner/comic_text_detector/utils/textblock.py:447-534`. Every path to `final_blk_list` is
   line-guaranteed: a yolo block either already carries assigned lines, or is `continue`d out on the
   `mask_score` filter (`:485-492`), or has the four corners of its own `xyxy` appended as one
   synthetic line before being kept; a scattered block (`:474`) is constructed with one line by
   definition; and the expansion loop only rewrites lines that already exist. No branch appends a
   line-less block.

   Two measurements agree with that reading, both 2026-07-30. Driven across five synthetic branch cases
   — line-less above and below the filter threshold, assigned-lines, scattered, and both together — the
   pinned `group_output` returned **6** blocks and **0** with `len(lines) == 0`. Over the three real
   candidate pages of item 1(f): **14** blocks, **0** with `len(lines) == 0`. The `0/49` census the
   source record cites inherits item 1(f)'s provenance gap and is not restated as fact; the
   impossibility rests on the control flow, and these two runs are what keep the reading from being an
   argument nobody executed.

   **What this changes, and what it deliberately does not.** The degenerate branch is a property of
   the artifact **grammar**, which admits `lines: []`, and not a reachable upstream output. So the
   frozen tests over that grammar assert correct behaviour and are **not** replaced — cookbook rule 8
   **exit 1**, additive: replacement was refused. Four reachable-shape controls are added instead: a
   synthesized-corners pair, a dominating-lines pair, a multi-line pair, and a split-shaped unmatched
   structure. What the erratum removes is the *weighting* — the sentence licensed treating the
   degenerate branch as the well-evidenced case, and it is the one shape upstream cannot emit.

8. **Erratum 5 of 6 — item 18(k)'s "rare-but-real" repeats erratum 4's conflation at a second site.**
   Item 18(k) corrects the architect's *"the common case"* to *"rare-but-real, which is what §16.20
   item 3's 'not hypothetical' meant"*. As a statement about upstream's line-less **filter** branch
   that is right and item 9's measurement supports it. As a statement about a line-less **output**
   block it is the same error as erratum 4: one control-flow fact, two sites repeating it, which is
   why they are corrected in one entry rather than one at a time.

   **SUPERSEDES: §16.24 item 18(k)**

   Item 18(k)'s consequence survives, re-aimed. Its point that the degenerate branch is the
   least-evidenced part of the identity, and that item 18(e)'s per-block `len(lines)` census is the
   first real evidence on it, still holds — and the census now has a stated expectation to be checked
   against: **no gated upstream block carries zero lines**, with 0 of 14 on the three candidate pages
   as the standing prior. A census whose expected result is undeclared could not have failed, which is
   the shape cookbook rule 6 exists to catch.

9. **Erratum 6 of 6 — §16.6 item 4 cites an upstream symbol that does not exist; the clamp itself is
   KEPT and re-grounded as §14 register entry 18.** Item 4 justifies clamping rescaled detector boxes
   with *"This isn't new scope — upstream's yolov5 pipeline clips coordinates (`clip_coords`) as a
   normal part of the postprocess the spec already claims to port; the original §8.3 step 4 text
   simply omitted mentioning it."* There is no such symbol upstream.

   **SUPERSEDES: §16.6 item 4**

   **Verified while transcribing (2026-07-30):** `grep -rn clip_coords` over the whole pinned checkout
   **exits 1** — no hit in any file. The only clamping in the vendored yolov5 helper is IoU
   arithmetic: `yolov5_utils.py:166` is the comment naming `.clamp(0)` and `:169` is the call itself,
   both inside `box_iou`'s intersection computation, and they are the file's only two occurrences of
   `clamp` or `clip`. The source record places this in *"the same family as `PAD_VALUE=114`"* — a
   normative clause importing generic yolov5 behaviour that comic-text-detector does not have.

   **What changes is the justification, and only that.** The clamp is **KEPT**: no relaxation, no
   `validate()` change, no deletion. It is required so `run()` can produce a `PageDataRaw` that passes
   its own `validate()` on real frame-overhanging detector output, and that invariant is ours — which
   is precisely why the clamp is a deliberate divergence rather than a port. The outcome therefore
   differs from §16.20 item 8's disposition of the pad value, where upstream was normative for the
   line and the value was corrected to match; here upstream has no equivalent behaviour to match, so
   the clamp is registered as a deviation. §14 register entry **18** is added for it in the same
   change, and §16.6 item 4's clamp rule itself stands unaltered.

   **Owed at the code sites — AUTHORISED by this item together with §14 register entry 18, and
   carried by a separate follow-up commit rather than by this transcription.** Three sites, and they
   are enumerated rather than summarised because `grep -rn clip_coords` over **this repo** (run
   2026-07-30 while reviewing this transcription) found the withdrawn symbol at **two** live sites,
   while the first draft of this enumeration named only the two inside `yolo.rs` and missed the second
   live one entirely — an incomplete list would have left the false citation standing where nothing
   pointed at it:

   - `crates/pc-detect/src/yolo.rs` — the `DEVIATION(18)` comment at the clamp itself.
   - `crates/pc-detect/src/yolo.rs` — that file's doc comment, whose `clip_coords` citation is
     rewritten to name what upstream actually does: the IoU-intersection `.clamp(0)` at
     `yolov5_utils.py:166,169`, which is unrelated to a frame-bounds clamp.
   - `crates/pc-detect/tests/d5_yolo.rs:297` — a **comment** on
     `rescale_clips_boxes_that_overhang_image_bounds` reading *"matching upstream's `clip_coords`"*:
     the same withdrawn citation at a second site, and the one this entry originally missed. Only the
     comment text changes — it is rewritten to cite `yolov5_utils.py:166,169` like the doc comment
     above — and no assertion, test name or expected value in that file is touched, so the file's
     `FROZEN` header and cookbook rule 8 are not engaged. The two other `§16.6 item 4` citations in
     the suite (`crates/pc-detect/tests/d7_run.rs:309`, `crates/pc-detect/tests/d4_session.rs:163`)
     cite the **clamp rule**, which is KEPT, and are correct as they stand.

   This entry is spec-only. The follow-up commit cites *"§14 register entry 18 / §16.27 item 9"* as
   its authority, which is what distinguishes it from item 11's list: item 11 records work that is
   **not ratified**, whereas these three sites are ratified here and merely not yet landed.

10. **Closure note — item 18(d) is NOT a seventh erratum, and nothing is corrected for it.** Recorded
    so a future reader does not reopen it. The truncation-straddle branch was re-verified on
    2026-07-30 against the rebuilt oracle — **that run is the source record's, re-read while
    transcribing rather than re-executed here**, and its numbers are carried with that attribution. It
    instrumented the exact truncation site (`inference.py:114-124`, `postprocess_yolo`'s
    `.astype(np.int32)`, a citation checked line-for-line against the pinned checkout while
    transcribing) on both fixtures with the real weights: `darkrays` box 0 gives a
    pre-truncation `x2` of `110.9987564086914`, which truncates to `110` and matches the disputed line
    to four decimals, while `nightmare`'s sole YOLO box is nowhere near 110/111. **The fixture
    attribution is correct as written, and §16.20 item 12's transcription was correct as written.**

    Item 18(d)(2)'s own prose is not wrong either: its `ratio_x = 219/654` and `111.02 / 110.98`
    figures are `nightmare`'s geometry, used as an illustrative computation of the `r ≈ 3.0–3.8` regime
    both fixtures sit in (`darkrays` is 208×320, `r = 3.2`). The two were never the same measurement,
    and reading them as one shared worked example is the only thing that looked like a contradiction.
    No marker is declared for item 18(d) because nothing there is being replaced.

11. **Carried in the source ruling but deliberately NOT transcribed here, recorded rather than
    dropped (§16.23 item 1's rule).** **Three** items. Each needs its own transcription or its own
    ratification, and **none of these three may be read as ratified by this entry**:

    - **`UpstreamBoxOutsideFrame` as a gating row** (reported as 0/38 today). Part of the same
      disposition as item 9 above, but it is a new gating row rather than a correction of a cited
      fact, and this transcription's scope was the citation and the register entry. It needs its own
      entry before F1-C asserts it.
    - **The frozen-literal adaptation in `f1_oracle_comparator.rs`** — deferred, see item 1(g). It is
      unauthorised, not merely unscheduled. **(NOW AUTHORIZED — see §16.28, which resolves item
      1(g)'s deferral. This bullet's history stands as written; it no longer describes the present
      state.)**
    - **A worked discriminator example** in the source record naming a fixture that could not be
      located in this repo, in upstream's asset tree, or on any recovered page. It is recorded there
      as unverifiable and is not transcribed at all; the ruling it supported — that the discriminator
      is rejected in favour of item 2's gating row — does not rest on it.

    **Not a member of that list, and stated separately because the two statuses are opposite: item
    9's three code sites are AUTHORISED — they are simply not landed by this transcription.** The
    `DEVIATION(18)` comment, the doc-comment rewrite in `crates/pc-detect/src/yolo.rs`, and the
    corrected `clip_coords` citation in `crates/pc-detect/tests/d5_yolo.rs:297` are ratified by item 9
    together with §14 register entry **18**, which lands in the same change as this entry; the code
    itself cites *"§14 register entry 18 / §16.27 item 9"* as its authority. They are an
    authorised-but-not-yet-landed follow-up commit, **not** an open ratification question. An earlier
    draft listed them as a fourth bullet above, which read them as unratified — the exact opposite of
    item 9's disposition, and the reason the split is spelled out here rather than left to the reader.

## 16.28 The frozen literals in `f1_oracle_comparator.rs` — authorized scope (Fable tie-break, 2026-07-30)

**SUPERSEDES: §16.27 item 1(g)**

§16.27 item 1(g) deferred authorization for adapting the frozen struct literals in
`crates/pc-detect/tests/f1_oracle_comparator.rs` pending *"a joint architect-plus-engineer or Fable
ruling that authorises it explicitly and states its conditions."* That pass ran: the Technical
Architecture and Senior Rust Engineer agents were convened jointly, each blind to the other's
output, and produced full independent rulings. They agreed on most of the scope and disagreed on
one mechanism, so per `CLAUDE.md`'s tie-break rule `fable-adjudicator` reviewed both positions and
decided between them rather than designing from scratch. The full source record, including both
agents' verbatim reasoning and Fable's decisive quotes, is in `docs/RULINGS.md` under "§16.27 item
1(g) frozen-literal adaptation — Fable tie-break, 2026-07-30"; this entry transcribes the ratified
conclusions.

1. **The sequencing note's "mechanical adaptation" is refused as a category.** Both agents
   independently found that §16.27 item 1(a)'s closed 4-value `derivation` enum admits **no** value
   for a line-less block, and 24 of the 28 `OracleBlock` instances the file constructs have no line
   data at all. Writing any real `derivation`/`rect_yolo` value at those 24 sites is fabrication,
   not adaptation — there is no forced or defensible value to choose. What is authorized instead is
   narrower: a null completion.

   (a) **Nine `OracleBlock` construction sites** (the two `oracle_block()`/`bare_block()` helper
   bodies, plus seven inline literals) each gain the five new fields, all at their "unrecorded"
   value: `derivation: None`, `rect_yolo: None`, `eng_expanded: false`, `lines_pre_expand: None`,
   `expand_size: None`. Forced, not chosen: no enum member is admissible at 24 of the 28 instances,
   and the remaining 4 are refused per (b) below, so `None`/`false` is the only assignment that
   asserts nothing false. The 16 `bare_block(...)` and 5 `oracle_block(...)` *call sites* need no
   edit — both helpers absorb the new fields in their bodies, which is itself evidence the edit is
   genuinely mechanical at these nine sites.

   (b) **Two sites where the union arithmetic technically forces a `rect_yolo` value are refused
   anyway.** `upstream_truth()` blocks A and C solve uniquely under `Rect::merge`'s min/min/max/max
   semantics (verified: `crates/pc-core/src/geometry.rs:60-67`). Filling in block C is refused
   because it collides with the in-place `xyxy` mutation the fault-injection tests already perform
   on it at `f1_oracle_comparator.rs:171` and `:352` — a real fixture defect the enrichment would
   introduce. Filling in either block is refused on the second, independent ground that it would
   make these controls model a second subject (derivation classification) they were not written to
   model, for only the pass-side of the new gate's coverage. New controls, not enrichment of frozen
   ones, carry that coverage (item 5 below).

   (c) **Schema placement is a binding precondition, not a detail.** `derivation` is
   `Option<Derivation>` with `#[serde(default)]` on `OracleBlock` — **not** a required serde field.
   Forced by two existing frozen assertions: the `minimal` literal
   (`f1_oracle_comparator.rs:1739-1747`, a bare `{"xyxy":[1,2,3,4]}` block that must still parse)
   would panic under a required field, and the `"eng"`-language-rejection probe (`:1783-1786`) would
   pass for the wrong reason (missing-field, not the intended language-value rejection) — a silent
   weakening invisible to the compiler and to a green run. Both sites are protected by this schema
   placement and are **not edited**.

   (d) **`Totals` and its construction sites need no edit.** §16.27 item 3(b) already rules the
   authored anti-vacuity literals (`pairs`, `upstream_total`, `ours_total`) "gain nothing here"; this
   entry confirms that as a no-change row so a future reader does not infer one.

   (e) **Two ambiguities in item 1(a)'s text, resolved because getting them wrong breaks frozen
   vectors.** Item 1(a)'s "REQUIRED" binds the **recorder**, not the Rust type — (c) above is the
   consequence. And the comparator does **not** re-check the four derivation laws itself: item
   1(b)/(c) place them on the recorder, item 2 adds exactly one comparator row. Checking the laws
   inside `compare()` would make the fault-injection tests' in-place `xyxy` mutation at `:171` emit
   an extra divergence row into that test's already-frozen 3-row exact vector (a fourth row) and at
   `:352` into that test's already-frozen 1-row exact vector (a second row) — an amendment nobody has
   authorized.

2. **The one point of disagreement: how to close the "silent absence" hole, and Fable's decisive
   ruling on it.** With `derivation: Option<_>`, a real recorded artifact that dropped the field
   would parse and compare silently — nothing would catch that a real block should have carried a
   derivation but doesn't. The Rust Engineer proposed signing `derivation` on `ExpectedPair` itself
   (mirroring the existing `IdentityBranch` signed-and-verified field, `oracle.rs:213-216`) and
   gating a mismatch as a new `Divergence` variant. The Architect proposed routing it through the
   existing per-field `OracleCoverage`/`FieldCoverage` channel (§16.25's mechanism for
   `confidence`/`language`) with the "uniform absence" case gated only later, at the atomic
   recording commit.

   **Fable ruled for the Rust Engineer's mechanism**, quoted because this is the ruling's own
   reasoning:

   > The decisive ground is that a ratified clause already answers the Architect's proposal, and the
   > Architect did not cite it. §16.25 item 5 carries a two-condition scope test for exactly this
   > question ... `derivation` is none of those — it is an upstream-only recording probe, exactly
   > the shape of `base_xyxy_pretruncation`, which §16.25 item 5 names as the worked exclusion.

   and:

   > The Architect's design leaves the named hole open. Its uniform-absence case does not gate
   > inside `compare()` at all; the gate is deferred to a real-page authored assertion the Architect
   > itself flags as "extending §16.24 item 6's ratified two-part CI contract", "needing its own
   > ratification", and "not granted here". So as submitted, the design closes the silent-pass hole
   > only after a future ratification that does not exist.

   **Ratified: `ExpectedPair` gains `derivation: Option<Derivation>`.** All 21 frozen `ExpectedPair`
   literals in `f1_oracle_comparator.rs` gain `derivation: None` — a true statement that none of
   this file's existing controls exercises the derivation axis, compiler-forced, asserting nothing
   false.

   **Amendment 1, added by Fable — verification is exact BIDIRECTIONAL equality.** The mismatch
   fires on signed `Some(a)` vs. recorded `Some(b)` where `a ≠ b`, on signed `Some(a)` vs. recorded
   `None`, **and on signed `None` vs. recorded `Some(b)`.** The Rust Engineer's original proposal
   named only the first two directions; the third is what stops a future control from carrying a
   real recorded derivation while signing `None` and silently skipping the gate. This costs the
   frozen file nothing: every one of its 21 signatures and all 28 artifact blocks are `None`, which
   is equal, so no row fires and every existing exact-vector `assert_eq!(report.divergences, ...)`
   is untouched.

3. **The ratified `Divergence` variant count is 19**, composed of: the 16 variants that exist today
   (`crates/pc-detect/src/oracle.rs:317-423`), plus §16.27 item 2's `ours.rect == rect_yolo` leg-1
   gating row, plus item 2(c)'s structural guard (a pair whose upstream derivation is `YoloSplit` or
   `DbnetScattered` is itself a gating row), plus the signed-derivation-mismatch gating variant this
   entry ratifies in item 2 above. `the_divergence_class_partition_is_total_and_gating_excludes_diagnostics`
   moves from `samples.len() == 16` / `(gating, diagnostic) == (14, 2)` to `samples.len() == 19` /
   `(17, 2)`. This count is derived from this enumeration, never from counting the enum itself
   (cookbook rules 7 and 13). `UpstreamBoxOutsideFrame` is **not** among the 19 — §16.27 item 11
   leaves it untranscribed, and it needs its own entry before it can be asserted; landing it later
   makes it a 20th variant by its own ratification, not a consequence of this one. Rust names of all
   three new variants are left to F1-C; the mismatch variant's message must name both the signed and
   the recorded derivation, and item 2(c)'s guard's message must name the derivation (already
   required by item 2(c) itself).

4. **A `derivation == None` artifact block disables the derivation-conditional machinery for its
   pair.** Item 2's `ours.rect == rect_yolo` row, item 2(c)'s structural guard, and item 6's
   derivation-conditional 18(i) message do not apply when the recorded artifact carries no
   derivation. This must be stated in `oracle.rs`'s doc comment, not merely implemented, and it is
   what keeps every existing exact-vector assertion in `f1_oracle_comparator.rs` untouched (with
   `derivation: None` throughout, none of items 2/2(c)/6's machinery can fire on any input in that
   file today). It is safe specifically because Amendment 1's signature check is unconditional and
   independent of it.

5. **Anti-vacuity is authorized as a mechanism, not as a specific number.** The Rust Engineer
   proposed a literal `report.leg1_rows_checked == 14`, but 14 is §16.27 item 1(f)'s three-page
   *census total*, not the ratified oracle page P01's own paired-block count, which is **3**
   (§16.27 item 4 / `docs/RULINGS.md` R3: "3 paired at `[0,0,0,0]` ... 1 `ClassDuplicateOf`, 1
   `CoverageFilteredUpstream`"). Fable's finding: neither agent's ruling actually derived a number
   for P01, so no number is ratified here. What is ratified: `ComparisonReport` gains a counter of
   pairs on which item 2's leg-1 row was actually evaluated; a new test exercises it non-vacuously
   against its own authored count in the same commit that adds it; and the real-page gate asserts it
   against a literal **authored (hand-derived, never artifact-derived)** at the atomic recording
   commit (cookbook rule 7).

6. **New tests required in the same commit** — the frozen file's null completion gives the new
   machinery zero coverage on its own, so these are not optional follow-ups: the leg-1 row firing
   (`ours.rect != rect_yolo`, exact), the leg-1 row present and passing non-trivially, item 2(c)'s
   structural guard firing, and — the case this whole ruling exists to protect — the signed-
   derivation mismatch firing in its silent-drop direction (signed `Some`, recorded `None`). §16.27
   item 7's four already-authorized additive reachable-shape controls (synthesized-corners,
   dominating-lines, multi-line, split-shaped-unmatched) may supply pass-side coverage but may not
   substitute for these fire cases, and per Amendment 1 their `ExpectedPair` literals must sign a
   real `derivation` value to exercise the axis at all.

7. **Grafted from the Architect's losing position, ratified because Fable found no defect in
   either:** §16.24 item 1(f)'s two conditions carried verbatim into this authorization — every
   existing value in the nine sites is retained verbatim, and every new value is a literal, never
   derived from a run — and a comment obligation at `bare_block` recording the deliberate asymmetry
   against the populated full-shape JSON exemplar
   (item 8 below).

8. **The full-shape JSON literal at `f1_oracle_comparator.rs:1749-1772`** (the exemplar whose own
   comment states *"a dropped optional field is the quiet direction of drift"*) gains the five new
   fields **populated**, with new assertions on `derivation`, `rect_yolo`, and `eng_expanded` —
   additive strengthening, not mechanical, since nothing forces it but leaving it stale would
   falsify the comment's own claim. This is the only Rust-side gate on the not-yet-written Python
   recording script, so the serde spelling of `Derivation`'s enum values — left to F1-C — must be
   pinned here in the same commit that chooses it.

**Deferrals, recorded rather than dropped (§16.23 item 1's rule).** (i) Whether an *unmatched*
upstream entry's signed `DbnetScattered` mechanism is verified against that block's recorded
`derivation` inside `compare()` is a residual gap in **both** agents' designs, named by Fable but
not closed; F1-C may propose a fix, but it needs its own ruling. (ii) `UpstreamBoxOutsideFrame`
stays untranscribed per §16.27 item 11 and is not licensed by this entry. (iii) The Rust encoding of
the eng-expansion trio (flat three fields vs. a grouped `Option<EngExpansion { .. }>` that makes the
"REQUIRED together" conditional unrepresentable-if-violated) is left to F1-C; both encodings satisfy
this entry's authorization identically.

## 16.29 F1 Phase 2 plan: identity operand, oracle-page location, DETECTOR_ORACLE.md sequencing (Fable tie-break, 2026-07-30)

The joint architect + rust-engineer planning pass for #12/#13 Phase 2 (the Python detector-oracle
recording script, the real-page comparison run, the atomic recording commit) was spawned per
`CLAUDE.md`'s Plan step, each blind to the other. They converged on the one question that actually
blocks writing code and disagreed on two smaller design points; per `CLAUDE.md`'s tie-break rule
`fable-adjudicator` reviewed both positions on those two and decided between them. The source
record — a condensed account of both agents' positions plus Fable's ruling captured near-verbatim —
is in `docs/RULINGS.md` under "F1 Phase 2 plan: identity operand, oracle-page location,
DETECTOR_ORACLE.md sequencing — Fable tie-break, 2026-07-30"; this entry transcribes the ratified
conclusions. This entry introduces **no supersession** — no prior `§16.x` ruling is overturned; it
settles two open design questions and records an agreed fix.

1. **Agreed premise, not adjudicated: the geometry identity's operand is `lines_pre_expand` when
   recorded, else the served `lines`.** Both agents independently found that
   `crates/pc-detect/src/oracle.rs`'s `lines_bbox`/`identity_branch`/`reconstruct` union the
   upstream-SERVED `lines` field for §16.20 item 3(b)'s identity, but §16.27 item 4 requires the
   ratified oracle page to pass "over `lines_pre_expand`" — on a real English-expanded block the
   served lines overhang `xyxy` (measured: two blocks by 2 px on one `y` edge, one block by 4 px on
   both `y` edges), so the identity as shipped would spuriously gate on the real page. Both propose
   the same fix: read `lines_pre_expand` when the artifact recorded it (`Some`), else fall back to
   `lines`. Both verified this moves **zero** existing frozen `ExpectedPair`/`OracleBlock` literal —
   every one today has `lines_pre_expand: None` or a value identical to `lines`. Both agree **no new
   `Divergence` variant is required**: a block that should have carried `lines_pre_expand` but
   didn't already self-gates via the existing `GeometryIdentity` variant, since the served bbox is a
   superset of the pre-expansion bbox at every block where the two differ (expansion only grows the
   polygon), so a block whose `lines_pre_expand` is absent falls back to unioning the larger served
   bbox — which cannot make the identity's residual smaller, and is measured non-zero on the actual
   P01 blocks where the two lists differ. The architect additionally
   rebuilt the pinned upstream checkout (`0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`) and ran the real
   instrumentation, empirically reproducing this exact failure mode and confirming the fix against
   the real P01 page — this is measured, not merely argued. `OracleBlock`'s doc comment must state
   this operand rule explicitly (§16.28 item 4's precedent: a `None`-disabling behavior must be
   stated, not merely implemented). **Correction, found during implementation and recorded here
   rather than silently fixed:** an earlier draft of this item claimed `IdentityBranch` also stays
   keyed on the served `lines`, unaffected by the operand switch. That is wrong and was never
   something either agent's plan asserted — `IdentityBranch` (`LineInformed`/`LineLess`) describes
   whether the identity itself had lines to union, so it must follow the **same** operand as
   `lines_bbox`/`reconstruct` (`lines_pre_expand` when recorded, else served `lines`); a block with a
   recorded-but-empty `lines_pre_expand` is `LineLess` for the identity even though its served
   `lines` is non-empty. Only `PairResidual::upstream_line_count` — the §16.27 item 8 census, which
   is a claim about what upstream serves — stays keyed on the served `lines` regardless of which
   list the identity unions.

2. **RATIFIED (Fable) — the committed oracle page moves; it is not duplicated.** The rust-engineer
   read the shipped validator's comment (`crates/pc-testkit/src/provenance.rs:334-347`, citing
   "§16.24 item 2 requires the page to live inside the declaring group") as over-claiming ratified
   text, and proposed committing a second, byte-identical copy inside
   `tests/fixtures/recorded/detector/` alongside the existing
   `tests/fixtures/upstream/oracle_pages/` original. Fable built a scratch probe crate calling
   `pc_testkit::provenance::validate` directly on two otherwise-identical detector groups differing
   only in `input_page`, and measured: the `oracle_pages/`-prefixed path fails with
   `CommittedPathOutsideGroup`; the `recorded/detector/`-prefixed path passes with zero violations.
   The two frozen gates behind that measurement are
   `crates/pc-testkit/src/provenance.rs:334-347` (`validate_detector`, pinned by
   `crates/pc-testkit/tests/provenance_schema.rs::r8_reaches_the_source_and_input_page_path_slots_too`'s
   last case) and `crates/pc-testkit/tests/recorded_provenance.rs:315-321`
   (`assert_known_non_committed_form`, which panics on a declared path that is neither scratch-
   prefixed nor a bare `*.pt.onnx` filename).
   **(QUALIFIED by §16.31 item 3(e) — the wording above is kept verbatim, and describes the
   predicate AS IT STANDS TODAY, unchanged. §16.31 item 3 RATIFIES, but does not yet implement,
   turning this bare-filename branch into an enumerated three-name allow-list — that edit is P7c
   implementation work, still owed (§16.31 item 7). Once it lands, a `*.pt.onnx` name other than
   `comictextdetector.pt.onnx` will no longer pass, and `encoder_model.onnx`/`decoder_model.onnx`
   will. Until then, the predicate above is accurate exactly as written: only a bare `*.pt.onnx`
   name passes, and neither OCR filename does yet. This item's decision and the scratch-probe
   measurement above are untouched either way: a path-shaped value fails under both the current
   and the ratified form of the branch.)**
   Fable, quoted:

   > The engineer's factual premise — "the prefix is only a wrong comment on an unenforced
   > constraint" — is therefore wrong. Given a copy inside the group is mandatory, the
   > `oracle_pages/` original would be a second 441,914-byte file that nothing consumes … and that
   > no test pins — pure drift surface and repo weight. §16.24 item 17 ratifies the page's identity
   > (441,914 B, sha256 `3bef9922…`, "unmodified") and an ATTRIBUTION entry; it ratifies no
   > filesystem path, so the move violates nothing.

   **Decision: P01 moves** (`git mv`, byte-preserving) from `tests/fixtures/upstream/oracle_pages/`
   to `tests/fixtures/recorded/detector/`, inside the atomic recording commit — measured: an early,
   `PROVENANCE.json`-less move trips `recorded_provenance.rs::discover_groups`'s `EXPECTED_GROUPS`
   assertion (a 4th group directory) before `read_provenance` is ever reached, so the move cannot
   precede the atomic commit — with `tests/fixtures/upstream/ATTRIBUTION.md`'s row updated to the new
   path and `docs/HANDOVER.md:51`'s now-stale reference corrected. **No second copy is committed
   anywhere.** P02/P03 stay in `oracle_pages/` (used only for the `DbnetScattered` census, never
   declared as `input_page`). Grafted from the losing position: the `provenance.rs:336-341` comment
   does over-cite — item 2's text is single-declaration only; the lives-inside-the-group requirement
   actually flows from item 1's preserved frozen walker plus item 2's walker-visibility mandate — so
   the comment's citation should be corrected (comment text only; this authorizes no change to the
   validator's behavior); and at the atomic commit, assert the recorded `input_page_sha256` equals
   item 17's ratified literal
   `3bef9922e09cea66ab12271da0070025768ae9bc5d286f41ced617468131267e`, so an accidental re-encode
   during the move fails loudly. **Scope:** this covers P01 only; it does not decide the destination
   filename (bytes and digest are pinned; the name is the planners' choice), does not move P02/P03,
   does not touch any other fixture group, does not amend §16.24 item 2's text, and authorizes no
   behavior change to any validator — the comment fix is prose only.

3. **RATIFIED (Fable) — `docs/DETECTOR_ORACLE.md` may land early as a dormant-but-verified
   skeleton; its verdicts, signatures, and the real-page gate test remain atomic.** The
   rust-engineer read §16.24 item 6's atomic list literally and proposed bundling the doc and its
   verdicts into the single atomic commit. The architect proposed landing the doc early as a
   skeleton (`OPEN` rows, no verdicts, no signatures) with item 11's four doc-shape gates in
   "dormant but verified" form, by analogy with `xtask/tests/provenance_digests.rs`'s
   `EXPECTED_DETECTOR_PIN_GROUPS = 0` pattern. Fable's ruling, quoted:

   > The ratified sentence, verbatim (spec lines 3970-3972): "The recording commit is atomic:
   > fixtures + provenance + item 1(f)'s constant edits + the real-page gate test with its literal
   > `Expectations` + the doc verdicts + the three signatures land in one commit. Until then no
   > real-page gate test exists." It names "the doc verdicts", not the doc file. The engineer's
   > literal-reading position is literal about a paraphrase: the brief's quotation of item 6 inserted
   > "`docs/DETECTOR_ORACLE.md`" and "in one commit or none" — neither string appears in the spec
   > (`grep -n "one commit or none"` hits nothing). This is the project's own widening-by-paraphrase
   > defect, operating on the adjudication input itself.
   >
   > Item 11's ratified gate reads "no `OPEN` row once the fixture is present" (spec line 4059). That
   > conditional is vacuous unless a doc with `OPEN` rows may exist while the fixture is absent — the
   > ratifiers contemplated exactly the pre-fixture state the architect proposes.
   >
   > The "dormant but verified" precedent is real and already accepted practice:
   > `EXPECTED_DETECTOR_PIN_GROUPS: usize = 0` at `xtask/tests/provenance_digests.rs:64`, with
   > `detector_pins()` asserting the discovered count against that literal before any early return,
   > and a comment (`:208-211`) explicitly distinguishing this from the conditional-return gate item 6
   > rejects — citing item 6 by name. I ran it: 6 tests green, the dormant legs printing their
   > `DORMANT:` disclosure.

   **Decision:** `docs/DETECTOR_ORACLE.md` may land early as a skeleton (rows `OPEN`, no verdicts,
   no signatures) together with item 11's four doc-shape gates in dormant-but-verified form. The
   verdicts, the three signatures, the fixtures, the provenance, the item 1(f) constant edits, and
   the real-page gate test with its literal `Expectations` land in the atomic commit, exactly per
   item 6. **Two binding conditions:** (a) every dormant doc gate must follow the `detector_pins()`
   pattern — the dormancy premise asserted against a pinned literal (e.g. `EXPECTED_SIGNATURES = 0`,
   raised to 3 at the atomic commit) **before** any early return; a bare
   `if !fixture_present { return }` is forbidden, since item 6 rejects "a conditional-return gate" by
   name. (b) "Until then no real-page gate test exists" stands unqualified — the early task may not
   contain the real-page comparison test in any form, not `#[ignore]`d, not dormant, not skeletal;
   only item 11's four doc-shape gates (field-column↔key-set equality both directions;
   `EXPLAINED-§x.y` anchors resolve; no-`OPEN`-once-fixture-present; three role-distinct non-self
   signatures) may land early. **Scope:** licenses landing early exactly two things — the doc file's
   skeleton and item 11's four gates in dormant-but-verified form; does not license landing any
   other item-6 member early; does not touch the maintainer's still-open signature-scheduling
   decision (§16.24, "Maintainer decisions (ii)"); does not reinterpret "the doc verdicts" at any
   other site the phrase appears.

4. **Ruling 1 rests on executed measurement; Ruling 2 rests primarily on reading ratified text,
   corroborated by a measured precedent — the two are not settled the same way, and neither
   transcription nor any future citation should flatten that difference.** Ruling 1's decisive
   grounds are Fable's probe crate and the two frozen gates it exercises (item 2 above). Ruling 2's
   decisive ground is textual: the ratified sentence at spec lines 3970-3972 names "the doc
   verdicts", not the doc file, and neither "`docs/DETECTOR_ORACLE.md`" nor "in one commit or none"
   appears there — `grep -n "one commit or none"` **over `docs/PIPELINE_SPEC_V1.md` specifically**
   returns zero hits, confirming the rust-engineer's brief had paraphrased item 6 rather than quoted
   it. That grep result is scoped to the spec, not the repo: the same string appears, pre-existing
   and unrelated to this ruling, in a doc-comment at `crates/pc-detect/src/oracle.rs:16` and in
   `docs/HANDOVER.md:69` (both themselves paraphrases of item 6, not the ratified text). Ruling 2's
   textual reading is corroborated by the measured, already-accepted `EXPECTED_DETECTOR_PIN_GROUPS`
   precedent (`cargo test -p xtask --test provenance_digests`, 6 passed at HEAD) and by
   `cargo test -p pc-testkit --test recorded_provenance` (1 passed at HEAD) for Ruling 1's timing
   consequence — but the precedent is corroboration, not the primary ground. Item 3's reading of
   "once the fixture is present" as contemplating a pre-fixture doc state is interpretation of
   ratified prose, flagged as such by Fable itself and recorded verbatim in `docs/RULINGS.md`.

## 16.30 P7's ONNX artifact pin, the vendored vocab, the P7/P8 split (Fable tie-break, 2026-08-02), and the beam-search decode correction (joint architects, 2026-08-02)

**SUPERSEDES: §16.12 item 4** — item 3 condition (ii): the "inactive until P7" promise is discharged by P8, not by P7.

**SUPERSEDES: §9.5** — item 4: the P7 row's "greedy decode loop" does not describe upstream.

**SUPERSEDES: §13** — item 4: row 30's "greedy decode" does not describe upstream.

Both table anchors are **bare-section** targets, per §16.26 item 3(c) — *"a bare `§N` target with no
item scopes to the whole section — the only available reading"*. The two P7 rows live inside
markdown tables, which carry no `^N. ` item marker, so no item-scoped anchor exists to name; §16.26
item 3(b) records the same leniency for `step N` anchors into §8.3, for the same structural reason.
The back-pointers therefore sit in the two table rows themselves, which is where a reader actually
lands today. **This placement is convention, not something the gate enforces**: because both
targets are bare-section anchors, the gate is satisfied by a back-pointer anywhere in §9.5's span
(lines ~826-839) or §13's span (lines ~1294-1336) — moving either annotation to a different row in
the same table would still pass `every_supersession_marker_has_a_back_pointer_at_its_target`. The
row-level placement is maintained for the reader's benefit, not because anything would go red if it
slipped.

The P7 planning pass was spawned per `CLAUDE.md`'s Plan step — `architect` and `rust-engineer`, each
blind to the other. They **disagreed on which third-party ONNX artifact to pin**, so per
`CLAUDE.md`'s tie-break rule `fable-adjudicator` was convened and ruled on three points (items 1-3
below). **Item 1 rests on independently executed evidence** — Fable downloaded and hashed every
candidate file and diffed tensors against the true upstream checkpoint, rather than reading
repository descriptions, in the spirit of CLAUDE.md's "by running it, not only by reading it."
**Items 2 and 3 do not share that grounding**: item 2 rests on a spec clause plus reasoned
argument (vendoring convention, CRLF-hazard elimination), and item 3 rests on a textual reading of
§16.12 item 4 and `setup.rs` plus repo precedent — neither involved downloading or hashing a
candidate. Item 4 was already unanimous **background** the two agents had independently reported
before Fable was convened — Fable's own reply notes the beam-search question "is unanimous
background per the brief" and explicitly declines to re-verify it, so it was not adjudicated and
did not arise only after the ruling. Both agents then finalized their plans under Fable's ruling on
items 1-3, **independently and unprompted, each reaffirming the same fourth conclusion** (item 4)
when asked to finalize.

**Item 4 is a joint-architect finding, NOT a Fable ruling, and the distinction is load-bearing.**
Nothing about item 4 was adjudicated, because nothing about it was disputed: two agents working
blind reached the same conclusion from the same measurements. It reaches this spec by the step-1a
path that any joint-architect conclusion takes — which is precisely the "consensus with no
adversary" case `CLAUDE.md` step 1a exists for. A future citation must not upgrade it to an
adjudicated ruling, and must not downgrade items 1-3 to agreed premises.

**Source record:** `docs/RULINGS.md`, under "P7 manga-ocr artifact, vocab, task split — Fable
tie-break, 2026-08-02", marked `record quality: MIXED` there — its blockquoted passage is the
adjudicator's reply captured directly (that file's own "for future rulings" instruction, followed
here for the first time), but the framing prose around it in that entry is the Orchestrator's, not
Fable's, exactly as every other passage in this spec entry outside an explicit quote is. Two
consequences a reader must still carry: nothing below may be cited as verbatim adjudicator text
except the passages explicitly marked as quotes (the source record's own blockquote is the place to
check a wider quote against), and **this transcription re-measured none of the tensor comparisons in
items 1-2** — those are recorded as the ruling's measured literals. The full sha256 digests and byte
counts in item 1's table are **not** in the source record verbatim (it carries only abbreviated
8-hex prefixes, e.g. `15fa8155…`) — they were independently confirmed against the pinned commit's
HF-hosted artifacts (`paths-info` at `24b12778d85800835e2ca409236de281b8ab7b9f`) as part of this
step-1a review, and item 2(a)'s in-repo sha256 gate is what re-checks the vocab at P7 implementation
time. The artifact
digests in item 1 have no in-repo gate yet; giving them one is P7's job, not this entry's.

1. **RATIFIED (Fable) — the pinned manga-ocr ONNX artifact.** P7 pins
   `mayocream/manga-ocr-onnx` at commit `24b12778d85800835e2ca409236de281b8ab7b9f`, taking two
   files:

   | file | sha256 | bytes |
   |---|---|---|
   | `encoder_model.onnx` | `15fa8155fe9bc1a7d25d9bb353debaa4def033d0174e907dbd2dd6d995def85f` | 343,454,249 |
   | `decoder_model.onnx` | `ef7765261e9d1cdc34d89356986c2bbc2a082897f753a89605ae80fdfa61f5e8` | 117,480,262 |

   **Grounds, measured rather than argued.** This artifact had earlier been rejected on the reading
   that it was "a finetune" of manga-ocr rather than an export of it. Fable retracted that reading
   by running the comparison: mayocream's decoder matches the real `kha-white/manga-ocr-base`
   checkpoint (sha256 `c63e0bb5…`, at `pytorch_model.bin`) **bit-exact on 41/41 tensors, including
   the full word-embeddings matrix** — the tensor a finetune would be most likely to move — and the
   encoder matches **bit-exact on 126/126 tensors**. The "finetune" reading was a reading of
   repository prose; the tensor diff is the thing that settled it.

   **Fable's own scope statement, quoted verbatim rather than paraphrased (truncated only at the
   point marked `…`, which is where the quote moves from this ruling's scope to a different topic —
   what Fable did and did not verify beyond the artifact, carried below rather than dropped):**

   > this ruling holds for the two fp32 files `encoder_model.onnx` and `decoder_model.onnx` at
   > commit `24b12778…` only, with the sha256s `15fa8155…`/`ef776526…`. It certifies nothing about
   > any quantized variant, any other commit of either repo, or any other file in mayocream's repo —
   > in particular **not** its vocab.txt, which this ruling explicitly declines to use. …

   That last clause is not decoration: item 2 takes the vocab from a different repository entirely,
   and it does so *because* this ruling declines to certify mayocream's copy. The quote's own next
   sentence, carried rather than dropped: Fable states plainly that it **"ran no actual inference on
   either export — behavioral identity is inferred from total weight identity plus identical graph
   I/O and identical generation configs, not from an executed OCR run."** That caveat applies to this
   item's conclusion as much as the scope sentence above it, and belongs beside it for the same
   reason the scope sentence does.

2. **RATIFIED (Fable) — `vocab.txt` is vendored, from kha-white, as a repo-tracked file, not a
   `ModelSpec` registry entry.** P7 vendors the vocabulary, sourced from `kha-white/manga-ocr-base`
   at commit `aa6573bd10b0d446cbf622e29c3e084914df9741`: **24,072 bytes, LF line endings, sha256
   `344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d`.** The destination path
   `crates/pc-ocr/assets/vocab.txt` is **not part of the ruling** — Fable's own scope note says this
   entry "does not decide the loading mechanism (`include_str!` vs path-based read) — that stays
   with the joint plan"; the path named here is the joint plan's choice, carried for concreteness,
   and remains open to the same review the rest of the plan gets. Whatever path is used, it should
   follow the existing `tests/fixtures/upstream/ATTRIBUTION.md` convention Fable's own ground (b)
   leaned on for third-party vendored fixtures — attribution is owed regardless of which directory
   the file lands in.

   **Two things it is explicitly NOT.** (i) Not mayocream's own `vocab.txt` (30,216 bytes), even
   though that file was measured to carry **the exact same 6,144-token vocabulary** — it is merely
   CRLF-terminated, and the arithmetic is exact: 30,216 = 24,072 + 6,144 line endings, and stripping
   `\r` yields a byte-identical sha256 to kha-white's file. The choice between them is therefore not
   a content decision at all; it is a decision about which byte sequence the repo pins, and the
   LF-terminated original is the one with a first-party source. (ii) Not a `ModelSpec` registry
   entry — the vocabulary is repo content, not a downloaded artifact.

   **Two binding conditions, both from the ruling.** (a) The vendored file needs a **sha256 gate
   in-repo**, so the pinned bytes cannot drift silently. (b) A **`.gitattributes` entry marking it
   `-text`**, so eol normalization can never rewrite the pinned bytes on checkout or commit — the
   30,216-vs-24,072 measurement above is exactly what that hazard looks like when it fires.

3. **RATIFIED (Fable) — P7 splits into P7 + P8, and P8 is mandatory for v1.0.** P7 becomes the
   `pc-ocr` backend only (**heavy**); a new **P8** carries the CLI/pipeline wiring and depends on
   P7. **Grounds, paraphrased from the source record (not a quotation):** `§16.12 item 4` and the
   CLI's WARN text currently promise OCR becomes reachable "until P7" / "v1 ships no OCR engine
   (task P7)" — a pc-ocr-only P7 leaves that promise broken with no successor task. The split is not
   a convenience; without it the spec would ship a completion condition that completing P7 does not
   satisfy.

   **Three binding conditions.**

   (i) **v1.0 is not done until BOTH P7 and P8 land.** P8 is mandatory, not optional follow-up.

   (ii) **§16.12 item 4's "until P7" sentence and the CLI's WARN-text expectations get supersession
   markers at BOTH this new entry AND those sites.** The spec half is discharged in this entry: the
   marker at the top of §16.30, and the back-pointer written into §16.12 item 4. **The code half is
   NOT discharged here** — the two live WARN strings (`crates/pc-cli/src/setup.rs:94`, the `clean`
   path, and `crates/pc-cli/src/lib.rs:110`, the `ocr` path, both citing `spec §16.12 item 4`) and
   whatever pins them are untouched by this ratification commit, which is spec-and-test bookkeeping
   only. That half lands with P8, and this sentence is the record that it is owed.

   (iii) **P8 gets its own row in the §9.5 and §13 task tables, depending on P7.** Deliberately
   **not** done in this entry: adding a task row is plan content, and the P7/P8 task breakdown has
   not been planned yet. The two table rows carry a back-pointer here saying so, so a reader who
   lands on the table learns the row is owed rather than assuming P7 is the whole job.

4. **JOINT-ARCHITECT FINDING (not Fable) — upstream decodes with beam search, not greedily.** Both
   agents independently found that this spec's literal words for P7 contradict measured upstream
   behaviour, and both proposed the same replacement. The decode strategy P7 ports is:

   `num_beams=4`, `length_penalty=2.0`, `early_stopping=true`, `no_repeat_ngram_size=3`,
   `max_length=300`.

   **Three independent confirmations, each checked by both agents.** (a) `kha-white/manga-ocr-base`'s
   `config.json` carries these five values at top level, and that repo has no
   `generation_config.json`, so HuggingFace derives the generation config from `config.json`.
   (b) `manga_ocr/ocr.py` calls `self.model.generate(x[None], max_length=300)` with **no `num_beams`
   override**, so it takes the config's 4 — the "greedy" reading came from the call site looking
   bare, and the call site is bare precisely because the config already carries the value.
   (c) the pinned mayocream export's own `generation_config.json` independently declares the same
   five values.

   **Measured, on manga-ocr's own 12 published labels: beam reproduces 12/12, greedy reproduces
   10/12.** Torch-greedy and ONNX-greedy agree with each other 12/12, which is what rules out a
   torch-vs-onnx numerical difference and localises the 2-label gap to the search strategy itself.

   **Why this is ratified rather than adopted quietly (Orchestrator's paraphrase — no committed
   source record exists for this reasoning, so it is deliberately NOT presented as a quotation):**
   this is a spec-vs-upstream conflict, and "greedy because beam is too hard" is not a live argument
   against fixing it — a beam port over the pinned ONNX files also reproduces 12/12, so the strategy
   is implementable, not merely correct in principle. The trade being accepted is real decode-time
   cost (measured, in-session, at roughly 1.4x) against 2 of 12 labels reproduced wrong — which is
   why this goes through ratification rather than being silently adopted as "the obvious fix."

   The two spec sites are annotated, not rewritten: §9.5's P7 row ("greedy decode loop") and §13's
   row 30 ("greedy decode") keep their original wording with a back-pointer here, per §16.19's
   convention.

5. **What this entry does NOT do, enumerated because an unenumerated omission reads as an
   oversight.** It adds no P8 row to §9.5 or §13 (item 3(iii)); it touches no source file in
   `pc-cli`, `pc-ocr`, `pc-pipeline` or `pc-models`, so the two WARN strings and their expectations
   still say "task P7" (item 3(ii)); it vendors no file and adds no `.gitattributes` entry or sha256
   gate (item 2(a)/(b) are P7's work); it re-measured none of items 1-2's digests or byte counts;
   and it does not decide P7's or P8's task decomposition, test plan, or ordering relative to any
   other task. Those are the P7/P8 planning pass's output, not a ratification's.

6. **This entry was run through the gate it feeds, per cookbook rule 14b.** `§16.26`'s Layer B
   scanner pairs a prose verb with the anchors on the *same physical line*, so every line above was
   written to keep the verb list off any line carrying a live older anchor — the one-sentence rule
   at §16.26 item 8(b). The three claims above are therefore Layer A markers and nothing here adds a
   Layer B row; `cargo test -p pc-testkit --test spec_supersession` is what proves that (12 passed
   at transcription time, after 13→16 on the pinned count). The pinned claim count and the three
   ratified-marker rows in `crates/pc-testkit/tests/spec_supersession.rs` are raised in this entry's
   own commit, as §16.26 item 6 requires. Each of the three markers was additionally falsified
   individually before the entry was handed over — its back-pointer stripped, the suite re-run, the
   red confirmed at the named site, the file restored — because a marker whose removal keeps the
   suite green is decoration (cookbook rule 6).

## 16.31 P7c infrastructure: the generic signature schema, the OCR pin's home, and the frozen non-committed-form predicate (Fable tie-break, 2026-08-02)

**SUPERSEDES: §16.29 item 2** — item 3(e) below only: that entry's parenthetical description of this predicate's accepted set ("a bare `*.pt.onnx` filename") stops describing it once item 3's enumerated allow-list lands. That entry's decision — P01 moves, no second copy — and the measurement behind it stand entirely.

P7c records the ONNX graph signatures of §16.30 item 1's two pinned manga-ocr artifacts. It has to
extend `xtask`'s existing detector-only signature infrastructure, whose `parse_dim` bails on
symbolic ONNX dimensions — and both OCR graphs are almost entirely symbolic. The P7c planning pass
was spawned per `CLAUDE.md`'s Plan step: `architect` and `rust-engineer`, each blind to the other.
They converged on two points and **diverged on three**, so per `CLAUDE.md`'s tie-break rule
`fable-adjudicator` was convened and ruled on all three (items 1-3 below).

**Items 1-3 are Fable rulings, not joint-architect findings, and the distinction is the one §16.30
set out.** Each was adjudicated because it was disputed. Item 4 records what the two agents had
already agreed before Fable was convened; Fable did not rule on it, and a future citation must not
upgrade it to an adjudicated ruling.

**Source record:** `docs/RULINGS.md`, under "P7c infra — signature schema, pin home, frozen
provenance predicate — Fable tie-break, 2026-08-02", marked `record quality: MIXED` there, same as
§16.30's source entry — that is the **second** entry captured under that file's own "for future
rulings" instruction to record the adjudicator's reply verbatim before condensing it (§16.30's
source entry was the first). The scope of the verbatim claim is narrower than a first glance at
either entry's blockquote might suggest, and travels with the citation: only the `>`-blockquoted
passage is Fable's own text, the sole edit to it being removal of a closing usage/token-count line.
The "Question put to Fable" framing and the closing note in that entry are the Orchestrator's prose,
and everything in this spec entry outside an explicit quote is this transcription's wording, not
Fable's.

**Grounding differs across the three rulings, by Fable's own statement, and flattening it would
misdescribe the record.** Rulings A and B (items 1-2) rest on execution: Fable downloaded and hashed
both artifacts, re-implemented the protobuf walk and parsed both graphs, applied the engineer's
refactor to a scratch clone and ran the workspace suite, and probed the serde behaviour. Ruling C
(item 3) rests on ratified text plus cookbook rule 13's corollary — Fable marks it "the
reasoning-based ruling of the three", noting "there is no runnable oracle for a governance-shape
choice". Upstream PanelCleaner was not consulted: Fable states none of the three is a
ported-behaviour ambiguity, and that "the tiebreak oracle for artifact facts here is the pinned
files themselves, which I ran."

**What this transcription re-measured, and what it did not.** It re-measured none of Fable's ONNX
graph readings, neither artifact digest, and none of the suite runs. Four cheap code references were
re-checked while transcribing, and **two** figures moved: `pc_models::COMIC_TEXT_DETECTOR` resolves at
`crates/pc-models/src/lib.rs:25-31` (Fable cited `:26-31`; the `pub const` is line 26 and its doc
comment line 25), and Fable's "`xtask/src/record.rs` has ~10" unit tests re-counts to **12**
`#[test]` functions — the point it supports, that unit tests in the bin crate are established
practice, is unaffected. The other two references confirmed, unchanged: `pc-testkit` is a
dev-dependency of `pc-ocr` (`crates/pc-ocr/Cargo.toml:31`); `crates/pc-ocr/tests/p7_signature.rs`
does not exist yet, as expected.

1. **RATIFIED (Fable) — Ruling A: the signature schema becomes generic; no parallel OCR type set.**

   **Decision.** `crates/pc-testkit/src/model_signature.rs` is restructured as `TensorSig<D>` /
   `ModelSig<D>` with four aliases — `TensorSignature = TensorSig<i64>`,
   `ModelSignature = ModelSig<i64>`, `SymbolicTensorSignature = TensorSig<Dim>`,
   `SymbolicModelSignature = ModelSig<Dim>` — plus
   `#[serde(untagged)] enum Dim { Fixed(i64), Symbolic(String) }`. A parallel
   `OcrModelSignature`/`OcrTensorSignature` type set is **not** created.

   **Grounds, measured by Fable rather than accepted from either plan.** Applied verbatim to a
   scratch clone at commit `1158809`, the refactor left both frozen consumers
   (`crates/pc-testkit/tests/model_signature.rs`, `crates/pc-detect/tests/d4_signature.rs`) passing
   **with zero edits to either file**; `cargo test --workspace` gave **881 passed / 0 failed / 6
   ignored**, identical to a same-tree baseline taken after reverting the probe; and
   `cargo clippy --workspace --all-targets --all-features -- -D warnings` was clean. That is §16.24
   item 1's own acceptance standard for touching this territory — "green **with the test file
   unmodified**" — met here by measurement. Fable's three further grounds: the architect's
   "field-for-field parallel so a future unification is a rename, not a merge" prepays permanent
   duplication to make cheap a unification that is available now at a measured cost of zero; the
   frozen file's own header (`crates/pc-testkit/tests/model_signature.rs:4-6`) states the
   anti-duplication ground, *"Duplicating the shape in either consumer would let the two drift…"*
   (the sentence continues by naming §16.13 item 3 as the reason `xtask` may consume this crate at
   all — elided here as a cross-reference, not as part of the ground being cited), and two
   field-for-field-parallel type sets inside `pc-testkit` are that same drift vector one
   module over; and the safety property the parallel set would buy is already held, because
   `the_named_convenience_loader_matches_the_general_one` asserts `shape == vec![1, 3, 1024, 1024]`
   through the alias, so flipping `ModelSignature`'s alias target away from `i64` is a compile error
   inside a frozen file.

   **Two binding grafts — conditions on the win, not optional follow-ups.**

   (a) The refactor lands **with additive tests**, because no existing frozen test covers the new
   surface: (i) `SymbolicModelSignature` round-trips a mixed symbolic/fixed shape; (ii) the `i64`
   alias still **rejects** a symbolic shape element, as the old `Vec<i64>` did by type, and must
   observably keep doing so; (iii) `Dim` rejects a JSON float. All three passed under Fable's probe;
   the implementer writes their own test text.

   (b) **§16.16 item 2 is not relaxed by this ruling.** The `model_signature` group's recorder keeps
   rejecting symbolic, dynamic, zero and missing dims for the detector artifact. If `xtask`'s
   low-level walk is shared between the detector and OCR recorders, the detector path's symbolic bail
   must remain **and be provable**; unit tests in the bin crate are established practice.

   **Scope, from the ruling.** It holds for the type definitions in
   `crates/pc-testkit/src/model_signature.rs` and the four aliases named above. It does **not** decide
   `xtask`-side code organization — a shared walk or a new parse path are both acceptable so long as
   graft (b) holds. It does **not** decide which module hosts the OCR group's path constants and
   loader: Fable notes that a new `ocr_model_signature.rs` module holding constants, loader and pins
   **built on the shared generic types** would keep the architect's module boundary without the type
   duplication, and calls that compatible, not required. And it authorizes **no new schema fields**,
   `ir_version` included — "neither position proposed it; adding one needs its own argument."

2. **RATIFIED (Fable) — Ruling B: one `OcrModelPin` constant per file, in `pc-testkit`, with two
   importers.**

   **Decision.** The encoder and decoder pins (file name, sha256, size_bytes) live **once**, as
   constants in `pc-testkit` — under the type name `OcrModelPin`, which is Fable's own — imported by
   the `xtask` OCR recorder, which verifies the real file against the pin **before** parsing, as
   `xtask/src/model_signature.rs:76` does for the detector today, and by
   `crates/pc-ocr/tests/p7_signature.rs`. The two constant identifiers, `MANGA_OCR_ENCODER` and
   `MANGA_OCR_DECODER`, are carried over from the architect's plan rather than named by Fable —
   ratified as the mechanism (one constant, two importers), not as a naming requirement; an
   implementer is free to name them differently so long as the mechanism holds. Both dependency
   edges already exist: `xtask` depends on `pc-testkit`, and `pc-testkit` is a dev-dependency of
   `pc-ocr`.

   **The load-bearing factual question, settled by Fable reading the code rather than the
   transcripts.** The existing pattern is one constant, not two independently-typed literal copies.
   `pc_models::COMIC_TEXT_DETECTOR` is defined once and imported by the recorder, by the H2 gate
   (`xtask/tests/provenance_digests.rs::the_model_signature_source_digest_equals_the_pc_models_constant`)
   and by `pc-models`' own keystone, while `crates/pc-detect/tests/d4_signature.rs:62-69`
   deliberately carries no digest literal — its own inline comment there names the detector model
   task (`§8.3 step 3`) and the cross-crate dependency rule keeping `pc-detect` off `pc-models`, not
   §16.16 item 5 by number, but the substance is the same fact §16.16 item 5 states (spec
   `:2804-2805`: *"the single source of truth for model identity lives in `pc-models`"*). The
   engineer's contrary characterisation of the existing pattern was factually wrong about the digest,
   and the main ground for that position goes with it.

   **Binding graft, taken from the losing position.** The existing pattern is not constants-only:
   `d4_signature.rs:73-74` pins `size_bytes == 94_669_756` and `opset == 11` as **frozen-test
   literals**. Without that leg, one edit to a pin constant plus a re-record against a different real
   file keeps every test green, and nothing frozen anchors *which* artifact is pinned. So
   `crates/pc-ocr/tests/p7_signature.rs` must assert the pin constants' identity fields (sha256 and
   size_bytes) against §16.30 item 1's ratified literals, in the frozen test. That is duplication
   **with a comparator between the copies**, which is the H2 pattern — unlike the rejected proposal,
   in which the `xtask`-local and test-local copies were compared by nothing and a drift would
   surface only at the next re-record.

   **Scope, from the ruling, plus the obligation Fable asked be recorded with this transcription.**
   It holds for the two P7 pins (encoder/decoder sha256 + size) and their two named consumers: the
   `xtask` OCR recorder and `crates/pc-ocr/tests/p7_signature.rs`. It does **not** decide P8a's
   `ModelSpec` design. **Recorded here because the ruling directs that it be:** when P8a moves model
   identity into `pc_models`, it must either retire the `pc-testkit` pin or add an identity assertion
   binding the two — leaving both unbound would recreate exactly the drift this ruling exists to
   prevent. That obligation is owed by P8a and is not discharged here.

3. **RATIFIED (Fable) — Ruling C: an enumerated three-name allow-list in the frozen predicate, plus a
   compensating digest gate bound to the pin constants. A synthesis; neither position as stated was
   complete.**

   **Decision, two halves, both binding.** (a) `assert_known_non_committed_form`
   (`crates/pc-testkit/tests/recorded_provenance.rs`) has its `.ends_with(".pt.onnx")` suffix test
   replaced by an **enumerated allow-list of exactly three bare filenames**:
   `comictextdetector.pt.onnx`, `encoder_model.onnx`, `decoder_model.onnx`. (b) Additively, in
   `xtask/tests/provenance_digests.rs`, a new **bidirectional** gate asserts set equality between
   every bare-model-filename source declared anywhere in the tree and that three-name list, with each
   one's `source_sha256` bound to its **pin constant** — `pc_models::COMIC_TEXT_DETECTOR.sha256` for
   the detector, which is already H2, and item 2's `OcrModelPin` constants for the other two — and
   **not** to a second independently-typed literal table.

   **Grounds.** Against widening the suffix `.pt.onnx` → `.onnx`: cookbook rule 13's corollary is
   near-verbatim on point — *"`ignore dotfiles` or an extension allowlist would have re-opened it
   silently… the exempt set is itself asserted: a stray file must not pass by merely looking
   exemption-shaped."* Widening the suffix converts a narrow accident of the detector's filename into
   a genuine extension allowlist, and §16.24 item 1(e) already called loosening this exemption "the
   bypass wearing different clothes". Against the allow-list as the architect left it: membership
   without a digest binding lets `encoder_model.onnx` be declared with a garbage `source_sha256` and
   still pass the frozen gate, which never hashes non-committed sources. The engineer's objection was
   therefore correct, but its dichotomy was false — the compensating leg is not intrinsic to suffix
   widening and attaches to an allow-list just as well. One correction to that leg: a literal digest
   table in the xtask test would re-introduce the two-copies-no-comparator shape item 2 rejects, so
   it binds to the pin constants instead, and item 2's frozen-test literal leg is what anchors the
   constants themselves.

   (c) **Fable's direction claim, quoted, and the exact set relation the quote does not state.**
   Fable writes that an enumerated set "is the *opposite* move — it is **stricter than today's
   predicate**, not laxer, which is the direction every ratified precedent here prefers." Checked
   against the code while transcribing, that holds of the *pattern* but not of the *accepted set*,
   and the difference is worth one sentence rather than a future reader's surprise: today's branch
   accepts every bare `*.pt.onnx` filename, the new one accepts exactly three names, so the new set
   **drops** every `*.pt.onnx` name other than `comictextdetector.pt.onnx` and **admits** two names
   the old branch rejected. It is neither a subset nor a superset of the old one. What "stricter" is
   true of is closure — the accepted set stops being open-ended — and closure is the property the
   grounds above actually rest on.

   (d) **Housekeeping this ruling explicitly authorizes.** The comment at
   `crates/pc-testkit/tests/provenance_schema.rs:587-590`, which describes the exemption as "keyed on
   a single path component ending `.pt.onnx`", goes stale under (a) and gets a **comment-only**
   correction, authorizing no behaviour change — the same class as §16.29 item 2's precedent for a
   code comment that over-cites. Separately, the `EXPECTED_GROUPS` / `EXPECTED_DECLARED_PATHS` /
   `EXPECTED_COMMITTED_PATHS` additions the new OCR group's fixtures require are **already
   authorized** additive edits, extending §16.24 item 1(f)'s clause — quoted with its own scope,
   since that clause's stated trigger is the detector group, not this one: *"(f) When the detector
   group records, `EXPECTED_GROUPS` / `EXPECTED_DECLARED_PATHS` / `EXPECTED_COMMITTED_PATHS`
   (`:11-27`) gain the new entries. This is an authorised additive edit, invited by the test's own
   message (`:85`)..."* — on the same two conditions ratified there (every existing entry retained
   verbatim; the constants stay literal). The clause's own ground for being additive-not-ratified is
   the test's message itself (`"recorded fixture group count changed; inspect this test and update
   its expected count"`), which is group-agnostic wording, not detector-specific — that is why this
   entry extends it to the new OCR group rather than treating it as already covering one; if that
   reading is wrong, this paragraph is the thing to correct, not the underlying edit.

   (e) **The one spec site whose description this changes, and the two it does not.** §16.29 item 2
   describes this predicate as one "which panics on a declared path that is neither scratch-prefixed
   nor a bare `*.pt.onnx` filename". Under (a) that stops being an accurate description of the live
   predicate, so that site keeps its original wording and carries a back-pointer here, per §16.26's
   convention; the marker at the head of this entry is the other end of it. **This site was found
   while transcribing, not by the ruling** — Fable's prior-ratification check named §16.24 items
   1(e), 1(f) and 2 and cookbook rule 13, and did not reach §16.29 item 2. The other two spec sites
   naming this predicate are unaffected and carry no marker: §16.24 item 1(e)'s prohibition on adding
   upstream-path digests to the provenance files holds unchanged, because a path-shaped value still
   fails under either form of the branch; and §16.24 item 2's "the bare-model form is exactly the
   deliberately-unverifiable bucket" stays true of all three names.

   **Scope, from the ruling.** It holds for `assert_known_non_committed_form`'s bare-model-filename
   branch and the three named files. It does **not** widen the scratch-prefix branch, does **not**
   touch the key-name walker (§16.24 item 1(b): neither deleted nor patched), and does **not**
   pre-authorize any fourth filename — a future model file needs another explicit allow-list edit
   plus its own pin, by design.

4. **AGREED BACKGROUND, not adjudicated — recorded so a future citation does not upgrade it.** Three
   premises were common ground and are part of no ruling above. (i) The OCR signatures go into a
   **new provenance group** rather than widening the existing `model_signature` group — both agents
   converged on this independently, and it appears in the source record's "Question put to Fable"
   framing, which is the Orchestrator's prose, not Fable's. (ii) An in-place `Vec<i64>` → `Vec<Dim>`
   mutation of `TensorSignature` is rejected: both agents compiled it against the real frozen suite
   and got real `rustc` errors, and Fable accepted that without re-executing it. (iii) "Pins must not
   live in `pc_models` before P8a" is agreed ground, likewise accepted rather than measured. The name
   of the new group is not fixed by this entry.

5. **Verified versus accepted, in Fable's own split, because the two are not settled the same way.**
   Verified by execution: both artifacts' digests and sizes, re-derived and matching §16.30 item 1's
   table exactly; both graphs' full I/O signatures, including that the frozen `parse_dim` bails on
   both files; the generic refactor's 881/0/6-with-zero-test-edits against an identical same-tree
   baseline, with clean clippy; the `Dim` serde semantics (number → `Fixed`, string → `Symbolic`,
   round-trip stable, the `i64` alias rejecting a symbolic string, a JSON float rejected); and the
   single-constant pin pattern. Accepted without re-execution: items 4(ii) and 4(iii).

   Fable's measured graph facts, recorded because item 1's grafts are written against them: the
   encoder has 198 initializers, one true input `pixel_values` f32
   `[batch_size, num_channels, height, width]`, and output `last_hidden_state` f32 with three
   symbolic dims — **7 symbolic dims and zero concrete across its I/O**; the decoder has two true
   inputs (`input_ids` i64, both dims symbolic; `encoder_hidden_states` f32 `[sym, sym, 768]`) and
   output `logits` f32 `[sym, sym, 6144]`; both files declare **opset 14 and ir_version 7**.
   Recording `ir_version` is not authorized — see item 1's scope.

6. **Mode and integrity statement, recorded because the tie-break rule turns on it.** Fable's reply
   opens with a mode statement: Mode A, tie-break, final call, advisory-only status suspended for
   these three decisions; no files written in the repository; all probes run in a scratch clone and a
   scratch crate; `git status --porcelain` on the working tree empty. Fable also ran the
   prior-ratification check before ruling and reported that §16.30 reserves this ground rather than
   deciding it — "The artifact digests in item 1 have no in-repo gate yet; giving them one is P7's
   job, not this entry's" — that §16.16 item 2's reject-symbolic clause binds the `model_signature`
   **group's recorder** rather than the Rust representation of signatures generally, and that no
   ratified clause decides A or B.

7. **What this entry does NOT do, enumerated because an unenumerated omission reads as an oversight.**
   It writes no code: no `Dim` enum, no `TensorSig<D>`/`ModelSig<D>` refactor, no `OcrModelPin`
   constants, no `crates/pc-ocr/tests/p7_signature.rs`, no edit to `assert_known_non_committed_form`,
   no new gate in `xtask/tests/provenance_digests.rs`, and no correction to the
   `provenance_schema.rs` comment — each is P7c implementation work, owed and not done. It adds no
   provenance group, no `EXPECTED_GROUPS` entry, no fixture and no recorded signature. It does not
   name the new group, decide the `xtask` module layout, or decide P7c's task decomposition, test
   plan or batching — those are the P7c planning pass's output, not a ratification's. It decides
   nothing about P8a beyond recording item 2's obligation, and it re-measures none of item 5's
   figures.

8. **This entry was run through the gate it feeds, per cookbook rule 14b.** §16.26's Layer B scanner
   pairs a prose verb with the anchors on the same physical line, so every line above was written to
   keep the verb list off any line carrying a live older anchor — the one-sentence rule at §16.26
   item 8(b). The single claim above is a Layer A marker and this entry adds no Layer B row;
   `cargo test -p pc-testkit --test spec_supersession` is what proves that. The pinned claim count
   and the ratified-marker set in `crates/pc-testkit/tests/spec_supersession.rs` are raised in this
   entry's own commit, as §16.26 item 6 requires. The marker was falsified before handover — its
   back-pointer stripped, the suite re-run, the red confirmed at the named site, the file restored —
   because a marker whose removal keeps the suite green is decoration (cookbook rule 6).

## 16.32 The detector's CPU execution provider: denormal flushing plus thread confinement (Fable rulings, 2026-08-03 and 2026-08-04; transcribed 2026-08-04 — see docs/RULINGS.md)

1. **The surviving record, quoted not paraphrased.** This entry is transcribed from the
   institutional record of the pipeline run plus commit `3801b32`'s message, the only committed
   artifact that contains the original ruling's actual language. The reconstruction caveat — and
   the fact that neither subagent's position is recorded — is documented in docs/RULINGS.md,
   Entry 1; this entry does not assign positions. The commit message says:

   > The flag has a real hazard: ONNX Runtime sets DAZ/FTZ on the constructing thread and never restores it, non-deterministically depending on thread-scheduling order — a silent, order-dependent float-semantics change for every other pure-Rust pipeline stage sharing that thread. A Fable adjudicator ruling (independently reproduced by the ruling itself) required the flag ship only bundled with thread confinement.
   >
   > Thread confinement: `OnnxDetector` moved off the old `Mutex<Session>`
   > (§14.15/DEVIATION(15)) onto a dedicated worker thread that owns session
   > construction and inference exclusively, confining the flag's CPU-register side
   > effect. Preprocessing/decoding stay on the caller (a first cut wrongly ran them on
   > the worker, reintroducing the exact whole-run-poisoning regression §16.19 item 10
   > was ratified to prevent — proven and fixed via panic injection). Panic payloads now
   > cross the thread boundary as raw `Box<dyn Any + Send>` and get `resume_unwind`'d on
   > the receiving side, preserving the pinned `"panicked: <payload>"` rendering
   > (§16.19 item 4, §16.12 item 18, §5.2 item 2) without the double-prefix bug a naive
   > re-formatting fix would have caused (proven via a probe before implementing).
   >
   > 3. Flip the shipped default: `SessionTuning::default().flush_denormals` is
   > now `true`. A normal `OnnxDetector::from_path(...)` call — no special
   > configuration — now completes `detect()` in ~0.6s instead of ~17-20s.

2. **What changes going forward.**

   (a) The shipped default is `SessionTuning::default().flush_denormals == true` in
   `crates/pc-detect/src/onnx.rs`. A plain `OnnxDetector::from_path` now flushes. The pinning
   tests are `crates/pc-detect/tests/perf2_tuning.rs`'s default test, which runs by default, and
   `crates/pc-detect/tests/perf2_from_path_smoke.rs`, whose only test is `#[ignore]`d and opt-in,
   requiring `PANEL_OCR_ONNX_MODEL`, real ONNX weights, and ONNX Runtime.

   (b) The worker-thread invariant is exact: one `Session` per `OnnxDetector`, constructed and
   run only on a dedicated thread named `pc-detect-onnx`; preprocessing (`letterbox`, `to_nchw`)
   and decoding (`decode_outputs`) run on the CALLING thread. The split matters because a first
   cut ran preprocessing on the worker too, reintroducing the exact whole-run-poisoning regression
   §16.19 item 10 was ratified to prevent — found via panic injection, not by reading the diff.
   Of `crates/pc-detect/tests/perf2_mxcsr_hygiene.rs`'s four tests, the three opt-in tests —
   `flush_denormals_does_not_escape_the_detector_worker_thread`,
   `preprocessing_panic_is_per_image_and_does_not_kill_the_detector_worker`, and
   `inference_panic_is_per_request_and_does_not_kill_the_detector_worker` — are `#[ignore]`d and
   require `PANEL_OCR_ONNX_MODEL`, real ONNX weights, and ONNX Runtime; only
   `session_build_panic_reaches_caller_with_original_payload` runs by default, and it does not
   pin the worker/caller split.

   **SUPERSEDES: §16.19 item 10**
   **SUPERSEDES: §4.5**
   **SUPERSEDES: §14 item 15**

   (c) The panic-forwarding contract is raw and caller-rendered: the worker catches panics via
   `catch_unwind(AssertUnwindSafe(..))`, forwards the raw `Box<dyn Any + Send>` payload unformatted,
   and the receiving thread calls `resume_unwind`. The existing correct renderers —
   `pc-cli/src/detector.rs` for init panics per §16.19 item 4, and
   `pc-pipeline/src/batch.rs`'s `process_image_isolated` for per-image panics per §5.2/§16.12
   item 18 — therefore produce the pinned `"panicked: <payload>"` string exactly once, with no
   double-prefix. This was proven via a live probe before implementation; a naive re-format-on-
   worker approach would double-prefix.

3. **§16.21 item 6 closure:** see §16.21 item 6, closed by this entry.

4. **The authorised comment-only corrections enumeration.** This licenses task 29b's edits to
   the two FROZEN test files, and is an explicit enumeration:

   - `crates/pc-detect/tests/d4_session.rs`: five stale comment regions — the stale
     `intra_threads=1` rationale citing the old Mutex; the false "nothing in this file can run in
     this checkout" claim — MEASURED FALSE: `cargo test -p pc-detect --features onnx,testkit
     --test d4_session` runs 2 of 3 tests and passes both, only the real-weights test is
     `#[ignore]`d; the Mutex→worker-thread concurrency-note rationale; the falsified "if the
     wrapper is ever removed this stops compiling" claim — MEASURED FALSE: the Mutex wrapper WAS
     removed and it still compiles, because `mpsc::Sender` is `Sync` on the pinned toolchain; and
     the DEVIATION(15) rationale in the 4-thread smoke test comment.
   - `crates/pc-detect/tests/d4_onnx.rs`: two stale comment regions — the Mutex-serializes-
     inference rationale near the constants test; and a separate false "anything gated on that
     feature cannot even link, let alone run" claim near the top of the file — also measurably
     false the same way.
   - Precedent: commit `cecb9f1` ("Final review D4/D10: correct four stale doc comments") already
     established that correcting stale DOC COMMENTS (never assertions, literals, test names, or
     `#[ignore]` reasons) in frozen test files of this class does not require fresh joint-architect
     escalation.
     No assertion in either file may change; the correction is licensed for `//` and `//!` text only.

   `crates/pc-detect/src/onnx.rs`: the `DEVIATION(15)` marker was missing before this change and
   has been restored at its current location, on the `impl OnnxDetector` doc comment block that
   already explains the worker design, as required by §14's own closing convention ("Each of
   these must appear as a `// DEVIATION(n): ...` comment at the implementation site").

5. **PROVENANCE.json — recorded as an explicit deferral, per docs/RULINGS.md Entry 2 (Ruling 1).**
   The committed `tests/fixtures/recorded/detector/PROVENANCE.json` was recorded at commit
   `f6212eb`, before
   `SessionTuning` existed — so it implicitly reflects `flush_denormals=false`, which is the true
   historical fact. `xtask/src/record.rs` currently writes `"intra_threads": 0` and
   `"inter_threads": 0` as HARDCODED LITERALS rather than observed values (an existing defect
   against §16.21 item 5's "records the thread count actually used... facts, not inferences"
   principle). Fable ruled: do not add a `flush_denormals` field now (it would sit beside already-
   wrong literals and could only be added to the committed file by hand-edit, since no recorder in
   the tree would currently emit `false` for it correctly at HEAD — HEAD's recorder would emit
   `true`). Binding requirement: the NEXT real re-record (the one §16.23 item 5 already budgets,
   for DBNet lines) must land both the recorder fix (write observed values, not literals) and the
   `flush_denormals` schema field in the same change.

6. **§16.24 item 19(c)'s reader/writer table gap.** `xtask/src/ocr_model_signature.rs` is the
   third `PROVENANCE.json` writer and also a reader missing from that table: it deserializes an
   existing group and constructs/writes a `GroupProvenance` for the `ocr_model_signature` group.
   This was found during this transcription by re-running item 19's own prescribed enumeration
   untruncated; the table now records both roles.

7. **One explicitly recorded gate.** A gate now binds pc-cli's PRODUCTION feature resolution to
   exclude pc-detect/bench-tuning at the manifest-authoring level, formally verified end to end by
   `crates/pc-detect/tests/perf2_feature_containment.rs`. It is scoped to the normal/build
   dependency graph only, per docs/RULINGS.md Entry 2 (Ruling 2) — workspace test builds ARE tainted
   by pc-detect's own self-dev-dependency on itself with bench-tuning enabled, which is a separate,
   known, out-of-scope fact, not a defect this gate is meant to catch.

## 16.33 Windows becomes a supported platform at v1.1 (maintainer ratification, 2026-08-04; joint architect + Senior Rust Engineer plan pass)

**Status, stated first because it changes how to read the rest: this entry transcribes a PLAN and
its ratifications, not accomplished work.** At the time it lands, no source-side implementation
exists — `Platform`, `EnvSource`/`ProcessEnv`/`MapEnv`/`DirEnv`, `Shell`, `resolve_editor`, the
new `.gitattributes` rows and the CI matrix row are all owed by the TDD task that follows.
Recorded this way rather than as fact, following §16.27 item 9's precedent of enumerating
authorised follow-up work explicitly instead of leaving it implicit.

The item-15 test files are written first, and their status was **measured on 2026-08-04**, not
assumed — the first version of this preamble claimed none of the seven compiled, which was wrong
for four of them:

| file | status against today's source |
|---|---|
| `crates/pc-cli/tests/x1_platform_paths.rs` | does not compile (`E0432`: no `Platform`/`DirEnv`/`MapEnv`/`ProcessEnv` in `paths`) |
| `crates/pc-cli/tests/x1_shell_quoting.rs` | does not compile (`E0432`: no `Platform`/`Shell` in `paths`) |
| `crates/pc-cli/tests/x1_editor.rs` | does not compile (`E0432`: no `Platform`/`MapEnv`, no `resolve_editor`) |
| `crates/pc-testkit/tests/platform_claim_sites.rs` | compiles; 4 pass, 2 fail (four sites do not yet cite §16.33) |
| `crates/pc-testkit/tests/eol_normalisation.rs` | compiles; 6 pass, 1 fail (the seven `-text` rows are not in `.gitattributes` yet) |
| `crates/pc-testkit/tests/ci_matrix.rs` | compiles; 4 pass, 1 fail (`ci.yml`'s default tier lists no Windows runner) |
| `crates/pc-testkit/tests/verbatim_paths.rs` | compiles; **all pass**, because both documents it gates were written by this entry |

`verbatim_paths.rs` being green on arrival is worth stating plainly rather than glossing: it is
currently a gate against *future* drift and it proves nothing about undone work. Its own history is
the reason to distrust an all-green documentation gate — its first version passed a mutation that
reverted E1 (see the file's `RATIFIED_RESOLUTION_ROWS` doc comment), and it was rewritten to match
whole table rows because of it.

**This preamble does not exempt anything from CLAUDE.md's commit bar, and cannot.** That document
governs when work is committable; this one cannot grant itself a waiver from it. The plain fact is
only that tests-written-first leave `cargo test --workspace` non-building (four `E0432`s across the
three files above — `x1_editor.rs` raises two, one for its `paths` imports and one for
`resolve_editor`)
and `clippy` failing the same way, for the documented reason that the interface in item 5 does not
exist yet. Nothing from this pass is committed in that state: the commit happens after the TDD loop
below, once Codex's implementation makes the workspace green again — the normal
tests-first → implement → verify → commit sequence, with the bar applied at the end of it, not
redefined at the start.

1. **What was ratified, and what needed no ruling.** Three open escalations from the plan were
   decided by the maintainer on 2026-08-04; three were already settled by independent convergence
   of the two Opus subagents and are recorded here rather than re-argued. There was **no
   architect/Rust-Engineer disagreement on any design shape** — both passes independently
   produced a pure platform-parameterized resolver over an injectable environment, no `dirs`
   dependency, and a shell-flavor enum with a target-parameterized constructor plus a host
   constant — so Fable was not convened, and this is a consensus transcription, which is
   precisely the class §16.24's process notes and CLAUDE.md step 1a treat as the least-adversarial
   and therefore most defect-prone.

   | escalation | subject | disposition |
   |---|---|---|
   | E1 | cache/config root on Windows | **RATIFIED (maintainer):** `%LOCALAPPDATA%` for cache, `%APPDATA%` for config — the OS-convention split. Registered as §14 item 19 / `DEVIATION(19)`; see item 3. |
   | E2 | shell flavor for pasted recovery commands | Settled by convergence: PowerShell only. `cmd.exe` explicitly out of scope; see item 6. |
   | E3 | XDG precedence | Settled by convergence: XDG variables are honoured first on **every** platform, Windows included; see item 4. |
   | E4 | `xtask` scope on Windows | **RATIFIED (maintainer):** compile plus its non-model tests only; see item 9. |
   | E5 | editor fallback | Accepted as scoped: `notepad.exe` on Windows only; see item 7. |
   | E6 | `onnx` feature tier on Windows | **RATIFIED (maintainer):** ship the default tier now regardless of the `ort` Windows linking issue; see item 10. |

2. **The platform-claim-site enumeration — measured on 2026-08-04, and the measurement itself
   found a defect in how it was first taken.** Cookbook rule 14 requires enumerating every reader
   of a shared claim and recording the enumeration. The shared claim here is "this project targets
   Linux and macOS". A line-based `grep -rn "Linux + macOS"` over the tracked tree returns **7**
   hits and **misses an eighth**: §16.12 item 21's claim wraps across two physical lines
   (`... and v1 is Linux +` / `macOS only). ...`), so no line contains the phrase. Re-running the
   scan over whitespace-flattened file text returns **8**. The wrapped one is the single most
   important site in the list, because it is the clause that fixes the cache and config roots.
   That is why `crates/pc-testkit/tests/platform_claim_sites.rs` (item 15) flattens whitespace
   before scanning and carries a synthetic wrapped-claim control: the naive form of this check was
   measurably blind to the site that mattered most.

   The eight occurrences sit in **five** files. The table below has seven rows because it is keyed
   by *location within a file*, not by file — `docs/PIPELINE_SPEC_V1.md` carries three of them in
   three different places, and `README.md` carries two in one row. Counted three ways, so no reader
   has to guess which number is which: **8** occurrences, **7** locations, **5** files. (The gate in
   `crates/pc-testkit/tests/platform_claim_sites.rs` holds **6** files, which is a fourth number and
   deliberately not any of these three: it adds `docs/ARCHITECTURE_DECISIONS.md`, which carried zero
   occurrences of the claim and is included on the separate ground given below.)

   | location | occurrence(s) | what it governs |
   |---|---|---|
   | `docs/PIPELINE_SPEC_V1.md` line 6 | 1 | the "Fixed constraints" preamble, which forwards to `docs/ARCHITECTURE_DECISIONS.md` |
   | `docs/PIPELINE_SPEC_V1.md` §16.12 item 21 | 1 (wrapped) | the cache/config default roots and the "no new third-party dependency" argument |
   | `docs/PIPELINE_SPEC_V1.md` §16 out-of-scope list | 1 | the global out-of-scope bullet |
   | `README.md` lines 68, 99 | 2 | the user-facing platform statement and the roadmap checkbox |
   | `.github/workflows/ci.yml` line 8 | 1 | the CI matrix comment, which cites `docs/ARCHITECTURE_DECISIONS.md` |
   | `crates/pc-cli/src/paths.rs` line 3 | 1 | the module doc justifying hand-rolled directory discovery |
   | `crates/pc-cli/tests/x1_args.rs` line 363 | 1 | the doc comment justifying the `unix` gate on the paste-safety test (the comment's own prose says `#[cfg(unix)]`; the attribute is actually `#[cfg(all(feature = "onnx", unix))]`) |

   Two of those sites — the spec preamble and `ci.yml` — cite `docs/ARCHITECTURE_DECISIONS.md`
   for the platform list, and that file **had no platform-support section at all**: its "Fixed
   constraints" role was asserted by its readers and never discharged by the document. This pass
   adds one (`docs/ARCHITECTURE_DECISIONS.md`, "Platform support"), so both citations resolve to
   real text. That absence is itself the reason the claim drifted across five files with no single
   owner. It is also why `platform_claim_sites.rs` gates six files rather than the five measured
   ones: a gate on the citers that left the cited authority free to stay silent would protect the
   wrong end.

   `crates/pc-cli/tests/x1_args.rs` is a FROZEN test file. **This entry authorises a
   COMMENT-ONLY edit there** — adding a `§16.33` pointer to the doc comment at line 363, whose
   stated justification stops being true at v1.1. That justification, **re-flowed** (in the source it
   wraps across a `///` line break after `macOS`, so this is not a byte-verbatim quotation):
   "v1 is Linux + macOS only, so a POSIX shell is always present". No assertion, no `#[cfg]` attribute, no test name and no fixture in that
   file may change: the test stays `#[cfg(all(feature = "onnx", unix))]` and keeps asserting
   exactly what it asserts today. Recorded as an authorisation rather than done silently, because
   cookbook rule 8's three exits from a frozen test do not include "the comment went stale".

3. **E1 — RATIFIED: the OS-convention split, and it is a deliberate divergence from upstream.**

   **SUPERSEDES: §16.12 item 21**

   The scope of what is replaced, quoted verbatim rather than paraphrased — **as item 21 read
   before this entry**, since this entry also amends the live clause in place with a forward
   reference, so a character-compare against today's §16.12 item 21 will differ from the quotation
   below by exactly that inserted reference: *"Default cache
   location: `$XDG_CACHE_HOME/panel-ocr` (Linux) or `~/Library/Caches/panel-ocr` (macOS), falling
   back to `./.panel-ocr-cache` when neither `$XDG_CACHE_HOME` nor `$HOME` is set. No new
   third-party dependency is taken for this (`dirs` is not in `[workspace.dependencies]` and v1 is
   Linux + macOS only)."* Three things in that quotation change: the
   enumeration of platforms gains Windows; the parenthetical's reason for taking no dependency
   loses its second clause while keeping its first; and the fallback-trigger clause — *"falling back
   to `./.panel-ocr-cache` when neither `$XDG_CACHE_HOME` nor `$HOME` is set"* — becomes
   platform-dependent, because on Windows the second variable consulted before that fallback is
   `%LOCALAPPDATA%` and `HOME` is deliberately not consulted at all (item 5's flagged paragraph). The
   `./.panel-ocr-cache` destination itself is unchanged on every platform; only the condition that
   reaches it differs. Everything else in item 21 — the
   `--cache-dir > config.cache_dir > platform default` precedence, the single
   `paths::resolve_cache_root` funnel, the hidden-flag rationale, the
   recovery-command-from-resolved-state rule, and the non-UTF-8 `Path::display()` limitation —
   stands unchanged and is not touched by this entry. Widening any of those would be a separate
   step needing its own argument.

   Upstream PanelCleaner was run as the tiebreak oracle (cookbook rule 3). Its measured behaviour
   places **both** its cache and its config under `%APPDATA%` on Windows. v1.1 does **not** follow
   that: cache goes to `%LOCALAPPDATA%`, config to `%APPDATA%`, which is the split Windows itself
   specifies (`%APPDATA%` roams with the user profile; a regenerable model/image cache must not).
   This is therefore a deliberate divergence and is registered as **§14 item 19**, with a
   `// DEVIATION(19): ...` comment required at the Windows branch of the resolver in
   `crates/pc-cli/src/paths.rs`. That comment exists as of 2026-08-04, at
   `crates/pc-cli/src/paths.rs:90` in `cache_dir`'s Windows arm — the register entry landed with
   this ratification and the site comment landed with the implementation, which is the sequencing
   `DEVIATION(16)` also went through: §14 item 16 records it, and the comment has since been added
   at `crates/pc-cli/src/detector.rs:39` (**line-number correction, 2026-08-06, unrelated to
   §16.37**: this read `:36`, which was **correct when written** on 2026-08-04 in commit `83f7f89`
   — the comment genuinely sat at line 36 then. It went stale in commit `1f2bd73` (§16.36 G1-C,
   "session-creation refusal at detector + OCR seams"), which inserted lines above it and moved it
   to `:39`; verified with `git log -S`, and by reading the file at `1f2bd73^` (36) and `1f2bd73`
   (39). §14 item 19 carried the identical citation and is corrected the same way).
   **Pre-existing defect, noted rather than fixed here:**
   §14 item 16's own sentence still says "the concurrent implementation pass has not added that
   comment yet", and §16.13's note near line 3154 says the same; both are stale as of 2026-08-04,
   verified by grepping for `DEVIATION(16)`. Those are two other entries' text and are not amended
   by this one — the citation above deliberately does not lean on that stale sentence.

   `dirs` stays out of `[workspace.dependencies]`. The first half of item 21's parenthetical
   ("no new third-party dependency is taken for this") is unchanged and is the half that carries
   the decision; the second half ("and v1 is Linux + macOS only") was a supporting reason, not the
   conclusion, and its loss does not disturb the conclusion. Distinguishing the two is the point
   of quoting the clause instead of restating it.

4. **E3 — settled: XDG variables win on every platform, Windows included.** If `XDG_CACHE_HOME`
   is set, the cache root is `$XDG_CACHE_HOME/panel-ocr` on Linux, on macOS **and** on Windows;
   likewise `XDG_CONFIG_HOME` for config. Neither Opus pass argued for restricting XDG to Linux,
   and both gave the same reason: the variables are the project's only mechanism for a user or a
   test harness to relocate the roots without a CLI flag, and making that mechanism
   platform-conditional would make the resolver's own tests platform-conditional too. The two
   variables are independent: `XDG_CACHE_HOME` set with `XDG_CONFIG_HOME` unset yields an
   XDG-derived cache root and a platform-derived config root.

5. **The interface Codex implements, pinned here so the tests in item 15 have something exact to
   compile against.** All of it lives in `crates/pc-cli/src/paths.rs` except `resolve_editor`
   (item 7), which lives in `crates/pc-cli/src/lib.rs`.

   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq)]
   pub enum Platform { Linux, MacOs, Windows }

   impl Platform {
       /// The platform this binary was compiled for.
       pub const HOST: Platform = /* cfg! cascade */;
   }

   /// An injectable environment lookup. The resolver takes one of these instead of
   /// reading `std::env` directly, which is what makes every platform's branch
   /// reachable from a test on any host.
   pub trait EnvSource {
       /// `None` for an unset variable AND for one set to the empty string — the
       /// empty-is-unset rule the current `env_path` helper already applies.
       fn var(&self, key: &str) -> Option<std::ffi::OsString>;
   }

   /// Reads the real process environment via `std::env::var_os`.
   pub struct ProcessEnv;

   /// A fixed table, for tests. Public because the resolver's tests are integration
   /// tests and cannot reach a private double.
   #[derive(Debug, Default, Clone)]
   pub struct MapEnv(/* private */);
   impl MapEnv { pub fn from_pairs(pairs: &[(&str, &str)]) -> Self; }

   /// A platform plus an environment, and nothing else. No process state, no `cfg!`.
   pub struct DirEnv<'a> { /* private */ }
   impl<'a> DirEnv<'a> {
       pub fn new(platform: Platform, env: &'a dyn EnvSource) -> Self;
       pub fn cache_dir(&self) -> PathBuf;
       pub fn config_dir(&self) -> PathBuf;
       pub fn config_path(&self) -> PathBuf; // config_dir().join(CONFIG_FILE_NAME)
   }
   ```

   The three existing free functions keep their exact names, signatures and public paths, and
   become one-line host-bound wrappers (`DirEnv::new(Platform::HOST, &ProcessEnv).cache_dir()` and
   so on): `paths::default_cache_dir`, `paths::default_config_dir`, `paths::default_config_path`.
   That is a hard requirement, not a convenience — `crates/pc-cli/tests/x1_args.rs`'s frozen
   `the_cache_directory_is_overridable` calls two of them, and `paths::resolve_cache_root`'s
   signature is fixed by §16.12 item 21's single-funnel rule.

   Cache root, in order, first match wins:

   | platform | order |
   |---|---|
   | Linux | `$XDG_CACHE_HOME/panel-ocr` → `$HOME/.cache/panel-ocr` → `./.panel-ocr-cache` |
   | macOS | `$XDG_CACHE_HOME/panel-ocr` → `$HOME/Library/Caches/panel-ocr` → `./.panel-ocr-cache` |
   | Windows | `$XDG_CACHE_HOME/panel-ocr` → `%LOCALAPPDATA%\panel-ocr` → `./.panel-ocr-cache` |

   Config root, in order, first match wins:

   | platform | order |
   |---|---|
   | Linux | `$XDG_CONFIG_HOME/panel-ocr` → `$HOME/.config/panel-ocr` → `./.panel-ocr` |
   | macOS | `$XDG_CONFIG_HOME/panel-ocr` → `$HOME/Library/Application Support/panel-ocr` → `./.panel-ocr` |
   | Windows | `$XDG_CONFIG_HOME/panel-ocr` → `%APPDATA%\panel-ocr` → `./.panel-ocr` |

   The Linux and macOS rows are today's behaviour, unchanged, transcribed so the table is complete
   rather than a diff. `panel-ocr` and `config.toml` remain the existing `APP_DIR_NAME` and
   `CONFIG_FILE_NAME` constants, and the final component of every non-fallback row is produced by
   `.join(APP_DIR_NAME)`, which item 13 depends on.

   **One interface detail settled by this transcription rather than by the ratification, and
   flagged as such: on Windows the resolver consults neither `HOME` nor `USERPROFILE`.** The
   ratification fixed the two Windows variables and said nothing about either, so both exclusions
   are this transcription's judgment call. They have separate reasons and an earlier draft gave only
   the first, which left `USERPROFILE` excluded with no argument at all:

   - **`HOME`**: MSYS2, Cygwin and Git-for-Windows all set it to a POSIX-shaped path that is not
     where a Windows application's data belongs. Consulting it would put the cache in a location
     that depends on which terminal launched the binary.
   - **`USERPROFILE`**: it is a Windows-native path (`C:\Users\<name>`), so the MSYS2 argument does
     not apply to it. The reason is different: the only way to use it here would be to rebuild
     `%USERPROFILE%\AppData\Local` by hand, which hard-codes a layout Windows treats as
     relocatable — `LocalAppData` and `RoamingAppData` are known folders that can be redirected by
     policy or by a roaming-profile setup, and `%LOCALAPPDATA%`/`%APPDATA%` are how the OS reports
     where they actually are. Reconstructing the path would therefore risk writing somewhere
     Windows itself does not use, while adding a code path no test on a Linux CI host can validate
     against a real redirected profile. This reasoning is from the documented known-folder model,
     not from a measurement on a Windows host — no Windows host was available to this pass.

   Falling through to `./.panel-ocr-cache` when
   `%LOCALAPPDATA%` is genuinely absent is the same last-resort the other two platforms already
   take when `$HOME` is absent, so this adds no new failure mode. **Reversal path, if the maintainer
   disagrees with either exclusion:** it is one extra fallback step in the Windows row of one table
   (cache, config, or both), plus the corresponding row in `crates/pc-cli/tests/x1_platform_paths.rs`
   — which, per cookbook rule 8, means the frozen test goes back to the two architects jointly
   rather than being edited in place. The two exclusions are independent and either can be reversed
   without the other.

6. **E2 — settled: PowerShell is the only Windows shell whose quoting is implemented, and
   `cmd.exe` is out of scope with the reason recorded.**

   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq)]
   pub enum Shell { Posix, PowerShell }
   impl Shell {
       pub const HOST: Shell = Shell::for_target(Platform::HOST);
       pub const fn for_target(platform: Platform) -> Shell;
       pub fn quote(self, path: &Path) -> String;
   }
   ```

   `for_target` maps Linux and macOS to `Posix` and Windows to `PowerShell`. `Posix` keeps
   today's exact rule from §16.12 item 21, quoted verbatim so it is visibly unchanged: *"wrap the
   path in single quotes and represent each embedded single quote as `'\''` (close quote, escaped
   literal quote, reopen quote)"*. `PowerShell` wraps in single quotes and doubles each embedded
   single quote (`''`); a PowerShell single-quoted string is fully literal, so no other character
   needs escaping. `crates/pc-cli/src/models.rs`'s `shell_quote` is replaced by
   `Shell::HOST.quote(...)` at its one call site.

   `cmd.exe` is out of scope because it has **no** quoting rule that round-trips an arbitrary
   path: `%VAR%` expansion happens before quote processing, so a path containing a literal `%`
   cannot be expressed at all, and `^`-escaping is context-dependent on whether the line goes
   through a pipe. A recovery command exists only to be pasted verbatim (§16.12 item 21's own
   justification), so emitting one we cannot guarantee round-trips would be worse than emitting
   the PowerShell form. The consequence, stated rather than discovered: a user who pastes the
   suggestion into `cmd.exe` may get a wrong path for a `%`-containing cache root. That is a
   documented limitation of the same class as item 21's non-UTF-8 `Path::display()` limitation.

   `Shell` lives in `crates/pc-cli/src/paths.rs`, alongside `Platform` — pinned explicitly because
   `crates/pc-cli/tests/x1_shell_quoting.rs` imports `pc_cli::paths::Shell` and is frozen, so
   putting the enum anywhere else satisfies the prose while breaking the test.

   Two limits of the gating, so nobody reads more into green than is there. First, "the enum has
   exactly two variants and no `Cmd`" is **not** asserted by any test in this pass, so `cmd.exe`
   staying out of scope rests on this text alone. Note what that does *not* claim: a mechanical gate
   is possible — a non-wildcard `match` over `Shell` in a test would make adding a `Cmd` variant a
   compile error (`E0004`, non-exhaustive patterns), which is a real and cheap gate. The accurate
   statement is that `x1_shell_quoting.rs` as drafted contains no such exhaustive match, so today
   nothing catches a third variant; an earlier draft of this item claimed no gate was *possible*,
   which was wrong. Second, the existing round-trip-through-a-real-shell oracle
   (`the_models_download_suggestion_is_paste_safe_for_hostile_cache_paths` in
   `crates/pc-cli/tests/x1_args.rs`) is `#[cfg(all(feature = "onnx", unix))]` — the full attribute,
   as item 2 states it — and stays so; the PowerShell round-trip oracle is a new test that runs only
   on a Windows host, and off Windows the PowerShell rule is checked against a hard-coded literal
   table instead. Item 15 names which is which.

7. **E5 — accepted as scoped: `notepad.exe` is the Windows fallback, and only on Windows.**

   ```rust
   /// `$EDITOR` when set, else the platform's fallback, else `None`.
   pub fn resolve_editor(platform: Platform, env: &dyn EnvSource) -> Option<std::ffi::OsString>;
   ```

   | platform | `$EDITOR` set | `$EDITOR` unset or empty |
   |---|---|---|
   | Linux | that value | `None` |
   | macOS | that value | `None` |
   | Windows | that value | `Some("notepad.exe")` |

   `$VISUAL` is deliberately **not** consulted: today's `profile edit` reads `EDITOR` only, and
   adding a second variable is a behaviour change on Linux and macOS that nothing asked for.
   `None` keeps today's exact rendered error, `"$EDITOR is not set"`.

   **Two Linux/macOS cases DO change, and saying otherwise was wrong.** An earlier draft of this
   item claimed the Linux and macOS surfaces were "byte-identical to before". They are not. Today's
   call site is `crates/pc-cli/src/lib.rs:196`,
   `std::env::var("EDITOR").context("$EDITOR is not set")?`, and reading it against item 5's
   `EnvSource::var` contract gives three cases, not one:

   | `EDITOR` | today | v1.1 on Linux/macOS | same? |
   |---|---|---|---|
   | genuinely unset | `Err(NotPresent)` → `"$EDITOR is not set"` | `None` → `"$EDITOR is not set"` | yes |
   | set to `""` | `Ok("")` → spawns `""` → `"failed to start $EDITOR"` | `None` → `"$EDITOR is not set"` | **no** |
   | set to non-UTF-8 bytes | `Err(NotUnicode)` → `"$EDITOR is not set"` | `Some(bytes)` → spawn attempted | **no** |

   Both changes are deliberate and both are improvements, which is why they are kept rather than
   worked around; but they are behaviour changes and are recorded as such, because
   `crates/pc-cli/tests/x1_editor.rs` freezes the empty-`EDITOR` case and a frozen test resting on a
   false "nothing changes" claim is exactly what cookbook rule 8 is for. The empty case: `EDITOR=""`
   is common in stripped environments, and `"$EDITOR is not set"` describes it far better than a
   failure to launch a program named the empty string. The non-UTF-8 case: today's message is
   actively wrong — the variable *is* set — and item 5's `OsString`-valued contract means the value
   no longer has to be discarded to be used, since `Command::new` takes an `OsStr`. Neither case is
   reachable from `x1_args.rs` or any other frozen test outside `x1_editor.rs`, checked by grepping
   the tracked tree for `EDITOR` on 2026-08-04.

8. **End-of-line normalisation is a Windows correctness hazard for digest-pinned text artifacts,
   and it is measured.** Git-for-Windows defaults to `core.autocrlf=true`, which rewrites LF to
   CRLF in the working tree for any file Git considers text. Seven committed **text** artifacts
   have their exact bytes pinned by a SHA-256 digest recorded in a `PROVENANCE.json`, verified by
   `verify_committed_artifact` in `crates/pc-testkit/tests/recorded_provenance.rs`. Enumerated by
   reading every `tests/fixtures/recorded/*/PROVENANCE.json` record whose `output` is not an image
   or model file, on 2026-08-04 — none of the seven carries an end-of-line attribute today
   (`git ls-files --eol` reports `attr/` empty for all seven):

   1. `tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_detector_blocks.json`
   2. `tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01#raw.json`
   3. `tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_upstream_oracle.json`
   4. `tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_upstream_group_output_equality.json`
   5. `tests/fixtures/recorded/model_signature/comictextdetector.signature.json`
   6. `tests/fixtures/recorded/ocr_model_signature/encoder_model.signature.json`
   7. `tests/fixtures/recorded/ocr_model_signature/decoder_model.signature.json`

   Every one of them must be `-text` in `.gitattributes`, the same treatment
   `crates/pc-ocr/assets/vocab.txt` already has under §16.30 item 2. Without it, a Windows
   checkout fails seven digest checks for a reason that has nothing to do with any of the code
   under test, which is the worst possible first impression of a new platform. The `-text` marker
   is kept **narrow** — scoped to digest-pinned artifacts, not applied as a blanket `* -text` —
   so that its presence on a file continues to mean "these bytes are pinned"; a blanket rule would
   make the marker uninformative and would silence the gate in item 15 by construction.
   `.gitattributes` glob syntax note, since one filename needs it: `#` is only a comment
   introducer at the start of a line, so `...E01P01#raw.json` needs no escaping mid-pattern.

9. **E4 — RATIFIED: on Windows, `xtask` must compile and its non-model tests must pass; nothing
   more.** `xtask` is a workspace member (§16.13 item 2), so `cargo test --workspace
   --all-targets` on a Windows runner already compiles it and runs its unit tests — 67 of them
   were discovered by `cargo test -p xtask --bins -- --list` on 2026-08-04. Its
   Python-subprocess-dependent paths — fixture recording (`xtask/src/record.rs`), golden
   calibration (`xtask/src/calibrate.rs`), the interpreter probe (`xtask/src/env.rs`) and the
   model-signature recorders — stay **maintainer-local, Linux and macOS only**, documented as
   such rather than silently untested. A scan of every `#[test]` body in `xtask/src/*.rs` on
   2026-08-04 found none that spawns a Python interpreter or any subprocess, so the 67 are
   expected to be platform-neutral already; that is an expectation about today's bodies, not a
   guarantee, and the CI Windows job is what will settle it. CI must not invoke `cargo xtask` on
   the Windows runner, and item 15's `ci_matrix.rs` asserts it does not.

10. **E6 — RATIFIED: the default feature tier ships on Windows now, and the `onnx` tier is
    allowed to be a documented partial.** The plan flagged W6: `ort` 2.0.0-rc.12's Windows
    linking behaviour is unverified by this project and may not resolve within reasonable
    iteration. The decision is to ship the default tier regardless. If W6 does not close, the
    result is a **ratified, explicitly documented partial** — default tier supported on Windows,
    `onnx` tier on Windows deferred or best-effort — and not a release blocker and not a silent
    gap. The binding mechanical consequence: the `onnx` CI tier must never make Windows a
    *required* check unless that is separately ratified. `ci_matrix.rs` asserts the default
    `test` job's operating-system set as a set (Windows present) and asserts the implication for
    the `test-onnx` job: if it lists a Windows runner at all, that job carries
    `continue-on-error: true`.

11. **The DirectML rejection is re-grounded, because the ground it stood on was that Windows was
    not a target.**

    **SUPERSEDES: §16.22 item 1**

    The scope of what is replaced is one paragraph of that item, quoted verbatim: *"DirectML is
    rejected on read evidence rather than packaging preference: `ort/src/ep/directml.rs` gates
    `supported_by_platform()` on `cfg!(target_os = "windows")`. TensorRT/NVRTX need
    `libnvinfer.so.10`, which no `ort` distribution ships."* Item 1's **decision** — CUDA at
    v1.5 opt-in; CoreML, DirectML, `.pt`/torch loading and multi-device dispatch at v2 — is
    untouched, and so is the TensorRT/NVRTX sentence, whose reason (no distribution ships the
    library) is platform-independent. What changes is only the DirectML *justification*: an
    argument that a provider is unavailable because it requires Windows is vacuous only while
    Windows is not a target, and §16.23 item 4's mirror rule forbids carrying a
    vacuous-for-this-version argument silently into a version where it is no longer vacuous.

    The replacement ground, which reaches the same verdict from evidence that does not depend on
    the platform list: DirectML is a **non-CPU execution provider**, so §16.22 item 2's carve-out
    is what applies to it. §16.22 item 2 holds §5 item 7's determinism guarantee
    ("identical inputs + identical config must produce identical outputs … regardless of thread
    count") unconditionally for the CPU execution provider only, and requires any other provider
    to be quarantined from CI, fixtures, recordings and every gate. DirectML would therefore need
    its own quarantine, its own opt-in surface and its own measurement of run-to-run and
    cross-provider divergence — the three things §16.22 items 3, 4 and 5 spell out for CUDA and
    none of which exist for DirectML — and there is **no Windows GPU CI runner** on which any of
    it could be verified. v1.1 ships the CPU execution provider on a third platform. It adds no
    execution provider. DirectML stays at v2 on that basis.

12. **The global out-of-scope bullet moves.**

    **SUPERSEDES: §16**

    The bullet's full text, quoted verbatim: *"**Windows support** — v2 (Linux + macOS only)."*
    It is replaced by a v1.1 statement in the same list. Nothing else in §16's out-of-scope list
    is touched by this entry — in particular the GPU-execution-provider bullet keeps CoreML and
    DirectML at v2, which item 11 re-grounds rather than moves.

    **The gate's granularity here is section-wide, not line-level.** The supersession cross-check
    row for this claim is `("16.33", "16")`, and `§16` is the whole summary section, so a back-pointer
    anywhere within its span satisfies it — not specifically the one at the out-of-scope bullet. That
    is ratified §16.26 item 3(c) leniency ("a bare `§N` target with no item scopes to the whole
    section"), documented at the constant in `crates/pc-testkit/tests/spec_supersession.rs`
    rather than a defect, but it means the gate would stay green if the marker migrated off the
    bullet a reader actually lands on, which is the failure mode the cross-check exists to prevent.
    Worth knowing before treating this row as strong evidence; the other two rows this entry adds
    name numbered `§16.x` subsections and are correspondingly tighter.

13. **Cross-check of the frozen assertion, stated explicitly instead of left implicit.**
    `crates/pc-cli/tests/x1_args.rs`'s frozen `the_cache_directory_is_overridable` asserts
    `paths::default_cache_dir().ends_with("panel-ocr")` and
    `paths::default_config_path().ends_with("config.toml")`. Both survive E1, and the reason is
    structural rather than lucky: `Path::ends_with` compares whole trailing components, every
    non-fallback row of item 5's two tables ends in `.join(APP_DIR_NAME)`, and `config_path` is
    `config_dir().join(CONFIG_FILE_NAME)`. So `%LOCALAPPDATA%\panel-ocr` satisfies it on a
    Windows host, where `\` is a component separator, exactly as `$HOME/.cache/panel-ocr` does on
    Linux.

    The one case that does **not** satisfy it is the last-resort relative fallback
    (`./.panel-ocr-cache`, whose final component is `.panel-ocr-cache`) — and that is a
    **pre-existing** conditionality of the frozen assertion, identical on all three platforms and
    not introduced by E1: on Linux today, an environment with neither `XDG_CACHE_HOME` nor `HOME`
    already fails it. E1 adds a third platform to that shape without changing it. Recorded here
    because a DEVIATION that quietly changed the meaning of a frozen assertion would be the exact
    defect cookbook rule 8 exists to catch, and "we checked and it still holds" is only useful if
    it says *why*.

14. **Sequencing: contiguous, and before GPU-1.** Both Opus passes independently placed the
    Windows work as one contiguous block ahead of the v1.5 GPU task. The reason is that the GPU
    task's own quarantine design has to know how many platforms it is quarantining from, and item
    11's re-grounding is a prerequisite for it rather than a side note: a GPU plan drafted while
    §16.22 item 1 still justified the DirectML rejection by "it needs Windows" would inherit an
    argument this entry retires. Splitting the Windows block would also mean a CI matrix that
    lists a Windows runner before the resolver can pass on it, which is a red required check for
    however long the split lasts.

15. **The tests, written first, and what each one traces to.** Seven new files; each names the
    requirement in this entry it verifies. Their measured status against today's source is in this
    entry's status preamble and is **not** uniform: three do not compile, three compile with one or
    two failures each, and `verbatim_paths.rs` is fully green because this entry wrote both
    documents it gates. Do not read this table as an all-red list.

    | file | verifies |
    |---|---|
    | `crates/pc-testkit/tests/platform_claim_sites.rs` | item 2 — every enumerated claim site carries a `§16.33` pointer, with a synthetic wrapped-claim control for the measured line-based blindness |
    | `crates/pc-cli/tests/x1_platform_paths.rs` | items 3, 4, 5 — the two resolution tables, per platform and per environment state, including that Windows cache is `%LOCALAPPDATA%` and Windows config is `%APPDATA%` when the two differ |
    | `crates/pc-testkit/tests/verbatim_paths.rs` | items 3, 5, 9 — the ratified path literals and the E4 scope sentence appear verbatim in this entry and in `docs/ARCHITECTURE_DECISIONS.md`, so code and documentation cannot drift apart the way they did in item 2 |
    | `crates/pc-cli/tests/x1_shell_quoting.rs` | item 6 — `Shell::for_target`'s mapping and both quoting rules; PowerShell against a hard-coded literal table everywhere, and against a real `pwsh` round trip on a Windows host |
    | `crates/pc-cli/tests/x1_editor.rs` | item 7 — all six cells of the editor table |
    | `crates/pc-testkit/tests/eol_normalisation.rs` | item 8 — each of the seven digest-pinned text artifacts is `-text` per `git check-attr`, and the marker stays narrow |
    | `crates/pc-testkit/tests/ci_matrix.rs` | items 9, 10 — the default `test` job's operating-system set, the `onnx` tier implication, and that the Windows job does not invoke `cargo xtask` |

    Task classification, per CLAUDE.md's plan requirement: the resolver plus the quoting and
    editor changes are **simple** and may be batched into one Codex call, because they share one
    file and one injectable-environment shape. The `.gitattributes` and `ci.yml` changes are
    **simple** and batchable with each other. `verbatim_paths.rs`'s documentation targets are
    **simple**. Nothing here is heavy: there is no model, no fixture re-recording and no
    numerical calibration in this task.

## 16.34 `panel-ocr ocr` misclassifies OCR'd pages as `NoTextDetected` (joint architect + Senior Rust Engineer plan pass, 2026-08-04; user-reported against the packaged Windows `onnx` binary; **step-1a fresh-reader pass found five citation/attribution defects in the first transcription, corrected below — 2026-08-05**)

**Status.** This entry transcribes a ratified bug-fix PLAN, not yet-implemented work. Both
subagents independently reproduced the defect (the architect's `run_stages` probe and the
engineer's `cargo test -p pc-pipeline --test x1_ocr_report_path`, both against today's `HEAD`)
before proposing the same predicate. The first transcription of this plan overclaimed test
coverage that does not exist and mis-cited three spec clauses; a fresh `fresh-reader` spawn
caught all of it before commit (step 1a's purpose exactly), and this version corrects each
finding rather than silently replacing the prior text — see the parenthetical items below.

1. **The defect, in one sentence: `panel-ocr ocr` reports "no text detected" and an empty
   CSV/TXT for every image that has any, because §15's report-path overrides intentionally empty
   `PageData::text_boxes` and `pc-pipeline` reads that emptiness as absence.** Reproduced against
   the real Windows `onnx` binary on a real manga page: `clean` (with OCR enabled) detects and
   OCRs 8 boxes; `ocr` on the identical image with the identical profile reports
   `0 completed, 1 skipped, 0 failed … no text detected` and writes an empty report. §15 item 5
   and §16.11 item 11 already establish *why* `text_boxes` ends up empty there (every box is
   OCR'd and moved into `OcrAnalytic.removed`, by design); what was never checked is that
   `crates/pc-pipeline/src/single.rs`'s `no_text = page.text_boxes.is_empty()` (the one
   `§5 item 6` trigger) cannot tell that intentional emptiness apart from the genuine one.

2. **Fix: the skip decision is declared per-path in one named function, not inferred from
   `text_boxes` alone.**

   ```rust
   /// §5 item 6's skip decision, declared per path rather than inferred from one field: the
   /// `clean` and `ocr` paths mean different things by an empty `text_boxes` (§16.34).
   ///
   ///   * `clean` (`performing_ocr == false`) — unchanged, bit for bit: `text_boxes.is_empty()`
   ///     is exactly §5 item 6's trigger.
   ///   * `ocr` with an OCR analytic present — §15's overrides move every OCR'd box out of
   ///     `text_boxes` into `OcrAnalytic.removed` (§16.11 item 11), so `text_boxes` is empty *by
   ///     design*. The population §5.6 counts is `OcrAnalytic.num_boxes`, already defined by
   ///     §16.8 item 11 as "the number of tight boxes at entry to step 7 (pre-removal)" — no new
   ///     field is added anywhere.
   ///   * `ocr` with no analytic (no factory, or the pass never ran) — the pass never ran, so
   ///     nothing was consumed and `text_boxes` is still the whole population; falls back to the
   ///     `clean` reading.
   fn no_text_for(performing_ocr: bool, text_boxes_empty: bool, ocr: Option<&OcrAnalytic>) -> bool {
       match (performing_ocr, ocr) {
           (true, Some(ocr)) => ocr.num_boxes == 0,
           _ => text_boxes_empty,
       }
   }
   ```

   Rejected candidate, named because it was on the table: gating on `ocr.removed.is_empty()`
   instead of `ocr.num_boxes == 0`. Wrong — a page whose every box hit `StageError` in
   `recognize()` (DEVIATION(9), fail-open) has `num_boxes > 0` and `removed == []`; that reading
   would report an OCR **failure** as an **absence**, the same category error this entry fixes.

   **What does not change, stated because each is a live constraint this fix must not cross:**
   `outcome_for`, `ChainOutputs.no_text`'s call sites, `process_image`, and §16.14 item 2's
   strip conjunction (`no_text` stays the conjunction over segments) are untouched — only the
   value fed into the existing `no_text` slot changes, for the `performing_ocr` path only. The
   OCR discard pass itself (`crates/pc-preprocess/src/ocr_filter.rs`) is untouched: §16.8 item 12
   already closes the alternative fix ("make the pass keep boxes when `performing_ocr`") by
   pinning that flag to step 2 only. `ImageOutcome::Skipped`'s shape (§16.12 item 12) is
   untouched — it still carries no analytics field, which is exactly why the fix has to sit at
   the `Completed`/`Skipped` classification and not downstream of it: once a page is misclassified
   `Skipped`, its OCR data has nowhere to go. `ChainOutputs.no_text`'s doc comment at
   `single.rs:116` ("`true` when `PageData::text_boxes` was empty") becomes false the moment this
   lands and must be corrected in the same diff to name `no_text_for`'s three-way reading instead.

3. **SUPERSEDES: §15 item 5** — upstream's `run_ocr` sets **three** overrides
   (`pcleaner/main.py:862-868`, pinned commit `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`, quoted
   verbatim): `profile.preprocessor.ocr_enabled = True`, `profile.preprocessor.ocr_max_size =
   10**10`, and `profile.preprocessor.ocr_blacklist_pattern = ".*"`. The Rust port's
   `apply_report_overrides` (`crates/pc-cli/src/ocr.rs`) sets only the last two. Item 5's
   clincher sentence quotes only those two and is narrowed accordingly; its actual conclusion
   (the CSV/TXT fixtures define report format, not filter behavior, because the blacklist is
   inert on that path) is untouched by this addition — the third override does not change the
   blacklist's behavior. §9.6's fixtures bullet also states the two-override fact without
   enumerating a third and remains accurate for the same reason (its claim is about the
   blacklist's inertness, not about the exhaustive override list); not marked. §16.11 item 11's
   two-override mention is likewise unaffected, but its further claim that "every box is
   'removed' and `removed` is the complete, ordered box list" holds only where every
   `recognize()` call on a candidate box succeeds — item 5 below documents a `Completed` page
   where a box survives OCR failure and is never removed at all, so `removed` there is *not* the
   complete box list. That is a scope gap in item 11's "every", not a wrong override count, and
   is not a reversal of item 11's actual point (reading only `removed` on the happy path is still
   not a gap); recorded here rather than left as a silent "both remain accurate."

   Without the third override, a profile with `ocr_enabled = false` makes `panel-ocr ocr` a
   *second*, independent silent-empty-report path: `apply_report_overrides` never forces
   `ocr_enabled`, so the step-7 OCR pass simply never runs — §16.8 item 12 pins that
   `performing_ocr` alone gates only the strict-language drop in step 2 and does **not** by
   itself enable the step-7 pass, which needs a factory **and** `ocr_enabled` (§16.8 item 3) —
   `analytics.ocr` stays `None`, and `no_text_for`'s fallback arm (item 2 above) reads the
   still-full `text_boxes` as "has text": `Completed` with an empty analytics slice, rendering
   `""` at exit 0. §16.8 item 3's own stated rationale for requiring a factory **and**
   `ocr_enabled` gate together is exactly this hazard ("an all-empty analytic is indistinguishable
   from 'ran and found no candidates'"). Fix: `apply_report_overrides` also sets
   `profile.preprocessor.ocr_enabled = true`.

4. **SUPERSEDES: §9.5** — the P8 row there enumerates the two overrides in parenthetical form
   ("`run_ocr`'s §15.5 report-path overrides (`ocr_blacklist_pattern = ".*"`,
   `ocr_max_size = 10^10`)"), which item 3 narrows to name the third. Bare-section target per
   §16.26 item 3(c)'s convention (a markdown-table row carries no `^N. ` item marker), so a
   `§16.34` back-pointer anywhere in §9.5 satisfies the gate. **§13's row 31 is deliberately NOT
   marked**, corrected from the first transcription of this entry, which claimed both rows
   "name only two overrides": row 31's actual text is a bare pointer — "`run_ocr`'s §15.5
   report-path overrides" — with no enumeration to narrow, so nothing there is stale. §13's own
   claim that `run_ocr` wires the OCR factory *unconditionally* is about factory construction, not
   about which profile fields get overridden, and stays true after this fix. Both rows' actual
   task scope (P8, already shipped) is unaffected either way.

5. **Open question, decided rather than left implicit — and the reachability claim in the first
   transcription of this item was WRONG, corrected here rather than silently fixed: a
   `Completed` page whose OCR pass ran but whose `removed` list is empty (every candidate box
   failed OCR, had no engine for its language, or missed the canvas) renders a phantom header** —
   `write_txt` emits `"path: \n"` with no lines, `write_csv` emits a bare header row, for a page
   that produced no usable text. **This is reachable at today's `HEAD`, before any part of this
   fix lands** — a page on which the OCR pass keeps every box via one of those three fail-open
   paths already has a non-empty `text_boxes` today, so `no_text` is already `false` and the page
   is already `Completed` carrying an analytic with `removed == []`; the fresh-reader pass
   confirmed this with a standalone probe against `HEAD` (`MockOcrEngine::failing()`, 2 detected
   boxes, `performing_ocr = true`): `num_boxes=2, removed=[]`, `ImageOutcome::Completed`, TXT
   `"page01.png: \n"`, CSV header-only. This fix does not create the case or change whether it is
   `Completed` — item 2's predicate still reads `num_boxes == 0` as false here, so the page stays
   `Completed` exactly as it is today. **Decided: `pc-cli::ocr_report` skips an `OcrAnalytic`
   whose `removed` is empty**, matching upstream's own per-box (not per-page) report shape
   (`pcleaner/ocr/ocr.py`'s `format_output_plain` writes the path header only inside the per-box
   loop, so a page contributing zero boxes contributes no header). One line in
   `crates/pc-cli/src/lib.rs::ocr_report`; `pc_export::render_ocr_report` and both frozen writers
   are untouched. This is a judgment call grounded in a *reading* of upstream, not a run of it
   (provoking an all-boxes-fail page there needs a broken engine) — recorded as such rather than
   silently promoted to a verified fact. Residual, recorded rather than fixed here: after this
   change a page whose every box failed OCR still renders `""` while the batch summary counts it
   `completed`; the per-box `WARN` `run_ocr_pass` already logs on engine failure is the only
   signal, which is acceptable but is the mirror of the category error item 2 rejects and is
   worth a reader noticing.

6. **Tests. The first transcription of this item claimed two tests exist that do not — corrected
   here to state only what is actually on disk, and what the implementation task must still add.**
   Two files exist today, untracked pending this ratification, each demonstrated red against
   `HEAD` (not green — no fix is implemented yet):
   - `crates/pc-pipeline/tests/x1_ocr_report_path.rs` — four integration tests: an OCR run over 2
     detected boxes completes with both boxes in `analytics.ocr.removed` (identity-checked
     against a hand-derived vector, not just a count); a page with 0 detected boxes is still
     `Skipped { NoTextDetected }` even under `performing_ocr`; a `clean`-path page whose OCR
     filter discarded every box is still `Skipped` and still exported (the control that stops
     this fix from widening past the report path — pins `g1_chain.rs`'s frozen
     `a_page_without_text_is_skipped_but_still_exported` sibling behavior, grounded in
     §5 item 6/§16.12 items 5 and 12); and the literal
     batch-summary line `"1 completed, 0 skipped, 0 failed"` for the reproduced symptom. Measured
     at `HEAD`: 2 passed (the two controls), 2 failed (the two bug rows), exactly as expected
     pre-fix.
   - `crates/pc-cli/tests/x1_ocr_report_rows.rs` — three integration tests driving the real
     `apply_report_overrides` + `run_batch` + `ocr_report` composition (cookbook rule 12: the
     defect lived in the join, not in any one piece): the CSV report carries one row per OCR'd
     box for a page with text, against a hand-derived expected string; the TXT report carries
     both recognised strings; a page with zero detected boxes renders `""`, not a bare header,
     preserving §16.11 item 12's existing rule that "an empty `analytics` slice renders `\"\"`
     for both formats (not a bare CSV header)" — **corrected citation: the first transcription
     named item 11, which is the `removed`-is-the-complete-list clause, not this rule.** This file
     does not compile at `HEAD` (`ocr_report` and `apply_report_overrides` are private), which is
     itself expected pre-fix.
   - **Not yet drafted, and must be added by the engineer before Codex implements — the first
     transcription incorrectly claimed these already existed:** (a) a truth-table unit test over
     `no_text_for` itself, covering all six `(performing_ocr, text_boxes_empty, ocr)` rows named
     in item 2's doc comment, including the two rows no integration test reaches cheaply
     (`performing_ocr = true` with no analytic; a `clean`-path page where every box was OCR-filtered
     away, confirming `no_text_for` still returns the `clean` reading and does not accidentally
     key off `ocr` being `Some`); (b) a test proving `apply_report_overrides` forces
     `profile.preprocessor.ocr_enabled` back to `true` starting from a profile that set it
     `false` — built from `false` rather than `Profile::default()`'s already-`true` value, so it
     cannot pass vacuously (cookbook rule 1); (c) a test for item 5's decided behaviour: an
     `OcrAnalytic` with `num_boxes > 0` and `removed == []` (an all-boxes-fail-open page) renders
     `""`, not a bare header — the one case in this entire fix that changes previously-observable
     output.
   - `report_profile_overrides_disable_report_filtering` (`crates/pc-cli/src/ocr.rs`) is
     **frozen and unedited** — it is a real instance of cookbook rule 1 (its name claims more
     than "two struct fields hold the values I just assigned"), but the exit is addition, not
     amendment (cookbook rule 8 exit 1): the tests above and the two still to be drafted gate the
     seam it never reached.
   - Not required as a CI gate: an `onnx`-tier end-to-end `panel-ocr ocr` smoke test needs real
     manga-ocr weights and cannot run without them; the tests above use `MockDetector` +
     `pc-ocr`'s test-kit OCR double and gate the defect without a model.

7. **Task classification: simple, one Codex call, sequential within it.** Scope: `no_text_for` +
   its call site + the corrected `ChainOutputs.no_text` doc comment (`crates/pc-pipeline`); then
   the third override + the `ocr_report` empty-`removed` skip + `pub` visibility on `ocr_report`
   and `apply_report_overrides` needed for the new integration tests + a `pc-ocr` test-kit
   dev-dep in `crates/pc-cli/Cargo.toml` (`crates/pc-cli`, `crates/pc-preprocess` untouched). Not
   split into two calls: `x1_ocr_report_rows.rs` cannot compile without the `pc-cli` half and
   cannot go green without the `pc-pipeline` half. Must not be batched with any other task
   touching `crates/pc-pipeline/src/single.rs` or `crates/pc-cli/src/lib.rs` (cookbook rule 11's
   collision pattern). The three tests named in item 6 as not-yet-drafted must be written and
   confirmed red before Codex touches source, per CLAUDE.md's TDD loop.

## 16.35 Mask-quality polish: a lowest-deviation rescue tier for the masking fail-safe (joint architect + Rust Engineer plan, Fable tie-break on tie-direction and default, 2026-08-05)

Not a bug fix. Investigation this session found `pc-mask`'s masking fail-safe (§10.3 step 10) working exactly as specified: it deliberately leaves a box unmasked rather than paint a bad mask, matching upstream. This section ratifies a v1.1 improvement — recovering some of those left-unmasked boxes without weakening the fail-safe — found independently by both plan agents by reading the actual selection code, not by loosening the threshold.

1. **The mechanism, quoted from the code it modifies.** §10.3 step 9's greedy selection is a *ratchet*, not an argmin: candidate `i` is accepted iff `i == 0` or `dev_i <= best_dev * (1.0 - mask_improvement_threshold)`. Because a later candidate must *improve* on the current best by the full margin to win, the ratchet can stop on a candidate whose deviation is worse than one it already scored and passed over — e.g. deviations `[16.0, 14.5]` at the default `mask_improvement_threshold = 0.1`: index 1 needs `<= 14.4` and is rejected, so the ratchet's final pick is `16.0`. If `mask_max_standard_deviation = 15.0`, that pick fails the fail-safe (§10.3 step 10) and the box is left unmasked — even though the discarded candidate at `14.5` would have passed it. **DECIDED:** when this happens, retry with the lowest-`std_deviation` candidate among those already scored in §10.3 step 9, and paint it only if that candidate itself is `<= mask_max_standard_deviation`; if none qualifies, behaviour is unchanged (`mask: None`).

   **SUPERSEDES: §10.3 step 10** — the failure-threshold clause there stays exactly correct for `mask_fallback_to_lowest_deviation = false`; this item adds the retry that runs first when the flag is `true` (the shipped default, item 4 below).

2. **Why this cannot weaken the fail-safe, stated as checkable properties.** (P1) Every painted mask still satisfies `std_deviation <= mask_max_standard_deviation` — the identical predicate §10.3 step 10 already applies to the greedy pick, applied here to a different candidate, never relaxed. (P2) Every region masked today is unaffected: the rescue only fires on the branch that currently returns `None`. (P3) The masked-region set per page is therefore a **superset** of today's. (P4) A rescued candidate never leaves visible text that the greedy pick would have covered, because every growth candidate is a dilation of the same precise cut (larger index ⊇ smaller) and the box candidate ⊇ the masking rect — so a thinner rescue pick paints less margin, never less text. These four properties, not an appeal to "it's a small change," are what makes the default-on decision in item 4 safe.

3. **Raising `mask_max_standard_deviation` itself is rejected, not merely deferred.** Verified against the pinned upstream oracle (`0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`, `pcleaner/config.py:556-564`): every one of our nine masker defaults, including `mask_max_standard_deviation = 15`, is value-identical to upstream's — a claim about the values, not a byte-for-byte claim: §16.5 item 4 already names this exact key as the reason a float literal cannot be asserted byte-identical across the two languages' formatters. Raising it would be a new, unargued deviation, and — measured on the session's repro page — would paint visibly bad masks over four boxes scoring 38–57 in border deviation, which is exactly the outcome the fail-safe exists to prevent. The rescue tier in item 1 is the alternative that needed no such trade.

4. **RULING (Fable, 2026-08-05) — default `true`, with a mandatory `DEVIATION(21)` register entry.** The two plan agents disagreed on the default value and how much justification it needed; Fable's ruling: `mask_fallback_to_lowest_deviation` defaults to `true`, because a default of `false` would ship the polish switched off and make this whole task a no-op for real users, and P1–P4 above are exactly what makes `true` acceptable — nothing is ever painted above the existing threshold. **Binding condition:** because upstream has no such retry and our default therefore paints some boxes upstream leaves unmasked, this is registered as `DEVIATION(21)` at §14 item 21, and `mask_fallback_to_lowest_deviation = false` must be tested to reproduce upstream's exact selection output on a rescue-eligible fixture.

5. **RULING (Fable, 2026-08-05) — tie-break: lowest index wins.** When two or more already-scored candidates share the exact-equal lowest `std_deviation`, the rescue picks the **lowest index**. Neither agent's original ground survives scrutiny: §5.7 (line 464) only requires *some* fixed rule, not a direction, and a scratchpad probe against the real `select_candidate` showed the engineer's "consistency with the existing inclusive `<=`" claim is empirically backwards for the reachable case — at the default `mask_improvement_threshold = 0.1`, equal nonzero deviations already resolve to the *earlier* candidate today, and "later wins" is a corner reachable only at an exact `0.0` deviation or an explicit `mask_improvement_threshold = 0.0`, both of which the rescue tier can never reach (a scored `0.0` is always accepted by the greedy loop and always passes the fail-safe, since `mask_max_standard_deviation > 0` is a validated invariant, so the rescue never runs on a region whose candidate list contains one). Lowest-index also keeps the algorithm's own stated intent — a tie is not an *improvement*, so the smaller/earlier candidate keeps its claim — and paints strictly fewer pixels for identical measured border quality. Implementation note: `Iterator::min_by` returns the first minimum (verified by probe), so a straightforward `min_by` over `(index, deviation)` in candidate order already implements this ruling; a frozen test with two equal passing deviations (e.g. `[16.0, 14.5, 14.5]`, threshold `15.0`) must assert the lower index wins.

6. **RULING (Fable, 2026-08-05) — landing is not blocked on running upstream against the session's repro page, but *crediting* that repro as fixed by this feature is.** The repro image (`rockéQè¬046.png`) is not committed to this repo, so nobody has run upstream PanelCleaner on it. The mechanism argument in items 1–2 holds independent of any one page, so it does not gate landing. But before any commit message, changelog, or spec entry cites that specific repro as *resolved* by this feature, upstream must be run on it: if upstream successfully masks a box that we mask only via the rescue tier, that is presumptive evidence of a scoring-parity defect elsewhere (border extraction, luma, or candidate generation) being papered over, not a genuine fix, and it must be investigated as a bug first.

7. **API surface** (`crates/pc-mask/src/fit.rs`, additive only — `select_candidate`'s and `fit_region`'s existing signatures are unchanged and every existing frozen test in `crates/pc-mask/tests/m4_fit.rs` and `m56_run.rs` keeps compiling untouched):
   - `Scored { index: usize, std_deviation: f64, median_color: [u8; 3] }` — one scored candidate.
   - `Selection { greedy: Scored, lowest: Scored, deviations: Vec<f64> }` — everything §10.3 step 9's loop learned, returned by `select_candidate_with_fallback`, which `select_candidate` becomes a thin wrapper over (`.map(|s| s.greedy.into())`).
   - `fit_accepted(std_deviation, mask_max_standard_deviation) -> bool` — the fail-safe predicate, named once so it has exactly one call site.
   - `resolve_fallback(selection: &Selection, mask_max_standard_deviation: f64, enabled: bool) -> Scored` — applies item 1's policy; `enabled = false` reproduces today's `select_candidate` output bit for bit (the `DEVIATION(21)` control in item 4).
   - `FitReport { fitment: Fitment, candidate_deviations: Vec<f64>, greedy_index: usize, rescued: bool }` and `fit_region_scored(...) -> Option<FitReport>`, which `fit_region` becomes a thin wrapper over. **`Fitment` itself gains no fields** — `crates/pc-mask/tests/m56_run.rs:23` and `crates/pc-core/tests/page_data.rs` construct `Fitment`/`MaskFittingAnalytic` with exhaustive struct literals in frozen tests, so a new field there would break a frozen test at compile time for no correctness gain; the diagnostic travels beside `Fitment` in `FitReport` instead. When the rescue fires, `Fitment.std_deviation`/`candidate_index`/`thickness` report the **rescued** candidate's values, keeping `Fitment::failed() == (std_deviation > mask_max_standard_deviation)` true on both branches.
   - Diagnostics are **log lines only** (`WARN` on a failed-even-with-rescue region naming both the greedy and lowest deviations; `DEBUG` with the full deviation vector on success) — deliberately **not** added to `MaskRegionStats`, `MaskFittingAnalytic`, or the persisted `#mask_data.json` schema, per §16.25's "one output channel, one kind of claim": the schema-versioned pipeline-data channel is for data later stages consume, and nothing downstream needs the per-candidate vector.
   - One real cross-crate consequence, stated rather than left to be discovered: a rescued region's `MaskRegionStats.failed` flips `true → false` and its `std_deviation` drops, so `pc-denoise`'s noise-mask filter (§11.3 step 3: `!failed && std_deviation > noise_min_standard_deviation`) newly considers it for edge blending. This is the intended behaviour — a painted mask should be eligible for the same seam treatment as any other success — and is recorded here because it is a real behaviour change in a different crate, not a silent side effect.

8. **Task sequencing.** T1 (**heavy**): the `fit.rs` refactor above with zero behaviour change (rescue not wired) — the frozen `m4_fit.rs` suite must stay green with the identical trace, which is the regression gate for the refactor itself. T2 (**simple**, its own call, not batched with T1): the diagnostics logging. T3 (**heavy**): a `cargo xtask mask-sweep` tool (`--replay`, CI-runnable off the committed detector fixture; `--pages DIR --detector onnx:...`, maintainer-local) that measures, for real pages, how often the rescue fires and against what deviation values — non-gating, written to `docs/MASK_QUALITY_CALIBRATION.md` (reviewer/date/method, mirroring `GOLDEN_CALIBRATION.md`'s role) — this is what item 6's upstream-comparison check and any future retune argument must be grounded in, not a single hand-measured page. T4 (**simple**): the config key (`[masker] mask_fallback_to_lowest_deviation`, §6) plus its `Default`, plus `crates/pc-config/tests/`. T5 (**heavy**): wire the rescue into `fit_region` behind the flag. Do not batch T4 with any task touching `[general] device` (§16.36) in the same Codex call — see item 9.

9. **Conflict avoidance against the parallel v1.5 GPU-1 branch, explicit.** `crates/pc-config/src/profile.rs`: this branch's one field lives in `MaskerConfig` (§6's `[masker]` block); GPU-1's device field lives in `GeneralConfig` (§6's `[general]` block, per §16.36 item 1) — different structs, different `Default` blocks, low textual conflict risk, but land the two config-surface edits **sequentially, not via parallel Codex calls**, because `crates/pc-config/tests/defaults.rs::table_registry_matches_default_document` compares the whole registry against the whole shipped document and a half-merged state fails in a way that reads like a defect in whichever branch lands second. Spec section numbers are pre-assigned to avoid collision: this is **§16.35**, GPU-1 is **§16.36**. `DEVIATION` numbers are likewise pre-assigned: this is `DEVIATION(21)`, GPU-1's opt-in-only divergence is `DEVIATION(22)` — neither takes `20`, left unused per §14's note above.

10. **Out of scope, recorded so it isn't silently dropped.** No change to `mask_growth_steps`/`min_mask_thickness`/`mask_growth_step_pixels` (a denser ladder is not safety-preserving by construction — the candidates are generated by iterated in-place dilation of one shared buffer, so a different `(min, step)` pair produces different masks, not just different labels for the same ones; any retune needs T3's measured data, not a plausibility argument). No LaMa inpainting fallback for boxes the rescue still can't save (that is v1.5 scope, §16.23). No touch to `pc-core`, `pc-detect`, `pc-denoise`, `pc-export`, `pc-preprocess`, `pc-cli`, or `xtask/src/calibrate.rs`.

## 16.36 GPU-1: device config, the model-agnostic policy resolver, and §16.19-integrated fatal refusal (joint architect + Rust Engineer plan, Fable tie-break on three points, 2026-08-05)

Scope, quoted from §16.23 item 5 so it travels with this entry: **"device config, the model-agnostic policy resolver, §16.19-integrated fatal refusal, recording refusal — no CUDA linkage."** GPU-2 (the `cuda` feature and its guards) is a separate, later task. GPU-1 ships **no `ort` execution-provider registration code and no CUDA-conditional compilation** — every claim below holds in a build with no `cuda` feature at all.

1. **RULING (Fable, 2026-08-05) — the device key lives at `[general] device = "cpu"`, a single global choice, not a per-stage key.** Both plan agents' upstream appeals were reconciled by re-verification at the pinned commit `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`: upstream has **no device config key anywhere** in any of its six config section classes — it auto-detects globally (`pcleaner/ctd_interface.py:64`, `pcleaner/main.py:440`/`:815`), which §16.22 item 5(a) already forbids us from doing. What survives from upstream is only that device is one process-wide decision, never per-model. Decisive ground on our side: §16.22 item 5(f), *"the device policy resolver is model-agnostic, so v1.5's LaMa inpainting reuses it instead of growing a second one"* — v1 has no `[ocr]` or `[inpainter]` section, so a `[text_detector] device` key would force LaMa and OCR (item 3 below) to read a foreign section's key or grow a sibling one, which is the second policy surface 5(f) forbids. `[general]` already hosts the sibling machine-resource knob `max_threads`. §6's TOML block and §8.3 step 3's execution-provider bullet already reflect this ruling (this commit).

   **SUPERSEDES: §8.3 step 3** — the execution-provider bullet's prior text left `device`'s section unstated; this item pins it to `[general]`.

2. **RULING (Fable, 2026-08-05) — the resolver lives in `pc_core::device`, pure, with no `ort` types and no EP registration.** Verified dependency edges (every `Cargo.toml` read directly, not inferred): `pc-ocr` depends on `pc-core` **only** — it does **not** depend on `pc-config`. Every consumer of the resolver (`pc-detect`, `pc-ocr`, `pc-config`, `xtask`) already depends on `pc-core`, so this placement adds **zero** new dependency edges. Placing it in `pc-config` would force a new `pc-ocr → pc-config` edge that §16.5 item 1's enumerated dependency list does not mandate (that list names `pc-detect, pc-preprocess, pc-mask, pc-denoise, pc-export` — not `pc-ocr`). This ruling does **not** add `pc-ocr` to §16.5 item 1's list; `pc-ocr` still does not depend on `pc-config`.

   API (pure, testable with no model and no ONNX Runtime, in the default no-`onnx` CI tier):
   ```
   enum Device { Cpu (default), Cuda }                          // serde snake_case; Device::ALL = [Cpu, Cuda]
   struct DeviceSupport { .. }                                   // which devices THIS BUILD can register; a VALUE, not a cfg! read
   impl DeviceSupport { const CPU_ONLY; const WITH_CUDA; fn compiled() -> Self; fn supports(self, Device) -> bool }
   enum ConvAlgorithmSearch { Heuristic, Default }               // no `Exhaustive` variant: unrepresentable, per §16.22 item 3
   struct DevicePolicy { .. }                                    // resolved decision; no model, no path, no stage identity
   impl DevicePolicy {
       fn cpu() -> Self;
       fn requested(&self) -> Device;
       fn registration_must_error_on_failure(&self) -> bool;     // true whenever any explicit provider is requested (§16.22 item 5(c))
       fn provenance_execution_provider(&self) -> &'static str;  // derived from the resolved policy, never a separate literal (§16.22 item 6)
       fn report(&self) -> String;                                // requested + registered only, never per-node (§16.22 item 5(e))
   }
   enum DeviceRefusal { NotCompiledIn { requested: Device } }
   impl DeviceRefusal { fn message(&self) -> String }
   fn resolve(requested: Device, support: DeviceSupport) -> Result<DevicePolicy, DeviceRefusal>;
   ```
   `DeviceSupport::compiled()` is unconditionally `CPU_ONLY` in GPU-1; GPU-2 makes it conditional on its `cuda` feature. There is deliberately no `Cpu` variant of an execution-provider-request type: ONNX Runtime's built-in CPU provider is implicit today (`build_session` registers nothing), and registering one explicitly could perturb §16.32's bit-exact recorded floats — a CPU policy is structurally "no explicit registration."

3. **RULING (Fable, 2026-08-05) — GPU-1 wires the same ungated refusal into `pc-ocr`'s session creation, not only the detector's.** §16.22 item 5(c) ("loud refusal at session creation") and §16.23 item 5 ("§16.19-integrated fatal refusal") are both written unqualified, in terms of `device` and sessions, not "the detector's session" — item 5's measurements (items 3–4) happen to be detector measurements, but the binding conditions are not scoped to the detector. Leaving `pc-ocr` unwired would make `device = "cuda"` silently per-stage the instant CUDA linkage exists at GPU-2 (detector on CUDA, OCR silently still on CPU, stated nowhere) — exactly the invisible-divergence failure §14 item 7 and §16.22 item 5(c) exist to prevent. It is also the site that fires first in practice: verified in `crates/pc-cli/src/lib.rs`, the OCR factory builds a real ONNX session **eagerly** (before the first detector call, in both `run_clean` and `run_ocr`), while the detector's session is lazy per §16.19 item 1 — so with the shipped default `ocr_enabled = true`, the OCR refusal is what a user actually sees first.

   **Binding condition, so no future reader is surprised:** this grants `pc-ocr` a **refusal path only** — no CUDA execution, now or automatically at GPU-2. §16.22's nondeterminism and cross-provider measurements (items 3–4) were taken on the detector only; extending actual CUDA registration to OCR needs its own measurements and its own future ratification. Consequence to state plainly: after GPU-2, `device = "cuda"` with the shipped default `ocr_enabled = true` will **refuse fatally** — by design — until the user disables OCR or a later ratification grants OCR a CUDA path.

4. **Failure classification (§16.19 item 5 / cookbook rule 4) — no new machinery.** The refusal is raised inside session construction (`OnnxProvider::initialize_detector` for the detector; the eager OCR factory constructor for OCR), which takes no image, so §16.19 item 5(b)'s causal criterion classifies it run-fatal by construction, and `OnnxProvider::failures_are_run_fatal()` is **already** `true` in the shipped code. GPU-1 therefore produces a `StageError::Model(refusal.message())`, reuses the existing `OnceLock` latch (§16.19 item 2's preamble), and renders through the already-pinned shape: `error: model error: <message>` on stderr, exit `1`. No new `StageError` variant, no new `PipelineError` arm, and `DetectorProvider::detector_for`'s signature is untouched. With `--detector replay`/`mock` and `ocr_enabled = false`, `device = "cuda"` is accepted by config and inert (no session is ever created) — correct under the session-scoped framing above, and deliberately given **no** WARN (a WARN would be unratified surface dragging in §14 item 7's one-time-WARN machinery for no session that ever ran).

5. **Recording hard-refusal (§16.22 item 5(b)), made reachable rather than decorative.** `xtask/src/record.rs`'s existing `ensure_cpu_execution_provider(DETECTOR_EXECUTION_PROVIDER)` compares a constant against itself and can never fail; its recorded `execution_provider`/thread-count pins are literals, not values read from the config that built the session. Fix: add one `--device <cpu|cuda>` flag to the `record-fixtures` **command** (default `cpu`), applying uniformly to every `--only <group>` it dispatches — not a per-group flag — so §16.22 item 5(b)'s unqualified *"every fixture-producing path"* is not narrowed: the refusal fires **on the request** (before dispatching to any group at all — this is "recordings must be CPU," and must still fire once GPU-2 makes CUDA resolvable) and again **on the resolved policy** (immediately before session construction, for the two groups that create one, catching any future code path that obtains a device from elsewhere). `calibrate-goldens` gets its own `--device <cpu|cuda>` under the identical rule. Derive the recorded `execution_provider`/thread-count fields from the `TextDetectorConfig` that actually built the session, never a separate literal. `pc-testkit`'s `REQUIRED_EXECUTION_PROVIDER` becomes `Device::Cpu.as_str()` rather than a second independently-typed `"cpu"` literal — the same one-constant-not-two-independent-copies principle §16.31 item 2 ratified for the OCR model pins. **Every fixture-producing path, enumerated** (`xtask::record::Group::ALL`, six variants, `xtask/src/record.rs`): `Detector` and `calibrate-goldens` build a session and write a committed artifact → the request-level refusal above covers both. `ModelSignature`, `OcrModelSignature`, `Nlm`, `InterArea`, `FindEdges` build no `ort` session at all — the request-level refusal still applies to them when dispatched via `record-fixtures --device cuda`, uniformly, per the no-narrowing rule above; there is simply no *second*, session-level refusal for them to also trip, since they create no session, which is the sense in which they need "nothing further," not an exemption from the flag. `xtask bench` builds a session but writes only to scratch (deleted after) → **report the resolved device**, not refuse, since bench numbers get quoted into spec sections and an unlabelled number is the same invisible-provenance failure in a different shape. `docs/GOLDEN_CALIBRATION.md` inlines a copy of the detector `PROVENANCE.json` and is therefore a second reader of `execution_provider` that the acceptance gate must cover. The schema itself does not change — only the producer's derivation — so the committed `PROVENANCE.json` fixture is expected to remain byte-for-byte unmodified; do not retype `OursPins.execution_provider` from `String` to `Device`, since the frozen `crates/pc-testkit/tests/provenance_schema.rs` constructs it as a `String` literal and typing it would break a frozen test for no correctness gain.

6. **Task sequencing.** G1-A + G1-B (**simple**, one Codex call): `pc_core::device` (item 2) plus the `[general] device` config surface (item 1) — `crates/pc-config/tests/defaults.rs` gains `device` in `general_defaults`, plus a `device = "cuda"` **loads-and-validates** test mirroring the existing `annotation_refine_mode_loads_successfully` pattern (§16.5 item 3's ratified split: config accepts it, the stage refuses it — the refusal belongs at session creation, not config load, or `--detector replay`/`mock` runs would break for a device nothing in them ever uses). G1-C (**heavy**, its own call): the session-creation refusal at both seams (item 3) plus `pc-cli` wiring. Two distinct plumbing hazards, not one: `OnnxProvider`'s existing construction copies out only `intra_threads`/`inter_threads` from `TextDetectorConfig` (`crates/pc-cli/src/detector.rs`), so any *other* `TextDetectorConfig` field added later would be silently dropped by that pattern — but `device` is **not** a `TextDetectorConfig` field (it lives in `GeneralConfig`, per item 1), and `build_provider`'s call site passes only `&profile.text_detector` today, so `device` must be threaded through as its own new argument from `profile.general`, not smuggled in by "carry the whole config." Verify `device` actually reaches `initialize_detector` end to end before calling G1-C done — a refusal test that never observes a non-default `device` because the plumbing dropped it silently would pass while the feature does nothing. G1-D (**heavy**, its own call, depends on G1-A/G1-C): the recording/calibration refusal and provenance derivation (item 5). Do not batch G1-A/G1-B with §16.35's `[masker]` config-surface task (item 8/T4 there) in the same Codex call — see §16.35 item 9.

7. **Hand-offs to GPU-2, pre-authorised here so they are not discovered mid-task as frozen-test conflicts.** `DeviceSupport::compiled()` becomes conditional on GPU-2's `cuda` feature — GPU-1 deliberately does not freeze `compiled() == CPU_ONLY` as an assertion, only the capability-explicit `CPU_ONLY`/`WITH_CUDA` constructors, which never need editing. The end-to-end refusal test in G1-C (`a_cuda_device_refusal_is_attempted_once_and_declared_run_fatal`-shaped) will need a `#[cfg(not(feature = "cuda"))]` gate once GPU-2's feature exists, following the same pre-authorised-frozen-test-edit precedent §16.23 item 5 already set for PERF-1. GPU-2 also owns: the downgrade-guard test for `ort-sys`'s silent fallback to its CPU-only `none` distribution (§16.22 item 5(d) — the guard file cannot be written in GPU-1, since `cfg(feature = "cuda")` on an undeclared feature is an `unexpected_cfgs` clippy failure), and the `#[non_exhaustive]`-equivalent question of whether `ort`'s CUDA session options actually honour `ConvAlgorithmSearch::Heuristic` (item 2's type only makes `Exhaustive` unrepresentable in *our* policy — it does not itself prove `ort`'s behaviour, which needs an assertion at the `ort` boundary GPU-2 introduces).

8. **Conflict avoidance against the parallel v1.1 mask-polish branch.** See §16.35 item 9 for the shared statement: `[general]` (this branch) vs `[masker]` (mask-polish) are different structs in `profile.rs` and different blocks in `default_profile.toml`, low textual risk, but land the two config-surface edits sequentially rather than via parallel Codex calls, because of `table_registry_matches_default_document`'s whole-registry-vs-whole-document comparison. This is **§16.36**; mask-polish is **§16.35**. This branch's opt-in-only divergence is `DEVIATION(22)`; neither branch takes `20`, left unused per §14's note above.

## 16.37 Annotation-mode refinement (Task A), R0: the tie-order deviation, two corrections of statements that are false today, and the R0→A6 sequence (joint architect + Senior Rust Engineer plan, 2026-08-06)

**Historical record (2026-08-06, R0).** This section was written before A4: it recorded the Annotation port as unimplemented, `pc-detect` rejected the mode, `MaskRefineMode::Simple` was the shipped default, and no frozen golden value moved. A4 later landed the markers and wiring described by §16.39; the dated record is retained for provenance, not as a current-state claim. At R0, the markers and rejection-clause amendments were reserved for **A4**, not written here; §16.39 later records their landing, which is the sequencing §14 item 16 already states in the opposite direction (*"The implementation site will carry `DEVIATION(16)`; the concurrent implementation pass has not added that comment yet."*) — the register entry lands with the ratification, the site marker with the implementation.

**Two** of the items below — items 4 and 5, the only two labelled `CORRECTION` — correct statements that are **false as this section is written**. Both are false about `Simple`, which ships today, and neither is a consequence of this port: reading them as caused by Annotation would misattribute a pre-existing defect to a change that has not landed.

1. **Upstream's Annotation output is not reproducible across environments, and the mechanism is measured rather than argued.** `get_topk_color` (`pcleaner/comic_text_detector/utils/textmask.py:19`, pinned commit `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`) orders its candidate colours with `idx = np.argsort(bins * -1)`. NumPy's default `kind='quicksort'` is introsort, whose ordering **among equal keys is not part of NumPy's API contract**, and the input is saturated with equal keys. Measured per `argsort` call, as the size of the **largest** group of equal bin counts in that call: 47 calls on `ja_Pepper-and-Carrot_by-David-Revoy_E01P02` ranging **35 to 255** of 255 bins, and 22 calls on `…E01P03` ranging **69 to 255** — so every call carries a large tie group, but the sizes are per-call and per-page, not a fixed set. Changing only the tie rule — default introsort to `kind='stable'`, nothing else — changes upstream's own refined mask on E01P02 by **277 differing pixels (251 turned on, 26 turned off, net +225: 6230 → 6455 non-zero)**, and on E01P03 by **27 differing pixels (0 on, 27 off, net −27: 1831 → 1804)**, while leaving **`…E01P01` byte-identical**. The two figures per page are different metrics and are stated separately on purpose: the symmetric difference is the parity-relevant one, the net non-zero delta is not, and quoting one as though it were the other is what the first draft of this item did. Independently, CPU dispatch alone (AVX2 vs SSE, via `NPY_DISABLE_CPU_FEATURES`) changes the same tie order enough to change how many candidate colours `get_topk_color` returns (2 vs 3) on synthetic input, and likewise leaves E01P01 byte-identical. Neither hazard is pinned by anything we record: `tests/fixtures/recorded/detector/PROVENANCE.json` pins `numpy`, `opencv`, `python` and `torch`, and pins **no CPU feature set**. Both readings on the record are claims about the pages they were measured on and neither generalises — E01P01's stability is a property of E01P01's data exactly as E01P02/E01P03's instability is a property of theirs.

2. **DECIDED: v1.5 declares its own tie rule rather than chasing an unspecified one.** `get_topk_color`'s histogram ordering is a **stable** sort on `(Reverse(count), ascending bin index)`. Registered as `DEVIATION(23)` in §14; the implementation site is the histogram/tie-rule function landed by A1 — `top_k_colors` in `crates/pc-detect/src/annotate.rs` — and must carry that comment, which **exists** as of 2026-08-06, at `crates/pc-detect/src/annotate.rs:278` (the doc comment declaring the rule) and `:295` (the sort site inside `top_k_colors`). **Line-number correction, 2026-08-06, landing with item 10:** those two numbers read `:266` and `:283`, correct at A1 and moved by A2's twelve-line growth of this file's module doc comment above them — see §14 item 23's second correction for the verification. **Correction, 2026-08-06:** this clause read *"which does not exist yet"*, **correct when written** at R0 — spec-only, no code — and stale the moment A1 landed, which is the register-entry-then-site-comment sequencing §14 item 19 describes; staleness, not an original error, and corrected the same way item 19 corrects its own citation. This is the same shape as §14 item 2, whose grounds transfer verbatim — *"upstream iterates a Python `set`, making the merge order (and therefore the merged boxes) nondeterministic between runs. We use index order (FIFO)."* Item 1's measurement is what makes porting introsort **not merely expensive but ill-posed**: an introsort port would still be a function of a CPU feature set nothing pins, so there is no upstream ordering to be faithful *to*.

3. **The frozen gate is ours-vs-ours and proves DETERMINISM, not parity — stated here because a future reader will otherwise read "gate" as "parity gate".** The A5 gate asserts our refinement is byte-identical across runs and thread counts on the recorded page. Ask what change turns it red: only nondeterminism. **A port that is wrong in the same way on every run passes it forever.** Correctness evidence therefore comes from A1–A3's per-function hand-derived assertions, not from A5. Any comparison against upstream is a **non-gating calibration report**, the same downgrade §15.2 item 2 already applied to this stage's parity claim — *"§10.7(B) item 15's upstream-image comparison is downgraded from a frozen gate to a **non-gating calibration report**"* — and that report must additionally pin the CPU feature set it was produced under, or item 1's second hazard makes the report itself irreproducible. **Widened by item 10 below (2026-08-06), which adds a second thing the same report must pin:** OpenCV's **IPP build and enable status** — the `cv2.getBuildInformation()` IPP line *and* whether IPP was enabled for the run — because item 10 measures a second environment-dependent path (Otsu's) that the CPU-feature-set pin does not cover. The requirement is widened, not replaced: both pins are required, and A5 records both or the report is not reproducible for either reason. **Widened a second time by item 11(d) below (2026-08-07) — on SCOPE, not on pins:** the gate and the report must run against the **complete** algorithm, *"both `refine_mask` and `refine_undetected_mask`"*, not a partial one, because *"A calibration report produced against `refine_mask` alone would compare our half-algorithm with upstream's whole one and attribute the difference to a port defect"*. Pointer added here so the precondition is visible at the item a reader lands on for the A5 gate, mirroring the item 10 pointer above; the ruling itself is item 11(d)'s and this note adds nothing to it.

4. **CORRECTION, true of `Simple` today and unrelated to this port: §14 item 17's coverage-filter measurement is false on the committed fixture.** Item 17 states, verbatim: *"measured across two real manga pages the filter has never fired (minimum coverage 0.3025 against a 0.1 threshold)"*. Measured under the shipped `Simple` mode against `tests/fixtures/recorded/detector/…E01P01_raw_mask.png` — **that file, and not `…_detector_mask.png`**, because despite its name `_raw_mask.png` holds the *refined* mask (`crates/pc-detect/src/lib.rs:105-106` writes `refined_mask` to `raw_mask_dest`) while `_detector_mask.png` holds the unrefined U-Net output; reading the wrong one yields 0.0330 for this block, the same conclusion by a different number — block `[438, 1407, 498, 1446]` has `mask_coverage` **0.0867521** and **is dropped** by the `>= DEFAULT_MIN_MASK_COVERAGE` filter — which is why `…_detector_blocks.json` holds four blocks and `…#raw.json` holds three. The filter has fired, on the one page every consumer of that fixture reads, since before this section was written. Method: `mask_coverage` (`crates/pc-detect/src/lib.rs:41`) reimplemented independently and validated against the three committed `#raw.json` rows to within 3.2e-8 before being trusted. Upstream drops the same block (its own pre-filter `mask_score` for that box is 0.03446 with `len_lines: 0`), so **behaviour agrees and only the sentence is wrong**; item 17's scope and operand rulings are untouched by this correction.

   **SUPERSEDES: §14 item 17**

5. **CORRECTION, true of `Simple` today: §8.3 step 5's `expand_textwindow` parenthetical claims a parity that does not hold.** Step 5 states, verbatim: *"For each detected block, `expanded = rect.pad(16, image_size)` (upstream `expand_textwindow(expand_r=16)`)"*, and the same claim is repeated at `crates/pc-detect/src/mask.rs:12`. Upstream's `expand_r` is a **divisor**, not a pad: `paddings = int(round((max(h, w) * 0.25 + min(h, w) * 0.75) / expand_r))` (`comic_text_detector/utils/imgproc_utils.py:168`). Run on the four committed E01P01 boxes it yields **3, 4, 5 and 3 px**, against our flat 16. A second divergence in the same function, not previously recorded anywhere: upstream clamps the far edges to **`im_w - 1` / `im_h - 1`**, while `pc_core::Rect::pad` (`crates/pc-core/src/geometry.rs:91-92`) clamps to `canvas.0` / `canvas.1` — off by one at every image edge. **What this corrects is a provenance CLAIM, not `Simple`'s behaviour.** §8.3 step 5 describes `Simple` as *"ported from koharu's `refine_segmentation_mask`"*, not from PanelCleaner, and whether koharu uses a flat 16 is **unverified** — koharu is not vendored in this tree and was not consulted. `Simple`'s padding is therefore left exactly as it is; A0 and A1 bind the corrected formula and the `- 1` clamp to `Annotation` only. Deciding whether `Simple` is also wrong needs koharu as its own oracle and is not decided here.

   **SUPERSEDES: §8.3 step 5**

6. **Measured, NOT decided here: Annotation collapses `mask_coverage` and would silently drop blocks.** Same method as item 4, comparing the shipped `Simple` mask against upstream's Annotation output with our boxes held fixed: E01P01 `0.7812269 → 0.3920916`, `0.7731718 → 0.1736820`, `0.5636277 → 0.1226258` (kept, but only **22.6% above** the 0.1 threshold, i.e. `(0.1226258 − 0.1) / 0.1`); E01P02 `[349,824,492,878]` **`0.4071484 → 0.0238280`** and `[531,836,686,882]` **`0.1088359 → 0.0000000`**; E01P03 `[794,1310,905,1344]` **`0.3497615 → 0.0312666`**. Three blocks that `Simple` keeps would be dropped — no mask, no OCR, no diagnostic. A0 must measure this end to end rather than inherit it: the counterfactual recorded during planning (painted regions 3/9 → 6/9 across the three pages) held the box set fixed and so is an **upper bound on the improvement that does not include this feedback**. What to do about it — accept, re-tune `min_mask_coverage` for this mode, or change the filter's operand — is an A4 decision and is deliberately left open.

   **Which of the two Annotation outputs the figures above came from, settled by measurement (Fable's empirical ruling on the comparand, 2026-08-07, added with item 11 below).** *"upstream's Annotation output"* is ambiguous between two things upstream can produce: `refine_mask` alone, and `refine_mask` followed by `refine_undetected_mask` — the latter being what PanelCleaner's own pipeline runs (`keep_undetected_mask=True` at `pcleaner/ctd_interface.py:162`, reaching `refine_undetected_mask` at `pcleaner/comic_text_detector/inference.py:205-208`). Both variants were run at the pinned commit under opencv 5.0.0 with IPP disabled, with A1's corrected proportional window, on all three pages. **At PAGE level the two differ substantially:** E01P02 by **3772** differing pixels (6230 non-zero with the undetected pass against 2458 without) and E01P03 by **1038** (1831 against 793). E01P01 differs by **0** px — on that page the undetected pass invents no block at all, so its "agreement" there is degenerate and carries no information about the other two. **On every block rect this item records, on all three pages, the two variants' `mask_coverage` values are bit-identical** — not merely equal to seven decimal places — so no figure above depends on which variant produced it, and the coverage decision A4 owes is unaffected by the choice. The reason is that on these pages the invented components' pixels fall entirely outside these rects.

   **Scope, and this half is the load-bearing one:** the bit-identity is a property of **these specific block rects, on these specific pages, in this environment** — it is **not** a property of the algorithm, and it must not be restated as *"`refine_undetected_mask` never affects `mask_coverage`"*. An invented component can in principle overlap a detected rect and still be invented (the invention test is an intersection-area ratio below 0.5, not disjointness — see item 11(b)), and then it would move that rect's coverage. Method: both variants computed over each page's `_base.png` + `_detector_mask.png` + `_detector_blocks.json`, with the harness validated first by reproducing `Simple` byte-identically against the committed `…E01P01_raw_mask.png` before any Annotation number was read off it; `mask_coverage` per `crates/pc-detect/src/lib.rs:41`.

   **Left OPEN by that ruling, and recorded as open rather than guessed: whether A0's attribution baseline included the undetected pass.** The *"painted regions 3/9 → 6/9"* counterfactual quoted above was recorded during planning, and **which of the two variants its Annotation arm used is unaudited**. This section does not resolve it and nothing sequenced before A4 depends on it, but it is not a nothing either: the page-level figures above move a page by thousands of pixels, which could move a painted-region count. It is a **follow-up check A0 owes**, not a blocker.

7. **Authorised, landing with A4, not with this section: the SPEC clauses that reject Annotation, plus the one frozen test. This item's enumeration is NOT a complete list of every site in the repo that says so.** `crates/pc-detect/tests/d7_run.rs:196-215`'s `run_rejects_annotation_refine_mode` asserts `StageError::InvalidInput` and `detector.calls() == 0`. Under cookbook rule 8's test — *"Does the corrected assertion claim **less about the system** than the original did?"* — the corrected assertion does (from always-rejected to rejected-unless-configured), so this is **exit 2 and routes to both architects**, which this section is. **Predicted, not verified:** that test is expected to go red when A4 lands, but whether it actually does depends on A4's opt-in design, which does not exist yet — if A4 gates on a config flag whose default keeps the refusal, the test could stay green, and A4 must check rather than assume.

   **The clauses this item authorises amending, and the exact nature of each:** §16.5 item 3 (*"**`mask_refine_mode = "annotation"` is accepted by config, rejected by `pc-detect`** (`StageError::InvalidInput`)"*) and §8.3 step 5's *"where only `Simple` is implemented in v1 (`Annotation` → `StageError::InvalidInput`)"* both become **false** and need markers. §15 item 2's *"`Annotation` mode remains the v1.5 door for a full refinement port"* **rejects nothing and does not become false — it becomes stale**, describing a door as unopened once it is opened; A4 decides whether staleness warrants a marker at all, and this item does not pre-judge that.

   **Anchor spelling, settled here to prevent a later panic:** the third clause must be cited as **`§15 item 2`**, never `§15.2 item 2`. There is no `## 15.2` header — §15's items live under `## 15.` — so `§15.2` resolves to a nonexistent section, and `every_supersession_marker_has_a_back_pointer_at_its_target` **panics** on an unresolvable anchor rather than skipping it. The registry already uses the correct form at `("16.34", "15 item 5")`. Prose writes `§15.2 item 2` in exactly two places, both inside this section — item 3 above and this sentence — and nowhere else in the file (`grep -n "§15\.2 item"`); that form is harmless in prose and fatal in a marker.

   **Code sites are A4's own enumeration, deliberately not asserted complete here** (cookbook rule 14; §14 item 18 in this register sets the precedent of enumerating comment sites explicitly rather than gesturing at them). Known starting points, each **re-verified by reading the cited line** while writing this item, and **not** claimed exhaustive: `docs/PIPELINE_SPEC_V1.md:499` and `crates/pc-config/src/default_profile.toml:20` (both the `mask_refine_mode` default line and its trailing "annotation is rejected" comment; the TOML is locked by `crates/pc-config/tests/defaults.rs`'s literal-§6-block comparison, so editing it without the test is a red suite), `crates/pc-config/src/profile.rs:208-210`, `crates/pc-detect/src/lib.rs:71`, and two sites in the **stale-not-false** category described above for §15 item 2 rather than the rejection category — `crates/pc-detect/src/mask.rs:152` (a "v1.5 door" sentence in `refine_simple`'s `DEVIATION(12)` comment, which rejects nothing) and `README.md:102` (a roadmap checkbox, `- [ ] \`Mask RefineMode::Annotation\``). Both merely stop describing reality once A4 lands. **`crates/pc-cli/src/detector.rs` is NOT such a site** and was removed from an earlier draft of this list: `grep -rn -i annotation crates/pc-cli/src/` returns nothing at all. A4 re-derives this list with a fresh grep and records the result; it does not inherit this one.

8. **Task sequencing.** **R0** (this section, spec-only, no code). **A0** (**simple**, non-gating): re-run the Simple-vs-Annotation counterfactual with item 5's corrected window *and* item 6's block-set feedback both live, and attribute how much of the measured improvement is each — two variables, so report them separately or the result is uninterpretable. **A1** (**heavy**): the greyscale/histogram/top-k-colour path including item 2's tie rule — pure functions over committed arrays, hand-derivable, no model. **A2** (**heavy**): Otsu thresholding and the XOR-minimising channel selection. **A3** (**heavy**): connected components plus the merge and hole-filling loops. Connected-component **label order** is a measured non-issue on all three pages — reversing the entire non-background label numbering changes upstream's output by 0 px on E01P01, E01P02 and E01P03 — but that is a property of those pages, so A3 states it as a recorded observation, never as a licence to ignore ordering. **A3b** (**heavy**, added 2026-08-07, sequenced HERE — after A3 and **before A4**): `refine_mask`'s page-level driver plus `refine_undetected_mask`; item 11 below is its ruling and carries its full scope, and the grounds for placing it before A4 rather than after. The letters A0–A6 keep exactly the meanings they have in this list — the task is inserted as `A3b` rather than by renumbering, so no existing citation of `A4`, `A5` or `A6` anywhere in this file or the tree moves. **A4** (**heavy**): wiring, the config gate, item 6's coverage decision, item 7's clause amendment **and its own fresh enumeration of the code sites**, and the `DEVIATION(12)` retirement marker. **A5** (**simple**): item 3's determinism gate plus the calibration document. **A6**: a committed upstream fixture group, a separate task, later, and **only if a future §16.x wants a gated comparison** — item 1 is why it is not wanted now.

9. **Scope quarantine, so no gate moves under this work.** `Annotation` ships **non-default**; `Simple` stays the default and every value frozen against it stays put. Adding a second recorded page or a new fixture group is **not** in A0–A5: `xtask/src/record.rs`'s `detector_plan` is single-page by §16.29 item 2, and `crates/pc-testkit/tests/recorded_provenance.rs` carries a hard-coded group count and hard-coded artifact paths. That is A6's problem, under its own ruling.

10. **A2's measurements: OpenCV's Otsu is environment-dependent, which is why `DEVIATION(29)` exists — and the measurements, not the argument, are what carries it (independent architect + Senior Rust Engineer takes, Fable tie-break on the `PROVENANCE.json` sub-point, 2026-08-06).** A register entry without its supporting measurement is the bare assertion the cookbook warns against, so the runs are recorded here rather than summarised into item 29.

    (a) **IPP-on and IPP-off disagree on real measured samples.** Method: OpenCV's published `getThreshVal_Otsu_8u` transcribed independently into Python, validated against `cv2.threshold(..., THRESH_OTSU)` with `cv2.ipp.setUseIPP(False)` **before** being trusted, then compared against the same call with IPP enabled. **There is no reproducible rate, and none is stated here.** Every measurement on record, with its provenance, each over 20 000 arrays:

    | disagreements / 20 000 | ≈ rate | corpus | measured by |
    | --- | --- | --- | --- |
    | 0 | none at all | smooth bimodal (two Gaussian modes, well separated) | this correction, 2026-08-07 |
    | 1 | 1 in 20 000 | uniform-random small arrays | this correction, 2026-08-07 |
    | 5 | 1 in 4 000 | not characterised in the report that reached this transcription | fresh reader, 2026-08-07 (as reported; not re-run here) |
    | 10 | 1 in 2 000 | random | Senior Rust Engineer, 2026-08-06 |
    | 20 | 1 in 1 000 | random, second corpus | Senior Rust Engineer, 2026-08-06 |
    | 21 | 1 in 952 | random | architect, 2026-08-06 |
    | 54 | 1 in 370 | small arrays over narrow value ranges, tie-dense by construction | this section's original transcription, 2026-08-06 |
    | 148 | 1 in 135 | not characterised in the report that reached this transcription | fresh reader, 2026-08-07 (as reported; not re-run here) |
    | 154 | 1 in 130 | narrow value ranges (2–6 distinct values, 2×2 to 12×12) | this correction, 2026-08-07 |
    | 460 | 1 in 43 | tiny arrays, ≤4 distinct values, values repeated in equal runs | this correction, 2026-08-07 |

    **The spread is from zero to 1 in 43 — over two orders of magnitude across the non-zero runs, and one corpus in which the two paths never disagreed at all.** It tracks the corpus's **tie density** (how often two candidate between-class variances land equal or within a few ULPs) and, per (c) below, the **CPU dispatch level**; it is not a property of "the measuring environment" and cannot be pinned to a band. This sentence previously read *"on the order of 1 in 1000–2000 array samples in the measuring environment"*, which covered only the 10- and 20-disagreement runs and was already contradicted by the 54 run in its own paragraph, then by five further runs; it is replaced rather than re-tightened, because a tighter band would fail the same way. Quote a rate only with the corpus that produced it.

    **`DEVIATION(29)` does not rest on the magnitude, so nothing above weakens it.** Item 29 is grounded on `cv2.threshold(..., THRESH_OTSU)` not being **single-valued** — that some inputs disagree at all, plus (c)'s dispatch-level dependence and (e)'s x86-only availability. A single non-zero run establishes that; a corpus measuring 0 does not unestablish it, and a corpus measuring 460 does not strengthen it. The rate matters only for judging exposure, which (d) settles for the committed page.

    Every run **that reported it** agreed on one thing: with IPP **disabled**, agreement with the hand-transcribed reference was **total** — 0 disagreements out of 20 000. Confirmed first-hand for the four corpora measured for this correction, which include the two highest IPP-on rates in the table (460 and 154); reported, not re-run here, for the earlier runs, and the two fresh-reader rows carry no IPP-off figure at all. So the IPP-off half is *strongly* evidenced and not uniformly re-verified, and the table's rate column is IPP-on only.

    (b) **Both mechanisms occur — exact ties AND ULP-level near-ties. This corrects an earlier statement in the A2 test file.** `crates/pc-detect/tests/a2_annotate_otsu.rs`'s module doc comment read *"None of the 10 is an exact tie in double arithmetic: in every one the two candidate between-class variances differ in the last few ULPs, so the mechanism is IPP's arithmetic, not a tie rule"*. That is an **overclaim** — a property of one corpus restated as a property of the mechanism — found independently by both reviewers: the architect found an exact, bit-identical-sigma tie arising from **random, non-constructed** data on which IPP and the reference disagree, which contradicts the sentence directly. Re-measured during this transcription on corpus (a)'s fourth run: of 54 disagreements, **15 had bit-identical `f64` between-class variances** (verified by comparing the two `f64` values for equality and printing both as hex, e.g. input `[152]*7 + [153]*7 + [154]*7`, both sigmas `0x1.0000000000000p-1`, reference 152, IPP 153) and **39 differed in the last few ULPs**. So **both** mechanisms are present and neither explains the divergence alone. The doc comment is corrected to match; that is a **doc-comment-only** change — no frozen assertion in that file moves, and the test that pins the tie direction (`otsu_threshold_breaks_exact_between_class_variance_ties_toward_the_lowest_index`) already asserted the reference rule and still does.

    (c) **IPP's own answer varies by CPU dispatch level with IPP held enabled, which is what makes "match IPP" not even well-defined as a target.** `OPENCV_IPP=sse42` against `avx2` and `avx512` returns different thresholds on one fixed array corpus (re-verified here over 4 000 arrays: the `sse42` result set differs from `avx2`'s and `avx512`'s, which are identical to each other and to the unset default). So "be faithful to IPP" would have to name a dispatch level as well as a library, and neither is pinned by anything anyone ships.

    (d) **The committed CC-BY test page is NOT exposed — a real mitigating measurement, and the reason item 29's severity is not item 23's.** All **12** per-channel Otsu thresholds across `ja_Pepper-and-Carrot_by-David-Revoy_E01P01`'s four boxes are **identical** under the reference and IPP paths. Verified independently by both reviewers and re-verified by the adjudicator, and re-verified a fourth time during this transcription against the committed `…_base.png` on the four A1-derived windows: block 0 `[76, 117, 131]`, block 1 `[122, 153, 134]`, block 2 `[126, 156, 134]`, block 3 `[78, 116, 125]` in upstream's B, G, R order, byte-identical with `cv2.ipp.setUseIPP(True)` and `(False)`. **`DEVIATION(29)`'s real-world severity is therefore qualitatively different from `DEVIATION(23)`'s**, which did move a real page (277 px on E01P02) — item 29 does not inherit item 23's severity framing, only its grounds. As with item 1's E01P01 result, this is a property of **this page's data** and generalises to nothing.

    (e) **An IPP-faithful rule would not even be implementable uniformly across the targets we ship.** `.github/workflows/release.yml` builds `aarch64-apple-darwin` (verified: that target appears at two rows of the matrix), and Intel IPP is **x86-only**. So "match IPP" has no meaning at all on one of this project's own release targets — an additional and independent reason to prefer the reference path, beyond it being the principled choice.

    (f) **DECIDED, against the alternative reading: `tests/fixtures/recorded/detector/PROVENANCE.json` is NOT extended.** This was the one genuine disagreement between the two takes; the adjudicator ruled for the architect's position. Two reasons, both checked rather than argued: **no artifact that file pins is IPP-sensitive** — the detector oracle recorder discards the refined mask (`xtask/scripts/record_detector_oracle.py:646` binds it as `_mask_refined` and never reads it), so no Otsu-derived value can propagate into a pinned artifact; and the file is **generator-produced from a fixed key list** (`xtask/src/record.rs:894-903`, `["python", "opencv_version", "numpy_version", "torch_version"]`), so a hand-added IPP field would be dropped silently at the next re-record. Extending it would place a pin on files that do not carry the risk while leaving the pin unenforceable — cookbook rule 12's "gate the artifact carrying the risk", inverted.

    (g) **Instead, two obligations at the artifacts that do carry the risk.** The first is item 3's, already widened above: A5's calibration report pins OpenCV's IPP build and enable status as well as the CPU feature set. The second is **binding on A6**: if A6 ever commits upstream's Annotation-mode mask as a fixture, that fixture's provenance **must** pin IPP status at that point. **Guidance for whoever does A6, added by the adjudicator and deliberately not a requirement here:** a bare on/off boolean is **insufficient** given (c)'s dispatch-level sensitivity — pin a **measured canary** alongside the build string, i.e. the Otsu threshold actually returned in the recording environment on a known tie-exercising array (item (b)'s 21-element input is one such array, with `152` on the reference path and `153` on this machine's IPP path), so a future re-record can detect a dispatch-level change and not merely a library swap. A6 decides the format.

11. **DECIDED: `refine_mask`'s page-level driver and `refine_undetected_mask` are one task of their own — `A3b`, heavy, sequenced after A3 and BEFORE A4 (independent architect + Senior Rust Engineer reviews of A3, Fable tie-break on the sequencing, 2026-08-07).** The two reviews independently found the same gap in item 8's task list, and the finding is three separate facts rather than one:

    (a) **`refine_undetected_mask` (`pcleaner/comic_text_detector/utils/textmask.py:161-192`, pinned commit `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`) is named by NO task in item 8's original A0–A6 enumeration** — not by A3, whose scope is *"connected components plus the merge and hole-filling loops"*, and not by A4, whose scope is wiring and policy. **It is not dead code, which is the reading that would have made the gap harmless:** PanelCleaner's real pipeline reaches it on every page, via `keep_undetected_mask=True` at `pcleaner/ctd_interface.py:162` and the `if keep_undetected_mask:` branch at `pcleaner/comic_text_detector/inference.py:205-208`. Item 6's note measures what it does on real pages — 3772 differing pixels on E01P02, 1038 on E01P03 — so it is live and it is not small. **And `DEVIATION(12)` names both halves:** §14 item 12 reads *"v1 ships the "Simple" refinement, not upstream's `refine_mask`/`refine_undetected_mask`"*. If A4 retires that entry while `refine_undetected_mask` stays unported, the retirement asserts a parity the tree does not have — the exact class of false claim items 4 and 5 of this section exist to correct. **So this is a scope gap, not a scope choice**, and closing it before the retirement is what makes the retirement honest.

    (b) **A3b's scope, in two parts.** **Part 1, `refine_mask`'s PAGE-LEVEL DRIVER (`textmask.py:195-212`)** — a thin per-block composition loop: expand the window with A1's `expand_text_window`, crop image and pred mask, build the candidate list through A1's top-k path and A2's Otsu path (upstream's `get_topk_masklist` + `get_otsuthresh_masklist(per_channel=False)`, which our `candidate_mask_list` already composes), call A3's `merge_mask_list`, and OR the result into a page-sized mask at the window's coordinates. **This is pulled OUT of A4's scope and into A3b** (Fable's ruling): it is *"numerical work with an upstream oracle, not integration work"*, and both `refine_mask` proper and `refine_undetected_mask`'s recursive call into it need it as one standalone callable unit — writing it inside A4's wiring would make the recursive caller depend on A4. **Part 2, `refine_undetected_mask` itself**, whose five load-bearing details are, in upstream's order: **(i)** `mask_pred[np.where(mask_refined > 30)] = 0` — a zero-where-already-refined step at threshold **30**, and note it mutates the CALLER's array in place, which upstream then returns as its "unrefined" mask from `inference.py:210` (verified by probe, 2026-08-07: a 40×40 pred of uniform 200 with a 5×5 refined patch comes back with **25** pixels zeroed in the caller's own array). A3b decides deliberately whether our port mutates or copies, and records which; **(ii)** `cv2.threshold(mask_pred, 30, 255, THRESH_BINARY)` then a connected-components labelling, reusing A3's `connected_components`; **(iii)** `valid_labels = np.where(stats[:, -1] > 50)[0]` then `for lab_index in valid_labels[1:]` — and **this is NOT "skip the background label"**. It drops the **first surviving entry** of the area-filtered list, which coincides with label 0 only when the background's own area also exceeds 50. Measured 2026-08-07 on an 8×8 all-foreground image with a single background pixel: `stats` areas are `[1, 63]`, `valid_labels == [1]`, and `valid_labels[1:] == []` — so the **only real component is silently dropped**. A port that writes "skip label 0" is a different function; **(iv)** the invention test, `union_area(blk.xyxy, bbox) / w / h < 0.5`, maximised over every detected block. `union_area` (`pcleaner/comic_text_detector/utils/imgproc_utils.py:15-22`) is **misnamed**: it computes the **intersection** area of the two boxes, and returns the sentinel **`-1`** when they are disjoint (so a component overlapping nothing scores `-1/w/h`, which is `< 0.5`, and is invented). Port the behaviour under its own name, not under its upstream name; **(v)** the recursive `refine_mask(img, mask_pred, seg_blk_list, ...)` over the invented blocks, taking the **already-zeroed** `mask_pred` from (i), OR-ed into the refined mask — which is why part 1 must be a callable unit.

    (c) **Connectivity: port the EFFECTIVE 8, not the literal 4 the source text passes, and no `DEVIATION` is needed for it.** Upstream writes `cv2.connectedComponentsWithStats(pred_mask_t, 4, cv2.CV_16U)` (`textmask.py:170`), whose positional arguments bind to the `labels` and `stats` **output** slots, so `connectivity` keeps its default **8** and `ltype` keeps `CV_32S`. Re-measured on opencv 5.0.0, 2026-08-07: on `[[0,0,0,0,255],[0,255,255,0,0],[255,0,255,255,0]]` the positional calls `(m, 4, CV_16U)` and `(m, 8, CV_16U)` both return `num_labels = 3`, matching `connectivity=8`, while `connectivity=4` returns `4`; and the returned `labels` dtype is `int32`, not `uint16`. So upstream **intends** 4 here and **runs** 8 — a latent upstream defect, recorded rather than reproduced-as-intended. Matching what upstream *does* rather than what it *says* needs no register entry: that is the precedent `DEVIATION(29)` set in the same section, where the choice between two implementations of one nominally-specified OpenCV function was resolved by what the code actually runs. (The same reading for `merge_mask_list`'s call site — where intent and effect agree, so the finding is free — is recorded in `crates/pc-detect/src/annotate_merge.rs`'s module header note (a), **not** in item 8, whose A3 note is only about connected-component label order and carries no connectivity statement.)

    (d) **The sequencing was the one genuine disagreement, and Fable ruled for BEFORE A4.** The Senior Rust Engineer proposed sequencing it after A4, on the grounds that A4's wiring gives it a place to plug in. Fable ruled the other way: A3b is numerical work with an upstream oracle to check against, and A4 is integration and policy; running the numerical work first keeps A4's `DEVIATION(12)` retirement honest at the moment it lands rather than provisionally true pending a later task. **Consequences, stated so nothing is inferred:** A4's scope is exactly the list in item 8 — wiring, the config gate, item 6's coverage decision, item 7's clause amendments and its own fresh enumeration of the code sites, and the `DEVIATION(12)` retirement — with the page-level driver removed from it, since that driver was never named in item 8's A4 text but **was** described as A4's by `crates/pc-detect/src/annotate_merge.rs`'s module header, which this item corrects. **A5 gains an explicit stated precondition:** its determinism gate and calibration report must run against the **complete** algorithm — both `refine_mask` and `refine_undetected_mask` — not a partial one. A calibration report produced against `refine_mask` alone would compare our half-algorithm with upstream's whole one and attribute the difference to a port defect.

12. **DECIDED, and deliberately NOT a `DEVIATION`: A3's connected-component label numbering differs from OpenCV's, and no §14 entry is registered for it (independent architect + Senior Rust Engineer reviews of A3, no disagreement, 2026-08-07).** `pc_detect::annotate_merge::connected_components` numbers labels in ascending order of each component's **first raster pixel**; `cv2`'s default 8-way algorithm is block-based (Grana/BBDT) and numbers by 2×2-block scan order. The two orders differ — measured on `[[0,255,0,0,0,255,255,255,255],[0,0,0,255,0,255,255,255,0]]`, where `cv2` gives the single pixel at `(1, 3)` label 2 and the right-hand blob label 3, while first-raster-pixel order gives the blob 2 and the pixel 3. **The reason no entry is registered is that the difference is provably output-identical, not that it was judged unimportant:** §14 registers behavioural divergences, and there is no divergence to register here.

    **The proof's scope, quoted verbatim from the module header it lives in rather than paraphrased, because the whole risk in this decision is a scope that travels wider than its evidence:** *"the merged mask does not depend on the order at all"*, established for **`merge_mask_list`'s two loops only**. The proof is that a visit to label `L` writes `merged[bbox] |= indicator(L)`, which turns on only `L`'s own pixels; for every pixel outside `L` the candidate equals the origin, so those terms are bit-identical in the two XOR sums and cancel out of the strict comparison, leaving `L`'s own pixels as the only load-bearing term — and distinct labels are pixel-disjoint, so no other visit can have touched them. Both `L`'s decision and `L`'s effect are therefore functions of the loop-entry state alone.

    **What this decision does NOT license, spelled out because the blanket phrasing was the review finding.** `ConnectedComponents` is `pub`, with `pub labels` and `pub stats`, so a **different** future caller can be order-sensitive where these two loops are not — item 11(b)(iii)'s `valid_labels[1:]`, which drops whichever filtered entry sorts first, is exactly such a caller. The no-`DEVIATION` conclusion is scoped to `merge_mask_list`'s two loops and does not travel to A3b or to any other consumer; each new consumer of the label numbering re-establishes order-independence for itself or registers the divergence it has. `connected_components`'s doc comment carries that scoping so the blanket sentence cannot be inherited into a context where it is false.

    §16.37 item 8's independent measurement agrees and is recorded as it is worded there — *"reversing the entire non-background label numbering changes upstream's output by 0 px on E01P01, E01P02 and E01P03 — but that is a property of those pages"* — and it is evidence beside the proof, not the ground for it: a three-page measurement could not carry this conclusion, and a proof scoped to two loops does not need it to.

    **A3b discharged this item's obligation by re-establishing PARITY with upstream's label order, not by registering a divergence — the ruling is item 13 below.** This item named `valid_labels[1:]` as the anticipated order-sensitive consumer; that consumer has now landed and takes its label sequence in `cv2`'s own order through a local accessor, so it has no divergence to register. This sentence records which of the two exits this item offered was taken and nothing more; item 13 carries the ruling, its measurements and its grounds.

13. **DECIDED, and deliberately NOT a `DEVIATION`: `undetected_blocks` takes its label sequence in OpenCV's block-scan ORDER through a new local accessor, restoring parity; `connected_components`'s global numbering is unchanged (two independent joint architect + Senior Rust Engineer reviews of A3b, converging, 2026-08-07).** A3b's implementation surfaced a **real** divergence, not a theoretical one, exactly where item 12 predicted a consumer could be order-sensitive. `refine_undetected_mask`'s ported `valid_labels[1:]` (item 11(b)(iii)) selects a label by **position in the label sequence**, so unlike `merge_mask_list`'s two loops — where item 12's proof shows the per-label decisions commute — its output is a function of the numbering itself. Constructed 20×20 fixture, background **43** px with components of **300** and **57** px: `cv2` drops the 57-px component and invents `[5, 0, 20, 20]`, while first-raster-pixel numbering drops the 300-px one and invents `[0, 1, 3, 20]`. Two genuinely different outputs from the same input.

    (a) **What was rejected, and why, because the rejected options are the reason the chosen one is narrow.** **Porting OpenCV's BBDT algorithm** was rejected: nothing in this port needs OpenCV's pixel-to-label *assignment*, only the order the components arrive in, and a BBDT port is a large surface added to satisfy a small requirement. **Renumbering `connected_components` globally** was rejected on a checked, not argued, ground: both reviewers independently enumerated its readers — `annotate_merge.rs`'s `merge_mask_list` and `hole_fill_area_threshold`, plus A3's frozen test `connected_components_label_numbering_is_first_raster_pixel_order_not_opencvs`, whose **name** asserts the current rule — so a global renumber would turn a frozen test red for a reason its name does not admit. **The chosen fix is a LOCAL accessor**, `ConnectedComponents::labels_in_opencv_block_scan_order`, returning the non-background labels of the *same* labeling permuted into `cv2`'s order, with `labels` and `stats` untouched. Its sole caller is `undetected_blocks`, and its doc comment states in its own words that it is *"the label ORDER OpenCV assigns, NOT a port of the BBDT algorithm"*.

    (b) **The order rule, measured rather than read out of OpenCV's source.** A component's `cv2` label rank is the ascending order of `min over its pixels of (y / 2, x / 2)`, compared row-major — block row first, then block column. **cv2 5.0.0**, three independent runs, **0 mismatches in every one**: the architect measured **2614 + 299** fixtures, the Senior Rust Engineer **2091**, and the transcription run re-measured **2000** fixtures carrying **8318** non-background components. Both reviewers independently confirmed that all **block-based** variants agree (`CCL_DEFAULT`, `CCL_BBDT`, `CCL_GRANA`, `CCL_SPAGHETTI`) and only the **pixel-based** ones disagree (`CCL_SAUF`, `CCL_WU`) — re-measured here as 0 mismatches for the four block variants against **71 of 536** two-or-more-component fixtures for each pixel variant, with the pixel variants matching *our* first-raster-pixel order instead. That is both why the two orders differ at all and why the block rule is upstream's genuinely-intended answer: item 11(c) establishes that `cv2`'s effective connectivity at this call site is **8**, whose default family is block-based.

    (c) **Why NO `DEVIATION` entry is registered — and the contrast with `DEVIATION(23)` and `DEVIATION(29)` is the whole argument.** Those two are registered because **no single upstream answer exists** to be faithful to: item 23's ground is *"an introsort port would still be a function of a CPU feature set nothing pins, so there is no upstream ordering to be faithful to"*, and item 29's is the same sentence with the noun swapped for IPP and its dispatch level. Here upstream's answer is **single-valued, deterministic and measured**: one order, agreed by every block-based CCL variant, reproducible across three independent measurement runs, depending on no CPU feature set and no optional library. So there is no choice for us to declare — the ground that made 23 and 29 registrable is absent, and after this change the algorithm (item 11(c)'s effective 8) and the order both match upstream. §14 registers behavioural divergences; there is none here. **Note the identifier: the Otsu entry referenced above is item 29**, renumbered from 24 on this branch on 2026-08-07 to resolve the `lama-inpaint` cross-branch collision recorded in the gap note above §14's item 21; nothing in this item cites 24.

    (d) **The frozen A3b test was corrected under cookbook rule 8, with its NAME changed too.** `undetected_blocks_label_numbering_decides_which_component_is_dropped_when_the_background_is_small` asserted `[0, 1, 3, 20]` — this crate's pre-fix answer — and deliberately froze the divergence pending a ruling. It now asserts **`[5, 0, 20, 20]`**, hard-coded from the cv2 5.0.0 measurement, and is renamed `undetected_blocks_matches_upstreams_block_scan_label_order_when_the_background_is_small`. **The rename is part of the ruling, not cosmetic:** the old name claims the numbering *decides* the drop, which is a divergence claim, and the test now verifies parity — leaving the name would be this project's dominant defect class (a name claiming more, or other, than its assertion verifies). This is a joint-ruling correction that makes the test claim something **different** about the system, which is rule 8's legitimate exit, not a unilateral edit.

    (e) **Two further test obligations, both discharged.** First, an **anti-regression** test — `connected_components_global_numbering_is_still_first_raster_pixel_after_the_block_scan_accessor` — pins on the *same* 20×20 fixture, where the two orders are opposed, that `label_at(5, 0) == 1`, `label_at(0, 1) == 2`, `stats[1].area == 300` and `stats[2].area == 57`, i.e. the exact inverse of `cv2`'s `[43, 57, 300]`. It fails if the block-scan order ever leaks into the struct's own fields. Second, **`undetected_blocks_drops_the_lowest_numbered_survivor_when_the_background_is_filtered_out` is in the order-sensitive regime too** — its background area is **8**, so `valid[0]` is a real component there as well — and gets the same answer under both orders only by coincidence, because each blob's minimum 2×2 block happens to sort as its first raster pixel does. Both reviewers found this independently; a comment now says so at the test, so it cannot be read later as evidence the fix was unnecessary.

    (f) **The committed page is not exposed and cannot corroborate this.** Its residual background area is **1 987 219**, far above the 50-px filter, so `valid[0] == 0` under both orders and both drop the background. `refine_undetected_mask_on_the_recorded_page_invents_no_block` is therefore silent on this ruling, and its comment says so rather than being offered as support.

## 16.38 LaMa inpainting (v1.5): the measured ONNX signature, per-tile inference, the weights substitution, and the L1–L6 sequence (joint architect + Senior Rust Engineer plan pass, 2026-08-06)

Scope, quoted from §16.23 item 1 so it travels with this entry: **"v1.5 ships: … LaMa inpainting"**. This entry ratifies the *plan* for it — measurements, deviations, rulings, sequencing. It ships **no code**: every clause below describes work owed by tasks L1–L6 (item 16), and nothing here asserts that any of it exists yet.

**One exception to the sentence above, disclosed rather than silently reinterpreted — the same treatment item 17's preamble gives D3's "Ruled by this entry".** Item 19 was added to this entry *after* the preamble was written, and item 19(g)'s optional-aware refusal hint **"is added now, unwired"** — that code exists, is real and is tested (`pc_cli::models::models_download_optional_command`); it simply has no caller until L5. So read the preamble as "ships no code that is *wired* into a pipeline path", not "ships no code at all". Item 19's own clauses are the authority on what exists; the preamble's blanket wording is left as written because item 19(g) is the only clause in this entry that declares code landing *with the entry* rather than owed by a task. (Code for items 13, 16(a) and 19(a)–(f) does now exist, but it arrived through tasks L1–L3 as the preamble describes, not with the entry.)

**Numbering note, stated because it is a merge hazard rather than a style choice.** This entry is **§16.38**, not §16.37. §16.37 is already occupied on the sibling `mask-parity` branch, which forked from the same base commit `02b5613`; numbering this entry §16.37 would produce two `## 16.37` headings the moment the two branches merge, and `outline()` in `crates/pc-testkit/tests/spec_supersession.rs` would then resolve one number to two spans. §16.37 is therefore left unused **on this branch only** — it is not a reserved gap.

1. **MEASURED (L0 spike, 2026-08-06) — the pinned ONNX artifact and its graph signature.** The artifact is `lama-manga.onnx` from the Hugging Face repository `mayocream/koharu` at revision `15439cba09df388c51de6e47c6020bc31edab41f`, URL `https://huggingface.co/mayocream/koharu/resolve/15439cba09df388c51de6e47c6020bc31edab41f/lama-manga.onnx`, **207,482,644 bytes**, sha256 **`50a1abae0d73bd46d08eae36c8590cd59ad09029494c9698702b050ef00b0100`**. Size and digest were re-verified for this entry independently of the spike, by reading Hugging Face's `X-Linked-Size` and `X-Linked-ETag` response headers on that exact revision URL (both matched the spike's values character-for-character).

   (a) **`ir_version = 9`, `producer_name = "pytorch"`, `producer_version = "2.7.0"`, one opset import at version 20 with no domain (so the default `ai.onnx` domain).** Re-derived for this entry from the raw protobuf: a 256-byte range request on the head decodes as `08 09` (field 1, `ir_version = 9`), `12 07 "pytorch"`, `1a 05 "2.7.0"`; a 100,000-byte range request on the tail decodes `42 02 10 14` (field 8 `opset_import`, sub-field 2 `version = 20`).

   (b) **The two inputs are separate tensors, and their H/W axes are FIXED at 512.** Re-derived for this entry from the same tail range request, decoding `GraphProto` fields 11 and 12 byte by byte:

   * `image`: `elem_type = 1` (float32), dims `[dim_param "batch", 3, 512, 512]`
   * `mask`: `elem_type = 1` (float32), dims `[dim_param "batch", 1, 512, 512]`

   There is **no** concatenated 4-channel input. The `512` dims are literal `dim_value`s (`0a 03 08 80 04` — varint `0x80,0x04` = 512), not symbols.

   (c) **CORRECTION to the spike's own report, found while re-deriving (b): the OUTPUT's declared shape is not `[batch,3,512,512]`.** The decoded `output` value-info is `elem_type = 1`, dims `[dim_param "batch", 3, dim_param "batch", dim_param "Sigmoidoutput_dim_3"]` — three of four axes symbolic, and the **height axis carries the same symbol name as the batch axis**, which is a shape-inference artifact of the export and is false at run time for any batch size other than 512. **Binding consequence for L5: the implementation must validate the *runtime* output shape and must never derive geometry from the declared one.** This is the same obligation `validate_output_shapes` already discharges for the detector (§16.19 item 2(c) names it as a `Model`-only emitter). **NOTE, MEASURED BY TASK L5's REAL-MODEL TEST RUN (2026-08-07) — this narrows the clause above rather than contradicting it, and the narrowing is about WHERE the ambiguity lives, not about whether it exists.** The symbolic dims quoted above are what the **raw ONNX protobuf** declares; they are not what an `ort` caller sees. Loaded through `ort` 2.0.0-rc.12, ONNX Runtime runs its own shape inference at load time and concretises both spatial symbols, so `session.outputs()` reports the output as `("output", f32, [-1, 3, 512, 512])` — batch free, the height axis no longer sharing the batch symbol. Running real inference on the pinned artifact (sha256 `50a1abae…b0100`, 207,482,644 bytes, CPU execution provider) then produced an actual output shape of `[1, 3, 512, 512]`: fully concrete, with no ambiguity in practice. Both figures are **measured by running the model** in `crates/pc-inpaint/tests/l5_real_model.rs`, not inferred from the graph and not assumed. **What this does NOT do is relax the binding consequence.** The obligation stands exactly as written — `crates/pc-inpaint/src/onnx.rs` validates against the hard-coded `EXPECTED_OUTPUT_SHAPE` and never reads `session.outputs()[0]`'s shape — for three reasons stated so a later reader does not read this note as permission: an inference pass is a property of the runtime version rather than of the pinned artifact, so a different or future `ort` may report the protobuf's symbols unchanged; the clause's obligation was on the *runtime* shape from the start, which this note supplies a value for rather than removing; and deriving geometry from a shape the loader inferred is still deriving it from the declared one. So the sole correction to the clause above is that *"three of four axes symbolic"* describes the protobuf's declared symbols, not what `ort` reports or produces at run time.

   (d) **The fixed H/W axes were also confirmed empirically by the spike**, which built a real `ort::Session` from the artifact (this workspace's pinned `ort` 2.0.0-rc.12, features `ndarray, std, download-binaries, copy-dylibs, tls-rustls`) and fed 256×256, 640×512 and 1024×1024 inputs: all three failed identically with `Got invalid dimensions for input: image … Got: 256 Expected: 512` or its per-size equivalent. The `batch` axis is genuinely dynamic (`b = 2` and `b = 4` both ran) but **scales linearly** — 1.58 s / 3.03 s / 6.55 s for `b = 1 / 2 / 4` — so batching buys no throughput and is not a design lever.

   (e) **No FFT operator-support risk.** The FFC blocks' Fourier transform was exported as an explicit DFT over a cos/sin twiddle basis, not as a native ONNX FFT operator, so `ort` loads and runs the graph with zero unsupported-operator errors (the spike also cross-checked the same signature and the same successful load under Python `onnxruntime` 1.23.2 on the CPU EP). Re-derived for this entry, and **scoped exactly to what was measured**: a byte-string scan of the **first 20 MiB** of the artifact finds **0** occurrences of `DFT`, `STFT`, `Rfft`, `RFFT` or `fft`, against 864 of `Einsum`, 864 of `MatMul`, 828 of `Range`, 865 of `Cos` and 865 of `Sin`. Those are occurrence counts of a byte string over a *prefix*, not node counts over the whole graph — they corroborate the mechanism, they do not enumerate the graph.

   (f) **Measured CPU wall clock, one 512×512 tile, on the spike machine (20 logical cores): ~1.6 s with default all-core threading, ~4.7 s single-threaded. Session build, one-time at construction: ~6.3 s.** These are the only real numbers for this feature's ONNX path and they are machine-specific.

   (g) **The model does not composite; the caller must.** With an all-zero mask (nothing to fill) only **1 of 786,432** output pixels matched the input exactly — the model regenerates the whole tile. Mask convention confirmed: value 1 = fill, value 0 = keep. So blending the unmasked region back from the original is a **hard requirement**, not an optimisation.

2. **The retired numbers: what must NOT appear anywhere as a description of this feature.** An earlier Python/TorchScript benchmark (0.55–0.83 s per region) and upstream's own full-page timing describe a different artifact, a different call pattern and a different framework; a per-region number in particular cannot describe a per-tile call pattern at all. Neither may be quoted in spec, config comments, README or commit messages as if it characterised this feature. Item 1(f) is the whole citable set, and it is a measurement on one machine — a future reader wanting a number for their own machine must measure it, per cookbook rule 6's re-measure-never-quote discipline.

   **What item 1 does NOT establish, enumerated rather than summarised.** It does not establish output *quality* on any real page (the spike ran no manga page end to end and produced no reviewed image); it does not establish that per-tile output is visually comparable to upstream's single full-page call (item 5's open question, decision point D2 in item 17); it does not establish `lama-manga.onnx`'s numerical agreement with upstream's `.pt` checkpoint at any tolerance (item 6); and it does not establish behaviour on any execution provider other than CPU.

3. **Upstream's algorithm, transcribed with line cites, at the pinned commit `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`.** This is the port target; every clause below was read in the fetched source, not recalled. The whole of upstream's inpainting is `pcleaner/inpainting.py` (178 lines).

   (a) **Inputs.** `inpaint_page` reads exactly two cached JSON documents — the page JSON and `#mask_data.json` (`:57`, `:59`) — plus three images: `page_data.mask_path` (`:58`), `mask_data.mask_path` (`:60`) and `mask_data.original_path` (`:62`). `page_data.mask_path` is set to `path_gen.raw_mask` at `pcleaner/ctd_interface.py:173`, i.e. **`_raw_mask.png`** — our `PageData.raw_mask` (`crates/pc-core/src/page.rs:61`). `mask_data.mask_path` is the combined fill mask — our `MaskData.combined_mask` (`crates/pc-core/src/mask_data.rs:13`).

   (b) **`boxes_with_stats` matches `MaskRegionStats` field for field.** Upstream types it `Sequence[tuple[Box, float, bool, int | None]]` (`pcleaner/structures.py:643`) and destructures it as `box, deviation, failed, thickness` (`:78`, `:84`). Ours is `{ rect, std_deviation, failed, thickness: Option<u32> }` (`crates/pc-core/src/mask_data.rs:21-29`). **`pc-mask` is therefore not modified by this feature** — the new stage reads what is already persisted.

   (c) **Two eligibility sets, unioned.** Failed boxes: every row with `failed` true (`:76-80`). Poorly-fitted boxes: `not failed and deviation >= inpainting_min_std_dev and thickness is not None and thickness <= min_inpainting_radius` (`:82-90`). The `thickness is not None` clause carries upstream's own comment: *"For box masks, this is none. We don't need to inpaint those, they are always good."*

   (d) **Two fill-mask sources.** Failed boxes take their mask from the **raw** mask, cropped to the box and grown by `masker.min_mask_thickness` (`:93-97`). Poorly-fitted boxes take theirs from the **combined fill** mask, cropped to the box (`:99-101`).

   (e) **Growth arithmetic, exactly.** `growth = min(min_inpainting_radius + int(deviation * inpainting_radius_multiplier), max_inpainting_radius)` (`:106-108`); `growth_with_isolation = growth + inpainting_isolation_radius` (`:111`); the box is padded by `growth_with_isolation` clamped to the canvas (`:113`, via `structures.py:98-110`'s `max(x1-a,0) / min(x2+a,w)`), the cropped mask is pasted into a canvas-sized blank at the box's offset (`:114-116`), and *that* is grown by `growth` (`:117`). Note `int(...)` truncates toward zero, and `config.py:932`'s `fix()` already enforces `max_inpainting_radius = max(min_inpainting_radius, max_inpainting_radius)`.

   (f) **One page-global fill mask, one page-global isolation mask, and the paste OFFSET matters.** Each padded mask from (e) is canvas-sized but filled at coordinates relative to the padded box (`:114-116` pastes the crop at `box.x1 - box_padded.x1, box.y1 - box_padded.y1`), so `:124` pastes it back at `(box.x1, box.y1)` to restore absolute position. A port that reads (e) and (f) as operating in one absolute frame would place every fill region at the wrong coordinates — which item 5(a) makes load-bearing, since it tells the implementer to follow (e)–(f) exactly. **A direct absolute-frame port is not bit-equivalent, and that is a real subtlety rather than a note:** `grow_mask` pads with `mode="edge"` (`image_ops.py:812`), so growth near a canvas boundary replicates whatever sits at that boundary — and the boundary differs between upstream's padded-box-relative canvas and an absolute page canvas. Whichever frame L4 chooses, it must choose deliberately and pin the near-edge case. All padded masks are then pasted into a single `combined_mask` (`:122-124`) and resized to the original image size with NEAREST when the page was scaled (`:127-128`). The isolation mask is each padded mask grown *again* by `inpainting_isolation_radius` (`:137-144`).

   (g) **One model call per page.** `inpainted_image = model(original_image, combined_mask)` (`:132`), guarded by `if boxes_to_inpaint:` — with nothing eligible, upstream skips the model entirely and uses the original (`:131-134`). `InpaintingModel.__call__` crops the result back to the input size when the model returned something larger (`:37-39`), which is how upstream tolerates arbitrary page dimensions.

   (h) **Compositing.** `cleaned_image` is rebuilt from scratch inside `inpaint_page` — `original_image.convert("RGBA")` (`:149`), the fill mask pasted over it (`:151-153`), then the noise mask when denoising ran (`:155-157`) — so **`_clean_inpaint.png` is not derived from `_clean.png`**. The fill mask is Gaussian-faded by `inpainting_fade_radius` (`:159-161`, via `image_ops.py:820-830`), the faded mask is pasted through the isolation mask into a `final_mask` (`:166-167`), that becomes the inpainted image's alpha (`:168`), and the result is alpha-composited over the cleaned image before alpha is flattened to 255 (`:170-171`).

   (i) **Two cache artifacts, named verbatim.** `_inpainting.png` (`output_structures.py:409`) and `_clean_inpaint.png` (`:413`).

4. **DEVIATION(24) — per-tile inference, in place of upstream's single full-page call.** Upstream calls the model once per page on the full-resolution original with one page-global mask (item 3(g)); v1.5 calls it once per 512×512 tile. **This is forced, not chosen:** item 1(b) measured the ONNX artifact's H and W axes as fixed literal 512s, and item 1(d) measured three off-size inputs being rejected. Upstream's own artifact is a TorchScript `.pt` whose fully-convolutional generator accepts any size (the sibling weights repo declares `pad_multiple: 8`), which is why upstream never needed a tile loop. The register entry is owed at §14 as item 24, and the `// DEVIATION(24)` comment is owed at the tile loop in `crates/pc-inpaint/`. **The quality consequence is real and unmeasured** — a per-tile call sees a 512×512 context where upstream saw the page — and item 5 declares the policy while item 17's decision point D2 keeps the unmeasured half from being read as settled.

5. **The tiling and stride policy, DECLARED rather than left to implementation.** Cookbook rule 4's spirit generalised: a policy nobody wrote down gets inferred differently by every later reader.

   (a) The fill mask and the isolation mask are built **page-global and exactly as item 3(e)–(f)**, at the original image size. No deviation is taken here; the tiling sits strictly downstream of them.

   (b) **The tile cover is a function of the padded boxes, merged.** Take the padded boxes of item 3(e) and close them transitively under rectangle intersection, replacing each intersecting group by its bounding union until no two rectangles intersect. **The order-independence argument, corrected — the obvious one does not work.** "The transitive closure of an intersection relation is order-independent" would be enough only if merging did not change the relation, and it does: replacing a group by its bounding union can create intersections that no pair in the original relation had, so the closure is taken over a relation the procedure itself edits. The conclusion still holds, by a different route: the merge step is **monotone** (unioning rectangles only grows them, and a grown rectangle intersects a superset of what it intersected before) and the procedure runs to the point where no two rectangles intersect, so it computes the **least fixpoint** of a monotone operator on a finite set — which is unique, hence independent of visiting order. Stated at this length because the first draft asserted the wrong argument for a true conclusion, and that is the same precedent-as-derivation slip item 9(a) rejects in someone else's reasoning; the step-1a fresh reader caught it here. L4 must still pin it empirically: a test that permutes `#mask_data.json` region order and asserts an identical cover.

   (c) **One window per merged rectangle when it fits.** A merged rectangle whose width and height are both `<= 512` gets one 512×512 window centred on it and then translated minimally to lie inside the frame. A merged rectangle exceeding 512 on either axis is covered by a stride-512 lattice anchored at its own top-left corner, with the final row and column translated inward to stay in frame, and windows whose intersection with the fill mask is empty are dropped.

   (d) **Pages smaller than 512 on either axis** are edge-replicated up to 512 and the result cropped back. Edge replication rather than a constant fill is the in-family choice: upstream's own `grow_mask` pads with `mode="edge"` (`image_ops.py:812`).

   (e) **Every fill pixel is written exactly once.** Windows from different merged rectangles may overlap even though the rectangles do not, so ownership is assigned rather than left to write order: order the merged rectangles by `(y1, x1)` and, within one, its lattice windows row-major; a fill pixel belongs to the first window in that order whose region contains it, and write-back touches only owned pixels, masked by `faded_fill AND isolation` per item 3(h). L4 must pin "exactly once" as an assertion over the whole page, not as a comment.

   (f) **The seam risk is named, not hidden.** A fill region spanning a lattice boundary is generated from two different 512 contexts, and the discontinuity lands *inside* the filled area. Clause (c)'s centred-window rule removes this for every region that fits in 512×512, which item 3(e)'s radii make the common case, and the boundary is exact rather than approximate. Padding is applied on both sides, so the padded extent is `extent + 2 × (growth + inpainting_isolation_radius)` and `growth ∈ [min_inpainting_radius, max_inpainting_radius]` = `[7, 20]` at the defaults, giving a per-side pad in `[12, 25]`. Therefore a box of **≤ 462 px** on an axis always fits (512 − 2×25), a box of **≥ 489 px** never fits (512 − 2×12 = 488), and between 463 and 488 px it depends on that region's own `std_deviation` through `inpainting_radius_multiplier`. `Box.pad` also clamps to the canvas (item 3(e)), so a box against the frame edge pads less on that side and fits at a larger extent. It does not remove it for larger regions, and no measurement here says how bad it looks.

6. **DEVIATION(25) — the weights substitution, with the provenance chain verified end to end.** Upstream downloads `https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt`, sha256 `479d3afdcb7ed2fd944ed4ebcc39ca45b33491f0f2e43eb1000bd623cfb41823` (`pcleaner/model_downloader.py:21-22`), and loads it through `simple_lama_inpainting` by setting `LAMA_MODEL` (`inpainting.py:22-24`). v1.5 loads the ONNX artifact of item 1 instead, because this project has no TorchScript loader (`.pt`/torch loading is v2 per §16's out-of-scope list).

   (a) **The provenance chain, and exactly how far it was verified.** The sibling Hugging Face repository `mayocream/lama-manga` declares in its `config.json` a field `"source_checkpoint": "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt"` — **the same URL, character for character, as `model_downloader.py:21`** — alongside `"architecture": "FFCResNetGenerator"`, `"input_channels": 4`, `"output_channels": 3`, `"pad_multiple": 8`. Both ends of that comparison were fetched for this entry. **What this does NOT establish, stated rather than blurred into it:** it is a claim made by the republisher, not a numerical check; nothing here verifies that `mayocream/koharu`'s `lama-manga.onnx` was exported from `mayocream/lama-manga`'s weights, nor that either matches the `.pt` at any tolerance. Establishing that needs a side-by-side run of both artifacts on one page, which no task below performs.

   (b) **Licence, decided by the maintainer on a direct question.** `mayocream/koharu` is AGPL-3.0. The weights are **downloaded at run time and never vendored into this repository**, which is the basis on which the maintainer accepted it. Do not re-litigate; do not add the artifact to the tree.

   (c) The register entry is owed at §14 as item 25, and the `// DEVIATION(25)` comment at the new `pc_models::ModelSpec` constant.

7. **`inpainting_max_mask_radius` — RULED: ship the key, defaulted to upstream's value, deliberately unread by the algorithm; the WARN around it is DEVIATION(26).**

   (a) **The measurement, done by grepping the whole upstream tarball rather than one file.** `inpainting_max_mask_radius` occurs on exactly seven lines, in six places across three files (the seventh line is the second half of one `if`): `pcleaner/config.py:819` (declaration), `:861` (INI export), `:910` (INI import), `:920-921` (`fix()` clamp), `media/default.conf:265`, and `pcleaner/gui/image_file.py:345`. The last is a **cache-staleness key list**, not an algorithm input — it sits in `inpaint_settings`, the list of settings whose change invalidates a cached output. `pcleaner/inpainting.py`, which is the entirety of upstream's inpainting algorithm, references it **zero** times. The gate its config comment describes — *"The maximum radius of a mask to perform inpainting on. Masks larger than this will be left as they are"* — is implemented at `inpainting.py:89` as `thickness <= i_conf.min_inpainting_radius`, i.e. with a **different key**.

   (b) **Wiring it as upstream's evident intent is REFUSED, and the reason is decisive rather than cautious.** Which direction the fix moves eligibility depends on which of upstream's two disagreeing default sets you read. `config.py`'s dataclass defaults are `min_inpainting_radius = 7`, `inpainting_max_mask_radius = 6` (`:819-820`) — wiring would *narrow* eligibility. `media/default.conf` ships `inpainting_max_mask_radius = 6` (`:265`) and `min_inpainting_radius = 5` (`:269`) — wiring would *widen* it. A change to the feature's eligibility filter whose sign depends on which upstream file you opened is not "evident intent". (`media/default.conf` is read by **zero** Python files in the tarball — `grep -rn "default\.conf" --include=*.py` returns nothing — so `config.py`'s dataclass defaults are the runtime authority and the values in item 13 come from there. The other file is documentation, in the same `media/` directory §15 item 2 already discounted for the `demo_bubbles` fixtures.)

   (c) **Omitting the key is also refused**, because the cost lands on a later task rather than this one: §16.23 item 1 ships legacy INI import at v1.5, and §6 makes an unknown key a `WARN` that is preserved on round-trip — so omitting it means importing a perfectly valid upstream profile emits a spurious warning.

   (d) **Ruled: parse it, validate it, default it to `6`, and never read it in the eligibility filter, which uses `min_inpainting_radius` exactly as `inpainting.py:89` does.** That half is exact parity and is *not* a deviation. What L4 owes is a gate that keeps it inert: a test asserting that two runs differing only in `inpainting_max_mask_radius` select the identical eligible-region set. A comment saying "unused" enforces nothing.

   (e) **DEVIATION(26) is the WARN, not the key.** `pc-config` emits a one-time `WARN` when a loaded profile sets `inpainting_max_mask_radius` to a value other than the default, naming `min_inpainting_radius` as the key that actually gates and stating that the key is inert upstream too. Ground: §14 item 7's rule that *"an opt-in setting must not silently behave differently"* — a user who tunes a knob and gets no effect is exactly that case. Bounded deliberately: no WARN at the default value, so a user who never touched it sees nothing. The machinery already exists and is reused rather than built — `warn_colored_images_once` at `crates/pc-config/src/round_trip.rs:259`, with its message in `crates/pc-config/src/error.rs:45-67`. This is surface upstream does not have, which is why it is registered at §14 as item 26 rather than described as parity. **Cheapest thing to overturn in this entry:** if a reviewer prefers silence, delete clause (e) and keep (d); nothing else depends on it.

8. **RULING — the ONNX session is constructed LAZILY and its outcome LATCHED, and this agrees with the architect's lean but not with the architect's stated ground. DEVIATION(27).**

   (a) **Upstream's placement, measured.** `pcleaner/main.py:390-392` sets `skip_inpainting = True` when `inpainting_enabled` is false; `main.py:625-626` calls `md.ensure_inpainting_available(config)` and then constructs `ip.InpaintingModel(config)` **inside** the `if not skip_inpainting:` block at `:605` and **before** the per-page loop at `:642`. So upstream is *conditional on the flag* and *eager with respect to the page loop*.

   (b) **The config flag alone does not force laziness — it forces conditionality, and conflating the two would be an over-claim.** `inpainting_enabled` defaults to `False` (`config.py:817`), and an eager-but-flag-gated construction would already avoid the 207 MB download for every default run. Said plainly so nobody cites the default-off flag as the reason for laziness.

   (c) **The forcing ground is §16.19 item 1's, transferred, plus one that is stronger here.** §16.19 item 1 ratified lazy detector construction because *"a run whose `#raw.json` is cached never executes stage 1 and must not be blocked by a model it will never read"*; the same holds for a resumed run that never reaches `Step::Inpaint`. The stronger one is specific to this stage: a batch may have `inpainting_enabled = true` and **zero eligible regions on every page** — upstream's own `if boxes_to_inpaint:` guard at `inpainting.py:131` proves the empty case is expected, not pathological — and such a run must not pay a 207 MB download plus a ~6.3 s session build (item 1(f)) to inpaint nothing.

   (d) **The latch is the same mechanism, for the same two reasons, and is not a new design.** One `OnceLock<Result<…, String>>` behind a double-checked `Mutex<()>`, storing the rendered message rather than the error, per §16.19 item 2 and 2(d). §16.19 item 2(a)'s cost argument applies unchanged at 207 MB, and 2(b)'s determinism argument applies verbatim: without the latch, two workers can render different text for one cause and stdout becomes scheduling-dependent, violating §5.7.

   (e) **Why a new register item rather than widening DEVIATION(16).** Item 16's text is scoped to the detector — *"upstream constructs the detector before its per-image loop"*. Widening a ratified claim past the subject it names is a separate, argued step, per this file's own transcription rule; so the inpainter's placement deviation is registered at §14 as item 27 with its own text and its own `// DEVIATION(27)` site comment. This entry does **not** amend item 16, whose already-recorded stale sentence about a missing site comment is a separate pre-existing matter (§14 item 19 says so).

9. **RULING — a missing or uninitializable inpainting model is RUN-FATAL. This agrees with the architect's conclusion and rejects the architect's ground.**

   (a) **The ground offered — "the existing OCR provider is classified that way" — is not a valid one.** Cookbook rule **5** records *“‘We recovered poison elsewhere’ is never a reason”* (`docs/COOKBOOK.md:236` — inside rule 5, not rule 4; this spec carries its own copy of the same sentence in §16.19 item 10's grounds) and rule **4**'s own table states the signature test licences a provider to *declare* run-fatal, using "may". A precedent is not a derivation.

   (b) **The valid part of the signature test.** Session construction takes no image, so it is image-independent by construction and §16.19 item 5(b)'s criterion permits the provider to declare its failures run-fatal. That licences the classification; it does not by itself choose it.

   (c) **What actually decides it: per-image classification buys the user nothing here.** §5.1 states that on `Failed`, *"later stages for that image are **not** attempted"*. `Step::Inpaint` sits before `Step::Export` (item 11), so a per-image `Failed { step: Inpaint }` suppresses that page's export too. The outcome is the same zero exported files as run-fatal, at the cost of N error lines instead of one, N provisioning attempts instead of one, and exit 2 instead of 1. The UX argument the brief raised for per-image is therefore an argument for a *third* option, not for per-image.

   (d) **That third option — fail open, WARN and export the non-inpainted result — is refused, on §16.22 item 5(c)'s ruling about exactly this shape.** There, an opt-in accelerator whose provisioning fails was ruled a rendered fatal refusal specifically to countermand a silent fallback, because *"a run reporting success while secretly executing at ~150 s/page is worse than a crash because it is invisible"* A run that reports success while silently not inpainting is the same failure in a different costume, and §14 item 7's *"an opt-in setting must not silently behave differently"* points the same way.

   (e) **Upstream agrees, and this was checked rather than assumed.** `InpaintingModel.__init__` raises `FileNotFoundError` on a missing model (`inpainting.py:20-21`) at `main.py:626`, outside any per-page guard, and upstream's export block does not begin until after the inpainting block. So upstream too exports **nothing** when the inpainting model is absent. This ruling is upstream-faithful, not merely defensible.

   (f) **The residual cost, stated plainly rather than minimised.** With `inpainting_enabled = true` and the model absent or unloadable, **zero** pages are exported and the exit code is 1. The remedy is provisioning, not a runtime fallback — §16.19 item 3's *“Retry” means the user re-running `panel-ocr models download`* applies unchanged. The blast radius is confined to users who explicitly set the flag, since it defaults to false.

   (g) **The other half of the classification, declared because rule 4 says fatality is declared and not inferred.** A failure *inside* `Session::run` on a tile receives the image and is therefore **per-image**: `Failed { step: Inpaint }`, exit 2. One stage, two classifications, both declared here. No new `StageError` variant and no new `PipelineError` arm: construction failures render as `StageError::Model`, per-tile inference failures as `StageError::Inference`, matching §16.19 item 2(c)'s split.

   (h) **This entry records no disagreement between the two Opus subagents.** Both rulings above reach the architect's conclusion. Item 8 and item 9(a) record that the *grounds* differ, which is deliberate: cookbook rule 14a records that this pipeline's defects cluster where everyone agreed and nobody checked, so an agreement whose grounds were never compared is worth less than one whose were. Nothing here routes to `fable-adjudicator`.

10. **A transitive divergence neither plan flagged: the failed-box fill mask inherits DEVIATION(12).** Item 3(d) has upstream sampling failed boxes' masks from `_raw_mask.png`, which upstream writes as `mask_refined` — the output of `refine_mask` (`ctd_interface.py:182`). v1 ships `MaskRefineMode::Simple` instead (§14 item 12, §15 item 2), and cookbook rule 7 records the measured agreement of that artifact with upstream's as **IoU 0.258**. So for every `failed` region the filled area differs from upstream's *before any tiling happens*, and it would differ even if item 4's deviation did not exist. This gets no new register number — it is DEVIATION(12) propagating into a new consumer — but it must be stated at the fill-mask synthesis site, because a future parity investigation comparing our inpainted output against upstream's will otherwise attribute the whole difference to tiling.

11. **`Step::Inpaint` lands between `Denoise` and `Export`; three frozen-test value corrections are PRE-AUTHORISED here; NO new `Output` variants are added.**

    **SUPERSEDES: §2.8** — its preamble reads "Mirror upstream `output_structures.py` minus inpainting", and its `Step` literal has five variants.

    (a) `pub enum Step { Detect = 1, Preprocess, Mask, Denoise, Inpaint, Export }`, so `Step::Export as i32` becomes **6** and `Step::Export.prev()` becomes `Some(Step::Inpaint)`. Upstream has the same stage (`ost.Step.inpainter`). Considered and rejected: giving `Inpaint` a discriminant above `Export` to preserve `Export == 5` — the derived `Ord` on a fieldless enum compares discriminants, so that would order `Export < Inpaint` and break §4.4's resume comparison, which is the one thing the ordering exists for.

    (b) **The three frozen tests this turns red, each with its exact new value, so L6 does not discover them.** `crates/pc-core/tests/language_output_step.rs:42` (`assert_eq!(Step::Export as i32, 5)` → `6`); `:52` (`Step::Export.prev() == Some(Step::Denoise)` → `Some(Step::Inpaint)`); `crates/pc-export/tests/e3_run.rs:22` (same change). Adding `Step::Denoise < Step::Inpaint`, `Step::Inpaint < Step::Export` and `Step::Inpaint.prev() == Some(Step::Denoise)` alongside them is a frozen-test **addition**, cookbook rule 8 exit 1, and needs no authorisation.

    (c) **Why these are corrections and not amendments, by rule 8's own test.** Rule 8 asks whether the corrected assertion claims *less about the system*: `== 6` claims exactly as much as `== 5`, and `Some(Inpaint)` exactly as much as `Some(Denoise)`. Same variant, same subject, and once clause (a) lands the changed value is the only one the field admits. They are forced — but forced *by this entry*, so they are authorised **here**, following the precedent §16.23 item 5 set for PERF-1's authorised frozen-test edit and §16.36 item 7 cited for GPU-2.

    (d) **`pc_core::Output` gains nothing, and this upholds two ratified clauses rather than bending them.** §16.11 item 2 records *"Considered and rejected: adding export variants to `pc_core::Output`"* and §16.12 item 10 records *"`pc-core` is frozen and §2.8's variant list is closed."* Both stand. The two cache artifacts of item 3(i) get **pipeline-local suffix constants** in `pc-pipeline`, exactly as §16.12 item 10 did for `SPLITS_SUFFIX`: `"_inpainting.png"` and `"_clean_inpaint.png"`, upstream-verbatim. Consequences worth stating because they look like omissions: `Output::ALL.len() == 13` and `cache_suffixes_are_upstream_verbatim` stay green and unedited, and §2.8's non-surjectivity note is untouched.

    (e) **Longest-first suffix matching (§16.12 item 9) must be re-checked, not assumed.** `_clean_inpaint.png` and `_clean.png` both end a stem, the same collision `_clean_denoised.png` already has; L6 owes a test that the new suffixes disambiguate under the existing longest-match rule.

12. **The export seam.**

    **SUPERSEDES: §12.2** — its `ExportSources` struct lists five fields and none is an inpainting artifact.

    **SUPERSEDES: §12.3** — step 2's precedence reads "`cleaned: denoised > masked`; `mask: denoise_mask > final_mask`".

    **SUPERSEDES: §16.11 item 3** — it pins that precedence "exactly" as `cleaned = if denoising_enabled { denoised.or(masked) } else { masked }`.

    (a) `ExportSources` gains `inpainted: Option<ImageHandle>` (`_clean_inpaint.png`) and `inpainted_mask: Option<ImageHandle>` (`_inpainting.png`). Precedence becomes `cleaned = inpainted.or(<the existing expression>)` and `mask = inpainted_mask.or(<the existing expression>)`, matching upstream's stated `Precedence: masker < denoiser < inpainter` (`image_export.py:304`) and its ascending whitelists at `main.py:660-664` (cleaned) and `:665-669` (mask).

    **SUPERSEDED IN PART by §16.38 item 22 — read it before citing the `mask =` half of (a) above.** The `mask = inpainted_mask.or(…)` expression describes a plain replacement; the branch it cites is a three-layer alpha composite. The `cleaned =` half of (a), and the whitelist citations, stand unchanged.

    (b) **The stale-artifact rule of §12.3 step 2 extends with it, and must:** a populated `sources.inpainted` with `inpainting_enabled == false` (or `--skip-inpaint`) is a stale cached artifact and is ignored, exactly as a stale `denoised` is. Without this, disabling inpainting would silently resurrect a previous run's inpainted output.

    (c) **No new `Output` variant is needed for the category map either.** §16.11 item 2's map already routes `MaskedOutput`/`DenoisedOutput` to `Category::Cleaned` and `FinalMask`/`DenoiseMask` to `Category::Mask`, and a default run therefore already requests both categories; the inpainted sources win *within* a category rather than requesting one.

13. **The `[inpainter]` config surface — eight keys, transcribed from `config.py:817-824`.**

    **SUPERSEDES: §6** — its preamble reads "v1 sections and defaults — **exactly** upstream's values (`config.py`), minus `[inpainter]` (v1.5) and minus GUI/post-action keys (v2)".

    (a) `inpainting_enabled = false`, `inpainting_min_std_dev = 15.0`, `inpainting_max_mask_radius = 6`, `min_inpainting_radius = 7`, `max_inpainting_radius = 20`, `inpainting_radius_multiplier = 0.2`, `inpainting_isolation_radius = 5`, `inpainting_fade_radius = 4`. Every value is `config.py`'s dataclass default, which item 7(b) established is the runtime authority; the disagreeing values in `media/default.conf` are documentation and are not used.

    (b) **Validation at config load, mirroring `config.py:917-932`'s `fix()` as errors rather than silent clamps** (the project's existing convention: `pc-config` rejects, upstream repairs) — `inpainting_min_std_dev >= 0`, **`inpainting_radius_multiplier >= 0.0`**, and each of the five `Pixels` keys `>= 0`, plus `max_inpainting_radius >= min_inpainting_radius`, which is upstream's own `:932` invariant. The multiplier is easy to omit from that list because it is the only non-`Pixels`, non-threshold key in the block, and omitting it would be a real gap rather than a tidiness one: upstream clamps it at `config.py:926-927`, and unlike `inpainting_max_mask_radius` it is **live** — it feeds item 3(e)'s `growth = min(min_inpainting_radius + int(deviation * multiplier), max_inpainting_radius)`, so a negative value drives `growth` below `min_inpainting_radius` and potentially negative, into a growth kernel whose `size` is unsigned. Caught by the step-1a fresh reader.

    (c) **`inpainting_enabled = true` must load and validate successfully even in a build with no ONNX**, exactly as §16.36 item 6 ruled for `device = "cuda"`: config accepts, the stage refuses. Otherwise `--detector replay`/`mock` runs break for a model nothing in them ever reads.

14. **The achromatic-output question — RESOLVED by measurement, not deferred, and the premise it was raised on is corrected.** The concern was that upstream's inpaint path skips the greyscale treatment `pc_mask::cleaned_image`'s `all_achromatic` branch applies (`crates/pc-mask/src/combine.rs:107-121`), leaving us inconsistent if we mirror upstream.

    (a) **True of the cache artifact, false of the exported file.** Upstream's `_clean_inpaint.png` is indeed RGBA (`inpainting.py:149`, `:171`). But it is exported through `export_single_image(cache_path_gen.clean_inpaint, cleaned_out_path, original_image)` at `image_export.py:234-239` — with `original_image` supplied — so `save_optimized` captures `original.mode` at `:58` and converts at `:61-62`, which is **byte-for-byte the same treatment** the masked output gets at `:195-200` and the denoised output at `:211-216`. Upstream is not inconsistent at the export boundary.

    (b) **Our own code already does the equivalent, independently of `pc-mask`.** `pc-export` reads the mode off the original file (`crates/pc-export/src/lib.rs:267`, `formats::read_color_mode`) and converts (`:188`, `convert_to_mode`), per §12.3 step 3 and §16.11 item 5. So no new decision is required and no schema change to `#mask_data.json` is required — which matters, because the alternative reading would have needed `median_color` persisted into `MaskRegionStats` to give `pc-inpaint` an `all_achromatic` signal, and that is a shared-format change with enumerated readers (cookbook rule 14).

    (c) **What L6 owes instead is an obligation, not a decision:** `_clean_inpaint.png` must reach the user through the same `export_cleaned` seam and the same `original_mode` argument that `_clean.png` uses, pinned by a test on a greyscale input asserting the exported inpainted file's colour mode. `pc-inpaint` producing an RGB/RGBA cache artifact is upstream-faithful and creates no export-visible inconsistency.

15. **§16.23 item 5's sequence is REVISED: LaMa moves ahead of GPU-2 and ahead of the INI-import + Lab-NLM batch.**

    **SUPERSEDES: §16.23 item 5**

    (a) **The sequence as ratified, quoted verbatim so the change is visible:** *"`V15-0` is this ratification pass (§16.21, §16.22, §16.23) and **lands before any v1.5 code**; then **PERF-1** …; then **F1** …; then **GPU-1** …; then **GPU-2** (the `cuda` feature and its guards); then §16.21 item 6's investigation, which may run any time after PERF-1; then legacy INI import and Lab NLM batched; then LaMa; then PSD; then **DBNet lines**, which change default detector output and therefore end with an F1 re-record and full re-sign, budgeted once, up front; then **Annotation** last."*

    (b) **The amendment: LaMa may land immediately after GPU-1.** Everything before GPU-1 in that list is unchanged, and has landed on this branch's base — checked rather than assumed: §16.21, §16.22 and §16.23 exist as spec sections (V15-0); `docs/DETECTOR_ORACLE.md` and the tracked `tests/fixtures/recorded/detector/` set exist, including the committed upstream-oracle record (F1); and GPU-1 itself merged as §16.36 at `02b5613`, this branch's base commit. This entry reorders nothing before GPU-1. Annotation stays last. PSD and DBNet lines keep their relative order.

    (c) **One correction to how the jump was described to this pass.** §16.21 item 6 was *not* jumped: item 5's own text places it "any time after PERF-1", so it is not an ordering constraint at all. The real content of this amendment is LaMa moving ahead of **GPU-2** and ahead of the **INI-import + Lab-NLM batch**.

    (d) **Grantable, and the argument is a dependency check rather than a convenience.** LaMa depends on GPU-1, not GPU-2: §16.22 item 5(f) already states *"the device policy resolver is model-agnostic, so v1.5's LaMa inpainting reuses it instead of growing a second one"*, and GPU-1 shipped that resolver (§16.36 item 2). Nothing in item 5's LaMa placement is justified by GPU-2, INI import or Lab NLM, and no clause makes any of them a prerequisite.

    (e) **The load-bearing half: LaMa implies no fixture re-record.** `inpainting_enabled` defaults to `false` (`config.py:817`, item 13(a)), so no default-path artifact changes and no committed fixture moves. That is what distinguishes it from DBNet lines, for which item 5 budgets an F1 re-record and full re-sign explicitly. §16.23 item 2's four grounds for keeping Annotation last are untouched — none of them mentions inpainting — and item 4's re-ratification gate on §14 item 17 is likewise untouched, since LaMa synthesises no DBNet lines.

    (f) **What this amendment does NOT grant.** It does not reorder PSD, DBNet lines or Annotation; it does not defer GPU-2, INI import or Lab NLM, all of which remain in v1.5 scope per §16.23 item 1; and it does not license any further reordering by analogy — the next one needs its own clause.

16. **Task sequencing, with the `heavy`/`simple` classification `CLAUDE.md`'s batching rule requires.** L0 is the spike already executed; its findings are items 1–2 and it is not a Codex task.

    (a) **L1 — hoist the morphology into `pc_imageops::morph` (heavy, its own call).** Prerequisite, not cleanup: a third verbatim copy of `kernel`/`dilate` in `pc-inpaint` is not acceptable, and §16.10 item 2 already scheduled the hoist as the "v1.5 consolidation ticket". `pc_mask::grow` and `pc_denoise::morph` become re-exports. **Acceptance gate, stated as a set rather than as "tests pass":** the existing frozen kernel-matrix tests in `crates/pc-mask/tests/` and `crates/pc-denoise/tests/` must be green **and unedited** — `git diff --stat` over those two directories must be empty at the end of the task. A behaviour-preserving move that needed a test edit was not behaviour-preserving.

    **SUPERSEDES: §16.9 item 2** — it places `grow.rs` in `pc-mask` on the ground that the kernels are masking policy, and gives `pc-imageops` "exactly one new module, `mask.rs`". Its `pc-imageops` "no `pc-config` dependency" property is **not** touched by this entry and must survive the hoist.

    **SUPERSEDES: §16.10 item 2** — it describes `pc_denoise::morph::{kernel, dilate}` as "a verbatim restatement" of `pc_mask::grow`'s and pins a frozen test "so the two copies cannot drift silently"; after the hoist there is one implementation and no second copy.

    (b) **L2 + L3 (simple, one sequential Codex call).** L2 is the `[inpainter]` config surface of item 13. L3 is the `pc_models` registry entry of item 6 — **blocked on decision point D1**, so if D1 is unresolved when this call is made, L2 ships alone. Land L2's `default_profile.toml` edit **sequentially** with respect to any other branch's config-surface edit, for the reason §16.35 item 9 and §16.36 item 8 both give: `table_registry_matches_default_document` compares the whole registry against the whole document.

    (c) **L4 — `pc-inpaint`'s pure geometry and mask pipeline, with no ONNX at all (heavy, its own call).** Eligibility filter (item 3(c)), the two fill-mask sources (3(d)), the growth arithmetic (3(e)), the page-global fill and isolation masks (3(f)), the tile cover (item 5(b)–(e)) and the compositing (3(h)). All of it behind an `Inpainter` trait whose test double returns a fixed tile, so **every test in L4 runs in the default no-`onnx` tier** — the tier cookbook rule 6 records as the one that actually executes. This is where the bulk of the frozen tests live.

    (d) **L5 — the ONNX session behind the `onnx` feature (heavy, its own call).** `ort` session construction through `pc_core::device::resolve` (§16.36 item 2, no second policy surface), the lazy `OnceLock` latch of item 8, the run-fatal declaration of item 9, NCHW tensor construction for two separate inputs (item 1(b)), and **runtime** output-shape validation per item 1(c).

    (e) **L6 — pipeline and export wiring (heavy, its own call, depends on L1–L5).** `Step::Inpaint` and the three pre-authorised frozen-test corrections (item 11(b)), the pipeline-local cache suffixes (11(d)) and their longest-match check (11(e)), `ExportSources` and precedence (item 12), the stale-artifact rule (12(b)), the greyscale export obligation (item 14(c)), and a `--skip-inpaint` flag matching the existing `--skip-denoise` shape.

    (f) **Verify the flag actually reaches the stage before calling L6 done**, for the reason §16.36 item 6 gives for `device`: a test that never observes a non-default `inpainting_enabled` because the plumbing dropped it silently would pass while the feature does nothing.

17. **Named decision points. These were recorded as OPEN, not resolved by this entry as first written.** D1 has since been resolved and its ruling is transcribed at **item 19 below**, added to this entry rather than as a new section; D2 and D4 remain open, and D3's own clause (c) already reads "Ruled by this entry", which contradicts this preamble's blanket wording and is left as written rather than silently reinterpreted here.

    (a) **D1 — RESOLVED by item 19 below; read that item before citing this one, whose options list and recommendation it replaces.** The paragraph that follows is kept verbatim as the record of the question **as it stood at ratification**, and its enumeration of `pc_models::ALL`'s use sites is now **stale — do not cite it as current**. What L3 changed, re-derived 2026-08-06 with `grep -rn "pc_models::ALL" --include=*.rs .`: `models download` and `models verify` now iterate `pc_models::selected(include_optional)` instead, so `models path` is the **only** remaining `ALL` site in `crates/pc-cli/src/lib.rs` (**1** hit, `:308`, not the three below); and `crates/pc-models/tests/d1_resolve.rs` now holds **6** occurrences — five inside four tests (`:143`, `:202`, `:276`, `:282`, `:298`) plus one in the `names_at` helper (`:21`), which is itself called exactly twice, both calls inside the single test `the_required_optional_partition_is_pinned_by_name` (`:118-151`) — not "three … in two tests". The sentence below that "no frozen assertion … pins the model count" is likewise stale: `:143` now asserts `pc_models::ALL.len() == 4` as **the test's own anti-vacuity literal**, additional to what item 19(f) itself requires — 19(f) mandates the partition be asserted "as **sets of names** rather than counts — cardinality is not identity", and does not call for a bare total-count assertion; `:141-142`'s own comment states the literal's purpose ("a hard-coded total that cannot be computed from the registry"). So the count *is* pinned, and changing the registry's size is now a deliberate frozen-test question rather than a free edit. **D1 — `pc_models::ALL` and the 207 MB question. Needs the user or a joint-architect ruling, because it changes shipped CLI behaviour.** `pc_models::ALL` is iterated unconditionally at three sites, all in `crates/pc-cli/src/lib.rs` — `:263` (`models download`), `:279` (`models verify`), `:335` (`models path`) — and at three more in the frozen `crates/pc-models/tests/d1_resolve.rs` (`:128`, `:134`, `:150`). That is the complete enumeration; the count per file was taken with `grep -rn "pc_models::ALL" --include=*.rs . | awk -F: '{print $1}' | sort | uniq -c`, which reports `3` and `3`, so a truncated read would be visible as a wrong number (cookbook rule 14). Adding the inpainting spec to `ALL` grows every user's `models download` by 207,482,644 bytes for a feature that defaults off, and adds a row to `models verify` and `models path`. Options: **(i) accept**; **(ii) partition** — `ALL` keeps the required three, a new `OPTIONAL` slice holds the inpainting spec, `models download` gains an opt-in flag, and `verify`/`path` report an absent optional model as a distinct non-failing state (note `models verify` currently sets `all_ok = false` on any `Missing` and returns `EXIT_FATAL`, so this state must be added deliberately or verify starts failing for everyone); **(iii) a selector flag** on `models download` only. Recommendation: **(ii)**. Not decided here. Mechanically, no frozen assertion in `crates/pc-models/tests/d1_resolve.rs` pins the model count, so all three options are reachable without a frozen-test edit — its three `pc_models::ALL` use sites sit in **two** tests (lines 128 and 134 in `all_lists_every_model_the_cli_can_manage`, line 150 in `every_declared_digest_is_lowercase_hex_of_the_right_length`) and assert detector reachability, file-name uniqueness and digest form, never a cardinality.

    (b) **D2 — per-tile visual quality against upstream's full-page call.** Item 5 declares a policy that is deterministic and testable; it says nothing about how the result looks, and item 2 records that no quality measurement exists. Closing this needs a side-by-side run of upstream and of this port on a real page with a human verdict, which no task above performs, and which §15.10(a)'s independence rule would govern. **Until then, no clause anywhere may claim visual parity with upstream's inpainting.** Not a blocker for landing L1–L6.

    (c) **D3 — whether the `inpainting_max_mask_radius` WARN of item 7(e) is wanted.** Ruled by this entry, flagged because it is the one place here that adds surface upstream lacks, and because deleting it costs nothing else (item 7(e)).

    (d) **D4 — THREE independent collisions with the sibling branch, not one.** Both branches fork from `02b5613`, so each numbered its additions against the same base and each must be reconciled at the merge. **(i) The §-section number**, handled by this entry's numbering note: this is §16.38 because `mask-parity`'s committed `1b4e12f` holds §16.37. **(ii) The §14 DEVIATION register number.** §14 ended at items `…19, 21, 22` on the shared base (20 deliberately unused), so both branches' next free number looked like 23, and `mask-parity`'s `1b4e12f` **already claims `DEVIATION(23)`** for `get_topk_color`'s histogram tie order — an unrelated deviation with its own site comment. This entry therefore takes **24-27**, which was verified free by reading that branch's own §14 **as it stood at transcription time** (it then claimed 21, 22, 23 and nothing above). **THAT VERIFICATION WENT FALSE, AND THE COLLISION ON 24 IS NOW RESOLVED BY RULING — collision re-measured 2026-08-07, ruling the same day.** The sibling `mask-parity` branch minted its **own, unrelated `DEVIATION(24)`** — an Otsu tie rule, in its commit `70a617c` (`2026-08-07T00:23:14+07:00`), about five minutes after this branch's `534734e` — with its own site comments (`crates/pc-detect/src/annotate.rs:410` and `:449`, plus `crates/pc-detect/tests/a2_annotate_otsu.rs:50`, measured with `git grep -n "DEVIATION(24)" mask-parity -- '*.rs'`), so `git show mask-parity:docs/PIPELINE_SPEC_V1.md | grep -o "DEVIATION(2[0-9])" | sort -u` reported **21, 22, 23 and 24**. **Why there was a collision at all, stated as cause rather than as coincidence:** `mask-parity` minted its own 24 independently, without checking this branch's prior reservation of the 24-27 range — this was not two branches racing for the same next-free number in the same minute. **THE RULING (architect, 2026-08-07): `mask-parity` renumbers its Otsu tie rule to `DEVIATION(29)`, together with its site comments; this branch's `DEVIATION(24)` — per-tile inference, §14 item 24 — stays at 24 and required no change on this side.** The ground is timestamp precedence, measured on the commits rather than argued from which branch matters more: this branch's reservation of 24-27 landed in `ee7cb60` at `2026-08-06T16:47:04+07:00`, and `mask-parity`'s mint landed in `70a617c` at `2026-08-07T00:23:14+07:00`, **7 hours 36 minutes later** (the ruling states the gap as "7.5 hours"; the exact delta is 7h36m10s and is the same measurement, quoted here in both forms so neither reading looks like a correction of the other). The reservation was readable on the sibling branch's own base for most of a working day before the number was minted a second time. **Decided is not the same as landed, and the two halves are kept apart here deliberately** — that separation is the whole lesson of the "28 is confirmed free on both branches" clause below. The *decision* is settled and needs nothing further from this branch. The *sibling-side edit* is a separate dispatch on the other worktree and, measured from this worktree at `f517aa8`, is **not yet observable here**: `git show mask-parity:docs/PIPELINE_SPEC_V1.md | grep -o "DEVIATION(2[0-9])" | sort -u` still reports **21, 22, 23 and 24**, and `git grep -n "DEVIATION(24)" mask-parity -- '*.rs'` still returns Otsu sites — **four** of them at branch tip `dacd666`, not the three the mint commit carried, and the difference is stated rather than folded in because the renumber has one more site to move than the enumeration above implies. The three named above are unchanged (`crates/pc-detect/src/annotate.rs:410`, `:449`, `crates/pc-detect/tests/a2_annotate_otsu.rs:50`); the fourth is `crates/pc-detect/src/annotate_merge.rs:56`, a **prose cross-reference** rather than a site comment (*"the `DEVIATION(24)` precedent: match what upstream"*), added after the mint by that branch's A3 commit `dacd666` and therefore absent from the `70a617c` measurement above — both readings are correct at the commit each names. It nonetheless **names 24 and so must move to 29 with the rest** when the sibling-side renumber lands: it cites the deviation for its precedent value, so leaving it at 24 would point a reader of `mask-parity` at this branch's per-tile-inference entry. Re-measure before that renumber is dispatched, since a later sibling commit may add more mentions the same way this one did. So a reader who follows a `// DEVIATION(24)` comment on `mask-parity` to §14 item 24 on this branch still lands on an unrelated deviation until that dispatch lands and is fetched here — the hazard is scheduled for removal, not yet removed, and this clause does not claim the tree already shows otherwise. Stated per-number rather than summarised, because a summary is what went stale here: **24 is ruled to this branch**, and becomes held on exactly one side once the sibling's renumber lands; **29 is the number `mask-parity`'s Otsu tie rule moves to**, so any cross-branch reference to that deviation reads 29 and not 24; **25, 26 and 27** were absent from `mask-parity` at the measurement above and remain free; **no collision on 28**: the sibling branch is measured clean (`git show mask-parity:docs/PIPELINE_SPEC_V1.md | grep -c "DEVIATION(28)"` returns `0`), and **this branch itself takes 28** — §14 item 28, registered by item 21 of this entry — so 28 is claimed on exactly one side, which is what makes it non-colliding rather than free. (This clause read "28 is confirmed free on both branches" while item 21 of this very entry was claiming it, which was false the moment that item landed; item 21(e) below already states the same number in the correct tense, and this is worded to match it.) **This still must be reconciled before the two branches merge**, and the ruling above names the form: `mask-parity`'s §14 entry moves to 29 together with its site comments, and nothing on this side moves. This clause previously read *"Which side renumbers is deliberately NOT decided here"*, and that sentence is quoted rather than deleted so a reader arriving from an older copy can see exactly which open question closed — the decision closed; what remains open is only the sibling's edit, which is a fact about that worktree and not about this one. The earlier 23-collision was found by the step-1a fresh reader, not by this pass: the first transcription assigned 23-26 and would have merged into two different `DEVIATION(23)`s with two `// DEVIATION(23)` comments pointing at unrelated code. §16.35 item 9 and §16.36 item 8 already handled exactly this hazard by pre-assigning numbers across parallel branches, and naming only the section number was a narrower reading of their precedent than the precedent supports. **(iii) `RATIFIED_SUPERSESSIONS` and `EXPECTED_PARSED_CLAIMS`** in `crates/pc-testkit/tests/spec_supersession.rs`: each branch raised the count against the shared base, so the merge must contain both branches' rows and the sum, not either branch's number. Any future parallel branch owes the same three-part check before it picks any number.

18. **What this entry does NOT do, listed so absences are not read as oversights.**

    (a) **It does not supersede §16's out-of-scope inpainting bullet.** That bullet already assigns inpainting, `InpainterConfig`, `_inpainting.png` and `_clean_inpaint.png` to **v1.5**, and this is v1.5, so landing them discharges the bullet rather than contradicting it. The precedent is §16.36, which landed GPU-1 without touching §16's CUDA bullet; §16.33 marked §16's Windows bullet only because that bullet's *version* claim changed.

    (b) **It does not extend §5.3's fatal list.** Item 9's run-fatal outcome falls under §5.3's existing *"model file missing or hash mismatch (and download unavailable)"*, so no new fatal condition is declared and §16.19 item 1(c) needs no companion.

    (c) **It does not modify `pc-mask`.** Item 3(b) is the reason: everything the stage reads is already persisted in `#mask_data.json` and `#clean.json`.

    (d) **It does not touch `pc_core::Output`, `Output::ALL`, or §2.8's non-surjectivity note** (item 11(d)), and it does not change any on-disk schema — `Step`'s serde form is `rename_all = "snake_case"`, so the wire value is the variant name and the renumbered discriminant is not serialised.

    (e) **It ratifies no benchmark number beyond item 1(f)**, and item 2 names the numbers that must not reappear.

19. **D1 — RESOLVED: `ALL` stays complete, optionality is a mandatory field, and `--include-optional` is a visible flag on BOTH `models download` and `models verify`** (independent joint architect + Senior Rust Engineer pass, mechanism decided by Fable tie-break, 2026-08-06). This item resolves the decision item 17(a) recorded as OPEN, and it is the one clause of §16.38 that changes shipped CLI surface.

    **SUPERSEDES: §13.1** — its `models` usage line reads `panel-ocr models   download|verify|path`, with no flag on any of the three.

    (a) **`pc_models::ALL` stays the COMPLETE registry.** Item 17(a)'s own option (ii) sketched shrinking `ALL` to the required three and adding a separate `OPTIONAL` slice; that sketch is **rejected**. `ALL` is what `models path` iterates and what any future consumer will reach for first, and a registry that omits a model the application can manage is a trap for exactly the reader who trusts its name. Callers scope their own iteration instead.

    (b) **`models download` without the flag fetches only `Required` models; with `--include-optional`, `Required` + `Optional`.** `models verify` scopes its reported rows the same way. `models path` is **unchanged** — it already lists every entry with `EXIT_OK` unconditionally, which is the behaviour this item wants there. The flag is **visible**, unlike `--cache-dir`'s `hide = true` (§16.12 item 21): that one is a testing affordance, this one is a decision a user makes.

    **RULING (Orchestrator, 2026-08-06), recorded as a named tradeoff rather than left to be re-litigated: `models verify` scoping its rows the same way `download` does is DELIBERATE, and it stands as implemented.** The step-1a fresh reader measured the cost and it is real, so it is stated and not softened: **without** `--include-optional` an `Optional` model's row does not appear at all, so a *corrupted-but-present* optional artifact — digest or size mismatch — is **invisible** to a plain `models verify`, which exits `0`; only `models verify --include-optional` observes it (or its `NOT INSTALLED (optional)` state). The alternative reading — `verify` always reporting all four rows while only `download` scopes — was considered and **rejected**, on two grounds. **(1)** Plain `models verify`'s default output stays byte-identical to pre-L3 behaviour (see the carve-out below), which was an explicit design goal of this item: `models verify` is a CI preflight (clause (d)'s own reason for locking the `Required`-absent failure), and existing users' preflight checks must not regress or gain rows. **(2)** The residual risk is low-severity **because it cannot reach output**: L5's inpainter runtime path must independently verify the LaMa artifact's integrity before use, regardless of what `models verify` last reported, so a corrupt optional model yields a run-fatal refusal at the stage rather than corrupted inpainting — what goes undetected is a *stale preflight signal for a feature the user has not enabled*, not silently wrong pixels. A future reader who wants the other behaviour is changing a decision, not fixing an oversight.

    **The one carve-out on "byte-identical", stated because the claim above would otherwise be an over-claim.** One reachable path did change: previously an `fs::metadata` failure on a *present* cached file aborted the whole subcommand through `anyhow`'s `with_context` (`crates/pc-cli/src/lib.rs`, pre-L3), leaving every model after it unreported; it now renders an `ERROR` row for that model and continues to the rest, with the exit code still `1` because `VerifyStatus::Error` is a failure. That is a **minor and arguably improved** behaviour change, not a regression — no row disappears and no failure becomes a success — and it is the only one: `resolve` with a `None` override cannot return `Err`, so `fs::metadata` was the only abort the old loop could actually take.

    (c) **A skipped optional model is NAMED, never silently skipped.** `models download` prints one line per omitted optional model, naming the model and the flag that would fetch it. Ground: a user who has enabled `inpainting_enabled` and run `models download` must not be left inferring why the stage still refuses.

    (d) **`models verify`'s four outcomes, stated as a table because getting one of them wrong is how this change breaks.** A `Required` model that is absent is **still a failure** — it still clears `all_ok` and still exits `EXIT_FATAL` — and that is the regression this item's frozen tests lock, because `models verify` is used as a CI preflight and a weakened version of it would start passing with no detector weights present. An `Optional` model that is absent is a **new, non-failing status**, and its label must **not** contain the literal string `MISSING`, so a script grepping that word cannot confuse the two. An `Optional` model that is **present and fails** its digest or size check **is still a failure**, regardless of optionality: optionality licenses **absence only, never corruption**. A `Required` model that verifies clean is unchanged.

    (e) **Mechanism, decided by the Fable tie-break and binding: a mandatory two-variant field on `ModelSpec`** — `pub enum Requirement { Required, Optional }`, with **no `Default` impl**, so every construction site must state the answer and a new model cannot silently inherit "required". A derived `OPTIONAL`/`REQUIRED` slice and an `is_optional()` predicate were both **explicitly rejected**: either lets a new registry entry be added without anyone deciding, which is the failure mode the field exists to prevent. The ruling **pre-authorises** the one-line consequential edit to the frozen `FAKE_SPEC` helper at `crates/pc-models/tests/common/mod.rs` (marked `Required`, matching its doc comment describing it as a stand-in for `COMIC_TEXT_DETECTOR`): it is compile-forced, it changes no assertion, and it is therefore not a unilateral test edit under cookbook rule 8.

    (f) **A frozen test pinning the partition BY NAME is a condition of this ruling, not an optional extra** (Fable's "graft"). `crates/pc-models/tests/d1_resolve.rs` must assert that `COMIC_TEXT_DETECTOR`, `MANGA_OCR_ENCODER` and `MANGA_OCR_DECODER` are `Required` and the LaMa entry is `Optional`, as **sets of names** rather than counts — cardinality is not identity, and a swap between the two partitions keeps every count while silently changing what a default `models download` fetches.

    (g) **An optional-aware refusal hint is added now, unwired.** The existing hint hard-codes `panel-ocr models download` (`pc-models/src/lib.rs`'s `ModelError::Unavailable`, and `pc_cli::models::models_download_command`); a sibling helper naming `--include-optional` lands with this item so L5 can call it when the inpainter refuses on a missing LaMa artifact. `ModelError::Unavailable`'s own text is **not** changed: a required model's refusal must never suggest a flag that fetches 207 MB the user did not ask for.

    (h) **What this item does NOT do.** It does not add a size/`ProgressSink` policy of any kind for the larger artifact; it does not make `inpainting_enabled = true` imply the flag (config load stays independent of provisioning, per item 13(c)); and it does not resolve D2 or D4.

20. **L4 also hoists the separable Gaussian blur into `pc_imageops::gaussian` (part of L4's own heavy call, not a separate task)** — independent joint architect + Senior Rust Engineer review, Fable tie-break on the sibling deviation question, 2026-08-07. This item is to `gaussian` exactly what item 16(a) is to `kernel`/`dilate`, and is written in the same form deliberately.

    (a) **The task.** `crates/pc-denoise/src/gaussian.rs` moves to `crates/pc-imageops/src/gaussian.rs`, **body unchanged**, and `pc_denoise::gaussian` becomes a re-export of it. Nothing else moves.

    **Ground.** Prerequisite, not cleanup, on the same reasoning item 16(a) gives for the morphology: `pc-inpaint` needs this blur for `inpainting_fade_radius` (item 3(h), porting `image_ops.py:820-830`'s `fade_mask_edges`), §1 rule 2 forbids the stage-to-stage dependency that would let it reach `pc-denoise`, and a second verbatim copy of the taps-and-two-passes arithmetic is not acceptable. §16.10 item 1 already scheduled this move in its own closing clause, quoted verbatim so its scope travels with it: *"Both stay `pub` (`pc_denoise::nlm`, `pc_denoise::gaussian`) and free of any `pc-config` dependency, so §11.1's "independently benchmarkable" property is preserved and a v1.5 hoist into `pc-imageops` is a mechanical move plus a re-export."* The `pc-imageops` "no `pc-config` dependency" property that item 16(a) protects is likewise **not** touched here and must survive this hoist.

    **Acceptance gate, stated as a set rather than as "tests pass":** the frozen `crates/pc-denoise/tests/n2_gaussian.rs` must be green **and unedited** — `git diff --stat` over `crates/pc-denoise/tests` must be empty at the end of the task. **Measured, not asserted:** that command produced empty output on the L4 working tree at the time this entry was written, with `git status --porcelain` over the same directory likewise empty. A behaviour-preserving move that needed a test edit was not behaviour-preserving.

    **SUPERSEDES: §16.10 item 1** — it places `nlm` **and** `gaussian` in `pc-denoise` on the ground that `pc-imageops` was frozen at the end of Stage 3 and neither module had a second consumer in v1.

    (b) **The supersession is SCOPED TO `gaussian` ALONE, and the scope is the load-bearing half of this item.** `nlm` stays in `pc-denoise`: it still has exactly one consumer, so the whole ground §16.10 item 1 gives still holds for it, and **this entry is not authority for hoisting it**. Both reviewers raised the same objection independently — an unscoped in-part marker would let a future reader hoist `nlm` on this entry's authority, which is precisely the widening §16.38 item 8(e) declined for `DEVIATION(16)` and which this file's own transcription rule forbids as a separate, argued step. The back-pointer at §16.10 item 1 therefore states the `gaussian`-only scope at the target site too, where a reader who never reaches this entry will land.

    (c) **Two further targets, found by the architect and independently confirmed by the Senior Rust Engineer, handled ASYMMETRICALLY because the two cases are not the same shape.**

    **SUPERSEDES: §13** — its row 14 attributes "Gaussian blur" to `pc-denoise`, which stops being true of the module's home at v1.5. The annotation is inline in the row, following row 30's precedent for a stale cell (§16.19's convention: original wording kept, note appended), and it names the one clause affected rather than the row: the noise-mask build, `run()`, the 1-bit shortcut and the analytics are all still `pc-denoise`, and row 15's `nlm` is untouched. As with §16.30's `13` row, this is a bare-section target under ratified §16.26 item 3(c) leniency — §13's rows live in a markdown table and carry no `^N. ` item marker — so the gate accepts a back-pointer anywhere in §13 and the row-level precision of the annotation is a convention `RATIFIED_SUPERSESSIONS` cannot enforce.

    **§11.5's N2 row gets a NOTE and deliberately NO marker.** That row reads "`pc-imageops`: separable Gaussian blur", which §16.10 item 1 contradicted for v1 and which this hoist makes accurate again. A restoration is not a supersession — there is nothing left stale at that site to warn a reader about — so a marker there would claim the wrong relation, and the note says which of the two it is. **Disclosed rather than discovered later:** because there is no marker, nothing in `crates/pc-testkit/tests/spec_supersession.rs` gates that note's existence, and deleting it would leave the suite green. That is the same class of unenforced-placement gap the `("16.38", "6")` row in that file already documents, accepted for the same reason: the alternative is a marker asserting a relation that does not hold. §11.5's N1 row is left alone and stays inaccurate for `nlm`, exactly as §16.10 item 1 left it.

    (d) **`DEVIATION(6)` now reaches a SECOND consumer, with NO new register number.** Upstream's `fade_mask_edges` calls `ImageFilter.GaussianBlur(radius)` (`image_ops.py:820-830`) — the *same* upstream call §14 item 6 already registers for the noise fade — so `pc-inpaint`'s fade inherits the existing divergence (PIL's three-pass box approximation versus this project's true separable Gaussian, `sigma = radius`, truncated at `3*sigma`, pinned by §16.10 item 12). This is item 10's propagation shape exactly: one upstream call, one existing register entry, a new caller. Contrast §14 item 28, registered by item 21 below, which does **not** take this treatment — a different upstream call with a mechanism §14 item 5 never describes needs its own number, and that asymmetry is the point of stating this one. The note is owed at the call site and **exists**, at `crates/pc-inpaint/src/fade.rs` on `fade_fill_mask`, together with the fact that the "a few levels on a soft edge" figure — which is **§11.3 step 5's** wording, not §14 item 6's; item 6 is one sentence about the box-approximation-versus-true-Gaussian choice and contains no such phrase — was stated at `noise_fade_radius = 1` and that **nothing has measured the difference at the inpainting default `inpainting_fade_radius = 4`**, where the kernel is wider. The un-measured half is stated, not inherited silently.

    (e) **What this item does NOT do.** It does not hoist `nlm` (clause (b)); it does not touch the composite duplication (`blend_channel`, `resize_nearest_rgba`, `alpha_composite_over`, `composite_rgb`) that §16.10 item 3 pins across `pc-mask`/`pc-denoise` and that `pc-inpaint` now also needs — that consolidation is tracked as its own task and this entry is not authority for it; and it does not change the blur's arithmetic, its border rule, or `pc_denoise::gaussian`'s public path.

21. **RULING (Fable tie-break, 2026-08-07) — the combined fill mask's alpha binarisation gets its OWN register number, `DEVIATION(28)`. It is NOT `DEVIATION(5)` reaching a new consumer.** Item 3(d) declares only that the fill mask is "cropped to the box" and is silent on how the RGBA combined mask becomes binary; `crates/pc-inpaint/src/fill.rs`'s `combined_fill_binary` escalated the question rather than settling it, which is what this item answers. The architect read it as item 5 propagating (item 10's shape); the Senior Rust Engineer read it as a distinct divergence; Fable ruled for the second reading.

    (a) **What v1.5 does.** `combined_fill_binary` takes coverage from **alpha** — `pixel[3] > 0` — on the RGBA `_combined_mask.png`. Upstream binarises the same image with `mask_image.convert("1")` at `pcleaner/inpainting.py:63`. The input really is RGBA and not greyscale: upstream builds it through `combine_best_masks` / `convert_mask_to_rgba`, so the fill colour and its opacity are both carried. PIL's `convert("1")` routes through `L`, i.e. the luma of the RGB channels, and **alpha is discarded** — so a black opaque fill `(0,0,0,255)` reads as *uncovered*, and on the `black_bubble` demo page — a demo fixture, not a profile — every poorly-fitted region silently gets nothing inpainted.

    (b) **Why this is a DIFFERENT deviation and not item 5's, stated as the two facts that separate them.** **(i) A different upstream call site.** §14 item 5 is registered against `denoiser.py:62`'s path into `grow_mask`, whose discard happens at `mask.convert("L")`; this is `inpainting.py:63`'s `mask_image.convert("1")`, a different call in a different file feeding a different stage. Item 10's propagation shape applies when one upstream call acquires a second caller — the shape §16.38 item 20(d) uses for `DEVIATION(6)`, where the call is `ImageFilter.GaussianBlur` in both places. That is not this case. **(ii) A mechanism item 5 never describes: dithering.** `convert("1")` does not threshold; PIL's documented default for mode `"1"` is Floyd-Steinberg error diffusion, so on the *intermediate* luma values a real anti-aliased or coloured fill produces, the output is a stippled checkerboard rather than a clean bilevel cut. §14 item 5's text describes luma-versus-alpha and nothing else, so an item-5-propagation framing would leave the dithering unrecorded at the only site where it exists. **This also forecloses one of the two options that were on the table:** "replicate upstream verbatim, bug included" is not a one-line change to a threshold — it would require porting Floyd-Steinberg error diffusion — and it is therefore rejected on cost as well as on correctness.

    **Provenance, stated rather than implied.** Clauses (a) and (b)(ii) transcribe findings the architect and the Senior Rust Engineer each reached independently against upstream; PIL's Floyd-Steinberg default for mode `"1"` is that library's documented behaviour. This transcription did **not** re-run upstream — the tie-break oracle was exercised by the reviewers, not by the transcriber — and a pinned-checkout re-measurement of the stippling on a real `black_bubble` page is owed before anyone cites a *magnitude* here. What is claimed is the mechanism's existence, not a measured pixel difference.

    (c) **Why a new register item rather than widening §14 item 5, following §16.38 item 8(e)'s precedent exactly.** Item 5's text is scoped to the denoiser's cutout — it names `denoiser.py:62`, `generate_noise_mask` and `grow_mask` — and widening a ratified claim past the subject it names is a separate, argued step, per this file's own transcription rule. So the inpainting binarisation is registered at §14 as item 28, with its own text and its own `// DEVIATION(28)` site comment. This entry does **not** rewrite item 5, and item 5's own scope is unchanged by it. What the two share is the **ground**, quoted verbatim from §14 item 5 so its wording travels with it: *"A mask's 'is this pixel covered' signal must not depend on the brightness of its fill color"* — that general principle is the warrant for `alpha > 0` here, and reusing a principle is not the same act as widening the claim built on it.

    (d) **This stage binarises TWO masks TWO different ways, DELIBERATELY. Do not unify them.** `crates/pc-inpaint/src/lib.rs` uses `combined_fill_binary` (`alpha > 0`) for the RGBA combined mask and `BinaryMask::from_gray_threshold(raw_mask, PIL_BINARY_THRESHOLD)` (strict `> 127`, §16.9 item 4) for the greyscale `_raw_mask.png`. The asymmetry is upstream-faithful and follows from the inputs: the raw mask has no alpha channel to discard and is already strictly bilevel, so `convert("1")`'s dithering is a no-op on it and a luma threshold is exactly right there. A future reader who "fixes the inconsistency" by routing the combined mask through `from_gray_threshold` reintroduces `DEVIATION(28)`'s bug, and one who routes the raw mask through an alpha test has no channel to read. Stated here because the two lines sit four apart in `run()` and look like an oversight.

    (e) **Number collision check, re-run on this branch at transcription time rather than inherited.** **Both greps below are stated as of BEFORE this entry was added, because this entry is itself what changes their output** — the same tensing problem §14's marker-inventory parenthetical already handles for its predecessor, and re-asserting them in the present tense inside the document that falsifies them is exactly the staleness this file keeps producing. Measured **before** this entry landed: `grep -rn "DEVIATION(2[5-9])" docs/PIPELINE_SPEC_V1.md` returned hits for 25, 26 and 27 only, and **it now also returns 28 — this entry's own**; `DEVIATION(3[0-9])` returned none and still returns none; and `grep -rhno "DEVIATION([0-9]*)" --include=*.rs crates/*/src/ xtask/src/` topped out at 27, and **now reaches 28**, that being `combined_fill_binary`'s site comment in `crates/pc-inpaint/src/fill.rs`, which this entry requires. So 28 was free **on this branch** when it was claimed, and is now taken by this entry and by nothing else. Item 17(d)(ii)'s three-part cross-branch check still applies and is **not** discharged by this clause: the sibling `mask-parity` branch numbers against the same base, and reconciling both branches' §14 items — and both branches' raises of `EXPECTED_PARSED_CLAIMS` — belongs to whoever merges them. Nothing in this repo gates the register-number half. On 28 specifically, `mask-parity` is measured clean (item 17(d)(ii) records the command and its `0`); on **24** that branch collided and the collision is now resolved in this branch's favour — that branch renumbers to 29 — and item 17(d)(ii) is the entry to read for both the ground and for what part of it had not yet landed on the sibling when it was written.

    (f) **Site-comment obligations, enumerated because a summary has already lost one of them twice in this file.** `crates/pc-inpaint/src/fill.rs`'s `combined_fill_binary` doc comment currently frames the choice as `DEVIATION(5)` propagating and as unratified; that framing is **wrong under this ruling** and is replaced by `DEVIATION(28)` with this item as its ground — done with this entry. `crates/pc-inpaint/src/lib.rs` carries clause (d)'s two-binarisations note at the pair of call sites — done with this entry. **Also DONE with this entry, and no longer owed:** `crates/pc-inpaint/tests/l4_eligibility.rs`'s `a_black_but_opaque_combined_mask_pixel_counts_as_covered_and_a_transparent_white_one_does_not` previously cited `DEVIATION(5)` in its doc comment and in an assertion message, and its doc comment repeated the "escalated, not settled" framing. That file is a frozen test, so the change is named rather than made quietly: it is **authorised by this entry** in the same way §14 item 18 authorised the corresponding fix at `crates/pc-detect/tests/d5_yolo.rs:297`, and it touches **no assertion and no expected value** — the doc comment at `:176-188` now calls the reading **ratified** and cites `DEVIATION(28)` at §14 item 28, and the assertion message at `:202` names `DEVIATION(28)`. So this clause's three obligations are all discharged by this same commit, and nothing in `crates/pc-inpaint/` still **cites** `DEVIATION(5)` as this binarisation's ground. Stated as "cites as its ground" rather than "names", because `DEVIATION(5)` does still appear twice there — `crates/pc-inpaint/src/fill.rs:79` and `crates/pc-inpaint/tests/l4_eligibility.rs:178`, both saying it is **not** the applicable entry — and a bare "names" reading is falsified by a grep that finds exactly those two negations.

22. **CORRECTED — item 12(a)'s `mask =` half is wrong: upstream's inpainted-mask export is a THREE-LAYER alpha composite, not a replacement** (independent joint architect + Senior Rust Engineer L6 planning pass, 2026-08-07). Both planning agents read the pinned upstream source separately and reached this finding independently, so per `CLAUDE.md`'s pipeline this is a **consensus correction ratified jointly**, not a `fable-adjudicator` escalation — the adjudicator is convened for disagreement, and there is none here.

    **SUPERSEDES: §16.38 item 12(a)** — its `mask = inpainted_mask.or(<the existing expression>)` describes the exported mask being replaced by `_inpainting.png`, which is not what the branch it cites does. Its `cleaned =` half and its two whitelist citations are untouched by this item.

    (a) **The evidence, quoted verbatim** — `pcleaner/image_export.py` at the pinned commit `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`, **lines 241-263**, fetched from `raw.githubusercontent.com` for this transcription rather than recalled. **One departure from strict verbatim, named so the claim is exact:** every line is dedented by the 4 spaces of the enclosing function body, so relative indentation — which is what carries the `if denoising_enabled:` scope — is preserved and absolute column numbers are not. Nothing else is altered, including the comment on `:242`.

    ```python
    if ost.Output.inpainted_mask in outputs:
        # Special case: Here we need to take the final mask, scale it up, and then paste the denoising
        final_mask = Image.open(cache_path_gen.combined_mask)
        final_mask = final_mask.resize(original_size, Image.NEAREST)
        final_mask = final_mask.convert("RGBA")

        add_image_layer(final_mask, masker_mask_name)

        if denoising_enabled:
            denoised_mask = Image.open(cache_path_gen.noise_mask)
            denoised_mask = denoised_mask.convert("RGBA")
            final_mask.alpha_composite(denoised_mask)

            add_image_layer(denoised_mask, denoiser_mask_name)

        inpainted_mask = Image.open(cache_path_gen.inpainting)
        inpainted_mask = inpainted_mask.convert("RGBA")

        add_image_layer(inpainted_mask, inpainter_mask_name)

        final_mask.alpha_composite(inpainted_mask)

        export_single_image(final_mask, masked_out_path)
    ```

    (b) **What the quote establishes, enumerated rather than summarised** — item 12(a) is itself what a paraphrase of this branch produced, so a second paraphrase is not the way to correct it. **(i)** Three layers, in one fixed order: `_combined_mask.png` is the base (`:243`), the noise mask is composited over it (`:252`), and the inpainting output is composited **last**, i.e. on top (`:261`). **(ii)** The noise layer is conditional on **`denoising_enabled`** (`:249`), not on the mere presence of `_noise_mask.png`. **(iii)** The inpainting layer is unconditional *within* this branch — there is no `if` around `:256-261`. **(iv)** The base is resized with `Image.NEAREST` (`:244`); neither the noise layer nor the inpainting layer is resized at all, both being original-size artifacts already. **(v)** Item 12(a)'s *whitelist* reading is correct and survives: `main.py:665-669` lists `final_mask, denoise_mask, inpainted_mask` in ascending priority, so `.or()` correctly describes **which branch runs**. What item 12(a) got wrong is **what that branch then does**.

    (c) **The ratified `MaskChoice` shape**, extending the existing two-variant enum in `crates/pc-export/src/discover.rs:48-54` rather than replacing it:

    ```rust
    pub enum MaskChoice {
        FinalOnly(ImageHandle),
        WithDenoise {
            final_mask: ImageHandle,
            denoise_mask: ImageHandle,
        },
        WithInpaint {
            final_mask: ImageHandle,
            denoise_mask: Option<ImageHandle>,
            inpainted_mask: ImageHandle,
        },
    }
    ```

    **`denoise_mask` is an `Option` inside one variant, and NOT a pair of variants, and the reason is (b)(ii) rather than taste:** upstream's `if denoising_enabled:` guard sits *inside* a single branch, so the two cases share one composite ordering by construction. Splitting them into `WithInpaint` / `WithDenoiseAndInpaint` would give a four-variant enum whose two inpaint arms differ by one optional layer, and would let a later edit composite the layers in a different order in one arm without any test noticing.

    (d) **`resolve`'s corrected precedence** (`crates/pc-export/src/discover.rs`'s `resolve`), stated so L6-3 does not have to infer it: when `inpainting_enabled` is true and both `sources.final_mask` and `sources.inpainted_mask` are present, the result is `WithInpaint { final_mask, denoise_mask: sources.denoise_mask.filter(|_| denoising_enabled), inpainted_mask }`. Every other combination falls through to the existing two-arm rule **unchanged**. An `inpainted_mask` present with `inpainting_enabled == false` is a stale cached artifact and is dropped, which is item 12(b) applied here and not a new rule.

    (e) **An `inpainted_mask` with no `final_mask` takes the existing WARN-and-export-no-mask path**, the same one `(None, Some(_), true)` already takes for a stray denoise mask, with a message naming the inpainting artifact rather than reusing the denoise text verbatim. Ground: upstream opens `cache_path_gen.combined_mask` unconditionally at `:243` with no guard, so a missing combined mask is a state upstream never contemplates; our `discover` reaches it only because it probes the cache for files, which is exactly the situation the existing arm was written for.

    (f) **`export_mask`'s composite, ordered.** Base = `resize_nearest_rgba(final_mask, original_size)`; then `alpha_composite_over` the noise mask at `(0, 0)` **when the `Option` is `Some`**; then `alpha_composite_over` the inpainting layer at `(0, 0)`. The order is load-bearing and is the thing L6-3's frozen test must pin — see (h).

    (g) **The resize filter is EXACT parity in this branch, and DEVIATION(8) does not propagate into it.** `DEVIATION(8)` / §15.8 records that we resize masks nearest-neighbour uniformly where upstream uses `Image.BILINEAR` for the denoise-mask branch (`image_export.py:221`). Upstream's inpainted-mask branch uses `Image.NEAREST` (`:244`), so uniform-nearest matches upstream here rather than diverging from it. **This was already on record and is not a new measurement:** §15 item 8's five-site survey lists *"`:244` (inpainted_mask) NEAREST"* verbatim, and its conclusion that `:221` is "the sole outlier across 5 sites" is exactly the reason no divergence lands here. Restated at this site because §15 item 8 is not where a reader of item 12 will look. Stated because the opposite is the natural assumption, and because a future parity investigation should not spend time looking for a divergence that is not there. The noise and inpainting layers keep `export_mask`'s existing tolerance of resizing a layer only when its size differs from the base.

    (h) **What L6-3 owes as an assertion, not as an intent.** A test over three distinguishable single-colour RGBA layers — pick values whose composite is different under every permutation of the three — asserting the exported pixel equals the **hard-coded** expected value for the `combined → noise → inpainting` order, plus a second case with `denoising_enabled = false` asserting the two-layer result. Cardinality is not identity here either: asserting "three layers were opened" would pass under a wrong order. The expected colours are literals in the test, never computed by running the composite under test.

    (i) **What this item does NOT change.** Item 12(b)'s stale-artifact rule, item 12(c)'s category map, and the `cleaned` half of item 12(a) all stand. That last one was re-checked against the same file rather than assumed — `image_export.py:234-239` really is a plain replacement, with no compositing at all:

    ```python
    if ost.Output.inpainted_output in outputs:
        export_single_image(
            cache_path_gen.clean_inpaint,
            cleaned_out_path,
            original_image,
        )
    ```

    So `cleaned = inpainted.or(<the existing expression>)` is right as written, and the asymmetry between the two halves of item 12(a) is upstream's, not a transcription slip.

23. **RATIFIED (L6-DP1) — eligibility is computed BEFORE the inpainter session is asked for; a page with nothing eligible must produce ZERO `InpainterProvider::inpainter()` calls.** Both L6 planning agents designed this same shape independently, so this too is a consensus ratification rather than a tie-break.

    **Label collision, stated because it is a real reading hazard in this entry.** Item 17 already uses `D1`-`D4` for a different set of decision points, **two** of which — `D2` and `D4` — are still live; `D1` was resolved by item 19 and `D3`'s own clause (c) reads "Ruled by this entry", exactly as item 17's preamble states (*"D2 and D4 remain open"*) and as item 19(h) repeats (*"it does not resolve D2 or D4"*). The L6 pass's points are therefore written **`L6-DP1`-`L6-DP5`** everywhere below; a bare `D2` in this entry always means item 17's, never L6's. (`L6-DP5` was not one of the planning pass's four — it was found while writing this transcription and is recorded at item 25(d).)

    (a) **The shape.** `run_inpaint` calls `pc_inpaint::select_regions(regions, config)` first and inspects the result. Only when that slice is non-empty does it ask the provider for a session. Nothing about a provider, a model file or a download is reachable from a page with no eligible region.

    (b) **The ground, quoted verbatim from item 8(c) so its scope travels with it:** *"a batch may have `inpainting_enabled = true` and **zero eligible regions on every page** — upstream's own `if boxes_to_inpaint:` guard at `inpainting.py:131` proves the empty case is expected, not pathological — and such a run must not pay a 207 MB download plus a ~6.3 s session build (item 1(f)) to inpaint nothing."* Item 8(c) is the requirement; this item is the mechanism that discharges it, and the two are not the same thing.

    (c) **This is a SECOND mechanism, not a restatement of item 8's latch, and either one alone leaves a real hole.** Item 8's `OnceLock` makes the first provisioning attempt lazy and caches its outcome; it does not stop that first attempt from happening. L6-DP1 stops the attempt from being made at all. Without (a), a run with `inpainting_enabled = true` and nothing eligible anywhere still trips the latch exactly once and still pays the 207 MB; without item 8, a resumed run that never reaches `Step::Inpaint` pays it at startup. Both are needed.

    (d) **`select_regions` is called twice on an eligible page, and that duplication is DELIBERATE.** Once by `run_inpaint` as the gate, once inside `inpaint_page` (`crates/pc-inpaint/src/lib.rs:181`). It is a pure function of `(regions, config)`, both already in hand, and threading the first result into `inpaint_page` would move the eligibility contract out of the crate that owns it and admit a caller-supplied set `select_regions` would never produce. Do not "optimise" this.

    (e) **The frozen test L6-4 owes, stated as assertions with their literals.** A counting `InpainterProvider` double whose `inpainter()` increments a shared counter. **Case 1:** a page whose `#mask_data.json` regions are all ineligible (`failed == false`, `std_deviation` below `inpainting_min_std_dev`, `thickness == None`), run with `inpainting_enabled = true`; assert the counter is **0**, and assert the run still reaches `Completed` with the non-inpainted cleaned artifact exported. **Case 2, and it is not optional:** the same fixture with one region made eligible; assert the counter is exactly **1**. Case 1 alone is vacuous — it passes identically if the provider was never wired into the pipeline at all, which is precisely the plumbing failure item 16(f) warns about. The pair is what makes the gate capable of failing.

    (f) **`InpainterProvider` ALREADY EXISTS — L5 shipped it, and this item adds a caller rather than a trait.** The trait is `crates/pc-cli/src/inpainter.rs:42`, with `fn inpainter(&self) -> Result<Arc<dyn Inpainter>, StageError>` and a **mandatory, no-default** `fn failures_are_run_fatal(&self) -> bool` that both shipped implementations answer `true`. Stated explicitly because the first draft of this item asserted the opposite — that no such predicate should exist — which contradicted code already on this branch; that assertion is retracted here rather than left to be discovered, and the shipped shape stands. The asymmetry with `DetectorProvider`'s *defaulted* `false` (`crates/pc-pipeline/src/ctx.rs:20-26`) is deliberate on the shipped side: the detector's predicate is defaulted because `ReplayDetector` binds per image and may legitimately answer `false`, while the inpainter's has no default because item 9 leaves no discretion — construction takes no image in every implementation, including the test doubles. What L6-4 owes is only that `run_inpaint` **consult** the predicate and map a `true` answer to a run-fatal `PipelineError`, so fatality stays declared and never inferred at the call site (cookbook rule 4).

24. **The L6 task breakdown — five tasks, four Codex calls, with the `heavy`/`simple` classification `CLAUDE.md`'s batching rule requires.** Item 16(e) scoped L6 as one heavy call. This pass splits it, and the ground is concrete rather than tidiness: 16(e)'s single call would have mixed a `pc-core` enum change, a `pc-export` precedence change and a `pc-pipeline` stage wiring into one invocation, where the *first* of those turns three frozen tests red in two other crates before any of the rest compiles. Item 16(e) is not otherwise altered, and (a)-(e) below cover everything 16(e) named — but they are **not** an exact union of it: they also carry material 16(e) never named, namely item 22's three-layer mask composite and its frozen test, item 25(c)'s `run_inpaint` adapter shape, item 23's eligibility-first gate with 23(e)'s two-case test, the provider's construction in `pc-cli` and injection into `PipelineCtx` (item 25(d)), checkpointing at the new step, `export_sources` populating the two new fields, and item 19(g)'s refusal hint getting its first caller. Those additions arrived with items 19-25 after 16(e) was written; the split is a re-scoping, not a re-partition of a fixed set.

    (a) **L6-1 — `Step::Inpaint` in `pc-core` (simple).** Item 11(a)'s enum plus item 11(b)'s three pre-authorised value corrections and the three additions it names. Acceptance, as literals: `Step::Export as i32 == 6`; `Step::Export.prev() == Some(Step::Inpaint)`; `Step::Inpaint.prev() == Some(Step::Denoise)`; `Step::Denoise < Step::Inpaint < Step::Export`; and `Output::ALL.len() == 13` still green **and unedited**, per item 11(d).

    (b) **L6-2 — the two pipeline-local cache suffixes (simple; batched sequentially with L6-1 in ONE Codex call).** `"_inpainting.png"` and `"_clean_inpaint.png"` as `pc-pipeline` constants beside `SPLITS_SUFFIX` (`crates/pc-pipeline/src/cache.rs:18`), added to `known_suffixes()`. Batchable with L6-1 because the two touch disjoint files and L6-2 does not depend on the enum. Acceptance: both new suffixes round-trip through `CachePaths::from_existing`, **and** the real invariant of item 25(b) below.

    (c) **L6-3 — `pc-export` precedence and the three-layer composite (heavy, its own call).** `ExportSources`' two new fields, `MaskChoice::WithInpaint` and `resolve` per item 22(c)-(e), `export_mask`'s ordered composite per 22(f) with 22(h)'s frozen test, item 12(b)'s stale-artifact rule, and item 14(c)'s greyscale export obligation pinned on a greyscale input.

    (d) **L6-4 — `pc-pipeline` stage wiring (heavy, its own call; depends on L6-1, L6-2, L6-3).** The `run_inpaint` adapter of item 25(c), making L5's existing `InpainterProvider` reachable from `PipelineCtx` per item 25(d), item 23's eligibility-first gate with 23(e)'s two-case test, the cache writes for both artifacts, checkpointing at the new step, and `export_sources` populating the two new fields.

    (e) **L6-5 — `pc-cli`'s `--skip-inpaint` and the provider handoff (simple; depends on L6-4).** The flag in the existing `--skip-denoise` shape, the provider constructed in `pc-cli` and injected, and item 16(f)'s check that a non-default `inpainting_enabled` is actually observed at the stage. Item 19(g)'s optional-aware refusal hint gets its first caller here, which is what "added now, unwired" was waiting for.

    (f) **The batching, stated explicitly so it is not re-derived:** call 1 = L6-1 then L6-2, sequential; call 2 = L6-3; call 3 = L6-4; call 4 = L6-5. Four calls for five tasks.

25. **The L6 pass's remaining decision points, with each clause's current status stated separately so a reader does not average them.** Clauses (a) and (d) are **RESOLVED by §16.40**, (b) is a **finding** about existing text and not a defect that was fixed, and (c) is **RATIFIED**.

    (a) **`L6-DP2` — RESOLVED by §16.40; read that entry before implementing or citing long-strip inpainting.** This clause formerly left open whether `Step::Inpaint` runs per strip segment or once on the stitched strip and deferred every strip test. §16.40 chooses per-segment execution plus mixed-source stitching for the scope it quotes verbatim. The measurement remains useful: `crates/pc-pipeline/src/single.rs` runs stages 1-4 per segment (`run_stages`, `:6`) and exports the stitched result once (`merged_strip_export`, `:475`). Its former statement that no clause may define long-strip inpainting is no longer live within §16.40's stated scope.

    **SUPERSEDED IN PART by §16.40 — this back-pointer qualifies the former OPEN, DEFERRED status and the former omission of strip tests; the measurement above remains unchanged.**

    (b) **`L6-DP3` — item 11(e)'s suffix collision does not exist. Recorded as a FINDING, not as a defect that was fixed.** Measured 2026-08-07 over the full suffix set — the thirteen `Output::cache_suffix()` values, `SPLITS_SUFFIX`, and the two new constants — testing every ordered pair for one being a proper tail of the other: **zero pairs**. Concretely, `_clean.png` is not a tail of `_clean_inpaint.png`, and it is not a tail of `_clean_denoised.png` either, so item 11(e)'s clause naming "the same collision `_clean_denoised.png` already has" describes a collision that has never existed at any point in this repo. **The obligation item 11(e) states nonetheless stands** — it reads "L6 owes a test that the new suffixes disambiguate under the existing longest-match rule" — and L6-2 discharges it by asserting the **real** invariant (no suffix in `known_suffixes()` is a proper tail of another) rather than a disambiguation that never applied. **No marker is raised against item 11(e), and that is a judgement rather than a measurement:** the obligation survives intact and only its parenthetical characterisation is wrong. A reader who thinks the wrong characterisation itself warrants one is making a defensible call and should raise it with both architects; nothing in `crates/pc-testkit/tests/spec_supersession.rs` gates it either way.

    (c) **`L6-DP4` — RATIFIED: no `pc_core::Stage` impl for inpainting is possible, and the `run_inpaint` free function in `pc-pipeline` is the shape.** `pc_core::Stage` requires `type Input: Serialize + DeserializeOwned` (`crates/pc-core/src/stage.rs:15`), and `pc_inpaint::PageInput<'a>` (`crates/pc-inpaint/src/lib.rs:102-118`) holds **seven** fields, six of which borrow: four image references — `original: &'a RgbImage`, `raw_mask: &'a GrayImage`, `combined_mask: &'a RgbaImage`, `noise_mask: Option<&'a RgbaImage>` — plus `regions: &'a [MaskRegionStats]` and `config: &'a InpainterConfig`, with only `min_mask_thickness: u32` held by value; a type with a lifetime parameter cannot implement `DeserializeOwned`. Two alternatives were considered and rejected: an owned `InpaintInput` mirror (it would clone the full-resolution original plus three masks per page to satisfy a trait impl nothing consumes), and turning the fields into `ImageHandle`s loaded inside the stage (item 16(e) already assigns loading to L6, and `pc-inpaint` is deliberately filesystem-free). Ratified: `run_inpaint` is a free function in `crates/pc-pipeline/src/single.rs` following the existing `denoise_dests`/`export_sources` shape, and `Step::Inpaint` exists as an ordering and checkpoint value with no `Stage` impl behind it.

    **The consequence of (c), stated rather than left to be discovered.** `Step::Inpaint` becomes the **first** `Step` variant with no `impl pc_core::Stage` — measured, not assumed: `grep -rn "impl pc_core::Stage for" crates/*/src/` returns exactly five hits today, one for each of `Detect`, `Preprocess`, `Mask`, `Denoise`, `Export`. So §3's rule — *"each stage crate exposes one pure function"* (`docs/PIPELINE_SPEC_V1.md:321`, quoted verbatim) — acquires one stage whose function is not routed through the trait, and any future code that enumerates `Stage` impls in order to enumerate stages will silently miss inpainting. Nothing in this repo gates that; it is disclosed here because the alternative was to let a later reader find it.

    (d) **`L6-DP5` — RESOLVED by §16.40; read item 2(d) there before implementing the provider boundary.** The measurements and options below are retained as the record of the question, not as live alternatives. The trait is in `crates/pc-cli/src/inpainter.rs:42` (L5), and `PipelineCtx` (`crates/pc-pipeline/src/ctx.rs:47-50`) must carry it for L6-4 — but `pc-pipeline` **cannot** depend on `pc-cli`; the edge runs the other way (`crates/pc-cli/Cargo.toml:28`). Two shapes were available. **(i)** Move the trait alone into `pc_pipeline::ctx` beside `DetectorProvider` and re-export it from `pc-cli`, leaving `UnavailableInpainterProvider` and `OnnxInpainterProvider` where they are; that move **does add a new non-dev edge**, and an earlier draft of this clause claimed the opposite — corrected here rather than left standing. `pc-inpaint` is today only a **dev**-dependency of `pc-pipeline`: `crates/pc-pipeline/Cargo.toml:47` sits under the `[dev-dependencies]` table that opens at `:33`, and that file's own comment at `:45-46` says so verbatim — *"This is NOT L6's wiring: nothing under `src/` references `pc-inpaint`, and `Step::Inpaint` does not exist yet."* Re-measured 2026-08-07: `grep -rn "pc_inpaint" crates/pc-pipeline/src/` returns **0** hits. So putting the trait in `pc_pipeline::ctx` — a `src/` module that names `pc_inpaint::Inpainter`, which is where the returned `Arc<dyn Inpainter>` is defined — requires promoting `pc-inpaint` to a real `[dependencies]` entry, a genuinely new edge in the workspace graph. **That edge is an allowed one:** the same `Cargo.toml` records at `:15-16` and `:43-44` that *"`pc-pipeline` is the ONE crate allowed to depend on every stage"*, §1 rule 2 forbidding only stage-to-stage edges. **(ii)** Leave the trait in `pc-cli` and have `PipelineCtx` take a provider-shaped closure instead. The former lean was (i); §16.40 makes (i) binding and rejects (ii). `DEVIATION(27)`'s declared site is the **lazy latch**, which lives in `OnnxInpainterProvider` and does not move, so §14 item 27's *"the inpainter provider in `crates/pc-cli/`"* remains true of the implementation carrying the deviation.

    **SUPERSEDED IN PART by §16.40 — this back-pointer resolves the OPEN provider-location choice; the measurements and dependency-edge analysis above remain unchanged.**

## 16.39 A4 ratification: the `Annotation` coverage operand, `DEVIATION(12)` narrowed rather than retired, the clause markers, and the A4-a…A4-d breakdown (joint architect + Senior Rust Engineer plan pass, converging independently, 2026-08-07)

**This section is spec-only. Nothing here asserts that `Annotation` is wired into `pc_detect::run`.** At the moment it is written, `run` still refuses the mode at `crates/pc-detect/src/lib.rs:90`, `run_rejects_annotation_refine_mode` is still green, and `MaskRefineMode::Simple` is still the shipped default and stays so afterwards. What lands here is the set of decisions A4 implements, transcribed before the code, exactly as R0 was.

**Why this is §16.39 and not §16.38, checked rather than assumed.** §16.38 is claimed by the sibling `lama-inpaint` branch — verified by reading `docs/PIPELINE_SPEC_V1.md` in that worktree at its committed `ead8bd1`, where `## 16.38 LaMa inpainting (v1.5)…` exists. Taking 38 here would manufacture the same cross-branch collision §14's item-20 note records for `DEVIATION(24)`, and would do it after the collision was already paid for once. The union of top-level `16.x` numbers claimed on any branch is **16.5–16.38** — re-checked by listing `^## 16\.` in all three worktrees on 2026-08-07: the register starts at §16.5 and no branch has ever claimed §16.1–§16.4, so the union is not 16.1–16.38 as an earlier draft of this sentence said. The conclusion is unaffected, because the top of the range is what decides it: 39 is the first free one. As §16.33 did, this section is inserted **before** §16, whose span runs to EOF.

**Both planning passes reached items 1 and 2 independently and agreed.** That is convergence, not a dispute, so it is a joint ratification and not a Fable escalation; Fable is convened for disagreement, and there was none on these two. Neither agent's argument is treated here as evidence for the other's — where a number below has one source, it says so.

1. **DECIDED: under `MaskRefineMode::Annotation`, the §8.3 step 6 coverage filter scores each block against the UNREFINED, letterbox-cropped, resized detector mask. `Simple`'s operand is unchanged.** This closes what §16.37 item 6 left open in its own words — *"What to do about it — accept, re-tune `min_mask_coverage` for this mode, or change the filter's operand — is an A4 decision and is deliberately left open"* — and it takes the third option.

   (a) **Upstream's operand, quoted rather than paraphrased, at pinned commit `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3`.** Both planning agents reached this by running upstream; the lines below were additionally re-read first-hand from a fresh checkout at that commit while writing this transcription (read, not run — the running is the two agents' and is not re-claimed here). `pcleaner/comic_text_detector/inference.py:193-208`:

    ```python
    193:        # map output to input img
    194:        mask = mask[: mask.shape[0] - dh, : mask.shape[1] - dw]
    195:        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
    ...
    203:        blk_list = group_output(blks, lines, im_w, im_h, mask)
    204:        mask_refined = refine_mask(img, mask, blk_list, refine_mode=refine_mode)
    205:        if keep_undetected_mask:
    206:            mask_refined = refine_undetected_mask(
    207:                img, mask, mask_refined, blk_list, refine_mode=refine_mode
    208:            )
    ```

    The filter lives inside `group_output`, which is called at `:203` — **before** `refine_mask` at `:204` — and is handed `mask`, the array built at `:194-195` by cropping the letterbox padding off and resizing to the base image size. `pcleaner/comic_text_detector/utils/textblock.py`, the three lines the citation names:

    ```python
    456:    mask_score_thresh = 0.1
    471:                mask_score = mask[by1:by2, bx1:bx2].mean() / 255
    488:                mask_score = mask[by1:by2, bx1:bx2].mean() / 255
    ```

    `:488` is the one that decides a **detected** block's fate; it sits inside `if len(blk.lines) == 0:` at `:485`, which is every block for us because we synthesize no line polygons. `:471` is the same expression in the scattered-lines branch at `:469-473`, which we do not reach. `:456`'s `mask_score_thresh = 0.1` is numerically **exactly** `pc_detect::DEFAULT_MIN_MASK_COVERAGE` (`crates/pc-detect/src/lib.rs:51`), and `mean() / 255` is exactly `mask_coverage`'s definition. So under this decision the whole quantity — operand, expression and threshold — matches upstream, and the only surviving difference at this step is the scope one §14's item 17 already registers (upstream tests only line-less blocks; for us every block is line-less, so that difference is vacuous for v1).

   (b) **Scope, stated narrowly because the evidence is narrow.** This is an `Annotation`-only decision. Under `Simple` the operand stays the refined mask, the values frozen against it stay frozen, and `DEVIATION(17)` keeps describing `Simple` accurately. Nothing here is a claim that scoring the unrefined mask is *better*; it is a claim that it is what upstream does, and `Annotation`'s whole purpose is upstream parity.

   (c) **REJECTED: re-tune `min_mask_coverage` for this mode.** Refuted by a measurement, not by preference: §16.37 item 6 records `[531,836,686,882]` on E01P02 collapsing to `0.0000000`. There is no positive threshold that keeps a block whose coverage is zero, so the option does not merely cost accuracy — it cannot work on a case already on the record. A threshold low enough to matter for `0.0238280` would also stop rejecting anything, which is the filter's whole job.

   (d) **REJECTED: accept the drops.** Three blocks that `Simple` keeps would be dropped with no mask, no OCR and no diagnostic. That breaks the precondition §16.37 item 11(d) attaches to A5 — *"its determinism gate and calibration report must run against the **complete** algorithm — both `refine_mask` and `refine_undetected_mask` — not a partial one"* — because a calibration report over a block set our own filter has already thinned is not a comparison with upstream's page at all, and the difference would be attributed to the port. Accepting the drops also silently makes `Annotation` worse than `Simple` on real pages, which inverts the reason the port exists.

   (e) **Implementation, so A4-b has one operand and not two.** `run` currently computes coverage against `refined_mask` at `crates/pc-detect/src/lib.rs:128`, and the unrefined operand exists only *inside* `refine_simple` (`crates/pc-detect/src/mask.rs`, as `resize_bilinear(&crop_letterbox(mask, dw, dh)?, image_size)`). A4-b lifts that composition to `run` so both modes name the same two arrays explicitly, and the mode selects which one the filter reads. The lift must not change `Simple`'s bytes — A4-a's byte-identity lock is what catches it if it does.

   **SUPERSEDES: §8.3 step 6**

   **SUPERSEDES: §14 item 17**

   Both markers are this transcription's application of the marker convention to item 1's consequences, not extra decisions: neither clause's proposition changes for `Simple`, and both stop being mode-independent the moment `Annotation` runs. They are called out as the transcription's own act so a reviewer can reject them without touching the ruling.

2. **DECIDED: `DEVIATION(12)` is narrowed, not retired — and this contradicts the word "retirement" in §16.37 items 8 and 11(d).** Both planning agents reached this independently, with the same argument, having each noticed that the existing ratified text says "retirement".

   (a) **The argument.** Upstream has no mode switch: `inference.py:204` runs `refine_mask` unconditionally on every page, and `:205-208` runs `refine_undetected_mask` whenever `keep_undetected_mask` is set, which `pcleaner/ctd_interface.py:162` sets. panel-ocr's default remains `MaskRefineMode::Simple` — §16.37 item 9 mandates exactly that (*"`Annotation` ships **non-default**; `Simple` stays the default and every value frozen against it stays put"*). So after A4 a **default** panel-ocr run still does something upstream never does. Retiring the entry outright would assert a by-default parity the tree does not have, which is the same class of false claim §16.37 items 4 and 5 exist to correct and which item 11(a) already warned about in the other direction.

   (b) **The narrowed proposition, in the form the register entry now carries.** *Not* "we did not port `refine_mask`/`refine_undetected_mask`" — as of A4 we did, and A3b ported both halves. What survives is: **the shipped default is `Simple`, so a default run diverges from upstream's unconditional refinement; parity is reachable only by opting in with `mask_refine_mode = "annotation"`.** The divergence is real, it is a default-value divergence, and it is the only one left at this site.

   (c) **The site comment moves, because the artifact carrying the divergence moved.** The primary `DEVIATION(12)` comment is at `crates/pc-detect/src/mask.rs:150-153`, on `refine_simple` — an algorithm that is no longer the divergence, since we now ship upstream's algorithm too. It moves to the `#[default] Simple` variant of `MaskRefineMode` in `crates/pc-config/src/profile.rs`. That is cookbook rule 12 applied literally: gate and annotate the artifact carrying the risk. **No line number is given for the destination on purpose** — this register has already had to correct three line citations that were right when written (§14 items 19 and 23, §16.37 item 2), and A4 edits that file in the same commit, so a number written here would be stale before it was read. The obligation is discharged by the comment sitting on that variant, and A4-d's gate is what checks it.

   (d) **What `refine_simple` keeps.** A pointer, not the primary comment: a sentence saying `Simple` is koharu's algorithm and that the register entry now lives at the default. The "v1.5 door" sentence at `mask.rs:152` stops describing reality and goes.

   **SUPERSEDES: §14 item 12**

3. **Three clause markers, all landing with A4 — but item 7 pre-authorised only two of them, and the third is a call it explicitly left open.** Stated this way because an earlier draft's lead read *"§16.37 item 7's three clause markers"*, which credits item 7 with authorising a marker it declined to pre-judge. Item 7 authorised amending **§16.5 item 3** and **§8.3 step 5** and deliberately did not write either marker at R0; for **§15 item 2** it wrote that *"A4 decides whether staleness warrants a marker at all, and this item does not pre-judge that"*. (c) below is where that decision is made, and it is A4's, not item 7's.

   (a) **§16.5 item 3 — FALSE, marker written.** Its *"rejected by `pc-detect` (`StageError::InvalidInput`)"* half stops being true. Its layer split (config accepts, stage decides) is untouched and is in fact what makes the opt-in work.

   (b) **§8.3 step 5 — FALSE, marker written.** The out-of-scope bullet's *"where only `Simple` is implemented in v1 (`Annotation` → `StageError::InvalidInput`)"* stops being true. Note this is a **second** marker against §8.3 step 5 from a **different** claiming section; §16.37 item 5 already carries one against a different sentence of the same step (the `expand_textwindow` parenthetical). Both resolve to §8.3's whole span by the `step N` fallback, so one back-pointer would satisfy both — they are written at their own sentences anyway, and the registry carries both rows.

   (c) **§15 item 2 — STALE, and the call is that it DOES get a marker.** Item 7 posed this exactly: *"§15 item 2's `Annotation` mode remains the v1.5 door for a full refinement port` **rejects nothing and does not become false — it becomes stale**, describing a door as unopened once it is opened; A4 decides whether staleness warrants a marker at all, and this item does not pre-judge that."* Both planning passes leaned yes; the call recorded here is **yes**. Grounds, and they are about the reader rather than about severity: §15 is the sign-off register a future reader lands on to learn what was decided and what is still open, and "remains the v1.5 door" reads as *not yet done*. A reader who lands there and does not read this section will conclude the port has not happened — which is precisely the failure the back-pointer convention exists to prevent, and it does not become less of a failure because the sentence was true when written. The cost is one line; the alternative is a register that quietly misdescribes the tree. The rest of item 2 — ship `Simple` as the default, keep the upstream comparison non-gating — is unchanged and is a **premise** of item 2 above.

   (d) **The one thing item 7 predicted and could not verify.** Item 7 wrote: *"**Predicted, not verified:** that test is expected to go red when A4 lands, but whether it actually does depends on A4's opt-in design, which does not exist yet — if A4 gates on a config flag whose default keeps the refusal, the test could stay green, and A4 must check rather than assume."* The design decided here adds **no** second flag: `mask_refine_mode` is itself the opt-in, so `run` stops refusing and `run_rejects_annotation_refine_mode` goes red. That is a statement about the design, not a measurement — A4-a confirms it by running the suite and reports what it saw. Under cookbook rule 8 this is exit 2 (the corrected assertion claims something **different**, not less), routed to both architects, which is this section; A4-a replaces the test and **renames** it, because a test named `run_rejects_annotation_refine_mode` that no longer asserts a rejection is this project's dominant defect class.

   **SUPERSEDES: §16.5 item 3**

   **SUPERSEDES: §8.3 step 5**

   **SUPERSEDES: §15 item 2**

4. **DECIDED: an empty expanded window is a per-image `StageError::InvalidInput` that propagates out of `run`, never a silently skipped block.** A3b already made this classification and declared it rather than leaving it to be inferred from a signature (cookbook rule 4) — `crates/pc-detect/src/annotate_refine.rs:86-91` states it, and `refine_mask` returns that error at `:183-185`. What A4 owes is not to re-decide it but to **not swallow it**: the wiring must propagate, so the image fails and the run continues. Upstream-faithful, and measured rather than argued: A3b recorded that upstream reaches `cv2.cvtColor` on a zero-sized array and dies with `(-215:Assertion failed)` at `modules/imgproc/src/color.cpp:199`, i.e. upstream itself fails on this input rather than skipping the block. Skipping would be strictly worse than both options — a page silently missing a mask region with no error anywhere. Run-fatal would be wrong for the opposite reason: one malformed block in one image is not grounds to abort a batch, and §5's per-image isolation is the rule.

5. **The fresh code-site enumeration, re-derived by grep here and NOT inherited from §16.37 item 7 — which is what item 7 required (*"A4 re-derives this list with a fresh grep and records the result; it does not inherit this one"*).** Command: `grep -rn -i annotation crates/ README.md docs/PIPELINE_SPEC_V1.md`, 2026-08-07, run against this branch after this section's own back-pointer edits landed, so the spec line numbers below are live. **Two of item 7's own entries were wrong, in different ways, and the difference matters:**

   (a) **`crates/pc-detect/src/lib.rs:71` is STALENESS, not an original error.** At R0 (commit `1b4e12f`) line 71 was inside `run`'s doc comment, verified by `git show 1b4e12f:crates/pc-detect/src/lib.rs`; A1 through A3b added `pub use` lines and module doc comments above it. The live sites are **`:86-88`** (the doc comment sentence *"Rejects `MaskRefineMode::Annotation` with `StageError::InvalidInput` **before** any detection work"*) and **`:90-94`** (the check itself). Same distinction §14's items 19 and 23 draw about their own citations.

   (b) **The `default_profile.toml` claim is an ORIGINAL ERROR, and it is refuted by running the suite, not by reading it.** Item 7 says of that file's line 20: *"the TOML is locked by `crates/pc-config/tests/defaults.rs`'s literal-§6-block comparison, so editing it without the test is a red suite"*. There is no such comparison. `defaults.rs` compares **by value** — which is §16.5 item 4's explicit ratified choice (*"Config defaults are frozen by value, not by re-serialized text"*) — plus a table/key registry check; none of that reads a comment. The only byte comparison is `crates/pc-config/tests/round_trip.rs:79-83`'s `default_profile_round_trips_byte_for_byte`, which compares `DEFAULT_PROFILE_TOML` against **itself** after a parse/serialize cycle: it proves comments survive the round trip and is invariant to what they say, because editing the comment changes both sides together. **Measured, 2026-08-07:** line 20's trailing comment was replaced wholesale with `# PROBE COMMENT: this text is not gated by any test` and `cargo test --workspace` ran to **exit 0 with 139 `test result: ok` lines and no failures**; the file was then restored from git. So the comment is a **free-standing prose site** with no gate behind it — which is exactly why A4-d's gate must cover it explicitly, and is a live example of cookbook rule 12: item 7 pointed at a gate that was not gating the artifact.

   (c) **Sites whose text becomes FALSE when A4 lands** (each must change in A4's commit). **Which of these carry a §16.39 marker, counted rather than asserted — a draft of this parenthetical said "the four with markers above are the spec ones" and was wrong in both halves.** This list holds **four** `docs/PIPELINE_SPEC_V1.md` entries — `:499`, `:699`, `:1475-1477` and `:1527` — and **two** of the four carry a §16.39 back-pointer: `:699` (§8.3 step 5, marker at `:701`) and `:1475-1477` (§16.5 item 3, marker at `:1479`). `:499` does not, and should not: it is §6's reproduction of the shipped TOML, edited in the same commit as the `default_profile.toml:20` line it mirrors, not a ratified proposition with its own back-pointer. `:1527` does not either, and **that one is flagged as open rather than settled — see the note at the end of this sub-item.** Going the other way, four of §16.39's six markers target clauses that appear **nowhere** in this list, because they are narrowed rather than falsified: §8.3 step 6 and §14 item 17 (item 1's two transcription markers, `Annotation`-only operand), §14 item 12 (item 2, narrowed not retired) and §15 item 2 (item 3(c), stale not false, and listed in 5(d) below). Marker count and FALSE-site count are different quantities and neither is derivable from the other.
   `docs/PIPELINE_SPEC_V1.md:499` (§6's default profile block, trailing comment *"annotation" rejected in v1*) · `:699` (§8.3 step 5's out-of-scope bullet) · `:1475-1477` (§16.5 item 3) · `:1527` (§16.6 item 1's *"Required so `pc-detect` can reject `MaskRefineMode::Annotation`"* rationale — **moved here from 5(d), where a draft filed it as merely stale**; it asserts the same proposition as `crates/pc-detect/src/detector.rs:29-31` below, which that draft filed as false, and one sentence cannot be both. The proposition *"the stage can reject `Annotation`"* stops being true, so FALSE is the class for both; what stays true at both sites is that the `config` field is **required**, for the branch) · `crates/pc-config/src/default_profile.toml:20` · `crates/pc-config/src/profile.rs:208-210` (`MaskRefineMode`'s doc comment) · `crates/pc-detect/src/lib.rs:86-88` and `:90-94` · `crates/pc-detect/src/detector.rs:29-31` (the `config` field's doc comment, *"required so the stage can reject `MaskRefineMode::Annotation`"* — the field is still required, for the branch; the word "reject" is what goes, and it sits on `:29`. An earlier draft cited `:30-32`, which starts one line below the comment and so excludes the very word it names) · `crates/pc-config/tests/defaults.rs:113` and `:356-358` (comments only — `annotation_refine_mode_loads_successfully`'s **assertions** stay true and stay frozen, since config still accepts the value; its `.expect("config accepts annotation; pc-detect is what rejects it")` message at `:361` is the false part) · `crates/pc-detect/tests/d7_run.rs:196-215` (the frozen test, item 3(d) above).

   **OPEN, and deliberately not settled by this transcription: whether `:1527` (§16.6 item 1) needs its own back-pointer marker.** Item 3 above enumerates the clause markers this ratification authorises — §16.5 item 3, §8.3 step 5, §15 item 2 — and §16.6 item 1 is **not** among them; item 7 of §16.37 did not name it either. Reclassifying it from stale to false is a correction to *this* item's own bookkeeping and is within a transcription's remit; **adding a seventh marker would be a new ratification act and is not.** The argument for one is that §16.6 item 1 is a ratified decision whose stated rationale A4 falsifies, which is exactly the §16.19 trigger; the argument against is that the decision itself (`DetectInput` gains `config`) survives untouched and only its *because*-clause moves. This is recorded for the architects to close, not resolved here; the enumeration obligation of cookbook rule 14 is discharged by the site being listed either way.

   (d) **Sites that become STALE without becoming false** (they stop describing the tree; no marker is claimed for any of them, and item 3(c) explains why §15 item 2 is treated differently from these):
   `docs/PIPELINE_SPEC_V1.md:1440` (§15 item 2's "v1.5 door" — the one that **does** get a marker) · the `MaskRefineMode::Annotation` bullet in §16's out-of-scope list at the end of this file · `:3805` (**§16.22 item 7**, not §16.23 — the line is inside §16.22, whose span ends at `:3806`, and it *points at* §16.23 item 2 without being in it; an earlier draft of this item attributed it to §16.23) · `:3835-3854` (§16.23 item 2's *"`MaskRefineMode::Annotation` must NOT land before F1 records"* — its precondition is satisfied, F1 recorded long ago, so the constraint is spent rather than violated) · `:3821` (§16.23's **v1.5-ships** list, which names `MaskRefineMode::Annotation` as still to come) · `:3899` (§16.23's v1.5 sequencing, naming Annotation as the historical-final step — the exact words "Annotation is still last" are on the following line, `:3900`) — **these last two were missed by the first draft of this item and added on re-grep**; they are the same spent-constraint class as `:3835-3854`, from the same section · `crates/pc-detect/src/mask.rs:150-153` (`DEVIATION(12)`, item 2(c)-(d) above) · `README.md:102` (the roadmap checkbox). (Re-pinned 2026-08-09, corrected again 2026-08-09 after a fresh-reader mismatch: the five `docs/PIPELINE_SPEC_V1.md` line numbers above — `:3805`, `:3806`, `:3835-3854`, `:3821`, `:3899` — each independently re-verified against the actual quoted sentence at that line, not just shifted by a flat +25; content unchanged.)

   **(d.i) RECLASSIFIED out of this list, because they become FALSE and not merely stale — they carry 5(c)'s obligation, not this one.** The first draft of this item filed the "not wired" headers as stale; they are not. *"This module is not wired into anything"* and *"`d7_run.rs::run_rejects_annotation_refine_mode` still passes"* are propositions A4 falsifies outright, the second because item 3(d) replaces and renames that test. The live spans, each re-read rather than inherited: `crates/pc-detect/src/lib.rs:10-11`, `:13-15`, `:17-19` (three module doc comments, *"**Not wired into [`run`]**, which still rejects that mode"*; the first draft cited only `:14` and `:18` and missed `:10-11` entirely, because line 11 carries the claim without the token this grep searches for — see 5(f)) · `crates/pc-detect/src/annotate.rs:7-10` · `crates/pc-detect/src/annotate_merge.rs:14-16` · `crates/pc-detect/src/annotate_refine.rs:7-10` · `crates/pc-detect/tests/a1_annotate_topk.rs:19-21` · `crates/pc-detect/tests/a2_annotate_otsu.rs:65-67` · `crates/pc-detect/tests/a3_annotate_merge.rs:60-62`. That is **nine spans across seven files** — recounted by reading the list above item by item, after a draft of this sentence said "six module/scope headers", which is neither the number of spans nor the number of files. The seven files are `lib.rs` (three spans), `annotate.rs`, `annotate_merge.rs`, `annotate_refine.rs`, `a1_annotate_topk.rs`, `a2_annotate_otsu.rs` and `a3_annotate_merge.rs` (one span each). **The first draft named two of them**, not three: `lib.rs:14` and `lib.rs:18`, per this item's own parenthetical above, and 5(f) below independently records the additions as *"`lib.rs:10-11` and the six headers now in (d.i)"* — seven added to two already listed is the nine here, and the two statements corroborate each other rather than one being derived from the other. (Each of the nine spans was re-read on 2026-08-07 and the wording is not uniform, so it is recorded rather than summarised: `lib.rs`'s three carry *"**Not wired into [`run`]**, which still rejects that mode"*; `annotate.rs:7-10` carries *"**This module is not wired into anything.** `pc_detect::run` still rejects `MaskRefineMode::Annotation`"* citing §16.37's preamble and **not** the test name; `annotate_merge.rs:14-16` and `annotate_refine.rs:7-10` carry that sentence **plus** *"`d7_run.rs::run_rejects_annotation_refine_mode` still passes"*; the three test files carry a *"**Scope.** A1/A2/A3 is *not* Annotation mode"* header with the same two clauses and not the "not wired into anything" one.) Separately, three of them — `annotate_merge.rs:22`, `annotate_refine.rs:10`, `a3_annotate_merge.rs:68` — say A4 performs *"the `DEVIATION(12)` **retirement**"*, which **item 2 above makes false in a second, independent way**: the entry is narrowed, not retired.

   (e) **`crates/pc-cli/` is still NOT a site**, re-confirmed by this grep rather than inherited: `grep -rn -i annotation crates/pc-cli/` returns nothing.

   (f) **What is asserted complete, and what the first draft of this item got wrong.** The first draft said *"This list is asserted complete for the token `annotation` … Hits inside [the Apache licence], [`spec_supersession.rs`]'s own comments, and every `annotate*` identifier are excluded as unrelated"* — i.e. that 5(c) and 5(d) between them covered every hit outside three exclusion classes. **That was false, and re-running the grep on 2026-08-07 refuted it rather than an argument doing so:** `docs/PIPELINE_SPEC_V1.md:3821` and `:3899` (re-pinned 2026-08-09; the prior pins `:3798`/`:3876` already carried a pre-existing 2-line drift before the §16.42 insertions, so the net move from those old numbers is +23, not a flat +25 — each line independently re-verified against its target passage, same as item 5(d)'s treatment of these same two sites) were in neither list and in no exclusion class, and so were `crates/pc-detect/src/lib.rs:10-11` and the six headers now in (d.i). All are added above.

   The command, re-run in full for this correction, is `grep -rn -i annotation crates/ README.md docs/PIPELINE_SPEC_V1.md`. It returned **138** hits before the first correction was written and **145** after — and **147** after the second correction pass (2026-08-07) that recounted 5(d.i)'s spans, re-classified `:1527`, and added 5(g). **The total is not a stable number, because this section's own prose is inside the search scope**: 34 of the 147 are in §16.39 itself, which now spans `:7504`–`:7653`. Anyone re-running it will get a different total the moment this section is edited again; the durable claim is the classification below, not the count. What is asserted, and only this: **every hit whose prose A4 makes false or stale is enumerated in 5(c), 5(d) or 5(d.i)**, and every remaining hit falls into one of these five named classes, each of which stays accurate after A4:

   1. **The Apache licence text**, `crates/pc-ocr/assets/LICENSE-APACHE-2.0.txt` (1 hit) — the licence's own wording, unrelated.
   2. **Identifiers and literals rather than prose:** the `Annotation` variant itself (`crates/pc-config/src/profile.rs:216`), `ANNOTATION_EXPAND_R` and its re-export and uses (`crates/pc-detect/src/annotate.rs:41` and `:44`, `crates/pc-detect/src/lib.rs:35`, `annotate_refine.rs:98` and `:181`, and the A1 test call sites), the `"annotation"` TOML value under test (`crates/pc-config/tests/defaults.rs:345`, `:350`, `:360`, `:364`, and the test's own name at `:359`), upstream's `REFINEMASK_ANNOTATION` wherever it is quoted (including `docs/PIPELINE_SPEC_V1.md:3384`, re-pinned 2026-08-09 from `:3359`, +25, by the §16.42 spec insertions above this point; content unchanged), the `annotation_*_matches_the_upstream_oracle` test names, `docs/PIPELINE_SPEC_V1.md:7389`'s reference to `annotation_refine_mode_loads_successfully` as a *pattern* (**UNVERIFIED, flagged rather than re-pinned 2026-08-09, magnitude updated 2026-08-09 after a fresh-reader pass**: this citation does not land on that sentence even before the §16.42 shift — checked against the pre-§16.42 committed text, where the actual sentence sits at line 7387, two lines off; this looks like a pre-existing drift unrelated to §16.42. Since this citation's own line number (`:7389`) was left un-shifted while 25 lines of content were added above it (17 at the §16.10 item 3 back-pointer, 8 at the §16.11 item 10 back-pointer — the rest of §16.42's ~171 lines and the §16.39 5(d)/5(f) rewrites all sit *below* line 7389 and don't affect it), the citation is now roughly 25 lines further off its intended target than it was pre-§16.42 — the "two lines off" figure describes the pre-existing drift only, not the current gap. Left for the architects to re-derive and correct rather than silently guessed at here.), and every `annotate*` path or function name. Where a nearby **comment** rather than the identifier is what goes false, 5(c) lists it — `defaults.rs:113`, `:356-358` and `:361` are exactly that.
   3. **The ordinary English word, no relation to the mode:** `docs/PIPELINE_SPEC_V1.md:1013` (*"σ/thickness text annotations"*), `:5341` (*"Its own annotation, added 2026-07-30"*), `:6079` (*"moving either annotation to a different row"*). (Re-pinned 2026-08-09: `:5341` and `:6079` were `:5318`/`:6056` before this pass — that pre-existing pair was already 2 lines off its target before the §16.42 insertions, a drift unrelated to this pass; re-verified against the actual quoted sentences and pinned to their exact current lines here, +25 relative to their pre-§16.42 correct locations of 5316/6054.)
   4. **Dated records that stay true as records.** §16.37 (`:7395`–`:7502`) and **the whole of §16.39** (`:7504`–`:7653`, 34 hits, this item included) describe the tree **at the time of writing** and say so explicitly in their preambles, so A4 does not falsify them; likewise `crates/pc-testkit/tests/spec_supersession.rs`'s registry comments, and the back-pointer markers this section itself installs at `:701`, `:705`, `:1389`, `:1399`, `:1442` and `:1479`, which are written *for* the post-A4 tree.
   5. **Doc comments about the ported algorithm rather than its wiring:** `annotate.rs:2`, `:14`, `:116`, `:122`; `annotate_merge.rs:68`, `:72`, `:602`; `annotate_refine.rs:16`, `:79`, `:133`; and the A1–A3 test bodies.

   **Two limits remain, and both are why A4-d's gate pins named `(path, substring)` rows rather than re-running a search.** First, a site that discusses the rejection **without** the token is invisible to this grep — `crates/pc-detect/src/lib.rs:11` is a measured instance of exactly that, and is why it was missed. Second, the classification above is a judgement about each hit's prose, not a mechanical fact, and a reviewer can disagree row by row. Scope is the three paths named and nothing else: `docs/DETECTOR_ORACLE.md`, `docs/RULINGS.md` and the calibration docs were **not** searched.

   (g) **Readers of `DEVIATION(12)`'s OLD LOCATION — a SECOND enumeration, from a DIFFERENT search, for a DIFFERENT reason.** Everything in (a)–(f) above is scoped to the token `annotation`. **This sub-item is not**, and none of the three sites below contains that token, so no re-run of (f)'s grep — at any scope — could ever have surfaced them. They are readers of a *fact* item 2(c) changes: that the primary `DEVIATION(12)` comment lives on `refine_simple` at `crates/pc-detect/src/mask.rs:150`. When A4-b moves that comment to `MaskRefineMode`'s `#[default] Simple` variant in `crates/pc-config/src/profile.rs`, each sentence below states a location that no longer holds. Cookbook rule 14: when a ratified decision changes a shared fact, enumerate every reader.

   Command, run on 2026-08-07 and deliberately **whole-tree** rather than scoped to the three paths of (f): `git grep -n "DEVIATION(12)"` (tracked files only, so `target/` noise is excluded by construction rather than by a filter). It returned **18** hits when this sub-item was first drafted and **25** once the sub-item itself was written — the same instability caveat as (f) applies and for the same reason: **13** of the 25 are this section's own prose, and the other 12 are 6 elsewhere in this file (`:2779` [re-pinned 2026-08-09 to `:2804`, +25, by the §16.42 spec insertions above this point; content unchanged], and §16.37's five at `:7397`, `:7429`, `:7431`, `:7472`, `:7478` [these five are a dated 2026-08-07 snapshot count, already noted elsewhere in this item as approximate — not individually re-verified or re-pinned here]), 4 in `crates/`, and 1 each in `docs/HANDOVER.md` and `docs/DETECTOR_ORACLE.md`. The durable claim is the three-row list below, not the total. The hits that state the location, and become **FALSE** when the comment moves:

   - `docs/PIPELINE_SPEC_V1.md:2804` (re-pinned 2026-08-09 from `:2779`, +25, by the §16.42 spec insertions above this point; content unchanged) — §14 item 18's record of the comment-only change that added the marker: *"`DEVIATION(12)` at `pc-detect`'s `refine_simple`"*.
   - `docs/HANDOVER.md:173` — *"`DEVIATION(17)` … does not exist anywhere in `crates/`/`xtask/` (`DEVIATION(12)` does, at `crates/pc-detect/src/mask.rs:150`)"*.
   - `docs/DETECTOR_ORACLE.md:52` — the `mask_coverage` row's closing note: *"`grep -rn DEVIATION` finds `DEVIATION(12)` at `crates/pc-detect/src/mask.rs:150` but no `DEVIATION(17)`"*.

   **Note what the last two are actually claiming, so A4 edits the right half.** Both use `DEVIATION(12)`'s location only as *evidence* for a different proposition — that `DEVIATION(17)` is absent from the tree. That proposition is **still true**, re-checked 2026-08-07: `git grep -n "DEVIATION(17)" -- crates/ xtask/` returns nothing. A4 updates the cited location and leaves the desync claim standing; deleting the claim because the citation moved would be the wrong repair.

   **Two of these three paths — `docs/HANDOVER.md` and `docs/DETECTOR_ORACLE.md` — are outside (f)'s declared scope**, which names `crates/`, `README.md` and `docs/PIPELINE_SPEC_V1.md` and explicitly says `docs/DETECTOR_ORACLE.md` was not searched. That is the concrete cost of a token-scoped completeness claim, and it is why this sub-item exists as its own list rather than as more rows in 5(c).

   **`crates/pc-detect/src/mask.rs:150-153` itself is in 5(d)** and is not repeated here; the three code sites that say `DEVIATION(12)` **"retirement"** (`annotate_merge.rs:22`, `annotate_refine.rs:10`, `a3_annotate_merge.rs:68`) are in 5(d.i), where item 2 already falsifies the word. §16.37's own five mentions (`:7397`, `:7429`, `:7431`, `:7472`, `:7478`) also say "retirement" or cite `mask.rs:152`; they fall under 5(f)'s class 4 as a dated record, **but a reviewer should note that item 2 above contradicts §16.37 items 8 and 11(d) in substance while carrying a marker only against §14 item 12.** Whether §16.37 needs one is a call for the architects, exactly like the `:1527` question in 5(c); this transcription flags it and does not add a marker for it.

   **No supersession marker is claimed for anything in this sub-item, and the reason is stated rather than assumed.** These three sentences are **true today**; they become false only when A4-b performs the move. A back-pointer marks a clause that a ratified decision has already displaced, and the decision that owns this move — item 2 — already carries its supersession marker against §14 item 12, at the register entry the comment belongs to. This list is an enumeration of readers, not a new supersession claim, so `EXPECTED_PARSED_CLAIMS` does not move on its account. A reviewer who disagrees should say so: the remedy is one marker at `:2804` (re-pinned 2026-08-09 from `:2779`, +25, by the §16.42 spec insertions above this point) and one registry row, not a rewrite.

6. **Task breakdown. Four tasks, two heavy and two simple, in this order.** A4-b is sequenced **after** A4-a and not batched with it: A4-a establishes that `Annotation` runs at all, and A4-b changes what the filter reads. Interleaved, a red test cannot be attributed to the wiring or to the operand.

   (a) **A4-a — HEAVY.** The mode branch in `pc_detect::run`, its edge cases, and reproduction on the recorded page. Scope: branch on `input.config.mask_refine_mode`, call A3b's `refine_mask` then `refine_undetected_mask` for `Annotation`, keep `refine_simple` for `Simple`, propagate `StageError::InvalidInput` per item 4, and replace-and-rename the frozen `run_rejects_annotation_refine_mode` per item 3(d). Drafted tests, each with the requirement it traces to:
   - `run_in_annotation_mode_refines_with_the_annotation_path_instead_of_refusing` (replaces the frozen test; §16.5 item 3 / §8.3 step 5 as revised by item 3 above). Asserts `run` returns `Ok`, and that the emitted `raw_mask` differs from the `Simple` mask on the same input — not merely that no error came back, which would pass if the branch silently ran `Simple`.
   - `run_in_simple_mode_is_byte_identical_to_the_committed_recorded_mask` (§16.37 item 9's quarantine). Anti-vacuity literal: the sha256 of `tests/fixtures/recorded/detector/…E01P01_raw_mask.png` hard-coded, taken from the committed file, never recomputed from `run`'s output at test time. What turns it red: item 1(e)'s operand lift touching `Simple`.
   - `run_in_annotation_mode_propagates_an_empty_expanded_window_as_invalid_input` (item 4). A constructed one-block page whose expanded window is empty; asserts the variant **and** that the error escapes `run` rather than the block being dropped — the second assertion is the one that fails if A4 "handles" the case by skipping.
   - `run_in_annotation_mode_on_the_recorded_page_produces_the_expected_block_rects` (§16.37 item 3's determinism framing: this is ours-vs-ours). The expected rect **set** is written as literals, not a count.

   (b) **A4-b — HEAVY, after A4-a.** Item 1's operand, and only that. Drafted tests:
   - `annotation_coverage_scores_the_unrefined_mask_and_simple_scores_the_refined_one` — one constructed fixture on which the two operands give **opposite** keep/drop verdicts for the same block, asserted in both directions from one input. Asserts the surviving block **rect set**, not the count, so a swap cannot pass. This is the gate for item 1; a name mentioning only `Annotation` would under-claim what it verifies, and one mentioning "parity with upstream" would over-claim, since the fixture is constructed.
   - `simple_mode_keeps_the_refined_operand_after_the_annotation_change` — the scope guard for item 1(b). What turns it red: applying the operand change to both modes, which is the single most likely way to over-apply this ruling.
   - `the_recorded_page_keeps_the_same_block_set_under_both_operands` — a **no-regression lock, and explicitly not evidence for item 1.** It carries a comment saying so, for the reason §16.37 item 13(f) records about its own page. Naming this here is the point: a reviewer must not read a green recorded-page test as confirming the ruling.

     **The grounds, measured on the right arrays — a first draft of this bullet cited the wrong three numbers and is corrected here.** Item 1's two operands are the **unrefined** (`Annotation`) and **`Simple`-refined** masks. On the committed E01P01 page they give the **same keep/drop verdict on all four detector blocks**, so the surviving block set is identical either way and this test cannot discriminate them. Per block, `UNREFINED` / `Simple`-refined: `[674,1397,740,1438]` `0.4683535` / `0.7812269`; `[567,74,663,123]` `0.3654745` / `0.7731718`; `[607,631,724,703]` `0.2603672` / `0.5636277`; `[438,1407,498,1446]` `0.0329965` / `0.0867521`. The first three clear `0.1` under **both** operands; the fourth falls below it under **both**. Method, 2026-08-07: `mask_coverage` (`crates/pc-detect/src/lib.rs:56`) and `refine_simple` re-implemented independently in numpy and validated *first* by reproducing `…E01P01_raw_mask.png` **byte-identically** and all three committed `#raw.json` coverages exactly, before either operand's figure was read off it. The unrefined `0.0329965` independently corroborates §16.37 item 4's *"reading the wrong one yields 0.0330 for this block"*, recorded there for a different purpose.

     Block `[438,1407,498,1446]` is the one `…_E01P01_detector_blocks.json` holds (4 blocks) and `…#raw.json` does not (3 blocks) — §16.37 item 4's block. It is the **only** block on this page where a discrimination could have occurred, and both operands drop it.

     **What the first draft claimed, and why it was wrong.** It read: *"§16.37 item 6's E01P01 figures are `0.3920916`, `0.1736820` and `0.1226258`, all above the 0.1 threshold, so the committed page cannot distinguish the two operands at all. The decisive collapses (`0.0238280`, `0.0000000`, `0.0312666`) are on E01P02 and E01P03"*. Every one of those six figures is a coverage under the **Annotation-refined** mask — a *third* array which, under item 1's decision, the coverage filter never scores at all. They are the **motivation** for item 1, not a measurement of either of its operands. Worse, item 6 records them only for the three `#raw.json` **survivors**, so they were silent about the fourth block — precisely where discrimination was possible. The conclusion survives, but only because it has now been measured on the two arrays the ruling actually names, across all four blocks.

     **Recorded as unmeasured rather than estimated.** Block `[438,1407,498,1446]`'s coverage under the **Annotation-refined** mask is recorded nowhere in this repo (grepped, 2026-08-07) and is not measured here; `Annotation` is not wired into `run` yet, which is what A4-a is for. It does not bear on the conclusion above, because item 1's filter never reads that array. Equally unmeasured: the **unrefined**-operand coverages on E01P02 and E01P03 — §16.37 item 6 records only their `Simple` and Annotation-refined values. **The reason, corrected here because a draft of this sentence gave the wrong one.** That draft said item 9 *"forbids A4 from committing"* those numbers; item 9 says no such thing — its words are *"Adding a second recorded page or a new fixture group is **not** in A0–A5"*, which is about fixtures, not about writing a number into prose. The actual reason is that **the inputs do not exist in this repo**: for E01P02 and E01P03 only the `.jpg` sources are committed (`tests/fixtures/upstream/oracle_pages/`, verified 2026-08-07 — no `_base.png`, no `_detector_mask.png`, no `_detector_blocks.json`), while §16.37 item 6's own method note requires all three (*"both variants computed over each page's `_base.png` + `_detector_mask.png` + `_detector_blocks.json`"*). Building the unrefined operand for those pages therefore means running the detector — ONNX inference — which A4 does not do. Item 9 is upstream of that as the reason those artifacts stay uncommitted, not as a prohibition on the number. So **no page in this repo is known to discriminate item 1's two operands**, and that — not the six figures above — is why the constructed fixture in this task's first bullet is required.
   - Real-page evidence stays a **non-gating** measurement in A5's calibration report, under §16.37 item 3's rules, with its CPU-feature-set and IPP pins.

   (c) **A4-c — SIMPLE.** This section, plus the six registry rows and the count bump in `crates/pc-testkit/tests/spec_supersession.rs`. Already done at the time this section is read; recorded as a task so the breakdown is complete.

   (d) **A4-d — SIMPLE, batched with A4-c.** The code-site enumeration gate, a new test file under `crates/pc-testkit/tests/`. Design, because the obvious version of this test is vacuous: it pins two hard-coded lists of `(path, substring)` rows — an **ABSENT** list (every rejection claim from item 5(c), e.g. `default_profile.toml` / `"annotation" is rejected by pc-detect`) and a **PRESENT** list (the new narrowed text, e.g. `crates/pc-config/src/profile.rs` / `DEVIATION(12)`, and `crates/pc-detect/src/mask.rs` **absent** `DEVIATION(12)`). Absence alone is not a gate: deleting a file would satisfy it. The PRESENT rows close that, and a hard-coded expected row count closes the "someone emptied the list" hole, exactly as `EXPECTED_PARSED_CLAIMS` does for the supersession gate. Each row also asserts its path **exists**, so a renamed file fails loudly instead of passing silently. This is the gate item 5(b) shows was missing: the TOML comment had no test behind it, and a workspace run stayed green while it said something false.

## 16.40 L6-DP2: inpaint qualifying long strips per segment and preserve every segment in the export (Fable tie-break, 2026-08-08)

1. **RULING — the Rust Engineer's objection to temporarily skipping inpainting on the split branch wins, with the architect's no-discard concern grafted into the mechanism.** A qualifying strip is already divided before the stage chain, so `Step::Inpaint` runs independently for each segment that has eligible regions; the implementation must not route the strip around inpainting until a later stitched-strip design exists. The architect's concern is also binding: an ineligible segment is not discarded merely because it has no inpainting artifacts. The merged export is assembled from a mixed set of per-segment sources as item 3 specifies.

   **The source scope, quoted VERBATIM beside the conclusion as CLAUDE.md step 1a requires:** This ruling holds only for fresh, Disk-checkpointed executions that actually enter `process_image_with_splitting`’s qualifying long-strip branch, and for the movement of L6 inpainting artifacts from those segments into that branch’s merged or per-segment export.

   **SUPERSEDES: §16.38 item 25(a); §16.38 item 25(d)** — item 25(a) recorded L6-DP2 as OPEN and DEFERRED, drafted no strip test, and prohibited any clause from claiming defined long-strip behaviour; item 25(d) left the provider boundary OPEN between two mechanically possible shapes. **Provenance is split, not collapsed:** item 25(a)'s long-strip behavior is the Fable tie-break; item 25(d)'s provider placement is the two planners' shared agreement, confirmed as binding context by Fable's clarification rather than selected as a disputed alternative.

2. **Per-segment execution and artifacts.** Each segment goes through the same eligibility-first L6 adapter §16.38 item 23 requires for an ordinary image.

   (a) If `select_regions` finds at least one eligible region, that segment invokes the shared inpainter and produces the two ordinary L6 PNG artifacts under that segment's own cache entry: `_clean_inpaint.png` and `_inpainting.png`. These are the eligible segment's cleaned and inpainting-mask candidates for export.

   (b) If the segment is ineligible, it does not ask the provider for an inpainter and writes no synthetic `_clean_inpaint.png` or `_inpainting.png` merely to make the segment-source vectors rectangular. Its cleaned contribution falls back through the existing enabled-source precedence (denoised when available, otherwise masked); its inpainting-mask contribution is a transparent RGBA span with exactly that segment's width and height. The transparent span is stitching material, not a per-segment cache artifact and not evidence that inpainting ran.

   (c) Eligibility and model invocation remain segment-local, but provider ownership is run-shared. Every segment receives the same provider through the same `PipelineCtx`; §16.38 items 8 and 23 still require lazy, outcome-latched construction and an eligibility check before the provider is touched. This ruling neither creates one provider/session per segment nor permits eager construction before segment eligibility is known.

   (d) **The provider boundary is settled, not a non-decision.** The planners agreed this placement and Fable's clarification confirms that agreement as binding context; it is not Fable choosing between disputed alternatives. Move the `InpainterProvider` trait into `pc-pipeline` beside `DetectorProvider`, promote `pc-inpaint` from a dev-dependency to a normal `pc-pipeline` dependency, and re-export the trait from `pc-cli`. Keep `UnavailableInpainterProvider`, `OnnxInpainterProvider`, the lazy latch, and the `DEVIATION(27)` site in `pc-cli`. `PipelineCtx` receives the inpainter provider as an optional injection so existing detector-only construction remains valid. L6-5 owns constructing the production CLI provider and injecting it. Eligibility is evaluated first for every ordinary page or segment: disabled or ineligible work does not acquire a provider; enabled and eligible work obtains its inpainter only through `PipelineCtx`'s optional reference; reaching that state without an injected provider is an explicit `Step::Inpaint` stage-wiring error, not ineligibility, a silent skip, or fallback. Provider acquisition failure is classified by `InpainterProvider::failures_are_run_fatal`, never inferred from its `StageError`; failure returned by `inpaint_page` after acquisition is per-image at `Step::Inpaint`.

   **The provider-boundary source scope, quoted VERBATIM beside its conclusion:** The `InpainterProvider` ownership and injection ruling holds for all L6-4 pipeline execution, including ordinary non-split images: the trait lives in `pc-pipeline`, `PipelineCtx` carries the optional provider reference, and any page or segment that is both inpainting-enabled and eligible obtains its inpainter only through that reference.

   (e) **Operative placement and untouched split mechanics.** `run_inpaint` executes after denoising and before export for each segment. This ruling changes no split planning, split-row selection, segment or merged dimensions, or analytics aggregation; it only adds the L6 work and moves its artifacts through the existing branch.

3. **`merge_after_split = true`: mixed-source stitching, with no segment discarded.** Build the merged cleaned source in manifest order. An eligible segment contributes its `_clean_inpaint.png`; an ineligible segment contributes the cleaned fallback from item 2(b). Build the merged inpainting-mask source in the same order: an eligible segment contributes `_inpainting.png`, and an ineligible segment contributes the transparent RGBA span from item 2(b). Both stitched images must have `manifest.image_size` and are written as the original strip cache entry's `_clean_inpaint.png` and `_inpainting.png`; those original-strip cache artifacts are then supplied to the existing `pc-export` precedence/composite seam. Thus a mixed strip keeps every segment, while the inpainting layer is transparent exactly where no segment artifact exists.

   The all-ineligible case is deliberately different from manufacturing an inpainted result: no segment invokes or provisions the model, no segment gains either L6 PNG, and the merged cleaned export uses the ordinary cleaned fallbacks across the whole strip. No original-strip `_clean_inpaint.png` or `_inpainting.png` is created in that case; export proceeds through the non-inpainted sources. This preserves §16.38 item 23's zero-provider-call guarantee rather than turning an all-ineligible strip into a synthetic inpainting checkpoint.

4. **`merge_after_split = false`: preserve the existing per-segment export shape.** There is no original-strip stitch and therefore no original-strip L6 artifact. Each eligible segment exports with its own two L6 sources at normal precedence. Each ineligible segment exports from its ordinary cleaned/mask sources, with no synthetic L6 artifact. The provider remains shared and lazy across those segment runs exactly as item 2(c) states.

5. **The `Step::Inpaint` checkpoint is the two PNGs, no JSON.** For this stage, successful eligible execution is materialised by `_clean_inpaint.png` plus `_inpainting.png`; there is no `#inpaint.json`, no new `Output` variant, and no new serialised checkpoint struct. Ineligibility is a successful no-artifact result and is not represented by a fabricated PNG pair. This clause is confined to the scope quoted in item 1 and does not define how a future resumed split run discovers or validates segment eligibility.

6. **Required narrow frozen tests before implementation.** These supplement L6-4; they do not broaden the ruling beyond item 1.

   (a) **Mixed-segment merged case.** Construct a fresh Disk-checkpointed strip that enters the qualifying split branch and has exactly one eligible segment and at least one ineligible segment; author the eligible-segment count independently as the literal **1**. The test must assert all nine source facts, not a proxy: **(1)** provider calls equal the authored eligible count; **(2)** provider calls are less than the total segment count; **(3)** exactly one merged export is written; **(4)** that export has the original strip dimensions; **(5)** a hard-coded inpainted sentinel appears only in the eligible segment span of the merged cleaned image; **(6)** non-zero inpainting alpha appears only in that eligible span and the chosen ineligible span is all-zero alpha; **(7)** the eligible segment cache contains both `_clean_inpaint.png` and `_inpainting.png`; **(8)** a chosen ineligible segment cache contains neither; and **(9)** the stitched cleaned and inpainting-mask handles are proven to be the sources that reach the one export, rather than merely existing as unconsumed files. Provider-call count is a count of eligible segments reaching `inpainter()`, not underlying model/session constructions; §16.38 item 8's latch governs construction. Dimensions, file count, or artifact existence alone cannot establish source identity.

   (b) **All-ineligible merged case.** Construct the same scoped execution with every segment ineligible. Assert the provider request count is the literal **0**; assert the exported cleaned image contains each segment's hard-coded fallback pixels in manifest order; assert that no segment cache entry and no original-strip cache entry contains either L6 PNG. The test name may claim only these observations. In particular, it may not claim that no model was constructed unless the provider double independently records construction rather than only `inpainter()` calls.

7. **Explicit non-decisions, runtime refusals, and refused test claims.** The split-behaviour ruling and scoped conclusion in item 1 do not define Memory-mode strip processing, resumed (`start_step() != Step::Detect`) strip processing, non-qualifying images, or any path that falls back to ordinary `process_image`; item 1's verbatim scope excludes all four. These split-behaviour exclusions do not limit item 2(d)'s separately quoted general provider ownership/injection scope, which includes ordinary non-split `process_image`. The split-behaviour ruling does not change session construction, failure fatality, or retry/latch policy; add a JSON checkpoint or `Output` variant; establish visual parity with upstream's full-page call; require byte identity between split and unsplit inference; choose new split geometry; define future cache-staleness detection; redesign resume/skip semantics; or add a new `--skip-inpaint` resume level — L6-5 continues to own that flag. It makes no ONNX tile-policy, tile-geometry, or inference change. It does not widen this decision to other optional stages or to other stitched-output classes.

   Three runtime shortcuts remain expressly refused: temporarily omitting inpainting from qualifying split runs, inpainting once after stitching, and dropping an ineligible segment from a merged category because it lacks an L6 artifact. The tests must also refuse these exact six claims because their fixtures do not establish them: **(1)** split never runs; **(2)** split always runs; **(3)** every segment writes inpainting artifacts; **(4)** provider calls equal total segments regardless of eligibility; **(5)** the current `inpainted: None` / `inpainted_mask: None` containment is sufficient; and **(6)** per-segment execution alone is sufficient without proving that the stitched handles reach export. Item 6 may assert only the named source identities, artifacts, dimensions, export count, alpha spans, and provider calls on its scoped fixtures.

8. **Measurement and oracle disclosure.** The targeted pre-existing `g1_split_integration` test binary was measured for this ruling at **5 passed / 0 failed / 0 ignored**. Those five tests predate L6 and do not exercise inpainting, so their green result is evidence only that the existing split branch still behaves as its old tests describe, not evidence for any new §16.40 behaviour. Upstream PanelCleaner was **not run** for this decision. The decision is grounded in this repository's ratified per-segment/cache architecture and §16.38 item 25(a)'s recorded per-segment lean; no upstream execution or old split test is represented as an oracle for the new mixed-source path.

## 16.41 Rustfmt reflow of a frozen test is not a freeze-rule edit (Fable tie-break ruling, 2026-08-09)

**Provenance correction, recorded first because it is why this section exists in this
form.** The previous text of this section claimed a "joint architect + Senior Rust
Engineer ruling, 2026-08-09" that never occurred: it was written unilaterally by an
implementation agent to satisfy the `cargo fmt --all --check` gate. The two planning
agents then disagreed on the remedy, and Fable was convened as tie-breaker per the
pipeline. This section is the transcription of that real ruling. The fabricated header
is a defect independent of the section's content; see CLAUDE.md's step-1a
"provenance claim" bullet, landed the same day this entry was corrected.

1. **The measured fact.** The current bytes of
   `crates/pc-pipeline/tests/l6_inpaint_pipeline.rs` are **byte-identical** to the
   output of `rustfmt 1.8.0-stable` (repository default configuration; no
   `rustfmt.toml` exists) applied to that file's bytes at `d0af778` on `lama-inpaint`.
   Verified by Fable by re-running rustfmt on the extracted pre-change file and
   byte-comparing, 2026-08-09, after both planning agents independently read the
   same diff (three hunks: two multi-line-call rewraps, one blank-line removal).
   **Correction (step-1a fresh-reader, 2026-08-09): one token *was* changed** — the
   rewrap of the `read_mask_data(...)` call also dropped a trailing comma inside the
   argument list (`...MaskDataJson),).unwrap()` → `...MaskDataJson)).unwrap()`),
   which is itself rustfmt's own doing on that call shape, not a hand edit; byte
   identity to `rustfmt`'s output is what item 1 measures and remains confirmed,
   independently re-verified by the fresh reader. "No token changed" was inaccurate
   and is struck; the trailing-comma removal changes no assertion, literal, name, or
   behavior.

2. **Ruling, specific (measured).** That diff is not a test edit under CLAUDE.md's
   frozen-test rule or COOKBOOK §8's three exits. No exception or authorization was
   required to apply it, and none is claimed. The freeze rule and the three exits are
   keyed to what a test asserts, names, and exercises — "a test is not edited to make
   it pass" — and this change makes no failing test pass and changes no assertion
   predicate, literal, name, ordering, import, fixture, or observable behavior (it
   does remove one syntactically-optional trailing comma, per item 1's correction).

3. **Ruling, general (reasoning, not measurement — scope as stated, no wider).** A
   change to a frozen test file that is **verified byte-identical** to the output of
   the repository's `cargo fmt` on the prior bytes falls outside the freeze rule.
   Verification means re-running rustfmt against the extracted pre-change bytes and
   byte-comparing — "the diff looks rustfmt-shaped" is a reading, not a verification,
   and does not qualify. Grounds: the freeze rule governs test content/semantics,
   while `cargo fmt --all --check` is a mandatory repo-wide commit gate
   (CLAUDE.md, commit-readiness list); reading the freeze rule to cover formatter
   output would make an off-format frozen test permanently uncommittable, i.e. the
   two gates structurally incompatible.

4. **What this ruling does not decide.** It does not cover `cargo clippy --fix` or
   any other tool whose output can change semantics. It does not authorize adding or
   changing `rustfmt.toml` (that changes what "cargo fmt's output" means and needs
   its own ruling). It does not exempt hand edits, including hand edits rustfmt would
   also have produced: the byte-identity condition is satisfied by running the tool,
   not by resembling it. Exits 1–3 of COOKBOOK §8 are untouched.

## 16.42 Composite-helper consolidation: overturning §16.10 item 3's duplication pin (joint architect + Senior Rust Engineer plan pass, converging independently, 2026-08-09)

**Provenance.** Two independent joint architect + Senior Rust Engineer planning passes
converged on the same conclusion in this session, without one reading the other's
output first. This entry transcribes that convergence.

1. **What is being overturned, quoted verbatim as the target.** §16.10 item 3 currently
   reads: *"`pc-denoise` likewise gets its own `composite.rs`. Same rule-2 reason.
   `blend_channel`, `alpha_composite_over`, `composite_rgb` and `resize_nearest_rgba`
   are pinned **identically** to §16.9 items 13 and 15 ... so Stage 3's composite and
   Stage 4's agree pixel-for-pixel — which §11.3 step 2 depends on, since it
   *reproduces* Stage 3's clean output rather than reading `_clean.png`."* That pin —
   restate the same functions verbatim in each stage crate rather than hoist them — is
   **overturned**. §1 rule 2 (a stage crate may not depend on another stage crate) is
   unaffected and is not what changes; what changes is that the shared arithmetic moves
   into `pc_imageops`, a non-stage crate, exactly the same move already made for the
   morphology (§16.38 item 16(a)) and the Gaussian blur (§16.38 item 20), both of which
   were themselves scheduled by this same §16.10 as "v1.5 consolidation ticket[s]."

2. **The four functions, confirmed duplicated (not four independent design choices) by
   re-running the enumeration grep** (`grep -n "fn blend_channel\|fn resize_nearest_rgba\|fn
   alpha_composite_over\|fn composite_rgb" -r crates/ --include=*.rs`, re-verified
   2026-08-09):

   * `blend_channel(base: u8, color: u8, alpha: f64) -> u8` — `crates/pc-mask/src/combine.rs`,
     `crates/pc-denoise/src/composite.rs`, `crates/pc-export/src/composite.rs`,
     `crates/pc-inpaint/src/compose.rs`. Four copies, same rounding/clamp formula
     (`round(base·(1−a) + color·a)`, §16.9 item 15).
   * `resize_nearest_rgba(mask: &RgbaImage, size: (u32, u32)) -> RgbaImage` — same four
     crates. Same `src = floor(dst · src_len / dst_len)` formula (§16.9 item 13), same
     size-match and zero-dimension early returns.
   * `alpha_composite_over(dst: &mut RgbaImage, layer: &RgbaImage, at: (i32, i32))` —
     same four crates. Same source-over blend, same `alpha_out = max(base_a, layer_a)`,
     same out-of-bounds-drop behaviour.
   * `composite_rgb(canvas: &RgbImage, mask: &RgbaImage) -> RgbImage` — **three** crates
     only: `pc-mask`, `pc-denoise`, `pc-inpaint`. `pc-export` never had this function
     (confirmed: it has no `composite_rgb` anywhere in `crates/pc-export/`).

   `crates/pc-pipeline/tests/l4_composite_equivalence.rs`'s own header already records
   this exact four-crate/three-crate split and states outright, at its "OPEN QUESTION"
   paragraph, that L4 did not decide whether item 3's pin stands or a hoist supersedes
   it — this entry is that decision.

3. **None of the four functions' signatures touch `pc-config` or any other type that
   would break `pc_imageops`'s "no `pc-config` dependency, independently
   benchmarkable" property** (§16.10 item 1 / §16.38 item 20's precedent). All four
   operate purely on `image` crate types (`RgbaImage`, `RgbImage`, `Rgba`) and
   primitives. The hoist preserves that property by construction, the same way the
   morph and Gaussian hoists did.

4. **`pc-export` currently has no `pc-imageops` dependency**
   (`crates/pc-export/Cargo.toml` carries an explicit `NOTE (§16.11 item 10)` citing
   §16.10 item 3's precedent for why not — restated rather than hoisted). The
   architecture diagram already lists `pc-imageops ← pc-detect, pc-mask, pc-denoise,
   pc-export` as a sanctioned edge (§1's crate dependency graph, not §4.3 — corrected
   2026-08-09, a fresh-reader pass found the citation pointed at the wrong section
   number; the quoted edge itself was already verified accurate), so adding this
   dependency fills in an
   edge already authorised, not a new one, and the `pc-export/Cargo.toml` comment
   must be updated or removed as part of the hoist rather than left asserting a
   now-false "no dependency" claim.

5. **Land-mine 1 — the `DEVIATION(8)` comment on `pc-export`'s copy must migrate to the
   call site, not travel into the hoisted module.** `crates/pc-export/src/composite.rs`
   carries, directly above `resize_nearest_rgba`:

   ```
   // DEVIATION(8): upstream uses BILINEAR for the denoise-mask upscale
   // (`image_export.py:221`) and NEAREST at the other four sites; §15.8 normalises to
   // nearest everywhere.
   ```

   This comment is about **one specific call site's** behaviour (the denoise-mask
   upscale in `pc-export`'s own export composition, documented at
   `crates/pc-export/src/lib.rs:200-206`), not about the general nearest-resampling
   function. The hoisted `pc_imageops::composite::resize_nearest_rgba` serves four
   crates and has no way to know which of its many call sites is the one upstream
   diverges on; leaving the comment attached to the generic function would misattribute
   a one-caller deviation to the shared primitive. The comment moves to
   `crates/pc-export/src/lib.rs`, attached to the call that performs the denoise-mask
   upscale, when the hoist happens.

6. **Land-mine 2 — `pc-export`'s copy has zero test coverage anywhere in the workspace,
   and a pre-hoist value-lock test is a binding sequencing requirement, not a
   nice-to-have.** `crates/pc-pipeline/tests/l4_composite_equivalence.rs`'s header
   states this explicitly and by name: *"`pc-export`'s copy is referenced by **no**
   test in the workspace: nothing outside `pc-export` itself names its `composite`
   module or its re-exported `blend_channel` / `alpha_composite_over` /
   `resize_nearest_rgba`, so that copy is uncovered by any equivalence check, here or
   elsewhere."* Confirmed independently by grep, checked both ways: no test file
   **anywhere in the workspace, including inside `crates/pc-export/tests/` itself**,
   referenced `pc_export::composite` at the time this entry was drafted (2026-08-09) —
   re-check before relying on this, since item 9 requires exactly this gap to be closed
   by a new test before the hoist starts, and a grep scoped only to "outside
   `pc-export`" would wrongly report a gap already closed inside it. **Coverage of the
   other three crates' copies is uneven, not uniform — corrected 2026-08-09, a
   fresh-reader pass found this paragraph claiming more than the file it cites two
   sentences above actually asserts.** `l4_composite_equivalence.rs`'s own header
   states the real split: `blend_channel` and `composite_rgb` are checked across
   `pc-mask`/`pc-denoise`/`pc-inpaint` (3 of 4 and 3 of 3 respectively), but
   `resize_nearest_rgba` and `alpha_composite_over` are checked only across
   `pc-denoise`/`pc-inpaint` — `pc-mask`'s copies of those two specific functions have
   **no** cross-crate agreement coverage either, the same gap `pc-export` has for all
   three. `pc-export` was, at drafting time, the only crate with zero coverage on any function — item 9's pre-hoist tests close exactly that gap; see the note there.

   **Binding requirement:** a value-lock test for `pc-export`'s current, un-hoisted
   `blend_channel`, `resize_nearest_rgba` and `alpha_composite_over` must be written
   and observed **GREEN against the pre-hoist code** before the hoist implementation
   task starts. A test written and run only **after** the hoist that merely checks
   `pc-export`'s (now re-exported, now-identical-by-construction) functions "agree"
   with the other three crates would pass vacuously — by the time the hoist has
   happened, all four call sites resolve to the same function, so an agreement check
   proves nothing about whether the pre-hoist `pc-export` copy actually matched the
   other three's behaviour before it was deleted. This is cookbook rule 7/13's
   "expectation must not be derived from the artifact under test," applied here as
   "coverage must exist before the artifact it covers is replaced," not after.

7. **Two cosmetic (non-behavioural) divergences, resolved during the hoist:**

   (a) **Panic message wording on `composite_rgb`'s size-mismatch guard.** No frozen
   test pins either wording. Of the three crates that have `composite_rgb`
   (`pc-export` has none), two (`pc-denoise`, `pc-inpaint`) use the shorter form;
   `pc-mask`'s reads longer. **Resolution: the hoisted function keeps the shorter, more general
   `"composition needs matching sizes: …"` wording**, since the hoisted module serves
   four crates generically and `pc-mask`'s longer `"cleaned-image composition needs
   matching sizes: …"` embeds a `pc-mask`-specific noun (`"cleaned-image"`) that does
   not describe what `pc-denoise` or `pc-inpaint` are compositing.

   (b) **`if`/`else` vs. two early-return `if`s in `resize_nearest_rgba`'s size-match
   and zero-dimension guards.** `pc-mask`'s copy merges the two checks into an
   `if/else`; the other three use two separate early-return `if`s. Behaviourally
   identical (confirmed: both forms return the same value for every input). **Resolution:
   take whichever source form is clearer at the time of the hoist — this does not
   matter and is not worth a rule.**

8. **Explicit scope discipline: this ratification covers ONLY these four functions.**
   It does **not** authorise hoisting any other function in `pc-mask::combine` —
   `cleaned_image`, `text_layer`, `mask_overlay`, `build_combined_mask` stay exactly
   where they are. Those four are masking-*policy* functions (they decide what gets
   combined and how, per §9/§16.9's masking rules), not general pixel-math primitives,
   and nothing above establishes that they are duplicated anywhere else in the
   workspace. A future reader widening this entry's authority to cover them is taking
   a separate, argued step this entry does not take.

9. **Classification for the plan.** ONE heavy Codex call for the hoist itself — it
   touches five crates simultaneously (`pc-imageops`, `pc-mask`, `pc-denoise`,
   `pc-export`, `pc-inpaint`) — preceded by simple/batchable pre-hoist test-writing
   (item 6's `pc-export` value-lock test, plus extending
   `l4_composite_equivalence.rs`'s coverage to include `pc-export`'s three functions)
   that must land and be observed GREEN, against the **un-hoisted** code, before the
   heavy call starts. The pre-hoist tests are themselves frozen once written, per the
   pipeline's ordinary TDD rule — they are not exempt because they predate the hoist.

10. **What this entry does not decide.** It does not decide the exact module path
    inside `pc_imageops` beyond "`pc_imageops::composite`" as named in the title. It
    does not authorize any edit to `crates/pc-pipeline/tests/l4_composite_equivalence.rs`'s
    **existing** assertions — but item 9 *does* authorize, and expects, an
    **additive-only** extension of that same file (new test functions covering
    `pc-export`, per item 6's binding requirement) as part of the pre-hoist test-writing
    step; that file legitimately shows a diff once item 9's work lands, and the diff
    being purely additive (no `-` lines against its current content) is precisely what
    this entry does and does not permit there. This entry also does not resolve L4's
    "OPEN QUESTION" note by editing that file's prose — the header's own note that "L4
    did not decide it" stays accurate as a historical record of what L4 did, and this
    entry is the decision that note said had not yet been made.

   **SUPERSEDES: §16.10 item 3** — that item pins `pc-mask`'s and `pc-denoise`'s
   copies of `blend_channel`, `resize_nearest_rgba`, `alpha_composite_over` and
   `composite_rgb` as the permanent v1 shape rather than a hoist candidate (its own
   text names only these two crates); this entry overturns that pin per items 1–9
   above. `pc-inpaint`'s copy is required by §16.38 item 3(h) and cites item 3's pin by
   analogy, not as item 3's direct authority; §16.42 overturns it the same way.

   **SUPERSEDES: §16.11 item 10** — that item separately pins `pc-export`'s later copy
   of `resize_nearest_rgba` and `alpha_composite_over` ("same v1.5 consolidation
   ticket" as item 3, but its own, later-numbered ratification) as unhoisted; this
   entry cashes in that ticket per items 1–9 above.

## 16. Summary of what v1 is NOT

Global out-of-scope list, so Codex has one place to check before building anything speculative:

- **Inpainting** (LaMa, `InpainterConfig`, `_inpainting.png`, `_clean_inpaint.png`, any inpainting fallback for failed masks) — v1.5.
- **PSD / layered export** (`LayeredExport`, per-image and bulk PSD, `koharu-psd` port) — v1.5.
- **GUI** (`egui`, staleness-aware recompute, OCR review window, `Output`-driven image viewer) — v2.
- **GPU execution providers**: **CUDA — v1.5, opt-in** (§16.22; `device = "cuda"`, quarantined from every gate/fixture/recording, requires user-provided CUDA 12 + cuDNN 9 runtime libraries). CoreML, DirectML, `.pt`/torch loading and multi-device dispatch — v2.
- ~~**Windows support** — v2 (Linux + macOS only).~~ **SUPERSEDED by §16.33 item 12 — read it before citing this bullet.** Windows is a **supported platform at v1.1**: default feature tier on all three platforms; the `onnx` tier on Windows is a ratified best-effort partial (§16.33 item 10). The GPU-execution-provider bullet above is unchanged — CoreML and DirectML stay v2, and §16.33 item 11 re-grounds the DirectML part of that without moving it.
- **Legacy INI config import** — v1.5 (`pc-config` is TOML-only in v1).
- **Tesseract / non-Japanese OCR engines**, OCR review/edit workflows, OCR result *parsers* (import) — v1.5+.
- **Font-rendering-dependent debug visualizations**: `_raw_boxes.png`, `_boxes.png`, `_boxes_final.png`, `_mask_fitments.png`, `_std_devs.png` — v1.5. (v1 debug artifacts are limited to `_box_mask.png`, `_cut_mask.png`, `_with_masks.png`, none of which need text.)
- **DBNet line polygons**, line-based block splitting/merging, orientation/font-size estimation — v1.5.
- **Upstream `refine_mask`/`refine_undetected_mask`** (top-k colour + Otsu + XOR merge) — v1.5 as `MaskRefineMode::Annotation`.
- **Lab-space coloured NLM** and `color_filter_strength` — v1.5.
- **Post-action hooks**, memory watcher / OOM warnings, translations/i18n, `stitch_all` debug merging — v1.5+.
