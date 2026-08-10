# panel-ocr

A from-scratch Rust rewrite of [PanelCleaner](https://github.com/VoxelCubes/PanelCleaner)
(GPL-3): detect text in comic/manga panels, mask it out, denoise/inpaint the result, and
export clean pages — no Python at runtime. [koharu](https://github.com/mayocream/koharu) is
used purely as architectural inspiration, not a dependency; panel-ocr stands on its own.

Licensed **GPL-3.0-only**, same as PanelCleaner.

## What it does

- **Detects** the text bubbles/SFX on a manga/comic page (real ONNX text detector, not a mock).
- **Cleans** them: masks the text out, then either flat-fills or **inpaints** the hole with
  generated content (LaMa) so the result looks like blank paper instead of a colored patch.
- **Extracts text** as an isolated layer and/or runs OCR over it, producing a CSV/TXT report.
- Batches over folders of images, resumes/caches between stages, and isolates per-image
  failures so one bad page doesn't kill a run.

## Installing

**Download a prebuilt binary** from the
[Releases page](https://github.com/ducphamhoang/panel-ocr/releases) — Linux, macOS
(Intel + Apple Silicon), and Windows archives are built and published for every tagged
release, each bundling the `panel-ocr` binary alongside this README and the license.
Unpack the archive and run the binary directly; no installer or runtime dependencies needed.

**Or build from source:**

```
cargo build --release --features pc-cli/onnx -p pc-cli
```

Requires Rust 1.94.1+ (pinned via `rust-toolchain.toml`). CPU-only — no GPU acceleration yet
(see [Roadmap](#roadmap)). The `onnx` feature is required for real detection/OCR/inpainting;
without it the binary still builds and runs, but only against `mock`/`replay` detector
backends useful for pipeline testing, not real pages. On Windows, the `onnx` feature's build
is not yet verified in CI (§16.33) — the default (non-`onnx`) tier is fully supported there.

## Quick start

Fetch the model weights once (into a per-user cache directory, not the repo):

```
panel-ocr models download            # required detector + OCR models (~90 MB)
panel-ocr models download --include-optional   # + LaMa inpainting weights (~207 MB)
```

Clean a folder of pages, writing output next to each input under `cleaned/`:

```
panel-ocr clean my-manga-folder/
```

Useful flags for `clean` (run `panel-ocr clean --help` for the full list):

| Flag | Effect |
|---|---|
| `--output-dir <DIR>` | Where cleaned images go (default: `cleaned`, next to each input) |
| `--extract-text` | Also write the isolated text layer (`_text.png`) |
| `--skip-inpaint` | Disable inpainting even if the profile has it enabled |
| `--skip-denoise` | Disable denoising entirely |
| `--profile <NAME>` / `--profile-path <FILE>` | Use a specific profile instead of the default |
| `--threads <N>` | Worker thread count |
| `-v` / `-vv` / `-vvv` | Increase log verbosity |

Run OCR and write a report:

```
panel-ocr ocr my-manga-folder/ --format csv --output report.csv
```

### Getting the best cleaning quality

The shipped defaults are deliberately conservative (a documented divergence from
upstream PanelCleaner, DEVIATION(12)). For the highest-accuracy bubble cleaning, opt into
both of the following in your profile (`panel-ocr profile new my-profile.toml` to get a
starting file, then edit it and pass `--profile-path`):

```toml
[text_detector]
mask_refine_mode = "annotation"   # default is "simple"; annotation matches upstream's mask refinement

[inpainter]
inpainting_enabled = true         # default is false; fills bubbles instead of flat-color masking
```

`panel-ocr profile show` prints the full default profile as TOML so you can see every
tunable option and its default value.

## Architecture

A stage-based pipeline, not a feature-first codebase — closer to an ETL pattern. Each
stage is a pure function behind a shared `Stage` trait, with typed input/output contracts
and disk-checkpointing between stages for batch/resume support:

```
detect → preprocess → mask → denoise → inpaint (optional) → export
```

| Crate | Role |
|---|---|
| `pc-core` | Shared types: `Rect`, `ImageHandle`, page data, the `Stage` trait, error types |
| `pc-config` | TOML profile/config, format-preserving via `toml_edit` |
| `pc-testkit` | Dev-only: fixture paths, image metrics (SSIM/IoU/diff), golden comparison |
| `pc-detect` + `pc-imageops` | Stage 1 — text detection (YOLO postprocess, resize/rescale, mask refinement, long-strip splitting) |
| `pc-preprocess` + `pc-ocr` | Stage 2 — box filtering, merging, reading order, OCR pass |
| `pc-mask` | Stage 3 — growth/dilation, border-color fitting, mask composition |
| `pc-denoise` | Stage 4 — non-local-means denoising (from-scratch, exact) |
| `pc-inpaint` | Stage 4b — LaMa-based inpainting for masked regions (opt-in) |
| `pc-export` | Stage 5 — format coercion, compositing, OCR reports |
| `pc-pipeline` + `pc-cli` | Orchestrator + the `panel-ocr` binary |
| `xtask` | Fixture recording (`cargo xtask record-fixtures`), golden calibration (`cargo xtask calibrate-goldens`), and the Simple/Annotation/LaMa mode-comparison benchmark (`cargo xtask mode-bench`) |

The full design spec — including every algorithm, data contract, and resolved ambiguity
— lives in [`docs/PIPELINE_SPEC_V1.md`](docs/PIPELINE_SPEC_V1.md). Architecture rationale
(why Rust, why no Python at runtime, per-component tooling choices) is in
[`docs/ARCHITECTURE_DECISIONS.md`](docs/ARCHITECTURE_DECISIONS.md). Measured parity
numbers against real OpenCV/PIL references are in
[`docs/GOLDEN_CALIBRATION.md`](docs/GOLDEN_CALIBRATION.md). Mask-mode/inpainting
quality/timing comparisons are in [`docs/MODE_COMPARISON.md`](docs/MODE_COMPARISON.md).

## Building from source

```
cargo build --workspace
cargo test --workspace
```

Supported platforms: Linux, macOS and Windows for the default feature tier. The optional
`onnx` tier (real detection/OCR/inpainting) is verified in CI on Linux and macOS; on
Windows it builds but is not yet CI-verified (§16.33 item 10).

## Roadmap

### v1 — done
- [x] Text detection (YOLO-based, with resize/rescale, mask refinement, long-strip split)
- [x] Preprocessing (box filtering/merging, reading order, optional OCR pass)
- [x] Masking (growth, border-color fitting, composition)
- [x] Denoising (from-scratch non-local-means)
- [x] Export (multi-format, OCR reports)
- [x] Pipeline orchestrator + CLI, batch processing, checkpointing, per-image error isolation
- [x] Fixture recording + golden calibration tooling, with real OpenCV/PIL parity gates
      wherever the build environment allows

### v1.1 — done
- [x] Windows support for the default feature tier — config/cache root discovery, shell quoting and
      editor launching (§16.33). The optional `onnx` tier is not yet attempted on Windows CI; that
      partial is deferred by §16.33 item 10, not claimed as working.

### v1.2 — done
- [x] `Annotation` mask-refine mode — opt-in parity with upstream PanelCleaner's mask
      refinement (`mask_refine_mode = "annotation"`; `Simple` stays the default, DEVIATION(12))
- [x] LaMa inpainting — fills masked regions with generated content instead of a flat
      color, wired into the pipeline end-to-end (opt-in via `inpainting_enabled = true`,
      or force it off per-run with `--skip-inpaint`)
- [x] Fixed a real compositing defect (§16.45): partial-alpha regions (denoise/mask rims
      composited onto a transparent base) were darkening visibly instead of blending
      correctly — found via visual inspection of a real page, root-caused to
      `alpha_composite_over` not implementing real source-over, fixed and reviewed
- [x] `cargo xtask mode-bench` — a pinned, non-gating Simple/Annotation/LaMa comparison
      tool, with a real committed report in `docs/MODE_COMPARISON.md`

### v1.5 — not started
- [ ] GPU-1: device config, policy resolver, fatal-refusal wiring (no CUDA linkage yet)
- [ ] GPU-2: the `cuda` feature itself, opt-in and quarantined from every gate/fixture/
      recording (§16.22) — CPU stays the only execution provider covered by
      determinism guarantees
- [ ] Legacy INI config import + Lab-space non-local-means denoising (batched)
- [ ] PSD / layered export
- [ ] DBNet line-polygon synthesis (highest-risk item in v1.5; forces a second F1
      detector-fixture re-record + re-sign when it lands)

### v2
- [ ] `egui`-based GUI, with incremental/staleness-aware recompute
- [ ] Additional ONNX execution providers (CoreML, DirectML), torch `.pt` model loading,
      multi-device dispatch (§16.22 item 1)

## Contributing

This project was built test-first: every stage has a frozen test suite tied to a spec
section, and `docs/PIPELINE_SPEC_V1.md`'s `§16.x` subsections are the running log of every
design decision and deviation from upstream PanelCleaner, with rationale. Read the spec
section for the area you're touching before changing behavior — most "obvious" choices
(rounding rules, rect inclusivity, float precision) were already litigated there.
