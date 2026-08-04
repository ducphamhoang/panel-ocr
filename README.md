# panel-ocr

A from-scratch Rust rewrite of [PanelCleaner](https://github.com/VoxelCubes/PanelCleaner)
(GPL-3): detect text in comic/manga panels, mask it out, denoise the result, and export
clean pages — no Python at runtime. [koharu](https://github.com/mayocream/koharu) is used
purely as architectural inspiration, not a dependency; panel-ocr stands on its own.

Licensed **GPL-3.0-only**, same as PanelCleaner.

## Status: v1 complete

Every v1 pipeline stage is implemented, tested, and wired into a working CLI:

```
panel-ocr clean <images...> [--detector mock|replay:<dir>|onnx] [options...]
panel-ocr ocr <images...>
panel-ocr profile ...
panel-ocr cache ...
panel-ocr models ...
```

~700 tests passing across the workspace, `clippy --all-features -D warnings` and
`cargo fmt --check` both clean.

**One real gap today**: there's no bundled ONNX text-detector model yet, so `--detector
onnx` fails fast with a clear message rather than silently doing nothing. Use `--detector
mock` (produces no detections, useful for exercising the rest of the pipeline) or
`--detector replay:<dir>` (replays a previously recorded detection) until a model-download
path lands. See [Roadmap](#roadmap).

## Architecture

A stage-based pipeline, not a feature-first codebase — closer to an ETL pattern. Each
stage is a pure function behind a shared `Stage` trait, with typed input/output contracts
and disk-checkpointing between stages for batch/resume support:

```
detect → preprocess → mask → denoise → export
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
| `pc-export` | Stage 5 — format coercion, compositing, OCR reports |
| `pc-pipeline` + `pc-cli` | Orchestrator + the `panel-ocr` binary |
| `xtask` | Fixture recording (`cargo xtask record-fixtures`) and golden calibration (`cargo xtask calibrate-goldens`) |

The full design spec — including every algorithm, data contract, and resolved ambiguity
— lives in [`docs/PIPELINE_SPEC_V1.md`](docs/PIPELINE_SPEC_V1.md). Architecture rationale
(why Rust, why no Python at runtime, per-component tooling choices) is in
[`docs/ARCHITECTURE_DECISIONS.md`](docs/ARCHITECTURE_DECISIONS.md). Measured parity
numbers against real OpenCV/PIL references are in
[`docs/GOLDEN_CALIBRATION.md`](docs/GOLDEN_CALIBRATION.md).

## Building

```
cargo build --workspace
cargo test --workspace
```

Requires Rust 1.94.1+ (pinned via `rust-toolchain.toml`). CPU-only. Supported platforms: Linux,
macOS and Windows — Windows landed in v1.1 for the default feature tier; v1 shipped Linux + macOS
only, and the optional `onnx` tier on Windows is still deferred (§16.33).

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

### Near-term — completing v1's promise
- [ ] **Real ONNX text detector** (`pc-models` for model provisioning/download, `pc-detect`'s
      `ort`-backed inference session). This is the one piece of v1 that exists as a
      well-defined contract (`TextDetector` trait, `--detector onnx` flag already wired)
      but has no real backing implementation yet — `--detector mock`/`replay:<dir>` are
      the working substitutes today.
  - [ ] Fixture recording for the detector boundary once real weights exist
        (`cargo xtask record-fixtures --only detector`), unblocking the 4 tests
        currently `#[ignore]`d pending this.
  - [ ] GPU execution provider as an optional build feature (CPU-only is the v1 baseline).

### v1.5
- [ ] LaMa inpainting (fill masked regions with generated content instead of a flat color)
- [ ] PSD / layered export

### v2
- [ ] `egui`-based GUI, with incremental/staleness-aware recompute

## Contributing

This project was built test-first: every stage has a frozen test suite tied to a spec
section, and `docs/PIPELINE_SPEC_V1.md`'s `§16.x` subsections are the running log of every
design decision and deviation from upstream PanelCleaner, with rationale. Read the spec
section for the area you're touching before changing behavior — most "obvious" choices
(rounding rules, rect inclusivity, float precision) were already litigated there.
