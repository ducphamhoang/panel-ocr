# Architecture Decisions

Captured from the pre-planning research/discussion phase, before the formal
Opus Technical Architecture + Senior Rust Engineer planning pipeline kicks off.
These are settled defaults — revisit only if a spec surfaces a real conflict.

## Project identity

- panel-ocr is a **standalone Rust rewrite** of PanelCleaner
  (https://github.com/VoxelCubes/PanelCleaner) — a tool that detects
  speech/text bubbles in manga/comic panels and cleans (masks/denoises/
  inpaints) them for translation/typesetting workflows.
- **License: GPL-3**, matching both PanelCleaner and koharu.
- **No runtime or build dependency on koharu** (https://github.com/mayocream/koharu).
  koharu is a *translation* tool; panel-ocr is a *cleaning* tool. Different
  product goals, different masking philosophy (koharu dilates a mask to feed
  an inpainter; PanelCleaner/panel-ocr grows a mask and scores its border to
  *avoid* needing an inpainter). We do not integrate with or call into koharu.
- **Code reuse from koharu is allowed and intended**, since both projects are
  GPL-3: we will port/adapt koharu's model-inference code (comic-text-detector,
  manga-ocr, LaMa — all in `crates/koharu-ml`) and its PSD writer
  (`crates/koharu-psd`) as starting points for our own `pc-detect`/`pc-ocr`/
  `pc-inpaint`/`pc-export` crates, rather than reimplementing YOLO postprocess,
  the manga-ocr decode loop, LaMa's FFT port, and PSD writing from scratch.
  Give credit/attribution per GPL-3 requirements when porting.

## Rust vs. Python

**Full Rust, no Python at runtime, no Python sidecar.** Python survives only
as an offline, maintainer-run model-conversion script (torch → ONNX), never
shipped to users. Rationale: koharu already proves all three ML models
(comic-text-detector, manga-ocr, LaMa) run natively in Rust via `ort` and/or
`candle`, on CPU and GPU, so the usual "sidecar for the ML parts" compromise
isn't necessary here.

## Per-component verdicts

| Component | Verdict |
|---|---|
| Text detection (YOLOv5+U-Net+DB) | ONNX-via-Rust (`ort`), backend behind `trait Detector` |
| manga-ocr (Japanese OCR) | ONNX-via-Rust (`ort`) — no Tesseract-quality regression |
| Tesseract (other languages) | Pure Rust via `leptess` FFI |
| LaMa inpainting | ONNX-via-Rust, **v1.5** (deferred, optional/off-by-default like upstream) |
| Mask growth / border-std-dev scoring | Pure Rust (`image`, `imageproc`, `ndarray`, `rayon`) |
| `cv2.fastNlMeansDenoising` equivalent | Reimplement in Rust — no crate provides it |
| PSD layered export | Pure Rust, adapted from `koharu-psd`, **deferred out of v1** |
| Config/profiles | **Format change: INI → TOML** via `toml_edit`; one-way legacy INI importer for migration |
| GUI | Pure Rust (`egui`), **v2 milestone**, not v1 |
| CLI + parallelism | Pure Rust (`clap` + `rayon`) |
| Model download/hashing | Pure Rust (`reqwest`, `sha2`); `hf-hub` for HF-hosted weights |

## Resolved open questions

1. **Config format**: TOML via `toml_edit`, not INI-compatible. The old
   format's comments were code-generated, not user-authored, so nothing of
   real value is lost; `toml_edit` gives genuine round-trip (it's what Cargo
   itself uses).
2. **`ort` version risk**: accept `ort` 2.0.0-rc.12, pinned exact. Pre-1.0 but
   production-used elsewhere (HF Text Embeddings Inference, Google Magika).
   Mitigated by keeping every ML call behind a trait so a `candle` fallback
   (koharu proves it works for these exact models) is a real, not
   hypothetical, escape hatch.
3. **GPU support**: CPU-only ONNX execution provider at v1; CUDA/CoreML/
   DirectView EPs added later as an opt-in build feature. Packaging concern,
   not an architecture blocker.
4. **Denoiser test tolerance**: perceptual/threshold-based (e.g. SSIM or
   max-per-channel-delta), **not bit-exact** — bit-exact parity with OpenCV's
   NLM isn't achievable from an independent reimplementation. Settled now,
   before any tests are drafted, per the "tests are frozen once written" rule.
5. **Inpaint weight provenance**: adopt an existing manga-finetuned
   ONNX/safetensors equivalent (e.g. `mayocream/lama-manga`) rather than
   re-exporting PanelCleaner's exact `anime-manga-big-lama.pt` ourselves, for
   v1.5. Revisit exact-weight parity only if bit-identical output becomes an
   actual requirement later.
6. **CLI compatibility**: fresh, idiomatic `clap` design — no requirement to
   mirror PanelCleaner's `docopt` flag surface. This is a rewrite under a new
   name, not a drop-in replacement.
7. **GUI scope/timing**: CLI-first for v1 (fully usable standalone, as
   PanelCleaner's own CLI is); `egui` GUI is a separate v2 milestone.

## Pipeline architecture pattern

Stage-based (pipeline/ETL-style), not feature-first: each pipeline stage
(detect, preprocess, mask, denoise, inpaint, export) is its own crate exposing
one pure function `fn run(input: StageInput) -> Result<StageOutput>`, with
`serde`-typed `StageInput`/`StageOutput` structs mirroring PanelCleaner's
`PageData`/`MaskData` JSON contracts. The CLI is a thin wrapper that calls
these functions and reads/writes JSON+image sidecar files — same
disk-checkpoint pattern as PanelCleaner (enables skip/resume flags) — but the
in-out contract is enforced by the type system, not by convention across
separate binaries. Most tests operate directly on `StageInput`/`StageOutput`
in-process (fast, no file I/O); a smaller set of end-to-end tests exercise the
actual CLI binary against golden fixture files.

## Milestones

- **v1**: CLI + detect + preprocess + mask + denoise + export (parity core).
- **v1.5**: LaMa inpainting + PSD export.
- **v2**: `egui` GUI with incremental/staleness-aware recompute.
