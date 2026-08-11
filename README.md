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
- Runs on **CPU by default**, or on an **NVIDIA GPU** (CUDA) if you build it that way — see
  [GPU acceleration](#gpu-acceleration) below.

## Installing

**Download a prebuilt binary** from the
[Releases page](https://github.com/ducphamhoang/panel-ocr/releases) — Linux, macOS
(Intel + Apple Silicon), and Windows archives are built and published for every tagged
release, each bundling the `panel-ocr` binary alongside this README and the license.
Unpack the archive and run the binary directly; no installer or runtime dependencies needed.
**The prebuilt binaries are CPU-only** — GPU support requires building from source
yourself (see [GPU acceleration](#gpu-acceleration)); there is no technical reason it
couldn't be added to the release matrix later, it just isn't there yet.

**Or build from source (CPU):**

```
cargo build --release --features pc-cli/onnx -p pc-cli
```

Requires Rust 1.94.1+ (pinned via `rust-toolchain.toml`). The `onnx` feature is required
for real detection/OCR/inpainting; without it the binary still builds and runs, but only
against `mock`/`replay` detector backends useful for pipeline testing, not real pages. On
Windows, the `onnx` feature's build is not yet verified in CI (§16.33) — the default
(non-`onnx`) tier is fully supported there.

## Prerequisites

- **CPU (default):** nothing beyond the Rust toolchain to build, and nothing at all to run
  a downloaded binary. No GPU, no CUDA, no extra runtime libraries.
- **GPU (optional, build-from-source only):** an NVIDIA GPU, the **CUDA 12.x** toolkit, and
  **cuDNN 9** — both need to be installed and discoverable on `PATH` (Windows) or the
  dynamic linker's search path (Linux) at build time *and* run time. See
  [GPU acceleration](#gpu-acceleration) for exact commands and what happens if either is
  missing (a loud, non-zero-exit error — never a silent fallback to CPU).
- **Either way:** the model files (~763 MB total) are downloaded once into a per-user cache
  directory the first time you run `panel-ocr models download` — see
  [Quick start](#quick-start).

## Quick start

Fetch the model weights once (into a per-user cache directory, not the repo):

```
panel-ocr models download            # every required model: detector, OCR encoder and
                                     # decoder, and the LaMa inpainter (~763 MB in total)
```

All of them are required models, the LaMa inpainter included, because inpainting is on by
default. There is nothing optional left to add: `--include-optional` still exists and
currently selects no extra artifact. `panel-ocr models verify` checks what's already cached
without downloading anything; `panel-ocr models path` prints where the cache lives.

Clean a folder of pages, writing output next to each input under `cleaned/`:

```
panel-ocr clean my-manga-folder/
```

Useful flags for `clean` (run `panel-ocr clean --help` for the full list):

| Flag | Effect |
|---|---|
| `--output-dir <DIR>` | Where cleaned images go (default: `cleaned`, next to each input) |
| `--extract-text` | Also write the isolated text layer (`_text.png`) |
| `--skip-inpaint` | Turn inpainting off for this run (it is on by default) |
| `--skip-denoise` | Disable denoising entirely |
| `--profile <NAME>` / `--profile-path <FILE>` | Use a specific profile instead of the default |
| `--threads <N>` | Worker thread count |
| `-v` / `-vv` / `-vvv` | Increase log verbosity |

Run OCR and write a report:

```
panel-ocr ocr my-manga-folder/ --format csv --output report.csv
```

Inpaint a brush mask over a single image directly — no detection, masking, or denoising
re-run, no cache:

```
panel-ocr inpaint page.png --mask brush.png --output fixed.png
```

`--mask` is an RGBA PNG the same pixel dimensions as `IMAGE`; alpha > 0 marks a painted
pixel (RGB is ignored), and disjoint painted blobs become separate inpaint regions. This is
for continuing or fixing a result without re-running the pipeline, not a replacement for
`clean`. This always runs LaMa inpainting regardless of `[inpainter].inpainting_enabled`,
and needs a build with the `onnx` feature and the LaMa model downloaded
(`panel-ocr models download`). The inpainted area extends beyond the drawn brush strokes: at
the default profile the painted silhouette is grown by `min_mask_thickness` + the inpaint
`growth` radius + a fade band (roughly 10 px wider than drawn at shipped defaults), so a UI
sizing its brush or its preview should not assume a pixel-exact result.

### Opting out of the default refinement and inpainting

Out of the box you get upstream PanelCleaner's own mask refinement and LaMa inpainting —
both are **on by default**, so nothing needs configuring to get them. The LaMa model is 207 MB of
that download — fetched whether or not you use it, since it is a required model — and
inpainting is what runs it over every eligible region.

To turn either off, opt **out** in your profile (`panel-ocr profile new my-profile.toml` to
get a starting file, then edit it and pass `--profile-path`):

```toml
[text_detector]
mask_refine_mode = "simple"       # opt out of the default "annotation"; koharu-style
                                  # refinement, a v1 addition upstream has no equivalent of

[inpainter]
inpainting_enabled = false        # opt out of the default LaMa inpainting; masked regions get
                                  # a flat fill instead of generated content
```

This project publishes no measurement ranking the four combinations; `cargo xtask mode-bench`
and [`docs/MODE_COMPARISON.md`](docs/MODE_COMPARISON.md) record what each one produced on the
same inputs and deliberately declare no winner.

`--skip-inpaint` does the second of those for a single run without editing a profile.

`panel-ocr profile show` prints the full default profile as TOML so you can see every
tunable option and its default value.

## GPU acceleration

Every stage that touches an ONNX model — the text detector, OCR, and LaMa inpainting — has
a real, measured CUDA execution path. It is **opt-in and not in the prebuilt binaries**:
you build it yourself.

**1. Install the prerequisites** (see [Prerequisites](#prerequisites)): an NVIDIA GPU, the
CUDA 12.x toolkit, and cuDNN 9. Make sure cuDNN's `bin`/`lib` directory is on `PATH` (Windows)
or your linker's search path (Linux) — the CUDA toolkit installer normally puts itself on
`PATH` already, but cuDNN is usually a separate download you place and path yourself.

**2. Build with the `cuda` feature:**

```
cargo build --release --features cuda -p pc-cli
```

This pulls a prebuilt CUDA execution-provider distribution (~82 MB) at build time — no
GPU needed to *build*, only to *run*.

**3. Set `device = "cuda"` in your profile:**

```toml
[general]
device = "cuda"
```

then run `panel-ocr clean --profile-path your-profile.toml ...` as usual. Everything else
about the command line is identical to the CPU case.

**What happens if CUDA can't actually register** (missing cuDNN, no GPU, driver too old):
a loud, non-zero-exit error naming the problem — **never** a silent fallback to CPU. This
project treats a silent GPU→CPU downgrade as a defect, not a convenience.

**What to actually expect, measured on real pages, not estimated:**
- CUDA execution is **not bit-identical to CPU**, run to run — small, real, non-gating
  numeric differences exist for both OCR and LaMa (documented in
  [`docs/OCR_DEVICE_DIVERGENCE.md`](docs/OCR_DEVICE_DIVERGENCE.md) and
  [`docs/LAMA_DEVICE_DIVERGENCE.md`](docs/LAMA_DEVICE_DIVERGENCE.md)). Decoded text and
  visual output matched on every real page we measured.
- **Single-page runs pay a fixed CUDA context/driver setup cost every invocation** — for one
  page with these model sizes, that fixed cost can make a single-shot GPU run *slower* than
  CPU, especially if the GPU was idle beforehand (driver cold-start adds several extra
  seconds on top).
- **Batches are where GPU wins**: measured on a real 10-page batch, GPU finished in ~12s
  against CPU's ~28s — **roughly 2.3x faster** — because the fixed setup cost is paid once
  for the whole batch, not once per page. LaMa inpainting was the single largest stage on
  both devices (~53–64% of total time), and had the clearest GPU win (~1.9x); OCR benefited
  most proportionally (~4.3x).
- If you're timing your own runs and see GPU looking *slower* than CPU on a single image,
  run it a second time back-to-back before concluding anything — the first CUDA call after
  the GPU has been idle is measurably slower than a subsequent one.

## Architecture

A stage-based pipeline, not a feature-first codebase — closer to an ETL pattern. Each
stage is a pure function behind a shared `Stage` trait, with typed input/output contracts
and disk-checkpointing between stages for batch/resume support:

```
detect → preprocess → mask → denoise → inpaint (optional) → export
```

| Crate | Role |
|---|---|
| `pc-core` | Shared types: `Rect`, `ImageHandle`, page data, the `Stage` trait, error types, the device policy resolver (CPU/CUDA) |
| `pc-config` | TOML profile/config, format-preserving via `toml_edit` |
| `pc-testkit` | Dev-only: fixture paths, image metrics (SSIM/IoU/diff), golden comparison |
| `pc-ort` | Shared `ort` execution-provider boundary — real CUDA registration, reused by every ONNX-backed stage |
| `pc-detect` + `pc-imageops` | Stage 1 — text detection (YOLO postprocess, resize/rescale, mask refinement, long-strip splitting) |
| `pc-preprocess` + `pc-ocr` | Stage 2 — box filtering, merging, reading order, OCR pass |
| `pc-mask` | Stage 3 — growth/dilation, border-color fitting, mask composition |
| `pc-denoise` | Stage 4 — non-local-means denoising (from-scratch, exact; CPU-only, no GPU path) |
| `pc-inpaint` | Stage 4b — LaMa-based inpainting for masked regions (on by default; opt out with `inpainting_enabled = false` or `--skip-inpaint`) |
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
Windows it builds but is not yet CI-verified (§16.33 item 10). The `cuda` feature builds on
any platform `ort` ships a CUDA distribution for (see [GPU acceleration](#gpu-acceleration))
but is not part of CI or the release matrix.

## FAQ

**Do I need a GPU?** No. CPU is the default, fully supported, and what every prebuilt
binary runs. GPU is an optional, build-it-yourself speedup for batches — see
[GPU acceleration](#gpu-acceleration).

**Why isn't GPU support in the downloaded binary?** It needs a CUDA toolkit and cuDNN
installed on the machine that *builds* it, and GitHub's hosted CI runners don't have
either preinstalled or a GPU to verify against. Nothing prevents adding it later; it just
isn't there today. Build it yourself with `--features cuda` if you want it.

**How much faster is GPU, really?** On a single page, often *not* faster — see the timing
caveats in [GPU acceleration](#gpu-acceleration). On a real 10-page batch we measured, GPU
was about 2.3x faster overall, with LaMa inpainting (the biggest single cost either way)
about 1.9x faster and OCR about 4.3x faster. Denoising has no GPU path at all — it's the
same CPU code either way.

**Is GPU output identical to CPU output?** Not bit-for-bit, run to run — small, measured,
non-gating numerical differences exist (see the linked divergence reports). Decoded text
and visual results matched on every real page tested.

**Where are the model files, and how big are they?** A per-user cache directory —
`panel-ocr models path` prints the exact location. ~763 MB total across all four required
models (detector, OCR encoder, OCR decoder, LaMa inpainter); `panel-ocr models verify`
checks what's already there without downloading.

**Can I import my old PanelCleaner `.ini` config?** Not yet. `pc-config` is TOML-only today;
legacy `.ini` import is deferred to a future release (see [Roadmap](#roadmap)).

**Does it export layered PSD files for editing in Photoshop?** Not yet — also deferred to a
future release. Today's output is flattened PNG/JPEG/etc. per your profile's
`preferred_file_type`.

**Inpainting looks wrong / I'd rather have a flat fill.** Set `inpainting_enabled = false`
in your profile, or pass `--skip-inpaint` for a single run. See
[Opting out of the default refinement and inpainting](#opting-out-of-the-default-refinement-and-inpainting).

**One bad page killed my whole batch — is that expected?** No — per-image failures are
isolated by design; a bad page is reported and skipped, not fatal to the batch, *except*
for run-fatal errors (a missing required model file, a device that can't be resolved at
all), which are deliberately fatal because retrying page-by-page against a broken
environment wouldn't help.

**What platforms are supported?** Linux, macOS (Intel + Apple Silicon), and Windows for the
default feature tier. The `onnx` tier (real models) is CI-verified on Linux and macOS;
on Windows it builds but isn't yet CI-verified. GPU is Linux/Windows-with-an-NVIDIA-card,
build-from-source only, on any platform `ort` ships a CUDA distribution for.

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
- [x] `Annotation` mask-refine mode — parity with upstream PanelCleaner's mask refinement,
      and the shipped default since §16.46 (`mask_refine_mode = "simple"` is the opt-out;
      DEVIATION(12), the default-value divergence, is retired by that entry)
- [x] LaMa inpainting — fills masked regions with generated content instead of a flat
      color, wired into the pipeline end-to-end and on by default (opt out with
      `inpainting_enabled = false`, or per-run with `--skip-inpaint`)
- [x] Fixed a real compositing defect (§16.45): partial-alpha regions (denoise/mask rims
      composited onto a transparent base) were darkening visibly instead of blending
      correctly — found via visual inspection of a real page, root-caused to
      `alpha_composite_over` not implementing real source-over, fixed and reviewed
- [x] `cargo xtask mode-bench` — a pinned, non-gating Simple/Annotation/LaMa comparison
      tool, with a real committed report in `docs/MODE_COMPARISON.md`

### v1.5 — done
- [x] GPU-1: device config, policy resolver, fatal-refusal wiring (§16.36)
- [x] GPU-2: real CUDA registration for the text detector, opt-in via the `cuda` feature,
      quarantined from every gate/fixture/recording (§16.47)
- [x] GPU-3: real, measured CUDA registration for OCR — manga-ocr's encoder and decoder
      (§16.48)
- [x] GPU-4: real, measured CUDA registration for LaMa inpainting, and full removal of the
      stage-scoped CUDA refusal that bridged the gap while OCR/LaMa lacked their own paths
      — every device-touching stage now has exactly one gate, `NotCompiledIn` (§16.49)
- Legacy INI config import, Lab-space coloured NLM, PSD/layered export, and DBNet
  line-polygon synthesis were originally scoped to v1.5 but are **deferred to a future
  release** by direct maintainer decision (§16.50) — see [Deferred backlog](#deferred-backlog)
  below.

### Deferred backlog
Originally scoped to v1.5, now unscheduled — moved here rather than dropped, so the scope
and citations aren't lost:
- [ ] Legacy INI config import — read upstream PanelCleaner's own `.ini` profile format
- [ ] Lab-space coloured NLM (`color_filter_strength`'s real effect, vs. today's
      joint-channel RGB approximation)
- [ ] PSD / layered export
- [ ] DBNet line-polygon synthesis (line-based block splitting/merging,
      orientation/font-size estimation) — the highest-risk item in this backlog; forces a
      detector-fixture re-record + re-sign when it lands, and needs its own gate
      re-ratified first (§14 item 17)

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
