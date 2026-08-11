# LaMa device divergence (GPU-4 G4-C)

**Generated in full by `cargo xtask lama-device-compare` — do not hand-edit.** Every value here is measured at generation time. This is a **non-gating** diagnostic: no number here is a pass/fail gate (§16.22 item 2's CPU-determinism carve-out applying in the negative direction: CUDA is exempt from determinism, so none of this asserts). It records what the SAME real LaMa `lama-manga.onnx` graph produced on the CPU execution provider vs the CUDA execution provider on one machine, across three independent full `inpaint_page` runs sharing one `PageInput`.

## 1. Method and provenance

- **Model:** C:/Users/ducph/AppData/Local/panel-ocr/models/lama-manga.onnx — sha256 `50a1abae0d73bd46d08eae36c8590cd59ad09029494c9698702b050ef00b0100`
- **Runs:** three independent full `inpaint_page` runs per page, each wrapped in its own fresh `RecordingInpainter` around its own `OnnxInpainter`: `cpu = OnnxInpainter::from_path_with_policy(&model, &DevicePolicy::cpu())`, `cuda_a` and `cuda_b` each from `resolve(Device::Cuda, DeviceSupport::compiled())`.
- **No cross-tile feedback loop:** `cover`/`owners` (the tile windows) are computed once, before the inference loop starts, from pure CPU arithmetic over `original`/the masks, with zero dependency on any tile's own inference output — so three independent runs sharing one `PageInput` receive byte-identical tile crops by construction.
- **Tile-identity assertion:** after all three runs, every recorded `(tile_rgb, mask_bits)` pair at index `i` is byte-identical across all three recordings, for every `i`. Confirmed at generation time: **PASSED**. If this ever failed, the producer aborts with a loud error rather than reporting divergence numbers.
- **Source A (default-config pages):** the six `tests/fixtures/upstream/demo_bubbles/ *_bubble_raw.png` pages that actually inpaint under the shipped default `InpainterConfig` (`black`, `darkrays`, `nightmare`, `ray`, `spikey`, `square`; `handwritten` has 0 eligible regions and is skipped). Each page is real content, real detection (a real `TextDetector` session), real masking, single-tile (every page is under 512px on both axes, verified).
- **Source A config:** the shipped default `InpainterConfig` verbatim — `min_inpainting_radius: 7, max_inpainting_radius: 20, inpainting_isolation_radius: 5, inpainting_fade_radius: 4`.
- **Source B (forced multi-tile):** the committed detector fixture `tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg` (a real, replayable page via `pc_detect::ReplayDetector`), run through the SAME real detect → preprocess → mask pipeline but with a deliberately widened `InpainterConfig`: `min_inpainting_radius = max_inpainting_radius = 300` (both equal, so `padded_region`'s growth clamp is exact — see `crates/pc-inpaint/src/growth.rs`), `inpainting_isolation_radius` and `inpainting_fade_radius` left at their shipped defaults (`5` and `4`). This is a real, hard-coded producer constant, not a CLI flag — a later run cannot silently change the disclosed method. Observed `tiles_inferred` this run: **4**.
- **CPU-pinned upstream stages:** detect, preprocess and mask run on the CPU in all three runs — the CUDA policy is only ever handed to the inpainter session itself.
- **Disclosure stated up front:** the min-quantization-margin column below is **CPU-only** — a characterization of the CPU run's own raw values, measuring how close its byte rounding was to flipping. It is **not** a CPU-vs-CUDA comparison.

## 2. Device

- **CPU policy:** device: requested cpu; no execution provider registered explicitly (ONNX Runtime's built-in CPU provider is implicit); per-node operator placement is not claimed
- **CUDA policy:** device: requested cuda; execution providers registered explicitly: cuda (conv algorithm search: heuristic, error on registration failure); per-node operator placement is not claimed

## 3. Environment

| Property | Value |
|---|---|
| `ort` crate version pin | `2.0.0-rc.12` |
| CUDA_PATH | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1 |
| CUDNN_PATH | D:\Temp\nvidia-cudnn\nvidia\cudnn\bin |
| driver version | 591.86 |
| CUDA version | Cuda compilation tools, release 12.1, V12.1.66 |
| CUDA provider requests | 1 |

## 4. Channels

Channels 1 and 2 are per-tile: channel 1 compares the CPU run's tile output to CUDA-A's, channel 2 compares CUDA-A's to CUDA-B's (self-nondeterminism, the analogue of GPU-3's channel 4 at the tile level). Channels 3 and 4 are page-level: channel 3 compares the CPU run's `inpainting`/`clean_inpaint` artifacts to CUDA-A's (whole artifact and write-region-restricted to `final_mask > 0`, available as `inpainting`'s own alpha channel), channel 4 the same CUDA-A vs CUDA-B. A whole-artifact number can understate a real difference by orders of magnitude when most of the image is untouched, so the write-region-restricted number is reported alongside it.

### 4.1 Source A — per-page rows (shipped default config)

| Page | tiles | tiles_inferred | Ch1 decoded diff | Ch1 raw bitwise-diff | Ch1 raw max\|Δ\| | Ch2 decoded diff | Ch2 raw bitwise-diff | Ch2 raw max\|Δ\| | Ch3 inpainting diff | Ch3 inpainting where diff | Ch3 clean diff | Ch3 clean where diff | Ch4 inpainting diff | Ch4 clean diff | min quant margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| black | 1 | 1 | 21449 | 786100 | 0.005065 | 0 | 0 | 0.000000 | 266 | 266 | 188 | 188 | 0 | 0 | 0.000000 |
| darkrays | 1 | 1 | 20999 | 695418 | 0.031810 | 0 | 0 | 0.000000 | 1881 | 1881 | 1642 | 1642 | 0 | 0 | 0.000000 |
| nightmare | 1 | 1 | 11183 | 671009 | 0.003266 | 0 | 0 | 0.000000 | 224 | 224 | 161 | 161 | 0 | 0 | 0.000000 |
| ray | 1 | 1 | 4490 | 777806 | 0.003024 | 0 | 0 | 0.000000 | 472 | 472 | 280 | 280 | 0 | 0 | 0.000000 |
| spikey | 1 | 1 | 6557 | 657448 | 0.004485 | 0 | 0 | 0.000000 | 263 | 263 | 146 | 146 | 0 | 0 | 0.000000 |
| square | 1 | 1 | 11925 | 683776 | 0.006607 | 0 | 0 | 0.000000 | 608 | 608 | 463 | 463 | 0 | 0 | 0.000000 |

### 4.2 Source B — forced multi-tile page and per-tile rows

| Page | tiles | tiles_inferred | Ch1 decoded diff | Ch1 raw bitwise-diff | Ch1 raw max\|Δ\| | Ch2 decoded diff | Ch2 raw bitwise-diff | Ch2 raw max\|Δ\| | Ch3 inpainting diff | Ch3 inpainting where diff | Ch3 clean diff | Ch3 clean where diff | Ch4 inpainting diff | Ch4 clean diff | min quant margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| forced-multi-tile | 4 | 4 | 555540 | 3119804 | 0.017849 | 0 | 0 | 0.000000 | 170705 | 170705 | 170215 | 170215 | 0 | 0 | 0.000000 |

Per-tile channel 1/2 rows (decoded RGB and raw f32, per recorded tile index):

| Tile | Ch1 decoded diff | Ch1 raw bitwise-diff | Ch1 raw max\|Δ\| | Ch1 raw mean\|Δ\| | Ch2 decoded diff | Ch2 raw bitwise-diff | Ch2 raw max\|Δ\| | Ch2 raw mean\|Δ\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 151536 | 786280 | 0.010982 | 0.000755 | 0 | 0 | 0.000000 | 0.000000 |
| 1 | 67614 | 781264 | 0.004913 | 0.000337 | 0 | 0 | 0.000000 | 0.000000 |
| 2 | 273009 | 780683 | 0.017849 | 0.001369 | 0 | 0 | 0.000000 | 0.000000 |
| 3 | 63381 | 771577 | 0.004737 | 0.000315 | 0 | 0 | 0.000000 | 0.000000 |

### 4.3 Global aggregates (all tiles, both sources)

| Aggregate | Ch1 raw max-of-maxes | Ch1 raw mean-of-means | Ch2 raw max-of-maxes | Ch2 raw mean-of-means |
|---|---:|---:|---:|---:|
| global | 0.031810 | 0.000350 | 0.000000 | 0.000000 |

## 5. Verdict

**Non-gating.** No number above is a pass/fail gate. CUDA is exempt from determinism, so none of this asserts. The numbers record what actually happened on this machine at generation time, for the ratification step (G4-D) to cite.

