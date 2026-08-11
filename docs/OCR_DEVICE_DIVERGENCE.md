# OCR device divergence (GPU-3 G3-D)

**Generated in full by `cargo xtask ocr-device-compare` — do not hand-edit.** Every value here is measured at generation time. This is a **non-gating** diagnostic: no number here is a pass/fail gate (§16.22 item 2's CPU-determinism carve-out applying in the negative direction: CUDA is exempt from determinism, so none of this asserts). It records what the SAME real manga-ocr encoder+decoder ONNX graphs produced on the CPU execution provider vs the CUDA execution provider on one machine.

## 1. Method and provenance

- **Crop:** E:/game-prototypes/panel-ocr/tests/fixtures/upstream/demo_bubbles/handwritten_bubble_raw.png (loaded with `image::open`; preprocessed by `pc_ocr::preprocess::pixel_values`).
- **Encoder:** C:/Users/ducph/AppData/Local/panel-ocr/models/encoder_model.onnx — sha256 `15fa8155fe9bc1a7d25d9bb353debaa4def033d0174e907dbd2dd6d995def85f`
- **Decoder:** C:/Users/ducph/AppData/Local/panel-ocr/models/decoder_model.onnx — sha256 `ef7765261e9d1cdc34d89356986c2bbc2a082897f753a89605ae80fdfa61f5e8`
- **Sessions:** two independent `MangaOcrSessions` via `from_paths_for_device(.., Device::Cpu)` and `from_paths_for_device(.., Device::Cuda)`.
- **Channel 2 mechanism (teacher-forced):** the real CPU beam search runs and records every `(prefix, logits)` pair it serves; each is replayed through the CUDA decoder with the **CPU** encoder (`&cpu_enc`), isolating the decoder's own divergence from the encoder's and from the beam search's feedback loop.
- **Channel 3/4 mechanism:** two fully independent free-running `encode` + `beam_search` decodes (one CPU, two CUDA), compared by `first_divergence_index` and decoded-string equality.

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

## 4. Channels

| Channel | What is compared | Values | Bitwise-differing | max abs Δ | mean abs Δ | per-row max-of-maxes | min top-2 margin |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | encoder hidden states, CPU vs CUDA | 151296 | 151288 | 0.009014 | 0.000365 | — | — |
| 2 | decoder logits, teacher-forced (CPU encoder) | 423936 (69 rows) | 423837 | 0.004109 | 0.000506 | 0.004109 | 0.108847 |
| 3 | end-to-end token IDs, CPU run vs CUDA run A | first divergence `None` | strings equal: true | — | — | — | — |
| 4 | end-to-end token IDs, CUDA run A vs CUDA run B | first divergence `None` | strings equal: true | — | — | — | — |

## 5. Decoded text

| Run | token IDs | decoded text |
|---|---|---|
| CPU | `[2, 2, 875, 1011, 869, 926, 861, 923, 3803, 893, 912, 929, 924, 970, 963, 861, 15, 45, 3]` | `すラこれから目にまわりダスか！？` |
| CUDA A | `[2, 2, 875, 1011, 869, 926, 861, 923, 3803, 893, 912, 929, 924, 970, 963, 861, 15, 45, 3]` | `すラこれから目にまわりダスか！？` |
| CUDA B | `[2, 2, 875, 1011, 869, 926, 861, 923, 3803, 893, 912, 929, 924, 970, 963, 861, 15, 45, 3]` | `すラこれから目にまわりダスか！？` |

## 6. Verdict

**Non-gating.** No number above is a pass/fail gate. CUDA is exempt from determinism, so none of this asserts. The numbers record what actually happened on this machine at generation time, for the ratification step (G3-E) to cite.

