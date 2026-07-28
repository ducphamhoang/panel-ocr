# Golden calibration (spec §7.3, task F2)

**Generated in full by `cargo xtask calibrate-goldens` — do not hand-edit.** Every
number here is measured at generation time; re-run the command to refresh it.
Recording provenance (tool versions, parameters, output hashes) is inlined below
from each fixture directory's `PROVENANCE.json`.

§7.3's rule, restated: these numbers must be *measured before a tolerance is
frozen*, never fitted afterwards. If a measurement misses a specified tolerance,
the discrepancy goes back to the two architects jointly (CLAUDE.md) — it is never
resolved by loosening the threshold here.

## 1. NLM vs. `cv2.fastNlMeansDenoising` — §11.7(B)12 (frozen gate)

Input: `tests/fixtures/upstream/demo_bubbles/<name>_bubble_raw.png` as luma8.
Reference: `tests/fixtures/recorded/nlm/<name>_h10_t7_s21.png`, produced by
`cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)`
(§16.10 item 19). Ours: `pc_denoise::nlm::denoise` with `NlmParams::defaults()`.
Specified tolerances (`GoldenThresholds::nlm_parity()`): SSIM ≥ 0.98,
mean |Δ| ≤ 1.0, max Δ ≤ 8.

| Name | Dimensions | Exact fraction | Max delta | Mean abs diff | SSIM | Shape IoU | Within dilation |
|---|---:|---:|---:|---:|---:|---:|:---:|
| nightmare | 219×343 | 0.991107 | 4 | 0.008946 | 0.999985 | — | — |
| ray | 256×329 | 0.996972 | 1 | 0.003028 | 0.999999 | — | — |

- `nightmare`: **all specified tolerances met.**
- `ray`: **all specified tolerances met.**

<details><summary>Recording provenance (`tests/fixtures/recorded/nlm/PROVENANCE.json`)</summary>

```json
{
  "numpy_version": "2.4.6",
  "opencv_version": "5.0.0",
  "params": {
    "h": 10,
    "searchWindowSize": 21,
    "templateWindowSize": 7
  },
  "python": "3.11.15",
  "records": [
    {
      "name": "nightmare",
      "output": "tests/fixtures/recorded/nlm/nightmare_h10_t7_s21.png",
      "sha256": "c6cc1002c209ffadd182f2ebd3162b53b55bbb92f99ee038ebaf419bfed94b8d",
      "size": [
        219,
        343
      ],
      "source": "tests/fixtures/upstream/demo_bubbles/nightmare_bubble_raw.png"
    },
    {
      "name": "ray",
      "output": "tests/fixtures/recorded/nlm/ray_h10_t7_s21.png",
      "sha256": "9cdd17366dbabf152f67b8d70bee43745c7e45011f8d1cf2fc55b83646fe1658",
      "size": [
        256,
        329
      ],
      "source": "tests/fixtures/upstream/demo_bubbles/ray_bubble_raw.png"
    }
  ],
  "tool": "cv2.fastNlMeansDenoising"
}
```

</details>

**§7.3 gate status:** SATISFIED — `n1_nlm.rs::b12_recorded_opencv_parity` may be (and is) unignored.

## 2. `resize_area` vs. `cv2.INTER_AREA` — §8.7(A)2 (frozen gate)

Downscale of `long_strip.jpg` to 500×4000. Specified tolerances: mean |Δ| ≤ 1.0 and
max per-channel Δ ≤ 2.

§16.13 item 5: the reference was recorded from a **lossless PNG re-encode of the
Rust-side JPEG decode**, so the gate measures resize arithmetic and not
libjpeg-turbo-vs-`image` decoder differences. The second row below is the
diagnostic that quantifies how large that decoder difference is; it is recorded
for visibility and is **not** a gate.

| Comparison | Dimensions | Mean abs diff | Max per-channel Δ | SSIM (as gray) |
|---|---:|---:|---:|---:|
| ours vs. cv2 INTER_AREA (gate) | 500×4000 | 0.000000 | 0 | 1.000000 |
| cv2-from-JPEG vs. cv2-from-PNG (decoder diagnostic, non-gating) | 500×4000 | 0.003438 | 1 | 0.999999 |

**§8.7(A)2 status:** tolerances met — `d2_resize.rs::a2_inter_area_matches_the_recorded_opencv_reference` may be (and is) unignored.

<details><summary>Recording provenance (`tests/fixtures/recorded/inter_area/PROVENANCE.json`)</summary>

```json
{
  "jpeg_decode_diagnostic": {
    "metrics_vs_reference": {
      "max_delta": 1,
      "mean_abs_diff": 0.003438,
      "ssim_as_gray": 0.9999985218837129
    },
    "numpy_version": "2.4.6",
    "opencv_version": "5.0.0",
    "output": "target/xtask-scratch/long_strip_inter_area_500x4000_cv2jpeg.png",
    "output_size": [
      500,
      4000
    ],
    "python": "3.11.15",
    "sha256": "cf16b4bdc28f0e58fefc838a8286a4c42f0f3cc687cbe2fe2129a1d9fadfb252",
    "source": "tests/fixtures/upstream/long_strip.jpg",
    "source_size": [
      1000,
      8000
    ],
    "tool": "cv2.resize/INTER_AREA"
  },
  "reference": {
    "numpy_version": "2.4.6",
    "opencv_version": "5.0.0",
    "output": "tests/fixtures/recorded/inter_area/long_strip_inter_area_500x4000.png",
    "output_size": [
      500,
      4000
    ],
    "python": "3.11.15",
    "sha256": "812b37cc1168f608d56b99a17fc0dd58d0b105834c1a310a7133e991dc113fd9",
    "source": "target/xtask-scratch/long_strip_decoded_rgb.png",
    "source_size": [
      1000,
      8000
    ],
    "tool": "cv2.resize/INTER_AREA"
  }
}
```

</details>

## 3. Measurements blocked on the detector recording (F1, detector group)

None of the following can be measured until spec §8.5 tasks **D1** (`pc-models`) and
**D4** (`pc-detect/src/onnx.rs`) exist and the recording is run on a machine holding
`comictextdetector.pt.onnx`. `cargo xtask record-fixtures --only detector` prints the
same explanation with the exact artifact list.

| Measurement | Spec | Blocked test |
|---|---|---|
| demo_bubbles masking calibration report (IoU, exact %, max Δ, SSIM per fixture) | §10.7(B)15 | *(non-gating report; no test)* |
| Hand-traced review of the detect-stage `insta` snapshot | §8.7(A)6 / §15.10(a) | `pc-detect d7_run.rs::a6_pending_insta_snapshot_of_recorded_page` |
| Recorded-page box-count/coordinate regression lock | §8.7(B)9 | `pc-detect d7_run.rs::b9_pending_recorded_page_regression_lock` |
| Hand-traced review of the preprocess-stage `insta` snapshot | §9.7(B)11 / §15.10(a) | `pc-preprocess p5_run.rs::b11_pending_insta_snapshot_of_recorded_page_tiers` |
| End-to-end denoise golden PNGs (`_noise_mask.png`, `_clean_denoised.png`) | §11.7(B)13 / §16.10 item 20 | `pc-denoise n4_run.rs::b13_pending_recorded_page_end_to_end_golden` |

§10.7(B)15 additionally needs one or two license-clean full manga pages (≤ 400 KB
each) supplied by the maintainer (§7.2), which no automated step can source.

## 4. PIL `FIND_EDGES` cross-check — §10.3 step 2 / §16.9 item 21

Not a fixture: `pc_mask::border` consumes no recorded file. `cargo xtask
record-fixtures --only find-edges` runs real `PIL.ImageFilter.FIND_EDGES` over all
512 distinct 3×3 masks plus 500 random masks and asserts agreement with §10.3
step 2's closed form, and that a fully-set 3×3 mask yields **8** edges. That is the
empirical confirmation of §16.9 item 21's hand proof; a disagreement is escalated,
never patched. See the run log for the current result.

## 5. Verdict

| Gate | Spec | Status |
|---|---|---|
| NLM parity | §11.7(B)12 | MET |
| INTER_AREA parity | §8.7(A)2 | MET |
| Detector-dependent goldens | §8.7(A)6, §8.7(B)9, §9.7(B)11, §10.7(B)15, §11.7(B)13 | BLOCKED on D1+D4 |

