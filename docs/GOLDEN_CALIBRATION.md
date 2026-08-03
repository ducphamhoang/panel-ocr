# Golden calibration (spec §7.3, task F2)

**Generated in full by `cargo xtask calibrate-goldens` — do not hand-edit.** Every
number here is measured at generation time; re-run the command to refresh it.
Section 4's real demo_bubbles measurements require `--features onnx --detector
onnx:<path>`; running without a verified model explicitly replaces prior real
Section 4 measurements with a BLOCKED row and emits a warning below.
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
  "schema_version": 1,
  "group": "nlm",
  "tool": "cv2.fastNlMeansDenoising",
  "command_line": "cargo xtask record-fixtures --only nlm",
  "tool_versions": {
    "numpy": "2.4.6",
    "opencv": "5.0.0",
    "python": "3.11.15"
  },
  "records": [
    {
      "name": "nightmare",
      "output": "tests/fixtures/recorded/nlm/nightmare_h10_t7_s21.png",
      "output_sha256": "c6cc1002c209ffadd182f2ebd3162b53b55bbb92f99ee038ebaf419bfed94b8d",
      "committed": true,
      "source": "tests/fixtures/upstream/demo_bubbles/nightmare_bubble_raw.png",
      "params": {
        "h": 10,
        "searchWindowSize": 21,
        "size": [219, 343],
        "templateWindowSize": 7
      }
    },
    {
      "name": "ray",
      "output": "tests/fixtures/recorded/nlm/ray_h10_t7_s21.png",
      "output_sha256": "9cdd17366dbabf152f67b8d70bee43745c7e45011f8d1cf2fc55b83646fe1658",
      "committed": true,
      "source": "tests/fixtures/upstream/demo_bubbles/ray_bubble_raw.png",
      "params": {
        "h": 10,
        "searchWindowSize": 21,
        "size": [256, 329],
        "templateWindowSize": 7
      }
    }
  ]
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
  "schema_version": 1,
  "group": "inter_area",
  "tool": "cv2.resize/INTER_AREA",
  "command_line": "cargo xtask record-fixtures --only inter-area",
  "tool_versions": {
    "numpy": "2.4.6",
    "opencv": "5.0.0",
    "python": "3.11.15"
  },
  "records": [
    {
      "name": "reference",
      "output": "tests/fixtures/recorded/inter_area/long_strip_inter_area_500x4000.png",
      "output_sha256": "812b37cc1168f608d56b99a17fc0dd58d0b105834c1a310a7133e991dc113fd9",
      "committed": true,
      "source": "target/xtask-scratch/long_strip_decoded_rgb.png",
      "params": {
        "output_size": [500, 4000],
        "source_size": [1000, 8000]
      }
    },
    {
      "name": "jpeg_decode_diagnostic",
      "output": "target/xtask-scratch/long_strip_inter_area_500x4000_cv2jpeg.png",
      "output_sha256": "cf16b4bdc28f0e58fefc838a8286a4c42f0f3cc687cbe2fe2129a1d9fadfb252",
      "committed": false,
      "source": "tests/fixtures/upstream/long_strip.jpg",
      "params": {
        "output_size": [500, 4000],
        "source_size": [1000, 8000]
      }
    }
  ],
  "diagnostics": {
    "metrics_vs_reference": {
      "max_delta": 1,
      "mean_abs_diff": 0.003438,
      "ssim_as_gray": 0.9999985218837129
    }
  }
}
```

</details>

## 3. Detector-dependent test status

Measured from the test attributes in the current workspace tree; `UNKNOWN` is never treated as live or ignored.

| Test | Spec | Status |
|---|---|---|
| Hand-written detect determinism + committed-raw equality (`a6_pending_recorded_page_equality_and_determinism`) | §8.7(A)6 | LIVE |
| Recorded-page box-count/coordinate regression lock (`b9_pending_recorded_page_regression_lock`) | §8.7(B)9 | LIVE |
| Hand-written preprocess tier arithmetic (`b11_pending_recorded_page_tier_arithmetic`) | §9.7(B)11 | LIVE |
| End-to-end denoise golden PNGs (_noise_mask.png, _clean_denoised.png) (`b13_pending_recorded_page_end_to_end_golden`) | §11.7(B)13 | IGNORED — pending task F1: needs the recorded page fixture and its committed golden PNGs |

`End-to-end denoise golden PNGs (_noise_mask.png, _clean_denoised.png)` (`b13_pending_recorded_page_end_to_end_golden`) is ignored with measured reason: `pending task F1: needs the recorded page fixture and its committed golden PNGs`.

Among the 4 tracked tests, §11.7(B)13's denoise golden requires special handling: deriving its golden references by running our own denoiser would violate the "no self-oracle" rule (cookbook rule 7); its derivation/signing decision needs the joint architects, not a unilateral fix.

Those four tests un-ignore only against the signed maintainer page, never against `demo_bubbles`. (§16.24 item 12)

## 4. demo_bubbles masking calibration — §10.7(B)15

Input: each `<name>_bubble_raw.png` in `pc_testkit::paths::DEMO_BUBBLES`; reference: the matching vendored `<name>_bubble_clean.png`. Detector and mask artifacts are scratch-only under `target/xtask-scratch/` (§16.24 item 12); only the cleaned PNG is written for human inspection.

This is a non-gating calibration report (§15.2). A shortfall against reference values (IoU ≥ 0.99, ≥99.5% exact, max Δ ≤ 2, SSIM ≥ 0.995) may reflect the `MaskRefineMode::Simple` vs upstream's full refinement difference and/or §10.7(A)9 border-uniformity failures that leave a region untouched. The per-crop counts distinguish succeeded, failed, and dropped regions; fitting statistics describe only regions that reached fitting, and this report does not isolate causal contributions. This is evidence, not a build failure.

| Name | Dimensions | Exact fraction | Max delta | Mean abs diff | SSIM | Shape IoU | Within dilation |
|---|---:|---:|---:|---:|---:|---:|:---:|
| black | 202×319 | 0.842624 | 255 | 7.841926 | 0.786630 | 0.113704 | true |
| darkrays | 208×320 | 0.812109 | 255 | 13.989694 | 0.729663 | 0.000000 | true |
| handwritten | 72×132 | 0.661616 | 255 | 20.541982 | 0.621282 | 0.000000 | true |
| nightmare | 219×343 | 0.775377 | 254 | 7.396528 | 0.826528 | 0.000000 | true |
| ray | 256×329 | 0.511208 | 255 | 50.733995 | 0.373859 | 0.000000 | true |
| spikey | 354×354 | 0.873488 | 255 | 10.659174 | 0.832276 | 0.273086 | true |
| square | 144×270 | 0.785082 | 255 | 14.689841 | 0.680407 | 0.000000 | true |

- `black`: measured — detector boxes: 2; masking regions: 2 (reached fitting: 2; succeeded: 1, failed: 1, dropped: 0; border std devs: [25.107338, 2.345173]); output changes: present Shortfall may reflect the `MaskRefineMode::Simple` vs upstream's full refinement difference; this report does not isolate that causal contribution..
- `darkrays`: measured — detector boxes: 2; masking regions: 1 (reached fitting: 1; succeeded: 0, failed: 1, dropped: 0; border std devs: [59.063776]); output changes: none.
- `handwritten`: measured — detector boxes: 0; masking regions: 0 (reached fitting: 0; succeeded: 0, failed: 0, dropped: 0; border std devs: —); output changes: none.
- `nightmare`: measured — detector boxes: 1; masking regions: 1 (reached fitting: 1; succeeded: 0, failed: 1, dropped: 0; border std devs: [26.972765]); output changes: none.
- `ray`: measured — detector boxes: 1; masking regions: 1 (reached fitting: 1; succeeded: 0, failed: 1, dropped: 0; border std devs: [30.387882]); output changes: none.
- `spikey`: measured — detector boxes: 2; masking regions: 2 (reached fitting: 2; succeeded: 1, failed: 1, dropped: 0; border std devs: [42.926213, 0.000000]); output changes: present Shortfall may reflect the `MaskRefineMode::Simple` vs upstream's full refinement difference; this report does not isolate that causal contribution..
- `square`: measured — detector boxes: 1; masking regions: 1 (reached fitting: 1; succeeded: 0, failed: 1, dropped: 0; border std devs: [43.097891]); output changes: none.

> For a crop with `output changes: none`, Within dilation=true is vacuous because the ours-change set is empty; it is not evidence that masking succeeded.

## 5. Detector box-count comparison — §15.1

Recorded page: `ja_Pepper-and-Carrot_by-David-Revoy_E01P01`.

| Ours total | Upstream total | Pairs compared | Gating rows |
|---:|---:|---:|---:|
| 4 | 4 | 3 | 0 |

| Side | Unmatched index | Mechanism |
|---|---:|---|
| ours | 3 | `CoverageFilteredUpstream` |
| upstream | 1 | `ClassDuplicateOf` |

The unmatched upstream `ClassDuplicateOf` entry at index 1 is the expected, accepted consequence of §15.1's ratified class-agnostic NMS decision and deviation §14.13; it is not an anomaly.

<details><summary>Recording provenance (`tests/fixtures/recorded/detector/PROVENANCE.json`)</summary>

```json
{
  "schema_version": 1,
  "group": "detector",
  "tool": "PanelCleaner detector oracle recorder",
  "command_line": "cargo xtask record-fixtures --only detector",
  "tool_versions": {
    "numpy": "2.5.1",
    "opencv": "5.0.0",
    "python": "3.12.3",
    "torch": "2.13.0+cpu"
  },
  "records": [
    {
      "name": "detector_mask",
      "output": "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_detector_mask.png",
      "output_sha256": "8052d7331648165cc01718e32cfd156ea24b50daf1b2544a6934878f94c7eac4",
      "committed": true
    },
    {
      "name": "detector_blocks",
      "output": "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_detector_blocks.json",
      "output_sha256": "ea5a2e36ddb5ce591b5c41ca9f5cd5ef229b0bf961a510259359848b54b52b36",
      "committed": true
    },
    {
      "name": "base",
      "output": "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_base.png",
      "output_sha256": "af3700d436a3a0a90f84c32c6d2b2d5ff79ecbdfca7bf3cbdb46709e1b508943",
      "committed": true
    },
    {
      "name": "raw_mask",
      "output": "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_raw_mask.png",
      "output_sha256": "d33e5359541962c563cf89c42d9ab99c52b690a7f425f40d7a9e4535911c78d7",
      "committed": true
    },
    {
      "name": "raw_page",
      "output": "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01#raw.json",
      "output_sha256": "d3e6965b2a50ed1f925f4d320369e98b3304a81884967e220e3385b4f706853b",
      "committed": true
    },
    {
      "name": "upstream_oracle",
      "output": "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_upstream_oracle.json",
      "output_sha256": "c6cb214360132a1b52b09a5a084735930ee9f757c560067a810fa37c1cb6d686",
      "committed": true
    },
    {
      "name": "group_output_equality",
      "output": "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01_upstream_group_output_equality.json",
      "output_sha256": "e335276911a150d83394679ab4af32483e631a674f9685a280dd4bb2ff0bead3",
      "committed": true
    }
  ],
  "detector": {
    "input_page": "tests/fixtures/recorded/detector/ja_Pepper-and-Carrot_by-David-Revoy_E01P01.jpg",
    "input_page_sha256": "3bef9922e09cea66ab12271da0070025768ae9bc5d286f41ced617468131267e",
    "model": "comictextdetector.pt.onnx",
    "model_digest": "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f",
    "ours": {
      "backend": "ort",
      "decoded_rgb_digest": "82c9b93e3bd95420df7c8bf9f5ffb5b322ec04c7d8576e8cf87ad254adf385e4",
      "decoded_from": "input_page",
      "execution_provider": "cpu",
      "intra_threads": 0,
      "inter_threads": 0,
      "pad_value": 0,
      "panel_ocr_commit": "cecb9f102177c50f3e8bb0af73f66bc08408db07"
    },
    "upstream": {
      "backend": "cv2_dnn",
      "decoded_rgb_digest": "82c9b93e3bd95420df7c8bf9f5ffb5b322ec04c7d8576e8cf87ad254adf385e4",
      "decoded_from": "input_page",
      "version": "2.11.11",
      "commit": "0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3",
      "command_line": "record_detector_oracle.py PanelCleaner comictextdetector.pt.onnx ja_Pepper-and-Carrot_by-David-Revoy_E01P01_decoded_rgb.png detector ja_Pepper-and-Carrot_by-David-Revoy_E01P01",
      "dependency_versions": {
        "numpy": "2.5.1",
        "opencv": "5.0.0",
        "pcleaner": "2.11.11",
        "torch": "2.13.0+cpu"
      }
    }
  },
  "diagnostics": {
    "confidence_mapping": {
      "status": "ok"
    },
    "derivation_census": {
      "dbnet_scattered": 0,
      "yolo_split": 0,
      "yolo_synthesized_corners": 2,
      "yolo_unioned": 2
    },
    "instrumented_pristine_equality": {
      "blocks": 4,
      "checked": true,
      "fields": [
        "xyxy",
        "lines",
        "language",
        "vertical",
        "font_size"
      ]
    },
    "line_count_census": {
      "blocks": 4,
      "with_zero_lines": 0
    },
    "upstream_pre_filter_mask_scores": [
      {
        "index": 0,
        "len_lines": 2,
        "mask_score": null,
        "xyxy": [
          567,
          74,
          663,
          123
        ]
      },
      {
        "index": 1,
        "len_lines": 0,
        "mask_score": 0.46598988449777545,
        "xyxy": [
          674,
          1397,
          740,
          1438
        ]
      },
      {
        "index": 2,
        "len_lines": 3,
        "mask_score": null,
        "xyxy": [
          607,
          630,
          723,
          703
        ]
      },
      {
        "index": 3,
        "len_lines": 0,
        "mask_score": 0.2605883283987859,
        "xyxy": [
          607,
          631,
          724,
          703
        ]
      },
      {
        "index": 4,
        "len_lines": 0,
        "mask_score": 0.03446455505279035,
        "xyxy": [
          438,
          1407,
          498,
          1446
        ]
      }
    ]
  }
}
```

</details>

## 6. PIL `FIND_EDGES` cross-check — §10.3 step 2 / §16.9 item 21

Not a fixture: `pc_mask::border` consumes no recorded file. `cargo xtask
record-fixtures --only find-edges` runs real `PIL.ImageFilter.FIND_EDGES` over all
512 distinct 3×3 masks plus 500 random masks and asserts agreement with §10.3
step 2's closed form, and that a fully-set 3×3 mask yields **8** edges. That is the
empirical confirmation of §16.9 item 21's hand proof; a disagreement is escalated,
never patched. See the run log for the current result.

## 7. Verdict

| Gate | Spec | Status |
|---|---|---|
| NLM parity | §11.7(B)12 | MET |
| INTER_AREA parity | §8.7(A)2 | MET |
| demo_bubbles masking calibration report | §10.7(B)15 | REPORTED (non-gating) |
| Hand-written detect determinism + committed-raw equality | §8.7(A)6 | LIVE |
| Recorded-page box-count/coordinate regression lock | §8.7(B)9 | LIVE |
| Hand-written preprocess tier arithmetic | §9.7(B)11 | LIVE |
| End-to-end denoise golden PNGs (_noise_mask.png, _clean_denoised.png) | §11.7(B)13 | IGNORED — pending task F1: needs the recorded page fixture and its committed golden PNGs |

