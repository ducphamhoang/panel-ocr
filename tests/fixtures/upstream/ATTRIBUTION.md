# Vendored upstream test fixtures — attribution

The files in this directory are **copied verbatim from PanelCleaner** and are not
original works of panel-ocr. They are vendored so that panel-ocr's parity and
calibration tests (spec §7.1) can run without a network fetch or a submodule.

## Source

| | |
|---|---|
| Project | PanelCleaner |
| Author | VoxelCubes and PanelCleaner contributors |
| Repository | https://github.com/VoxelCubes/PanelCleaner |
| Commit | `0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3` |
| Commit date | 2026-07-16 |
| Commit subject | "Update flatpak metadata" |
| Upstream version at that commit | `pcleaner 2.11.11` (`pcleaner/__init__.py`) |
| License | GNU General Public License v3.0 (see the upstream `LICENSE`) |

## License

PanelCleaner is licensed under the **GNU General Public License, version 3**.
panel-ocr is likewise licensed under **GPL-3.0**, so these files are redistributed
here under the same terms. The full licence text is in this repository's root
`LICENSE`. No copyright notice, licence text, or attribution has been removed from
any copied file, and none of these files have been modified — they are byte-for-byte
copies of the upstream originals.

## Files and their upstream paths

| Vendored path (relative to this directory) | Upstream path |
|---|---|
| `demo_bubbles/black_bubble_raw.png` | `media/demo_bubbles/black_bubble_raw.png` |
| `demo_bubbles/black_bubble_clean.png` | `media/demo_bubbles/black_bubble_clean.png` |
| `demo_bubbles/darkrays_bubble_raw.png` | `media/demo_bubbles/darkrays_bubble_raw.png` |
| `demo_bubbles/darkrays_bubble_clean.png` | `media/demo_bubbles/darkrays_bubble_clean.png` |
| `demo_bubbles/handwritten_bubble_raw.png` | `media/demo_bubbles/handwritten_bubble_raw.png` |
| `demo_bubbles/handwritten_bubble_clean.png` | `media/demo_bubbles/handwritten_bubble_clean.png` |
| `demo_bubbles/nightmare_bubble_raw.png` | `media/demo_bubbles/nightmare_bubble_raw.png` |
| `demo_bubbles/nightmare_bubble_clean.png` | `media/demo_bubbles/nightmare_bubble_clean.png` |
| `demo_bubbles/ray_bubble_raw.png` | `media/demo_bubbles/ray_bubble_raw.png` |
| `demo_bubbles/ray_bubble_clean.png` | `media/demo_bubbles/ray_bubble_clean.png` |
| `demo_bubbles/spikey_bubble_raw.png` | `media/demo_bubbles/spikey_bubble_raw.png` |
| `demo_bubbles/spikey_bubble_clean.png` | `media/demo_bubbles/spikey_bubble_clean.png` |
| `demo_bubbles/square_bubble_raw.png` | `media/demo_bubbles/square_bubble_raw.png` |
| `demo_bubbles/square_bubble_clean.png` | `media/demo_bubbles/square_bubble_clean.png` |
| `long_strip.jpg` | `tests/mock_files/long_strip.jpg` |
| `ocr_output/good_detected_text.csv` | `tests/mock_files/ocr_output/good_detected_text.csv` |
| `ocr_output/good_detected_text.txt` | `tests/mock_files/ocr_output/good_detected_text.txt` |

## Measured properties (spec §7.1 — use as test constants)

All `demo_bubbles` images are 8-bit **grayscale** PNGs, and each `raw`/`clean` pair
shares one size:

| fixture | size |
|---|---|
| `black` | 202 x 319 |
| `darkrays` | 208 x 320 |
| `handwritten` | 72 x 132 |
| `nightmare` | 219 x 343 |
| `ray` | 256 x 329 |
| `spikey` | 354 x 354 |
| `square` | 144 x 270 |

`long_strip.jpg` is 1000 x 8000 RGB, progressive JPEG, 300 dpi.

## Status of the `*_clean.png` images — read before writing a test against them

The `demo_bubbles` files live in upstream's `media/` directory: they are **README
demo assets**, not part of upstream's own test suite, and the PanelCleaner version
and profile that produced each `*_clean.png` are not recorded anywhere upstream.

Per spec §15.2 (decided by Fable) they are therefore **calibration fixtures, not
frozen gates**: §10.7(B) item 15 compares against them and records the numbers in
`docs/GOLDEN_CALIBRATION.md` without gating CI. Do not add a pass/fail assertion
against a `*_clean.png` image. The one frozen assertion that touches this data is
§10.7(B) item 16 (`black_bubble`'s chosen fill colour must be dark and must not hit
the off-white snap), which is independent of mask-refinement mode.

## Re-vendoring

If these fixtures are ever refreshed from a newer upstream commit, update the commit
hash, date, and version in the table above in the same change, and re-run
`cargo xtask calibrate-goldens` (task F2) so `docs/GOLDEN_CALIBRATION.md` still
describes the files actually present.
