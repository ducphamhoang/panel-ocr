#!/usr/bin/env python3
"""Record the `cv2.fastNlMeansDenoising` reference for spec §11.6 / §11.7(B)12.

Committed so the provenance of `tests/fixtures/recorded/nlm/*.png` is auditable
(§11.6 says so explicitly). Invoked by `cargo xtask record-fixtures`; runnable
standalone.

Contract, pinned by §16.10 item 19 — do not "improve" any of it:
  input     tests/fixtures/upstream/demo_bubbles/<name>_bubble_raw.png, as luma8
  operation cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
  output    tests/fixtures/recorded/nlm/<name>_h10_t7_s21.png, 8-bit grayscale PNG

Usage: record_nlm.py <upstream_dir> <out_dir> <name> [<name> ...]
"""

import hashlib
import json
import sys

import cv2
import numpy as np

# §16.10 item 19 / §6 denoiser defaults. Positional in OpenCV's signature:
# fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize).
H = 10
TEMPLATE_WINDOW = 7
SEARCH_WINDOW = 21


def main(argv):
    if len(argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    upstream_dir, out_dir = argv[1], argv[2]
    names = argv[3:]

    import os

    os.makedirs(out_dir, exist_ok=True)
    records = []

    for name in names:
        src_path = os.path.join(upstream_dir, "demo_bubbles", f"{name}_bubble_raw.png")
        # IMREAD_GRAYSCALE is the exact analogue of the Rust side's `to_luma8()` on an
        # already-8-bit-grayscale PNG: both are the identity on the stored samples.
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"could not read {src_path}")
        if img.dtype != np.uint8:
            raise SystemExit(f"{src_path} decoded as {img.dtype}, expected uint8")

        out = cv2.fastNlMeansDenoising(img, None, H, TEMPLATE_WINDOW, SEARCH_WINDOW)

        out_path = os.path.join(out_dir, f"{name}_h{H}_t{TEMPLATE_WINDOW}_s{SEARCH_WINDOW}.png")
        if not cv2.imwrite(out_path, out):
            raise SystemExit(f"could not write {out_path}")

        with open(out_path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        records.append(
            {
                "name": name,
                "source": src_path,
                "output": out_path,
                "size": [int(img.shape[1]), int(img.shape[0])],
                "sha256": digest,
            }
        )

    print(
        json.dumps(
            {
                "tool": "cv2.fastNlMeansDenoising",
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
                "python": sys.version.split()[0],
                "params": {
                    "h": H,
                    "templateWindowSize": TEMPLATE_WINDOW,
                    "searchWindowSize": SEARCH_WINDOW,
                },
                "records": records,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
