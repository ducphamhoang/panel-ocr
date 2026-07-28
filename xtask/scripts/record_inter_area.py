#!/usr/bin/env python3
"""Record the `cv2.INTER_AREA` downscale reference for spec §8.3 step 2 / §8.7(A)2.

Committed for the same provenance reason as `record_nlm.py`. Invoked by
`cargo xtask record-fixtures`; runnable standalone.

  operation cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
  output    an 8-bit RGB PNG

NOTE (§16.13 item 5): the caller is expected to hand this script a **losslessly
decoded PNG**, not the original progressive JPEG, so that the recorded reference
isolates the resize arithmetic from JPEG-decoder differences between OpenCV's
libjpeg-turbo and the Rust `image` crate's decoder. `cargo xtask record-fixtures`
also runs this script against the raw JPEG once, purely to report the decoder
delta in `docs/GOLDEN_CALIBRATION.md`.

Usage: record_inter_area.py <input> <output> <width> <height>
"""

import hashlib
import json
import sys

import cv2
import numpy as np


def main(argv):
    if len(argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    src_path, out_path, width, height = argv[1], argv[2], int(argv[3]), int(argv[4])

    # IMREAD_COLOR gives BGR; keep everything in BGR and let cv2.imwrite write it back
    # as RGB-ordered PNG, so no channel juggling can silently swap R and B.
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {src_path}")
    if img.dtype != np.uint8:
        raise SystemExit(f"{src_path} decoded as {img.dtype}, expected uint8")

    out = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    if not cv2.imwrite(out_path, out):
        raise SystemExit(f"could not write {out_path}")

    with open(out_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()

    print(
        json.dumps(
            {
                "tool": "cv2.resize/INTER_AREA",
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
                "python": sys.version.split()[0],
                "source": src_path,
                "source_size": [int(img.shape[1]), int(img.shape[0])],
                "output": out_path,
                "output_size": [width, height],
                "sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
