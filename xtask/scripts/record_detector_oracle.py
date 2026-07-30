#!/usr/bin/env python3
"""Record the pinned PanelCleaner detector oracle.

Usage: record_detector_oracle.py <checkout> <model> <page> <out_dir> <stem>
       record_detector_oracle.py --self-test

The normal invocation is deliberately quiet on stdout: the final line is one JSON
manifest for ``cargo xtask``.  Diagnostics and failures go to stderr.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from contextlib import redirect_stdout


UPSTREAM_COMMIT = "0afa21fd6caab5bee0ab8ef51a5a19fc4bd9dda3"
MODEL_DIGEST = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"
UPSTREAM_BACKEND = "cv2_dnn"
FIELDS = ["xyxy", "lines", "language", "vertical", "font_size"]
DERIVATIONS = (
    "yolo_unioned",
    "yolo_synthesized_corners",
    "yolo_split",
    "dbnet_scattered",
)


class RecordingError(RuntimeError):
    """A fatal recording error which must not produce a partial recording."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise RecordingError(f"could not read {path.name}: {exc}") from exc


def fail(message: str) -> None:
    raise RecordingError(message)


def rect_union(left: list[int], right: list[int]) -> list[int]:
    return [
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    ]


def bbox_lines(lines: list[list[list[int]]]) -> list[int] | None:
    points = [point for polygon in lines for point in polygon]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def check_derivation_law(block: dict, parent_xyxy: list[int] | None = None) -> None:
    """Apply §16.27 item 1(c)'s exact, epsilon-free law to one tagged block."""

    derivation = block["derivation"]
    xyxy = block["xyxy"]
    lines = block["lines_pre_expand"]
    line_bbox = bbox_lines(lines)
    if line_bbox is None:
        fail(f"{derivation} block has no pre-expansion line points")

    if derivation == "yolo_unioned":
        rect_yolo = block["rect_yolo"]
        if rect_yolo is None:
            fail("yolo_unioned block has no rect_yolo")
        if xyxy != rect_union(rect_yolo, line_bbox):
            fail(
                "yolo_unioned law failed: "
                f"xyxy={xyxy}, union={rect_union(rect_yolo, line_bbox)}"
            )
    elif derivation == "yolo_synthesized_corners":
        rect_yolo = block["rect_yolo"]
        if rect_yolo is None or rect_yolo != xyxy or xyxy != line_bbox:
            fail(
                "yolo_synthesized_corners law failed: "
                f"xyxy={xyxy}, rect_yolo={rect_yolo}, lines_bbox={line_bbox}"
            )
        if len(block["lines"]) != 1:
            fail("yolo_synthesized_corners block does not have exactly one served line")
    elif derivation == "yolo_split":
        rect_yolo = block["rect_yolo"]
        if rect_yolo is None or xyxy != line_bbox:
            fail(
                "yolo_split law failed: "
                f"xyxy={xyxy}, rect_yolo={rect_yolo}, lines_bbox={line_bbox}"
            )
        if parent_xyxy is None or rect_yolo != parent_xyxy:
            fail(
                "yolo_split parent law failed: "
                f"parent={parent_xyxy}, rect_yolo={rect_yolo}"
            )
    elif derivation == "dbnet_scattered":
        if block["rect_yolo"] is not None or xyxy != line_bbox:
            fail(
                "dbnet_scattered law failed: "
                f"xyxy={xyxy}, rect_yolo={block['rect_yolo']}, lines_bbox={line_bbox}"
            )
    else:
        fail(f"unrecognised derivation {derivation!r}")


def validate_expansion(block: dict, pre_font_size: int | float | None) -> None:
    delta = block["font_size"] - pre_font_size if pre_font_size is not None else None
    if delta is None:
        fail("missing pre-expansion font size")
    if block["eng_expanded"]:
        if block["language"] != "english" or block["vertical"]:
            fail("eng_expanded is set on a non-horizontal English block")
        if block["lines_pre_expand"] is None or block["expand_size"] is None:
            fail("eng_expanded requires lines_pre_expand and expand_size")
        if delta != block["expand_size"]:
            fail(
                "font-size expansion mismatch: "
                f"measured={delta}, recorded={block['expand_size']}"
            )
    else:
        if block["lines_pre_expand"] is not None or block["expand_size"] is not None:
            fail("non-expanded block carries expansion fields")
        if delta != 0:
            fail(f"font-size changed on a non-expanded block: delta={delta}")


def self_test() -> None:
    line = [[[10, 20], [40, 20], [40, 50], [10, 50]]]

    unioned = {
        "xyxy": [5, 15, 45, 55],
        "lines": line,
        "lines_pre_expand": line,
        "rect_yolo": [5, 15, 45, 55],
        "derivation": "yolo_unioned",
    }
    check_derivation_law(unioned)

    corners = {
        "xyxy": [10, 20, 40, 50],
        "lines": line,
        "lines_pre_expand": line,
        "rect_yolo": [10, 20, 40, 50],
        "derivation": "yolo_synthesized_corners",
    }
    check_derivation_law(corners)

    split = {
        "xyxy": [12, 20, 40, 50],
        "lines": [[[12, 20], [40, 20], [40, 50], [12, 50]]],
        "lines_pre_expand": [[[12, 20], [40, 20], [40, 50], [12, 50]]],
        "rect_yolo": [10, 20, 40, 50],
        "derivation": "yolo_split",
    }
    check_derivation_law(split, parent_xyxy=[10, 20, 40, 50])

    scattered = {
        "xyxy": [10, 20, 40, 50],
        "lines": line,
        "lines_pre_expand": line,
        "rect_yolo": None,
        "derivation": "dbnet_scattered",
    }
    check_derivation_law(scattered)

    expanded = {
        "language": "english",
        "vertical": False,
        "font_size": 14,
        "eng_expanded": True,
        "lines_pre_expand": line,
        "expand_size": 2,
    }
    validate_expansion(expanded, 12)

    def expects_failure(fn, label: str) -> None:
        try:
            fn()
        except RecordingError:
            return
        fail(f"self-test hard-fail path did not fire: {label}")

    bad_union = dict(unioned, xyxy=[5, 15, 44, 55])
    expects_failure(lambda: check_derivation_law(bad_union), "union law")
    bad_expansion = dict(expanded, expand_size=3)
    expects_failure(lambda: validate_expansion(bad_expansion, 12), "expansion delta")
    expects_failure(lambda: check_derivation_law({**scattered, "derivation": "other"}), "closed enum")

    if sha256_bytes(b"abc") != (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    ):
        fail("sha256 helper failed the abc vector")
    print("self-test: ok")


class _Tag:
    def __init__(self, *, yolo_index: int | None = None, rect_yolo=None):
        self.yolo_index = yolo_index
        self.rect_yolo = rect_yolo
        self.synthesized = False
        self.split = False
        self.parent_xyxy = None
        self.pre_lines = None
        self.pre_font_size = None
        self.pre_xyxy = None


class _Context:
    def __init__(self, textblock):
        self.textblock = textblock
        self.yolo = []
        self.used_yolo = set()
        self.synth_rects = []
        self.tags_by_id = {}
        self.final_tags = []
        self.instrumented_blocks = None
        self.pre_filter_mask_scores = []
        self.nms_before_scale = None
        self.base_after_scale = None
        self.confidence_decline_reason = None


def _as_int_rect(value) -> list[int]:
    return [int(value[0]), int(value[1]), int(value[2]), int(value[3])]


def _normalise_lines(lines) -> list[list[list[int]]]:
    if lines is None:
        return []
    return [
        [[int(point[0]), int(point[1])] for point in polygon]
        for polygon in lines
    ]


def _language_for_class(cls: int) -> str | None:
    if cls == 0:
        return "english"
    if cls == 1:
        return "japanese"
    if cls == 2:
        return None
    fail(f"upstream returned an unknown class index {cls}")


def _language_for_block(language: str) -> str | None:
    if language == "eng":
        return "english"
    if language == "ja":
        return "japanese"
    if language == "unknown":
        return None
    fail(f"upstream returned an unknown language code {language!r}")


def _yolo_entries(ctx: _Context, blks) -> None:
    boxes, classes, confs = blks
    if len(boxes) != len(classes) or len(boxes) != len(confs):
        fail("group_output yolo arrays have different lengths")
    ctx.yolo = []
    for index, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
        cls_int = int(cls)
        ctx.yolo.append(
            {
                "index": index,
                "xyxy": _as_int_rect(box),
                "class_index": cls_int,
                "language": _language_for_class(cls_int),
                "confidence": float(conf),
            }
        )


def _line_assignment_counts(blks, lines) -> list[int]:
    boxes = [_as_int_rect(box) for box in blks[0]]
    counts = [0] * len(boxes)
    for line in lines:
        bx1, bx2 = int(line[:, 0].min()), int(line[:, 0].max())
        by1, by2 = int(line[:, 1].min()), int(line[:, 1].max())
        line_area = (by2 - by1) * (bx2 - bx1)
        if line_area == 0:
            fail("zero-area DBNet line encountered while recording diagnostics")
        best_score, best_index = -1.0, -1
        for index, box in enumerate(boxes):
            x1, y1 = max(box[0], bx1), max(box[1], by1)
            x2, y2 = min(box[2], bx2), min(box[3], by2)
            score = -1 if y2 < y1 or x2 < x1 else (y2 - y1) * (x2 - x1)
            score /= line_area
            if best_score < score:
                best_score, best_index = score, index
        if best_score > 0.4:
            counts[best_index] += 1
    return counts


def _mask_score(mask, rect):
    x1, y1, x2, y2 = rect
    score = float(mask[y1:y2, x1:x2].mean() / 255)
    return score if math.isfinite(score) else None


def _tag_for_block(ctx: _Context, blk) -> _Tag | None:
    existing = ctx.tags_by_id.get(id(blk))
    if existing is not None:
        return existing
    rect = _as_int_rect(blk.xyxy)
    language = _language_for_block(blk.language)
    rect_candidates = [entry for entry in ctx.yolo if entry["xyxy"] == rect]
    candidates = [
        entry
        for entry in rect_candidates
        if entry["index"] not in ctx.used_yolo
        and entry["language"] == language
    ]
    if candidates:
        if len(candidates) > 1:
            ctx.confidence_decline_reason = (
                f"yolo block {rect} has {len(candidates)} indistinguishable yolo candidates"
            )
        entry = candidates[0]
        ctx.used_yolo.add(entry["index"])
        tag = _Tag(yolo_index=entry["index"], rect_yolo=entry["xyxy"][:])
        ctx.tags_by_id[id(blk)] = tag
        return tag
    if rect_candidates:
        fail(
            f"language/class disagreement on yolo block {rect}: "
            f"language={language!r}, candidates={[entry['class_index'] for entry in rect_candidates]}"
        )
    return None


def _block_snapshot(blk) -> dict:
    return {
        "xyxy": _as_int_rect(blk.xyxy),
        "lines": _normalise_lines(blk.lines),
        "language": _language_for_block(blk.language),
        "vertical": bool(blk.vertical),
        "font_size": blk.font_size,
    }


def _install_hooks(ctx: _Context):
    tb = ctx.textblock
    originals = {name: getattr(tb, name) for name in (
        "merge_textlines",
        "split_textblk",
        "xywh2xyxypoly",
        "examine_textblk",
        "sort_textblk_list",
    )}

    def merge_textlines(value):
        result = originals["merge_textlines"](value)
        for blk in result:
            tag = _Tag()
            tag.pre_lines = None
            ctx.tags_by_id[id(blk)] = tag
        return result

    def split_textblk(blk):
        split, result = originals["split_textblk"](blk)
        source_tag = ctx.tags_by_id.get(id(blk))
        for sub in result:
            tag = _Tag()
            tag.split = bool(split)
            tag.yolo_index = source_tag.yolo_index if source_tag else None
            tag.rect_yolo = source_tag.rect_yolo[:] if source_tag and source_tag.rect_yolo else None
            tag.synthesized = source_tag.synthesized if source_tag else False
            tag.parent_xyxy = _as_int_rect(blk.xyxy) if split else None
            if split and source_tag and source_tag.synthesized:
                fail("a synthesized-corners block entered split_textblk")
            ctx.tags_by_id[id(sub)] = tag
        return split, result

    def xywh2xyxypoly(value, *args, **kwargs):
        result = originals["xywh2xyxypoly"](value, *args, **kwargs)
        for row in value:
            rect = _as_int_rect([row[0], row[1], row[0] + row[2], row[1] + row[3]])
            ctx.synth_rects.append(rect)
        return result

    def examine_textblk(blk, im_w, im_h, sort=False):
        tag = _tag_for_block(ctx, blk)
        if tag is None:
            # This is either a scattered line or a yolo block that is going to be filtered.
            # The latter never reaches this hook, so no untagged final block is acceptable.
            tag = _Tag()
            ctx.tags_by_id[id(blk)] = tag
        if tag.yolo_index is not None and _as_int_rect(blk.xyxy) in ctx.synth_rects:
            tag.synthesized = True
        result = originals["examine_textblk"](blk, im_w, im_h, sort=sort)
        return result

    def sort_textblk_list(value, im_w, im_h):
        result = originals["sort_textblk_list"](value, im_w, im_h)
        for blk in result:
            tag = ctx.tags_by_id.get(id(blk))
            if tag is None:
                fail(
                    "sort_textblk_list returned an untagged block: "
                    f"xyxy={getattr(blk, 'xyxy', None)}, "
                    f"language={getattr(blk, 'language', None)!r}, "
                    f"lines={len(getattr(blk, 'lines', []))}"
                )
            tag.pre_lines = _normalise_lines(blk.lines)
            tag.pre_font_size = blk.font_size
            tag.pre_xyxy = _as_int_rect(blk.xyxy)
        return result

    tb.merge_textlines = merge_textlines
    tb.split_textblk = split_textblk
    tb.xywh2xyxypoly = xywh2xyxypoly
    tb.examine_textblk = examine_textblk
    tb.sort_textblk_list = sort_textblk_list
    return originals


def _restore_hooks(module, originals) -> None:
    for name, function in originals.items():
        setattr(module, name, function)


def _block_fields_equal(left, right) -> bool:
    return (
        _as_int_rect(left.xyxy) == _as_int_rect(right.xyxy)
        and _normalise_lines(left.lines) == _normalise_lines(right.lines)
        and _language_for_block(left.language) == _language_for_block(right.language)
        and bool(left.vertical) == bool(right.vertical)
        and left.font_size == right.font_size
    )


def _group_output_wrapper(real_group_output, ctx: _Context, blks, lines, im_w, im_h, mask=None, sort_blklist=True):
    ctx.yolo = []
    _yolo_entries(ctx, blks)
    assignment_counts = _line_assignment_counts(blks, lines)
    for entry, count in zip(ctx.yolo, assignment_counts):
        entry["len_lines"] = count
        entry["mask_score"] = None if count else _mask_score(mask, entry["xyxy"]) if mask is not None else None
    ctx.pre_filter_mask_scores = [
        {
            "index": entry["index"],
            "xyxy": entry["xyxy"],
            "mask_score": entry["mask_score"],
            "len_lines": entry["len_lines"],
        }
        for entry in ctx.yolo
    ]

    pristine_inputs = copy.deepcopy((blks, lines, im_w, im_h, mask))
    pristine = real_group_output(
        pristine_inputs[0], pristine_inputs[1], pristine_inputs[2], pristine_inputs[3],
        pristine_inputs[4], sort_blklist=sort_blklist
    )

    originals = _install_hooks(ctx)
    try:
        instrumented_inputs = copy.deepcopy((blks, lines, im_w, im_h, mask))
        instrumented = real_group_output(
            instrumented_inputs[0], instrumented_inputs[1], instrumented_inputs[2], instrumented_inputs[3],
            instrumented_inputs[4], sort_blklist=sort_blklist
        )
    finally:
        _restore_hooks(ctx.textblock, originals)

    if len(pristine) != len(instrumented):
        fail(f"pristine/instrumented block count differs: {len(pristine)} != {len(instrumented)}")
    for index, (clean, tagged) in enumerate(zip(pristine, instrumented)):
        if not _block_fields_equal(clean, tagged):
            fail(f"pristine/instrumented block {index} differs in one of {FIELDS}")
        tag = ctx.tags_by_id.get(id(tagged))
        if tag is None:
            fail(f"instrumented block {index} has no hook tag")
        ctx.final_tags.append(tag)
    ctx.instrumented_blocks = instrumented
    return pristine


def _make_block(block, tag: _Tag, ctx: _Context, confidence_allowed: bool, pre_base) -> dict:
    language = _language_for_block(block.language)
    if tag.yolo_index is not None:
        entry = ctx.yolo[tag.yolo_index]
        expected_language = entry["language"]
        if language != expected_language:
            fail(
                f"language/class disagreement on yolo block {tag.yolo_index}: "
                f"language={language!r}, class={entry['class_index']}"
            )

    served = _block_snapshot(block)
    pre_lines = tag.pre_lines
    if pre_lines is None:
        fail("final block did not pass through sort_textblk_list")
    pre_font = tag.pre_font_size
    expansion = served["font_size"] - pre_font
    is_expanded = (
        language == "english" and not served["vertical"] and len(served["lines"]) > 0
    )
    if is_expanded:
        expand_size = max(int(pre_font * 0.1), 2)
        if expansion != expand_size:
            fail(f"English expansion measured {expansion}, expected {expand_size}")
        lines_pre_expand = pre_lines
    else:
        expand_size = None
        lines_pre_expand = None
    result = {
        "xyxy": served["xyxy"],
        "lines": served["lines"],
        "confidence": (
            ctx.yolo[tag.yolo_index]["confidence"]
            if tag.yolo_index is not None and confidence_allowed
            else None
        ),
        "language": language,
        "vertical": served["vertical"],
        "font_size": served["font_size"],
        "base_xyxy_pretruncation": pre_base,
        "derivation": None,
        "rect_yolo": tag.rect_yolo[:] if tag.rect_yolo is not None else None,
        "eng_expanded": bool(is_expanded),
        "lines_pre_expand": lines_pre_expand,
        "expand_size": expand_size,
    }
    if tag.yolo_index is None:
        result["derivation"] = "dbnet_scattered"
    elif tag.split:
        result["derivation"] = "yolo_split"
    elif tag.synthesized:
        result["derivation"] = "yolo_synthesized_corners"
    else:
        result["derivation"] = "yolo_unioned"
    law_block = dict(result)
    law_block["lines_pre_expand"] = pre_lines
    check_derivation_law(law_block, tag.parent_xyxy)
    validate_expansion(result, pre_font)
    result.pop("vertical")
    result.pop("font_size")
    return result


def _base_pretruncation(ctx: _Context, index: int) -> list[float] | None:
    if ctx.base_after_scale is None or index >= len(ctx.base_after_scale):
        return None
    return [float(value) for value in ctx.base_after_scale[index]]


def _record_one(checkout: Path, model: Path, page: Path, out_dir: Path, stem: str) -> dict:
    if not checkout.is_dir():
        fail(f"upstream checkout does not exist: {checkout.name}")
    if model.suffix != ".onnx":
        fail("wrong backend: the model must be a .onnx file for cv2.dnn")
    actual_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if actual_commit.returncode != 0 or actual_commit.stdout.strip() != UPSTREAM_COMMIT:
        found = actual_commit.stdout.strip() or "unavailable"
        fail(f"wrong upstream commit: expected {UPSTREAM_COMMIT}, found {found}")
    if sha256_file(model) != MODEL_DIGEST:
        fail("model digest mismatch")
    if not page.is_file():
        fail(f"input page does not exist: {page.name}")

    # Heavy dependencies are imported only after --self-test has been handled.
    sys.path.insert(0, str(checkout))
    import cv2
    import numpy as np
    import torch
    import pcleaner
    from pcleaner.comic_text_detector import inference
    from pcleaner.comic_text_detector.utils import textblock
    from pcleaner.ctd_interface import resize_to_target
    from pcleaner.comic_text_detector.utils.textmask import REFINEMASK_ANNOTATION
    TextDetector = inference.TextDetector

    image_bgr = cv2.imread(str(page), cv2.IMREAD_COLOR)
    if image_bgr is None:
        fail(f"could not read {page.name}")
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        fail(f"{page.name} did not decode as an 8-bit 3-channel image")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    decoded_rgb_digest = sha256_bytes(np.ascontiguousarray(image_rgb).tobytes(order="C"))
    image, scale = resize_to_target(image_bgr, 1000, 4000)
    height, width = image.shape[:2]

    detector = TextDetector(model_path=str(model), input_size=1024, device="cpu")
    if getattr(detector, "backend", None) != "opencv":
        fail("wrong backend: TextDetector did not select cv2.dnn")
    if not isinstance(detector.model, cv2.dnn_Net):
        fail("wrong backend: detector model is not a cv2.dnn_Net")

    ctx = _Context(textblock)
    real_group_output = inference.group_output
    real_nms = inference.non_max_suppression
    real_postprocess_yolo = inference.postprocess_yolo

    def capture_nms(*args, **kwargs):
        result = real_nms(*args, **kwargs)
        try:
            ctx.nms_before_scale = copy.deepcopy(result[0].detach().cpu().numpy())
        except AttributeError:
            ctx.nms_before_scale = copy.deepcopy(np.asarray(result[0]))
        return result

    def wrapped_group_output(*args, **kwargs):
        return _group_output_wrapper(real_group_output, ctx, *args, **kwargs)

    def capture_postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
        result = real_postprocess_yolo(
            det, conf_thresh, nms_thresh, resize_ratio, sort_func=sort_func
        )
        if ctx.nms_before_scale is not None and sort_func is None:
            scaled = ctx.nms_before_scale.copy()
            scaled[..., [0, 2]] *= resize_ratio[0]
            scaled[..., [1, 3]] *= resize_ratio[1]
            ctx.base_after_scale = scaled[..., 0:4].tolist()
        return result

    inference.non_max_suppression = capture_nms
    inference.postprocess_yolo = capture_postprocess_yolo
    inference.group_output = wrapped_group_output
    try:
        with redirect_stdout(io.StringIO()):
            _mask, _mask_refined, blocks = detector(
                image, refine_mode=REFINEMASK_ANNOTATION, keep_undetected_mask=True
            )
    finally:
        inference.non_max_suppression = real_nms
        inference.postprocess_yolo = real_postprocess_yolo
        inference.group_output = real_group_output

    if len(ctx.final_tags) != len(blocks):
        fail("instrumentation did not produce one tag per returned block")
    confidence_allowed = ctx.confidence_decline_reason is None
    # `postprocess_yolo`'s float boxes are captured after network-to-image scaling and before
    # its exact `astype(np.int32)` truncation.  The NMS wrapper supplies the unmodified tensor;
    # the postprocess wrapper applies the same resize factors without reimplementing NMS.
    pre_base = [_base_pretruncation(ctx, index) for index in range(len(ctx.yolo))]
    blocks_out = [
        _make_block(
            block,
            tag,
            ctx,
            confidence_allowed,
            pre_base[tag.yolo_index] if tag.yolo_index is not None else None,
        )
        for block, tag in zip(blocks, ctx.final_tags)
    ]

    derivation_census = {name: 0 for name in DERIVATIONS}
    for block in blocks_out:
        derivation = block["derivation"]
        if derivation not in derivation_census:
            fail(f"unclassifiable block derivation {derivation!r}")
        derivation_census[derivation] += 1

    equality = {
        "fields": FIELDS,
        "pristine": [_block_snapshot(block) for block in blocks],
        "instrumented": [_block_snapshot(block) for block in ctx.instrumented_blocks],
    }
    if equality["pristine"] != equality["instrumented"]:
        fail("pristine/instrumented equality witness disagrees")
    if not ctx.instrumented_blocks:
        # The equality check is still meaningful on an empty page, but the real oracle corpus
        # and its classification census require a non-vacuous returned block list.
        fail("upstream returned no blocks")

    pre_filter_blocks = []
    for entry in ctx.yolo:
        pre_filter_blocks.append(
            {
                "xyxy": entry["xyxy"],
                "lines": [],
                "confidence": None if not confidence_allowed else entry["confidence"],
                "language": entry["language"],
                "base_xyxy_pretruncation": pre_base[entry["index"]],
                "derivation": None,
                "rect_yolo": None,
                "eng_expanded": False,
                "lines_pre_expand": None,
                "expand_size": None,
            }
        )

    zero_lines = sum(1 for block in blocks_out if not block["lines"])
    input_page_sha256 = sha256_file(page)
    model_digest = sha256_file(model)
    oracle = {
        "scale": float(scale),
        "image_size": [int(width), int(height)],
        "blocks": blocks_out,
        "pre_filter_blocks": pre_filter_blocks,
    }

    dependency_versions = {
        "pcleaner": str(pcleaner.__version__),
        "opencv": str(cv2.__version__),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
    }
    command_line = (
        "record_detector_oracle.py PanelCleaner comictextdetector.pt.onnx "
        f"{page.name} {out_dir.name} {stem}"
    )
    diagnostics = {
        "instrumented_pristine_equality": {
            "checked": True,
            "blocks": len(blocks_out),
            "fields": FIELDS,
        },
        "derivation_census": derivation_census,
        "line_count_census": {"blocks": len(blocks_out), "with_zero_lines": zero_lines},
        "upstream_pre_filter_mask_scores": ctx.pre_filter_mask_scores,
        "confidence_mapping": (
            {"status": "declined", "reason": ctx.confidence_decline_reason}
            if not confidence_allowed
            else {"status": "ok"}
        ),
    }
    upstream = {
        "version": str(pcleaner.__version__),
        "commit": UPSTREAM_COMMIT,
        "command_line": command_line,
        "backend": UPSTREAM_BACKEND,
        "dependency_versions": dependency_versions,
        "decoded_rgb_digest": decoded_rgb_digest,
        "decoded_from": "input_page",
    }

    artifact_bytes = {
        f"{stem}_upstream_oracle.json": json.dumps(oracle, indent=2, ensure_ascii=False) + "\n",
        f"{stem}_upstream_group_output_equality.json": json.dumps(
            equality, indent=2, ensure_ascii=False
        )
        + "\n",
    }
    records = [
        {
            "name": "upstream_oracle" if name.endswith("_upstream_oracle.json") else "group_output_equality",
            "output": name,
            "sha256": sha256_bytes(text.encode()),
        }
        for name, text in artifact_bytes.items()
    ]
    manifest = {
        "tool": "PanelCleaner detector oracle recorder",
        "command_line": command_line,
        "python": sys.version.split()[0],
        "opencv_version": str(cv2.__version__),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "upstream": upstream,
        "model": model.name,
        "model_digest": model_digest,
        "input_page": page.name,
        "input_page_sha256": input_page_sha256,
        "records": records,
        "diagnostics": diagnostics,
    }

    generated = [oracle, equality, manifest]
    for value in generated:
        _assert_no_absolute_paths(value)
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = []
    try:
        for name, text in artifact_bytes.items():
            target = out_dir / name
            temp = out_dir / f".{name}.tmp"
            temp.write_text(text, encoding="utf-8")
            temp_paths.append(temp)
            os.replace(temp, target)
    except OSError as exc:
        for temp in temp_paths:
            try:
                temp.unlink()
            except OSError:
                pass
        raise RecordingError(f"could not write recording artifacts: {exc}") from exc
    return manifest


def _assert_no_absolute_paths(value) -> None:
    if isinstance(value, str):
        if os.path.isabs(value) or (len(value) > 2 and value[1] == ":" and value[2] in "\\/"):
            fail(f"absolute path would leak into an artifact: {value}")
    elif isinstance(value, dict):
        for child in value.values():
            _assert_no_absolute_paths(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_absolute_paths(child)


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        try:
            self_test()
        except RecordingError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if len(argv) != 6:
        print(__doc__, file=sys.stderr)
        return 2
    chatter = io.StringIO()
    try:
        with redirect_stdout(chatter):
            manifest = _record_one(
                Path(argv[1]), Path(argv[2]), Path(argv[3]), Path(argv[4]), argv[5]
            )
    except (RecordingError, OSError, subprocess.SubprocessError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if chatter.getvalue():
        print(chatter.getvalue(), file=sys.stderr, end="")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
