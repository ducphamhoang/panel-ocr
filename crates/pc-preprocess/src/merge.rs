//! Task P3 — spec §9.3 steps 4 and 9: the two overlap-resolution passes.
//!
//! Both are the same shape: a FIFO queue in index order; pop the front box; select its
//! partners **from the queue as it stands before any merging** (spec §16.8 item 2 —
//! snapshot semantics, which is what makes §9.7(A)4's `[A∪B, C]` true); merge them all
//! into the popped box; emit it. One pass, no transitive re-scan.
//!
//! DEVIATION(2): upstream iterates a Python `set` here, so its merge order — and
//! therefore the merged boxes themselves — differ between runs. We use index order.
//! Required by §5.7's determinism rule.

use pc_core::{Rect, TextBox};

/// spec §9.3 step 4 (`resolve_total_overlaps`): merge boxes that contain each other's
/// centres.
///
/// DEVIATION(3): upstream mutates its `boxes` list without touching the parallel
/// `box_language` list, silently desynchronising the two whenever a merge happens. Our
/// `TextBox` pairing makes that impossible, so we specify it: **the earliest (popped)
/// box's language wins**.
pub fn resolve_total_overlaps(boxes: Vec<TextBox>) -> Vec<TextBox> {
    let mut queue: Vec<TextBox> = boxes;
    let mut resolved: Vec<TextBox> = Vec::with_capacity(queue.len());

    while !queue.is_empty() {
        let mut current = queue.remove(0);
        // §16.8 item 2: partners are chosen against `current` as popped, before any
        // merge widens it.
        let popped = current.rect;
        let mut remaining: Vec<TextBox> = Vec::with_capacity(queue.len());
        for candidate in std::mem::take(&mut queue) {
            if popped.overlaps_center(&candidate.rect) {
                current.rect = current.rect.merge(&candidate.rect);
            } else {
                remaining.push(candidate);
            }
        }
        queue = remaining;
        resolved.push(current);
    }

    resolved
}

/// spec §9.3 step 9 (`resolve_overlaps`): merge extended boxes that overlap by more than
/// `threshold_percent` of the smaller box's area (`Rect::overlaps` is strictly greater).
pub fn resolve_overlaps(rects: Vec<Rect>, threshold_percent: f64) -> Vec<Rect> {
    let mut queue: Vec<Rect> = rects;
    let mut resolved: Vec<Rect> = Vec::with_capacity(queue.len());

    while !queue.is_empty() {
        let popped = queue.remove(0);
        let mut merged = popped;
        // §16.8 item 2: snapshot — every candidate is tested against `popped`, never
        // against the progressively grown `merged`.
        let mut remaining: Vec<Rect> = Vec::with_capacity(queue.len());
        for candidate in std::mem::take(&mut queue) {
            if popped.overlaps(&candidate, threshold_percent) {
                merged = merged.merge(&candidate);
            } else {
                remaining.push(candidate);
            }
        }
        queue = remaining;
        resolved.push(merged);
    }

    resolved
}
