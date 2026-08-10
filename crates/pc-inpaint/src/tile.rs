//! spec §16.38 items 5(b)–(e) — the tile cover and pixel ownership.
//!
//! The policy is **declared** in the spec rather than left to this file (item 5's own
//! preamble: "a policy nobody wrote down gets inferred differently by every later reader"), so
//! every clause below is a transcription and the tests are its lock.

use pc_core::Rect;
use pc_imageops::BinaryMask;

use crate::TILE;

/// The sentinel [`owner_map`] uses for "no window owns this pixel". A page pixel that is not a
/// fill pixel carries this; a **fill** pixel that carries it is a hole in the cover, which is
/// exactly what §16.38 item 5(e)'s "exactly once" obligation forbids.
pub const UNOWNED: u32 = u32::MAX;

/// One 512×512 inference window, and which merged rectangle produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileWindow {
    /// Always `TILE` wide and `TILE` tall. `x1`/`y1` may be negative only if the tiling canvas
    /// is smaller than `TILE`, which [`crate::inpaint_page`] rules out by enlarging the canvas
    /// per item 5(d).
    pub window: Rect,
    /// Index into the merged-rectangle slice this window covers.
    pub merged_index: usize,
}

/// spec §16.38 item 5(b): "Take the padded boxes of item 3(e) and close them transitively
/// under rectangle intersection, replacing each intersecting group by its bounding union until
/// no two rectangles intersect."
///
/// **The order-independence argument, transcribed with its correction, because the obvious one
/// does not work.** "The transitive closure of an intersection relation is order-independent"
/// would suffice only if merging did not change the relation, and it does: replacing a group by
/// its bounding union can create intersections no pair in the original relation had. The
/// conclusion holds by a different route — the merge step is **monotone** (unioning rectangles
/// only grows them, and a grown rectangle intersects a superset of what it intersected before)
/// and the procedure runs until no two rectangles intersect, so it computes the **least
/// fixpoint** of a monotone operator on a finite set, which is unique and hence independent of
/// visiting order.
///
/// The implementation is the naive run-to-fixpoint loop, deliberately: it is the definition,
/// and the number of padded boxes on a page is small.
///
/// Returned **sorted by `(y1, x1, y2, x2)`** — item 5(e)'s ordering, with `y2`/`x2` appended
/// only to make ties total (two distinct non-degenerate non-intersecting rectangles cannot
/// share a top-left corner, so the extra keys never actually decide anything).
pub fn merge_rects(rects: &[Rect]) -> Vec<Rect> {
    let mut merged: Vec<Rect> = rects
        .iter()
        .copied()
        .filter(|rect| rect.width() > 0 && rect.height() > 0)
        .collect();

    loop {
        let mut absorbed = None;
        'outer: for i in 0..merged.len() {
            for j in (i + 1)..merged.len() {
                if intersects(&merged[i], &merged[j]) {
                    absorbed = Some((i, j));
                    break 'outer;
                }
            }
        }
        let Some((i, j)) = absorbed else { break };
        let union = merged[i].merge(&merged[j]);
        merged.remove(j);
        merged[i] = union;
    }

    merged.sort_by_key(|rect| (rect.y1, rect.x1, rect.y2, rect.x2));
    merged
}

/// Strict, half-open intersection: `x2`/`y2` are exclusive (§16.9 item 1), so two rectangles
/// that merely touch along an edge do **not** intersect.
fn intersects(a: &Rect, b: &Rect) -> bool {
    a.x1 < b.x2 && b.x1 < a.x2 && a.y1 < b.y2 && b.y1 < a.y2
}

/// spec §16.38 item 5(c) — every window, before the fill-emptiness drop.
///
///   * "A merged rectangle whose width **and** height are both `<= 512` gets one 512×512
///     window centred on it and then translated minimally to lie inside the frame."
///   * "A merged rectangle exceeding 512 on either axis is covered by a stride-512 lattice
///     anchored at its own top-left corner, with the final row and column translated inward to
///     stay in frame."
///
/// Note the two cases are chosen for the **rectangle**, not per axis: a 900×100 rectangle takes
/// the lattice branch on both axes, so its single `y` anchor is its own `y1` translated inward
/// — not centred. Windows are emitted merged-rectangle by merged-rectangle, and **row-major**
/// within one, which is the order item 5(e)'s ownership rule reads.
///
/// `canvas` must be at least `TILE` on both axes; [`crate::inpaint_page`] guarantees that by
/// enlarging a smaller page per item 5(d).
pub fn tile_windows(merged: &[Rect], canvas: (u32, u32)) -> Vec<TileWindow> {
    let mut out = Vec::new();
    for (merged_index, rect) in merged.iter().enumerate() {
        let fits = rect.width() <= TILE as i32 && rect.height() <= TILE as i32;
        let xs = if fits {
            vec![centred_anchor(rect.x1, rect.x2, canvas.0)]
        } else {
            lattice_anchors(rect.x1, rect.x2, canvas.0)
        };
        let ys = if fits {
            vec![centred_anchor(rect.y1, rect.y2, canvas.1)]
        } else {
            lattice_anchors(rect.y1, rect.y2, canvas.1)
        };
        for y in &ys {
            for x in &xs {
                out.push(TileWindow {
                    window: Rect::new(*x, *y, *x + TILE as i32, *y + TILE as i32),
                    merged_index,
                });
            }
        }
    }
    out
}

/// "one 512×512 window centred on it and then translated minimally to lie inside the frame".
///
/// The centring pads the short side by `(TILE - extent) / 2` — integer division, so an odd
/// slack puts the extra pixel on the **right/bottom**. "Translated minimally" is the clamp: the
/// nearest in-frame anchor to the centred one.
fn centred_anchor(start: i32, end: i32, canvas: u32) -> i32 {
    let extent = end - start;
    let anchor = start - (TILE as i32 - extent) / 2;
    clamp_anchor(anchor, canvas)
}

/// "a stride-512 lattice anchored at its own top-left corner, with the final row and column
/// translated inward to stay in frame".
///
/// Anchors are `start, start + 512, …` while the previous anchor has not yet reached `end`;
/// each is then clamped into frame, which is what performs the inward translation of the final
/// row/column. Clamping can collapse two anchors onto one (a canvas barely wider than `TILE`),
/// so equal neighbours are de-duplicated — a duplicate window would be a second inference call
/// producing the same tile, and the later one would own nothing.
fn lattice_anchors(start: i32, end: i32, canvas: u32) -> Vec<i32> {
    let mut anchors = Vec::new();
    let mut at = start;
    loop {
        let clamped = clamp_anchor(at, canvas);
        if anchors.last() != Some(&clamped) {
            anchors.push(clamped);
        }
        at += TILE as i32;
        if at >= end {
            break;
        }
    }
    anchors
}

fn clamp_anchor(anchor: i32, canvas: u32) -> i32 {
    let last = canvas as i32 - TILE as i32;
    anchor.clamp(0, last.max(0))
}

/// spec §16.38 item 5(c)'s last clause: "windows whose intersection with the fill mask is empty
/// are dropped".
///
/// A window can be empty even though its merged rectangle is not: the rectangle is the bounding
/// union of padded boxes, so its interior includes area no write pixel reaches, and a lattice
/// window can land entirely in that gap.
///
/// `write_region` is the set of pixels the tile loop may write. [`crate::inpaint_page`] passes
/// `final_mask > 0` rather than the bare fill mask, for the reason recorded at its call site —
/// a superset, so this drops a subset of what item 5(c)'s literal text drops and no write pixel
/// can be left unowned.
pub fn tile_cover(
    merged: &[Rect],
    write_region: &BinaryMask,
    canvas: (u32, u32),
) -> Vec<TileWindow> {
    tile_windows(merged, canvas)
        .into_iter()
        .filter(|tile| window_touches(&tile.window, write_region))
        .collect()
}

fn window_touches(window: &Rect, fill: &BinaryMask) -> bool {
    let (width, height) = fill.dimensions();
    let x_start = window.x1.max(0);
    let y_start = window.y1.max(0);
    let x_end = window.x2.min(width as i32);
    let y_end = window.y2.min(height as i32);
    for y in y_start..y_end {
        for x in x_start..x_end {
            if fill.get(x as u32, y as u32) {
                return true;
            }
        }
    }
    false
}

/// spec §16.38 item 5(e) — "Every fill pixel is written exactly once."
///
/// Quoted in full because the whole rule is here: "Windows from different merged rectangles may
/// overlap even though the rectangles do not, so ownership is assigned rather than left to write
/// order: order the merged rectangles by `(y1, x1)` and, within one, its lattice windows
/// row-major; a fill pixel belongs to the first window in that order whose region contains it,
/// and write-back touches only owned pixels, masked by `faded_fill AND isolation`."
///
/// Returns a `write_region`-sized row-major map: the index into `cover` of the owning window, or
/// [`UNOWNED`] where the pixel is outside the write region — **or** where the cover has a hole,
/// which is the failure mode this map makes visible instead of silent. `cover` is already in item
/// 5(e)'s order, because [`merge_rects`] sorts and [`tile_windows`] emits row-major.
///
/// `write_region` is the set of writable pixels; [`crate::inpaint_page`] passes `final_mask > 0`,
/// which contains the fill mask, so item 5(e)'s "every fill pixel is written exactly once" is
/// implied by "every write pixel has exactly one owner". The reading is recorded at that call site.
pub fn owner_map(cover: &[TileWindow], write_region: &BinaryMask) -> Vec<u32> {
    let fill = write_region;
    let (width, height) = fill.dimensions();
    let mut owners = vec![UNOWNED; (width as usize) * (height as usize)];

    for (index, tile) in cover.iter().enumerate() {
        let x_start = tile.window.x1.max(0);
        let y_start = tile.window.y1.max(0);
        let x_end = tile.window.x2.min(width as i32);
        let y_end = tile.window.y2.min(height as i32);
        for y in y_start..y_end {
            for x in x_start..x_end {
                if !fill.get(x as u32, y as u32) {
                    continue;
                }
                let slot = &mut owners[(y as usize) * (width as usize) + (x as usize)];
                if *slot == UNOWNED {
                    *slot = index as u32;
                }
            }
        }
    }

    owners
}
