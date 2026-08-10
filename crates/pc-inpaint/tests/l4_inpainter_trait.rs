//! L4 — the [`pc_inpaint::Inpainter`] seam L5 replaces: spec §16.38 items 1(b), 1(d), 1(g), 5(d) and
//! 9(g). Every assertion here is a constraint L5's `ort`-backed implementation must satisfy, pinned
//! now so L5 is a drop-in rather than a renegotiation.

mod common;

use common::{flat_rgb, gray_rect, radii, region, transparent};
use image::Rgb;
use pc_core::{Rect, StageError};
use pc_inpaint::stub::{StubInpainter, StubMode};
use pc_inpaint::{inpaint_page, PageInput, TILE};

fn one_region() -> Vec<pc_core::MaskRegionStats> {
    vec![region(Rect::new(290, 290, 310, 310), 30.0, true, Some(2))]
}

/// §16.38 item 1(b): the pinned artifact's `image` input is `[batch, 3, 512, 512]` and its `mask` input
/// is `[batch, 1, 512, 512]`, with the `512`s as literal `dim_value`s; item 1(d) fed a real `ort`
/// session 256×256, 640×512 and 1024×1024 and got three identical rejections.
///
/// So `TILE` is a hard-coded `512` here, **not** read from `pc_inpaint::TILE`: if someone changes that
/// constant this test must fail, because changing it does not change what the model accepts.
#[test]
fn the_tile_side_is_512_and_every_call_receives_exactly_that_shape() {
    assert_eq!(TILE, 512, "the artifact's H and W axes are literal 512s");

    let config = pc_config::InpainterConfig::default();
    let original = flat_rgb((600, 600), [200, 200, 200]);
    let stub = StubInpainter::flat(0, 0, 0);

    inpaint_page(
        PageInput {
            original: &original,
            raw_mask: &gray_rect((600, 600), Rect::new(295, 295, 305, 305)),
            combined_mask: &transparent((600, 600)),
            noise_mask: None,
            regions: &one_region(),
            min_mask_thickness: 0,
            config: &config,
        },
        &stub,
    )
    .expect("the stub never fails");

    let seen = stub.seen();
    assert_eq!(seen.len(), 1);
    for call in &seen {
        assert_eq!(call.tile.dimensions(), (512, 512));
        assert_eq!(call.mask.dimensions(), (512, 512));
    }
}

/// §16.38 item 1(g): "Mask convention confirmed: value 1 = fill, 0 = keep." The mask the model receives
/// must be the fill mask cropped to the window — non-empty (or the call is pointless) and strictly
/// smaller than the tile (or the model is being asked to regenerate everything).
///
/// The bounds are hand-derived: the fill is a 10×10 blob dilated by `growth = 13`, so roughly 1150
/// pixels of a 262,144-pixel tile.
#[test]
fn the_mask_handed_to_the_model_is_the_window_crop_of_the_fill_mask() {
    let config = pc_config::InpainterConfig::default();
    let original = flat_rgb((600, 600), [200, 200, 200]);
    let stub = StubInpainter::flat(0, 0, 0);

    inpaint_page(
        PageInput {
            original: &original,
            raw_mask: &gray_rect((600, 600), Rect::new(295, 295, 305, 305)),
            combined_mask: &transparent((600, 600)),
            noise_mask: None,
            regions: &one_region(),
            min_mask_thickness: 0,
            config: &config,
        },
        &stub,
    )
    .expect("the stub never fails");

    let set = stub.seen()[0].mask.count_set();
    assert!(
        (800..2000).contains(&set),
        "the tile mask holds {set} set pixels; a 10x10 blob dilated by 13 is near 1150"
    );
    assert!(
        set < (512 * 512),
        "the mask must not be the whole tile — 1 = fill, 0 = keep"
    );
}

/// §16.38 item 5(d): "Pages smaller than 512 on either axis are edge-replicated up to 512 and the
/// result cropped back. Edge replication rather than a constant fill is the in-family choice: upstream's
/// own `grow_mask` pads with `mode='edge'` (`image_ops.py:812`)."
///
/// The page's last column is painted a distinctive colour, so a constant-fill pad (black, white or the
/// page's flat background) is distinguishable from replication.
#[test]
fn a_page_smaller_than_512_is_edge_replicated_into_the_tile_and_the_output_is_cropped_back() {
    let page = (100_u32, 80_u32);
    let config = radii(2, 2, 0.0, 1, 0);
    let mut original = flat_rgb(page, [200, 200, 200]);
    for y in 0..page.1 {
        original.put_pixel(page.0 - 1, y, Rgb([1, 2, 3]));
    }
    let stub = StubInpainter::flat(0, 0, 0);

    let output = inpaint_page(
        PageInput {
            original: &original,
            raw_mask: &gray_rect(page, Rect::new(48, 38, 52, 42)),
            combined_mask: &transparent(page),
            noise_mask: None,
            regions: &[region(Rect::new(45, 35, 55, 45), 0.0, true, Some(1))],
            min_mask_thickness: 0,
            config: &config,
        },
        &stub,
    )
    .expect("the stub never fails");

    assert_eq!(
        output.clean_inpaint.dimensions(),
        page,
        "the result is cropped back to the page"
    );
    assert_eq!(output.inpainting.dimensions(), page);

    let tile = &stub.seen()[0].tile;
    assert_eq!(tile.dimensions(), (512, 512));
    assert_eq!(
        *tile.get_pixel(300, 10),
        Rgb([1, 2, 3]),
        "x = 300 is past the 100-wide page, so it replicates column 99 — not a constant fill"
    );
    assert_eq!(
        *tile.get_pixel(10, 300),
        Rgb([200, 200, 200]),
        "y = 300 is past the 80-tall page, so it replicates row 79"
    );
    assert_eq!(
        *tile.get_pixel(99, 10),
        Rgb([1, 2, 3]),
        "and the last in-page column is genuinely that colour, so the check above is meaningful"
    );
}

/// §16.38 item 9(g), which declares this half of the split rather than leaving it to be inferred:
/// *"A failure **inside** `Session::run` on a tile receives the image and is therefore **per-image**:
/// `Failed { step: Inpaint }`, exit 2. … construction failures render as `StageError::Model`, per-tile
/// inference failures as `StageError::Inference`."*
///
/// The variant is what the pipeline's fatality classification reads, so the assertion is on the variant
/// and not on the message.
#[test]
fn a_per_tile_inference_failure_surfaces_as_stage_error_inference() {
    let config = pc_config::InpainterConfig::default();
    let original = flat_rgb((600, 600), [200, 200, 200]);
    let stub = StubInpainter::new(StubMode::Fail);

    let result = inpaint_page(
        PageInput {
            original: &original,
            raw_mask: &gray_rect((600, 600), Rect::new(295, 295, 305, 305)),
            combined_mask: &transparent((600, 600)),
            noise_mask: None,
            regions: &one_region(),
            min_mask_thickness: 0,
            config: &config,
        },
        &stub,
    );

    assert!(
        matches!(result, Err(StageError::Inference(_))),
        "a per-tile failure is `Inference`, never `Model` (which is reserved for the run-fatal \
         construction half) — got {result:?}",
        result = result.map(|_| "Ok")
    );
}

/// §16.38 item 1(c): the graph's *declared* output shape is unusable — "three of four axes symbolic, and
/// the **height axis carries the same symbol name as the batch axis**" — so "the implementation must
/// validate the *runtime* output shape and must never derive geometry from the declared one."
///
/// L5 owes that validation at the session; this pins the caller's own guard, so an implementation that
/// returns the wrong size is rejected with a per-image error instead of panicking on an out-of-bounds
/// read or silently misplacing pixels.
#[test]
fn an_inpainter_returning_a_wrongly_sized_tile_is_rejected_as_an_inference_error() {
    let config = pc_config::InpainterConfig::default();
    let original = flat_rgb((600, 600), [200, 200, 200]);
    let stub = StubInpainter::new(StubMode::WrongSize { side: 256 });

    let result = inpaint_page(
        PageInput {
            original: &original,
            raw_mask: &gray_rect((600, 600), Rect::new(295, 295, 305, 305)),
            combined_mask: &transparent((600, 600)),
            noise_mask: None,
            regions: &one_region(),
            min_mask_thickness: 0,
            config: &config,
        },
        &stub,
    );

    assert!(
        matches!(result, Err(StageError::Inference(_))),
        "got {result:?}",
        result = result.map(|_| "Ok")
    );
}

/// The stub that copies its input at an offset exists so a test can prove the tile the model sees is the
/// **window crop of the original**, not a blank or a re-scaled page: with `CopyOffset { 0, 0 }` the
/// returned tile is the input tile, so the composited fill area must carry the original's own pixels.
#[test]
fn a_pass_through_inpainter_leaves_the_page_pixel_identical_which_proves_the_tile_is_the_page_crop()
{
    let config = pc_config::InpainterConfig::default();
    let page = (600_u32, 600_u32);
    let mut original = flat_rgb(page, [200, 200, 200]);
    // A marker inside what will be the filled area.
    original.put_pixel(300, 300, Rgb([7, 8, 9]));
    let stub = StubInpainter::new(StubMode::CopyOffset { dx: 0, dy: 0 });

    let output = inpaint_page(
        PageInput {
            original: &original,
            raw_mask: &gray_rect(page, Rect::new(295, 295, 305, 305)),
            combined_mask: &transparent(page),
            noise_mask: None,
            regions: &one_region(),
            min_mask_thickness: 0,
            config: &config,
        },
        &stub,
    )
    .expect("the stub never fails");

    assert_eq!(
        output.clean_inpaint.get_pixel(300, 300).0[..3],
        [7, 8, 9],
        "a pass-through inpainter must reproduce the original pixel exactly, which it can only do \
         if the tile it was handed was the window crop of the page at the right offset"
    );
}
