//! Task F3 -- the recorded model signature, and the closure of the D4a self-reference.
//! spec §8.3 step 3, §16.15 (the `N_CLASSES` correction), §16.16. FROZEN per CLAUDE.md.
//!
//! **Why this file exists.** Every other runnable assertion about the detector model's
//! tensor arity descends from *our reading of the spec*: `d5_yolo.rs` pins the literals
//! `N_CLASSES == 2` / `ROW_STRIDE == 7`, and `d4_onnx.rs`'s `shipped_outputs()` fixture
//! writes `7` by hand. Those legs pin each other, so no single edit can move the constant
//! unnoticed -- but none of them measures the shipped artifact. Had §16.15's correction
//! landed on 6 instead of 7, the whole suite would have agreed with 6. The assertions
//! below are the one leg anchored to the file itself, via a signature recorded from the
//! sha256-verified `comictextdetector.pt.onnx` by `cargo xtask record-fixtures --only
//! model-signature` (task F3).
//!
//! **This is also the probe's acceptance test.** The recorder is a dependency-free
//! protobuf walk (ruled: the authority here is the bytes of a verified file, and any
//! correct deserializer of the published wire format is a measurement instrument, not a
//! reference implementation -- unlike §16.13 item 4's *behavioural* references). A
//! hand-rolled walk can be plausibly wrong rather than obviously wrong, so the expected
//! values below -- 4 tensors, 14 dimensions, 4 element types, the anchor identity, the
//! input count -- are independently known and are what hold it honest.
//!
//! **This file is the ONLY D4 test file with a fixture dependency**, and it loads through
//! the panicking `pc_testkit::paths::recorded` helper -- never `recorded_opt`. The
//! signature JSON is a committed ~300-byte artifact, so its absence is a broken checkout,
//! not an un-run recording step; a skip here would silently restore the exact gap this
//! file closes. `d4_onnx.rs` stays synthetic and fixture-free.
//!
//! No model weights and no ONNX Runtime are required by anything in this file.

use pc_detect::onnx::{bind_outputs, OutputBinding, OutputMeta, NET_SIZE};
use pc_detect::yolo::{N_CLASSES, ROW_STRIDE};
use pc_testkit::model_signature::{comic_text_detector_signature, ModelSignature, TensorSignature};

/// `[1, 3, NET_SIZE, NET_SIZE]` -- the shape `letterbox` + `to_nchw` must produce.
fn expected_input_shape() -> Vec<i64> {
    vec![1, 3, i64::from(NET_SIZE), i64::from(NET_SIZE)]
}

/// The number of raw yolov5 candidate rows: 3 anchors at each of the three head strides
/// (8, 16, 32) over a `NET_SIZE` square input. `3 * (128^2 + 64^2 + 32^2) = 64512`.
fn expected_anchor_rows() -> i64 {
    let net = i64::from(NET_SIZE);
    3 * ((net / 8).pow(2) + (net / 16).pow(2) + (net / 32).pow(2))
}

/// The recorded outputs as `bind_outputs` consumes them, preserving recorded order.
/// `pc-testkit` deliberately knows nothing about `pc_detect::onnx::OutputMeta` (it is a
/// dev-dependency of this crate, not the reverse), so the conversion lives here.
fn recorded_output_metas(signature: &ModelSignature) -> Vec<OutputMeta> {
    signature
        .outputs
        .iter()
        .map(|tensor: &TensorSignature| OutputMeta {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
        })
        .collect()
}

// ------------------------------------------------------------ the recorded artifact

#[test]
fn the_signature_describes_the_model_the_spec_names() {
    // spec §8.3 step 3 names exactly one detector model. The digest *identity* -- that this
    // signature was taken from `pc_models::COMIC_TEXT_DETECTOR` and not from some other
    // export -- is asserted in `pc-models`' own suite, which owns that constant (`pc-detect`
    // must not depend on `pc-models`, spec §1 as amended). Here we assert only the file
    // identity and that the digest is well formed, so the digest literal lives in exactly
    // one place.
    let signature = comic_text_detector_signature();

    assert_eq!(signature.model_file_name, "comictextdetector.pt.onnx");
    assert_eq!(signature.size_bytes, 94_669_756);
    assert_eq!(signature.opset, 11);
    assert_eq!(signature.sha256.len(), 64);
    assert!(
        signature
            .sha256
            .chars()
            .all(|character| matches!(character, '0'..='9' | 'a'..='f')),
        "the recorded digest must be lowercase hex: {}",
        signature.sha256
    );
}

// ------------------------------------------------------------ the input tensor

#[test]
fn the_model_declares_exactly_one_input() {
    // PROBE OBLIGATION, stated here so it is visible from the test side: an ONNX graph may
    // list weight initializers in `graph.input`, and this model has 280 initializers. A
    // walk that does not subtract `graph.initializer` from `graph.input` would record 281
    // inputs. Every other input assertion in this file assumes a single entry, so this is
    // the assertion that keeps that assumption honest.
    let signature = comic_text_detector_signature();

    assert_eq!(
        signature.inputs.len(),
        1,
        "recorded inputs: {:?}",
        signature
            .inputs
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn net_size_matches_the_models_declared_input_shape() {
    // Closes the SECOND self-reference in D4a: `d4_onnx.rs::constants_match_the_spec`
    // asserts `NET_SIZE == 1024` as a transcription of §8.3 step 3, and the entire
    // letterbox suite is built on that number. This ties it to the artifact.
    let signature = comic_text_detector_signature();
    let input = &signature.inputs[0];

    assert_eq!(input.shape, expected_input_shape());
    assert_eq!(
        input.shape[1], 3,
        "spec §8.3 step 3: RGB, i.e. 3 channels, NCHW"
    );
    assert_eq!(input.shape[0], 1, "one image per inference (§4.5)");
}

#[test]
fn the_model_input_is_named_images() {
    // DECISION, and why the name is pinned even though §8.3 step 3 only mandates binding
    // *outputs* by index: the graph has exactly one input, so an index-bound
    // implementation is unambiguous and this assertion constrains nothing about D4b's
    // choice. It is pinned anyway because IF D4b binds by name, a re-exported model with a
    // renamed input would fail at inference time with an `ort` error that says nothing
    // about why -- and this test names the cause in one line, offline, with no weights.
    // Cheap certification of an artifact property; not a requirement on the binding style.
    let signature = comic_text_detector_signature();

    assert_eq!(signature.inputs[0].name, "images");
}

#[test]
fn the_model_input_is_f32() {
    // spec §8.3 step 3: `f32 / 255.0`. A wrong element type is a silent decode disaster --
    // f16 weights fed f32 bytes produce plausible-looking garbage, not an error -- and it
    // is a declared graph property, so it is asserted rather than merely recorded.
    let signature = comic_text_detector_signature();

    assert!(
        signature.inputs[0].is_f32(),
        "input element type was recorded as `{}`",
        signature.inputs[0].element_type
    );
}

// ------------------------------------------------------------ the output tensors

#[test]
fn the_model_declares_three_f32_outputs_named_blk_seg_det_in_that_order() {
    // spec §8.3 step 3: "Bind by index". That instruction is only correct if the graph's
    // output ORDER is what we think it is, so the order is itself a certified fact here --
    // note that a name-sorted probe would emit blk/det/seg, i.e. `seg` and `det` swapped.
    let signature = comic_text_detector_signature();

    let names: Vec<&str> = signature
        .outputs
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    assert_eq!(names, ["blk", "seg", "det"]);
    for tensor in &signature.outputs {
        assert!(
            tensor.is_f32(),
            "output `{}` element type was recorded as `{}`",
            tensor.name,
            tensor.element_type
        );
    }
}

#[test]
fn row_stride_matches_the_models_declared_blk_arity() {
    // ==================== THE CLOSURE OF THE TAUTOLOGY ====================
    // spec §16.15 corrected §8.3 step 3's `n_classes` from 3 to 2 on the evidence of
    // upstream's `nc = shape[2] - 5` and this model's `[1, 64512, 7]`. Until this
    // assertion existed, that correction was pinned only by literals we wrote ourselves.
    //
    // If this fails, our constant disagrees with the shipped artifact. That is a
    // spec/frozen-test conflict and escalates to BOTH architects per CLAUDE.md: do NOT
    // edit `N_CLASSES`, `ROW_STRIDE`, `d5_yolo.rs`'s literals, `d4_onnx.rs`'s
    // `shipped_outputs()` literal, or the recorded fixture to make it pass.
    //
    // Why a mismatch here cannot pass unnoticed as a coincidence: a mis-parsed protobuf
    // field yields garbage or a length, not a plausible single-digit integer that happens
    // to equal an independently known expectation.
    let signature = comic_text_detector_signature();
    let blk = signature.output("blk");

    assert_eq!(blk.shape.len(), 3, "blk is `[1, N, 5 + n_classes]`");
    assert_eq!(blk.shape[0], 1);
    let columns = *blk.shape.last().expect("blk has a last dimension");
    assert_eq!(
        columns, ROW_STRIDE as i64,
        "the model's blk column count is {columns}, but the frozen yolo::ROW_STRIDE is \
         {ROW_STRIDE}"
    );
    // The same statement in §16.15's own terms, so the constant the correction actually
    // changed is the one measured against the artifact.
    assert_eq!(
        N_CLASSES as i64,
        columns - 5,
        "spec §8.3 step 3's `5 + n_classes`: the model implies n_classes = {}, but \
         yolo::N_CLASSES is {N_CLASSES}",
        columns - 5
    );
}

#[test]
fn the_recorded_blk_row_count_is_the_three_scale_anchor_grid() {
    // A property of the RECORDED ARTIFACT, certified in the artifact's own test file --
    // NOT a runtime requirement. Nothing in v1 depends on the candidate-row count:
    // `bind_outputs` reads dimension 1 only for the mask/lines channel guard, and
    // `decode_blocks` derives the row count from the payload length, so a re-export with a
    // different grid would work unchanged. It is asserted because it is a strong, cheap
    // signal that the export is the standard yolov5 three-head graph we believe it to be:
    // 3 anchors x (128^2 + 64^2 + 32^2) = 3 x 21504 = 64512 for a 1024^2 input.
    let signature = comic_text_detector_signature();

    assert_eq!(expected_anchor_rows(), 64_512);
    assert_eq!(signature.output("blk").shape[1], expected_anchor_rows());
}

#[test]
fn the_recorded_mask_and_lines_shapes_are_what_the_swap_guard_expects() {
    // spec §8.3 step 3's swap guard fires when output 1 has 2 channels and output 2 has 1.
    // `d4_onnx.rs` proves the guard's logic on synthetic input; the claim that it does NOT
    // fire on the real weights depends on the real channel counts being 1 then 2, which
    // was a literal we wrote until now. §8.3 step 5's dw/dh crop additionally assumes the
    // mask arrives at the letterbox resolution, which `NET_SIZE` here certifies.
    let signature = comic_text_detector_signature();
    let net = i64::from(NET_SIZE);

    assert_eq!(signature.output("seg").shape, vec![1, 1, net, net]);
    assert_eq!(signature.output("det").shape, vec![1, 2, net, net]);
}

// ------------------------------------------------------------ the binding, non-vacuously

#[test]
fn bind_outputs_binds_the_real_model_by_index() {
    // The former tautology, now anchored: the input comes from the recorded artifact and
    // the expectation from `OutputBinding::BY_INDEX`, so the two sides no longer descend
    // from the same constant.
    //
    // STRICT equality, not `is_ok()`: `bind_outputs` returns `Ok` for the swapped
    // arrangement too, so `is_ok()` would silently accept a probe that emitted
    // blk/det/seg (which is what name-sorted output would produce) -- the exact class of
    // plausible-looking wrongness this file exists to catch.
    let signature = comic_text_detector_signature();

    let binding =
        bind_outputs(&recorded_output_metas(&signature)).expect("the real model binds cleanly");

    assert_eq!(binding, OutputBinding::BY_INDEX);
    assert_eq!(
        binding,
        OutputBinding {
            blks: 0,
            mask: 1,
            lines_map: 2
        }
    );
}
