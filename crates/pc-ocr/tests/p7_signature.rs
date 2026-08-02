//! Task P7c Phase 3 — assertions over the committed manga-ocr graph signatures.
//!
//! This is the dependency-free/default-tier half of the P7 session contract. It must remain
//! runnable without `ort` and without either ONNX weight file.

use pc_ocr::vocab::VOCAB_LEN;
use pc_testkit::{
    model_signature::{Dim, ELEMENT_TYPE_F32, ELEMENT_TYPE_I64},
    ocr_model_signature::{
        manga_ocr_decoder_signature, manga_ocr_encoder_signature, MANGA_OCR_DECODER,
        MANGA_OCR_ENCODER,
    },
};

#[test]
fn encoder_signature_matches_its_pin_and_graph_metadata() {
    let signature = manga_ocr_encoder_signature();

    assert_eq!(signature.schema_version, 1);
    assert_eq!(signature.model_file_name, MANGA_OCR_ENCODER.file_name);
    assert_eq!(signature.sha256, MANGA_OCR_ENCODER.sha256);
    assert_eq!(signature.size_bytes, MANGA_OCR_ENCODER.size_bytes);
    assert_eq!(
        signature.sha256,
        "15fa8155fe9bc1a7d25d9bb353debaa4def033d0174e907dbd2dd6d995def85f"
    );
    assert_eq!(signature.size_bytes, 343_454_249);
    assert_eq!(signature.opset, 14);

    assert_eq!(signature.inputs.len(), 1);
    let input = &signature.inputs[0];
    assert_eq!(input.name, "pixel_values");
    assert_eq!(input.element_type, ELEMENT_TYPE_F32);
    assert_eq!(
        input.shape,
        vec![
            Dim::Symbolic("batch_size".into()),
            Dim::Symbolic("num_channels".into()),
            Dim::Symbolic("height".into()),
            Dim::Symbolic("width".into()),
        ]
    );

    assert_eq!(signature.outputs.len(), 1);
    let output = &signature.outputs[0];
    assert_eq!(output.name, "last_hidden_state");
    assert_eq!(output.element_type, ELEMENT_TYPE_F32);
    assert_eq!(
        output.shape,
        vec![
            Dim::Symbolic("batch_size".into()),
            Dim::Symbolic("Addlast_hidden_state_dim_1".into()),
            Dim::Symbolic("Addlast_hidden_state_dim_2".into()),
        ]
    );
}

#[test]
fn encoder_signature_has_zero_concrete_dimensions_deliberately() {
    let signature = manga_ocr_encoder_signature();

    // Deliberate measured fact: do not “fix” this by adding the earlier, wrong planning
    // assumption of concrete 197/768 dimensions to the fixture or this assertion.
    assert!(signature
        .inputs
        .iter()
        .chain(signature.outputs.iter())
        .flat_map(|tensor| tensor.shape.iter())
        .all(|dimension| dimension.fixed().is_none()));
}

#[test]
fn decoder_signature_matches_its_pin_and_graph_metadata() {
    let signature = manga_ocr_decoder_signature();

    assert_eq!(signature.schema_version, 1);
    assert_eq!(signature.model_file_name, MANGA_OCR_DECODER.file_name);
    assert_eq!(signature.sha256, MANGA_OCR_DECODER.sha256);
    assert_eq!(signature.size_bytes, MANGA_OCR_DECODER.size_bytes);
    assert_eq!(
        signature.sha256,
        "ef7765261e9d1cdc34d89356986c2bbc2a082897f753a89605ae80fdfa61f5e8"
    );
    assert_eq!(signature.size_bytes, 117_480_262);
    assert_eq!(signature.opset, 14);

    assert_eq!(signature.inputs.len(), 2);
    assert_eq!(signature.inputs[0].name, "input_ids");
    assert_eq!(signature.inputs[0].element_type, ELEMENT_TYPE_I64);
    assert_eq!(
        signature.inputs[0].shape,
        vec![
            Dim::Symbolic("batch_size".into()),
            Dim::Symbolic("decoder_sequence_length".into()),
        ]
    );

    assert_eq!(signature.inputs[1].name, "encoder_hidden_states");
    assert_eq!(signature.inputs[1].element_type, ELEMENT_TYPE_F32);
    assert_eq!(
        signature.inputs[1].shape,
        vec![
            Dim::Symbolic("batch_size".into()),
            Dim::Symbolic("encoder_sequence_length".into()),
            Dim::Fixed(768),
        ]
    );

    assert_eq!(signature.outputs.len(), 1);
    assert_eq!(signature.outputs[0].name, "logits");
    assert_eq!(signature.outputs[0].element_type, ELEMENT_TYPE_F32);
    assert_eq!(
        signature.outputs[0].shape,
        vec![
            Dim::Symbolic("batch_size".into()),
            Dim::Symbolic("decoder_sequence_length".into()),
            Dim::Fixed(6144),
        ]
    );
}

#[test]
fn both_graphs_report_opset_14() {
    assert_eq!(manga_ocr_encoder_signature().opset, 14);
    assert_eq!(manga_ocr_decoder_signature().opset, 14);
}

#[test]
fn decoder_logits_width_matches_the_frozen_vocabulary() {
    let signature = manga_ocr_decoder_signature();
    let logits_width = signature
        .output("logits")
        .shape
        .last()
        .and_then(Dim::fixed)
        .expect("recorded logits width is concrete");

    // A failure is a spec/artifact conflict. Do not “fix” it by editing VOCAB_LEN, the
    // vendored vocabulary, or the recorded signature; escalate to both architects.
    assert_eq!(logits_width, VOCAB_LEN as i64);
}

#[test]
fn dim_fixed_and_symbol_discriminate_the_two_variants() {
    let fixed = Dim::Fixed(768);
    let symbolic = Dim::Symbolic("encoder_sequence_length".into());

    assert_eq!(fixed.fixed(), Some(768));
    assert_eq!(fixed.symbol(), None);
    assert_eq!(symbolic.fixed(), None);
    assert_eq!(symbolic.symbol(), Some("encoder_sequence_length"));
}
