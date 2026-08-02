use pc_testkit::model_signature::{
    Dim, SymbolicModelSignature, SymbolicTensorSignature, TensorSignature,
};

#[test]
fn symbolic_signature_round_trips_mixed_shape_dimensions() {
    let signature = SymbolicModelSignature {
        schema_version: 1,
        model_file_name: "encoder_model.onnx".into(),
        sha256: "a".repeat(64),
        size_bytes: 42,
        opset: 14,
        inputs: vec![SymbolicTensorSignature {
            name: "pixel_values".into(),
            element_type: "f32".into(),
            shape: vec![
                Dim::Symbolic("batch_size".into()),
                Dim::Fixed(3),
                Dim::Symbolic("height".into()),
            ],
        }],
        outputs: Vec::new(),
    };

    let json = serde_json::to_string(&signature).expect("serialize symbolic signature");
    let round_tripped: SymbolicModelSignature =
        serde_json::from_str(&json).expect("deserialize symbolic signature");

    assert_eq!(round_tripped, signature);
}

#[test]
fn integer_signature_rejects_symbolic_shape_dimensions() {
    let result = serde_json::from_str::<TensorSignature>(
        r#"{"name":"x","element_type":"f32","shape":[1,"batch_size",3]}"#,
    );

    assert!(result.is_err());
}

#[test]
fn dim_rejects_json_float() {
    let result = serde_json::from_str::<Dim>("768.5");

    assert!(result.is_err());
}
