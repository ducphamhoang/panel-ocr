//! Dependency-free recorder for the two pinned manga-ocr ONNX graph signatures.

use anyhow::{bail, Context, Result};
use pc_testkit::model_signature::{
    Dim, SymbolicModelSignature, SymbolicTensorSignature, ELEMENT_TYPE_F32,
};
use pc_testkit::ocr_model_signature::{
    OcrModelPin, MANGA_OCR_DECODER, MANGA_OCR_DECODER_SIGNATURE, MANGA_OCR_ENCODER,
    MANGA_OCR_ENCODER_SIGNATURE,
};
use pc_testkit::provenance::{ArtifactRecord, GroupProvenance};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

const GROUP: &str = "ocr_model_signature";
const PROVENANCE_REL: &str = "ocr_model_signature/PROVENANCE.json";
const TOOL: &str = "cargo xtask record-fixtures --only ocr-model-signature";

pub fn capability_status(
    encoder_explicit: Option<&Path>,
    decoder_explicit: Option<&Path>,
) -> String {
    let encoder_output = crate::paths::recorded_root().join(MANGA_OCR_ENCODER_SIGNATURE);
    let decoder_output = crate::paths::recorded_root().join(MANGA_OCR_DECODER_SIGNATURE);
    let provenance = crate::paths::recorded_root().join(PROVENANCE_REL);
    if encoder_output.is_file() && decoder_output.is_file() && provenance.is_file() {
        return format!(
            "RECORDED — {} and {} (sources can be re-verified with --ocr-encoder PATH and --ocr-decoder PATH)",
            crate::paths::display_relative(&encoder_output),
            crate::paths::display_relative(&decoder_output),
        );
    }

    let encoder = source_path(encoder_explicit, "PANEL_OCR_MANGA_OCR_ENCODER");
    let decoder = source_path(decoder_explicit, "PANEL_OCR_MANGA_OCR_DECODER");
    match (encoder, decoder) {
        (Some(encoder), Some(decoder)) if encoder.is_file() && decoder.is_file() => {
            format!(
                "AVAILABLE — encoder: {}, decoder: {}",
                encoder.display(),
                decoder.display()
            )
        }
        (encoder, decoder) => format!(
            "SKIPPED — {}",
            missing_sources_reason(encoder.as_deref(), decoder.as_deref())
        ),
    }
}

pub fn source_path(explicit: Option<&Path>, env_name: &str) -> Option<PathBuf> {
    explicit
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os(env_name).map(PathBuf::from))
}

pub fn record(
    encoder_explicit: Option<&Path>,
    decoder_explicit: Option<&Path>,
    force: bool,
) -> Result<super::record::Outcome> {
    let encoder_output = crate::paths::recorded_root().join(MANGA_OCR_ENCODER_SIGNATURE);
    let decoder_output = crate::paths::recorded_root().join(MANGA_OCR_DECODER_SIGNATURE);
    let provenance_path = crate::paths::recorded_root().join(PROVENANCE_REL);

    if encoder_output.is_file() && decoder_output.is_file() && provenance_path.is_file() && !force {
        validate_existing_group(&encoder_output, &decoder_output, &provenance_path)?;
        return Ok(super::record::Outcome::Recorded {
            detail: format!(
                "{} and {} already present (use --force to re-record)",
                crate::paths::display_relative(&encoder_output),
                crate::paths::display_relative(&decoder_output),
            ),
        });
    }

    let encoder = source_path(encoder_explicit, "PANEL_OCR_MANGA_OCR_ENCODER");
    let decoder = source_path(decoder_explicit, "PANEL_OCR_MANGA_OCR_DECODER");
    if encoder.as_ref().is_none_or(|path| !path.is_file())
        || decoder.as_ref().is_none_or(|path| !path.is_file())
    {
        return Ok(super::record::Outcome::Skipped {
            reason: format!(
                "{}; no fixture was emitted.",
                missing_sources_reason(encoder.as_deref(), decoder.as_deref())
            ),
        });
    }
    let encoder = encoder.expect("encoder source checked above");
    let decoder = decoder.expect("decoder source checked above");

    // Read and verify both real artifacts (once each, in memory) before parsing either
    // protobuf. This is deliberately a pair: a one-file recording is not a valid OCR
    // group. `read_and_verify_source` returns the exact bytes it hashed, and
    // `parse_signature` parses those same bytes rather than re-reading the path.
    let (encoder_bytes, encoder_sha256) = read_and_verify_source(&encoder, &MANGA_OCR_ENCODER)?;
    let (decoder_bytes, decoder_sha256) = read_and_verify_source(&decoder, &MANGA_OCR_DECODER)?;
    let encoder_signature = parse_signature(
        &encoder,
        &encoder_bytes,
        &encoder_sha256,
        &MANGA_OCR_ENCODER,
    )?;
    let decoder_signature = parse_signature(
        &decoder,
        &decoder_bytes,
        &decoder_sha256,
        &MANGA_OCR_DECODER,
    )?;
    println!("parsed OCR encoder signature:\n{encoder_signature:#?}");
    println!("parsed OCR decoder signature:\n{decoder_signature:#?}");

    let mut encoder_json = serde_json::to_vec_pretty(&encoder_signature)?;
    encoder_json.push(b'\n');
    let mut decoder_json = serde_json::to_vec_pretty(&decoder_signature)?;
    decoder_json.push(b'\n');
    let encoder_output_sha256 = sha256_hex_bytes(&encoder_json);
    let decoder_output_sha256 = sha256_hex_bytes(&decoder_json);
    let provenance = GroupProvenance {
        schema_version: pc_testkit::provenance::PROVENANCE_SCHEMA_VERSION,
        group: GROUP.into(),
        tool: TOOL.into(),
        command_line: TOOL.into(),
        tool_versions: BTreeMap::new(),
        records: vec![
            artifact_record(
                "encoder",
                "tests/fixtures/recorded/ocr_model_signature/encoder_model.signature.json",
                &encoder_output_sha256,
                &encoder_signature,
                &MANGA_OCR_ENCODER,
            ),
            artifact_record(
                "decoder",
                "tests/fixtures/recorded/ocr_model_signature/decoder_model.signature.json",
                &decoder_output_sha256,
                &decoder_signature,
                &MANGA_OCR_DECODER,
            ),
        ],
        detector: None,
        diagnostics: BTreeMap::new(),
    };
    let mut provenance_json = serde_json::to_vec_pretty(&provenance)?;
    provenance_json.push(b'\n');

    let output_dir = encoder_output
        .parent()
        .expect("OCR signature output has a parent")
        .to_path_buf();
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating {}", output_dir.display()))?;

    let encoder_temp = pc_models::partial_path(&encoder_output);
    let decoder_temp = pc_models::partial_path(&decoder_output);
    let provenance_temp = pc_models::partial_path(&provenance_path);
    let _staged_files = StagedFiles::new([
        encoder_temp.clone(),
        decoder_temp.clone(),
        provenance_temp.clone(),
    ]);
    write_synced(&encoder_temp, &encoder_json)?;
    write_synced(&decoder_temp, &decoder_json)?;
    write_synced(&provenance_temp, &provenance_json)?;

    // Keep publication adjacent: all fallible serialization, staging, and path work is complete
    // before the three destination renames begin.
    std::fs::rename(&encoder_temp, &encoder_output)
        .with_context(|| format!("publishing {}", encoder_output.display()))?;
    std::fs::rename(&decoder_temp, &decoder_output)
        .with_context(|| format!("publishing {}", decoder_output.display()))?;
    std::fs::rename(&provenance_temp, &provenance_path)
        .with_context(|| format!("publishing {}", provenance_path.display()))?;

    validate_existing_group(&encoder_output, &decoder_output, &provenance_path)?;
    Ok(super::record::Outcome::Recorded {
        detail: format!(
            "{} and {} ({} / {} bytes, source sha256 {} / {})",
            crate::paths::display_relative(&encoder_output),
            crate::paths::display_relative(&decoder_output),
            encoder_signature.size_bytes,
            decoder_signature.size_bytes,
            encoder_signature.sha256,
            decoder_signature.sha256,
        ),
    })
}

fn missing_sources_reason(encoder: Option<&Path>, decoder: Option<&Path>) -> String {
    let mut missing = Vec::new();
    for (label, path, pin, env_name) in [
        (
            "encoder",
            encoder,
            &MANGA_OCR_ENCODER,
            "PANEL_OCR_MANGA_OCR_ENCODER",
        ),
        (
            "decoder",
            decoder,
            &MANGA_OCR_DECODER,
            "PANEL_OCR_MANGA_OCR_DECODER",
        ),
    ] {
        match path {
            Some(path) if path.is_file() => {}
            Some(path) => missing.push(format!(
                "{label} file is absent at {}; provide --ocr-{} PATH or set {env_name} to the sha256-verified {}",
                path.display(), label, pin.file_name
            )),
            None => missing.push(format!(
                "{label} file is not configured; provide --ocr-{} PATH or set {env_name} to the sha256-verified {}",
                label, pin.file_name
            )),
        }
    }
    missing.join("; ")
}

/// Reads the source file ONCE and verifies the exact bytes returned, rather than
/// verifying a path's contents (`verify_sha256`/`std::fs::metadata`) and then having a
/// separate later read (`parse_signature`'s old `std::fs::read` + a second
/// `sha256_hex(path)`) re-open the same path — a real TOCTOU gap on a path that could
/// change between the two reads (a symlink swap, a concurrent write, a network mount).
/// Everything downstream parses these returned bytes and records the returned digest;
/// nothing re-reads `path` or re-derives the digest from the pin.
fn read_and_verify_source(path: &Path, pin: &OcrModelPin) -> Result<(Vec<u8>, String)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading the OCR model source {}", path.display()))?;
    if bytes.len() as u64 != pin.size_bytes {
        bail!(
            "`{}` has size {} bytes, expected pinned size {}",
            path.display(),
            bytes.len(),
            pin.size_bytes
        );
    }
    let sha256 = sha256_hex_bytes(&bytes);
    if !sha256.eq_ignore_ascii_case(pin.sha256) {
        bail!(
            "`{}` hashes to {sha256}, expected pinned sha256 {}",
            path.display(),
            pin.sha256
        );
    }
    Ok((bytes, sha256))
}

fn artifact_record(
    name: &str,
    output: &str,
    output_sha256: &str,
    signature: &SymbolicModelSignature,
    pin: &OcrModelPin,
) -> ArtifactRecord {
    ArtifactRecord {
        name: name.into(),
        output: output.into(),
        output_sha256: output_sha256.into(),
        committed: true,
        source: Some(pin.file_name.into()),
        source_sha256: Some(signature.sha256.clone()),
        params: BTreeMap::from([
            ("opset".into(), serde_json::json!(signature.opset)),
            ("size_bytes".into(), serde_json::json!(signature.size_bytes)),
        ]),
    }
}

fn validate_existing_group(
    encoder_output: &Path,
    decoder_output: &Path,
    provenance_path: &Path,
) -> Result<()> {
    let encoder_json = std::fs::read(encoder_output)
        .with_context(|| format!("reading existing {}", encoder_output.display()))?;
    let decoder_json = std::fs::read(decoder_output)
        .with_context(|| format!("reading existing {}", decoder_output.display()))?;
    let provenance_json = std::fs::read(provenance_path)
        .with_context(|| format!("reading existing {}", provenance_path.display()))?;
    let encoder_signature: SymbolicModelSignature = serde_json::from_slice(&encoder_json)
        .with_context(|| {
            format!(
                "existing {} is not a symbolic model signature",
                encoder_output.display()
            )
        })?;
    let decoder_signature: SymbolicModelSignature = serde_json::from_slice(&decoder_json)
        .with_context(|| {
            format!(
                "existing {} is not a symbolic model signature",
                decoder_output.display()
            )
        })?;
    // The two checks below close a real gap: parsing the committed JSON into the schema
    // type proves it deserializes, not that its *content* still matches the pinned
    // artifact. Without this, a coordinated edit to the signature's own `sha256`/
    // `size_bytes` fields (left internally consistent with a matching edit to
    // PROVENANCE.json's `source_sha256`) would pass every check above and below.
    for (name, signature, pin) in [
        ("encoder", &encoder_signature, &MANGA_OCR_ENCODER),
        ("decoder", &decoder_signature, &MANGA_OCR_DECODER),
    ] {
        if !signature.sha256.eq_ignore_ascii_case(pin.sha256) {
            bail!(
                "existing OCR model-signature `{name}` records source sha256 {}, pinned {}",
                signature.sha256,
                pin.sha256
            );
        }
        if signature.size_bytes != pin.size_bytes {
            bail!(
                "existing OCR model-signature `{name}` records source size {} bytes, pinned {} bytes",
                signature.size_bytes,
                pin.size_bytes
            );
        }
    }
    let provenance: GroupProvenance = serde_json::from_slice(&provenance_json)
        .with_context(|| format!("existing {} is not provenance", provenance_path.display()))?;
    let violations = pc_testkit::provenance::validate(GROUP, &provenance);
    if !violations.is_empty() {
        bail!("existing OCR model-signature provenance is invalid: {violations:?}");
    }
    for (name, output, bytes, pin) in [
        ("encoder", encoder_output, &encoder_json, &MANGA_OCR_ENCODER),
        ("decoder", decoder_output, &decoder_json, &MANGA_OCR_DECODER),
    ] {
        let record = provenance
            .records
            .iter()
            .find(|record| record.name == name)
            .with_context(|| format!("OCR model-signature provenance has no {name} record"))?;
        let actual_output_sha256 = sha256_hex_bytes(bytes);
        if record.output_sha256 != actual_output_sha256 {
            bail!(
                "published OCR model-signature `{name}` output digest {} != {}",
                record.output_sha256,
                actual_output_sha256
            );
        }
        if record.source.as_deref() != Some(pin.file_name)
            || record.source_sha256.as_deref() != Some(pin.sha256)
        {
            bail!(
                "published OCR model-signature `{name}` source does not match pin {}",
                pin.file_name
            );
        }
        let expected_output = format!(
            "tests/fixtures/recorded/{}",
            match name {
                "encoder" => MANGA_OCR_ENCODER_SIGNATURE,
                "decoder" => MANGA_OCR_DECODER_SIGNATURE,
                _ => unreachable!(),
            }
        );
        if record.output != expected_output {
            bail!(
                "published OCR model-signature `{name}` output is `{}`, expected `{expected_output}`",
                record.output
            );
        }
        let _ = output;
    }
    if provenance.records.len() != 2 {
        bail!(
            "published OCR model-signature provenance has {} records, expected exactly two",
            provenance.records.len()
        );
    }
    Ok(())
}

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Parses already-verified bytes (from `read_and_verify_source`) — never re-reads `path`,
/// so the signature this emits describes exactly the bytes whose hash was checked, not
/// whatever a second, later `std::fs::read` of the same path happens to return.
fn parse_signature(
    path: &Path,
    bytes: &[u8],
    sha256: &str,
    pin: &OcrModelPin,
) -> Result<SymbolicModelSignature> {
    let model_file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "OCR model path has no valid UTF-8 file name: {}",
                path.display()
            )
        })?;
    if model_file_name != pin.file_name {
        bail!(
            "the verified OCR model is named `{model_file_name}`, expected `{}`",
            pin.file_name
        );
    }
    parse_symbolic_bytes(bytes, model_file_name, sha256.to_owned())
}

fn parse_symbolic_bytes(
    bytes: &[u8],
    model_file_name: &str,
    sha256: String,
) -> Result<SymbolicModelSignature> {
    let mut graph: Option<&[u8]> = None;
    let mut default_opsets = Vec::new();
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        match field.number {
            7 => {
                expect_wire(field.wire, 2, "ModelProto.graph")?;
                if graph
                    .replace(field.payload.expect("length-delimited field"))
                    .is_some()
                {
                    bail!("ModelProto.graph is repeated")
                }
            }
            8 => {
                expect_wire(field.wire, 2, "ModelProto.opset_import")?;
                let (domain, version) =
                    parse_opset(field.payload.expect("length-delimited field"))?;
                if domain.is_empty() {
                    default_opsets.push(version);
                }
            }
            _ => {}
        }
    }
    let graph = graph.ok_or_else(|| anyhow::anyhow!("ModelProto.graph is missing"))?;
    if default_opsets.len() != 1 {
        bail!(
            "expected exactly one default-domain opset import, found {}",
            default_opsets.len()
        );
    }
    let (inputs, outputs) = parse_graph(graph)?;
    Ok(SymbolicModelSignature {
        schema_version: 1,
        model_file_name: model_file_name.to_owned(),
        sha256,
        size_bytes: bytes.len() as u64,
        opset: default_opsets[0],
        inputs,
        outputs,
    })
}

fn parse_graph(
    bytes: &[u8],
) -> Result<(Vec<SymbolicTensorSignature>, Vec<SymbolicTensorSignature>)> {
    let mut initializer_names = HashSet::new();
    let mut raw_inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        match field.number {
            5 => {
                expect_wire(field.wire, 2, "GraphProto.initializer")?;
                initializer_names.insert(parse_initializer_name(
                    field.payload.expect("length-delimited field"),
                )?);
            }
            11 => {
                expect_wire(field.wire, 2, "GraphProto.input")?;
                raw_inputs.push(field.payload.expect("length-delimited field"));
            }
            12 => {
                expect_wire(field.wire, 2, "GraphProto.output")?;
                outputs.push(parse_value_info(
                    field.payload.expect("length-delimited field"),
                    "output",
                )?);
            }
            _ => {}
        }
    }
    let mut inputs = Vec::new();
    for raw in raw_inputs {
        let name = parse_value_info_name(raw, "input")?;
        if !initializer_names.contains(&name) {
            inputs.push(parse_value_info(raw, "input")?);
        }
    }
    if inputs.is_empty() {
        bail!("expected at least one true graph input after subtracting initializers");
    }
    Ok((inputs, outputs))
}

fn parse_opset(bytes: &[u8]) -> Result<(String, i64)> {
    let mut domain = String::new();
    let mut version = None;
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        match field.number {
            1 => {
                expect_wire(field.wire, 2, "OperatorSetIdProto.domain")?;
                domain = parse_string(
                    field.payload.expect("length-delimited field"),
                    "opset domain",
                )?;
            }
            2 => {
                expect_wire(field.wire, 0, "OperatorSetIdProto.version")?;
                version = Some(field.varint.expect("varint field") as i64);
            }
            _ => {}
        }
    }
    Ok((
        domain,
        version.ok_or_else(|| anyhow::anyhow!("OperatorSetIdProto.version is missing"))?,
    ))
}

fn parse_initializer_name(bytes: &[u8]) -> Result<String> {
    let mut name = None;
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        if field.number == 8 {
            expect_wire(field.wire, 2, "TensorProto.name")?;
            if name
                .replace(parse_string(
                    field.payload.expect("length-delimited field"),
                    "initializer name",
                )?)
                .is_some()
            {
                bail!("TensorProto.name is repeated")
            }
        }
    }
    name.ok_or_else(|| anyhow::anyhow!("GraphProto.initializer TensorProto.name is missing"))
}

fn parse_value_info_name(bytes: &[u8], context: &str) -> Result<String> {
    let mut name = None;
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        if field.number == 1 {
            expect_wire(field.wire, 2, "ValueInfoProto.name")?;
            if name
                .replace(parse_string(
                    field.payload.expect("length-delimited field"),
                    context,
                )?)
                .is_some()
            {
                bail!("{context} ValueInfoProto.name is repeated")
            }
        }
    }
    name.ok_or_else(|| anyhow::anyhow!("{context} ValueInfoProto.name is missing"))
}

fn parse_value_info(bytes: &[u8], context: &str) -> Result<SymbolicTensorSignature> {
    let mut name = None;
    let mut tensor_type = None;
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        match field.number {
            1 => {
                expect_wire(field.wire, 2, "ValueInfoProto.name")?;
                if name
                    .replace(parse_string(
                        field.payload.expect("length-delimited field"),
                        context,
                    )?)
                    .is_some()
                {
                    bail!("{context} ValueInfoProto.name is repeated")
                }
            }
            2 => {
                expect_wire(field.wire, 2, "ValueInfoProto.type")?;
                if tensor_type
                    .replace(parse_type_proto(
                        field.payload.expect("length-delimited field"),
                        context,
                    )?)
                    .is_some()
                {
                    bail!("{context} ValueInfoProto.type is repeated")
                }
            }
            _ => {}
        }
    }
    let name = name.ok_or_else(|| anyhow::anyhow!("{context} ValueInfoProto.name is missing"))?;
    let (element_type, shape) =
        tensor_type.ok_or_else(|| anyhow::anyhow!("{context} ValueInfoProto.type is missing"))?;
    Ok(SymbolicTensorSignature {
        name,
        element_type,
        shape,
    })
}

fn parse_type_proto(bytes: &[u8], context: &str) -> Result<(String, Vec<Dim>)> {
    let mut tensor_type = None;
    let mut non_tensor = None;
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        match field.number {
            1 => {
                expect_wire(field.wire, 2, "TypeProto.tensor_type")?;
                if tensor_type
                    .replace(parse_tensor_type(
                        field.payload.expect("length-delimited field"),
                        context,
                    )?)
                    .is_some()
                {
                    bail!("{context} TypeProto.tensor_type is repeated")
                }
            }
            2..=5 => non_tensor = Some(field.number),
            _ => {}
        }
    }
    if let Some(field) = non_tensor {
        bail!("{context} ValueInfoProto has non-tensor TypeProto field {field}")
    }
    tensor_type.ok_or_else(|| anyhow::anyhow!("{context} TypeProto.tensor_type is missing"))
}

fn parse_tensor_type(bytes: &[u8], context: &str) -> Result<(String, Vec<Dim>)> {
    let mut element_type = None;
    let mut shape = None;
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        match field.number {
            1 => {
                expect_wire(field.wire, 0, "TensorTypeProto.elem_type")?;
                if element_type
                    .replace(field.varint.expect("varint field") as i32)
                    .is_some()
                {
                    bail!("{context} TensorTypeProto.elem_type is repeated")
                }
            }
            2 => {
                expect_wire(field.wire, 2, "TensorTypeProto.shape")?;
                if shape
                    .replace(parse_shape(
                        field.payload.expect("length-delimited field"),
                        context,
                    )?)
                    .is_some()
                {
                    bail!("{context} TensorTypeProto.shape is repeated")
                }
            }
            _ => {}
        }
    }
    let element_type = element_type
        .ok_or_else(|| anyhow::anyhow!("{context} TensorTypeProto.elem_type is missing"))?;
    let shape =
        shape.ok_or_else(|| anyhow::anyhow!("{context} TensorTypeProto.shape is missing"))?;
    Ok((element_type_name(element_type)?, shape))
}

fn parse_shape(bytes: &[u8], context: &str) -> Result<Vec<Dim>> {
    let mut dims = Vec::new();
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        if field.number == 1 {
            expect_wire(field.wire, 2, "TensorShapeProto.dim")?;
            dims.push(parse_symbolic_dim(
                field.payload.expect("length-delimited field"),
                context,
            )?);
        }
    }
    if dims.is_empty() {
        bail!("{context} TensorShapeProto.shape has no dimensions")
    }
    Ok(dims)
}

fn parse_symbolic_dim(bytes: &[u8], context: &str) -> Result<Dim> {
    let mut value = None;
    let mut symbolic = None;
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        match field.number {
            1 => {
                expect_wire(field.wire, 0, "TensorShapeProto.Dimension.dim_value")?;
                value = Some(field.varint.expect("varint field") as i64);
            }
            2 => {
                expect_wire(field.wire, 2, "TensorShapeProto.Dimension.dim_param")?;
                symbolic = Some(parse_string(
                    field.payload.expect("length-delimited field"),
                    "dimension parameter",
                )?);
            }
            _ => {}
        }
    }
    match (value, symbolic) {
        (Some(_), Some(_)) => bail!("{context} has both dim_value and dim_param set"),
        (Some(value), None) => Ok(Dim::Fixed(value)),
        (None, Some(symbolic)) => Ok(Dim::Symbolic(symbolic)),
        (None, None) => bail!("{context} has neither dim_value nor dim_param set"),
    }
}

fn element_type_name(value: i32) -> Result<String> {
    let name = match value {
        1 => ELEMENT_TYPE_F32,
        2 => "u8",
        3 => "i8",
        4 => "u16",
        5 => "i16",
        6 => "i32",
        7 => "i64",
        8 => "str",
        9 => "bool",
        10 => "f16",
        11 => "f64",
        12 => "u32",
        13 => "u64",
        14 => "c64",
        15 => "c128",
        16 => "bf16",
        _ => bail!("unknown ONNX tensor element type {value}"),
    };
    Ok(name.into())
}

fn parse_string(bytes: &[u8], context: &str) -> Result<String> {
    String::from_utf8(bytes.to_vec()).with_context(|| format!("{context} is not UTF-8"))
}

fn expect_wire(actual: u8, expected: u8, field: &str) -> Result<()> {
    if actual != expected {
        bail!("{field} has wire type {actual}, expected {expected}")
    }
    Ok(())
}

struct Field<'a> {
    number: u32,
    wire: u8,
    payload: Option<&'a [u8]>,
    varint: Option<u64>,
}

fn next_field<'a>(input: &mut &'a [u8]) -> Result<Option<Field<'a>>> {
    if input.is_empty() {
        return Ok(None);
    }
    let key = read_varint(input)?;
    let wire = (key & 7) as u8;
    let number = (key >> 3) as u32;
    if number == 0 {
        bail!("protobuf field number 0 is invalid")
    }
    match wire {
        0 => Ok(Some(Field {
            number,
            wire,
            payload: None,
            varint: Some(read_varint(input)?),
        })),
        1 => {
            take(input, 8)?;
            Ok(Some(Field {
                number,
                wire,
                payload: None,
                varint: None,
            }))
        }
        2 => {
            let length = read_varint(input)? as usize;
            let payload = take(input, length)?;
            Ok(Some(Field {
                number,
                wire,
                payload: Some(payload),
                varint: None,
            }))
        }
        3 => {
            skip_group(input)?;
            Ok(Some(Field {
                number,
                wire,
                payload: None,
                varint: None,
            }))
        }
        4 => bail!("unexpected protobuf end-group wire type"),
        5 => {
            take(input, 4)?;
            Ok(Some(Field {
                number,
                wire,
                payload: None,
                varint: None,
            }))
        }
        _ => bail!("invalid protobuf wire type {wire}"),
    }
}

fn skip_group(input: &mut &[u8]) -> Result<()> {
    loop {
        let key = read_varint(input)?;
        match (key & 7) as u8 {
            0 => {
                read_varint(input)?;
            }
            1 => {
                take(input, 8)?;
            }
            2 => {
                let length = read_varint(input)? as usize;
                take(input, length)?;
            }
            3 => skip_group(input)?,
            4 => return Ok(()),
            5 => {
                take(input, 4)?;
            }
            wire => bail!("invalid protobuf wire type {wire} inside group"),
        }
    }
}

fn read_varint(input: &mut &[u8]) -> Result<u64> {
    let mut value = 0_u64;
    for shift in (0..64).step_by(7) {
        let byte = *input
            .first()
            .ok_or_else(|| anyhow::anyhow!("truncated protobuf varint"))?;
        *input = &input[1..];
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    bail!("protobuf varint is too long")
}

fn take<'a>(input: &mut &'a [u8], length: usize) -> Result<&'a [u8]> {
    if input.len() < length {
        bail!(
            "truncated protobuf field: need {length} bytes, have {}",
            input.len()
        )
    }
    let (head, tail) = input.split_at(length);
    *input = tail;
    Ok(head)
}

struct StagedFiles([PathBuf; 3]);

impl StagedFiles {
    fn new(paths: [PathBuf; 3]) -> Self {
        Self(paths)
    }
}

impl Drop for StagedFiles {
    fn drop(&mut self) {
        for path in &self.0 {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating staging file {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("writing staging file {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing staging file {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_symbolic_bytes, Dim};

    fn varint(mut value: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            bytes.push(byte);
            if value == 0 {
                return bytes;
            }
        }
    }

    fn field(number: u32, wire: u8, payload: &[u8]) -> Vec<u8> {
        let mut bytes = varint(u64::from(number << 3 | u32::from(wire)));
        match wire {
            0 => bytes.extend_from_slice(payload),
            2 => {
                bytes.extend_from_slice(&varint(payload.len() as u64));
                bytes.extend_from_slice(payload);
            }
            _ => panic!("test helper only needs varint and length-delimited fields"),
        }
        bytes
    }

    fn value_info(name: &str, shape: Vec<Vec<u8>>) -> Vec<u8> {
        let mut shape_bytes = Vec::new();
        for dim in shape {
            shape_bytes.extend(field(1, 2, &dim));
        }
        let tensor_type = [field(1, 0, &[1]), field(2, 2, &shape_bytes)].concat();
        let type_proto = field(1, 2, &tensor_type);
        [field(1, 2, name.as_bytes()), field(2, 2, &type_proto)].concat()
    }

    fn model_with_dim(dim: Vec<u8>) -> Vec<u8> {
        let input = value_info("input", vec![dim.clone(), field(1, 0, &[3])]);
        let output = value_info("output", vec![dim, field(1, 0, &[3])]);
        let graph = [
            field(1, 2, b"synthetic"),
            field(11, 2, &input),
            field(12, 2, &output),
        ]
        .concat();
        let opset = field(2, 0, &[14]);
        [field(7, 2, &graph), field(8, 2, &opset)].concat()
    }

    #[test]
    fn symbolic_parser_accepts_mixed_symbolic_and_fixed_shape() {
        let bytes = model_with_dim(field(2, 2, b"batch"));
        let signature = parse_symbolic_bytes(&bytes, "synthetic.onnx", "a".into()).unwrap();
        assert_eq!(signature.opset, 14);
        assert_eq!(
            signature.inputs[0].shape,
            vec![Dim::Symbolic("batch".into()), Dim::Fixed(3)]
        );
        assert_eq!(
            signature.outputs[0].shape,
            vec![Dim::Symbolic("batch".into()), Dim::Fixed(3)]
        );
    }

    #[test]
    fn symbolic_parser_rejects_both_set_dimension() {
        let both = [field(1, 0, &[2]), field(2, 2, b"batch")].concat();
        let error =
            parse_symbolic_bytes(&model_with_dim(both), "synthetic.onnx", "a".into()).unwrap_err();
        assert!(error.to_string().contains("both dim_value and dim_param"));
    }

    #[test]
    fn symbolic_parser_rejects_neither_set_dimension() {
        let error = parse_symbolic_bytes(&model_with_dim(Vec::new()), "synthetic.onnx", "a".into())
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("neither dim_value nor dim_param"));
    }
}
