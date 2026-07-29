//! Task F3: the dependency-free recorder for the declared ONNX graph signature.

use anyhow::{bail, Context, Result};
use pc_models::{sha256_hex, verify_sha256, COMIC_TEXT_DETECTOR};
use pc_testkit::model_signature::{ModelSignature, TensorSignature, ELEMENT_TYPE_F32};
use pc_testkit::provenance::{ArtifactRecord, GroupProvenance};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

const OUTPUT_REL: &str = "model_signature/comictextdetector.signature.json";
const PROVENANCE_REL: &str = "model_signature/PROVENANCE.json";

pub fn capability_status(explicit: Option<&Path>) -> String {
    let output = crate::paths::recorded_root().join(OUTPUT_REL);
    if output.is_file() {
        return format!(
            "RECORDED — {} (source can be re-verified with --model-signature PATH)",
            crate::paths::display_relative(&output)
        );
    }
    match source_path(explicit) {
        Some(path) if path.is_file() => format!("AVAILABLE — {}", path.display()),
        Some(path) => format!(
            "SKIPPED — model file is absent at {}; provide --model-signature PATH or set \
             PANEL_OCR_ONNX_MODEL",
            path.display()
        ),
        None => "SKIPPED — no model path; provide --model-signature PATH or set \
                  PANEL_OCR_ONNX_MODEL"
            .into(),
    }
}

pub fn source_path(explicit: Option<&Path>) -> Option<PathBuf> {
    explicit
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("PANEL_OCR_ONNX_MODEL").map(PathBuf::from))
}

pub fn record(explicit: Option<&Path>, force: bool) -> Result<super::record::Outcome> {
    let output = crate::paths::recorded_root().join(OUTPUT_REL);
    let provenance_path = crate::paths::recorded_root().join(PROVENANCE_REL);
    if output.is_file() && !force {
        validate_existing_pair(&output, &provenance_path)?;
        return Ok(super::record::Outcome::Recorded {
            detail: format!(
                "{} already present (use --force to re-record)",
                crate::paths::display_relative(&output)
            ),
        });
    }

    let Some(source) = source_path(explicit) else {
        return Ok(super::record::Outcome::Skipped {
            reason: "the model file is not configured. Provide --model-signature PATH or set \
                     PANEL_OCR_ONNX_MODEL to the sha256-verified comictextdetector.pt.onnx; \
                     no fixture was emitted."
                .into(),
        });
    };
    if !source.is_file() {
        return Ok(super::record::Outcome::Skipped {
            reason: format!(
                "model file is absent at {}. Provide --model-signature PATH or set \
                 PANEL_OCR_ONNX_MODEL to the sha256-verified comictextdetector.pt.onnx; no \
                 fixture was emitted.",
                source.display()
            ),
        });
    }

    // Verify the real artifact before parsing or emitting any signature/provenance bytes.
    verify_sha256(&source, COMIC_TEXT_DETECTOR.sha256)
        .with_context(|| format!("verifying the model-signature source {}", source.display()))?;
    let signature = parse_signature(&source)?;
    println!("parsed model signature:\n{signature:#?}");

    let output_dir = output
        .parent()
        .expect("model signature output has a parent")
        .to_path_buf();

    let mut signature_json = serde_json::to_vec_pretty(&signature)?;
    signature_json.push(b'\n');
    let output_sha256 = sha256_hex_bytes(&signature_json);
    let provenance = GroupProvenance {
        schema_version: pc_testkit::provenance::PROVENANCE_SCHEMA_VERSION,
        group: "model_signature".into(),
        tool: "cargo xtask record-fixtures --only model-signature".into(),
        command_line: "cargo xtask record-fixtures --only model-signature".into(),
        tool_versions: BTreeMap::new(),
        records: vec![ArtifactRecord {
            name: "signature".into(),
            output: "tests/fixtures/recorded/model_signature/comictextdetector.signature.json"
                .into(),
            output_sha256: output_sha256.clone(),
            committed: true,
            source: Some(COMIC_TEXT_DETECTOR.file_name.into()),
            source_sha256: Some(signature.sha256.clone()),
            params: BTreeMap::from([
                ("opset".into(), serde_json::json!(signature.opset)),
                ("size_bytes".into(), serde_json::json!(signature.size_bytes)),
            ]),
        }],
        detector: None,
        diagnostics: BTreeMap::new(),
    };
    let mut provenance_json = serde_json::to_vec_pretty(&provenance)?;
    provenance_json.push(b'\n');

    if let Some(old_signature_json) = output
        .is_file()
        .then(|| std::fs::read(&output))
        .transpose()
        .with_context(|| format!("reading existing {}", output.display()))?
    {
        let old_sha256 = sha256_hex_bytes(&old_signature_json);
        if old_signature_json == signature_json {
            println!(
                "signature unchanged: {} (sha256 {})",
                crate::paths::display_relative(&output),
                output_sha256
            );
        } else {
            println!(
                "signature bytes differ before overwrite: old sha256 {}, new sha256 {}",
                old_sha256, output_sha256
            );
        }
    }

    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating {}", output_dir.display()))?;

    let signature_temp = pc_models::partial_path(&output);
    let provenance_temp = pc_models::partial_path(&provenance_path);
    let _staged_files = StagedFiles::new([signature_temp.clone(), provenance_temp.clone()]);
    write_synced(&signature_temp, &signature_json)?;
    write_synced(&provenance_temp, &provenance_json)?;

    // Keep these renames adjacent: all fallible serialization, staging, and path work is
    // complete before the publication window begins.
    std::fs::rename(&signature_temp, &output)
        .with_context(|| format!("publishing {}", output.display()))?;
    std::fs::rename(&provenance_temp, &provenance_path)
        .with_context(|| format!("publishing {}", provenance_path.display()))?;

    let written_signature = std::fs::read(&output)
        .with_context(|| format!("reading published {}", output.display()))?;
    let written_provenance = std::fs::read(&provenance_path)
        .with_context(|| format!("reading published {}", provenance_path.display()))?;
    let written_signature_sha256 = sha256_hex_bytes(&written_signature);
    let written_provenance: GroupProvenance = serde_json::from_slice(&written_provenance)
        .with_context(|| format!("parsing published {}", provenance_path.display()))?;
    let written_record = written_provenance
        .records
        .iter()
        .find(|record| record.name == "signature")
        .context("published model-signature provenance has no signature record")?;
    if written_record.output_sha256 != written_signature_sha256 {
        bail!(
            "published model-signature pair is inconsistent: provenance output_sha256 {} != signature sha256 {}",
            written_record.output_sha256,
            written_signature_sha256
        );
    }

    Ok(super::record::Outcome::Recorded {
        detail: format!(
            "{} ({} bytes, source sha256 {})",
            crate::paths::display_relative(&output),
            signature.size_bytes,
            signature.sha256
        ),
    })
}

fn validate_existing_pair(output: &Path, provenance_path: &Path) -> Result<()> {
    let signature_json =
        std::fs::read(output).with_context(|| format!("reading existing {}", output.display()))?;
    let actual_sha256 = sha256_hex_bytes(&signature_json);
    serde_json::from_slice::<ModelSignature>(&signature_json).with_context(|| {
        format!(
            "existing model-signature pair is invalid: {} does not parse as a model signature; \
             re-recording with --force will repair it",
            output.display()
        )
    })?;

    let provenance_json = std::fs::read(provenance_path).with_context(|| {
        format!(
            "existing model-signature pair is incomplete: could not read {}; \
             re-recording with --force will repair it",
            provenance_path.display()
        )
    })?;
    let provenance: GroupProvenance =
        serde_json::from_slice(&provenance_json).with_context(|| {
            format!(
                "existing model-signature pair is invalid: {} does not parse as provenance; \
             re-recording with --force will repair it",
                provenance_path.display()
            )
        })?;
    let record = provenance
        .records
        .iter()
        .find(|record| record.name == "signature")
        .context("existing model-signature provenance has no signature record")?;
    if record.output_sha256 != actual_sha256 {
        bail!(
            "existing model-signature pair is inconsistent: provenance output_sha256 {} != actual signature sha256 {}; re-recording with --force will repair it",
            record.output_sha256,
            actual_sha256
        );
    }
    if record.source_sha256.as_deref() != Some(COMIC_TEXT_DETECTOR.sha256) {
        bail!(
            "existing model-signature pair is inconsistent: provenance source_sha256 {} != COMIC_TEXT_DETECTOR.sha256 {}; re-recording with --force will repair it",
            record.source_sha256.as_deref().unwrap_or("<missing>"),
            COMIC_TEXT_DETECTOR.sha256
        );
    }
    Ok(())
}

struct StagedFiles([PathBuf; 2]);

impl StagedFiles {
    fn new(paths: [PathBuf; 2]) -> Self {
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

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn parse_signature(path: &Path) -> Result<ModelSignature> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let size_bytes = bytes.len() as u64;
    let sha256 = sha256_hex(path).context("hashing the verified model")?;
    let model_file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "model path has no valid UTF-8 file name: {}",
                path.display()
            )
        })?;
    if model_file_name != COMIC_TEXT_DETECTOR.file_name {
        bail!(
            "the verified model is named `{model_file_name}`, expected `{}`",
            COMIC_TEXT_DETECTOR.file_name
        );
    }

    let mut graph: Option<&[u8]> = None;
    let mut default_opsets = Vec::new();
    let mut cursor = bytes.as_slice();
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
    let opset = default_opsets[0];
    if opset != 11 {
        bail!("comictextdetector pinned opset is 11, found {opset}");
    }

    // The parser extracts name/element-type/dims from top-level `graph.input`/`graph.output`
    // `ValueInfoProto`s, skipping unknown fields by wire type. It is not to grow into a
    // general ONNX reader.
    let (inputs, outputs) = parse_graph(graph)?;
    let signature = ModelSignature {
        schema_version: 1,
        model_file_name: model_file_name.to_owned(),
        sha256,
        size_bytes,
        opset,
        inputs,
        outputs,
    };
    validate_pinned_signature(&signature)?;
    Ok(signature)
}

fn parse_graph(bytes: &[u8]) -> Result<(Vec<TensorSignature>, Vec<TensorSignature>)> {
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
    if inputs.len() != 1 {
        bail!(
            "expected exactly one true graph input after subtracting initializers, found {}",
            inputs.len()
        );
    }
    Ok((inputs, outputs))
}

fn validate_pinned_signature(signature: &ModelSignature) -> Result<()> {
    let expected_input = TensorSignature {
        name: "images".into(),
        element_type: ELEMENT_TYPE_F32.into(),
        shape: vec![1, 3, 1024, 1024],
    };
    let expected_outputs = vec![
        TensorSignature {
            name: "blk".into(),
            element_type: ELEMENT_TYPE_F32.into(),
            shape: vec![1, 64_512, 7],
        },
        TensorSignature {
            name: "seg".into(),
            element_type: ELEMENT_TYPE_F32.into(),
            shape: vec![1, 1, 1024, 1024],
        },
        TensorSignature {
            name: "det".into(),
            element_type: ELEMENT_TYPE_F32.into(),
            shape: vec![1, 2, 1024, 1024],
        },
    ];
    if signature.inputs != [expected_input] {
        bail!("verified model's parsed input signature is not the pinned images f32 [1, 3, 1024, 1024]: {signature:?}");
    }
    if signature.outputs != expected_outputs {
        bail!("verified model's parsed output signature is not the pinned blk/seg/det signature: {signature:?}");
    }
    Ok(())
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

fn parse_value_info(bytes: &[u8], context: &str) -> Result<TensorSignature> {
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
    Ok(TensorSignature {
        name,
        element_type,
        shape,
    })
}

fn parse_type_proto(bytes: &[u8], context: &str) -> Result<(String, Vec<i64>)> {
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

fn parse_tensor_type(bytes: &[u8], context: &str) -> Result<(String, Vec<i64>)> {
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

fn parse_shape(bytes: &[u8], context: &str) -> Result<Vec<i64>> {
    let mut dims = Vec::new();
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        if field.number == 1 {
            expect_wire(field.wire, 2, "TensorShapeProto.dim")?;
            dims.push(parse_dim(
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

fn parse_dim(bytes: &[u8], context: &str) -> Result<i64> {
    let mut value = None;
    let mut symbolic = false;
    let mut cursor = bytes;
    while let Some(field) = next_field(&mut cursor)? {
        match field.number {
            1 => {
                expect_wire(field.wire, 0, "TensorShapeProto.Dimension.dim_value")?;
                value = Some(field.varint.expect("varint field") as i64);
            }
            2 => {
                expect_wire(field.wire, 2, "TensorShapeProto.Dimension.dim_param")?;
                symbolic = true;
            }
            _ => {}
        }
    }
    if symbolic {
        bail!("{context} has a symbolic/dynamic dimension")
    }
    let value = value.ok_or_else(|| anyhow::anyhow!("{context} has a missing dimension value"))?;
    if value <= 0 {
        bail!("{context} has an invalid zero/negative dimension {value}")
    }
    Ok(value)
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
