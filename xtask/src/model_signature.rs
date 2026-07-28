//! Task F3: the dependency-free recorder for the declared ONNX graph signature.

use anyhow::{bail, Context, Result};
use pc_models::{sha256_hex, verify_sha256, COMIC_TEXT_DETECTOR};
use pc_testkit::model_signature::{ModelSignature, TensorSignature, ELEMENT_TYPE_F32};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

const OUTPUT_REL: &str = "model_signature/comictextdetector.signature.json";
const PROVENANCE_REL: &str = "model_signature/PROVENANCE.json";

#[derive(Debug, Serialize)]
struct Provenance {
    tool: &'static str,
    source: String,
    source_sha256: String,
    output: &'static str,
    output_sha256: String,
    size_bytes: u64,
    opset: i64,
}

#[derive(Debug, Deserialize)]
struct PublishedProvenance {
    output_sha256: String,
    source_sha256: String,
}

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
    let provenance = Provenance {
        tool: "cargo xtask record-fixtures --only model-signature",
        source: COMIC_TEXT_DETECTOR.file_name.to_owned(),
        source_sha256: signature.sha256.clone(),
        output: "tests/fixtures/recorded/model_signature/comictextdetector.signature.json",
        output_sha256: output_sha256.clone(),
        size_bytes: signature.size_bytes,
        opset: signature.opset,
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
    let written_provenance: PublishedProvenance = serde_json::from_slice(&written_provenance)
        .with_context(|| format!("parsing published {}", provenance_path.display()))?;
    if written_provenance.output_sha256 != written_signature_sha256 {
        bail!(
            "published model-signature pair is inconsistent: provenance output_sha256 {} != signature sha256 {}",
            written_provenance.output_sha256,
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
    let provenance: PublishedProvenance =
        serde_json::from_slice(&provenance_json).with_context(|| {
            format!(
                "existing model-signature pair is invalid: {} does not parse as provenance; \
             re-recording with --force will repair it",
                provenance_path.display()
            )
        })?;
    if provenance.output_sha256 != actual_sha256 {
        bail!(
            "existing model-signature pair is inconsistent: provenance output_sha256 {} != actual signature sha256 {}; re-recording with --force will repair it",
            provenance.output_sha256,
            actual_sha256
        );
    }
    if provenance.source_sha256 != COMIC_TEXT_DETECTOR.sha256 {
        bail!(
            "existing model-signature pair is inconsistent: provenance source_sha256 {} != COMIC_TEXT_DETECTOR.sha256 {}; re-recording with --force will repair it",
            provenance.source_sha256,
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
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const ROUND_CONSTANTS: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let padded_len = (bytes.len() + 9).div_ceil(64) * 64;
    let mut padded = Vec::with_capacity(padded_len);
    padded.extend_from_slice(bytes);
    padded.push(0x80);
    padded.resize(padded_len - 8, 0);
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = INITIAL;
    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (word, bytes) in words[..16].iter_mut().zip(chunk.chunks_exact(4)) {
            *word = u32::from_be_bytes(bytes.try_into().expect("SHA-256 word is four bytes"));
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let big_sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(big_sigma1)
                .wrapping_add(choose)
                .wrapping_add(ROUND_CONSTANTS[index])
                .wrapping_add(words[index]);
            let big_sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = big_sigma0.wrapping_add(majority);
            [h, g, f, e, d, c, b, a] = [
                g,
                f,
                e,
                d.wrapping_add(temp1),
                c,
                b,
                a,
                temp1.wrapping_add(temp2),
            ];
        }
        for (value, addition) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *value = value.wrapping_add(addition);
        }
    }

    state.iter().map(|word| format!("{word:08x}")).collect()
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
