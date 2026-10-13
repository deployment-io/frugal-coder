//! Bounded metadata-only inspection of a flat, single-file safetensors layout.
use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    fs::{self, File},
    io::Read,
    path::Path,
};

pub const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
pub const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_TENSORS: usize = 100_000;
pub const MAX_RANK: usize = 32;
pub const MAX_DIRECTORY_ENTRIES: usize = 4096;

#[derive(Debug)]
pub struct InspectError(pub String);
impl fmt::Display for InspectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for InspectError {}
type Result<T> = std::result::Result<T, InspectError>;
fn err(message: impl Into<String>) -> InspectError {
    InspectError(message.into())
}

#[derive(Debug, Serialize)]
pub struct Architecture {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub text_model_type: Option<String>,
    pub recognized: bool,
    pub execution_implemented: bool,
}
#[derive(Debug, Serialize)]
pub struct TensorMetadata {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub data_offsets: [u64; 2],
    pub payload_bytes: u64,
}
#[derive(Debug, Serialize)]
pub struct WeightFile {
    pub name: String,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub payload_bytes: u64,
    pub tensors: Vec<TensorMetadata>,
}
#[derive(Debug, Serialize)]
pub struct Inspection {
    pub schema_version: u32,
    pub architecture: Architecture,
    pub layout: String,
    pub files: Vec<WeightFile>,
    pub total_tensor_payload_bytes: u64,
    pub total_weight_file_bytes: u64,
    pub limitations: Vec<String>,
}

// serde_json::Value normally accepts duplicate object keys by overwriting them.
// Reject them recursively so config fields, tensors, and offsets are unambiguous.
struct UniqueJson(Value);
impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct UniqueVisitor;
        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = UniqueJson;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("JSON without duplicate keys")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut result = serde_json::Map::new();
                while let Some((key, value)) = map.next_entry::<String, UniqueJson>()? {
                    if result.insert(key.clone(), value.0).is_some() {
                        return Err(de::Error::custom(format!("duplicate key: {key}")));
                    }
                }
                Ok(UniqueJson(Value::Object(result)))
            }
            fn visit_seq<S: SeqAccess<'de>>(
                self,
                mut seq: S,
            ) -> std::result::Result<Self::Value, S::Error> {
                let mut result = Vec::new();
                while let Some(value) = seq.next_element::<UniqueJson>()? {
                    result.push(value.0);
                }
                Ok(UniqueJson(Value::Array(result)))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(v.into())))
            }
            fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(v)))
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::Bool(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| UniqueJson(Value::Number(n)))
                    .ok_or_else(|| de::Error::custom("nonfinite number"))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }
        }
        deserializer.deserialize_any(UniqueVisitor)
    }
}

fn parse_json(bytes: &[u8], context: &str) -> Result<Value> {
    serde_json::from_slice::<UniqueJson>(bytes)
        .map(|v| v.0)
        .map_err(|e| err(format!("{context}: invalid JSON: {e}")))
}

fn open_regular(path: &Path, name: &str) -> Result<(File, u64)> {
    let metadata = fs::symlink_metadata(path).map_err(|e| err(format!("{name}: {e}")))?;
    if !metadata.file_type().is_file() {
        return Err(err(format!(
            "unsupported layout: {name} must be a regular file, not a symlink or directory"
        )));
    }
    let file = File::open(path).map_err(|e| err(format!("{name}: {e}")))?;
    let metadata = file.metadata().map_err(|e| err(format!("{name}: {e}")))?;
    if !metadata.is_file() {
        return Err(err(format!("{name}: not a regular file")));
    }
    Ok((file, metadata.len()))
}

fn nonempty_string(value: Option<&Value>, field: &str) -> Result<String> {
    value
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| err(format!("config.json: {field} must be a nonempty string")))
}

fn configuration(directory: &Path) -> Result<Architecture> {
    let (file, len) = open_regular(&directory.join("config.json"), "config.json")?;
    if len > MAX_CONFIG_BYTES {
        return Err(err("config.json exceeds 1 MiB metadata limit"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| err(format!("config.json: {e}")))?;
    if bytes.len() as u64 > MAX_CONFIG_BYTES {
        return Err(err("config.json exceeds 1 MiB metadata limit"));
    }
    let config = parse_json(&bytes, "config.json")?;
    let object = config
        .as_object()
        .ok_or_else(|| err("config.json must be an object"))?;
    let model_type = nonempty_string(object.get("model_type"), "model_type")?;
    let declared = object
        .get("architectures")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| err("config.json: architectures must be a nonempty string array"))?;
    let architectures = declared
        .iter()
        .map(|a| nonempty_string(Some(a), "architectures entry"))
        .collect::<Result<Vec<_>>>()?;
    let text_model_type = match object.get("text_config") {
        None => None,
        Some(Value::Object(text)) => Some(nonempty_string(
            text.get("model_type"),
            "text_config.model_type",
        )?),
        Some(_) => return Err(err("config.json: text_config must be an object")),
    };
    let recognized = matches!((model_type.as_str(), architectures.as_slice()),
        ("qwen3_5", [name]) if name == "Qwen3_5ForConditionalGeneration")
        && text_model_type.as_deref() == Some("qwen3_5_text");
    Ok(Architecture {
        model_type,
        architectures,
        text_model_type,
        recognized,
        execution_implemented: false,
    })
}

fn validate_layout(directory: &Path) -> Result<()> {
    let entries = fs::read_dir(directory).map_err(|e| err(format!("model directory: {e}")))?;
    for (count, entry) in entries.enumerate() {
        if count >= MAX_DIRECTORY_ENTRIES {
            return Err(err("model directory exceeds 4096-entry limit"));
        }
        let entry = entry.map_err(|e| err(format!("model directory entry: {e}")))?;
        let name = entry.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| err("unsupported layout: non-UTF-8 filename"))?;
        let kind = entry.file_type().map_err(|e| err(format!("{name}: {e}")))?;
        if !kind.is_file() {
            return Err(err(format!(
                "unsupported layout: flat regular files required ({name})"
            )));
        }
        if name.ends_with(".index.json")
            || (name.ends_with(".safetensors") && name != "model.safetensors")
        {
            return Err(err(format!("unsupported layout: sharded, indexed, or multiple weight files ({name}); M0 requires only model.safetensors")));
        }
        if [".bin", ".gguf", ".pt", ".pth"]
            .iter()
            .any(|ext| name.ends_with(ext))
        {
            return Err(err(format!(
                "unsupported layout: non-safetensors weights ({name})"
            )));
        }
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn dtype_bytes(dtype: &str) -> Result<u64> {
    match dtype {
        "BOOL" | "I8" | "U8" | "F8_E4M3" | "F8_E5M2" => Ok(1),
        "I16" | "U16" | "F16" | "BF16" => Ok(2),
        "I32" | "U32" | "F32" => Ok(4),
        "I64" | "U64" | "F64" => Ok(8),
        _ => Err(err(format!("unsupported safetensors dtype: {dtype}"))),
    }
}

/// Inspects metadata only. Payload values, integrity, model completeness and
/// runtime RAM requirements are not checked. Files must remain stable while read.
pub fn inspect(directory: &Path) -> Result<Inspection> {
    validate_layout(directory)?;
    let architecture = configuration(directory)?;
    let (mut file, file_bytes) =
        open_regular(&directory.join("model.safetensors"), "model.safetensors")?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|e| err(format!("model.safetensors: truncated header length: {e}")))?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes == 0 || header_bytes > MAX_HEADER_BYTES {
        return Err(err("safetensors header must be 1 byte through 16 MiB"));
    }
    let payload_start = 8u64
        .checked_add(header_bytes)
        .ok_or_else(|| err("header-size overflow"))?;
    let payload_bytes = file_bytes
        .checked_sub(payload_start)
        .ok_or_else(|| err("model.safetensors: truncated header"))?;
    let mut header = vec![0; header_bytes as usize]; // Bound is portable even on 32-bit hosts.
    file.read_exact(&mut header)
        .map_err(|e| err(format!("model.safetensors: truncated header: {e}")))?;
    if header.first() != Some(&b'{') {
        return Err(err("safetensors header must begin with '{'"));
    }
    let parsed = parse_json(&header, "safetensors header")?;
    let entries = parsed
        .as_object()
        .ok_or_else(|| err("safetensors header must be an object"))?;
    if let Some(metadata) = entries.get("__metadata__") {
        serde_json::from_value::<BTreeMap<String, String>>(metadata.clone()).map_err(|e| {
            err(format!(
                "safetensors __metadata__ must map strings to strings: {e}"
            ))
        })?;
    }
    let mut tensors = Vec::new();
    for (name, value) in entries
        .iter()
        .filter(|(name, _)| name.as_str() != "__metadata__")
    {
        if tensors.len() >= MAX_TENSORS {
            return Err(err("safetensors exceeds 100000-tensor limit"));
        }
        let raw: RawTensor = serde_json::from_value(value.clone())
            .map_err(|e| err(format!("tensor {name:?}: {e}")))?;
        if raw.shape.len() > MAX_RANK {
            return Err(err(format!("tensor {name:?}: rank exceeds 32")));
        }
        // Empty tensors occupy zero bytes; still check nonzero dimension products.
        let nonzero_elements = raw
            .shape
            .iter()
            .filter(|&&d| d != 0)
            .try_fold(1u64, |n, &d| n.checked_mul(d))
            .ok_or_else(|| err(format!("tensor {name:?}: shape multiplication overflow")))?;
        let elements = if raw.shape.contains(&0) {
            0
        } else {
            nonzero_elements
        };
        let expected = elements
            .checked_mul(dtype_bytes(&raw.dtype)?)
            .ok_or_else(|| err(format!("tensor {name:?}: byte-size overflow")))?;
        let [start, end] = raw.data_offsets;
        if end < start || end > payload_bytes || end - start != expected {
            return Err(err(format!("tensor {name:?}: invalid byte range [{start}, {end}] for {expected} bytes and {payload_bytes}-byte payload")));
        }
        tensors.push(TensorMetadata {
            name: name.clone(),
            shape: raw.shape,
            dtype: raw.dtype,
            data_offsets: raw.data_offsets,
            payload_bytes: expected,
        });
    }
    // Empty ranges sort before nonempty ones at the same start. Require exact
    // coverage of the payload: no overlaps, holes, or unreferenced trailing bytes.
    let mut ranges: Vec<_> = tensors.iter().map(|t| t.data_offsets).collect();
    ranges.sort_unstable();
    let mut cursor = 0;
    for [start, end] in ranges {
        if start != cursor {
            return Err(err("safetensors payload has overlapping ranges or gaps"));
        }
        cursor = end;
    }
    if cursor != payload_bytes {
        return Err(err(
            "safetensors payload contains unreferenced trailing bytes",
        ));
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Inspection {
        schema_version: 1, architecture, layout: "single_safetensors".into(),
        files: vec![WeightFile { name: "model.safetensors".into(), file_bytes, header_bytes, payload_bytes, tensors }],
        total_tensor_payload_bytes: payload_bytes, total_weight_file_bytes: file_bytes,
        limitations: vec![
            "Metadata and byte layout validated; weight payload values and integrity not verified.".into(),
            "Architecture recognition is declarative; tensor names are not checked against a model architecture.".into(),
            "Model execution is not implemented; file and payload bytes are not runtime RAM estimates.".into(),
            "Only a flat directory with config.json and one model.safetensors is supported; indexed/sharded layouts are rejected.".into(),
            "Other regular assets are ignored; tokenizer, prompts, and configuration semantics are not validated.".into(),
        ],
    })
}
