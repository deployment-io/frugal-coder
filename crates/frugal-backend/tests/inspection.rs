use frugal_backend::inspect::{inspect, MAX_CONFIG_BYTES, MAX_HEADER_BYTES};
use serde_json::json;
use std::fs::{self, File};
#[path = "../../../evals/test-support/mod.rs"]
mod support;
use support::ModelDir;

fn fails(dir: &ModelDir, expected: &str) {
    let error = inspect(dir.path()).unwrap_err().to_string();
    assert!(
        error.contains(expected),
        "expected {expected:?}, got {error:?}"
    );
}

#[test]
fn valid_unknown_and_synthetic_recognized_model() {
    let dir = ModelDir::new();
    let report = inspect(dir.path()).unwrap();
    assert!(!report.architecture.recognized);
    assert!(!report.architecture.execution_implemented);
    assert_eq!(report.total_tensor_payload_bytes, 4);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../evals/fixtures/synthetic-model");
    let report = inspect(&path).unwrap();
    assert!(report.architecture.recognized);
    assert!(!report.architecture.execution_implemented);
    assert_eq!(report.schema_version, 1);
    assert_eq!(report.total_tensor_payload_bytes, 36);
    assert_eq!(
        report.total_weight_file_bytes,
        fs::metadata(path.join("model.safetensors")).unwrap().len()
    );
    let file = &report.files[0];
    assert_eq!(file.file_bytes, 8 + file.header_bytes + file.payload_bytes);
    assert_eq!(
        file.tensors
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["demo.scale", "demo.weight"]
    );
    assert_eq!(file.tensors[1].shape, [2, 3]);
    assert_eq!(file.tensors[1].dtype, "F32");
}

#[test]
fn malformed_configurations_and_recognition_mismatch() {
    let dir = ModelDir::new();
    for bad in [
        "{",
        "[]",
        "{}",
        "null",
        r#"{"model_type":3,"architectures":["X"]}"#,
        r#"{"model_type":"x","architectures":[]}"#,
        r#"{"model_type":"x","architectures":[3]}"#,
        r#"{"model_type":"x","architectures":["X"],"text_config":null}"#,
        r#"{"model_type":"x","architectures":["X"],"text_config":{}}"#,
        r#"{"model_type":"x","model_type":"y","architectures":["X"]}"#,
    ] {
        dir.write("config.json", bad.as_bytes());
        fails(&dir, "config.json");
    }
    dir.write(
        "config.json",
        br#"{"model_type":"qwen3_5","architectures":["Unrelated"]}"#,
    );
    assert!(!inspect(dir.path()).unwrap().architecture.recognized);
    dir.write("config.json", &[0xff]);
    fails(&dir, "invalid JSON");
}

#[test]
fn missing_files_and_metadata_limits() {
    let dir = ModelDir::new();
    fs::remove_file(dir.path().join("config.json")).unwrap();
    fails(&dir, "config.json");
    dir.write("config.json", b"{}");
    File::options()
        .write(true)
        .open(dir.path().join("config.json"))
        .unwrap()
        .set_len(MAX_CONFIG_BYTES + 1)
        .unwrap();
    fails(&dir, "1 MiB");
    let dir = ModelDir::new();
    fs::remove_file(dir.path().join("model.safetensors")).unwrap();
    fails(&dir, "model.safetensors");
    for length in [0, MAX_HEADER_BYTES + 1, u64::MAX] {
        dir.write("model.safetensors", &length.to_le_bytes());
        fails(&dir, "16 MiB");
    }
    let dir = ModelDir::new();
    assert!(inspect(&dir.path().join("missing")).is_err());
}

#[test]
fn malformed_and_truncated_headers() {
    let dir = ModelDir::new();
    for prefix_len in 0..8 {
        dir.write("model.safetensors", &100u64.to_le_bytes()[..prefix_len]);
        fails(&dir, "truncated header length");
    }
    dir.write("model.safetensors", &100u64.to_le_bytes());
    fails(&dir, "truncated header");
    for header in [
        "{",
        "[]",
        " {}",
        r#"{"__metadata__":{"x":1}}"#,
        r#"{"w":{}}"#,
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":1}}"#,
        r#"{"w":{"dtype":"F32","dtype":"I32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
    ] {
        dir.weights(header.as_bytes(), &[0; 4]);
        assert!(inspect(dir.path()).is_err(), "accepted {header}");
    }
    dir.weights(b"{\xff}", &[]);
    fails(&dir, "invalid JSON");
}

#[test]
fn invalid_shapes_dtypes_offsets_and_arithmetic() {
    let dir = ModelDir::new();
    for (shape, dtype, offsets, expected) in [
        (json!([-1]), "F32", json!([0, 4]), "tensor"),
        (json!([1.5]), "F32", json!([0, 4]), "tensor"),
        (json!(vec![1; 33]), "F32", json!([0, 4]), "rank exceeds"),
        (
            json!([u64::MAX, 2]),
            "F32",
            json!([0, 4]),
            "shape multiplication overflow",
        ),
        (
            json!([u64::MAX]),
            "F64",
            json!([0, 4]),
            "byte-size overflow",
        ),
        (
            json!([1]),
            "F4",
            json!([0, 4]),
            "unsupported safetensors dtype",
        ),
        (json!([1]), "F32", json!([4, 0]), "invalid byte range"),
        (json!([1]), "F32", json!([0, 5]), "invalid byte range"),
        (json!([2]), "F32", json!([0, 4]), "invalid byte range"),
        (
            json!([1]),
            "F32",
            json!([u64::MAX - 1, u64::MAX]),
            "invalid byte range",
        ),
        (json!([1]), "F32", json!([-1, 3]), "tensor"),
        (json!([1]), "F32", json!([0, 4, 4]), "tensor"),
    ] {
        let header = json!({"w":{"shape":shape,"dtype":dtype,"data_offsets":offsets}});
        dir.weights(header.to_string().as_bytes(), &[0; 4]);
        fails(&dir, expected);
    }
}

#[test]
fn rejects_overlaps_holes_and_trailing_bytes() {
    let dir = ModelDir::new();
    for (a, b, bytes) in [
        ([0, 4], [0, 4], 8),
        ([0, 4], [8, 12], 12),
        ([4, 8], [8, 12], 12),
        ([0, 4], [4, 8], 12),
    ] {
        let header = json!({"a":{"dtype":"F32","shape":[1],"data_offsets":a},"b":{"dtype":"F32","shape":[1],"data_offsets":b}});
        dir.weights(header.to_string().as_bytes(), &vec![0; bytes]);
        fails(&dir, "safetensors payload");
    }
    dir.weights(b"{}", &[0]);
    fails(&dir, "trailing bytes");
}

#[test]
fn scalar_empty_tensors_and_dtype_widths() {
    let dir = ModelDir::new();
    dir.weights(br#"{"empty":{"dtype":"F32","shape":[0,3],"data_offsets":[0,0]},"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#, &[0; 4]);
    assert_eq!(inspect(dir.path()).unwrap().total_tensor_payload_bytes, 4);
    dir.weights(b"{}", &[]);
    assert_eq!(inspect(dir.path()).unwrap().total_tensor_payload_bytes, 0);
    for (dtype, bytes) in [
        ("BOOL", 1),
        ("I8", 1),
        ("U8", 1),
        ("F8_E4M3", 1),
        ("F8_E5M2", 1),
        ("I16", 2),
        ("U16", 2),
        ("F16", 2),
        ("BF16", 2),
        ("I32", 4),
        ("U32", 4),
        ("F32", 4),
        ("I64", 8),
        ("U64", 8),
        ("F64", 8),
    ] {
        let header = json!({"w":{"dtype":dtype,"shape":[1],"data_offsets":[0,bytes]}});
        dir.weights(header.to_string().as_bytes(), &vec![0; bytes]);
        assert_eq!(
            inspect(dir.path()).unwrap().total_tensor_payload_bytes,
            bytes as u64
        );
    }
}

#[test]
fn rejects_all_partial_and_unsupported_layouts() {
    for name in [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model-00001-of-00002.safetensors",
        "other.safetensors",
        "pytorch_model.bin",
        "weights.gguf",
        "weights.pt",
        "weights.pth",
    ] {
        let dir = ModelDir::new();
        dir.write(name, b"{}");
        fails(&dir, "unsupported layout");
    }
    let dir = ModelDir::new();
    fs::create_dir(dir.path().join("nested")).unwrap();
    fails(&dir, "flat regular files");
}

#[cfg(unix)]
#[test]
fn rejects_symlinked_entries() {
    let dir = ModelDir::new();
    std::os::unix::fs::symlink("config.json", dir.path().join("link.json")).unwrap();
    fails(&dir, "flat regular files");
}

#[test]
fn sparse_large_payload_inspected_without_payload_read() {
    use std::io::Write;
    let dir = ModelDir::new();
    let bytes = 1024u64 * 1024 * 1024;
    let header = json!({"w":{"dtype":"U8","shape":[bytes],"data_offsets":[0,bytes]}}).to_string();
    let mut file = File::create(dir.path().join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.set_len(8 + header.len() as u64 + bytes).unwrap();
    let report = inspect(dir.path()).unwrap();
    assert_eq!(report.total_tensor_payload_bytes, bytes);
    // This sparse fixture supplies length, not a large buffer or real weights.
}
