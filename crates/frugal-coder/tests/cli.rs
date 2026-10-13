use serde_json::Value;
use std::{
    path::PathBuf,
    process::{Command, Output},
};
#[path = "../../../evals/test-support/mod.rs"]
mod support;
use support::ModelDir;

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_frugal-coder"))
        .args(args)
        .output()
        .unwrap()
}
fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../evals/fixtures/synthetic-model")
}

#[test]
fn help_version_and_unknown_commands() {
    for args in [
        &["--help"][..],
        &["-h"],
        &["inspect", "--help"],
        &["--version"],
        &["-V"],
    ] {
        let out = run(args);
        assert!(out.status.success());
        assert!(out.stderr.is_empty());
        assert!(!out.stdout.is_empty());
    }
    assert_eq!(
        String::from_utf8(run(&["--version"]).stdout).unwrap(),
        "frugal-coder 0.1.0\n"
    );
    for args in [
        &[][..],
        &["generate"],
        &["agent"],
        &["inspect"],
        &["inspect", "--json"],
        &["inspect", "a", "b"],
        &["inspect", "--unknown"],
        &["inspect", "--json", "--json", "a"],
    ] {
        let out = run(args);
        assert_eq!(out.status.code(), Some(2));
        assert!(out.stdout.is_empty());
        assert!(String::from_utf8_lossy(&out.stderr).starts_with("error:"));
    }
}

#[test]
fn human_and_deterministic_versioned_json_output() {
    let path = fixture();
    let path = path.to_str().unwrap();
    let out = run(&["inspect", path]);
    assert!(out.status.success());
    assert!(out.stderr.is_empty());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("Execution implemented: false"));
    assert!(text.contains("36"));
    let out = run(&["inspect", path, "--json"]);
    let repeated = run(&["inspect", "--json", path]);
    assert!(out.status.success());
    assert!(out.stderr.is_empty());
    assert_eq!(out.stdout, repeated.stdout);
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    let expected: Value =
        serde_json::from_str(include_str!("../../../evals/fixtures/inspect-v1.json")).unwrap();
    assert_eq!(report, expected);
}

#[test]
fn invalid_input_has_nonzero_exit_and_no_success_json() {
    let dir = ModelDir::new();
    for json in [false, true] {
        for bad in [b"{".as_slice(), b"null"] {
            dir.write("config.json", bad);
            let mut args = vec!["inspect", dir.path().to_str().unwrap()];
            if json {
                args.push("--json");
            }
            let out = run(&args);
            assert_eq!(out.status.code(), Some(1));
            assert!(out.stdout.is_empty());
            assert!(String::from_utf8_lossy(&out.stderr).contains("config.json"));
        }
    }
    let dir = ModelDir::new();
    dir.write("model.safetensors", &64u64.to_le_bytes());
    let out = run(&["inspect", dir.path().to_str().unwrap(), "--json"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("truncated header"));
}

#[test]
fn unsupported_layout_and_unknown_architecture() {
    let dir = ModelDir::new();
    let out = run(&["inspect", "--json", "--", dir.path().to_str().unwrap()]);
    assert!(out.status.success());
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["architecture"]["recognized"], false);
    assert_eq!(report["architecture"]["execution_implemented"], false);
    dir.write(
        "model.safetensors.index.json",
        br#"{"weight_map":{"w":"../escape.safetensors"}}"#,
    );
    let out = run(&["inspect", dir.path().to_str().unwrap(), "--json"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("unsupported layout"));
}
