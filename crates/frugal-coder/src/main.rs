use frugal_backend::inspect::{inspect, Inspection};
use std::{
    env,
    ffi::OsString,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

const HELP: &str = "Frugal Coder — experimental CPU-only foundation\n\nUsage:\n  frugal-coder --help\n  frugal-coder --version\n  frugal-coder inspect [--json] <model-directory>\n\nInspect local config.json and model.safetensors metadata without reading weight\npayloads. --json emits schema version 1. No model inference or agent execution.\n";

enum Command {
    Help,
    Version,
    Inspect { directory: PathBuf, json: bool },
}
fn parse(args: Vec<OsString>) -> Result<Command, String> {
    match args.as_slice() {
        [arg] if arg == "--help" || arg == "-h" => return Ok(Command::Help),
        [arg] if arg == "--version" || arg == "-V" => return Ok(Command::Version),
        [cmd, arg] if cmd == "inspect" && (arg == "--help" || arg == "-h") => {
            return Ok(Command::Help)
        }
        _ => {}
    }
    if args.first().is_none_or(|a| a != "inspect") {
        return Err("expected inspect <model-directory>, --help, or --version".into());
    }
    let mut json = false;
    let mut directory = None;
    let mut options = true;
    for arg in args.into_iter().skip(1) {
        if options && arg == "--" {
            options = false;
        } else if options && arg == "--json" && !json {
            json = true;
        } else if options && arg.to_string_lossy().starts_with('-') {
            return Err(format!(
                "unknown or repeated option: {}",
                arg.to_string_lossy()
            ));
        } else if directory.replace(PathBuf::from(arg)).is_some() {
            return Err("inspect accepts exactly one model directory".into());
        }
    }
    Ok(Command::Inspect {
        directory: directory.ok_or("inspect requires a model directory")?,
        json,
    })
}

fn human(report: &Inspection) -> String {
    use std::fmt::Write;
    let mut text = String::new();
    let a = &report.architecture;
    // Debug-format untrusted strings to escape control characters in terminal output.
    let _ = writeln!(
        text,
        "Architecture: {:?} ({:?})\nRecognized: {}\nExecution implemented: {}\nLayout: {}",
        a.architectures, a.model_type, a.recognized, a.execution_implemented, report.layout
    );
    let _ = writeln!(
        text,
        "Tensor payload bytes: {}\nWeight-file bytes: {}",
        report.total_tensor_payload_bytes, report.total_weight_file_bytes
    );
    for file in &report.files {
        let _ = writeln!(
            text,
            "File: {} (header: {} bytes)",
            file.name, file.header_bytes
        );
        for tensor in &file.tensors {
            let _ = writeln!(
                text,
                "  {:?}: {:?} {:?}, {} payload bytes, offsets {:?}",
                tensor.name, tensor.shape, tensor.dtype, tensor.payload_bytes, tensor.data_offsets
            );
        }
    }
    for limitation in &report.limitations {
        let _ = writeln!(text, "Limitation: {limitation}");
    }
    text
}

fn run() -> Result<(), (u8, String)> {
    let command = parse(env::args_os().skip(1).collect()).map_err(|e| (2, e))?;
    let output = match command {
        Command::Help => HELP.to_owned(),
        Command::Version => format!("frugal-coder {}\n", env!("CARGO_PKG_VERSION")),
        Command::Inspect { directory, json } => {
            let report = inspect(&directory).map_err(|e| (1, e.to_string()))?;
            if json {
                format!(
                    "{}\n",
                    serde_json::to_string_pretty(&report).map_err(|e| (1, e.to_string()))?
                )
            } else {
                human(&report)
            }
        }
    };
    io::stdout()
        .lock()
        .write_all(output.as_bytes())
        .map_err(|e| (1, format!("stdout: {e}")))
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err((code, error)) => {
            eprintln!("error: {error}");
            ExitCode::from(code)
        }
    }
}
