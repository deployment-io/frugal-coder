# Frugal Coder

Frugal Coder is an **experimental** Rust coding-agent project for delay-tolerant
engineering work on commodity CPUs. Its intended inference backend is independent
and owned here. Resource efficiency, model capability, and verified task success
are goals to measure, not demonstrated advantages.

**M0 performs local model metadata inspection and scalar numerical operations.
It does not perform model inference, generate text, execute tools, or complete
agent tasks.** Recognizing a model architecture does not mean it can execute it.

## Try M0

From the repository root, with Rust/Cargo 1.84.0 or a compatible newer toolchain:

```sh
cargo build --workspace --locked
cargo run --locked -p frugal-coder -- --help
cargo run --locked -p frugal-coder -- --version
cargo run --locked -p frugal-coder -- inspect evals/fixtures/synthetic-model
cargo run --locked -p frugal-coder -- inspect --json evals/fixtures/synthetic-model
```

The checked-in synthetic fixture is 236 weight-file bytes, including 36 tensor
payload bytes. It declares Qwen identifiers solely to demonstrate recognition;
it is not a Qwen checkpoint and cannot run. No model download is needed.

`inspect` reads bounded configuration and safetensors headers, checks tensor byte
ranges against file lengths, and reports names, shapes, dtypes, and byte counts.
It never reads weight payloads, executes repository code, or downloads files.
File sizes are **not runtime RAM estimates**. Metadata validation is **not payload
integrity verification**. Only a flat directory containing regular `config.json`
and `model.safetensors` files is supported. Indexed/sharded layouts, including
the official first-model checkpoint layout, are explicitly unsupported in M0.
See [the inspection contract](docs/inspection.md) and
[JSON schema v1](docs/inspect-v1.schema.json).

## Implemented and planned

- `frugal-backend`: checked contiguous f32 storage, scalar matmul, row RMSNorm,
  row softmax, SiLU, and local metadata inspection.
- `frugal-agent`: intentionally minimal library boundary; no execution API yet.
- `frugal-coder`: `--help`, `--version`, and `inspect [--json] <model-directory>`.

The product is CPU-only: no GPU/NPU paths, complete external inference engines,
or hosted-model fallback. Qwen/Qwen3.5-0.8B's text path is a **proposed correctness
candidate**, not a supported inference model or evidence of coding-agent ability.
See [the verified first-model plan](docs/first-model.md),
[architecture](docs/architecture.md), and [acceptance-based roadmap](docs/roadmap.md).

## Develop and validate

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --workspace --locked
```

`Cargo.lock` is checked in for reproducibility. Runtime dependencies are only
`serde` (typed serialization/deserialization) and `serde_json` (JSON); the tiny CLI
uses the standard library. No external numerical library is in the application.
After dependencies are cached, tests also work with `--offline` and require no
Python, credentials, network, or model downloads.

Local M0 validation used rustc 1.84.0 (`9fc6b4312`, LLVM 19.1.5), Cargo 1.84.0
(`66221abde`), on `aarch64-apple-darwin`. CI installs Rust 1.84.0 and runs the four
checks on Ubuntu with the lockfile. A configured workflow is not evidence of a
hosted CI run; actual outcomes are recorded in [the M0 log](docs/milestones/m0-foundation.md).

Read [the numerical contract](docs/numerics.md) and
[fixture provenance](evals/README.md) before changing numerics or tolerances.
Apache-2.0; see [LICENSE](LICENSE).
