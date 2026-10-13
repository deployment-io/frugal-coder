# Frugal Coder

Frugal Coder is an **experimental** project building a coding agent in Rust with
its own CPU-only inference backend. The vision is to make verified engineering
work practical on commodity hardware, including tasks where waiting longer for
a correct result is acceptable.

## Vision

We want an agent that can understand a repository, make a scoped change, run the
relevant checks, and return a result backed by verification evidence. Model
execution should happen locally on ordinary CPUs, with explicit resource limits
and a path to resuming longer-running work.

The intended product combines three parts:

- **An independent CPU inference backend.** We own model execution and the
  numerical implementation. No GPU/NPU execution paths, complete external
  inference engine, or hosted-model fallback.
- **A standalone coding agent and CLI.** The agent will use that backend to work
  on engineering tasks, report failures honestly, and distinguish changes made
  from results actually verified.
- **Future AgentBox integration.** Frugal Coder will become another agent
  executable for Deployment.io's existing Implement and Review stages.
  Deployment.io retains stage orchestration and acceptance policies. QA's
  integration contract is still developing.

Resource efficiency and useful coding capability must be demonstrated together.
We plan to measure task success, correctness, latency, memory, and relevant
energy/cost tradeoffs on documented hardware. No speed, cost, energy, or
coding-quality advantage has been demonstrated yet. Correctness comes before
performance optimization.

See the [architecture](docs/architecture.md) and
[roadmap with acceptance criteria](docs/roadmap.md) for the path from numerical
foundations to correct model inference, useful standalone agent tasks, and
integration.

## Current status: M0 foundation

**M0 performs local model metadata inspection and scalar numerical operations.
It does not perform model inference, generate text, execute tools, or complete
agent tasks.** Recognizing a model architecture does not mean it can execute it.

| Component | Code | Implemented today |
| --- | --- | --- |
| `frugal-backend` | [Numerics](crates/frugal-backend/src/tensor.rs), [inspection](crates/frugal-backend/src/inspect.rs) | Checked contiguous f32 tensors, scalar matmul, row RMSNorm, row softmax, SiLU, and local model metadata inspection. |
| `frugal-agent` | [Library boundary](crates/frugal-agent/src/lib.rs) | Module documentation only; no agent execution API yet. |
| `frugal-coder` | [CLI](crates/frugal-coder/src/main.rs) | `--help`, `--version`, and `inspect [--json] <model-directory>`. |

Qwen/Qwen3.5-0.8B's text path is the proposed first correctness candidate. It is
not a supported inference model or evidence of coding-agent ability. The
[first-model plan](docs/first-model.md) records verified source details and the
remaining validation work.

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
