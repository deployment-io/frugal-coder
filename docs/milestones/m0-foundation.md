# M0 foundation

## Plan recorded before implementation — 2026-10-12

- [x] Establish a small safe-Rust workspace and preserve repository/license state.
- [x] Implement checked scalar f32 tensors, matmul, row RMSNorm, row softmax, SiLU.
- [x] Implement bounded local config/safetensors inspection and a versioned JSON CLI.
- [x] Add independent numerical fixtures, adversarial metadata tests, and CLI tests.
- [x] Verify official first-model references; document a proposed text-only path.
- [x] Add architecture, roadmap, project guidance, license, and locked CI checks.
- [x] Run formatting, Clippy, tests, build, CLI examples, and final scope review.

Initial inspection: empty checkout, unborn `main`, no tracked files, user changes,
or license. Created `codex/m0-foundation`. Installed Rust/Cargo 1.84.0 with
rustfmt and Clippy; host `aarch64-apple-darwin`. No toolchain installation needed.
The parent project guidance and existing planning context were consulted;
Deployment.io retains orchestration ownership. No neighboring projects are edited.

## Initial decisions

Use edition 2021 and Rust 1.84.0. Keep only serde/serde_json as runtime utility
dependencies; parse the very small CLI with the standard library. The agent crate
will be a documented library boundary with no execution API. Reject indexed,
sharded, and multiple-weight layouts in M0. Accept only `config.json` plus
`model.safetensors`, with explicit byte limits, strict range/size validation, and
no payload reads. Inspection is never an execution-support claim.

Use f64 accumulators/intermediates for a portable scalar f32 correctness baseline;
reject zero dimensions and nonfinite inputs and reject unrepresentable outputs.
Independent fixtures will be generated with Python standard-library Decimal,
checked in, and consumed without Python or network access by ordinary Rust tests.

Validation evidence and final decisions are recorded below.

## Completion evidence — 2026-10-12

M0 acceptance criteria are satisfied locally. This does **not** complete a model
inference engine or coding agent. Changes are uncommitted on
`codex/m0-foundation`; nothing was pushed, published, or changed in remote settings.

Implemented three workspace crates, bounded metadata inspection, schema-v1
human/JSON CLI, checked scalar numerical operations, independent fixtures,
adversarial tests, documentation, Apache-2.0 license, and a locked CI workflow.
`frugal-agent` intentionally has no executable behavior or speculative API.

Final decisions:

- Numerical tensors support only rank 1/2 with positive dimensions and finite
  f32 storage. All four operations use f64 intermediates and return useful errors.
- Metadata limits are 1 MiB config, 16 MiB header, 100000 tensors, rank 32, 4096
  flat-directory entries. Metadata scalar/empty shapes are supported explicitly.
- Shards/indexes, extra weight files, nested directories, and symlink entries are
  rejected; no weight payload is read. Model/config semantic completeness and
  payload integrity are not asserted. Files must be stable during inspection.
- JSON schema version 1, sorted tensor names, relative filenames, explicit
  non-execution, exact u64 byte counts, and separate stdout/stderr.
- Only serde/serde_json runtime dependencies. Cargo.lock was generated from
  available cached crates and is included. No new Rust installation was needed;
  CI pins 1.84.0 without forcing a duplicate local toolchain installation.
- Independent reference generator uses CPython 3.14.2 Decimal at 80 digits; source
  and fixture hashes, formulas, and fixed tolerances are in `evals/README.md`.
- Official Qwen sources and tokenizer pipeline were accessible and pinned. The
  detailed model plan records architecture/state requirements, source-derived
  tensor shapes, the EOS metadata discrepancy, and unverified runtime evidence.
  No full model weights or third-party implementation code were added.

## Commands run locally and actual results

Host: `aarch64-apple-darwin`; rustc 1.84.0 (`9fc6b43126469e3858e2fe86cafb4f0fd5068869`,
2025-01-07), LLVM 19.1.5; Cargo 1.84.0 (`66221abde`, 2024-11-19).

| Command from repository root | Actual outcome |
| --- | --- |
| `cargo generate-lockfile --offline` | Passed; locked ten external utility packages from existing cache. |
| `cargo check --workspace --locked --offline` | Passed. |
| `cargo fmt --all -- --check` | Passed. |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | Passed; zero warnings. |
| `cargo test --workspace --locked` | Passed: 10 inspection, 7 numerical, 4 CLI integration tests; 0 failures. |
| `cargo build --workspace --locked` | Passed. |
| `cargo test --workspace --locked --offline` | Passed: same 21 tests without network or Python. |
| `target/debug/frugal-coder --help` | Exit 0; help stdout, empty stderr. |
| `target/debug/frugal-coder --version` | Exit 0; `frugal-coder 0.1.0`, empty stderr. |
| `target/debug/frugal-coder inspect evals/fixtures/synthetic-model` | Exit 0; two tensors, 36 payload bytes, 236 file bytes, recognized true, execution false. |
| `target/debug/frugal-coder inspect --json evals/fixtures/synthetic-model` | Exit 0; valid deterministic schema-v1 JSON, empty stderr; matches independently generated contract fixture. |
| `python3 evals/reference/generate.py` | Passed; produced checked-in numerical, metadata, and expected-report fixtures independently of Rust. |
| `cargo tree --workspace --locked` | Audited; only serde/serde_json and their utility/proc-macro dependencies. |

Additional explicit CLI exercises constructed local temporary fixtures and invoked
`target/debug/frugal-coder inspect --json <temporary-fixture>`: an eight-byte
prefix declaring a missing 100-byte header exited **1**, empty stdout, stderr
`model.safetensors: truncated header`; an added index file exited **1**, empty
stdout, explicit unsupported-layout diagnostic. CLI tests repeat these outcomes.
A 1 GiB sparse-file test completed using metadata only. Numerical tests also
cover finite extremes, rank/storage errors, arithmetic/allocation bounds,
nonfinite inputs, invalid epsilon, row axes, and constant-shift invariance.

All repository JSON parsed successfully. Schema field types/bounds and the
independent expected report were reviewed together. A separate Python
`jsonschema` validator was unavailable (module not installed); no claim is made
that an external JSON Schema validator ran. The normal tests require only Rust.

## CI configured, not run here

`.github/workflows/ci.yml` installs Rust 1.84.0 on Ubuntu and runs formatting,
Clippy, tests, and workspace build with locked dependency resolution where
applicable. It requires no project secrets or model files. **No hosted CI run has
occurred as part of this task.** Linux/Windows runtime behavior has not been
validated locally; the actual local host is the Mac architecture recorded above.

## Scope and hygiene review

Reviewed new-file diffs (the repository has no baseline commit), source, tests,
lockfile dependency tree, docs, and file sizes. No inference-engine, accelerator,
network-client, agent shell/tool execution, or fabricated generation paths.
All numerical expected outputs come from analytic cases or independent Decimal
formulas. Runtime command spawning exists only in CLI integration tests.
No credentials, machine-specific checkout paths, large weights, generated build
artifacts, or private task data are included in reviewable files. The only
safetensors fixture is 236 bytes; target output is ignored. No neighboring
repository files were modified.

## Remaining limitations and next step

No M0 blocker remains. Indexed/sharded layout support, tensor decoding, tokenizer
execution, all model execution, agent tools, and AgentBox integration are absent.
Metadata validation does not establish payload integrity or runtime RAM needs.
Real-model header dtypes/shapes, resolved EOS behavior, a runnable pinned reference
environment, tokenizer/prompt parity, cache/resume parity, and logits remain
unverified; `docs/first-model.md` distinguishes source facts from those items.

The smallest next implementation should add independently generated fixtures and
a separate Qwen zero-centered RMSNorm operation (including per-head application),
then work through the other model-specific primitives before full inference.
