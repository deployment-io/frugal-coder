# Project guidance

- Build an independent CPU-only backend. No GPU/NPU paths, complete external
  inference engine, hosted inference, or hidden external model delegation.
- Keep safe Rust and readable scalar correctness baselines. Do not add unsafe
  code, SIMD, quantization, or scheduling without a separately scoped milestone.
- Workspace boundaries: `frugal-backend` owns numerics/model execution;
  `frugal-agent` will own agent behavior; `frugal-coder` owns CLI presentation.
  M0 has inspection and numerics only; the agent has no execution API.
- Deployment.io owns stages, transitions, and acceptance policies. Future
  AgentBox integration targets Implement and Review first. QA is under
  development; do not invent its contract or duplicate orchestration here.
- Validate with `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  `cargo test --workspace --locked`, and `cargo build --workspace --locked`.
  Keep Cargo.lock. Tests must run offline after dependency installation.
- Use analytic cases and independent reference fixtures; record provenance and
  tolerances. Do not loosen tolerances just to pass. Report capabilities and
  measured results truthfully; recognition is never execution support.
- Inspect Git status and applicable guidance before edits. Preserve user work,
  existing licenses, and task branches. Keep changes in this repository and
  public paths repository-relative. Never commit credentials, real weights,
  private data, or build outputs. Do not commit, push, publish, change remotes,
  or edit neighboring projects without an explicit request.
