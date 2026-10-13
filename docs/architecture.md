# Architecture

## Existing M0 components

`frugal-coder` calls `frugal-backend::inspect::inspect` and formats its complete
report as human text or schema-v1 JSON. Argument/usage errors exit 2; inspection
or output errors exit 1; success exits 0. Diagnostics use stderr. Inspection
failures never emit a partial success report.

`frugal-backend::inspect` reads only a bounded config, an eight-byte header-length
prefix, a bounded safetensors header, and filesystem metadata. It checks declared
shape/dtype byte sizes and complete payload-range coverage against file length.
Architecture recognition compares declared identifiers, not weight semantics.
Execution support is always false. The implementation does not link an inference
engine, access networks, or load repository-provided code.

`frugal-backend::tensor` owns immutable, checked, contiguous rank-1/rank-2 f32
storage. Operations are portable scalar Rust with f64 intermediates. The public
constructor prevents invalid storage and nonfinite values from reaching kernels.
See [numerics](numerics.md) for axes, empty-input policy, and output precision.

`frugal-agent` contains only module documentation defining the future ownership
boundary. It has no task runner, trait framework, tools, shell execution, success
responses, or dependency on a hosted model. There is deliberately no CLI command
for generation or agent execution.

## Proposed evolution, not implemented

The backend will load locally supplied tensors, tokenize text through audited
utility code, and execute the first verified CPU model. It will have no knowledge
of Deployment.io task stages. The agent will eventually combine local generation
with bounded repository tools and explicit verification evidence. The CLI will
expose only capabilities that actually work.

A future AgentBox adapter can launch the executable for Deployment.io's existing
Implement and Review stages. Deployment.io retains stage transitions and
acceptance policy. QA integration remains future work with a developing contract.
M0 neither changes AgentBox nor defines another task lifecycle.

Backend optimizations come after reference comparisons establish correctness.
No speed, cost, energy, RAM, or coding-quality advantage has been measured here.
