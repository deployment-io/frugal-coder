# Roadmap and acceptance gates

Each milestone needs its own scoped implementation and evidence. Later items
below are proposed, not implemented.

## M0 — metadata and numerical foundation

Accept when the three-crate workspace builds, formatting/Clippy/tests pass, and
local synthetic CLI examples work. Invalid metadata/layouts must exit nonzero;
JSON must follow v1 with deterministic tensors and honest execution support.
Inspect only bounded metadata. Matmul, row RMSNorm, softmax, and SiLU require
analytic and independent fixture evidence, documented input domains, and checked
arithmetic. Document first-model verified facts and unresolved questions. See
[the M0 evidence log](milestones/m0-foundation.md).

## First correct model inference

Target Qwen/Qwen3.5-0.8B text-only after [the model plan](first-model.md). Accept only
when a pinned checkpoint can be loaded locally with complete tensor/index
validation and exact tokenizer/prompt comparisons; scalar intermediate outputs
and final logits agree with a pinned CPU reference under justified, predeclared
tolerances. Test prefill, incremental decode, cache resets, and serialized resume
against uninterrupted execution. Record memory/latency measurements without
claiming improvement. Reject unsupported modalities/configurations clearly.
No hosted fallback or external product inference engine.

## First useful standalone coding-agent task

Accept when the real local model completes a bounded, reproducible repository
change with tool permissions, failure reporting, and independent verification.
Record task success/failure and resource use on held-out synthetic/public tasks.
No fabricated tool results or success claims; stop cleanly on budgets or errors.
This requires a separate tool-execution design, not M0 placeholders.

## AgentBox Implement and Review integration

Accept when AgentBox can invoke the real executable for existing Implement and
Review stage contracts, including structured outcomes, verification evidence,
resume/cancellation, and explicit failure handling. Deployment.io alone owns
stage transitions and acceptance. Test end-to-end against the actual contract;
never add a parallel stage lifecycle inside the backend or agent.
QA integration remains future work whose contract is still developing.

## Profiling and optimization

After correctness gates, profile representative verified workloads on documented
commodity CPU hosts. Accept an optimization only with reproducible before/after
latency, peak memory, and relevant energy/cost measurements; capability and task
success must not regress. Quantization, SIMD, storage-aware execution, and
advanced scheduling require separate designs, error budgets, and regression
checks. No inferred advantage from architecture alone.
