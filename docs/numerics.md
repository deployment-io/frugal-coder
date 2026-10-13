# Scalar numerical contract

`Tensor::new(shape, data)` accepts only contiguous row-major f32 rank 1 or 2.
Dimensions must be nonzero; rank 0, rank >2, empty storage, and empty dimensions
are rejected. Element count must exactly match storage. Shape products, byte
products, and Rust's `isize::MAX` allocation bound are checked before allocation.
Output allocation uses `try_reserve_exact`; operating-system resource exhaustion
can still terminate a process after virtual memory reservation succeeds.

All data and RMS weights must be finite. NaNs and either infinity are rejected
at construction/parameter validation, never silently propagated. Immutable
storage/shape accessors maintain those invariants. Operations return errors for
unrepresentable nonfinite f32 outputs. Subnormal rounding and underflow to zero
are permitted; signed zero is not semantically distinguished. Scalar exp/sqrt
rounding can vary slightly across standard-library/host implementations.

| Operation | Shapes and axis | Precision and behavior |
| --- | --- | --- |
| `matmul` | `[M,K] @ [K,N] -> [M,N]`; rank 2 only | Multiplication and dot-product accumulation in f64; one f32 rounding per output. Incompatible inner dimensions fail. |
| `rms_norm` | Rank 1 = one row; rank 2 = independent rows; last dimension normalized | f64 sum of squares, mean, sqrt, scaling; f32 output. Weight slice must match row width. Epsilon is f32, finite, strictly positive. Formula `x / sqrt(mean(x*x)+epsilon) * weight`; no mean subtraction. |
| `softmax` | Rank 1 or 2, independently over last dimension | Row max subtraction, exponentials, and sum in f64; f32 output. Even subtraction of opposite f32 extremes cannot overflow f64. Output sums approximate one. No all-masked/-infinity convention is provided. |
| `silu` | Elementwise; preserves rank/shape | f64 sign-dependent sigmoid avoids exponent overflow: for negative x use `exp(x)/(1+exp(x))`; multiply by x and round to f32. |

f64 intermediates deliberately favor a transparent correctness baseline; this is
not a claim of equivalence to a model reference's f32/BF16 reduction order. There
is no compensated summation, autograd, broadcasting, arbitrary strides, compute
graph, SIMD, quantization, or automatic differentiation. Very large reductions
can still accumulate rounding error. M0 fixtures exercise small shapes only.

Fixture comparisons use `abs(actual-reference) <= 1e-6 + 2e-6*abs(reference)`.
Analytically exact integer products, equal-logit softmax, and SiLU(0) use exact
checks. The combined tolerance allows a small number of f32 rounding errors near
unit scale and accommodates near-zero values without dividing by zero. It is
not a promised tolerance for arbitrarily large matrices or full-model logits.
See [fixture provenance](../evals/README.md). Future model normalization variants
must be distinct from this generic direct-weight RMSNorm.
