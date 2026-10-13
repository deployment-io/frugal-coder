# Original synthetic fixtures and correctness evidence

All checked-in fixtures are original synthetic data under the repository's
Apache-2.0 license. They contain no real model weights, task data, or reference
implementation code. Tests never execute the generator or call a model library.

## Independent numerical reference

`fixtures/numerics.json` was produced with CPython 3.14.2, standard-library
`decimal`, 80 decimal digits, using `reference/generate.py` revision `decimal-v1`.
No numerical/model package is needed. Regenerate from the repository root:

```sh
python3 evals/reference/generate.py
```

Generator source SHA-256:
`59b9a7216b65eb410c8fe32e9929abdec0ebe0c48ff1d211eb4ebadbe1e94ffd`.
This content hash is the source revision because M0 was prepared without a Git
commit. Numerical fixture SHA-256:
`a0170258d9aa920c437653abd8978962389234b8a284d5747dd9c3a2186ba63b`.
If the generator changes, review/recompute provenance and expected values.

Inputs are rounded to IEEE f32 first with `struct.pack/unpack`; Decimal then
represents those binary inputs exactly. Matrix outputs are independent Decimal
dot products. RMS uses the direct per-row sum-of-squares/width formula, plus the
f32 epsilon, square root, and nontrivial per-column weights. Softmax uses direct
Decimal `exp(x)/sum(exp(x))` (not Rust's stabilized algorithm); Decimal's exponent
range supports the +/-1000 fixtures. SiLU uses `x/(1+exp(-x))`, independently of
Rust's sign-dependent formulation. Final reference values are stored as JSON
binary64-compatible numbers, **not rounded through Rust outputs**.

Cases include rectangular matrices, mixed-sign fractional values, cancellation,
row-dependent RMS with nonuniform/negative weights, zeros and large RMS inputs,
large positive/negative logits, fractional logits, and activations from -1000 to
1000. Rust tests supplement these with the exact analytic 2x3 @ 3x2 example,
equal-logit softmax, SiLU(0), RMS values derived from sqrt(5), shift invariance,
finite extremes, invalid shapes/parameters/storage, and allocation overflow.

Each comparison permits `1e-6 + 2e-6*abs(expected)` absolute error. This allows a
few f32 rounding errors for these small cases (including host exp/sqrt differences)
and a controlled absolute floor near zero. Exact analytic examples use exact
assertions. These tolerances are fixed, not fitted to failing tests; larger model
reductions need separately justified tolerances. See [numerics](../docs/numerics.md).

## Inspection fixture

`fixtures/synthetic-model` contains a generated regular `config.json` and a tiny
safetensors file: eight-byte prefix, 192-byte padded JSON header, and nine explicit
little-endian f32 values (36 payload bytes; 236 total). `demo.weight` is 2x3 and
`demo.scale` is length 3. The config deliberately declares Qwen identifiers for
recognition tests, but these are not real Qwen tensors and execution is false.

`fixtures/inspect-v1.json` is the complete expected report, independently assembled
by the Python generator from the authored metadata and byte sizes. It is not
captured from the Rust implementation. Synthetic weight-file SHA-256:
`9bcbb51ee00ad551e11ed89b32b31acba1bc8d09413e9c0e56230bada0a0fbdb`.

Adversarial fixtures are constructed in temporary directories by Rust tests:
malformed config/headers, duplicate keys, missing files, truncated prefixes,
invalid sizes/ranges, gaps/overlaps/trailing bytes, unknown dtypes, unsupported
layouts, and symlink entries. No malformed payload or model download is needed.
A sparse 1 GiB file exercises metadata-only inspection without allocating or
storing a weight buffer. Temporary directories are removed on test completion.

The CLI integration suite checks success/error exit codes, deterministic JSON
against the independent example, valid unknown architectures, and stderr/stdout
separation. All regular tests are ordinary Cargo tests; Python is development-only.
