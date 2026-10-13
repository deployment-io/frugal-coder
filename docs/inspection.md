# Model inspection and JSON v1

## Supported input

A local **flat** directory containing regular, non-symlink `config.json` and
`model.safetensors`. Other regular non-weight assets are ignored. M0 rejects
subdirectories, symlink entries, non-UTF-8 filenames, any `*.index.json`, any other
`*.safetensors`, and `*.bin`, `*.gguf`, `*.pt`, `*.pth` weight formats. A lone
shard-shaped safetensors filename is rejected, including `00001-of-00001`.
Inspection never follows index entries or silently reports one shard as a model.
These intentional restrictions also exclude common symlink-based cache layouts.

`config.json` must be a JSON object with a nonempty string `model_type` and a
nonempty array of nonempty strings `architectures`. Optional `text_config` must
be an object with a nonempty string `model_type`. Other fields are ignored.
Duplicate keys are rejected recursively. No auto-map, custom code, or tokenizer
is executed. Model files must remain stable while inspection runs; M0 does not
provide filesystem snapshot isolation against concurrent modification.

Recognition requires precisely `model_type = qwen3_5`, the single architecture
`Qwen3_5ForConditionalGeneration`, and `text_config.model_type = qwen3_5_text`.
All other well-formed declarations are unrecognized and may still be inspected.
Recognition validates **identifiers only**, not complete model configuration,
expected tensor names/shapes, or architecture compatibility.

## Limits and validation

- Configuration: at most 1 MiB (1,048,576 bytes), including whitespace.
- Safetensors header: 1 byte through 16 MiB (16,777,216 bytes), excluding its
  eight-byte little-endian length prefix. First byte must be `{`.
- At most 100,000 tensors, at most rank 32 per metadata shape, and at most 4,096
  directory entries. serde_json's default recursion limit (128) is retained.
- Header entries have exactly `dtype`, `shape`, `data_offsets` fields. The optional
  `__metadata__` field must map strings to strings. Unknown fields/duplicate keys,
  negative/noninteger sizes, and unknown dtypes fail.
- Supported byte widths: 1 for BOOL/I8/U8/F8_E4M3/F8_E5M2; 2 for I16/U16/F16/BF16;
  4 for I32/U32/F32; 8 for I64/U64/F64. Dtype recognition does not decode payloads.
- Shapes use checked u64 products; byte counts use checked u64 multiplication.
  Scalar shape `[]` means one element; zero dimensions mean an empty tensor.
  Even empty shapes' nonzero dimension products must fit u64. This conservative
  metadata policy differs from numerical tensors, which reject empty dimensions.
- Offsets are `[start,end)` relative to the payload after prefix/header. Each
  range must match its shape/dtype bytes and fit the actual file length. Sorted
  ranges must exactly cover the payload with no holes, overlaps, or trailing
  unreferenced bytes. Empty tensors occupy zero-length ranges at boundaries.

Only the prefix/header is read from the weight file. File length can be large
without corresponding payload allocation; tests use a 1 GiB sparse file. The
parser's decoded metadata needs additional RAM beyond raw header bytes. The
limits bound input sizes, not an exact peak parser-memory budget.

## Output and failure contract

`inspect --json <directory>` emits one UTF-8 JSON document plus a newline on
stdout, with no diagnostics on successful inspection. See the machine-readable
[schema](inspect-v1.schema.json) and [synthetic example](../evals/fixtures/inspect-v1.json).
Object field order is stable in this implementation; consumers must use keys.
Tensors are sorted lexicographically by name. Config architecture order is
preserved. No absolute input path, timestamps, or environment data is emitted.

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1`; incompatible contract changes require a new version. |
| `architecture` | Declared `model_type`, `architectures`, nullable `text_model_type`, `recognized` boolean, and `execution_implemented` (always false in M0). |
| `layout` | `single_safetensors`. |
| `files` | One entry: relative filename, total `file_bytes`, `header_bytes` excluding prefix, `payload_bytes`, and ordered `tensors`. |
| `files[].tensors[]` | Name, shape (u64 dimensions), dtype, payload-relative `data_offsets`, and `payload_bytes`. |
| `total_tensor_payload_bytes` | Sum of tensor byte ranges, excluding headers/prefix. |
| `total_weight_file_bytes` | Actual inspected weight-file length including prefix/header; excludes config and other assets. |
| `limitations` | Human-readable caveats; wording may evolve within v1, so do not parse these strings. |

u64 byte counts and dimensions are JSON integers. Consumers whose default number
type is IEEE double must use a parser preserving integers above 2^53 exactly.
The schema specifies types/bounds; cross-field arithmetic follows the rules above.

Usage errors exit 2; invalid input/I/O errors exit 1; success exits 0. Input errors
emit diagnostics only to stderr, with empty stdout. Output-device failure can
leave a partial write and exits nonzero. Human output escapes untrusted names.
No JSON error-object contract is promised in v1.

Byte-layout validation cannot establish tensor-value integrity, checkpoint
completeness, numerical validity of weights, runtime RAM needs, or inference
support. No payload hashes are computed. There is no implicit download, network
access, hosted fallback, or model execution.
