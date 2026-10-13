# Proposed first correct model: Qwen/Qwen3.5-0.8B text path

This is a future implementation plan, **not inference support**. M0 implements no
model, tokenizer, cache, generation, or agent task. Model capability for useful
coding work is unmeasured. Initial scope should be one unpadded text sequence,
CPU scalar f32 computation, deterministic greedy decoding, and explicit rejection
of multimodal input. Batching and packed sequences require separate evidence.

## Verified sources and provenance (2026-10-12)

Official checkpoint revision: `2fc06364715b967f1860aea9cf38778875588b17`.
Read the official [configuration][config], [tensor index][index],
[tokenizer configuration][tokenizer-config], [tokenizer pipeline][tokenizer],
[chat template][template], and [model card][card] at that revision. Only metadata,
tokenizer assets, and source were retrieved; **no weight payload was downloaded**.

Authoritative Transformers reference inspected at immutable commit
`812a950d7037ba93bb624b154030e9068afdce4d`: [modeling][modeling],
[configuration classes][configuration], [tokenization class][tokenization],
and [cache implementation][cache]. These are source inspections, not a verified
runnable dependency environment. The checkpoint's `transformers_version` field
is `4.57.0.dev0`; that is not an adequate reference-environment pin. A future
comparison must record the installed Transformers revision, PyTorch version,
tokenizers version, dtype, CPU host, and selected implementation paths.

Qwen's checkpoint and the inspected Transformers source declare Apache-2.0.
No upstream code, tokenizer assets, or real weights are vendored in this repo.
Descriptions below summarize the reference; the M0 fixtures are original
synthetic data, independently generated under this repository's license.

## Verified configuration

The top-level architecture is `Qwen3_5ForConditionalGeneration`, model type
`qwen3_5`, with separate text and vision configurations. The initial target uses
only `text_config` (`qwen3_5_text`). This is not a generic Qwen3 transformer.
The explicit checkpoint values below override reference-class defaults. [Source][config]

| Text field | Checkpoint value |
| --- | --- |
| `hidden_size`, `intermediate_size`, `vocab_size` | 1024, 3584, 248320 |
| `num_hidden_layers`, `layer_types` | 24; repeating three `linear_attention` then one `full_attention` |
| `full_attention_interval` | 4; full layers at zero-based 3, 7, 11, 15, 19, 23 |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | 8, 2, 256 (do not infer head size as 1024/8) |
| `attention_bias`, `attention_dropout`, `attn_output_gate` | false, 0.0, true |
| `linear_num_key_heads`, `linear_num_value_heads` | 16, 16 |
| `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim` | 128, 128, 4 |
| `hidden_act`, `rms_norm_eps` | silu, 1e-6 |
| `dtype`, `mamba_ssm_dtype` | bfloat16, float32 (declarations, not a tensor-header audit) |
| `tie_word_embeddings`, `use_cache` | true, true |
| `max_position_embeddings` | 262144; not a tested context length here |
| `rope_parameters` | default type, theta 10000000, partial factor 0.25, interleaved, sections [11,11,10] |
| `mtp_num_hidden_layers`, `mtp_use_dedicated_embeddings` | 1, false; MTP excluded from initial ordinary autoregressive path |
| `eos_token_id` | 248044; see the tokenizer discrepancy below |

## Tensor mapping and layout work required

The official index lists 488 tensor names in one indexed file named
`model.safetensors-00001-of-00001.safetensors`, with declared total tensor bytes
1,746,882,752. This is the index's claim, **not independently verified file/payload
size or runtime RAM**. M0 explicitly rejects this indexed layout, even though it
references just one file. A future loader must validate every index mapping,
referenced path containment, all referenced files, and tensor/index consistency.
Do not bypass validation by silently renaming a shard. [Source][index]

The following names are present in the index. Expected shapes are **derived from
configuration and reference constructors**, not observed weight headers. Validate
each actual dtype/shape before loading. Linear weights are `[out,in]`; M0 matmul
needs an explicit transpose/copy (future code), not a different multiplication
convention. [Index][index], [constructors][modeling]

| Checkpoint name (prefix `model.language_model.`) | Expected shape / use |
| --- | --- |
| `embed_tokens.weight` | [248320,1024]; embedding gather and tied output logits |
| `norm.weight` | [1024]; final zero-centered RMSNorm |
| `layers.i.input_layernorm.weight`, `layers.i.post_attention_layernorm.weight` | [1024]; pre-mixer and pre-MLP zero-centered RMSNorm |
| `layers.i.mlp.gate_proj.weight`, `up_proj.weight`, `down_proj.weight` | [3584,1024], [3584,1024], [1024,3584] |
| Full layer `self_attn.q_proj.weight` | [4096,1024]; reshape into 8 heads of 512, then split each into query and gate of 256 |
| Full layer `self_attn.k_proj.weight`, `v_proj.weight`, `o_proj.weight` | [512,1024], [512,1024], [1024,2048] |
| Full layer `self_attn.q_norm.weight`, `k_norm.weight` | [256]; normalize independently per head |
| Linear layer `linear_attn.in_proj_qkv.weight` | [6144,1024]; split into q/k/v blocks of 2048 after convolution |
| Linear layer `linear_attn.in_proj_z.weight`, `out_proj.weight` | [2048,1024], [1024,2048] |
| Linear layer `linear_attn.in_proj_a.weight`, `in_proj_b.weight` | [16,1024] each |
| Linear layer `linear_attn.conv1d.weight` | [6144,1,4]; depthwise causal convolution |
| Linear layer `linear_attn.A_log`, `dt_bias` | [16] each |
| Linear layer `linear_attn.norm.weight` | [128]; per-value-head direct-weight gated RMSNorm |

The index also contains `model.visual.*` and `mtp.*`. Audit and deliberately
exclude those from the text-only execution path; they remain real checkpoint
content. `lm_head.weight` is absent from this index; the reference conditional
class explicitly ties it to `model.language_model.embed_tokens.weight`. The
standalone reference `Qwen3_5ForCausalLM` uses a different prefix
(`model.embed_tokens.weight`). Prefix conversion and tie handling need explicit
loader tests, not heuristic suffix matching. [Index][index], [reference classes][modeling]

## Operations and stateful mechanisms

Each decoder block applies zero-centered RMSNorm, its configured mixer, a residual
add, another zero-centered RMSNorm, the SiLU-gated MLP, then another residual add.
The MLP is `down(SiLU(gate(x)) * up(x))`. Add gather, transpose, reshape/split,
concatenation, explicit elementwise operations, and weight conversion to the M0
primitives; there is no need for a general graph engine. [Reference][modeling]

Full attention needs per-head query/key normalization, partial RoPE, causal
masking, grouped-query attention (repeat each of two KV heads across four query
heads), scale `1/sqrt(256)`, stable softmax, and value mixing. Multiply the joined
attention result by sigmoid of the query-projection gate **before** `o_proj`.
M0 softmax rejects infinities, so a later explicit masked-softmax operation must
specify masked/all-masked behavior rather than weakening M0's finite-input
contract. Cache rotated keys and values for each full layer. [Reference][modeling]

Linear attention is Gated DeltaNet, not softmax attention. Project q/k/v, apply a
causal depthwise width-4 convolution followed by SiLU, reshape into heads, and L2
normalize q/k with sum-of-squares plus epsilon (not mean-of-squares RMSNorm).
Scale queries by `1/sqrt(128)`. The reference uses `beta=sigmoid(b)` and
`g=-exp(A_log)*softplus(a+dt_bias)`. Per token, decay state by `exp(g)`, compute
memory from key and decayed state, add the key outer-product with
`beta*(value-memory)`, then read out with the query. The result passes through
per-head direct-weight RMSNorm, multiplication by SiLU(z), and `out_proj`.
A straightforward recurrent scalar implementation is the proposed first oracle
path; compare it to the reference chunked prefill and recurrent decode paths.
[Recurrent rule and gated layer][modeling]

There are three distinct normalization requirements: zero-centered RMSNorm uses
`1+weight` for decoder/final/q/k norms; gated RMSNorm uses `weight` directly and
normalizes before SiLU gating; DeltaNet q/k L2 normalization uses a sum instead of
a mean. Reference casts/reduction order matter. M0's generic direct-weight
RMSNorm is not a drop-in replacement for all three. [Reference][modeling]

RoPE rotates only 64 of the 256 head channels (derived from factor 0.25). Implement
the exact inverse-frequency, interleaving/recomposition, half rotation, and
untouched suffix from the reference. The three spatial/temporal coordinate
streams share ordinary positions for plain text, but retain the reference's
four-component position input handling (text plus three RoPE streams) in
comparison fixtures. Correctness at nonzero positions and resume offsets must
be demonstrated; supporting this text convention does not add vision support.
[Rotary and text-model code][modeling]

For batch size B, persist full-attention KV caches `[B,2,T,256]` per full layer,
linear convolution history `[B,6144,4]`, and DeltaNet recurrent state
`[B,16,128,128]` per linear layer. These shapes are derived from the source; the
reference cache retains four convolution positions, including initialization
padding. Preserve initialization flags, number of consumed tokens, causal-mask
and position state. [Model][modeling], [cache][cache]

A proposed resume format must additionally bind checkpoint/config/tokenizer
revision hashes, chosen dtype, layer order, prompt/token history, cache layout
version, stop criteria, and (when sampling exists) RNG/decoder state. Resume must
reproduce uninterrupted logits; incompatible state must fail. Reconstructing
state from prompt replay is acceptable for an initial correctness experiment,
provided it is explicit and tested; it is not a storage optimization claim.

## Tokenizer, special tokens, and prompts

The pinned tokenizer JSON specifies NFC normalization, a Unicode regex split,
then byte-level BPE without added prefix space or byte fallback; 248044 base
vocabulary entries and 247587 merges. Preserve the exact regex, merges, added
special-token behavior, byte decoder, and whitespace. The config labels its class
`Qwen2Tokenizer`, while the inspected modern implementation offers
`Qwen3_5Tokenizer`; use the checkpoint pipeline as data and verify actual token
IDs instead of assuming class-name equivalence. [Pipeline][tokenizer],
[metadata][tokenizer-config], [class][tokenization]

Metadata declares no BOS insertion; padding is `<|endoftext|>` (248044).
`<|im_start|>` is 248045, `<|im_end|>` 248046, `<think>` 248068, and `</think>`
248069. **Discrepancy to resolve:** text config declares EOS 248044, while tokenizer
config's EOS string maps to 248046. No `generation_config.json` appears in the
inspected repository listing. Do not choose a stop set silently; validate the
reference's resolved generation configuration and desired conversation-ending
behavior with explicit stop-token tests. [Config][config], [tokenizer config][tokenizer-config]

For text chat, reproduce the pinned Jinja template's role delimiters, newlines,
content trimming, system placement, and generation suffix exactly. With
`add_generation_prompt`, it opens the assistant turn; `enable_thinking=true`
opens a thinking block, otherwise it emits an empty closed thinking block. The
model card identifies non-thinking as the 0.8B default. The template also defines
history reasoning handling and tool wrappers; document/reject unsupported message
forms initially rather than inventing tool execution. Golden prompt bytes and
token IDs are required for Unicode, code indentation, blank lines, multi-turn
history, and thinking toggles. Do not execute repository-supplied Python or
untrusted template extensions. [Template][template], [model card][card]

## Comparison plan and unresolved evidence

1. Establish a pinned **development-only CPU reference environment**, eager
   attention, no remote code or accelerator kernels, eval/no-grad, documented
   PyTorch/tokenizers/Transformers versions. Compare BF16 checkpoint conversion
   into f32; do not assume reference dtype-cast behavior matches M0 f64 reductions.
2. Generate original small synthetic layer weights/inputs and capture embedding,
   all normalization variants, RoPE, convolution, DeltaNet state, attention,
   gates, residuals, MLP, final norm, and logits. Record dtype/shape/hash and
   reference revision. Ordinary Rust tests consume checked-in small fixtures.
3. Compare one-token, short prefill, repeated single-token decode, chunk boundaries,
   nonzero positions, reset, and serialized resume. First use batch-one unpadded
   sequences; reject packed/padded inputs until specifically validated.
4. With an explicitly supplied local checkpoint later, validate every tensor name,
   dtype/shape/range, tied weights, prompt bytes, and tokenizer IDs. Compare
   intermediate error growth and final logits, then greedy token sequences. Set
   per-operation absolute/relative tolerances from precision analysis before
   testing; log max/RMS error and investigate divergence rather than weakening
   thresholds. M0 tolerances are not automatically full-model tolerances.

Still unverified: actual weight-header dtypes/shapes and values; a runnable pinned
reference dependency combination; tokenizer/prompt runtime parity; resolved EOS
behavior; reference cache equivalence across prefill/decode/resume; and any logits,
generation quality, useful coding capability, or resource-efficiency advantage.
Authoritative sources were accessible; these are explicit future validation
requirements, not invented facts or blockers to M0.

The smallest next step is an independently generated tiny fixture for
**Qwen's zero-centered RMSNorm and its per-head application**, then a separate
scalar implementation matching that fixture. Continue through gated normalization,
causal convolution and one recurrent DeltaNet layer before assembling the model.

[config]: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json
[index]: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/model.safetensors.index.json
[tokenizer-config]: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/tokenizer_config.json
[tokenizer]: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/tokenizer.json
[template]: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/chat_template.jinja
[card]: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md
[modeling]: https://github.com/huggingface/transformers/blob/812a950d7037ba93bb624b154030e9068afdce4d/src/transformers/models/qwen3_5/modeling_qwen3_5.py
[configuration]: https://github.com/huggingface/transformers/blob/812a950d7037ba93bb624b154030e9068afdce4d/src/transformers/models/qwen3_5/configuration_qwen3_5.py
[tokenization]: https://github.com/huggingface/transformers/blob/812a950d7037ba93bb624b154030e9068afdce4d/src/transformers/models/qwen3_5/tokenization_qwen3_5.py
[cache]: https://github.com/huggingface/transformers/blob/812a950d7037ba93bb624b154030e9068afdce4d/src/transformers/cache_utils.py
