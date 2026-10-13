#!/usr/bin/env python3
"""Development-only fixtures, independent 80-digit Decimal reference formulas.
Run from repository root: python3 evals/reference/generate.py
No third-party libraries or Rust implementation are used.
"""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def dec(x):
    return Decimal.from_float(f32(x))

def reference(op, shape, data, **extra):
    data = [f32(x) for x in data]
    xs = list(map(dec, data))
    width = shape[-1]
    with localcontext() as ctx:
        ctx.prec = 80
        if op == 'matmul':
            m, k = shape
            n = extra['rhs_shape'][1]
            bs = list(map(dec, extra['rhs']))
            ys = [sum((xs[r*k+i] * bs[i*n+c] for i in range(k)), Decimal(0))
                  for r in range(m) for c in range(n)]
        elif op == 'rms_norm':
            ys = []
            for start in range(0, len(xs), width):
                row = xs[start:start+width]
                rms = (sum(x*x for x in row)/Decimal(width) + dec(extra['epsilon'])).sqrt()
                ys.extend(x * dec(w) / rms for x, w in zip(row, extra['weight']))
        elif op == 'softmax':
            # Direct high-precision definition, deliberately independent of the
            # Rust max-subtraction algorithm. Decimal can represent exp(1000).
            ys = []
            for start in range(0, len(xs), width):
                exps = [x.exp() for x in xs[start:start+width]]
                ys.extend(x/sum(exps) for x in exps)
        elif op == 'silu':
            ys = [x/(1+(-x).exp()) for x in xs]
        else:
            raise ValueError(op)
    return dict(op=op, shape=shape, data=data, **extra, expected=[float(y) for y in ys])

cases = [
    reference('matmul', [2, 3], [1, 2, 3, 4, 5, 6], rhs_shape=[3, 2], rhs=[7, 8, 9, 10, 11, 12]),
    reference('matmul', [3, 2], [-.25, 2, 3.5, -4, .125, 0], rhs_shape=[2, 4], rhs=[1, -.5, 2, 3, -2, 4, .25, -1]),
    reference('matmul', [1, 3], [1e10, 1, -1e10], rhs_shape=[3, 1], rhs=[1, 1, 1]),
    reference('rms_norm', [2, 3], [1, -2, 3, -4, 0, 2], weight=[.5, -2, 1.5], epsilon=1e-5),
    reference('rms_norm', [3], [1e30, -1e30, 1e30], weight=[2, -1, .25], epsilon=1e-6),
    reference('rms_norm', [2, 2], [0, 0, 2, 2], weight=[-1, 3], epsilon=.25),
    reference('softmax', [2, 3], [1000, 1001, 999, -1000, -1001, -999]),
    reference('softmax', [3], [.1, -.3, 2.75]),
    reference('softmax', [2], [0, 0]),
    reference('softmax', [2], [1000, 1000]),
    reference('silu', [9], [-1000, -80, -10, -1, 0, .5, 2, 80, 1000]),
]
(ROOT/'evals/fixtures/numerics.json').write_text(json.dumps(dict(
    format_version=1, generator_revision='decimal-v1', precision_digits=80,
    absolute_tolerance=1e-6, relative_tolerance=2e-6, cases=cases), indent=2)+'\n')

# Own synthetic data, not a model checkpoint. Names deliberately aren't Qwen mappings.
model = ROOT/'evals/fixtures/synthetic-model'
model.mkdir(parents=True, exist_ok=True)
(model/'config.json').write_text(json.dumps(dict(
    architectures=['Qwen3_5ForConditionalGeneration'], model_type='qwen3_5',
    text_config=dict(model_type='qwen3_5_text'),
    fixture_note='Synthetic metadata only; not an executable Qwen checkpoint.'), indent=2)+'\n')
header = json.dumps({
    '__metadata__': {'purpose': 'synthetic M0 inspection fixture'},
    'demo.weight': {'dtype': 'F32', 'shape': [2, 3], 'data_offsets': [0, 24]},
    'demo.scale': {'dtype': 'F32', 'shape': [3], 'data_offsets': [24, 36]},
}, separators=(',', ':')).encode()
header += b' ' * (-len(header) % 8)
(model/'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<9f', 1, 2, 3, 4, 5, 6, 1, 1, 1))
print('Wrote independent numerical fixtures and synthetic metadata fixture.')

# Contract example computed from independently authored metadata above, not from
# invoking the Rust CLI. Caveat strings are presentation text, not numerical truth.
expected = dict(
    schema_version=1,
    architecture=dict(model_type='qwen3_5', architectures=['Qwen3_5ForConditionalGeneration'],
                      text_model_type='qwen3_5_text', recognized=True, execution_implemented=False),
    layout='single_safetensors',
    files=[dict(name='model.safetensors', file_bytes=8+len(header)+36,
                header_bytes=len(header), payload_bytes=36, tensors=[
                    dict(name='demo.scale', shape=[3], dtype='F32', data_offsets=[24,36], payload_bytes=12),
                    dict(name='demo.weight', shape=[2,3], dtype='F32', data_offsets=[0,24], payload_bytes=24)])],
    total_tensor_payload_bytes=36, total_weight_file_bytes=8+len(header)+36,
    limitations=[
        'Metadata and byte layout validated; weight payload values and integrity not verified.',
        'Architecture recognition is declarative; tensor names are not checked against a model architecture.',
        'Model execution is not implemented; file and payload bytes are not runtime RAM estimates.',
        'Only a flat directory with config.json and one model.safetensors is supported; indexed/sharded layouts are rejected.',
        'Other regular assets are ignored; tokenizer, prompts, and configuration semantics are not validated.',
    ],
)
(ROOT/'evals/fixtures/inspect-v1.json').write_text(json.dumps(expected,indent=2)+'\n')
