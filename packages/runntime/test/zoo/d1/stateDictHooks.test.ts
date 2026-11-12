import { describe, expect, it } from 'vitest';
import {
  derivedTensor,
  quantTensor,
  type LazyStateDict,
  type LazyTensor,
  type QuantTensor,
} from '../../../src/core/weights/safetensors.ts';
import { D1TextModel } from '../../../src/zoo/d1/model.ts';

const LAYER_TYPES = 'ccaccaccacacacac';
const HIDDEN = 1024;
const FFW = 4608;

function floats(shape: number[]): LazyTensor {
  const elems = shape.reduce((size, part) => size * part, 1);
  return derivedTensor(shape, () => Promise.resolve(new Float32Array(elems)), elems * 4);
}

function quant(name: string, shape: [number, number], base: number): QuantTensor {
  const [rows, cols] = shape;
  const groups = cols / 32;
  return quantTensor({
    name,
    shape,
    bits: 8,
    groupSize: 32,
    byteLength: rows * cols + rows * groups * 2,
    async ints() {
      const output = new Int8Array(rows * cols);
      for (let row = 0; row < rows; row++)
        output.fill((base + row) % 120, row * cols, (row + 1) * cols);
      return output;
    },
    async scales() {
      const output = new Float32Array(rows * groups);
      for (let row = 0; row < rows; row++)
        output.fill(base + row + 1, row * groups, (row + 1) * groups);
      return output;
    },
  });
}

function textCheckpoint(): LazyStateDict {
  const tensors = new Map<string, LazyTensor>();
  const add = (name: string, shape: number[], isQuant = false, base = 0) => {
    tensors.set(name, isQuant ? quant(name, shape as [number, number], base) : floats(shape));
  };
  add('encoder.embedding_norm.weight', [HIDDEN]);
  for (const [i, kind] of [...LAYER_TYPES].entries()) {
    const prefix = `encoder.layers.${i}`;
    add(`${prefix}.operator_norm.weight`, [HIDDEN]);
    add(`${prefix}.ffn_norm.weight`, [HIDDEN]);
    add(`${prefix}.feed_forward.w1.weight`, [FFW, HIDDEN], true, 20);
    add(`${prefix}.feed_forward.w3.weight`, [FFW, HIDDEN], true, 10);
    if (kind === 'c') {
      add(`${prefix}.conv.conv.weight`, [HIDDEN, 1, 3]);
    } else {
      add(`${prefix}.self_attn.q_proj.weight`, [HIDDEN, HIDDEN], true);
      add(`${prefix}.self_attn.k_proj.weight`, [HIDDEN / 2, HIDDEN], true);
      add(`${prefix}.self_attn.q_layernorm.weight`, [64]);
      add(`${prefix}.self_attn.k_layernorm.weight`, [64]);
    }
  }

  for (let i = 0; i < 2; i++) {
    const prefix = `head.head.layers.${i}`;
    add(`${prefix}.self_attn.in_proj_weight`, [3 * HIDDEN, HIDDEN], true);
    add(`${prefix}.self_attn.in_proj_bias`, [3 * HIDDEN]);
    for (const name of ['out_proj.weight', 'out_proj.bias']) {
      add(
        `${prefix}.self_attn.${name}`,
        name.endsWith('weight') ? [HIDDEN, HIDDEN] : [HIDDEN],
        name.endsWith('weight'),
      );
    }
    add(`${prefix}.linear1.weight`, [4096, HIDDEN], true);
    add(`${prefix}.linear1.bias`, [4096]);
    add(`${prefix}.linear2.weight`, [HIDDEN, 4096], true);
    add(`${prefix}.linear2.bias`, [HIDDEN]);
    for (const name of ['norm1', 'norm2']) {
      add(`${prefix}.${name}.weight`, [HIDDEN]);
      add(`${prefix}.${name}.bias`, [HIDDEN]);
    }
  }
  add('head.scorer.0.weight', [HIDDEN]);
  add('head.scorer.0.bias', [HIDDEN]);
  add('head.scorer.1.weight', [HIDDEN, HIDDEN], true);
  add('head.scorer.1.bias', [HIDDEN]);
  add('head.scorer.3.weight', [1, HIDDEN]);
  add('head.scorer.3.bias', [1]);
  add('vision.unused.weight', [HIDDEN, HIDDEN]);
  add('audio.unused.weight', [HIDDEN, HIDDEN]);
  return {
    metadata: {},
    tensors,
    totalBytes: () => [...tensors.values()].reduce((sum, t) => sum + t.byteLength, 0),
  };
}

describe('D1 quantized checkpoint transforms', () => {
  it('reorders and stacks native quant rows while keeping row scales', async () => {
    const model = new D1TextModel();
    const sd = textCheckpoint();
    model.transformStateDict(sd);

    const q = sd.tensors.get('encoder.layers.2.self_attn.q_proj.weight');
    if (q?.kind !== 'quant') throw new Error('q projection should stay quantized');
    const qInts = await q.ints();
    expect([qInts[0], qInts[HIDDEN], qInts[2 * HIDDEN], qInts[3 * HIDDEN]]).toEqual([0, 32, 1, 33]);
    const qScales = await q.scales();
    expect([qScales[0], qScales[HIDDEN / 32], qScales[2 * (HIDDEN / 32)]]).toEqual([1, 33, 2]);

    const w13 = sd.tensors.get('encoder.layers.0.feed_forward.w13.weight');
    if (w13?.kind !== 'quant') throw new Error('stacked feed-forward weight should stay quantized');
    const w13Ints = await w13.ints();
    const w13Scales = await w13.scales();
    expect(w13.shape).toEqual([2 * FFW, HIDDEN]);
    expect([w13Ints[0], w13Ints[FFW * HIDDEN]]).toEqual([10, 20]);
    expect([w13Scales[0], w13Scales[FFW * (HIDDEN / 32)]]).toEqual([11, 21]);

    const qkv = sd.tensors.get('head.layers.0.in_proj.weight');
    if (qkv?.kind !== 'quant') throw new Error('attention projection should stay quantized');
    const scales = await qkv.scales();
    expect(scales[0]).toBe(0.125);
    expect(scales[(HIDDEN - 1) * (HIDDEN / 32)]).toBe(128);
    expect(scales[HIDDEN * (HIDDEN / 32)]).toBe(1025);
    expect(sd.tensors.has('vision.unused.weight')).toBe(false);
    expect(sd.tensors.has('audio.unused.weight')).toBe(false);
    expect(sd.tensors.get('head.scorer_out.weight')?.kind).toBe('derived');
  });

  it('rejects a stacked linear weight with mixed float and quantized members', () => {
    const model = new D1TextModel();
    const sd = textCheckpoint();
    sd.tensors.set('encoder.layers.0.feed_forward.w1.weight', floats([FFW, HIDDEN]));
    expect(() => model.transformStateDict(sd)).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
  });
});
