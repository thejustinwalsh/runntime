import { describe, expect, it } from 'vitest';
import {
  derivedTensor,
  type LazyStateDict,
  type LazyTensor,
} from '../../../src/core/weights/safetensors.ts';
import { LFM3, Lfm3CausalLM } from '../../../src/zoo/lfm3/model.ts';

function floats(shape: number[], value: (index: number) => number = () => 0): LazyTensor {
  const elems = shape.reduce((size, part) => size * part, 1);
  return derivedTensor(
    shape,
    () => Promise.resolve(Float32Array.from({ length: elems }, (_, index) => value(index))),
    elems * 4,
  );
}

function stateDict(entries: Record<string, LazyTensor>): LazyStateDict {
  const tensors = new Map(Object.entries(entries));
  return {
    metadata: {},
    tensors,
    totalBytes: () => [...tensors.values()].reduce((sum, t) => sum + t.byteLength, 0),
  };
}

describe('LFM3 checkpoint transforms', () => {
  it('filters the multimodal checkpoint, renames norms and interleaves Q/K RoPE rows', async () => {
    const model = new Lfm3CausalLM();
    const sd = stateDict({
      'model.language_model.embed_tokens.weight': floats([LFM3.vocab, LFM3.hidden]),
      'model.language_model.layers.2.self_attn.q_proj.weight': floats(
        [LFM3.heads * LFM3.headDim, LFM3.hidden],
        (index) => Math.floor(index / LFM3.hidden),
      ),
      'model.language_model.layers.2.self_attn.k_proj.weight': floats(
        [LFM3.kvHeads * LFM3.headDim, LFM3.hidden],
        (index) => Math.floor(index / LFM3.hidden),
      ),
      'model.language_model.layers.2.self_attn.q_norm.weight': floats([LFM3.headDim], (i) => i),
      'model.language_model.layers.2.self_attn.k_norm.weight': floats([LFM3.headDim], (i) => i),
      'model.audio_encoder.layers.0.weight': floats([1]),
    });

    model.transformStateDict(sd);
    expect([...sd.tensors.keys()]).toEqual([
      'model.embed_tokens.weight',
      'model.layers.2.self_attn.q_proj.weight',
      'model.layers.2.self_attn.k_proj.weight',
      'model.layers.2.self_attn.q_norm.scale',
      'model.layers.2.self_attn.k_norm.scale',
      'lm_head.weight',
    ]);
    const q = await sd.tensors.get('model.layers.2.self_attn.q_proj.weight')!.f32();
    const kNorm = await sd.tensors.get('model.layers.2.self_attn.k_norm.scale')!.f32();
    expect(Array.from(q.slice(0, 8 * LFM3.hidden).filter((_, i) => i % LFM3.hidden === 0))).toEqual(
      [0, 64, 1, 65, 2, 66, 3, 67],
    );
    expect(Array.from(kNorm.slice(0, 8))).toEqual([0, 64, 1, 65, 2, 66, 3, 67]);

    model.lm_head.transformStateDict(sd, 'lm_head');
    expect(sd.tensors.has('lm_head.weight')).toBe(false);
    expect(
      Array.from(
        { length: 16 },
        (_, index) => sd.tensors.get(`lm_head.projections.${index}.weight`)!.shape,
      ),
    ).toEqual(Array.from({ length: 16 }, () => [LFM3.vocab / 16, LFM3.hidden]));
  });
});
