import { describe, expect, it } from 'vitest';
import {
  derivedTensor,
  quantTensor,
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

function qTensor(shape: number[], bits: 4 | 8, groupSize: number): LazyTensor {
  const elems = shape.reduce((size, part) => size * part, 1);
  return quantTensor({
    name: 'test GGUF tensor',
    shape,
    bits,
    groupSize,
    byteLength: (elems * bits) / 8,
    ints: () => Promise.resolve(new Int8Array(elems)),
    scales: () => Promise.resolve(new Float32Array(elems / groupSize)),
  });
}

function ggufMetadata(): Record<string, unknown> {
  return {
    'general.architecture': 'lfm3',
    'general.file_type': 15,
    'general.quantization_version': 2,
    'lfm3.block_count': LFM3.layers,
    'lfm3.context_length': LFM3.maxContext,
    'lfm3.embedding_length': LFM3.hidden,
    'lfm3.feed_forward_length': LFM3.ffn,
    'lfm3.attention.head_count': LFM3.heads,
    'lfm3.rope.freq_base': LFM3.fullRopeTheta,
    'lfm3.rope.freq_base_swa': LFM3.slidingRopeTheta,
    'lfm3.attention.layer_norm_rms_epsilon': Math.fround(LFM3.eps),
    'lfm3.attention.key_length': LFM3.headDim,
    'lfm3.attention.value_length': LFM3.headDim,
    'lfm3.rope.dimension_count': LFM3.headDim,
    'lfm3.attention.sliding_window': LFM3.slidingWindow,
    'lfm3.shortconv.l_cache': LFM3.convKernel,
    'lfm3.attention.head_count_kv': LFM3.layerTypes.map((type) =>
      type === 'conv' ? 0 : LFM3.kvHeads,
    ),
    'lfm3.attention.sliding_window_pattern': LFM3.layerTypes.map(
      (type) => type === 'sliding_attention',
    ),
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

  it('maps the pinned GGUF names and retains quantized tied embeddings', () => {
    const model = new Lfm3CausalLM();
    const sd = stateDict({
      'token_embd.weight': qTensor([LFM3.vocab, LFM3.hidden], 8, 16),
      'output_norm.weight': floats([LFM3.hidden]),
      'rope_freqs.weight': floats([64]),
      'blk.0.shortconv.conv.weight': floats([LFM3.hidden, LFM3.convKernel]),
      'blk.2.attn_q.weight': qTensor([LFM3.heads * LFM3.headDim, LFM3.hidden], 4, 32),
      'blk.2.attn_q_norm.weight': floats([LFM3.headDim]),
    });
    sd.ggufMetadata = ggufMetadata();

    model.transformStateDict(sd);
    expect([...sd.tensors.keys()]).toEqual([
      'model.embed_tokens.weight',
      'model.embedding_norm.scale',
      'model.layers.0.conv.conv.weight',
      'model.layers.2.self_attn.q_proj.weight',
      'model.layers.2.self_attn.q_norm.scale',
      'lm_head.weight',
    ]);
    expect(sd.tensors.get('model.embed_tokens.weight')?.kind).toBe('quant');
    expect(sd.tensors.get('model.layers.2.self_attn.q_proj.weight')).toMatchObject({
      kind: 'quant',
      bits: 4,
      groupSize: 32,
    });

    model.model.layers.items[0]!.conv!.conv.transformStateDict(sd, 'model.layers.0.conv.conv');
    expect(sd.tensors.get('model.layers.0.conv.conv.weight')?.shape).toEqual([
      LFM3.hidden,
      LFM3.convKernel,
    ]);

    model.lm_head.transformStateDict(sd, 'lm_head');
    expect(sd.tensors.has('lm_head.weight')).toBe(false);
    expect(sd.tensors.get('lm_head.projections.0.weight')?.kind).toBe('quant');
  });

  it('rejects GGUF metadata for a different architecture', () => {
    const model = new Lfm3CausalLM();
    const sd = stateDict({
      'token_embd.weight': qTensor([LFM3.vocab, LFM3.hidden], 8, 16),
      'rope_freqs.weight': floats([64]),
    });
    sd.ggufMetadata = { ...ggufMetadata(), 'general.architecture': 'llama' };
    expect(() => model.transformStateDict(sd)).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
  });
});
