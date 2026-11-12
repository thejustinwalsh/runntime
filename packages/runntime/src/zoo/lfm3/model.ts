/** LFM3-Omni's causal text backbone. The audio and vision towers in the
 *  original checkpoint are intentionally ignored by this text-only loader. */

import type { TgpuRoot } from 'typegpu';
import {
  add,
  cat,
  chunk,
  depthwiseConv1d,
  mul,
  nn,
  reshape,
  RunntimeError,
  rope,
  sdpa,
  sigmoid,
  silu,
  slice,
  tensor,
  type LazyStateDict,
  type LazyTensor,
  type Value,
} from '../../core/index.ts';
import { permuteRows, reshapeWeight, splitRows } from '../../core/weights/transforms.ts';
import { checkpointMismatch } from '../errors.ts';

/** Pinned architecture constants for Liquid's original text checkpoint. */
export const LFM3 = {
  vocab: 128_000,
  hidden: 2_048,
  ffn: 7_680,
  heads: 16,
  kvHeads: 4,
  headDim: 128,
  layers: 24,
  convKernel: 7,
  slidingWindow: 256,
  maxContext: 128_000,
  eps: 1e-5,
  fullRopeTheta: 1_000_000,
  slidingRopeTheta: 10_000,
  layerTypes: [
    'conv',
    'conv',
    'full_attention',
    'conv',
    'sliding_attention',
    'conv',
    'full_attention',
    'conv',
    'sliding_attention',
    'conv',
    'full_attention',
    'conv',
    'sliding_attention',
    'conv',
    'full_attention',
    'conv',
    'sliding_attention',
    'conv',
    'full_attention',
    'conv',
    'sliding_attention',
    'conv',
    'full_attention',
    'conv',
  ],
} as const;

/** One attention layer's retained key and value rows. */
export interface Lfm3AttentionCache {
  readonly key: Value;
  readonly value: Value;
}

/** Per-layer causal attention and short-convolution state. */
export interface Lfm3Cache {
  readonly attention: readonly (Lfm3AttentionCache | undefined)[];
  readonly convolution: readonly (Value | undefined)[];
  /** Number of token positions already evaluated. */
  readonly position: number;
}

/** Logits, next-token cache and caller-owned transient uploads for one forward. */
export interface Lfm3ModelStep {
  readonly logits: Value;
  readonly cache: Lfm3Cache;
  /** Caller-owned embeddings and rotary tables, released after evaluation. */
  readonly uploads: readonly Value[];
}

const projectionOrder = (heads: number): number[] => {
  const order: number[] = [];
  for (let head = 0; head < heads; head++) {
    const base = head * LFM3.headDim;
    for (let i = 0; i < LFM3.headDim / 2; i++) {
      order.push(base + i, base + i + LFM3.headDim / 2);
    }
  }
  return order;
};

const Q_ORDER = projectionOrder(LFM3.heads);
const K_ORDER = projectionOrder(LFM3.kvHeads);
const NORM_ORDER = projectionOrder(1);

function ropeTables(
  position: number,
  rows: number,
  theta: number,
  partial: number,
): {
  cos: Float32Array;
  sin: Float32Array;
} {
  const cos = new Float32Array(rows * LFM3.headDim);
  const sin = new Float32Array(rows * LFM3.headDim);
  const rotatedPairs = Math.floor((partial * LFM3.headDim) / 2);
  for (let row = 0; row < rows; row++) {
    for (let pair = 0; pair < LFM3.headDim / 2; pair++) {
      const angle =
        pair < rotatedPairs ? (position + row) * theta ** (-(2 * pair) / LFM3.headDim) : 0;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      cos[row * LFM3.headDim + pair * 2] = c;
      cos[row * LFM3.headDim + pair * 2 + 1] = c;
      sin[row * LFM3.headDim + pair * 2] = s;
      sin[row * LFM3.headDim + pair * 2 + 1] = s;
    }
  }
  return { cos, sin };
}

function lastRows(x: Value, rows: number): Value {
  const current = x.shape.dims![0]!;
  return current > rows ? slice(x, 0, current - rows, current) : x;
}

function appendRows(previous: Value | undefined, current: Value): Value {
  return previous ? cat([previous, current], 0) : current;
}

class Lfm3Mlp extends nn.Module<[Value], Value> {
  readonly w1 = new nn.Linear(LFM3.hidden, LFM3.ffn, { bias: false });
  readonly w3 = new nn.Linear(LFM3.hidden, LFM3.ffn, { bias: false });
  readonly w2 = new nn.Linear(LFM3.ffn, LFM3.hidden, { bias: false });

  override forward(x: Value): Value {
    return this.w2.forward(mul(silu(this.w1.forward(x)), this.w3.forward(x)));
  }
}

class Lfm3Attention extends nn.Module<
  [
    Value,
    number,
    Lfm3AttentionCache | undefined,
    { cos: Value; sin: Value },
    { cos: Value; sin: Value },
  ],
  { output: Value; cache: Lfm3AttentionCache }
> {
  readonly q_proj = new nn.Linear(LFM3.hidden, LFM3.heads * LFM3.headDim, { bias: false });
  readonly k_proj = new nn.Linear(LFM3.hidden, LFM3.kvHeads * LFM3.headDim, { bias: false });
  readonly v_proj = new nn.Linear(LFM3.hidden, LFM3.kvHeads * LFM3.headDim, { bias: false });
  readonly o_proj = new nn.Linear(LFM3.heads * LFM3.headDim, LFM3.hidden, { bias: false });
  readonly gate_proj = new nn.Linear(LFM3.hidden, LFM3.heads, { bias: false });
  readonly q_norm = new nn.RMSNorm(LFM3.headDim, LFM3.eps);
  readonly k_norm = new nn.RMSNorm(LFM3.headDim, LFM3.eps);

  forward(
    x: Value,
    layer: number,
    previous: Lfm3AttentionCache | undefined,
    fullRope: { cos: Value; sin: Value },
    slidingRope: { cos: Value; sin: Value },
  ): { output: Value; cache: Lfm3AttentionCache } {
    const rows = x.shape.dims![0]!;
    const layerType = LFM3.layerTypes[layer]!;
    const q = reshape(
      this.q_norm.forward(reshape(this.q_proj.forward(x), [rows * LFM3.heads, LFM3.headDim])),
      [rows, LFM3.heads * LFM3.headDim],
    );
    const k = reshape(
      this.k_norm.forward(reshape(this.k_proj.forward(x), [rows * LFM3.kvHeads, LFM3.headDim])),
      [rows, LFM3.kvHeads * LFM3.headDim],
    );
    const tables = layerType === 'full_attention' ? fullRope : slidingRope;
    const rotatedQ = rope(q, tables.cos, tables.sin, { headDim: LFM3.headDim });
    const rotatedK = rope(k, tables.cos, tables.sin, { headDim: LFM3.headDim });
    const scaledQ = mul(rotatedQ, 1 / Math.sqrt(LFM3.headDim));
    const value = this.v_proj.forward(x);
    const keys = appendRows(previous?.key, rotatedK);
    const values = appendRows(previous?.value, value);
    const keyOffset = previous?.key.shape.dims![0] ?? 0;
    const sliding = layerType === 'sliding_attention';
    const attended = sdpa(scaledQ, keys, values, {
      qHeads: LFM3.heads,
      kvHeads: LFM3.kvHeads,
      headDim: LFM3.headDim,
      windowLeft: sliding ? LFM3.slidingWindow - 1 : Infinity,
      windowRight: 0,
      qPosOffset: keyOffset,
    });
    const gated = mul(attended, sigmoid(this.gate_proj.forward(x)));
    const output = this.o_proj.forward(gated);
    return {
      output,
      cache: {
        key: sliding ? lastRows(keys, LFM3.slidingWindow) : keys,
        value: sliding ? lastRows(values, LFM3.slidingWindow) : values,
      },
    };
  }
}

class Lfm3DepthwiseConv extends nn.Module<[Value, number], Value> {
  readonly weight = new nn.Parameter({
    elems: LFM3.hidden * LFM3.convKernel,
    dtype: 'f32',
    dims: [LFM3.hidden, LFM3.convKernel],
  });

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const name = `${prefix}.weight`;
    const tensor = sd.tensors.get(name);
    if (!tensor) return;
    const expected = [LFM3.hidden, 1, LFM3.convKernel];
    if (tensor.shape.join() !== expected.join()) {
      throw checkpointMismatch(
        'LFM3 depthwise conv',
        `${name} has shape [${tensor.shape.join(', ')}], expected [${expected.join(', ')}]`,
      );
    }
    sd.tensors.set(name, reshapeWeight(tensor, [LFM3.hidden, LFM3.convKernel]));
  }

  forward(x: Value, padLeft: number): Value {
    return depthwiseConv1d(x, this.weight.value, {
      kernelSize: LFM3.convKernel,
      padLeft,
    });
  }
}

class Lfm3ShortConv extends nn.Module<[Value, Value | undefined], { output: Value; cache: Value }> {
  readonly in_proj = new nn.Linear(LFM3.hidden, 3 * LFM3.hidden, { bias: false });
  readonly out_proj = new nn.Linear(LFM3.hidden, LFM3.hidden, { bias: false });
  readonly conv = new Lfm3DepthwiseConv();

  forward(x: Value, previous: Value | undefined): { output: Value; cache: Value } {
    const [b, c, gate] = chunk(this.in_proj.forward(x), 3, 1);
    const gated = mul(b!, gate!);
    const combined = appendRows(previous, gated);
    const padLeft = Math.max(0, LFM3.convKernel - 1 - (previous?.shape.dims![0] ?? 0));
    const convolved = this.conv.forward(combined, padLeft);
    const output = this.out_proj.forward(mul(c!, convolved));
    return { output, cache: lastRows(combined, LFM3.convKernel - 1) };
  }
}

class Lfm3Layer extends nn.Module<
  [Value, number, Lfm3Cache | undefined, { cos: Value; sin: Value }, { cos: Value; sin: Value }],
  { hidden: Value; attention?: Lfm3AttentionCache; convolution?: Value }
> {
  readonly operator_norm = new nn.RMSNorm(LFM3.hidden, LFM3.eps);
  readonly ffn_norm = new nn.RMSNorm(LFM3.hidden, LFM3.eps);
  readonly post_operator_norm = new nn.RMSNorm(LFM3.hidden, LFM3.eps);
  readonly post_ffn_norm = new nn.RMSNorm(LFM3.hidden, LFM3.eps);
  readonly feed_forward = new Lfm3Mlp();
  readonly self_attn?: Lfm3Attention;
  readonly conv?: Lfm3ShortConv;

  constructor(layer: number) {
    super();
    if (LFM3.layerTypes[layer] === 'conv') this.conv = new Lfm3ShortConv();
    else this.self_attn = new Lfm3Attention();
  }

  forward(
    hidden: Value,
    layer: number,
    cache: Lfm3Cache | undefined,
    fullRope: { cos: Value; sin: Value },
    slidingRope: { cos: Value; sin: Value },
  ): { hidden: Value; attention?: Lfm3AttentionCache; convolution?: Value } {
    const residual = hidden;
    const normalized = this.operator_norm.forward(hidden);
    let op: Value;
    let attention: Lfm3AttentionCache | undefined;
    let convolution: Value | undefined;
    if (this.self_attn) {
      const result = this.self_attn.forward(
        normalized,
        layer,
        cache?.attention[layer],
        fullRope,
        slidingRope,
      );
      op = result.output;
      attention = result.cache;
    } else {
      const result = this.conv!.forward(normalized, cache?.convolution[layer]);
      op = result.output;
      convolution = result.cache;
    }
    hidden = add(this.post_operator_norm.forward(op), residual);
    const feedForward = this.post_ffn_norm.forward(
      this.feed_forward.forward(this.ffn_norm.forward(hidden)),
    );
    return {
      hidden: add(hidden, feedForward),
      attention,
      convolution,
    };
  }
}

class Lfm3LanguageModel extends nn.Module<
  [Value, number, number, Lfm3Cache | undefined, TgpuRoot, Value[]],
  { hidden: Value; cache: Lfm3Cache }
> {
  readonly embed_tokens = new nn.Embedding(LFM3.vocab, LFM3.hidden, { device: 'cpu' });
  readonly layers = new nn.ModuleList(
    Array.from({ length: LFM3.layers }, (_, layer) => new Lfm3Layer(layer)),
  );
  readonly embedding_norm = new nn.RMSNorm(LFM3.hidden, LFM3.eps);

  forward(
    input: Value,
    idsLength: number,
    position: number,
    previous: Lfm3Cache | undefined,
    root: TgpuRoot,
    uploads: Value[],
  ): { hidden: Value; cache: Lfm3Cache } {
    const full = ropeTables(position, idsLength, LFM3.fullRopeTheta, 0.5);
    const sliding = ropeTables(position, idsLength, LFM3.slidingRopeTheta, 1);
    const fullRope = {
      cos: track(tensor(root, full.cos, [idsLength, LFM3.headDim]), uploads),
      sin: track(tensor(root, full.sin, [idsLength, LFM3.headDim]), uploads),
    };
    const slidingRope = {
      cos: track(tensor(root, sliding.cos, [idsLength, LFM3.headDim]), uploads),
      sin: track(tensor(root, sliding.sin, [idsLength, LFM3.headDim]), uploads),
    };
    const attention: (Lfm3AttentionCache | undefined)[] = [];
    const convolution: (Value | undefined)[] = [];
    let hidden = input;
    for (let layer = 0; layer < LFM3.layers; layer++) {
      const result = this.layers.items[layer]!.forward(
        hidden,
        layer,
        previous,
        fullRope,
        slidingRope,
      );
      hidden = result.hidden;
      attention[layer] = result.attention;
      convolution[layer] = result.convolution;
    }
    return {
      hidden: this.embedding_norm.forward(hidden),
      cache: {
        attention,
        convolution,
        position: position + idsLength,
      },
    };
  }
}

const LM_HEAD_SHARDS = 16;
const LM_HEAD_SHARD_ROWS = LFM3.vocab / LM_HEAD_SHARDS;

class Lfm3LmHead extends nn.Module<[Value], Value> {
  readonly projections = new nn.ModuleList(
    Array.from(
      { length: LM_HEAD_SHARDS },
      () => new nn.Linear(LFM3.hidden, LM_HEAD_SHARD_ROWS, { bias: false }),
    ),
  );

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const name = `${prefix}.weight`;
    const weight = sd.tensors.get(name);
    if (!weight) throw checkpointMismatch('LFM3 checkpoint', `has no ${name} tied output weight`);
    if (weight.shape.join() !== [LFM3.vocab, LFM3.hidden].join()) {
      throw checkpointMismatch(
        'LFM3 output weight',
        `has shape [${weight.shape.join(', ')}], expected [${LFM3.vocab}, ${LFM3.hidden}]`,
      );
    }
    const shards = splitRows(weight, Array(LM_HEAD_SHARDS).fill(LM_HEAD_SHARD_ROWS));
    sd.tensors.delete(name);
    shards.forEach((shard, index) => {
      sd.tensors.set(`${prefix}.projections.${index}.weight`, shard);
    });
  }

  override forward(x: Value): Value {
    return cat(
      this.projections.items.map((projection) => projection.forward(x)),
      1,
    );
  }
}

function track(value: Value, uploads: Value[]): Value {
  uploads.push(value);
  return value;
}

/** Returns all retained GPU values in an LFM3 cache. */
export function cacheValues(cache: Lfm3Cache): Value[] {
  return [
    ...cache.attention.flatMap((entry) => (entry ? [entry.key, entry.value] : [])),
    ...cache.convolution.filter((entry): entry is Value => entry !== undefined),
  ];
}

/** Causal text model using Liquid's original tied-embedding LFM3 checkpoint. */
export class Lfm3CausalLM extends nn.Module<
  [readonly number[], TgpuRoot, Lfm3Cache?],
  Lfm3ModelStep
> {
  readonly model = new Lfm3LanguageModel();
  readonly lm_head = new Lfm3LmHead();

  override transformStateDict(sd: LazyStateDict): void {
    const prefix = 'model.language_model.';
    const original = [...sd.tensors.entries()];
    const language = original.filter(([name]) => name.startsWith(prefix));
    sd.tensors.clear();
    let embedding: LazyTensor | undefined;
    for (const [sourceName, sourceTensor] of language) {
      const name = sourceName.slice(prefix.length);
      const targetName = /(?:^|\.)[a-z_]*norm\.weight$/.test(name)
        ? name.replace(/\.weight$/, '.scale')
        : name;
      let tensor = sourceTensor;
      if (name === 'embed_tokens.weight') embedding = tensor;
      if (name.endsWith('.self_attn.q_proj.weight')) tensor = permuteRows(tensor, Q_ORDER);
      else if (name.endsWith('.self_attn.k_proj.weight')) tensor = permuteRows(tensor, K_ORDER);
      else if (
        name.endsWith('.self_attn.q_norm.weight') ||
        name.endsWith('.self_attn.k_norm.weight')
      ) {
        tensor = permuteRows(tensor, NORM_ORDER);
      }
      sd.tensors.set(`model.${targetName}`, tensor);
    }
    if (!embedding) {
      throw checkpointMismatch(
        'LFM3 checkpoint',
        'has no model.language_model.embed_tokens.weight',
      );
    }
    sd.tensors.set('lm_head.weight', embedding);
  }

  forward(ids: readonly number[], root: TgpuRoot, previous?: Lfm3Cache): Lfm3ModelStep {
    if (ids.length === 0) {
      throw new RunntimeError('INVALID_ARGUMENT', 'LFM3 forward needs at least one input token');
    }
    const position = previous?.position ?? 0;
    if (position + ids.length > LFM3.maxContext) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `LFM3 context exhausted: ${position + ids.length} > ${LFM3.maxContext}`,
      );
    }
    const uploads: Value[] = [];
    const embedding = track(this.model.embed_tokens.forward(ids, root), uploads);
    const result = this.model.forward(embedding, ids.length, position, previous, root, uploads);
    const final = slice(result.hidden, 0, ids.length - 1, ids.length);
    return {
      logits: this.lm_head.forward(final),
      cache: result.cache,
      uploads,
    };
  }
}
