/** Native TypeGPU LFM3-Omni audio encoder, depth decoder and Liquid codec. */

import {
  add,
  cat,
  chunk,
  deriveFromTensors,
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
  toChw,
  toHwc4,
  transpose,
  type LazyStateDict,
  type Value,
} from '../../../core/index.ts';
import type { TgpuRoot } from 'typegpu';
import { permuteRows, reshapeWeight, splitRows } from '../../../core/weights/transforms.ts';
import {
  quantTensor,
  type LazyTensor,
  type QuantTensor,
} from '../../../core/weights/safetensors.ts';
import { lfm3AudioFrameCount, lfm3AudioTokenCount, lfm3LogMelSpectrogram } from './frontend.ts';

const AUDIO_HIDDEN = 768;
const AUDIO_FFN = 1_792;
const AUDIO_HEADS = 8;
const AUDIO_KV_HEADS = 4;
const AUDIO_HEAD_DIM = 96;
const AUDIO_WINDOW = 32;
const AUDIO_LAYERS = 17;
const DEPTH_HIDDEN = 1_024;
const DEPTH_FFN = 2_816;
const DEPTH_HEADS = 32;
const DEPTH_KV_HEADS = 8;
const DEPTH_HEAD_DIM = 32;
const CODEBOOKS = 12;
const CODEBOOK_SIZE = 2_017;
const CODEBOOK_EOS = 2_016;
const CODEC_HIDDEN = 512;
const CODEC_FFN = 2_304;
const CODEC_SPECTRUM_WIDTH = 1_282;
const CODEC_OUTPUT_ROWS = Math.ceil(CODEC_SPECTRUM_WIDTH / 4) * 4;
const CODEC_HEADS = 8;
const CODEC_KV_HEADS = 2;
const CODEC_HEAD_DIM = 64;
const CODEC_FRAMES_PER_TOKEN = 6;
const AUDIO_LAYERS_WITH_ATTENTION = new Set([2, 5, 8, 11, 14, 15, 16]);
const CODEC_LAYERS_WITH_ATTENTION = new Set([2, 4, 6]);
const CODEBOOK_HEAD_ROWS = Math.ceil(CODEBOOK_SIZE / 4) * 4;

export interface Lfm3AudioAttentionCache {
  readonly key: Value;
  readonly value: Value;
}

export interface Lfm3AudioLayerCache {
  readonly attention?: Lfm3AudioAttentionCache;
  readonly convolution?: Value;
}

export interface Lfm3AudioDecoderCache {
  readonly layers: readonly (Lfm3AudioLayerCache | undefined)[];
  readonly position: number;
}

interface DecoderConfig {
  readonly hidden: number;
  readonly heads: number;
  readonly kvHeads: number;
  readonly headDim: number;
  readonly windowLeft: number;
  readonly causal: boolean;
  readonly ffn: number;
  readonly sandwich: boolean;
  readonly attentionLayers: ReadonlySet<number>;
  readonly layers: number;
}

function track(value: Value, uploads: Value[]): Value {
  uploads.push(value);
  return value;
}

function lastRows(value: Value, rows: number): Value {
  const total = value.shape.dims![0]!;
  return total > rows ? slice(value, 0, total - rows, total) : value;
}

function rowOrder(heads: number, headDim: number): number[] {
  const result: number[] = [];
  for (let head = 0; head < heads; head++) {
    const base = head * headDim;
    for (let i = 0; i < headDim / 2; i++) result.push(base + i, base + i + headDim / 2);
  }
  return result;
}

function ropeTables(
  position: number,
  rows: number,
  headDim: number,
  theta: number,
  partial: number,
): { cos: Float32Array; sin: Float32Array } {
  const cos = new Float32Array(rows * headDim);
  const sin = new Float32Array(rows * headDim);
  const rotatedPairs = Math.floor((partial * headDim) / 2);
  for (let row = 0; row < rows; row++) {
    for (let pair = 0; pair < headDim / 2; pair++) {
      const angle = pair < rotatedPairs ? (position + row) * theta ** (-(2 * pair) / headDim) : 0;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      cos[row * headDim + pair * 2] = c;
      cos[row * headDim + pair * 2 + 1] = c;
      sin[row * headDim + pair * 2] = s;
      sin[row * headDim + pair * 2 + 1] = s;
    }
  }
  return { cos, sin };
}

class AudioDownsampler extends nn.Module<[Float32Array, number, TgpuRoot, Value[]], Value> {
  readonly convIn = new nn.Conv2d(1, 256, {
    kernelSize: 3,
    stride: 2,
    padding: 1,
    bias: true,
    activation: 'silu',
  });
  readonly convMid = new nn.Conv2d(256, 256, {
    kernelSize: 3,
    stride: 2,
    padding: 1,
    groups: 256,
    bias: true,
  });
  readonly linMid = new nn.Conv2d(256, 256, { kernelSize: 1, bias: true, activation: 'silu' });
  readonly convEnd = new nn.Conv2d(256, 256, {
    kernelSize: 3,
    stride: 2,
    padding: 1,
    groups: 256,
    bias: true,
  });
  readonly linEnd = new nn.Conv2d(256, 256, { kernelSize: 1, bias: true, activation: 'silu' });
  readonly linOut = new nn.Linear(4_096, AUDIO_HIDDEN, { bias: true });

  override forward(
    features: Float32Array,
    frames: number,
    root: TgpuRoot,
    uploads: Value[],
  ): Value {
    const mel = track(
      tensor(root, features, { elems: features.length, dtype: 'f32', dims: [1, 128, frames] }),
      uploads,
    );
    let hidden = toHwc4(mel);
    hidden = this.convIn.forward(hidden);
    hidden = this.linMid.forward(this.convMid.forward(hidden));
    hidden = this.linEnd.forward(this.convEnd.forward(hidden));
    const chw = toChw(hidden, 'f16');
    const dimensions = chw.shape.dims!;
    const channels = dimensions[0]!;
    const mels = dimensions[1]!;
    const time = dimensions[2]!;
    if (channels !== 256 || mels !== 16) {
      throw new Error(
        `LFM3 audio downsampler produced [${chw.shape.dims!.join(', ')}], expected [256, 16, time]`,
      );
    }
    const rows = transpose(reshape(chw, [channels * mels, time]));
    return this.linOut.forward(rows);
  }
}

class AudioMlp extends nn.Module<[Value], Value> {
  readonly gate = new nn.Linear(AUDIO_HIDDEN, AUDIO_FFN, { bias: false });
  readonly up = new nn.Linear(AUDIO_HIDDEN, AUDIO_FFN, { bias: false });
  readonly down = new nn.Linear(AUDIO_FFN, AUDIO_HIDDEN, { bias: false });

  override forward(x: Value): Value {
    return this.down.forward(mul(silu(this.gate.forward(x)), this.up.forward(x)));
  }
}

class AudioShortConv extends nn.Module<[Value], Value> {
  readonly inProj = new nn.Linear(AUDIO_HIDDEN, AUDIO_HIDDEN * 3, { bias: false });
  readonly outProj = new nn.Linear(AUDIO_HIDDEN, AUDIO_HIDDEN, { bias: false });
  readonly depthwise = new nn.Parameter({
    elems: AUDIO_HIDDEN * 3,
    dtype: 'f32',
    dims: [AUDIO_HIDDEN, 3],
  });

  override forward(x: Value): Value {
    const [hidden, residual, gate] = chunk(this.inProj.forward(x), 3, 1);
    const convolved = depthwiseConv1d(mul(hidden!, gate!), this.depthwise.value, {
      kernelSize: 3,
      padLeft: 2,
    });
    return this.outProj.forward(mul(residual!, convolved));
  }
}

class AudioAttention extends nn.Module<[Value, TgpuRoot, Value[]], Value> {
  readonly qProj = new nn.Linear(AUDIO_HIDDEN, AUDIO_HEADS * AUDIO_HEAD_DIM, { bias: false });
  readonly kProj = new nn.Linear(AUDIO_HIDDEN, AUDIO_KV_HEADS * AUDIO_HEAD_DIM, { bias: false });
  readonly vProj = new nn.Linear(AUDIO_HIDDEN, AUDIO_KV_HEADS * AUDIO_HEAD_DIM, { bias: false });
  readonly outProj = new nn.Linear(AUDIO_HEADS * AUDIO_HEAD_DIM, AUDIO_HIDDEN, { bias: false });
  readonly qNorm = new nn.RMSNorm(AUDIO_HEAD_DIM, 1e-5);
  readonly kNorm = new nn.RMSNorm(AUDIO_HEAD_DIM, 1e-5);

  override forward(x: Value, root: TgpuRoot, uploads: Value[]): Value {
    const rows = x.shape.dims![0]!;
    const q = reshape(
      this.qNorm.forward(reshape(this.qProj.forward(x), [rows * AUDIO_HEADS, AUDIO_HEAD_DIM])),
      [rows, AUDIO_HEADS * AUDIO_HEAD_DIM],
    );
    const k = reshape(
      this.kNorm.forward(reshape(this.kProj.forward(x), [rows * AUDIO_KV_HEADS, AUDIO_HEAD_DIM])),
      [rows, AUDIO_KV_HEADS * AUDIO_HEAD_DIM],
    );
    const tables = ropeTables(0, rows, AUDIO_HEAD_DIM, 1_000_000, 1);
    const cos = track(tensor(root, tables.cos, [rows, AUDIO_HEAD_DIM]), uploads);
    const sin = track(tensor(root, tables.sin, [rows, AUDIO_HEAD_DIM]), uploads);
    const rotatedQ = rope(q, cos, sin, { headDim: AUDIO_HEAD_DIM });
    const rotatedK = rope(k, cos, sin, { headDim: AUDIO_HEAD_DIM });
    const attended = sdpa(
      mul(rotatedQ, 1 / Math.sqrt(AUDIO_HEAD_DIM)),
      rotatedK,
      this.vProj.forward(x),
      {
        qHeads: AUDIO_HEADS,
        kvHeads: AUDIO_KV_HEADS,
        headDim: AUDIO_HEAD_DIM,
        windowLeft: AUDIO_WINDOW - 1,
        windowRight: 0,
      },
    );
    return this.outProj.forward(attended);
  }
}

class AudioEncoderLayer extends nn.Module<[Value, TgpuRoot, Value[]], Value> {
  readonly operatorNorm = new nn.RMSNorm(AUDIO_HIDDEN, 1e-5);
  readonly ffnNorm = new nn.RMSNorm(AUDIO_HIDDEN, 1e-5);
  readonly attention?: AudioAttention;
  readonly shortConv?: AudioShortConv;
  readonly ffn = new AudioMlp();

  constructor(layer: number) {
    super();
    if (AUDIO_LAYERS_WITH_ATTENTION.has(layer)) this.attention = new AudioAttention();
    else this.shortConv = new AudioShortConv();
  }

  override forward(x: Value, root: TgpuRoot, uploads: Value[]): Value {
    const normalized = this.operatorNorm.forward(x);
    const operator = this.attention
      ? this.attention.forward(normalized, root, uploads)
      : this.shortConv!.forward(normalized);
    const withOperator = add(x, operator);
    return add(withOperator, this.ffn.forward(this.ffnNorm.forward(withOperator)));
  }
}

class AudioEncoderLfm extends nn.Module<[Value, TgpuRoot, Value[]], Value> {
  readonly layers = new nn.ModuleList(
    Array.from({ length: AUDIO_LAYERS }, (_, layer) => new AudioEncoderLayer(layer)),
  );
  readonly norm = new nn.RMSNorm(AUDIO_HIDDEN, 1e-5);

  override forward(x: Value, root: TgpuRoot, uploads: Value[]): Value {
    let hidden = x;
    for (const layer of this.layers.items) hidden = layer.forward(hidden, root, uploads);
    return this.norm.forward(hidden);
  }
}

class AudioProjector extends nn.Module<[Value], Value> {
  readonly layerNorm = new nn.LayerNorm(AUDIO_HIDDEN, 1e-5, { bias: true });
  readonly linear1 = new nn.Linear(AUDIO_HIDDEN, 2_048, { bias: true, activation: 'gelu' });
  readonly linear2 = new nn.Linear(2_048, 2_048, { bias: true });

  override forward(x: Value): Value {
    return this.linear2.forward(this.linear1.forward(this.layerNorm.forward(x)));
  }
}

class AudioEncoder extends nn.Module<[Float32Array, TgpuRoot, Value[]], Value> {
  readonly downsample = new AudioDownsampler();
  readonly lfm = new AudioEncoderLfm();
  readonly projector = new AudioProjector();

  override forward(audio: Float32Array, root: TgpuRoot, uploads: Value[]): Value {
    const frames = lfm3AudioFrameCount(audio.length);
    const features = lfm3LogMelSpectrogram(audio);
    return this.projector.forward(
      this.lfm.forward(this.downsample.forward(features, frames, root, uploads), root, uploads),
    );
  }
}

class DecoderMlp extends nn.Module<[Value], Value> {
  readonly gate: nn.Linear;
  readonly up: nn.Linear;
  readonly down: nn.Linear;

  constructor(
    readonly hidden: number,
    readonly intermediate: number,
  ) {
    super();
    this.gate = new nn.Linear(hidden, intermediate, { bias: false });
    this.up = new nn.Linear(hidden, intermediate, { bias: false });
    this.down = new nn.Linear(intermediate, hidden, { bias: false });
  }

  override forward(x: Value): Value {
    return this.down.forward(mul(silu(this.gate.forward(x)), this.up.forward(x)));
  }
}

class DecoderShortConv extends nn.Module<
  [Value, Value | undefined],
  { output: Value; cache: Value }
> {
  readonly inProj: nn.Linear;
  readonly outProj: nn.Linear;
  readonly depthwise: nn.Parameter;

  constructor(
    readonly hidden: number,
    readonly kernel: number,
  ) {
    super();
    this.inProj = new nn.Linear(hidden, hidden * 3, { bias: false });
    this.outProj = new nn.Linear(hidden, hidden, { bias: false });
    this.depthwise = new nn.Parameter({
      elems: hidden * kernel,
      dtype: 'f32',
      dims: [hidden, kernel],
    });
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const name = `${prefix}.depthwise`;
    const weight = sd.tensors.get(name);
    if (!weight) return;
    if (weight.shape.join() === [this.hidden, this.kernel].join()) return;
    if (weight.shape.join() !== [this.hidden, 1, this.kernel].join()) {
      throw new Error(
        `LFM3 short convolution '${name}' has shape [${weight.shape.join(', ')}], expected [${this.hidden}, ${this.kernel}]`,
      );
    }
    sd.tensors.set(name, reshapeWeight(weight, [this.hidden, this.kernel]));
  }

  override forward(x: Value, previous?: Value): { output: Value; cache: Value } {
    const [hidden, residual, gate] = chunk(this.inProj.forward(x), 3, 1);
    const gated = mul(hidden!, gate!);
    const combined = previous ? cat([previous, gated], 0) : gated;
    const convolved = depthwiseConv1d(combined, this.depthwise.value, {
      kernelSize: this.kernel,
      padLeft: Math.max(0, this.kernel - 1 - (previous?.shape.dims![0] ?? 0)),
    });
    return {
      output: this.outProj.forward(mul(residual!, convolved)),
      cache: lastRows(combined, this.kernel - 1),
    };
  }
}

class DecoderAttention extends nn.Module<
  [Value, TgpuRoot, Value[], DecoderConfig, Lfm3AudioAttentionCache | undefined, number],
  { output: Value; cache?: Lfm3AudioAttentionCache }
> {
  readonly qProj: nn.Linear;
  readonly kProj: nn.Linear;
  readonly vProj: nn.Linear;
  readonly gateProj: nn.Linear;
  readonly outProj: nn.Linear;
  readonly qNorm: nn.RMSNorm;
  readonly kNorm: nn.RMSNorm;

  constructor(
    readonly hidden: number,
    readonly heads: number,
    readonly kvHeads: number,
    readonly headDim: number,
  ) {
    super();
    this.qProj = new nn.Linear(hidden, heads * headDim, { bias: false });
    this.kProj = new nn.Linear(hidden, kvHeads * headDim, { bias: false });
    this.vProj = new nn.Linear(hidden, kvHeads * headDim, { bias: false });
    this.gateProj = new nn.Linear(hidden, heads, { bias: false });
    this.outProj = new nn.Linear(heads * headDim, hidden, { bias: false });
    this.qNorm = new nn.RMSNorm(headDim, 1e-5);
    this.kNorm = new nn.RMSNorm(headDim, 1e-5);
  }

  override forward(
    x: Value,
    root: TgpuRoot,
    uploads: Value[],
    config: DecoderConfig,
    previous: Lfm3AudioAttentionCache | undefined,
    position: number,
  ): { output: Value; cache?: Lfm3AudioAttentionCache } {
    const rows = x.shape.dims![0]!;
    const q = reshape(
      this.qNorm.forward(reshape(this.qProj.forward(x), [rows * config.heads, config.headDim])),
      [rows, config.heads * config.headDim],
    );
    const k = reshape(
      this.kNorm.forward(reshape(this.kProj.forward(x), [rows * config.kvHeads, config.headDim])),
      [rows, config.kvHeads * config.headDim],
    );
    const tables = ropeTables(position, rows, config.headDim, 1_000_000, 1);
    const cos = track(tensor(root, tables.cos, [rows, config.headDim]), uploads);
    const sin = track(tensor(root, tables.sin, [rows, config.headDim]), uploads);
    const rotatedK = rope(k, cos, sin, { headDim: config.headDim });
    const value = this.vProj.forward(x);
    const keys = previous ? cat([previous.key, rotatedK], 0) : rotatedK;
    const values = previous ? cat([previous.value, value], 0) : value;
    const attended = sdpa(
      mul(rope(q, cos, sin, { headDim: config.headDim }), 1 / Math.sqrt(config.headDim)),
      keys,
      values,
      {
        qHeads: config.heads,
        kvHeads: config.kvHeads,
        headDim: config.headDim,
        windowLeft: config.windowLeft,
        windowRight: config.causal ? 0 : Infinity,
        qPosOffset: previous?.key.shape.dims![0] ?? 0,
      },
    );
    return {
      output: this.outProj.forward(mul(attended, sigmoid(this.gateProj.forward(x)))),
      ...(config.causal
        ? {
            cache: {
              key: lastRows(keys, config.windowLeft + 1),
              value: lastRows(values, config.windowLeft + 1),
            },
          }
        : {}),
    };
  }
}

class DecoderLayer extends nn.Module<
  [Value, TgpuRoot, Value[], DecoderConfig, Lfm3AudioLayerCache | undefined, number],
  { hidden: Value; cache?: Lfm3AudioLayerCache }
> {
  readonly operatorNorm: nn.RMSNorm;
  readonly ffnNorm: nn.RMSNorm;
  readonly postOperatorNorm?: nn.RMSNorm;
  readonly postFfnNorm?: nn.RMSNorm;
  readonly attention?: DecoderAttention;
  readonly shortConv?: DecoderShortConv;
  readonly ffn: DecoderMlp;

  constructor(
    readonly hidden: number,
    readonly heads: number,
    readonly kvHeads: number,
    readonly headDim: number,
    readonly intermediate: number,
    readonly sandwich: boolean,
    useAttention: boolean,
  ) {
    super();
    this.operatorNorm = new nn.RMSNorm(hidden, 1e-5);
    this.ffnNorm = new nn.RMSNorm(hidden, 1e-5);
    if (sandwich) {
      this.postOperatorNorm = new nn.RMSNorm(hidden, 1e-5);
      this.postFfnNorm = new nn.RMSNorm(hidden, 1e-5);
    }
    if (useAttention) this.attention = new DecoderAttention(hidden, heads, kvHeads, headDim);
    else this.shortConv = new DecoderShortConv(hidden, 7);
    this.ffn = new DecoderMlp(hidden, intermediate);
  }

  override forward(
    x: Value,
    root: TgpuRoot,
    uploads: Value[],
    config: DecoderConfig,
    previous: Lfm3AudioLayerCache | undefined,
    position: number,
  ): { hidden: Value; cache?: Lfm3AudioLayerCache } {
    let operator: Value;
    let cache: Lfm3AudioLayerCache | undefined;
    if (this.attention) {
      const result = this.attention.forward(
        this.operatorNorm.forward(x),
        root,
        uploads,
        config,
        previous?.attention,
        position,
      );
      operator = result.output;
      if (result.cache) cache = { attention: result.cache };
    } else {
      const result = this.shortConv!.forward(this.operatorNorm.forward(x), previous?.convolution);
      operator = result.output;
      cache = { convolution: result.cache };
    }
    const withOperator = add(
      x,
      this.postOperatorNorm ? this.postOperatorNorm.forward(operator) : operator,
    );
    const ffn = this.ffn.forward(this.ffnNorm.forward(withOperator));
    return {
      hidden: add(withOperator, this.postFfnNorm ? this.postFfnNorm.forward(ffn) : ffn),
      ...(cache ? { cache } : {}),
    };
  }
}

class DecoderTrunk extends nn.Module<
  [Value, TgpuRoot, Value[], DecoderConfig, Lfm3AudioDecoderCache | undefined],
  { hidden: Value; cache?: Lfm3AudioDecoderCache }
> {
  readonly layers: nn.ModuleList<DecoderLayer>;
  readonly norm: nn.RMSNorm;

  constructor(config: DecoderConfig) {
    super();
    this.layers = new nn.ModuleList(
      Array.from(
        { length: config.layers },
        (_, layer) =>
          new DecoderLayer(
            config.hidden,
            config.heads,
            config.kvHeads,
            config.headDim,
            config.ffn,
            config.sandwich,
            config.attentionLayers.has(layer),
          ),
      ),
    );
    this.norm = new nn.RMSNorm(config.hidden, 1e-5);
  }

  override forward(
    x: Value,
    root: TgpuRoot,
    uploads: Value[],
    config: DecoderConfig,
    previous?: Lfm3AudioDecoderCache,
  ): { hidden: Value; cache?: Lfm3AudioDecoderCache } {
    let hidden = x;
    const layers: (Lfm3AudioLayerCache | undefined)[] = [];
    const position = previous?.position ?? 0;
    for (const [index, layer] of this.layers.items.entries()) {
      const result = layer.forward(
        hidden,
        root,
        uploads,
        config,
        previous?.layers[index],
        position,
      );
      hidden = result.hidden;
      layers.push(result.cache);
    }
    return {
      hidden: this.norm.forward(hidden),
      ...(config.causal ? { cache: { layers, position: position + x.shape.dims![0]! } } : {}),
    };
  }
}

const DEPTH_CONFIG = {
  hidden: DEPTH_HIDDEN,
  heads: DEPTH_HEADS,
  kvHeads: DEPTH_KV_HEADS,
  headDim: DEPTH_HEAD_DIM,
  windowLeft: Infinity,
  causal: false,
  ffn: DEPTH_FFN,
  sandwich: false,
  attentionLayers: new Set([0, 1, 2]),
  layers: 3,
} as const;
const CODEC_CONFIG = {
  hidden: CODEC_HIDDEN,
  heads: CODEC_HEADS,
  kvHeads: CODEC_KV_HEADS,
  headDim: CODEC_HEAD_DIM,
  windowLeft: 29,
  causal: true,
  ffn: CODEC_FFN,
  sandwich: true,
  attentionLayers: CODEC_LAYERS_WITH_ATTENTION,
  layers: 8,
} as const;

class AudioCodebook extends nn.Module {
  readonly embedding = new nn.Embedding(CODEBOOK_SIZE, DEPTH_HIDDEN, { device: 'cpu' });
  readonly norm = new nn.RMSNorm(DEPTH_HIDDEN, 1e-5);
  readonly head = new nn.Linear(DEPTH_HIDDEN, CODEBOOK_HEAD_ROWS, { bias: false });
}

class DepthDecoder extends nn.Module {
  readonly projIn = new nn.Linear(2_048, CODEBOOKS * DEPTH_HIDDEN, { bias: true });
  readonly maskEmbedding = new nn.Parameter({
    elems: DEPTH_HIDDEN,
    dtype: 'f32',
    dims: [DEPTH_HIDDEN],
  });
  readonly model = new DecoderTrunk(DEPTH_CONFIG);
  readonly books = new nn.ModuleList(Array.from({ length: CODEBOOKS }, () => new AudioCodebook()));

  forwardFrame(
    hidden: Value,
    codes: readonly number[],
    masked: readonly boolean[],
    root: TgpuRoot,
    uploads: Value[],
  ): Value[] {
    const projected = reshape(this.projIn.forward(hidden), [CODEBOOKS, DEPTH_HIDDEN]);
    const mask = reshape(this.maskEmbedding.value, [1, DEPTH_HIDDEN]);
    const slots: Value[] = [];
    for (let book = 0; book < CODEBOOKS; book++) {
      const base = slice(projected, 0, book, book + 1);
      const embedding = masked[book]
        ? mask
        : track(this.books.items[book]!.embedding.forward([codes[book]!], root), uploads);
      slots.push(add(base, embedding));
    }
    const trunk = this.model.forward(cat(slots, 0), root, uploads, DEPTH_CONFIG).hidden;
    return this.books.items.map((book, index) =>
      slice(
        book.head.forward(book.norm.forward(slice(trunk, 0, index, index + 1))),
        1,
        0,
        CODEBOOK_SIZE,
      ),
    );
  }
}

class LiquidCodec extends nn.Module {
  readonly codebook = new nn.Embedding(2_016, 4, { device: 'cpu' });
  readonly quantizerProjection = new nn.Linear(48, 256, { bias: true });
  readonly inputProjection = new nn.Linear(256, CODEC_HIDDEN, { bias: true });
  readonly model = new DecoderTrunk(CODEC_CONFIG);
  readonly outputProjection = new nn.Linear(CODEC_HIDDEN, CODEC_OUTPUT_ROWS, { bias: true });

  forwardFrame(
    codes: readonly number[],
    root: TgpuRoot,
    uploads: Value[],
    previous?: Lfm3AudioDecoderCache,
  ): { spectrum: Value; cache: Lfm3AudioDecoderCache } {
    if (
      codes.length !== CODEBOOKS ||
      codes.some((code) => !Number.isInteger(code) || code < 0 || code >= 2_016)
    ) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `LiquidCodec needs ${CODEBOOKS} codebook ids in [0, 2016)`,
      );
    }
    const codeValues = codes.map((code) => track(this.codebook.forward([code], root), uploads));
    const quantized = this.quantizerProjection.forward(reshape(cat(codeValues, 0), [1, 48]));
    const embedding = this.inputProjection.forward(quantized);
    const inputs = cat(
      Array.from({ length: CODEC_FRAMES_PER_TOKEN }, () => embedding),
      0,
    );
    const decoded = this.model.forward(inputs, root, uploads, CODEC_CONFIG, previous);
    if (!decoded.cache)
      throw new RunntimeError('EXECUTION_FAILED', 'LiquidCodec decoder lost its causal cache');
    return {
      spectrum: slice(this.outputProjection.forward(decoded.hidden), 1, 0, CODEC_SPECTRUM_WIDTH),
      cache: decoded.cache,
    };
  }
}

function mapAudioTensor(name: string): string | undefined {
  const downsample = /^a\.conv1d\.(0|2|3|5|6)\.(weight|bias)$/.exec(name);
  if (downsample) {
    const target = (
      {
        0: 'convIn',
        2: 'convMid',
        3: 'linMid',
        5: 'convEnd',
        6: 'linEnd',
      } as const
    )[downsample[1] as '0' | '2' | '3' | '5' | '6'];
    return `audioEncoder.downsample.${target}.${downsample[2]}`;
  }
  if (name.startsWith('a.pre_encode.out.')) {
    return `audioEncoder.downsample.linOut.${name.slice('a.pre_encode.out.'.length)}`;
  }
  const encoder = /^a\.blk\.(\d+)\.(.+)$/.exec(name);
  if (encoder) {
    const [, layer, suffix] = encoder;
    return `audioEncoder.lfm.layers.${layer}.${mapEncoderLayerSuffix(suffix!)}`;
  }
  if (name === 'a.post_ln.weight') return 'audioEncoder.lfm.norm.weight';
  if (name === 'mm.a.mlp.0.weight') return 'audioEncoder.projector.layerNorm.weight';
  if (name === 'mm.a.mlp.0.bias') return 'audioEncoder.projector.layerNorm.bias';
  if (name.startsWith('mm.a.mlp.1.'))
    return `audioEncoder.projector.linear1.${name.slice('mm.a.mlp.1.'.length)}`;
  if (name.startsWith('mm.a.mlp.3.'))
    return `audioEncoder.projector.linear2.${name.slice('mm.a.mlp.3.'.length)}`;
  const depth = /^a\.gen\.code\.blk\.(\d+)\.(.+)$/.exec(name);
  if (depth) return `depthDecoder.model.layers.${depth[1]}.${mapDepthLayerSuffix(depth[2]!)}`;
  if (name === 'a.gen.code.proj_in.weight') return 'depthDecoder.projIn.weight';
  if (name === 'a.gen.code.proj_in.bias') return 'depthDecoder.projIn.bias';
  if (name === 'a.gen.code.mask_embd') return 'depthDecoder.maskEmbedding';
  if (name === 'a.gen.code.output_norm.weight') return 'depthDecoder.model.norm.weight';
  if (name === 'a.gen.code.out_embd.weight') return 'audioOutputEmbedding.weight';
  if (name === 'a.gen.wav.quant.first.codebook.weight') return 'codec.codebook.weight';
  if (name.startsWith('a.gen.wav.quant.first.out_proj.')) {
    return `codec.quantizerProjection.${name.slice('a.gen.wav.quant.first.out_proj.'.length)}`;
  }
  if (name.startsWith('a.gen.wav.tfm.in_proj.')) {
    return `codec.inputProjection.${name.slice('a.gen.wav.tfm.in_proj.'.length)}`;
  }
  if (name.startsWith('a.gen.wav.tfm.out_proj.')) {
    return `codec.outputProjection.${name.slice('a.gen.wav.tfm.out_proj.'.length)}`;
  }
  if (name === 'a.gen.wav.tfm.output_norm.weight') return 'codec.model.norm.weight';
  const codec = /^a\.gen\.wav\.tfm\.blk\.(\d+)\.(.+)$/.exec(name);
  if (codec) return `codec.model.layers.${codec[1]}.${mapCodecLayerSuffix(codec[2]!)}`;
  return undefined;
}

function mapEncoderLayerSuffix(suffix: string): string {
  const projections: Record<string, string> = {
    'attn_q.weight': 'attention.qProj.weight',
    'attn_k.weight': 'attention.kProj.weight',
    'attn_v.weight': 'attention.vProj.weight',
    'attn_out.weight': 'attention.outProj.weight',
    'attn_q_norm.weight': 'attention.qNorm.weight',
    'attn_k_norm.weight': 'attention.kNorm.weight',
    'conv_dw.weight': 'shortConv.depthwise',
    'conv_pw1.weight': 'shortConv.inProj.weight',
    'conv_pw2.weight': 'shortConv.outProj.weight',
    'ffn_gate.weight': 'ffn.gate.weight',
    'ffn_up.weight': 'ffn.up.weight',
    'ffn_down.weight': 'ffn.down.weight',
    'ln1.weight': 'operatorNorm.weight',
    'ln2.weight': 'ffnNorm.weight',
  };
  const mapped = projections[suffix];
  if (!mapped) throw new Error(`unrecognized LFM3 audio encoder tensor suffix '${suffix}'`);
  return mapped;
}

function mapDepthLayerSuffix(suffix: string): string {
  const projections: Record<string, string> = {
    'attn_gate.weight': 'attention.gateProj.weight',
    'attn_q.weight': 'attention.qProj.weight',
    'attn_k.weight': 'attention.kProj.weight',
    'attn_v.weight': 'attention.vProj.weight',
    'attn_out.weight': 'attention.outProj.weight',
    'attn_q_norm.weight': 'attention.qNorm.weight',
    'attn_k_norm.weight': 'attention.kNorm.weight',
    'ffn_gate.weight': 'ffn.gate.weight',
    'ffn_up.weight': 'ffn.up.weight',
    'ffn_down.weight': 'ffn.down.weight',
    'ln1.weight': 'operatorNorm.weight',
    'ln2.weight': 'ffnNorm.weight',
  };
  const mapped = projections[suffix];
  if (!mapped) throw new Error(`unrecognized LFM3 depth-decoder tensor suffix '${suffix}'`);
  return mapped;
}

function mapCodecLayerSuffix(suffix: string): string {
  const projections: Record<string, string> = {
    'attn_gate.weight': 'attention.gateProj.weight',
    'attn_q.weight': 'attention.qProj.weight',
    'attn_k.weight': 'attention.kProj.weight',
    'attn_v.weight': 'attention.vProj.weight',
    'attn_out.weight': 'attention.outProj.weight',
    'attn_q_norm.weight': 'attention.qNorm.weight',
    'attn_k_norm.weight': 'attention.kNorm.weight',
    'conv_dw.weight': 'shortConv.depthwise',
    'conv_pw1.weight': 'shortConv.inProj.weight',
    'conv_pw2.weight': 'shortConv.outProj.weight',
    'ffn_gate.weight': 'ffn.gate.weight',
    'ffn_up.weight': 'ffn.up.weight',
    'ffn_down.weight': 'ffn.down.weight',
    'attn_post_norm.weight': 'postOperatorNorm.weight',
    'ffn_post_norm.weight': 'postFfnNorm.weight',
    'ln1.weight': 'operatorNorm.weight',
    'ln2.weight': 'ffnNorm.weight',
  };
  const mapped = projections[suffix];
  if (!mapped) throw new Error(`unrecognized LiquidCodec tensor suffix '${suffix}'`);
  return mapped;
}

function padQuantizedRows(tensor: QuantTensor, rows: number): QuantTensor {
  const originalRows = tensor.shape[0]!;
  const columns = tensor.shape[1]!;
  const groupsPerRow = columns / tensor.groupSize;
  const extraRows = rows - originalRows;
  if (
    tensor.shape.length !== 2 ||
    tensor.bits !== 8 ||
    columns % tensor.groupSize !== 0 ||
    extraRows < 0
  ) {
    throw new Error(`cannot pad quantized projection '${tensor.name}' to ${rows} rows`);
  }
  return quantTensor({
    name: `zero-padded(${tensor.name})`,
    shape: [rows, columns],
    bits: tensor.bits,
    groupSize: tensor.groupSize,
    byteLength: 0,
    sources: tensor.sources,
    async ints() {
      const source = await tensor.ints();
      const padded = new Int8Array(rows * columns);
      padded.set(source);
      return padded;
    },
    async scales() {
      const source = await tensor.scales();
      const padded =
        source instanceof Float32Array
          ? new Float32Array(rows * groupsPerRow)
          : new Uint16Array(rows * groupsPerRow);
      padded.set(source);
      return padded;
    },
  });
}

function padFloatVector(tensor: LazyTensor, rows: number): LazyTensor {
  if (
    tensor.kind === 'quant' ||
    tensor.shape.length !== 1 ||
    tensor.shape[0] !== CODEC_SPECTRUM_WIDTH
  ) {
    throw new Error(`cannot pad LiquidCodec output bias to ${rows} rows`);
  }
  return deriveFromTensors([rows], [tensor], async () => {
    const padded = new Float32Array(rows);
    padded.set(await tensor.f32());
    return padded;
  });
}

function permuteProjection(
  tensor: LazyStateDict['tensors'] extends Map<string, infer T> ? T : never,
  order: number[],
) {
  return permuteRows(tensor, order);
}

/** Full executable audio submodels loaded only from the pinned Q8_0 projector. */
export class Lfm3AudioModel extends nn.Module {
  readonly audioEncoder = new AudioEncoder();
  readonly depthDecoder = new DepthDecoder();
  readonly codec = new LiquidCodec();
  readonly audioOutputEmbedding = new nn.Embedding(CODEBOOKS * CODEBOOK_SIZE, 2_048, {
    device: 'cpu',
  });

  override transformStateDict(sd: LazyStateDict): void {
    const mapped = new Map<
      string,
      LazyStateDict['tensors'] extends Map<string, infer T> ? T : never
    >();
    const source = sd.tensors;
    const codeEmbeddings = source.get('a.gen.code.embd.weight');
    const codeHeadNorm = source.get('a.gen.code.head_norm.weight');
    if (!codeEmbeddings || !codeHeadNorm) {
      throw new Error('LFM3 audio projector is missing its depth codebook embeddings');
    }
    const codebooks = splitRows(codeEmbeddings, Array(CODEBOOKS).fill(1));
    const headNorms = splitRows(codeHeadNorm, Array(CODEBOOKS).fill(1));
    for (let book = 0; book < CODEBOOKS; book++) {
      const embedding = reshapeWeight(codebooks[book]!, [CODEBOOK_SIZE, DEPTH_HIDDEN]);
      const norm = reshapeWeight(headNorms[book]!, [DEPTH_HIDDEN]);
      mapped.set(`depthDecoder.books.${book}.embedding.weight`, embedding);
      if (embedding.kind !== 'quant') {
        throw new Error(`LFM3 depth codebook ${book} is not quantized`);
      }
      mapped.set(
        `depthDecoder.books.${book}.head.weight`,
        padQuantizedRows(embedding, CODEBOOK_HEAD_ROWS),
      );
      mapped.set(`depthDecoder.books.${book}.norm.weight`, norm);
    }
    const qOrder = rowOrder(AUDIO_HEADS, AUDIO_HEAD_DIM);
    const kOrder = rowOrder(AUDIO_KV_HEADS, AUDIO_HEAD_DIM);
    const depthQOrder = rowOrder(DEPTH_HEADS, DEPTH_HEAD_DIM);
    const depthKOrder = rowOrder(DEPTH_KV_HEADS, DEPTH_HEAD_DIM);
    const codecQOrder = rowOrder(CODEC_HEADS, CODEC_HEAD_DIM);
    const codecKOrder = rowOrder(CODEC_KV_HEADS, CODEC_HEAD_DIM);
    for (const [name, tensorValue] of source) {
      if (name === 'a.gen.code.embd.weight' || name === 'a.gen.code.head_norm.weight') continue;
      const target = mapAudioTensor(name);
      if (!target) {
        if (name.startsWith('a.') || name.startsWith('mm.a.')) {
          throw new Error(`unrecognized pinned LFM3 audio tensor '${name}'`);
        }
        continue;
      }
      let value = tensorValue;
      if (name === 'a.gen.wav.tfm.out_proj.weight') {
        if (value.kind !== 'quant') {
          throw new Error('LiquidCodec output projection must retain its published quantization');
        }
        value = padQuantizedRows(value, CODEC_OUTPUT_ROWS);
      } else if (name === 'a.gen.wav.tfm.out_proj.bias') {
        value = padFloatVector(value, CODEC_OUTPUT_ROWS);
      }
      if (name.startsWith('a.blk.')) {
        const layer = Number(name.split('.')[2]);
        if (name.endsWith('.attn_q.weight')) value = permuteProjection(value, qOrder);
        else if (name.endsWith('.attn_k.weight')) value = permuteProjection(value, kOrder);
        else if (name.endsWith('.attn_q_norm.weight'))
          value = permuteProjection(value, rowOrder(1, AUDIO_HEAD_DIM));
        else if (name.endsWith('.attn_k_norm.weight'))
          value = permuteProjection(value, rowOrder(1, AUDIO_HEAD_DIM));
        void layer;
      } else if (name.startsWith('a.gen.code.blk.')) {
        if (name.endsWith('.attn_q.weight')) value = permuteProjection(value, depthQOrder);
        else if (name.endsWith('.attn_k.weight')) value = permuteProjection(value, depthKOrder);
        else if (name.endsWith('.attn_q_norm.weight'))
          value = permuteProjection(value, rowOrder(1, DEPTH_HEAD_DIM));
        else if (name.endsWith('.attn_k_norm.weight'))
          value = permuteProjection(value, rowOrder(1, DEPTH_HEAD_DIM));
      } else if (name.startsWith('a.gen.wav.tfm.blk.')) {
        if (name.endsWith('.attn_q.weight')) value = permuteProjection(value, codecQOrder);
        else if (name.endsWith('.attn_k.weight')) value = permuteProjection(value, codecKOrder);
        else if (name.endsWith('.attn_q_norm.weight'))
          value = permuteProjection(value, rowOrder(1, CODEC_HEAD_DIM));
        else if (name.endsWith('.attn_k_norm.weight'))
          value = permuteProjection(value, rowOrder(1, CODEC_HEAD_DIM));
      }
      mapped.set(target, value);
    }
    sd.tensors.clear();
    for (const [name, value] of mapped) sd.tensors.set(name, value);
  }

  encodeAudio(audio: Float32Array, root: TgpuRoot, uploads: Value[]): Value {
    return this.audioEncoder.forward(audio, root, uploads);
  }

  forwardDepthFrame(
    hidden: Value,
    codes: readonly number[],
    masked: readonly boolean[],
    root: TgpuRoot,
    uploads: Value[],
  ): Value[] {
    return this.depthDecoder.forwardFrame(hidden, codes, masked, root, uploads);
  }

  audioOutEmbedding(codes: readonly number[], root: TgpuRoot, uploads: Value[]): Value {
    if (
      codes.length !== CODEBOOKS ||
      codes.some((code) => !Number.isInteger(code) || code < 0 || code >= CODEBOOK_SIZE)
    ) {
      throw new Error(`LFM3 audio output needs ${CODEBOOKS} codebook ids in [0, ${CODEBOOK_SIZE})`);
    }
    const ids = codes.map((code, book) => book * CODEBOOK_SIZE + code);
    const rows = track(this.audioOutputEmbedding.forward(ids, root), uploads);
    let output = slice(rows, 0, 0, 1);
    for (let book = 1; book < CODEBOOKS; book++)
      output = add(output, slice(rows, 0, book, book + 1));
    return output;
  }

  decodeSpeechCodes(
    codes: readonly number[],
    root: TgpuRoot,
    uploads: Value[],
    previous?: Lfm3AudioDecoderCache,
  ): { spectrum: Value; cache: Lfm3AudioDecoderCache } {
    return this.codec.forwardFrame(codes, root, uploads, previous);
  }
}

/** Audio-code EOS is 2016 in every codebook and is not a codec-table entry. */
export const LFM3_AUDIO_EOS_CODE = CODEBOOK_EOS;
export const LFM3_AUDIO_CODEBOOKS = CODEBOOKS;
export const LFM3_AUDIO_CODEBOOK_SIZE = CODEBOOK_SIZE;
export const LFM3_AUDIO_OUTPUT_FRAMES_PER_CODE = CODEC_FRAMES_PER_TOKEN;
export const lfm3AudioTokensForSamples = lfm3AudioTokenCount;
