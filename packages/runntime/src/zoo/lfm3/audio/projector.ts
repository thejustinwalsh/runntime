/** Pinned LFM3-Omni audio-projector GGUF metadata and tensor layout. */

import { RunntimeError, type GgufIndex } from '../../../core/index.ts';

export interface Lfm3AudioProjectorConfig {
  readonly encoder: {
    readonly sampleRate: 16_000;
    readonly fftSize: 400;
    readonly hopSize: 160;
    readonly melBins: 128;
    readonly hiddenSize: 768;
    readonly feedForwardSize: 1_792;
    readonly layers: 17;
    readonly attentionHeads: 8;
    readonly keyValueHeads: 4;
    readonly attentionWindow: 32;
    readonly outputSize: 2_048;
  };
  readonly depthDecoder: {
    readonly hiddenSize: 1_024;
    readonly feedForwardSize: 2_816;
    readonly layers: 3;
    readonly attentionHeads: 32;
    readonly keyValueHeads: 8;
    readonly codebooks: 12;
    readonly codebookSize: 2_017;
    readonly sampling: {
      readonly temperature: number;
      readonly repetitionPenalty: number;
      readonly repetitionWindow: number;
    };
  };
  readonly codec: {
    readonly sampleRate: 24_000;
    readonly hiddenSize: 512;
    readonly layers: 8;
  };
}

export interface Lfm3AudioProjectorTensorLayout {
  readonly name: string;
  readonly shape: readonly number[];
  readonly type: 'F32' | 'Q8_0';
}

const CONFIG: Lfm3AudioProjectorConfig = {
  encoder: {
    sampleRate: 16_000,
    fftSize: 400,
    hopSize: 160,
    melBins: 128,
    hiddenSize: 768,
    feedForwardSize: 1_792,
    layers: 17,
    attentionHeads: 8,
    keyValueHeads: 4,
    attentionWindow: 32,
    outputSize: 2_048,
  },
  depthDecoder: {
    hiddenSize: 1_024,
    feedForwardSize: 2_816,
    layers: 3,
    attentionHeads: 32,
    keyValueHeads: 8,
    codebooks: 12,
    codebookSize: 2_017,
    sampling: {
      temperature: 0.4,
      repetitionPenalty: 1.2,
      repetitionWindow: 64,
    },
  },
  codec: {
    sampleRate: 24_000,
    hiddenSize: 512,
    layers: 8,
  },
};

const AUDIO_ENCODER_ATTENTION_LAYERS = new Set([2, 5, 8, 11, 14, 15, 16]);
const AUDIO_GENERATOR_ATTENTION_LAYERS = new Set([2, 4, 6]);

function audioProjectorTensorLayout(): Lfm3AudioProjectorTensorLayout[] {
  const tensors: Lfm3AudioProjectorTensorLayout[] = [];
  const add = (name: string, shape: readonly number[], type: 'F32' | 'Q8_0'): void => {
    tensors.push({ name, shape, type });
  };
  const linear = (name: string, output: number, input: number, bias = false): void => {
    add(`${name}.weight`, [output, input], 'Q8_0');
    if (bias) add(`${name}.bias`, [output], 'F32');
  };

  for (let layer = 0; layer < CONFIG.encoder.layers; layer++) {
    const prefix = `a.blk.${layer}`;
    if (AUDIO_ENCODER_ATTENTION_LAYERS.has(layer)) {
      linear(`${prefix}.attn_q`, 768, 768);
      linear(`${prefix}.attn_k`, 384, 768);
      add(`${prefix}.attn_q_norm.weight`, [96], 'F32');
      add(`${prefix}.attn_k_norm.weight`, [96], 'F32');
      linear(`${prefix}.attn_v`, 384, 768);
      linear(`${prefix}.attn_out`, 768, 768);
    } else {
      add(`${prefix}.conv_dw.weight`, [768, 3], 'F32');
      linear(`${prefix}.conv_pw1`, 2_304, 768);
      linear(`${prefix}.conv_pw2`, 768, 768);
    }
    linear(`${prefix}.ffn_gate`, 1_792, 768);
    linear(`${prefix}.ffn_up`, 1_792, 768);
    linear(`${prefix}.ffn_down`, 768, 1_792);
    add(`${prefix}.ln1.weight`, [768], 'F32');
    add(`${prefix}.ln2.weight`, [768], 'F32');
  }
  add('a.post_ln.weight', [768], 'F32');

  for (const [layer, inputChannels, outputChannels, kernel, groups] of [
    [0, 1, 256, 3, 1],
    [2, 256, 256, 3, 256],
    [3, 256, 256, 1, 1],
    [5, 256, 256, 3, 256],
    [6, 256, 256, 1, 1],
  ] as const) {
    add(
      `a.conv1d.${layer}.weight`,
      [outputChannels, inputChannels / groups, kernel, kernel],
      'F32',
    );
    add(`a.conv1d.${layer}.bias`, [outputChannels, 1, 1], 'F32');
  }
  add('mm.a.mlp.0.weight', [768], 'F32');
  add('mm.a.mlp.0.bias', [768], 'F32');
  linear('mm.a.mlp.1', 2_048, 768, true);
  linear('mm.a.mlp.3', 2_048, 2_048, true);
  linear('a.pre_encode.out', 768, 4_096, true);

  for (let layer = 0; layer < CONFIG.depthDecoder.layers; layer++) {
    const prefix = `a.gen.code.blk.${layer}`;
    linear(`${prefix}.attn_gate`, 32, 1_024);
    linear(`${prefix}.attn_q`, 1_024, 1_024);
    linear(`${prefix}.attn_k`, 256, 1_024);
    add(`${prefix}.attn_q_norm.weight`, [32], 'F32');
    add(`${prefix}.attn_k_norm.weight`, [32], 'F32');
    linear(`${prefix}.attn_v`, 256, 1_024);
    linear(`${prefix}.attn_out`, 1_024, 1_024);
    linear(`${prefix}.ffn_gate`, 2_816, 1_024);
    linear(`${prefix}.ffn_up`, 2_816, 1_024);
    linear(`${prefix}.ffn_down`, 1_024, 2_816);
    add(`${prefix}.ln1.weight`, [1_024], 'F32');
    add(`${prefix}.ln2.weight`, [1_024], 'F32');
  }
  add('a.gen.code.embd.weight', [12, 2_017, 1_024], 'Q8_0');
  add('a.gen.code.head_norm.weight', [12, 1_024], 'F32');
  add('a.gen.code.mask_embd', [1_024], 'F32');
  add('a.gen.code.out_embd.weight', [24_204, 2_048], 'Q8_0');
  add('a.gen.code.output_norm.weight', [1_024], 'F32');
  linear('a.gen.code.proj_in', 12_288, 2_048, true);

  add('a.gen.wav.quant.first.codebook.weight', [2_016, 4], 'F32');
  add('a.gen.wav.quant.first.out_proj.weight', [256, 48], 'F32');
  add('a.gen.wav.quant.first.out_proj.bias', [256], 'F32');
  linear('a.gen.wav.tfm.in_proj', 512, 256, true);
  linear('a.gen.wav.tfm.out_proj', 1_282, 512, true);
  for (let layer = 0; layer < CONFIG.codec.layers; layer++) {
    const prefix = `a.gen.wav.tfm.blk.${layer}`;
    if (AUDIO_GENERATOR_ATTENTION_LAYERS.has(layer)) {
      linear(`${prefix}.attn_gate`, 8, 512);
      linear(`${prefix}.attn_q`, 512, 512);
      linear(`${prefix}.attn_k`, 128, 512);
      add(`${prefix}.attn_q_norm.weight`, [64], 'F32');
      add(`${prefix}.attn_k_norm.weight`, [64], 'F32');
      linear(`${prefix}.attn_v`, 128, 512);
      linear(`${prefix}.attn_out`, 512, 512);
    } else {
      add(`${prefix}.conv_dw.weight`, [512, 7], 'F32');
      linear(`${prefix}.conv_pw1`, 1_536, 512);
      linear(`${prefix}.conv_pw2`, 512, 512);
    }
    linear(`${prefix}.ffn_gate`, 2_304, 512);
    linear(`${prefix}.ffn_up`, 2_304, 512);
    linear(`${prefix}.ffn_down`, 512, 2_304);
    add(`${prefix}.attn_post_norm.weight`, [512], 'F32');
    add(`${prefix}.ffn_post_norm.weight`, [512], 'F32');
    add(`${prefix}.ln1.weight`, [512], 'F32');
    add(`${prefix}.ln2.weight`, [512], 'F32');
  }
  add('a.gen.wav.tfm.output_norm.weight', [512], 'F32');

  return tensors;
}

/** Exact audio-related tensor inventory in the pinned Q8_0 projector GGUF. */
export const LFM3_AUDIO_PROJECTOR_TENSOR_LAYOUT = audioProjectorTensorLayout();

/** Checks the pinned GGUF metadata and key tensor shapes before audio loading. */
export function validateLfm3AudioProjector(index: GgufIndex): Lfm3AudioProjectorConfig {
  const bad = (reason: string): never => {
    throw new RunntimeError('CHECKPOINT_MISMATCH', `LFM3-Omni audio projector: ${reason}`);
  };
  if (index.version !== 3) bad(`expected GGUF version 3, got ${index.version}`);
  const metadata = index.metadata;
  if (metadata['general.architecture'] !== 'clip') bad('expected general.architecture=clip');
  if (metadata['clip.has_audio_encoder'] !== true) bad('projector has no audio encoder');
  if (metadata['clip.has_gen_audio_encoder'] !== true) bad('projector has no audio generator');
  if (metadata['clip.audio.projector_type'] !== 'lfm3a') bad('expected audio projector type lfm3a');
  if (metadata['clip.gen.audio.projector_type'] !== 'lfm3a_gen') {
    bad('expected generated-audio projector type lfm3a_gen');
  }
  if (metadata['clip.use_silu'] !== true) bad('expected clip.use_silu=true');

  const numbers: Readonly<Record<string, number>> = {
    'general.file_type': 7,
    'general.quantization_version': 2,
    'clip.audio.projection_dim': CONFIG.encoder.outputSize,
    'clip.audio.embedding_length': CONFIG.encoder.hiddenSize,
    'clip.audio.feed_forward_length': CONFIG.encoder.feedForwardSize,
    'clip.audio.block_count': CONFIG.encoder.layers,
    'clip.audio.attention.head_count': CONFIG.encoder.attentionHeads,
    'clip.audio.attention.head_count_kv': CONFIG.encoder.keyValueHeads,
    'clip.audio.attention.layer_norm_epsilon': 1e-5,
    'clip.audio.num_mel_bins': CONFIG.encoder.melBins,
    'clip.audio.window_size': CONFIG.encoder.attentionWindow,
    'clip.gen.audio.projection_dim': CONFIG.encoder.outputSize,
    'clip.gen.audio.embedding_length': CONFIG.depthDecoder.hiddenSize,
    'clip.gen.audio.feed_forward_length': CONFIG.depthDecoder.feedForwardSize,
    'clip.gen.audio.block_count': CONFIG.depthDecoder.layers,
    'clip.gen.audio.attention.head_count': CONFIG.depthDecoder.attentionHeads,
    'clip.gen.audio.attention.head_count_kv': CONFIG.depthDecoder.keyValueHeads,
    'clip.gen.audio.attention.layer_norm_epsilon': 1e-5,
    'clip.gen.audio.sampling.temp': CONFIG.depthDecoder.sampling.temperature,
    'clip.gen.audio.sampling.penalty_repeat': CONFIG.depthDecoder.sampling.repetitionPenalty,
    'clip.gen.audio.sampling.penalty_last_n': CONFIG.depthDecoder.sampling.repetitionWindow,
  };
  for (const [key, expected] of Object.entries(numbers)) {
    const actual = metadata[key];
    if (typeof actual !== 'number' || Math.fround(actual) !== Math.fround(expected)) {
      bad(`${key} is ${JSON.stringify(actual)}, expected ${expected}`);
    }
  }

  const expectedNames = new Set<string>();
  for (const expected of LFM3_AUDIO_PROJECTOR_TENSOR_LAYOUT) {
    expectedNames.add(expected.name);
    const actual = index.tensors.get(expected.name);
    if (!actual) return bad(`missing ${expected.name}`);
    if (actual.type !== expected.type || actual.shape.join() !== expected.shape.join()) {
      bad(
        `${expected.name} is ${actual.type} [${actual.shape.join(', ')}], expected ${expected.type} [${expected.shape.join(', ')}]`,
      );
    }
  }
  for (const name of index.tensors.keys()) {
    if ((name.startsWith('a.') || name.startsWith('mm.a.')) && !expectedNames.has(name)) {
      bad(`unexpected audio tensor ${name}`);
    }
  }

  return CONFIG;
}
