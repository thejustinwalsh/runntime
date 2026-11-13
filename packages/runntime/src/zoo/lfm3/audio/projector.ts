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
  };
  readonly codec: {
    readonly sampleRate: 24_000;
    readonly hiddenSize: 512;
    readonly layers: 8;
  };
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
  },
  codec: {
    sampleRate: 24_000,
    hiddenSize: 512,
    layers: 8,
  },
};

/** Checks the pinned GGUF metadata and key tensor shapes before audio loading. */
export function validateLfm3AudioProjector(index: GgufIndex): Lfm3AudioProjectorConfig {
  const bad = (reason: string): never => {
    throw new RunntimeError('CHECKPOINT_MISMATCH', `LFM3-Omni audio projector: ${reason}`);
  };
  const metadata = index.metadata;
  if (metadata['general.architecture'] !== 'clip') bad('expected general.architecture=clip');
  if (metadata['clip.has_audio_encoder'] !== true) bad('projector has no audio encoder');
  if (metadata['clip.has_gen_audio_encoder'] !== true) bad('projector has no audio generator');
  if (metadata['clip.audio.projector_type'] !== 'lfm3a') bad('expected audio projector type lfm3a');
  if (metadata['clip.gen.audio.projector_type'] !== 'lfm3a_gen') {
    bad('expected generated-audio projector type lfm3a_gen');
  }

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
    'clip.gen.audio.sampling.temp': 0.4,
    'clip.gen.audio.sampling.penalty_repeat': 1.2,
    'clip.gen.audio.sampling.penalty_last_n': 64,
  };
  for (const [key, expected] of Object.entries(numbers)) {
    const actual = metadata[key];
    if (typeof actual !== 'number' || Math.fround(actual) !== Math.fround(expected)) {
      bad(`${key} is ${JSON.stringify(actual)}, expected ${expected}`);
    }
  }

  const tensors: ReadonlyArray<{
    name: string;
    shape: readonly number[];
    type: 'F32' | 'Q8_0';
  }> = [
    { name: 'a.pre_encode.out.weight', shape: [768, 4_096], type: 'Q8_0' },
    { name: 'a.blk.16.attn_q.weight', shape: [768, 768], type: 'Q8_0' },
    { name: 'a.gen.code.embd.weight', shape: [12, 2_017, 1_024], type: 'Q8_0' },
    { name: 'a.gen.code.blk.2.attn_q.weight', shape: [1_024, 1_024], type: 'Q8_0' },
    { name: 'a.gen.wav.quant.first.codebook.weight', shape: [2_016, 4], type: 'F32' },
    { name: 'a.gen.wav.tfm.blk.7.conv_dw.weight', shape: [512, 7], type: 'F32' },
    { name: 'a.gen.wav.tfm.out_proj.weight', shape: [1_282, 512], type: 'Q8_0' },
  ];
  for (const expected of tensors) {
    const actual = index.tensors.get(expected.name);
    if (!actual) return bad(`missing ${expected.name}`);
    if (actual.type !== expected.type || actual.shape.join() !== expected.shape.join()) {
      bad(
        `${expected.name} is ${actual.type} [${actual.shape.join(', ')}], expected ${expected.type} [${expected.shape.join(', ')}]`,
      );
    }
  }

  return CONFIG;
}
