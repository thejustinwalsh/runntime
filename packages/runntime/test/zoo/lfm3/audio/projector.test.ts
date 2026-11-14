/** Unit tests for the pinned LFM3-Omni audio-projector GGUF layout. */

import { expect, it } from 'vitest';
import { RunntimeError, type GgufIndex, type GgufTensorInfo } from '../../../../src/core/index.ts';
import {
  LFM3_AUDIO_PROJECTOR_TENSOR_LAYOUT,
  validateLfm3AudioProjector,
} from '../../../../src/zoo/lfm3/audio/projector.ts';

const metadata = {
  'general.architecture': 'clip',
  'general.file_type': 7,
  'general.quantization_version': 2,
  'clip.has_audio_encoder': true,
  'clip.audio.projector_type': 'lfm3a',
  'clip.audio.projection_dim': 2_048,
  'clip.audio.embedding_length': 768,
  'clip.audio.feed_forward_length': 1_792,
  'clip.audio.block_count': 17,
  'clip.audio.attention.head_count': 8,
  'clip.audio.attention.head_count_kv': 4,
  'clip.audio.attention.layer_norm_epsilon': 1e-5,
  'clip.audio.num_mel_bins': 128,
  'clip.audio.window_size': 32,
  'clip.has_gen_audio_encoder': true,
  'clip.use_silu': true,
  'clip.gen.audio.projector_type': 'lfm3a_gen',
  'clip.gen.audio.projection_dim': 2_048,
  'clip.gen.audio.embedding_length': 1_024,
  'clip.gen.audio.feed_forward_length': 2_816,
  'clip.gen.audio.block_count': 3,
  'clip.gen.audio.attention.head_count': 32,
  'clip.gen.audio.attention.head_count_kv': 8,
  'clip.gen.audio.attention.layer_norm_epsilon': 1e-5,
  'clip.gen.audio.sampling.temp': 0.4,
  'clip.gen.audio.sampling.penalty_repeat': 1.2,
  'clip.gen.audio.sampling.penalty_last_n': 64,
};

const tensor = (name: string, shape: number[], type: GgufTensorInfo['type']): GgufTensorInfo => ({
  name,
  shape,
  type,
  begin: 0,
  end: 4,
});

function projectorIndex(
  overrides: {
    metadata?: Record<string, unknown>;
    tensors?: Map<string, GgufTensorInfo>;
  } = {},
): GgufIndex {
  const tensors = new Map(
    LFM3_AUDIO_PROJECTOR_TENSOR_LAYOUT.map((entry) => [
      entry.name,
      tensor(entry.name, [...entry.shape], entry.type),
    ]),
  );
  for (const [name, value] of overrides.tensors ?? []) tensors.set(name, value);
  return {
    version: 3,
    metadata: { ...metadata, ...overrides.metadata },
    tensors,
    dataStart: 0,
    fileBytes: 4,
  };
}

it('returns the pinned encoder, depth-decoder and codec dimensions', () => {
  expect(LFM3_AUDIO_PROJECTOR_TENSOR_LAYOUT).toHaveLength(319);
  expect(validateLfm3AudioProjector(projectorIndex())).toEqual({
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
    codec: { sampleRate: 24_000, hiddenSize: 512, layers: 8 },
  });
});

it.each([
  { key: 'clip.audio.block_count', value: 16 },
  { key: 'clip.gen.audio.sampling.temp', value: 0 },
  { key: 'clip.gen.audio.sampling.penalty_repeat', value: 1.1 },
  { key: 'clip.gen.audio.sampling.penalty_last_n', value: 32 },
  { key: 'clip.has_gen_audio_encoder', value: false },
])('rejects mismatched projector metadata ($key)', ({ key, value }) => {
  expect(() => validateLfm3AudioProjector(projectorIndex({ metadata: { [key]: value } }))).toThrow(
    RunntimeError,
  );
});

it('rejects a missing or wrong-format codec tensor', () => {
  const index = projectorIndex();
  index.tensors.delete('a.gen.wav.tfm.out_proj.weight');
  expect(() => validateLfm3AudioProjector(index)).toThrow(
    /missing a\.gen\.wav\.tfm\.out_proj\.weight/,
  );

  expect(() =>
    validateLfm3AudioProjector(
      projectorIndex({
        tensors: new Map([
          [
            'a.gen.wav.tfm.out_proj.weight',
            tensor('a.gen.wav.tfm.out_proj.weight', [1_282, 512], 'F32'),
          ],
        ]),
      }),
    ),
  ).toThrow(/expected Q8_0/);
});

it('rejects a missing encoder tensor and an unexpected audio tensor', () => {
  const missing = projectorIndex();
  missing.tensors.delete('a.blk.16.attn_k_norm.weight');
  expect(() => validateLfm3AudioProjector(missing)).toThrow(
    /missing a\.blk\.16\.attn_k_norm\.weight/,
  );

  const unexpected = projectorIndex();
  const extra = tensor('a.unexpected.weight', [1], 'F32');
  unexpected.tensors.set(extra.name, extra);
  expect(() => validateLfm3AudioProjector(unexpected)).toThrow(/unexpected audio tensor/);
});

it('rejects unsupported GGUF versions and unexpected audio metadata', () => {
  expect(() => validateLfm3AudioProjector({ ...projectorIndex(), version: 2 })).toThrow(
    /expected GGUF version 3/,
  );
  expect(() =>
    validateLfm3AudioProjector(projectorIndex({ metadata: { 'clip.use_silu': false } })),
  ).toThrow(/clip\.use_silu=true/);
});
