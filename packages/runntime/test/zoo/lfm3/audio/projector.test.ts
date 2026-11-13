/** Unit tests for the pinned LFM3-Omni audio-projector GGUF layout. */

import { expect, it } from 'vitest';
import { RunntimeError, type GgufIndex, type GgufTensorInfo } from '../../../../src/core/index.ts';
import { validateLfm3AudioProjector } from '../../../../src/zoo/lfm3/audio/projector.ts';

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
    [
      tensor('a.pre_encode.out.weight', [768, 4_096], 'Q8_0'),
      tensor('a.blk.16.attn_q.weight', [768, 768], 'Q8_0'),
      tensor('a.gen.code.embd.weight', [12, 2_017, 1_024], 'Q8_0'),
      tensor('a.gen.code.blk.2.attn_q.weight', [1_024, 1_024], 'Q8_0'),
      tensor('a.gen.wav.quant.first.codebook.weight', [2_016, 4], 'F32'),
      tensor('a.gen.wav.tfm.blk.7.conv_dw.weight', [512, 7], 'F32'),
      tensor('a.gen.wav.tfm.out_proj.weight', [1_282, 512], 'Q8_0'),
    ].map((entry) => [entry.name, entry]),
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
    },
    codec: { sampleRate: 24_000, hiddenSize: 512, layers: 8 },
  });
});

it.each([
  { key: 'clip.audio.block_count', value: 16 },
  { key: 'clip.gen.audio.sampling.temp', value: 0 },
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
