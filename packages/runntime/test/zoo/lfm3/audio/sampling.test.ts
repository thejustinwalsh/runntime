import { describe, expect, it } from 'vitest';
import {
  createLfm3DepthSampler,
  lfm3MaskgitOrder,
  lfm3MaskgitRevealSizes,
  sampleLfm3DepthCode,
} from '../../../../src/zoo/lfm3/audio/sampling.ts';

describe('LFM3 depth sampling', () => {
  it('consumes the first seeded draw for the sampler key before MaskGIT permutation draws', () => {
    const sampler = createLfm3DepthSampler(1_234);

    // Golden values follow the pinned engine's dT seed stream and pT shuffle.
    expect(sampler.seed).toBe(314_799_534);
    expect(lfm3MaskgitOrder(12, sampler.nextRandom)).toEqual([
      2, 3, 5, 7, 6, 4, 1, 11, 0, 10, 9, 8,
    ]);
  });

  it('uses the pinned [4, 4, 4] schedule for the default three passes', () => {
    expect(lfm3MaskgitRevealSizes(12, 3)).toEqual([4, 4, 4]);
    expect(lfm3MaskgitRevealSizes(12, 1)).toEqual([12]);
  });

  it('selects a code with the reference PCG-hashed Gumbel-max draw', () => {
    const sampler = createLfm3DepthSampler(1_234);
    const logits = Float32Array.from({ length: 8 }, (_, index) => (index - 3) / 5);

    expect(
      sampleLfm3DepthCode({
        logits,
        codebookSize: 8,
        codebook: 2,
        frame: 7,
        samplerSeed: sampler.seed,
        temperature: 0.4,
        repetitionPenalty: 1.2,
        previousCodes: [3, 5],
      }),
    ).toBe(7);
  });

  it('uses greedy adjusted logits at zero temperature while retaining the codebook key', () => {
    const sampler = createLfm3DepthSampler(7);

    expect(
      sampleLfm3DepthCode({
        logits: Float32Array.of(1, 2, 1.5),
        codebookSize: 3,
        codebook: 0,
        frame: 1,
        samplerSeed: sampler.seed,
        temperature: 0,
        repetitionPenalty: 1.2,
        previousCodes: [1],
      }),
    ).toBe(1);
  });
});
