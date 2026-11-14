import { RunntimeError } from '../../../core/index.ts';

const UINT32_RANGE = 0x1_0000_0000;
const PCG_MULTIPLIER = 747_796_405;
const PCG_INCREMENT = 2_891_336_453;
const PCG_OUTPUT_MULTIPLIER = 277_803_737;

export interface Lfm3DepthSamplingSettings {
  readonly temperature: number;
  readonly repetitionPenalty: number;
  readonly repetitionWindow: number;
}

export interface Lfm3DepthSampler {
  readonly seed: number;
  readonly nextRandom: () => number;
}

/** Mirrors Liquid's seeded draw stream; its first value seeds per-code Gumbel draws. */
export function createLfm3DepthSampler(seed: number): Lfm3DepthSampler {
  let state = seed >>> 0;
  const nextUint32 = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return (value ^ (value >>> 14)) >>> 0;
  };
  const samplerSeed = nextUint32();
  return { seed: samplerSeed, nextRandom: () => nextUint32() / UINT32_RANGE };
}

/** Liquid's per-frame Fisher-Yates MaskGIT codebook order. */
export function lfm3MaskgitOrder(codebooks: number, nextRandom: () => number): number[] {
  const order = Array.from({ length: codebooks }, (_, index) => index);
  for (let index = order.length - 1; index > 0; index--) {
    const swap = Math.floor(nextRandom() * (index + 1));
    [order[index], order[swap]] = [order[swap]!, order[index]!];
  }
  return order;
}

/** Distributes codebooks evenly across the requested refinement passes. */
export function lfm3MaskgitRevealSizes(codebooks: number, passes: number): number[] {
  const sizes: number[] = [];
  let remaining = codebooks;
  for (let pass = 0; pass < passes && remaining > 0; pass++) {
    const count = Math.ceil(remaining / (passes - pass));
    sizes.push(count);
    remaining -= count;
  }
  return sizes;
}

/** Liquid's PCG-hashed Gumbel-max sampler for one codebook at one audio frame. */
export function sampleLfm3DepthCode(input: {
  readonly logits: Float32Array;
  readonly codebookSize: number;
  readonly codebook: number;
  readonly frame: number;
  readonly samplerSeed: number;
  readonly temperature: number;
  readonly repetitionPenalty: number;
  readonly previousCodes: readonly number[];
}): number {
  const {
    logits,
    codebookSize,
    codebook,
    frame,
    samplerSeed,
    temperature,
    repetitionPenalty,
    previousCodes,
  } = input;
  if (logits.length !== codebookSize) {
    throw new RunntimeError(
      'EXECUTION_FAILED',
      `LFM3 depth decoder returned ${logits.length} logits for codebook ${codebook}`,
    );
  }

  const samplerStream = pcgHash(pcgHash(frame) ^ samplerSeed);
  const temperature32 = Math.fround(temperature);
  const penalty32 = Math.fround(repetitionPenalty);
  let best = 0;
  let bestScore = -Infinity;
  for (let code = 0; code < logits.length; code++) {
    const value = logits[code]!;
    if (!Number.isFinite(value)) {
      throw new RunntimeError(
        'EXECUTION_FAILED',
        `LFM3 depth decoder produced non-finite logit ${code}`,
      );
    }
    const adjusted = previousCodes.includes(code)
      ? Math.fround(value < 0 ? Math.fround(value * penalty32) : Math.fround(value / penalty32))
      : value;
    let score = adjusted;
    if (temperature32 > 0) {
      const flattenedCode = (codebook * codebookSize + code) >>> 0;
      const randomBits = pcgHash(samplerStream ^ flattenedCode) >>> 8;
      const random = Math.fround(Math.fround(randomBits) + 0.5) * (1 / 16_777_216);
      const gumbel = Math.fround(Math.log(-Math.log(random)));
      score = Math.fround(Math.fround(adjusted / temperature32) - gumbel);
    }
    if (score > bestScore) {
      best = code;
      bestScore = score;
    }
  }
  return best;
}

function pcgHash(input: number): number {
  const value = input >>> 0;
  const state = (Math.imul(value, PCG_MULTIPLIER) + PCG_INCREMENT) >>> 0;
  const shift = (state >>> 28) + 4;
  const word = Math.imul(((state >>> shift) ^ state) >>> 0, PCG_OUTPUT_MULTIPLIER) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}
