/** LFM3-Omni's 16 kHz log-mel frontend and audio token dimensions. */

import { RunntimeError } from '../../../core/error.ts';

export const LFM3_AUDIO_SAMPLE_RATE = 16_000;
export const LFM3_AUDIO_FFT_SIZE = 400;
export const LFM3_AUDIO_HOP_SIZE = 160;
export const LFM3_AUDIO_MEL_BINS = 128;
export const LFM3_AUDIO_MAX_SECONDS = 32;

const MIN_PADDED_SAMPLES = 2 * LFM3_AUDIO_SAMPLE_RATE;
const MAX_AUDIO_SAMPLES = LFM3_AUDIO_MAX_SECONDS * LFM3_AUDIO_SAMPLE_RATE;
const LOG_FLOOR = 1e-12;
const LOG_POST_ADD = 4;
const LOG_POST_SCALE = 1 / 8;
const LOG10_E = 0.43429448190325176;
const FFT_BINS = LFM3_AUDIO_FFT_SIZE / 2 + 1;

interface FftPlan {
  readonly size: number;
  readonly radix: number;
  readonly subSize: number;
  readonly child?: FftPlan;
  readonly twiddleReal: Float64Array;
  readonly twiddleImag: Float64Array;
}

interface ComplexArray {
  readonly real: Float64Array;
  readonly imag: Float64Array;
}

const FFT_PLAN = makeFftPlan(LFM3_AUDIO_FFT_SIZE);
const WINDOW = periodicHannWindow(LFM3_AUDIO_FFT_SIZE);
const MEL_FILTERS = slaneyMelFilterbank(
  LFM3_AUDIO_MEL_BINS,
  LFM3_AUDIO_FFT_SIZE,
  LFM3_AUDIO_SAMPLE_RATE,
);

function makeFftPlan(size: number): FftPlan {
  if (size === 1) {
    return {
      size,
      radix: 1,
      subSize: 1,
      twiddleReal: Float64Array.of(1),
      twiddleImag: Float64Array.of(0),
    };
  }
  const radix = size % 2 === 0 ? 2 : size % 5 === 0 ? 5 : size;
  const subSize = size / radix;
  const child = radix === size ? undefined : makeFftPlan(subSize);
  const twiddleReal = new Float64Array(size * radix);
  const twiddleImag = new Float64Array(size * radix);
  for (let k = 0; k < size; k++) {
    for (let r = 0; r < radix; r++) {
      const angle = (2 * Math.PI * k * r) / size;
      twiddleReal[k * radix + r] = Math.cos(angle);
      twiddleImag[k * radix + r] = -Math.sin(angle);
    }
  }
  return { size, radix, subSize, child, twiddleReal, twiddleImag };
}

function fft(real: Float64Array, imag: Float64Array, plan: FftPlan): ComplexArray {
  if (plan.size === 1) return { real, imag };
  if (!plan.child) return directDft(real, imag, plan);
  const parts: ComplexArray[] = [];
  for (let r = 0; r < plan.radix; r++) {
    const partReal = new Float64Array(plan.subSize);
    const partImag = new Float64Array(plan.subSize);
    for (let index = 0; index < plan.subSize; index++) {
      partReal[index] = real[r + plan.radix * index]!;
      partImag[index] = imag[r + plan.radix * index]!;
    }
    parts.push(fft(partReal, partImag, plan.child));
  }

  const outputReal = new Float64Array(plan.size);
  const outputImag = new Float64Array(plan.size);
  for (let k = 0; k < plan.size; k++) {
    const subIndex = k % plan.subSize;
    for (let r = 0; r < plan.radix; r++) {
      const source = parts[r]!;
      const sourceReal = source.real[subIndex]!;
      const sourceImag = source.imag[subIndex]!;
      const twiddle = k * plan.radix + r;
      const twiddleReal = plan.twiddleReal[twiddle]!;
      const twiddleImag = plan.twiddleImag[twiddle]!;
      outputReal[k] = outputReal[k]! + sourceReal * twiddleReal - sourceImag * twiddleImag;
      outputImag[k] = outputImag[k]! + sourceReal * twiddleImag + sourceImag * twiddleReal;
    }
  }
  return { real: outputReal, imag: outputImag };
}

function directDft(real: Float64Array, imag: Float64Array, plan: FftPlan): ComplexArray {
  const outputReal = new Float64Array(plan.size);
  const outputImag = new Float64Array(plan.size);
  for (let k = 0; k < plan.size; k++) {
    for (let n = 0; n < plan.size; n++) {
      const angle = (2 * Math.PI * k * n) / plan.size;
      const cosine = Math.cos(angle);
      const sine = -Math.sin(angle);
      outputReal[k] = outputReal[k]! + real[n]! * cosine - imag[n]! * sine;
      outputImag[k] = outputImag[k]! + real[n]! * sine + imag[n]! * cosine;
    }
  }
  return { real: outputReal, imag: outputImag };
}

function periodicHannWindow(size: number): Float64Array {
  return Float64Array.from(
    { length: size },
    (_, index) => 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / size),
  );
}

function hzToMel(frequency: number): number {
  return frequency >= 1_000
    ? 15 + (Math.log(frequency / 1_000) * 27) / Math.log(6.4)
    : (3 * frequency) / 200;
}

function melToHz(mel: number): number {
  return mel >= 15 ? 1_000 * Math.exp((Math.log(6.4) * (mel - 15)) / 27) : (200 * mel) / 3;
}

function slaneyMelFilterbank(mels: number, fftSize: number, sampleRate: number): Float32Array[] {
  const frequencies = fftSize / 2 + 1;
  const maxFrequency = sampleRate / 2;
  const firstMel = hzToMel(0);
  const lastMel = hzToMel(maxFrequency);
  const edges = Array.from({ length: mels + 2 }, (_, index) =>
    melToHz(firstMel + ((lastMel - firstMel) * index) / (mels + 1)),
  );
  const filters: Float32Array[] = [];
  for (let mel = 0; mel < mels; mel++) {
    const lower = edges[mel]!;
    const center = edges[mel + 1]!;
    const upper = edges[mel + 2]!;
    const scale = 2 / (upper - lower);
    const weights = new Float32Array(frequencies);
    for (let bin = 0; bin < frequencies; bin++) {
      const frequency = (bin * maxFrequency) / (frequencies - 1);
      const rising = (frequency - lower) / (center - lower);
      const falling = (upper - frequency) / (upper - center);
      weights[bin] = Math.max(0, Math.min(rising, falling)) * scale;
    }
    filters.push(weights);
  }
  return filters;
}

/** Number of 400-sample log-mel frames after right-padding to the 160-sample hop. */
export function lfm3AudioFrameCount(samples: number): number {
  if (!Number.isInteger(samples) || samples < 0) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      'audio sample count must be a non-negative integer',
    );
  }
  if (samples <= LFM3_AUDIO_FFT_SIZE) return 1;
  return 1 + Math.ceil((samples - LFM3_AUDIO_FFT_SIZE) / LFM3_AUDIO_HOP_SIZE);
}

/** Number of audio placeholder tokens produced by the 3×2 convolutional stride. */
export function lfm3AudioTokenCount(samples: number): number {
  return Math.ceil(lfm3AudioFrameCount(samples) / 8);
}

/** Padded waveform length required by Liquid's 2–32 second graph variants. */
export function lfm3AudioPaddedLength(samples: number): number {
  if (!Number.isInteger(samples) || samples <= 0) {
    throw new RunntimeError('INVALID_ARGUMENT', 'audio must contain at least one sample');
  }
  if (samples > MAX_AUDIO_SAMPLES) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `LFM3 audio is limited to ${LFM3_AUDIO_MAX_SECONDS} seconds`,
    );
  }
  let padded = MIN_PADDED_SAMPLES;
  while (padded < samples) padded *= 2;
  return padded;
}

/** Computes the reference 128-band Slaney log-mel spectrogram in [mel, frame] order. */
export function lfm3LogMelSpectrogram(audio: Float32Array): Float32Array {
  const frames = lfm3AudioFrameCount(audio.length);
  const output = new Float32Array(LFM3_AUDIO_MEL_BINS * frames);
  const real = new Float64Array(LFM3_AUDIO_FFT_SIZE);
  const imag = new Float64Array(LFM3_AUDIO_FFT_SIZE);
  for (let frame = 0; frame < frames; frame++) {
    const begin = frame * LFM3_AUDIO_HOP_SIZE;
    for (let sample = 0; sample < LFM3_AUDIO_FFT_SIZE; sample++) {
      real[sample] = (audio[begin + sample] ?? 0) * WINDOW[sample]!;
      imag[sample] = 0;
    }
    const spectrum = fft(real, imag, FFT_PLAN);
    const powers = new Float64Array(FFT_BINS);
    for (let bin = 0; bin < FFT_BINS; bin++) {
      powers[bin] = spectrum.real[bin]! ** 2 + spectrum.imag[bin]! ** 2;
    }
    for (let mel = 0; mel < LFM3_AUDIO_MEL_BINS; mel++) {
      const filter = MEL_FILTERS[mel]!;
      let energy = 0;
      for (let bin = 0; bin < FFT_BINS; bin++) energy += powers[bin]! * filter[bin]!;
      output[mel * frames + frame] =
        (Math.log(Math.max(energy, LOG_FLOOR)) * LOG10_E + LOG_POST_ADD) * LOG_POST_SCALE;
    }
  }
  return output;
}
