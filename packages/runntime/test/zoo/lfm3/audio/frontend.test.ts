/** Unit tests for the LFM3-Omni audio frontend dimensions and spectrum. */

import { describe, expect, it } from 'vitest';
import {
  LFM3_AUDIO_FFT_SIZE,
  LFM3_AUDIO_HOP_SIZE,
  LFM3_AUDIO_MEL_BINS,
  lfm3AudioFrameCount,
  lfm3AudioPaddedLength,
  lfm3AudioTokenCount,
  lfm3LogMelSpectrogram,
} from '../../../../src/zoo/lfm3/audio/frontend.ts';

function directLogMel(audio: Float32Array): Float32Array {
  const fftSize = 400;
  const sampleRate = 16_000;
  const bins = fftSize / 2 + 1;
  const powers = new Float64Array(bins);
  for (let bin = 0; bin < bins; bin++) {
    let real = 0;
    let imaginary = 0;
    for (let sample = 0; sample < fftSize; sample++) {
      const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * sample) / fftSize);
      const value = (audio[sample] ?? 0) * window;
      const phase = (2 * Math.PI * bin * sample) / fftSize;
      real += value * Math.cos(phase);
      imaginary -= value * Math.sin(phase);
    }
    powers[bin] = real * real + imaginary * imaginary;
  }

  const toMel = (frequency: number) =>
    frequency >= 1_000
      ? 15 + (Math.log(frequency / 1_000) * 27) / Math.log(6.4)
      : (3 * frequency) / 200;
  const fromMel = (mel: number) =>
    mel >= 15 ? 1_000 * Math.exp((Math.log(6.4) * (mel - 15)) / 27) : (200 * mel) / 3;
  const lastMel = toMel(sampleRate / 2);
  const edges = Array.from({ length: LFM3_AUDIO_MEL_BINS + 2 }, (_, index) =>
    fromMel((lastMel * index) / (LFM3_AUDIO_MEL_BINS + 1)),
  );
  return Float32Array.from({ length: LFM3_AUDIO_MEL_BINS }, (_, mel) => {
    const lower = edges[mel]!;
    const center = edges[mel + 1]!;
    const upper = edges[mel + 2]!;
    let energy = 0;
    for (let bin = 0; bin < bins; bin++) {
      const frequency = (bin * (sampleRate / 2)) / (bins - 1);
      const triangle = Math.max(
        0,
        Math.min((frequency - lower) / (center - lower), (upper - frequency) / (upper - center)),
      );
      energy += powers[bin]! * triangle * (2 / (upper - lower));
    }
    return (Math.log10(Math.max(energy, 1e-12)) + 4) / 8;
  });
}

describe('LFM3-Omni audio frontend', () => {
  it.each([
    { samples: 0, frames: 1 },
    { samples: LFM3_AUDIO_FFT_SIZE, frames: 1 },
    { samples: LFM3_AUDIO_FFT_SIZE + 1, frames: 2 },
    { samples: LFM3_AUDIO_FFT_SIZE + LFM3_AUDIO_HOP_SIZE, frames: 2 },
    { samples: LFM3_AUDIO_FFT_SIZE + LFM3_AUDIO_HOP_SIZE + 1, frames: 3 },
  ])('pads $samples samples to $frames analysis frames', ({ samples, frames }) => {
    expect(lfm3AudioFrameCount(samples)).toBe(frames);
  });

  it('uses Liquid’s 2–32 second graph sizes and three convolutional strides', () => {
    expect(lfm3AudioPaddedLength(1)).toBe(32_000);
    expect(lfm3AudioPaddedLength(32_000)).toBe(32_000);
    expect(lfm3AudioPaddedLength(32_001)).toBe(64_000);
    expect(lfm3AudioPaddedLength(512_000)).toBe(512_000);
    expect(lfm3AudioTokenCount(32_000)).toBe(25);
    expect(lfm3AudioTokenCount(512_000)).toBe(400);
    expect(() => lfm3AudioPaddedLength(512_001)).toThrow(/limited to 32 seconds/);
  });

  it('returns a finite 128-band Slaney log-mel matrix in mel-major order', () => {
    const frames = lfm3AudioFrameCount(401);
    const mel = lfm3LogMelSpectrogram(new Float32Array(401));
    expect(mel).toHaveLength(LFM3_AUDIO_MEL_BINS * frames);
    expect(mel.every(Number.isFinite)).toBe(true);
    expect(mel).toEqual(
      Float32Array.from({ length: mel.length }, () => (Math.log10(1e-12) + 4) / 8),
    );
  });

  it('keeps a tonal input finite and distinct from silence', () => {
    const tone = Float32Array.from({ length: LFM3_AUDIO_FFT_SIZE }, (_, index) =>
      Math.sin((2 * Math.PI * 440 * index) / 16_000),
    );
    const silence = lfm3LogMelSpectrogram(new Float32Array(LFM3_AUDIO_FFT_SIZE));
    const speech = lfm3LogMelSpectrogram(tone);
    expect(speech.every(Number.isFinite)).toBe(true);
    expect(speech.some((value, index) => value !== silence[index])).toBe(true);
  });

  it('matches a direct-DFT reference for the published Slaney power frontend', () => {
    const audio = Float32Array.from(
      { length: LFM3_AUDIO_FFT_SIZE },
      (_, index) =>
        0.2 * Math.sin((2 * Math.PI * 440 * index) / 16_000) +
        0.07 * Math.cos((2 * Math.PI * 1_120 * index) / 16_000),
    );
    const actual = lfm3LogMelSpectrogram(audio);
    const expected = directLogMel(audio);
    expect(actual).toHaveLength(expected.length);
    for (let mel = 0; mel < expected.length; mel++) {
      expect(actual[mel]).toBeCloseTo(expected[mel]!, 4);
    }
  });
});
