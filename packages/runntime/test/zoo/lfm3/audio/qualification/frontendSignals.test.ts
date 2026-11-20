/** The log-mel frontend against a first-principles reference on synthesised signals. */

import { describe, expect, it } from 'vitest';
import {
  lfm3AudioFrameCount,
  lfm3LogMelSpectrogram,
} from '../../../../../src/zoo/lfm3/audio/frontend.ts';
import { melBandAt, referenceFrameCount, referenceLogMel } from './melReference.ts';
import { chirp, clippedSine, impulse, sine } from './signals.ts';

const SILENCE_FLOOR = (Math.log10(1e-12) + 4) / 4;

function maxAbsoluteDifference(actual: Float32Array, expected: Float32Array): number {
  expect(actual).toHaveLength(expected.length);
  let worst = 0;
  for (let index = 0; index < actual.length; index++) {
    worst = Math.max(worst, Math.abs(actual[index]! - expected[index]!));
  }
  return worst;
}

const SIGNALS: readonly { name: string; audio: Float32Array }[] = [
  { name: 'a 1 kHz sine', audio: sine(1_000, 1_600, 16_000, 0.5) },
  { name: 'a 100 Hz to 7 kHz chirp', audio: chirp(100, 7_000, 1_600, 16_000, 0.5) },
  { name: 'an impulse inside the first frame', audio: impulse(960, 200) },
  { name: 'an impulse on a frame boundary', audio: impulse(960, 480, 0.8) },
  { name: 'silence', audio: new Float32Array(800) },
  { name: 'a sine clipped at 4x gain', audio: clippedSine(440, 1_600, 4) },
  { name: 'a sine shorter than one frame', audio: sine(2_000, 150, 16_000, 0.3) },
  { name: 'a signal one sample past a frame boundary', audio: sine(300, 561, 16_000, 0.3) },
];

describe('LFM3-Omni frontend on synthesised signals', () => {
  it.each(SIGNALS)('matches the direct-DFT reference for $name', ({ audio }) => {
    const actual = lfm3LogMelSpectrogram(audio);
    expect(lfm3AudioFrameCount(audio.length)).toBe(referenceFrameCount(audio.length));
    // Float32 storage of a value near 0 to 1 bounds the comparison, not the DFT.
    expect(maxAbsoluteDifference(actual, referenceLogMel(audio))).toBeLessThan(1e-4);
  });

  it('puts a pure tone in the mel band that contains it', () => {
    const frames = lfm3AudioFrameCount(1_600);
    const mel = lfm3LogMelSpectrogram(sine(1_000, 1_600, 16_000, 0.5));
    const frame = 3;
    let loudest = 0;
    for (let band = 0; band < 128; band++) {
      if (mel[band * frames + frame]! > mel[loudest * frames + frame]!) loudest = band;
    }
    expect(Math.abs(loudest - melBandAt(1_000))).toBeLessThanOrEqual(1);
  });

  it('moves a chirp up the mel axis frame by frame', () => {
    const audio = chirp(200, 6_000, 4_800, 16_000, 0.5);
    const frames = lfm3AudioFrameCount(audio.length);
    const mel = lfm3LogMelSpectrogram(audio);
    const loudestBand = (frame: number): number => {
      let best = 0;
      for (let band = 0; band < 128; band++) {
        if (mel[band * frames + frame]! > mel[best * frames + frame]!) best = band;
      }
      return best;
    };
    expect(loudestBand(5)).toBeLessThan(loudestBand(frames >> 1));
    expect(loudestBand(frames >> 1)).toBeLessThan(loudestBand(frames - 8));
  });

  it('adds log10(4) / 4 to every band when the amplitude doubles', () => {
    const quiet = lfm3LogMelSpectrogram(sine(700, 1_200, 16_000, 0.1));
    const loud = lfm3LogMelSpectrogram(sine(700, 1_200, 16_000, 0.2));
    for (let index = 0; index < quiet.length; index++) {
      if (quiet[index]! > SILENCE_FLOOR + 0.05) {
        expect(loud[index]! - quiet[index]!).toBeCloseTo(Math.log10(4) / 4, 3);
      }
    }
  });

  it('holds silence at the log floor and keeps a clipped signal finite', () => {
    expect(
      lfm3LogMelSpectrogram(new Float32Array(1_000)).every((value) => value === SILENCE_FLOOR),
    ).toBe(true);
    expect(lfm3LogMelSpectrogram(clippedSine(220, 3_200, 50)).every(Number.isFinite)).toBe(true);
  });
});
