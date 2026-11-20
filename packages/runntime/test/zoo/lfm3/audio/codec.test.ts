import { describe, expect, it } from 'vitest';
import {
  decodeLiquidCodecSpectrum,
  LiquidCodecStream,
} from '../../../../src/zoo/lfm3/audio/codec.ts';

function dcOverlapAddReference(frames: number): Float32Array {
  const fftSize = 1_280;
  const hopSize = 320;
  const trim = (fftSize - hopSize) / 2;
  const output = new Float32Array(frames * hopSize * 6);
  for (let index = 0; index < output.length; index++) {
    const sample = index + trim;
    let numerator = 0;
    let envelope = 0;
    for (let frame = 0; frame < frames * 6; frame++) {
      const offset = sample - frame * hopSize;
      if (offset < 0 || offset >= fftSize) continue;
      const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * offset) / fftSize);
      numerator += window / fftSize;
      envelope += window * window;
    }
    output[index] = numerator / envelope;
  }
  return output;
}

function directSpectrumOverlapAddReference(spectrum: Float32Array): Float32Array {
  const fftSize = 1_280;
  const bins = fftSize / 2 + 1;
  const hopSize = 320;
  const edgeTrim = (fftSize - hopSize) / 2;
  const stftFrames = spectrum.length / (bins * 2);
  const frames: Float64Array[] = [];
  for (let frame = 0; frame < stftFrames; frame++) {
    const base = frame * bins * 2;
    const real = new Float64Array(fftSize);
    const imaginary = new Float64Array(fftSize);
    for (let bin = 0; bin < bins; bin++) {
      const magnitude = Math.exp(spectrum[base + bin]!);
      const phase = spectrum[base + bins + bin]!;
      const re = magnitude * Math.cos(phase);
      const im = magnitude * Math.sin(phase);
      real[bin] = re;
      imaginary[bin] = bin === 0 || bin === fftSize / 2 ? 0 : im;
      if (bin > 0 && bin < fftSize / 2) {
        real[fftSize - bin] = re;
        imaginary[fftSize - bin] = -im;
      }
    }

    const inverse = new Float64Array(fftSize);
    for (let sample = 0; sample < fftSize; sample++) {
      let value = 0;
      for (let bin = 0; bin < fftSize; bin++) {
        const angle = (2 * Math.PI * bin * sample) / fftSize;
        value += real[bin]! * Math.cos(angle) - imaginary[bin]! * Math.sin(angle);
      }
      inverse[sample] = value / fftSize;
    }
    frames.push(inverse);
  }

  const output = new Float32Array(stftFrames * hopSize);
  for (let index = 0; index < output.length; index++) {
    const sample = index + edgeTrim;
    let numerator = 0;
    let envelope = 0;
    for (let frame = 0; frame < stftFrames; frame++) {
      const offset = sample - frame * hopSize;
      if (offset < 0 || offset >= fftSize) continue;
      const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * offset) / fftSize);
      numerator += frames[frame]![offset]! * window;
      envelope += window * window;
    }
    output[index] = numerator / envelope;
  }
  return output;
}

describe('LiquidCodec STFT synthesis', () => {
  it('overlap-adds six 1280-point frames into one finite 1920-sample 24 kHz frame', () => {
    const spectrum = new Float32Array(6 * 1_282);
    for (let frame = 0; frame < 6; frame++) {
      spectrum.fill(-100, frame * 1_282, frame * 1_282 + 641);
      spectrum[frame * 1_282] = 0;
    }

    const samples = decodeLiquidCodecSpectrum(spectrum, 1);

    expect(samples).toHaveLength(1_920);
    expect(samples.every(Number.isFinite)).toBe(true);
    const reference = dcOverlapAddReference(1);
    for (let index = 0; index < samples.length; index++) {
      expect(Math.abs(samples[index]! - reference[index]!)).toBeLessThan(1e-7);
    }
  });

  it('matches an independent direct inverse-DFT and windowed overlap-add reference', () => {
    const bins = 641;
    const frameWidth = bins * 2;
    const spectrum = new Float32Array(6 * frameWidth);
    for (let frame = 0; frame < 6; frame++) {
      const base = frame * frameWidth;
      for (let bin = 0; bin < bins; bin++) {
        spectrum[base + bin] = -2 + bin / 2_000;
        spectrum[base + bins + bin] = Math.sin(bin * 0.07) * 0.4;
      }
    }

    const samples = decodeLiquidCodecSpectrum(spectrum, 1);
    const reference = directSpectrumOverlapAddReference(spectrum);

    expect(samples).toHaveLength(reference.length);
    for (let index = 0; index < samples.length; index++) {
      expect(Math.abs(samples[index]! - reference[index]!)).toBeLessThan(2e-5);
    }
  });

  it('refuses malformed frame counts, spectrum lengths and non-finite predictions', () => {
    expect(() => decodeLiquidCodecSpectrum(new Float32Array(), 0)).toThrow(/at least one/);
    expect(() => decodeLiquidCodecSpectrum(new Float32Array(6 * 1_282 - 1), 1)).toThrow(
      /expected 7692/,
    );
    const spectrum = new Float32Array(6 * 1_282);
    spectrum[0] = Number.NaN;
    expect(() => decodeLiquidCodecSpectrum(spectrum, 1)).toThrow(/non-finite spectrum/);
  });

  it('streams adjacent codec frames with the same overlap-add result as one batch', () => {
    const frameWidth = 6 * 1_282;
    const spectrum = new Float32Array(2 * frameWidth);
    for (let frame = 0; frame < 12; frame++) {
      spectrum.fill(-100, frame * 1_282, frame * 1_282 + 641);
      spectrum[frame * 1_282] = frame === 6 ? Math.log(0.5) : 0;
    }
    const stream = new LiquidCodecStream();
    const chunks = [
      stream.decode(spectrum.slice(0, frameWidth)),
      stream.decode(spectrum.slice(frameWidth)),
      stream.finish(),
    ];
    const streamed = new Float32Array(chunks.reduce((length, chunk) => length + chunk.length, 0));
    let offset = 0;
    for (const chunk of chunks) {
      streamed.set(chunk, offset);
      offset += chunk.length;
    }
    const batched = decodeLiquidCodecSpectrum(spectrum, 2);

    expect(streamed).toHaveLength(3_840);
    expect(batched).toHaveLength(streamed.length);
    for (let index = 0; index < streamed.length; index++) {
      expect(Math.abs(streamed[index]! - batched[index]!)).toBeLessThan(1e-5);
    }
  });
});
