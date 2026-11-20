/** First-principles log-mel reference: a direct DFT and Slaney triangles, no shared code with src. */

const FFT_SIZE = 400;
const HOP = 160;
const BINS = FFT_SIZE / 2 + 1;
const SAMPLE_RATE = 16_000;
const MELS = 128;

const toMel = (hz: number): number =>
  hz >= 1_000 ? 15 + (Math.log(hz / 1_000) * 27) / Math.log(6.4) : (3 * hz) / 200;
const fromMel = (mel: number): number =>
  mel >= 15 ? 1_000 * Math.exp((Math.log(6.4) * (mel - 15)) / 27) : (200 * mel) / 3;

/** Number of frames: one, then one more per hop until the right-padded signal is covered. */
export function referenceFrameCount(samples: number): number {
  return samples <= FFT_SIZE ? 1 : 1 + Math.ceil((samples - FFT_SIZE) / HOP);
}

/** Log-mel in [mel, frame] order: (log10(max(power energy, 1e-12)) + 4) / 4 on 128 Slaney bands. */
export function referenceLogMel(audio: Float32Array): Float32Array {
  const frames = referenceFrameCount(audio.length);
  const edges = Array.from({ length: MELS + 2 }, (_, index) =>
    fromMel((toMel(SAMPLE_RATE / 2) * index) / (MELS + 1)),
  );
  const output = new Float32Array(MELS * frames);
  const cosines = Float64Array.from({ length: FFT_SIZE }, (_, n) =>
    Math.cos((2 * Math.PI * n) / FFT_SIZE),
  );
  const sines = Float64Array.from({ length: FFT_SIZE }, (_, n) =>
    Math.sin((2 * Math.PI * n) / FFT_SIZE),
  );
  for (let frame = 0; frame < frames; frame++) {
    const windowed = Float64Array.from(
      { length: FFT_SIZE },
      (_, n) =>
        (audio[frame * HOP + n] ?? 0) * (0.5 - 0.5 * Math.cos((2 * Math.PI * n) / FFT_SIZE)),
    );
    const power = new Float64Array(BINS);
    for (let bin = 0; bin < BINS; bin++) {
      let real = 0;
      let imaginary = 0;
      for (let n = 0; n < FFT_SIZE; n++) {
        const index = (bin * n) % FFT_SIZE;
        real += windowed[n]! * cosines[index]!;
        imaginary -= windowed[n]! * sines[index]!;
      }
      power[bin] = real * real + imaginary * imaginary;
    }
    for (let mel = 0; mel < MELS; mel++) {
      const lower = edges[mel]!;
      const center = edges[mel + 1]!;
      const upper = edges[mel + 2]!;
      let energy = 0;
      for (let bin = 0; bin < BINS; bin++) {
        const hz = (bin * SAMPLE_RATE) / 2 / (BINS - 1);
        const triangle = Math.max(
          0,
          Math.min((hz - lower) / (center - lower), (upper - hz) / (upper - center)),
        );
        energy += power[bin]! * triangle * (2 / (upper - lower));
      }
      output[mel * frames + frame] = (Math.log10(Math.max(energy, 1e-12)) + 4) / 4;
    }
  }
  return output;
}

/** Index of the mel band whose triangle is centered nearest `hz`. */
export function melBandAt(hz: number): number {
  const edges = Array.from({ length: MELS + 2 }, (_, index) =>
    fromMel((toMel(SAMPLE_RATE / 2) * index) / (MELS + 1)),
  );
  let best = 0;
  for (let mel = 0; mel < MELS; mel++) {
    if (Math.abs(edges[mel + 1]! - hz) < Math.abs(edges[best + 1]! - hz)) best = mel;
  }
  return best;
}
