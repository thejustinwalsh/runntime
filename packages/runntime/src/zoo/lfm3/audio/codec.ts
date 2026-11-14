/** CPU overlap-add synthesis for LiquidCodec's predicted STFT frames. */

import { RunntimeError } from '../../../core/index.ts';
import { lfm3InverseFft } from './frontend.ts';

const FFT_SIZE = 1_280;
const FFT_BINS = FFT_SIZE / 2 + 1;
const HOP_SIZE = 320;
const FRAME_STACK = 6;
const SAMPLES_PER_CODEC_FRAME = HOP_SIZE * FRAME_STACK;
const EDGE_TRIM = (FFT_SIZE - HOP_SIZE) / 2;
const SPECTRUM_WIDTH = 2 * FFT_BINS;
const OVERLAP = FFT_SIZE - HOP_SIZE;
const WINDOW = Float64Array.from(
  { length: FFT_SIZE },
  (_, index) => 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / FFT_SIZE),
);

function validateSpectrum(spectrum: Float32Array, codecFrames: number): number {
  if (!Number.isInteger(codecFrames) || codecFrames < 1) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      'LiquidCodec needs at least one complete code frame',
    );
  }
  const stftFrames = codecFrames * FRAME_STACK;
  if (spectrum.length !== stftFrames * SPECTRUM_WIDTH) {
    throw new RunntimeError(
      'EXECUTION_FAILED',
      `LiquidCodec returned ${spectrum.length} spectrum values; expected ${stftFrames * SPECTRUM_WIDTH}`,
    );
  }
  return stftFrames;
}

function synthesizeSpectrum(
  spectrum: Float32Array,
  stftFrames: number,
): {
  audio: Float64Array;
  envelope: Float64Array;
} {
  const outputLength = (stftFrames - 1) * HOP_SIZE + FFT_SIZE;
  const audio = new Float64Array(outputLength);
  const envelope = new Float64Array(outputLength);
  const real = new Float64Array(FFT_SIZE);
  const imag = new Float64Array(FFT_SIZE);
  for (let frame = 0; frame < stftFrames; frame++) {
    const row = frame * SPECTRUM_WIDTH;
    real.fill(0);
    imag.fill(0);
    for (let bin = 0; bin < FFT_BINS; bin++) {
      const logMagnitude = spectrum[row + bin]!;
      const phase = spectrum[row + FFT_BINS + bin]!;
      if (!Number.isFinite(logMagnitude) || !Number.isFinite(phase)) {
        throw new RunntimeError(
          'EXECUTION_FAILED',
          `LiquidCodec returned non-finite spectrum data at frame ${frame}`,
        );
      }
      const magnitude = Math.exp(logMagnitude);
      const re = magnitude * Math.cos(phase);
      const im = magnitude * Math.sin(phase);
      real[bin] = re;
      imag[bin] = bin === 0 || bin === FFT_SIZE / 2 ? 0 : im;
      if (bin > 0 && bin < FFT_SIZE / 2) {
        real[FFT_SIZE - bin] = re;
        imag[FFT_SIZE - bin] = -im;
      }
    }
    const frameAudio = lfm3InverseFft(real, imag).real;
    const begin = frame * HOP_SIZE;
    for (let sample = 0; sample < FFT_SIZE; sample++) {
      const window = WINDOW[sample]!;
      audio[begin + sample] = audio[begin + sample]! + frameAudio[sample]! * window;
      envelope[begin + sample] = envelope[begin + sample]! + window * window;
    }
  }
  return { audio, envelope };
}

function normalizedSlice(
  audio: Float64Array,
  envelope: Float64Array,
  begin: number,
  end: number,
): Float32Array {
  const samples = new Float32Array(end - begin);
  for (let sample = begin; sample < end; sample++) {
    const weight = envelope[sample]!;
    if (weight <= 0) {
      throw new RunntimeError(
        'EXECUTION_FAILED',
        `LiquidCodec overlap-add envelope is empty at sample ${sample - begin}`,
      );
    }
    const value = audio[sample]! / weight;
    if (!Number.isFinite(value)) {
      throw new RunntimeError(
        'EXECUTION_FAILED',
        `LiquidCodec produced non-finite audio at sample ${sample - begin}`,
      );
    }
    samples[sample - begin] = value;
  }
  return samples;
}

/** Carries LiquidCodec's overlap-add tail between streamed code frames. */
export class LiquidCodecStream {
  private audioTail: Float64Array | undefined;
  private envelopeTail: Float64Array | undefined;
  private started = false;
  private finished = false;

  /** Adds codec frames and returns only samples no later frame can change. */
  decode(spectrum: Float32Array, codecFrames = 1): Float32Array {
    if (this.finished) {
      throw new RunntimeError('RESOURCE_DISPOSED', 'LiquidCodec stream is finished');
    }
    const stftFrames = validateSpectrum(spectrum, codecFrames);
    const { audio, envelope } = synthesizeSpectrum(spectrum, stftFrames);
    this.addTail(audio, envelope);
    const begin = this.started ? 0 : EDGE_TRIM;
    this.started = true;
    const end = audio.length - OVERLAP;
    const output = normalizedSlice(audio, envelope, begin, end);
    this.saveTail(audio, envelope, end);
    return output;
  }

  /** Flushes the final overlap tail. No samples are returned for an empty stream. */
  finish(): Float32Array {
    if (this.finished) return new Float32Array();
    this.finished = true;
    if (!this.started || !this.audioTail || !this.envelopeTail) return new Float32Array();
    const end = this.audioTail.length - EDGE_TRIM;
    return normalizedSlice(this.audioTail, this.envelopeTail, 0, end);
  }

  private addTail(audio: Float64Array, envelope: Float64Array): void {
    if (!this.audioTail || !this.envelopeTail) return;
    for (let index = 0; index < this.audioTail.length; index++) {
      audio[index] = audio[index]! + this.audioTail[index]!;
      envelope[index] = envelope[index]! + this.envelopeTail[index]!;
    }
  }

  private saveTail(audio: Float64Array, envelope: Float64Array, begin: number): void {
    this.audioTail = audio.slice(begin);
    this.envelopeTail = envelope.slice(begin);
  }
}

/** Turns LiquidCodec's log-magnitude/phase rows into 24 kHz overlap-add audio. */
export function decodeLiquidCodecSpectrum(
  spectrum: Float32Array,
  codecFrames: number,
): Float32Array {
  const decoder = new LiquidCodecStream();
  const first = decoder.decode(spectrum, codecFrames);
  const last = decoder.finish();
  if (first.length + last.length !== codecFrames * SAMPLES_PER_CODEC_FRAME) {
    throw new RunntimeError(
      'EXECUTION_FAILED',
      'LiquidCodec returned an incomplete overlap-add frame',
    );
  }
  const combined = new Float32Array(first.length + last.length);
  combined.set(first);
  combined.set(last, first.length);
  return combined;
}
