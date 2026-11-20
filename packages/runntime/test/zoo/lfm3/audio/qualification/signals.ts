/** Deterministic synthetic signals for the audio frontend and round-trip qualification. */

export function sine(
  frequency: number,
  samples: number,
  sampleRate = 16_000,
  amplitude = 1,
): Float32Array {
  return Float32Array.from(
    { length: samples },
    (_, index) => amplitude * Math.sin((2 * Math.PI * frequency * index) / sampleRate),
  );
}

/** Linear chirp from `from` to `to` Hz over the whole clip. */
export function chirp(
  from: number,
  to: number,
  samples: number,
  sampleRate = 16_000,
  amplitude = 1,
): Float32Array {
  const seconds = samples / sampleRate;
  return Float32Array.from({ length: samples }, (_, index) => {
    const time = index / sampleRate;
    return (
      amplitude *
      Math.sin(2 * Math.PI * (from * time + ((to - from) * time * time) / (2 * seconds)))
    );
  });
}

export function impulse(samples: number, at: number, amplitude = 1): Float32Array {
  const output = new Float32Array(samples);
  output[at] = amplitude;
  return output;
}

/** A sine driven `gain` times past full scale and hard-clipped to [-1, 1]. */
export function clippedSine(
  frequency: number,
  samples: number,
  gain: number,
  sampleRate = 16_000,
): Float32Array {
  return sine(frequency, samples, sampleRate, gain).map((value) =>
    Math.max(-1, Math.min(1, value)),
  );
}

/** Concatenates clips with `gap` samples of silence between them. */
export function joinWithSilence(clips: readonly Float32Array[], gap: number): Float32Array {
  const length =
    clips.reduce((sum, clip) => sum + clip.length, 0) + gap * Math.max(0, clips.length - 1);
  const output = new Float32Array(length);
  let offset = 0;
  for (const clip of clips) {
    output.set(clip, offset);
    offset += clip.length + gap;
  }
  return output;
}

/** Peak and RMS of a clip, for level checks on generated speech. */
export function levels(samples: Float32Array): { peak: number; rms: number; clipped: number } {
  let peak = 0;
  let sumSquares = 0;
  let clipped = 0;
  for (const sample of samples) {
    peak = Math.max(peak, Math.abs(sample));
    sumSquares += sample * sample;
    if (Math.abs(sample) >= 1) clipped++;
  }
  return { peak, rms: Math.sqrt(sumSquares / Math.max(1, samples.length)), clipped };
}

/** Mono 16-bit PCM RIFF/WAVE bytes at any sample rate. */
export function encodePcm16Wav(channels: readonly Float32Array[], sampleRate: number): ArrayBuffer {
  const frames = channels[0]!.length;
  const buffer = new ArrayBuffer(44 + frames * channels.length * 2);
  const view = new DataView(buffer);
  const text = (offset: number, value: string): void => {
    for (let index = 0; index < value.length; index++)
      view.setUint8(offset + index, value.charCodeAt(index));
  };
  text(0, 'RIFF');
  view.setUint32(4, buffer.byteLength - 8, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels.length, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels.length * 2, true);
  view.setUint16(32, channels.length * 2, true);
  view.setUint16(34, 16, true);
  text(36, 'data');
  view.setUint32(40, frames * channels.length * 2, true);
  for (let frame = 0; frame < frames; frame++) {
    for (let channel = 0; channel < channels.length; channel++) {
      const sample = Math.max(-1, Math.min(1, channels[channel]![frame]!));
      view.setInt16(
        44 + (frame * channels.length + channel) * 2,
        Math.round(sample * 32_767),
        true,
      );
    }
  }
  return buffer;
}
