/** Opt-in real-WebGPU check that repeated audio runs of varying length do not accumulate GPU memory.
 *
 *  Clip and prompt lengths differ per sentence, so a pool trimmed only after warm-up keeps one buffer
 *  set per length seen (1.97 to 2.17 GB in six cycles before the per-operation trim). With the trim,
 *  a second pass over the same sentences must land where the first did. */

import { expect, it } from 'vitest';
import { createLfm3Audio, models, resampleAudio } from '../../../src/zoo/index.ts';
import { gpu, liveBuffers } from '../../gpu.ts';

declare global {
  interface ImportMetaEnv {
    readonly VITE_LFM3_QUALIFICATION_MODE?: string;
    readonly VITE_LFM3_MODEL_URL?: string;
    readonly VITE_LFM3_PROJECTOR_URL?: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const MODE = import.meta.env.VITE_LFM3_QUALIFICATION_MODE;
const SENTENCES = [
  'The quick brown fox jumps over the lazy dog.',
  'Please turn left at the next street.',
  'She sells seashells by the seashore.',
  'The meeting has been moved to Thursday afternoon.',
  'A gentle breeze carried the scent of rain across the quiet valley.',
  'Remember to save your work before closing the window.',
] as const;
/** The second pass may differ from the first by allocator bookkeeping only. */
const REPEAT_SLACK_BYTES = 64 * 1024;

const gpuBufferBytes = (): number =>
  [...liveBuffers()].reduce((sum, buffer) => sum + buffer.size, 0);

it.skipIf(MODE !== 'audio-trim')(
  'returns to the same GPU size when the same sentences repeat and frees everything on dispose',
  async () => {
    await gpu();
    const before = liveBuffers();
    const audio = await createLfm3Audio({
      ...models.lfm3.LFM3_OMNI_2B.Q4_K_M,
      ...(import.meta.env.VITE_LFM3_MODEL_URL
        ? { modelPath: import.meta.env.VITE_LFM3_MODEL_URL }
        : {}),
      ...(import.meta.env.VITE_LFM3_PROJECTOR_URL
        ? { projectorPath: import.meta.env.VITE_LFM3_PROJECTOR_URL }
        : {}),
    });
    const loadedBytes = gpuBufferBytes();
    const rows: { cycle: number; gpuBytes: number }[] = [];
    try {
      for (let cycle = 0; cycle < SENTENCES.length * 2; cycle++) {
        const sentence = SENTENCES[cycle % SENTENCES.length]!;
        const speech = await audio.speak(sentence, { seed: 1_234, maxAudioFrames: 512 }).result;
        await audio.transcribe(await resampleAudio(speech.samples, speech.sampleRate));
        rows.push({ cycle, gpuBytes: gpuBufferBytes() });
      }
      console.log(
        `LFM3_AUDIO_TRIM_MEMORY ${JSON.stringify({ loadedBytes, rows, slack: REPEAT_SLACK_BYTES })}`,
      );
      for (const row of rows.slice(SENTENCES.length)) {
        const first = rows[row.cycle - SENTENCES.length]!;
        expect(Math.abs(row.gpuBytes - first.gpuBytes)).toBeLessThanOrEqual(REPEAT_SLACK_BYTES);
      }
    } finally {
      await audio.dispose();
    }
    // Small uniform and readback buffers stay in the root cache until the root is destroyed.
    const storage = [...liveBuffers()].filter(
      (buffer) => !before.has(buffer) && buffer.usage & GPUBufferUsage.STORAGE,
    );
    expect(storage.map((buffer) => buffer.size)).toEqual([]);
  },
  1_800_000,
);
