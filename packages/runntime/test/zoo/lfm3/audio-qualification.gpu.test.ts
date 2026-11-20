/** Audio qualification on a real WebGPU device, from local weights only.
 *  Modes (VITE_LFM3_QUALIFICATION_MODE): audio-roundtrip, audio-streaming, audio-input, audio-memory.
 *  Run them through `pnpm --filter runntime test:audio`. */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { decodeAudio, resampleAudio } from '../../../src/zoo/index.ts';
import { liveBuffers } from '../../gpu.ts';
import {
  adapterInfo,
  gpuBufferBytes,
  loadAudio,
  MODE,
  report,
  round,
  seeds,
  selectedVoices,
  SENTENCES,
  VOICES,
  type LoadedAudio,
} from './audio/qualification/harness.ts';
import {
  chirp,
  encodePcm16Wav,
  joinWithSilence,
  levels,
  sine,
} from './audio/qualification/signals.ts';
import { MEMORY_LIMITS, WER_LIMITS } from './audio/qualification/thresholds.ts';
import { wordErrorRate } from './audio/qualification/wer.ts';

let loaded: LoadedAudio;

function useModel(mode: string): void {
  beforeAll(async () => {
    loaded = await loadAudio();
    report('load', {
      mode,
      adapter: await adapterInfo(),
      loadMs: round(loaded.loadMs, 1),
      sourceBytes: loaded.sourceBytes,
      loadedGpuBytes: loaded.loadedGpuBytes,
    });
  }, 600_000);
  afterAll(async () => {
    await loaded?.audio.dispose();
    report('network', {
      mode,
      sameOriginRequests: loaded?.guard.requests.length,
      blocked: loaded?.guard.blocked,
    });
  }, 120_000);
}

describe.skipIf(MODE !== 'audio-roundtrip')('audio round trip: text to speech to text', () => {
  useModel('audio-roundtrip');

  it('speaks every sentence in every voice and transcribes it back within the measured word error rate', async () => {
    const rows: Record<string, unknown>[] = [];
    for (const seed of seeds()) {
      for (const { id, voice } of selectedVoices()) {
        for (const sentence of SENTENCES) {
          const speakStarted = performance.now();
          const generation = loaded.audio.speak(sentence, { seed, voice, maxAudioFrames: 512 });
          const chunks: Float32Array[] = [];
          for await (const chunk of generation) chunks.push(chunk);
          const speech = await generation.result;
          const speakMs = performance.now() - speakStarted;
          expect(speech.sampleRate).toBe(24_000);
          expect(speech.finishReason, `${id} "${sentence}" stopped on its own`).toBe('stop');
          expect(chunks.reduce((sum, chunk) => sum + chunk.length, 0)).toBe(speech.samples.length);
          const transcribeStarted = performance.now();
          const transcript = await loaded.audio.transcribe(
            await resampleAudio(speech.samples, speech.sampleRate),
          );
          const transcribeMs = performance.now() - transcribeStarted;
          const wer = wordErrorRate(sentence, transcript);
          const { peak, rms, clipped } = levels(speech.samples);
          rows.push({
            seed,
            voice: id,
            sentence,
            transcript,
            wer: round(wer),
            frames: speech.audioFrames,
            samples: speech.samples.length,
            seconds: round(speech.samples.length / 24_000, 3),
            chunks: chunks.length,
            peak: round(peak),
            rms: round(rms),
            clipped,
            speakMs: round(speakMs, 1),
            transcribeMs: round(transcribeMs, 1),
          });
        }
      }
    }
    const wers = rows.map((row) => row.wer as number);
    const summary = {
      items: rows.length,
      meanWer: round(wers.reduce((sum, value) => sum + value, 0) / wers.length),
      maxWer: Math.max(...wers),
      exact: wers.filter((value) => value === 0).length,
      byVoice: Object.fromEntries(
        selectedVoices().map(({ id }) => {
          const own = rows.filter((row) => row.voice === id).map((row) => row.wer as number);
          return [id, round(own.reduce((sum, value) => sum + value, 0) / own.length)];
        }),
      ),
    };
    report('roundtrip', { seeds: seeds(), summary, rows });
    expect(summary.meanWer).toBeLessThanOrEqual(WER_LIMITS.meanWer);
    expect(summary.maxWer).toBeLessThanOrEqual(WER_LIMITS.itemWer);
  }, 3_600_000);
});

const FOX = SENTENCES[0];
const STREAM_SEED = 1_234;

async function speakAll(text: string, seed = STREAM_SEED) {
  const generation = loaded.audio.speak(text, { seed, maxAudioFrames: 512 });
  const chunks: Float32Array[] = [];
  for await (const chunk of generation) chunks.push(chunk);
  return { chunks, result: await generation.result };
}

function identical(left: Float32Array, right: Float32Array): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

describe.skipIf(MODE !== 'audio-streaming')('speech streaming, cancellation and recovery', () => {
  useModel('audio-streaming');

  it('streams chunks that concatenate to the result and repeat exactly for one seed', async () => {
    const first = await speakAll(FOX);
    const second = await speakAll(FOX);
    const joined = new Float32Array(first.chunks.reduce((sum, chunk) => sum + chunk.length, 0));
    let offset = 0;
    for (const chunk of first.chunks) {
      joined.set(chunk, offset);
      offset += chunk.length;
    }
    expect(first.chunks.length).toBeGreaterThan(1);
    expect(identical(joined, first.result.samples)).toBe(true);
    const repeat = identical(first.result.samples, second.result.samples);
    report('streaming-determinism', {
      chunks: first.chunks.length,
      samples: first.result.samples.length,
      sameSeedIdenticalSamples: repeat,
    });
    expect(repeat).toBe(true);
  }, 600_000);

  it('stops at maxAudioFrames with finishReason length', async () => {
    const generation = loaded.audio.speak(FOX, { seed: STREAM_SEED, maxAudioFrames: 3 });
    const result = await generation.result;
    expect(result.finishReason).toBe('length');
    expect(result.audioFrames).toBe(3);
  }, 120_000);

  it('cancels on an early break, frees its buffers and the next speech is unchanged', async () => {
    const reference = await speakAll(FOX);
    const bytesAfterSuccess = gpuBufferBytes();
    const generation = loaded.audio.speak(FOX, { seed: STREAM_SEED, maxAudioFrames: 512 });
    for await (const chunk of generation) {
      expect(chunk.length).toBeGreaterThan(0);
      break;
    }
    await expect(generation.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
    const after = await speakAll(FOX);
    const bytesAfterRecovery = gpuBufferBytes();
    report('streaming-early-break', {
      bytesAfterSuccess,
      bytesAfterRecovery,
      recoveredIdentical: identical(reference.result.samples, after.result.samples),
    });
    expect(identical(reference.result.samples, after.result.samples)).toBe(true);
    expect(bytesAfterRecovery).toBe(bytesAfterSuccess);
  }, 600_000);

  it('cancels in flight with cancel() and with an AbortSignal, then recovers', async () => {
    const byCancel = loaded.audio.speak(FOX, { seed: STREAM_SEED, maxAudioFrames: 512 });
    const iterator = byCancel[Symbol.asyncIterator]();
    await iterator.next();
    byCancel.cancel();
    await expect(byCancel.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });

    const controller = new AbortController();
    const bySignal = loaded.audio.speak(FOX, {
      seed: STREAM_SEED,
      maxAudioFrames: 512,
      signal: controller.signal,
    });
    const signalIterator = bySignal[Symbol.asyncIterator]();
    await signalIterator.next();
    controller.abort();
    await expect(bySignal.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });

    const preAborted = new AbortController();
    preAborted.abort();
    await expect(
      loaded.audio.speak(FOX, { signal: preAborted.signal }).result,
    ).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });

    const recovered = await speakAll(FOX);
    expect(recovered.result.finishReason).toBe('stop');
  }, 600_000);

  it('keeps later queued work running when a queued request fails', async () => {
    const reference = await speakAll(FOX);
    const running = loaded.audio.speak(FOX, { seed: STREAM_SEED, maxAudioFrames: 512 });
    const doomed = loaded.audio.speak(SENTENCES[1], { seed: 7 });
    const afterDoomed = loaded.audio.transcribe(
      await resampleAudio(reference.result.samples, 24_000),
    );
    doomed.cancel();
    await expect(doomed.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
    const finished = await running.result;
    const transcript = await afterDoomed;
    report('streaming-queued-failure', {
      runningIdentical: identical(finished.samples, reference.result.samples),
      transcript,
    });
    expect(identical(finished.samples, reference.result.samples)).toBe(true);
    expect(wordErrorRate(FOX, transcript)).toBeLessThanOrEqual(WER_LIMITS.itemWer);
  }, 600_000);

  it('rejects invalid requests before queueing them and leaves the runner usable', async () => {
    expect(() => loaded.audio.speak('   ')).toThrowError(/must not be empty/);
    expect(() => loaded.audio.speak(FOX, { maxAudioFrames: 0 })).toThrowError(/maxAudioFrames/);
    expect(() => loaded.audio.speak(FOX, { seed: -1 })).toThrowError(/seed/);
    expect(() => loaded.audio.speak(FOX, { voice: { gender: 'robot' as 'male' } })).toThrowError(
      /gender/,
    );
    await expect(loaded.audio.transcribe(new Float32Array(0))).rejects.toMatchObject({
      code: 'INVALID_ARGUMENT',
    });
    const speech = await speakAll(SENTENCES[1]);
    expect(speech.result.finishReason).toBe('stop');
  }, 600_000);

  it('rejects in-flight work on dispose and refuses later calls', async () => {
    const pending = loaded.audio.speak(FOX, { seed: STREAM_SEED, maxAudioFrames: 512 });
    const iterator = pending[Symbol.asyncIterator]();
    await iterator.next();
    const disposing = loaded.audio.dispose();
    await expect(pending.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
    await disposing;
    expect(() => loaded.audio.speak(FOX)).toThrowError(/disposed/);
    const leaked = [...liveBuffers()].filter(
      (buffer) => !loaded.before.has(buffer) && (buffer.usage & GPUBufferUsage.STORAGE) !== 0,
    );
    report('streaming-dispose', { storageBuffersAfterDispose: leaked.length });
    expect(leaked).toEqual([]);
  }, 600_000);
});

/** Energy of `samples` at `frequency`, by a single-bin DFT, normalised to the amplitude of a sine. */
function amplitudeAt(samples: Float32Array, frequency: number, sampleRate: number): number {
  let real = 0;
  let imaginary = 0;
  for (let index = 0; index < samples.length; index++) {
    const phase = (2 * Math.PI * frequency * index) / sampleRate;
    real += samples[index]! * Math.cos(phase);
    imaginary += samples[index]! * Math.sin(phase);
  }
  return (2 * Math.hypot(real, imaginary)) / samples.length;
}

async function speech16k(text: string, voice?: (typeof VOICES)[number]['voice']) {
  const generation = loaded.audio.speak(text, { seed: STREAM_SEED, voice, maxAudioFrames: 512 });
  const result = await generation.result;
  return { result, samples16k: await resampleAudio(result.samples, result.sampleRate) };
}

describe.skipIf(MODE !== 'audio-input')(
  'audio input: boundaries, short clips, sample rates',
  () => {
    useModel('audio-input');

    it('keeps the transcript through every padded-graph boundary', async () => {
      const { samples16k } = await speech16k(FOX);
      const baseline = await loaded.audio.transcribe(samples16k);
      const rows = [
        { samples: samples16k.length, transcript: baseline, wer: wordErrorRate(FOX, baseline) },
      ];
      // Liquid pads the waveform to 2 s * 2^k; test each side of the first boundaries past the clip.
      for (const boundary of [64_000, 128_000]) {
        for (const length of [boundary, boundary + 1]) {
          if (length < samples16k.length) continue;
          const padded = new Float32Array(length);
          padded.set(samples16k);
          const transcript = await loaded.audio.transcribe(padded);
          rows.push({ samples: length, transcript, wer: wordErrorRate(FOX, transcript) });
        }
      }
      report('input-boundaries', { rows });
      for (const row of rows) expect(row.wer).toBeLessThanOrEqual(WER_LIMITS.itemWer);
    }, 600_000);

    it('transcribes a clip of the maximum 32 seconds and rejects one sample more', async () => {
      const clips: Float32Array[] = [];
      const spoken: string[] = [];
      let total = 0;
      for (const sentence of SENTENCES) {
        const { samples16k } = await speech16k(sentence);
        if (total + samples16k.length + 8_000 > 512_000) break;
        clips.push(samples16k);
        spoken.push(sentence);
        total += samples16k.length + 8_000;
      }
      const long = joinWithSilence(clips, 8_000);
      const maximum = new Float32Array(512_000);
      maximum.set(long);
      const started = performance.now();
      const transcript = await loaded.audio.transcribe(maximum, { maxNewTokens: 512 });
      const transcribeMs = performance.now() - started;
      const wer = wordErrorRate(spoken.join(' '), transcript);
      report('input-maximum-length', {
        sentences: spoken.length,
        speechSamples: long.length,
        samples: maximum.length,
        transcript,
        wer: round(wer),
        transcribeMs: round(transcribeMs, 1),
      });
      expect(wer).toBeLessThanOrEqual(WER_LIMITS.longWer);
      await expect(loaded.audio.transcribe(new Float32Array(512_001))).rejects.toMatchObject({
        code: 'INVALID_ARGUMENT',
      });
    }, 1_800_000);

    it('answers silence and very short clips without failing', async () => {
      const rows: Record<string, unknown>[] = [];
      for (const [name, samples] of [
        ['1 sample', new Float32Array(1)],
        ['25 ms of silence', new Float32Array(400)],
        ['50 ms tone', sine(440, 800, 16_000, 0.5)],
        ['1 s of silence', new Float32Array(16_000)],
        ['3 s of silence', new Float32Array(48_000)],
        ['2 s chirp', chirp(200, 4_000, 32_000, 16_000, 0.5)],
      ] as const) {
        const started = performance.now();
        const transcript = await loaded.audio.transcribe(samples);
        rows.push({
          name,
          samples: samples.length,
          transcript,
          ms: round(performance.now() - started, 1),
        });
      }
      report('input-silence-and-short', { rows });
      expect(rows).toHaveLength(6);
    }, 600_000);

    it('handles sample rates: resampling and decoding keep the tone and the words', async () => {
      const tone24 = sine(1_000, 24_000, 24_000, 0.5);
      const tone16 = await resampleAudio(tone24, 24_000);
      expect(tone16.length).toBe(16_000);
      const atTone = amplitudeAt(tone16.subarray(1_000, 15_000), 1_000, 16_000);
      const offTone = amplitudeAt(tone16.subarray(1_000, 15_000), 1_500, 16_000);
      expect(atTone).toBeGreaterThan(0.45);
      expect(atTone).toBeLessThan(0.55);
      expect(offTone).toBeLessThan(0.02);
      expect(await resampleAudio(tone16, 16_000)).toBe(tone16);

      const left = sine(1_000, 48_000, 48_000, 0.5);
      const decoded = await decodeAudio(encodePcm16Wav([left, left], 48_000));
      expect(decoded.length).toBe(16_000);
      expect(amplitudeAt(decoded.subarray(1_000, 15_000), 1_000, 16_000)).toBeGreaterThan(0.45);

      const generated = await loaded.audio.speak(FOX, { seed: STREAM_SEED, maxAudioFrames: 512 })
        .result;
      const viaResample = await loaded.audio.transcribe(
        await resampleAudio(generated.samples, 24_000),
      );
      const viaDecode = await loaded.audio.transcribe(
        await decodeAudio(encodePcm16Wav([generated.samples], 24_000)),
      );
      const unresampled = await loaded.audio.transcribe(generated.samples);
      const result = {
        resampled: { transcript: viaResample, wer: round(wordErrorRate(FOX, viaResample)) },
        decoded: { transcript: viaDecode, wer: round(wordErrorRate(FOX, viaDecode)) },
        unresampled: { transcript: unresampled, wer: round(wordErrorRate(FOX, unresampled)) },
      };
      report('input-sample-rates', result);
      expect(result.resampled.wer).toBeLessThanOrEqual(WER_LIMITS.itemWer);
      expect(result.decoded.wer).toBeLessThanOrEqual(WER_LIMITS.itemWer);
    }, 600_000);

    it('answers a spoken question through speech, transcription and the language backbone', async () => {
      const question = 'What is the capital of France?';
      const { samples16k } = await speech16k(question);
      const heard = await loaded.audio.transcribe(samples16k);
      const reply = await loaded.audio.generate(
        [{ role: 'user', content: `${heard} Reply with only the city.` }],
        { maxNewTokens: 16 },
      ).result;
      report('input-spoken-question', {
        heard,
        wer: round(wordErrorRate(question, heard)),
        reply: reply.text,
      });
      expect(reply.text).toMatch(/paris/i);
    }, 600_000);
  },
);

describe.skipIf(MODE !== 'audio-memory')('GPU memory across repeated audio runs', () => {
  useModel('audio-memory');

  it('returns to the same size when the same inputs repeat and releases everything on dispose', async () => {
    const cycles = 12;
    const rows: {
      cycle: number;
      sentence: string;
      gpuBytes: number;
      buffers: number;
      heapBytes: number | null;
    }[] = [];
    for (let cycle = 0; cycle < cycles; cycle++) {
      const sentence = SENTENCES[cycle % SENTENCES.length]!;
      const { samples16k } = await speech16k(sentence);
      await loaded.audio.transcribe(samples16k);
      const memory = Reflect.get(performance, 'memory');
      rows.push({
        cycle,
        sentence,
        gpuBytes: gpuBufferBytes(),
        buffers: liveBuffers().size,
        heapBytes:
          typeof memory === 'object' && memory !== null
            ? Reflect.get(memory, 'usedJSHeapSize')
            : null,
      });
    }
    await loaded.audio.dispose();
    const remaining = [...liveBuffers()].filter((buffer) => !loaded.before.has(buffer));
    const storage = remaining.filter((buffer) => (buffer.usage & GPUBufferUsage.STORAGE) !== 0);
    report('memory', {
      loadedGpuBytes: loaded.loadedGpuBytes,
      rows,
      storageBuffersAfterDispose: storage.length,
      rootCacheBytesAfterDispose: remaining.reduce((sum, buffer) => sum + buffer.size, 0),
    });
    // The second pass over the same sentences must land where the first pass did: nothing accumulates.
    for (const row of rows.slice(SENTENCES.length)) {
      const first = rows[row.cycle - SENTENCES.length]!;
      expect(Math.abs(row.gpuBytes - first.gpuBytes)).toBeLessThanOrEqual(
        MEMORY_LIMITS.repeatSlackBytes,
      );
    }
    expect(storage).toEqual([]);
  }, 1_800_000);
});
