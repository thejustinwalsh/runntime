/** Opt-in public packed-consumer qualification for real LFM3 audio on WebGPU. */

import { expect, it } from 'vitest';
import { createLfm3Audio, decodeAudio, models, resampleAudio } from '../../../src/zoo/index.ts';
import { gpu, liveBuffers } from '../../gpu.ts';

declare global {
  interface ImportMetaEnv {
    readonly VITE_LFM3_QUALIFICATION_MODE?: string;
    readonly VITE_LFM3_AUDIO_INPUT_BASE_URL?: string;
    readonly VITE_LFM3_MODEL_URL?: string;
    readonly VITE_LFM3_PROJECTOR_URL?: string;
    readonly VITE_LFM3_NATIVE_AUDIO_CAPTURE?: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const MODE = import.meta.env.VITE_LFM3_QUALIFICATION_MODE;
const AUDIO_INPUT_BASE_URL = import.meta.env.VITE_LFM3_AUDIO_INPUT_BASE_URL;
const AUDIO_CAPTURE = import.meta.env.VITE_LFM3_NATIVE_AUDIO_CAPTURE === '1';
const INPUTS = [
  {
    id: 'quick-brown-fox',
    file: 'quick-brown-fox.wav',
    transcript: 'The quick brown fox jumps over the lazy dog.',
    bytes: 89_560,
    sha256: '0497e70d431e9d827c10bbaff120c9cbabf0834a747107f83cb161dc1f759880',
  },
  {
    id: 'turn-left-next-street',
    file: 'turn-left-next-street.wav',
    transcript: 'Please turn left at the next street.',
    bytes: 70_164,
    sha256: 'f9ce57f4e548e339e944ba87860cd35a545db46e5eafd69f3692ede19f6cb054',
  },
] as const;

function gpuBufferBytes(): number {
  return [...liveBuffers()].reduce((sum, buffer) => sum + buffer.size, 0);
}

async function sha256(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

function normalizeWords(text: string): string {
  return text.toLowerCase().replace(/[.,!?;:'"()[\]{}]/g, '');
}

function waveformMetrics(samples: Float32Array) {
  let sumSquares = 0;
  let peak = 0;
  let clippedSamples = 0;
  for (const sample of samples) {
    sumSquares += sample * sample;
    peak = Math.max(peak, Math.abs(sample));
    if (Math.abs(sample) >= 1) clippedSamples++;
  }
  return {
    samples: samples.length,
    seconds: samples.length / 24_000,
    peak,
    rms: Math.sqrt(sumSquares / Math.max(1, samples.length)),
    clippedSamples,
  };
}

function encodePcm16Wav(samples: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset: number, text: string): void => {
    for (let index = 0; index < text.length; index++)
      view.setUint8(offset + index, text.charCodeAt(index));
  };
  writeText(0, 'RIFF');
  view.setUint32(4, buffer.byteLength - 8, true);
  writeText(8, 'WAVE');
  writeText(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 24_000, true);
  view.setUint32(28, 48_000, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index++) {
    const sample = Math.max(-1, Math.min(1, samples[index]!));
    view.setInt16(44 + index * 2, sample < 0 ? sample * 32_768 : sample * 32_767, true);
  }
  return buffer;
}

async function retainSpeech(filename: string, samples: Float32Array): Promise<void> {
  if (!AUDIO_CAPTURE) return;
  const response = await fetch(`/__native-audio-output/${filename}`, {
    method: 'POST',
    headers: { 'content-type': 'audio/wav' },
    body: encodePcm16Wav(samples),
  });
  expect(response.status).toBe(204);
}

it.skipIf(MODE !== 'audio')(
  'loads the exact published pair and transcribes and streams speech through the public zoo entry',
  async (ctx) => {
    if (!AUDIO_INPUT_BASE_URL) {
      throw new Error('set VITE_LFM3_AUDIO_INPUT_BASE_URL to the HTTPS frozen-audio route');
    }
    const { root } = await gpu();
    const before = liveBuffers();
    let sourceBytes = 0;
    let progressDone = 0;
    let progressTotal = 0;
    let peakGpuBytes = gpuBufferBytes();
    const loadStarted = performance.now();
    const audio = await createLfm3Audio(
      {
        ...models.lfm3.LFM3_OMNI_2B.Q4_K_M,
        ...(import.meta.env.VITE_LFM3_MODEL_URL
          ? { modelPath: import.meta.env.VITE_LFM3_MODEL_URL }
          : {}),
        ...(import.meta.env.VITE_LFM3_PROJECTOR_URL
          ? { projectorPath: import.meta.env.VITE_LFM3_PROJECTOR_URL }
          : {}),
      },
      {
        onBytes(bytes) {
          sourceBytes += bytes;
        },
        onProgress(_name, done, total) {
          progressDone = done;
          progressTotal = total;
          peakGpuBytes = Math.max(peakGpuBytes, gpuBufferBytes());
        },
      },
    );
    ctx.onTestFinished(() => audio.dispose());
    const loadMs = performance.now() - loadStarted;
    const loadedGpuBytes = gpuBufferBytes();
    peakGpuBytes = Math.max(peakGpuBytes, loadedGpuBytes);

    const fixtures = [];
    for (const input of INPUTS) {
      const response = await fetch(`${AUDIO_INPUT_BASE_URL.replace(/\/$/, '')}/${input.file}`);
      if (!response.ok)
        throw new Error(`failed to load frozen ${input.file}: HTTP ${response.status}`);
      const wav = await response.arrayBuffer();
      const digest = await sha256(wav);
      expect(wav.byteLength).toBe(input.bytes);
      expect(digest).toBe(input.sha256);
      const samples = await decodeAudio(wav);
      expect(samples.length).toBeGreaterThan(0);
      const started = performance.now();
      const transcript = await audio.transcribe(samples);
      const latencyMs = performance.now() - started;
      expect(normalizeWords(transcript)).toBe(normalizeWords(input.transcript));
      fixtures.push({
        ...input,
        sampleRate: 16_000,
        samples: samples.length,
        transcript,
        transcriptionMs: +latencyMs.toFixed(2),
      });
    }

    const requested = INPUTS[0]!.transcript;
    const runSpeech = async (depthPasses: 1 | 3) => {
      const chunks: Float32Array[] = [];
      const started = performance.now();
      const generation = audio.speak(requested, {
        seed: 1_234,
        depthPasses,
        maxAudioFrames: 256,
      });
      for await (const chunk of generation) chunks.push(chunk);
      const result = await generation.result;
      const latencyMs = performance.now() - started;
      expect(result.sampleRate).toBe(24_000);
      expect(result.samples.length).toBeGreaterThan(0);
      expect(result.samples.every(Number.isFinite)).toBe(true);
      const streamedLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      expect(streamedLength).toBe(result.samples.length);
      return { ...result, chunks, latencyMs, metrics: waveformMetrics(result.samples) };
    };

    const deepSpeech = await runSpeech(3);
    await retainSpeech('native-speech-depth-3.wav', deepSpeech.samples);
    const resampled = await resampleAudio(deepSpeech.samples, deepSpeech.sampleRate);
    const recovered = await audio.transcribe(resampled);
    console.info(
      `LFM3_PUBLIC_AUDIO_SPEECH_DIAGNOSTIC ${JSON.stringify({
        seed: 1_234,
        depthPasses: 3,
        requested,
        sampleRate: deepSpeech.sampleRate,
        samples: deepSpeech.samples.length,
        frames: deepSpeech.audioFrames,
        finishReason: deepSpeech.finishReason,
        streamChunks: deepSpeech.chunks.length,
        metrics: deepSpeech.metrics,
        recoveredTranscript: recovered,
      })}`,
    );
    const shallowSpeech = await runSpeech(1);
    await retainSpeech('native-speech-depth-1.wav', shallowSpeech.samples);

    const cancelled = audio.speak('This queued request is cancelled.', { seed: 7 });
    cancelled.cancel();
    await expect(cancelled.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
    const recoveredAfterAbort = await audio.transcribe(
      await decodeAudio(
        await fetch(`${AUDIO_INPUT_BASE_URL.replace(/\/$/, '')}/${INPUTS[1]!.file}`).then(
          (response) => response.arrayBuffer(),
        ),
      ),
    );
    expect(normalizeWords(recoveredAfterAbort)).toBe(normalizeWords(INPUTS[1]!.transcript));

    await audio.dispose();
    const remaining = [...liveBuffers()].filter((buffer) => !before.has(buffer));
    const report = {
      sourcePair: audio.sources,
      bytesRead: sourceBytes,
      progressDone,
      progressTotal,
      loadAndWarmUpMs: +loadMs.toFixed(2),
      loadedGpuBytes,
      peakGpuBytes,
      fixedClipTranscriptions: fixtures,
      speech: {
        seed: 1_234,
        input: requested,
        deep: {
          depthPasses: 3,
          frames: deepSpeech.audioFrames,
          finishReason: deepSpeech.finishReason,
          latencyMs: +deepSpeech.latencyMs.toFixed(2),
          streamChunks: deepSpeech.chunks.length,
          metrics: deepSpeech.metrics,
          recoveredTranscript: recovered,
        },
        shallow: {
          depthPasses: 1,
          frames: shallowSpeech.audioFrames,
          finishReason: shallowSpeech.finishReason,
          latencyMs: +shallowSpeech.latencyMs.toFixed(2),
          streamChunks: shallowSpeech.chunks.length,
          metrics: shallowSpeech.metrics,
        },
      },
      recoveredAfterAbort,
      gpuStorageBuffersAfterDispose: remaining
        .filter((buffer) => (buffer.usage & GPUBufferUsage.STORAGE) !== 0)
        .map((buffer) => ({ bytes: buffer.size, usage: buffer.usage })),
      gpuRootCacheBufferBytesAfterDispose: remaining.reduce((sum, buffer) => sum + buffer.size, 0),
      device: root.device.label,
    };
    console.info(`LFM3_PUBLIC_AUDIO_QUALIFICATION ${JSON.stringify(report)}`);
    expect(normalizeWords(recovered)).toBe(normalizeWords(requested));
    expect(progressDone).toBe(progressTotal);
    expect(peakGpuBytes).toBeGreaterThan(0);
    expect(report.gpuStorageBuffersAfterDispose).toEqual([]);
  },
  1_200_000,
);
