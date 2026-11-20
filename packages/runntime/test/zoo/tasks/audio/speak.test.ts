/** Public audio speech wiring: stream-only early break, cancellation and reuse.
 *
 *  The speech `result` carries an internal rejection observer mirroring the
 *  chat session, so abandoning the sample stream still settles `result`
 *  without an unhandled rejection. Removing that observer must fail this
 *  test: the early break leaves the abort rejection unobserved until the
 *  test awaits it.
 *
 *  Heavy IO boundaries (GGUF parsing/loading, the language backbone, audio
 *  weights, codec synthesis) are faked; the real `createLfm3Audio`
 *  scheduling, queueing, cancellation and stream/result wiring runs
 *  unmodified. */

import { describe, expect, it, vi } from 'vitest';

const fault = vi.hoisted(() => ({
  failNextForward: false,
  backboneLoads: 0,
  backboneDisposals: 0,
  trims: 0,
}));

vi.mock('../../../../src/core/index.ts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../../src/core/index.ts')>();
  return {
    ...actual,
    supportsF16: () => true,
    inGpuErrorScopes: async (_device: unknown, _name: string, fn: () => Promise<unknown>) => fn(),
    tensor: () => ({}),
    astype: (value: unknown) => value,
    cat: () => ({}),
    evalValues: () => {},
    toArray: async () => {
      const logits = new Float32Array(2017);
      logits[5] = 10;
      return logits;
    },
    parseGgufHeader: () => ({}),
    fromGguf: async () => ({ metadata: {}, tensors: new Map(), totalBytes: () => 0 }),
  };
});

vi.mock('../../../../src/zoo/tasks/nlp/lfm3.ts', () => ({
  loadLfm3Backbone: async () => {
    fault.backboneLoads++;
    return {
      root: { device: { queue: { onSubmittedWorkDone: async () => {} } } },
      executor: {
        releaseValues() {},
        trim() {
          fault.trims++;
        },
      },
      model: {},
      tokenizer: {
        bosId: 1,
        audioId: 2,
        audioOutId: 3,
        audioStartId: 4,
        textEndId: 5,
        stopIds: new Set([5]),
        encodeChat: () => [10, 11],
        decode: () => '',
      },
      runtime: {
        release() {},
        trim() {},
        forward: async () => {
          const logits = new Float32Array(128_000);
          logits[5] = 1;
          return { logits, cache: {} };
        },
        dispose: async () => {
          fault.backboneDisposals++;
        },
      },
      embedTokens: () => ({}),
      forwardEmbeddings: async () => ({
        logits: new Float32Array(4),
        hidden: new Float32Array(2048).fill(0.1),
        cache: {},
      }),
      forwardTokens: async () => {
        if (fault.failNextForward) {
          fault.failNextForward = false;
          throw new Error('injected language forward failure');
        }
        return {
          logits: new Float32Array(4),
          hidden: new Float32Array(2048).fill(0.1),
          cache: {},
        };
      },
    };
  },
}));

vi.mock('../../../../src/zoo/lfm3/audio/model.ts', () => ({
  LFM3_AUDIO_EOS_CODE: 2016,
  LFM3_AUDIO_CODEBOOKS: 12,
  LFM3_AUDIO_CODEBOOK_SIZE: 2017,
  lfm3AudioTokensForSamples: (samples: number) => Math.ceil(samples / 1280),
  Lfm3AudioModel: class {
    half() {}
    async loadStateDict() {}
    encodeAudio(audio: Float32Array) {
      return { shape: { dims: [Math.ceil(audio.length / 1280)] } };
    }
    forwardDepthFrame() {
      return Array.from({ length: 12 }, () => ({}));
    }
    decodeSpeechCodes() {
      return { spectrum: {}, cache: { layers: [] } };
    }
    audioOutEmbedding() {
      return {};
    }
    dispose() {}
  },
}));

vi.mock('../../../../src/zoo/lfm3/audio/codec.ts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../../../src/zoo/lfm3/audio/codec.ts')>();
  return {
    ...actual,
    LiquidCodecStream: class {
      decode() {
        return new Float32Array(480).fill(0.01);
      }
      finish() {
        return new Float32Array(240).fill(0.01);
      }
    },
  };
});

vi.mock('../../../../src/zoo/lfm3/audio/projector.ts', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('../../../../src/zoo/lfm3/audio/projector.ts')>();
  return {
    ...actual,
    validateLfm3AudioProjector: () => ({
      encoder: {
        sampleRate: 16_000,
        fftSize: 400,
        hopSize: 160,
        melBins: 128,
        hiddenSize: 768,
        feedForwardSize: 1792,
        layers: 17,
        attentionHeads: 8,
        keyValueHeads: 4,
        attentionWindow: 32,
        outputSize: 2048,
      },
      depthDecoder: {
        hiddenSize: 1024,
        feedForwardSize: 2816,
        layers: 3,
        attentionHeads: 32,
        keyValueHeads: 8,
        codebooks: 12,
        codebookSize: 2017,
        sampling: { temperature: 0.4, repetitionPenalty: 1.2, repetitionWindow: 64 },
      },
      codec: { sampleRate: 24_000, hiddenSize: 512, layers: 8 },
    }),
  };
});

import { createLfm3Audio, createLfm3Omni } from '../../../../src/zoo/index.ts';

describe('public audio speech stream', () => {
  it('generates text and speech with one backbone and disposes it once', async () => {
    const loads = fault.backboneLoads;
    const disposals = fault.backboneDisposals;
    const omni = await createLfm3Omni(
      { projectorPath: { read: async () => new Uint8Array() } },
      { warmUp: false },
    );
    try {
      const text = omni.generate([{ role: 'user', content: 'Hello' }], { maxNewTokens: 4 });
      const speech = omni.speak('hello world', { maxAudioFrames: 2, temperature: 0, seed: 7 });
      await expect(text.result).resolves.toMatchObject({ inputTokens: 2, finishReason: 'stop' });
      await expect(speech.result).resolves.toMatchObject({ audioFrames: 2 });
      expect(fault.backboneLoads - loads).toBe(1);
    } finally {
      await omni.dispose();
      await omni.dispose();
    }
    expect(fault.backboneDisposals - disposals).toBe(1);
  });

  it('early break cancels without an unhandled result rejection, then speaks again', async () => {
    const audio = await createLfm3Audio(
      { projectorPath: { read: async () => new Uint8Array() } },
      { warmUp: false },
    );
    try {
      const generation = audio.speak('hello world', {
        maxAudioFrames: 10,
        temperature: 0,
        seed: 7,
      });
      const received: Float32Array[] = [];
      for await (const chunk of generation) {
        received.push(chunk);
        break;
      }
      expect(received.length).toBe(1);
      // Let the abort rejection fire while the test is not awaiting result.
      // Without the internal observer this is an unhandled rejection.
      await new Promise((resolve) => setTimeout(resolve, 20));
      await expect(generation.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });

      const next = audio.speak('second run', { maxAudioFrames: 2, temperature: 0, seed: 7 });
      const chunks: Float32Array[] = [];
      for await (const chunk of next) chunks.push(chunk);
      const result = await next.result;
      expect(result.audioFrames).toBe(2);
      expect(result.samples.length).toBeGreaterThan(0);
      expect([...result.samples].every(Number.isFinite)).toBe(true);
    } finally {
      await audio.dispose();
    }
  });

  it('rejects a failed speech result and runs its already-queued successor', async () => {
    const audio = await createLfm3Audio(
      { projectorPath: { read: async () => new Uint8Array() } },
      { warmUp: false },
    );
    try {
      fault.failNextForward = true;
      const failed = audio.speak('injected failure', { maxAudioFrames: 1, temperature: 0 });
      const queued = audio.speak('queued recovery', { maxAudioFrames: 1, temperature: 0 });

      await expect(failed.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
      const chunks: Float32Array[] = [];
      for await (const chunk of queued) chunks.push(chunk);
      const result = await queued.result;

      expect(chunks.length).toBeGreaterThan(0);
      expect(result.audioFrames).toBe(1);
      expect([...result.samples].every(Number.isFinite)).toBe(true);
    } finally {
      await audio.dispose();
    }
  });

  describe('executor pool trimming', () => {
    const open = () =>
      createLfm3Audio({ projectorPath: { read: async () => new Uint8Array() } }, { warmUp: false });

    /** Clip and prompt lengths vary, so a pool trimmed only at warm-up keeps a buffer set for
     *  every length seen. Each queued operation must trim, however it ends. */
    it('trims after a completed speech, a transcription, a cancelled speech and a failed speech', async () => {
      const audio = await open();
      try {
        const trimmed = async (run: () => Promise<unknown>) => {
          const before = fault.trims;
          await run().catch(() => undefined);
          return fault.trims - before;
        };

        const speech = audio.speak('hello world', { maxAudioFrames: 2, temperature: 0, seed: 7 });
        expect(await trimmed(() => speech.result)).toBeGreaterThanOrEqual(1);

        const transcribed = audio.transcribe(new Float32Array(16_000), { maxNewTokens: 2 });
        expect(await trimmed(() => transcribed)).toBeGreaterThanOrEqual(1);

        const cancelled = audio.speak('cancel me', { maxAudioFrames: 10, temperature: 0, seed: 7 });
        cancelled.cancel();
        expect(await trimmed(() => cancelled.result)).toBeGreaterThanOrEqual(0);
        await expect(cancelled.result).rejects.toBeDefined();

        fault.failNextForward = true;
        const failed = audio.speak('injected failure', { maxAudioFrames: 1, temperature: 0 });
        expect(await trimmed(() => failed.result)).toBeGreaterThanOrEqual(1);
        await expect(failed.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
      } finally {
        await audio.dispose();
      }
    });

    it('trims once per queued operation, not once per load', async () => {
      const audio = await open();
      try {
        const before = fault.trims;
        for (let index = 0; index < 3; index++) {
          const speech = audio.speak(`clip ${index}`, {
            maxAudioFrames: 1 + index,
            temperature: 0,
            seed: 7,
          });
          await speech.result;
        }
        expect(fault.trims - before).toBe(3);
      } finally {
        await audio.dispose();
      }
    });
  });
});
