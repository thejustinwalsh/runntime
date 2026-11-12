/** Opt-in real-WebGPU loading and streaming through the public zoo entry. */

import { expect, it } from 'vitest';
import {
  createLfm3Chat,
  models,
  type Lfm3Message,
  type RangeSource,
} from '../../../src/zoo/index.ts';
import { gpu, liveBuffers } from '../../gpu.ts';
import liquidReference from './liquid-reference.json';

declare global {
  interface ImportMetaEnv {
    readonly VITE_LFM3_QUALIFICATION_MODE?: string;
    readonly VITE_LFM3_MODEL_URL?: string;
    readonly VITE_LFM3_TOKENIZER_URL?: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const MODE = import.meta.env.VITE_LFM3_QUALIFICATION_MODE;
const Q4_MODE = MODE === 'q4';
const MODEL_CONFIG = Q4_MODE ? models.lfm3.LFM3_OMNI_2B.Q4_K_M : models.lfm3.LFM3_OMNI_2B.DEFAULT;
const MODEL_URL = import.meta.env.VITE_LFM3_MODEL_URL ?? MODEL_CONFIG.modelPath;
const TOKENIZER_URL = import.meta.env.VITE_LFM3_TOKENIZER_URL ?? MODEL_CONFIG.tokenizerPath;
const REFERENCE_CASES = [
  {
    id: 'arithmetic',
    messages: [{ role: 'user', content: 'What is 6 times 7? Reply with only the number.' }],
  },
  {
    id: 'capital',
    messages: [
      { role: 'user', content: 'What is the capital of France? Reply with only the city.' },
    ],
  },
  {
    id: 'color',
    messages: [
      { role: 'user', content: 'What color is a clear daytime sky? Reply with one word.' },
    ],
  },
] satisfies readonly { id: string; messages: readonly Lfm3Message[] }[];

const gpuBufferBytes = (): number =>
  [...liveBuffers()].reduce((sum, buffer) => sum + buffer.size, 0);

function hostHeapBytes(): number | null {
  const memory = Reflect.get(performance, 'memory');
  const bytes =
    typeof memory === 'object' && memory !== null
      ? Reflect.get(memory, 'usedJSHeapSize')
      : undefined;
  return typeof bytes === 'number' ? bytes : null;
}

it.skipIf(MODE !== 'base' && MODE !== 'q4')(
  `loads the ${MODE === 'q4' ? 'published Q4_K_M GGUF' : 'original checkpoint'} and streams text through a packed public consumer`,
  async (ctx) => {
    await gpu();
    const before = liveBuffers();
    const invalidSource: RangeSource = { read: async () => new Uint8Array(8) };
    const loadController = new AbortController();
    loadController.abort('qualification checks pre-abort');
    await expect(
      createLfm3Chat(
        { ...MODEL_CONFIG, modelPath: invalidSource },
        { signal: loadController.signal },
      ),
    ).rejects.toMatchObject({ code: 'LOAD_ABORTED' });

    await expect(
      createLfm3Chat(
        { ...MODEL_CONFIG, modelPath: invalidSource, tokenizerPath: TOKENIZER_URL },
        { cacheId: 'lfm3-invalid' },
      ),
    ).rejects.toMatchObject({ code: 'LOAD_FAILED' });

    if (typeof MODEL_URL !== 'string' || typeof TOKENIZER_URL !== 'string') {
      throw new Error('LFM3 model and tokenizer URLs must be strings');
    }
    let sourceBytes = 0;
    const source: RangeSource = {
      async read(begin, end) {
        const response = await fetch(MODEL_URL, {
          headers: { Range: `bytes=${begin}-${end - 1}` },
        });
        if (response.status !== 206) {
          throw new Error(`LFM3 RangeSource expected HTTP 206, got ${response.status}`);
        }
        const data = new Uint8Array(await response.arrayBuffer());
        if (data.byteLength !== end - begin) {
          throw new Error(`LFM3 RangeSource returned ${data.byteLength} bytes for ${end - begin}`);
        }
        sourceBytes += data.byteLength;
        return data;
      },
    };

    let progressDone = 0;
    let progressTotal = 0;
    let peakGpuBytes = gpuBufferBytes();
    let peakJsHeapBytes = hostHeapBytes();
    const loadStarted = performance.now();
    const session = await createLfm3Chat(
      { ...MODEL_CONFIG, modelPath: source, tokenizerPath: TOKENIZER_URL },
      {
        cacheId: MODE === 'q4' ? 'lfm3-omni-q4km-public-3d29af2a' : 'lfm3-original-base-21d75123',
        onProgress(_name, done, total) {
          progressDone = done;
          progressTotal = total;
          peakGpuBytes = Math.max(peakGpuBytes, gpuBufferBytes());
          const heap = hostHeapBytes();
          if (heap !== null) peakJsHeapBytes = Math.max(peakJsHeapBytes ?? 0, heap);
        },
      },
    );
    ctx.onTestFinished(() => session.dispose());
    const loadMs = performance.now() - loadStarted;
    const loadedGpuBytes = gpuBufferBytes();
    const hostBytes = hostHeapBytes();
    peakGpuBytes = Math.max(peakGpuBytes, loadedGpuBytes);
    if (hostBytes !== null) peakJsHeapBytes = Math.max(peakJsHeapBytes ?? 0, hostBytes);

    const request = async (
      content: string,
      maxNewTokens = 24,
    ): Promise<{ text: string; latencyMs: number }> => {
      const started = performance.now();
      const generation = session.generate([{ role: 'user', content }], { maxNewTokens });
      let streamed = '';
      for await (const delta of generation) streamed += delta;
      const result = await generation.result;
      expect(streamed).toBe(result.text);
      expect(result.inputTokens).toBeGreaterThan(0);
      peakGpuBytes = Math.max(peakGpuBytes, gpuBufferBytes());
      const heap = hostHeapBytes();
      if (heap !== null) peakJsHeapBytes = Math.max(peakJsHeapBytes ?? 0, heap);
      return { text: result.text, latencyMs: performance.now() - started };
    };

    const first = await request('Say hello in one short sentence.');
    const second = await request('Say goodbye in one short sentence.');
    expect(first.text.trim().length).toBeGreaterThan(0);
    expect(second.text.trim().length).toBeGreaterThan(0);
    const referenceAnswers: { id: string; answer: string }[] = [];
    if (Q4_MODE) {
      for (const testCase of REFERENCE_CASES) {
        const fixture = liquidReference.fixtures.find((entry) => entry.id === testCase.id);
        if (!fixture) throw new Error(`Liquid reference fixture '${testCase.id}' is missing`);
        const result = await request(
          testCase.messages[0]!.content,
          fixture.generatedTokenIds.length,
        );
        expect(result.text.trim()).toBe(fixture.answer.trim());
        referenceAnswers.push({ id: testCase.id, answer: result.text.trim() });
      }
    }
    const abortController = new AbortController();
    const cancelled = session.generate([{ role: 'user', content: 'Write a short greeting.' }], {
      maxNewTokens: 24,
      signal: abortController.signal,
    });
    const firstUpdate = await cancelled[Symbol.asyncIterator]().next();
    expect(firstUpdate.done).toBe(false);
    abortController.abort('qualification cancellation');
    await expect(cancelled.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
    await session.reset();
    const recovered = await request('Say hello again briefly.');
    expect(recovered.text.trim().length).toBeGreaterThan(0);
    await session.dispose();
    const remainingBuffers = [...liveBuffers()].filter((buffer) => !before.has(buffer));
    const report = {
      checkpointRevision:
        MODE === 'q4'
          ? '3d29af2ad166f8299c2d820937e47dc0add20ffc'
          : '21d75123f86c1f7ffb4bef1ea202710c4e7ebe67',
      modelFormat: Q4_MODE ? 'gguf' : 'safetensors',
      rangeSource: true,
      bytesRead: sourceBytes,
      progressDone,
      progressTotal,
      loadAndWarmUpMs: +loadMs.toFixed(2),
      firstGenerationMs: +first.latencyMs.toFixed(2),
      secondGenerationMs: +second.latencyMs.toFixed(2),
      firstText: first.text,
      secondText: second.text,
      liquidReferenceAnswers: referenceAnswers,
      recoveredTextAfterCancellation: recovered.text,
      gpuBuffersAfterLoadBytes: loadedGpuBytes,
      peakGpuBuffersBytes: peakGpuBytes,
      peakJsHeapBytes,
      hostJsHeapBytesAfterLoad: hostBytes,
      gpuStorageBuffersAfterDispose: remainingBuffers
        .filter((buffer) => (buffer.usage & GPUBufferUsage.STORAGE) !== 0)
        .map((buffer) => ({ bytes: buffer.size, usage: buffer.usage })),
      gpuRootCacheBufferCountAfterDispose: remainingBuffers.length,
      gpuRootCacheBufferBytesAfterDispose: remainingBuffers.reduce(
        (sum, buffer) => sum + buffer.size,
        0,
      ),
    };
    console.info(`LFM3_PUBLIC_QUALIFICATION ${JSON.stringify(report)}`);
    expect(progressDone).toBe(progressTotal);
    expect(peakGpuBytes).toBeGreaterThan(0);
    expect(report.gpuStorageBuffersAfterDispose).toEqual([]);
  },
  1_200_000,
);
