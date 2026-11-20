/** Opt-in comparison of cached decode with a full forward on original weights. */

import { expect, it } from 'vitest';
import { evalValues, gpuExecutor, toArray, type Value } from '../../../src/core/index.ts';
import { openWeights, fetchJson } from '../../../src/zoo/load.ts';
import {
  cacheValues,
  LFM3,
  Lfm3CausalLM,
  type Lfm3ModelStep,
} from '../../../src/zoo/lfm3/model.ts';
import { createLfm3Tokenizer, LFM3_TOKENIZER_URL } from '../../../src/zoo/lfm3/tokenizer.ts';
import { gpu, liveBuffers } from '../../gpu.ts';

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
const MODEL_URL =
  import.meta.env.VITE_LFM3_MODEL_URL ??
  `https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/21d75123f86c1f7ffb4bef1ea202710c4e7ebe67/model.safetensors`;
const TOKENIZER_URL = import.meta.env.VITE_LFM3_TOKENIZER_URL ?? LFM3_TOKENIZER_URL;
const PROBABILITY_TOLERANCE = 0.001;

function greedy(logits: Float32Array): number {
  let token = 0;
  for (let index = 1; index < logits.length; index++) {
    if (logits[index]! > logits[token]!) token = index;
  }
  return token;
}

function maxProbabilityError(left: Float32Array, right: Float32Array): number {
  const softmax = (logits: Float32Array): Float64Array => {
    let max = -Infinity;
    for (const value of logits) max = Math.max(max, value);
    const probabilities = new Float64Array(logits.length);
    let total = 0;
    for (let index = 0; index < logits.length; index++) {
      const value = Math.exp(logits[index]! - max);
      probabilities[index] = value;
      total += value;
    }
    for (let index = 0; index < probabilities.length; index++) {
      probabilities[index] = probabilities[index]! / total;
    }
    return probabilities;
  };
  const a = softmax(left);
  const b = softmax(right);
  let error = 0;
  for (let index = 0; index < a.length; index++) {
    error = Math.max(error, Math.abs(a[index]! - b[index]!));
  }
  return error;
}

function destroyUploads(values: readonly Value[]): void {
  for (const value of values) {
    if (value.state === 'materialized') value.buffer.destroy();
  }
}

it.skipIf(MODE !== 'parity')(
  'matches cached decode with full-forward LFM3 tokens and probabilities',
  async () => {
    const { root } = await gpu();
    const before = liveBuffers();
    const executor = gpuExecutor(root);
    const model = new Lfm3CausalLM();
    const retained: Value[][] = [];

    const evaluate = async (step: Lfm3ModelStep): Promise<Float32Array> => {
      const cache = cacheValues(step.cache);
      retained.push(cache);
      evalValues([step.logits, ...cache], executor);
      const logits = await toArray(step.logits, executor);
      await root.device.queue.onSubmittedWorkDone();
      executor.releaseValues([step.logits]);
      destroyUploads(step.uploads);
      return logits;
    };

    try {
      const weights = await openWeights(MODEL_URL);
      const tokenizerJson = await fetchJson<unknown>(TOKENIZER_URL);
      const tokenizer = createLfm3Tokenizer(tokenizerJson);
      const loadStarted = performance.now();
      await model.loadStateDict(weights, { root });
      const loadMs = performance.now() - loadStarted;
      const prompt = tokenizer.encodeChat([{ role: 'user', content: 'Hello.' }]);
      const prefillStarted = performance.now();
      const prefill = model.forward(prompt, root);
      const prefillLogits = await evaluate(prefill);
      const prefillMs = performance.now() - prefillStarted;
      let cache = prefill.cache;
      let nextToken = greedy(prefillLogits);
      let cachedMs = 0;
      let fullMs = 0;
      let maxError = 0;
      const traceTokens: number[] = [];
      for (let stepIndex = 0; stepIndex < 3; stepIndex++) {
        traceTokens.push(nextToken);
        const prefix = [...prompt, ...traceTokens];
        const cachedStarted = performance.now();
        const cachedStep = model.forward([nextToken], root, cache);
        const cachedLogits = await evaluate(cachedStep);
        cachedMs += performance.now() - cachedStarted;
        expect(cachedLogits).toHaveLength(LFM3.vocab);

        const fullStarted = performance.now();
        const fullStep = model.forward(prefix, root);
        const fullLogits = await evaluate(fullStep);
        fullMs += performance.now() - fullStarted;
        expect(fullLogits).toHaveLength(LFM3.vocab);

        const error = maxProbabilityError(cachedLogits, fullLogits);
        maxError = Math.max(maxError, error);
        expect(greedy(cachedLogits)).toBe(greedy(fullLogits));
        expect(error).toBeLessThanOrEqual(PROBABILITY_TOLERANCE);
        cache = cachedStep.cache;
        nextToken = greedy(cachedLogits);
      }
      const report = {
        checkpointRevision: '21d75123f86c1f7ffb4bef1ea202710c4e7ebe67',
        loadedCheckpointBytes: weights.totalBytes(),
        promptTokens: prompt.length,
        nativeGreedyPrefixTokens: traceTokens,
        cachedFinalGreedyToken: nextToken,
        maxProbabilityError: maxError,
        tolerance: PROBABILITY_TOLERANCE,
        loadMs: +loadMs.toFixed(2),
        prefillMs: +prefillMs.toFixed(2),
        cachedDecodeMs: +cachedMs.toFixed(2),
        fullForwardMs: +fullMs.toFixed(2),
      };
      console.info(`LFM3_CACHE_PARITY ${JSON.stringify(report)}`);
    } finally {
      await root.device.queue.onSubmittedWorkDone().catch(() => undefined);
      for (const values of retained) {
        try {
          executor.releaseValues(values);
        } catch {
          // A failed device dispatch may already have invalidated pooled targets.
        }
      }
      model.dispose();
      executor.dispose();
    }
    const remaining = [...liveBuffers()].filter((buffer) => !before.has(buffer));
    const storage = remaining.filter((buffer) => (buffer.usage & GPUBufferUsage.STORAGE) !== 0);
    console.info(
      `LFM3_CACHE_PARITY_CLEANUP ${JSON.stringify({
        storageBufferCount: storage.length,
        rootCacheBufferCount: remaining.length - storage.length,
      })}`,
    );
    expect(storage).toEqual([]);
  },
  1_200_000,
);
