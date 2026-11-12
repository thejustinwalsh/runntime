/** Compares native Q4_K_M hidden states and greedy steps with pinned Liquid vectors. */

import { expect, it } from 'vitest';
import {
  evalValues,
  fromGguf,
  gpuExecutor,
  httpRangeSource,
  toArray,
  type Value,
} from '../../../src/core/index.ts';
import { cacheValues, Lfm3CausalLM, type Lfm3Cache } from '../../../src/zoo/lfm3/model.ts';
import { LFM3_Q4_K_M_SOURCE } from '../../../src/zoo/lfm3/gguf.ts';
import { gpu, liveBuffers } from '../../gpu.ts';
import liquidReference from './liquid-reference.json';

declare global {
  interface ImportMetaEnv {
    readonly VITE_LFM3_QUALIFICATION_MODE?: string;
    readonly VITE_LFM3_MODEL_URL?: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const MODE = import.meta.env.VITE_LFM3_QUALIFICATION_MODE;
const MODEL_URL = import.meta.env.VITE_LFM3_MODEL_URL ?? LFM3_Q4_K_M_SOURCE.url;
const HIDDEN_TOLERANCE = 0.25;
const MIN_HIDDEN_COSINE = 0.998;

function destroyUploads(values: readonly Value[]): void {
  for (const value of values) {
    if (value.state === 'materialized') value.buffer.destroy();
  }
}

function greedy(logits: Float32Array): number {
  let token = 0;
  for (let index = 1; index < logits.length; index++) {
    if (logits[index]! > logits[token]!) token = index;
  }
  return token;
}

function hiddenMetrics(actual: Float32Array, expected: readonly number[]) {
  let squaredError = 0;
  let dot = 0;
  let actualNorm = 0;
  let expectedNorm = 0;
  let maxAbsoluteError = 0;
  for (let index = 0; index < expected.length; index++) {
    const left = actual[index]!;
    const right = expected[index]!;
    const difference = left - right;
    squaredError += difference * difference;
    dot += left * right;
    actualNorm += left * left;
    expectedNorm += right * right;
    maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(difference));
  }
  return {
    rmsError: Math.sqrt(squaredError / expected.length),
    maxAbsoluteError,
    cosineSimilarity: dot / Math.sqrt(actualNorm * expectedNorm),
  };
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
  const leftProbabilities = softmax(left);
  const rightProbabilities = softmax(right);
  let error = 0;
  for (let index = 0; index < leftProbabilities.length; index++) {
    error = Math.max(error, Math.abs(leftProbabilities[index]! - rightProbabilities[index]!));
  }
  return error;
}

it.skipIf(MODE !== 'native-reference')(
  'matches native Q4_K_M teacher-forced vectors and next-token ids against Liquid',
  async () => {
    expect(liquidReference.engine).toMatchObject({
      revision: '9c2fb176f4c222da394e8037d3c0d0837f70b978',
      sha256: '77821949412f5dd6d60e37f22415f0e71454e65f6a73928529c6fbec0f2cf2cb',
    });
    expect(liquidReference.model).toMatchObject({
      revision: LFM3_Q4_K_M_SOURCE.revision,
      sha256: LFM3_Q4_K_M_SOURCE.sha256,
    });

    const { root } = await gpu();
    const before = liveBuffers();
    const executor = gpuExecutor(root);
    const model = new Lfm3CausalLM();
    let activeCache: Lfm3Cache | undefined;
    const measurements: {
      id: string;
      tokenId: number;
      predictedTokenId: number;
      fullForwardPredictedTokenId: number;
      cacheProbabilityError: number;
      rmsError: number;
      maxAbsoluteError: number;
      cosineSimilarity: number;
    }[] = [];

    try {
      const weights = await fromGguf(httpRangeSource(MODEL_URL), {
        fileBytes: LFM3_Q4_K_M_SOURCE.bytes,
        sha256: LFM3_Q4_K_M_SOURCE.sha256,
      });
      await model.loadStateDict(weights, { root });

      for (const fixture of liquidReference.fixtures) {
        const captureStart = fixture.inputTokenIds.length - fixture.vectors.length;
        const prefix = fixture.inputTokenIds.slice(0, captureStart);
        const prefill = model.forward(prefix, root);
        activeCache = prefill.cache;
        evalValues(cacheValues(activeCache), executor);
        await root.device.queue.onSubmittedWorkDone();
        destroyUploads(prefill.uploads);

        for (const [vectorIndex, vector] of fixture.vectors.entries()) {
          const previous = activeCache;
          const step = model.forward([vector.tokenId], root, previous);
          activeCache = step.cache;
          evalValues([step.hidden, step.logits, ...cacheValues(activeCache)], executor);
          const [hidden, logits] = await Promise.all([
            toArray(step.hidden, executor),
            toArray(step.logits, executor),
          ]);
          await root.device.queue.onSubmittedWorkDone();
          executor.releaseValues([step.hidden, step.logits]);
          if (previous) executor.releaseValues(cacheValues(previous));
          destroyUploads(step.uploads);

          const metrics = hiddenMetrics(hidden, vector.values);
          const predictedTokenId = greedy(logits);
          const fullStep = model.forward(
            fixture.inputTokenIds.slice(0, captureStart + vectorIndex + 1),
            root,
          );
          let fullLogits: Float32Array;
          try {
            evalValues([fullStep.logits, ...cacheValues(fullStep.cache)], executor);
            fullLogits = await toArray(fullStep.logits, executor);
            await root.device.queue.onSubmittedWorkDone();
          } finally {
            executor.releaseValues([fullStep.logits, ...cacheValues(fullStep.cache)]);
            destroyUploads(fullStep.uploads);
          }
          const fullForwardPredictedTokenId = greedy(fullLogits);
          const cacheProbabilityError = maxProbabilityError(logits, fullLogits);
          expect(predictedTokenId).toBe(vector.predictedTokenId);
          expect(predictedTokenId).toBe(fullForwardPredictedTokenId);
          expect(cacheProbabilityError).toBeLessThanOrEqual(0.001);
          expect(metrics.rmsError).toBeLessThan(HIDDEN_TOLERANCE);
          expect(metrics.cosineSimilarity).toBeGreaterThan(MIN_HIDDEN_COSINE);
          measurements.push({
            id: fixture.id,
            tokenId: vector.tokenId,
            predictedTokenId,
            fullForwardPredictedTokenId,
            cacheProbabilityError,
            ...metrics,
          });
        }
        executor.releaseValues(cacheValues(activeCache));
        activeCache = undefined;
      }
    } finally {
      if (activeCache) executor.releaseValues(cacheValues(activeCache));
      await root.device.queue.onSubmittedWorkDone().catch(() => undefined);
      model.dispose();
      executor.dispose();
    }

    const remaining = [...liveBuffers()].filter((buffer) => !before.has(buffer));
    const storage = remaining.filter((buffer) => (buffer.usage & GPUBufferUsage.STORAGE) !== 0);
    expect(storage).toEqual([]);
    console.info(
      `LFM3_NATIVE_REFERENCE ${JSON.stringify({
        modelRevision: LFM3_Q4_K_M_SOURCE.revision,
        modelSha256: LFM3_Q4_K_M_SOURCE.sha256,
        vectors: measurements.length,
        maxCacheProbabilityError: Math.max(
          ...measurements.map((entry) => entry.cacheProbabilityError),
        ),
        maxRmsError: Math.max(...measurements.map((entry) => entry.rmsError)),
        minCosineSimilarity: Math.min(...measurements.map((entry) => entry.cosineSimilarity)),
        maxAbsoluteError: Math.max(...measurements.map((entry) => entry.maxAbsoluteError)),
        storageBuffersAfterDispose: storage.length,
        measurements,
      })}`,
    );
  },
  1_200_000,
);
