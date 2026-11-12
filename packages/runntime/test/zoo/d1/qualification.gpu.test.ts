/** Opt-in pinned D1 reference qualification on a real WebGPU device. */

import { expect, it } from 'vitest';
import tgpu from 'typegpu';
import {
  fromSafetensors,
  type LazyStateDict,
  type LazyTensor,
} from '../../../src/core/weights/safetensors.ts';
import { tensor } from '../../../src/core/gpu/buffers.ts';
import { initRunntime, toArray } from '../../../src/core/gpu/eval.ts';
import { add } from '../../../src/core/graph/ops.ts';
import { createD1Decider, type D1Decider } from '../../../src/zoo/d1/decider.ts';
import type { Answer, D1Tokenizer, Question } from '../../../src/zoo/d1/prompt.ts';
import { buffersSince, expectClose, liveBuffers, gpu } from '../../gpu.ts';
import vectors from '../../../playground/d1/reference-vectors.json';

declare global {
  interface ImportMetaEnv {
    readonly VITE_D1_QUALIFICATION_MODE?: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const MODEL = 'LiquidAI/d1-omni-600M';
const REVISION = '02b55d7076f15129e59ab3f94783f32c4b088674';
const MODEL_URL = `https://huggingface.co/${MODEL}/resolve/${REVISION}/model.safetensors`;
const MODE = import.meta.env.VITE_D1_QUALIFICATION_MODE;
const TOLERANCE = 0.001;

interface ReferenceCase {
  id: string;
  state: string;
  question: Question;
  reference: {
    probabilities: number[];
    answer: string;
    inputTokens: number;
    temperatureKey: string;
    temperature: number;
  };
}

interface ReferenceCorpus {
  model: string;
  revision: string;
  probabilityTolerance: number;
  tokenizer: {
    revision: string;
    bosId: number;
    specialTokenIds: Record<string, number>;
    inputs: { text: string; ids: number[] }[];
  };
  cases: ReferenceCase[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isReferenceCase(value: unknown): value is ReferenceCase {
  if (!isRecord(value)) return false;
  const candidate = value;
  const reference = candidate.reference;
  if (!isRecord(reference)) return false;
  return (
    typeof candidate.id === 'string' &&
    typeof candidate.state === 'string' &&
    isRecord(candidate.question) &&
    typeof candidate.question.type === 'string' &&
    typeof candidate.question.instructions === 'string' &&
    Array.isArray(reference.probabilities) &&
    reference.probabilities.every((probability) => typeof probability === 'number') &&
    typeof reference.answer === 'string' &&
    typeof reference.inputTokens === 'number' &&
    typeof reference.temperatureKey === 'string' &&
    typeof reference.temperature === 'number'
  );
}

function isTokenizerInput(value: unknown): value is { text: string; ids: number[] } {
  return (
    isRecord(value) &&
    typeof value.text === 'string' &&
    Array.isArray(value.ids) &&
    value.ids.every((id) => typeof id === 'number')
  );
}

function isReferenceCorpus(value: unknown): value is ReferenceCorpus {
  if (!isRecord(value)) return false;
  const candidate = value;
  const tokenizer = candidate.tokenizer;
  if (!isRecord(tokenizer) || !isRecord(tokenizer.specialTokenIds)) return false;
  return (
    candidate.model === MODEL &&
    candidate.revision === REVISION &&
    candidate.probabilityTolerance === TOLERANCE &&
    tokenizer.revision === REVISION &&
    typeof tokenizer.bosId === 'number' &&
    Object.values(tokenizer.specialTokenIds).every((id) => typeof id === 'number') &&
    Array.isArray(tokenizer.inputs) &&
    tokenizer.inputs.every(isTokenizerInput) &&
    Array.isArray(candidate.cases) &&
    candidate.cases.every(isReferenceCase)
  );
}

function requireReferenceCorpus(value: unknown): ReferenceCorpus {
  if (!isReferenceCorpus(value)) throw new Error('pinned D1 reference corpus is malformed');
  return value;
}

function makeTokenizer(corpus: ReferenceCorpus): D1Tokenizer {
  const encodings = new Map(corpus.tokenizer.inputs.map(({ text, ids }) => [text, ids]));
  return {
    encode: (text) => {
      const ids = encodings.get(text);
      if (ids === undefined)
        throw new Error(`missing pinned tokenizer input ${JSON.stringify(text)}`);
      return [...ids];
    },
    tokenId: (token) => {
      const id = corpus.tokenizer.specialTokenIds[token];
      if (typeof id !== 'number') throw new Error(`missing D1 token ${token}`);
      return id;
    },
    bosId: corpus.tokenizer.bosId,
  };
}

function probabilities(answer: Answer, question: Question): number[] {
  if (answer.type === 'noul') return [answer.noul, 1 - answer.noul];
  if (answer.type === 'choice' && question.type === 'choice') {
    return Object.keys(question.criteria).map((name) => answer.probabilities[name]!);
  }
  if (answer.type === 'score' && question.type === 'score') {
    return question.criteria.map((_, index) => answer.probabilities[String(index)]!);
  }
  throw new Error(`D1 answer type did not match ${question.type}`);
}

function stableAnswer(answer: Answer, question: Question): string {
  if (answer.type === 'noul') return answer.noul >= 0.5 ? 'yes' : 'no';
  if (answer.type === 'choice') return answer.choice;
  if (question.type !== 'score') throw new Error('score answer has a non-score question');
  let best = 0;
  for (let i = 1; i < question.criteria.length; i++) {
    if (answer.probabilities[String(i)]! > answer.probabilities[String(best)]!) best = i;
  }
  return String(best);
}

function gpuBufferBytes(): number {
  return [...liveBuffers()].reduce((sum, buffer) => sum + buffer.size, 0);
}

function cloneStateDict(sd: LazyStateDict, tensors: Map<string, LazyTensor>): LazyStateDict {
  return {
    metadata: sd.metadata,
    tensors,
    totalBytes: () => [...tensors.values()].reduce((sum, tensor) => sum + tensor.byteLength, 0),
  };
}

function temperatures(corpus: ReferenceCorpus): Record<string, number> {
  return Object.fromEntries(
    corpus.cases.map((item) => [item.reference.temperatureKey, item.reference.temperature]),
  );
}

async function hostMemory(): Promise<{ jsHeapBytes: number | null; agentBytes: number | null }> {
  const memory = Reflect.get(performance, 'memory');
  const jsHeapBytes =
    typeof memory === 'object' &&
    memory !== null &&
    typeof Reflect.get(memory, 'usedJSHeapSize') === 'number'
      ? Reflect.get(memory, 'usedJSHeapSize')
      : null;
  const measure = Reflect.get(performance, 'measureUserAgentSpecificMemory');
  if (typeof measure !== 'function') return { jsHeapBytes, agentBytes: null };
  try {
    const sample: unknown = await measure.call(performance);
    const bytes =
      typeof sample === 'object' && sample !== null ? Reflect.get(sample, 'bytes') : undefined;
    return { jsHeapBytes, agentBytes: typeof bytes === 'number' ? bytes : null };
  } catch {
    return { jsHeapBytes, agentBytes: null };
  }
}

it.skipIf(MODE !== 'baseline')(
  'records the original D1 branch against the pinned Liquid text corpus',
  async (ctx) => {
    const corpus = requireReferenceCorpus(vectors);
    const started = performance.now();
    const { root } = await gpu();
    const before = liveBuffers();
    let bytesReceived = 0;
    let loadedWeightBytes = 0;
    let progressDoneBytes = 0;
    let progressTotalBytes = 0;
    let peakGpuBufferBytes = gpuBufferBytes();
    let peakJsHeapBytes: number | null;
    let peakAgentBytes: number | null;
    const sd = await fromSafetensors(MODEL_URL, {
      onBytes: (bytes) => (bytesReceived += bytes),
    });
    const tokenizer = makeTokenizer(corpus);
    const loadStarted = performance.now();
    const decider = await createD1Decider(sd, tokenizer, {
      temperatures: temperatures(corpus),
      onProgress: (_name, done, total) => {
        loadedWeightBytes = total;
        progressDoneBytes = done;
        progressTotalBytes = total;
        peakGpuBufferBytes = Math.max(peakGpuBufferBytes, gpuBufferBytes());
      },
    });
    const loadMs = performance.now() - loadStarted;
    const afterLoadMemory = await hostMemory();
    peakJsHeapBytes = afterLoadMemory.jsHeapBytes;
    peakAgentBytes = afterLoadMemory.agentBytes;
    peakGpuBufferBytes = Math.max(peakGpuBufferBytes, gpuBufferBytes());

    const decisions: Record<string, unknown>[] = [];
    for (const item of corpus.cases) {
      const t = performance.now();
      const result = await decider.systemOne(item.state, { answer: item.question });
      const latencyMs = performance.now() - t;
      const answer = result.answers.answer!;
      const got = probabilities(answer, item.question);
      const maxAbsError = Math.max(
        ...got.map((p, index) => Math.abs(p - item.reference.probabilities[index]!)),
      );
      expect(result.inputTokens).toBe(item.reference.inputTokens);
      expect(stableAnswer(answer, item.question)).toBe(item.reference.answer);
      expect(maxAbsError).toBeLessThanOrEqual(TOLERANCE);
      decisions.push({
        id: item.id,
        inputTokens: result.inputTokens,
        answer: stableAnswer(answer, item.question),
        probabilities: got,
        maxAbsError,
        latencyMs: +latencyMs.toFixed(2),
      });
      peakGpuBufferBytes = Math.max(peakGpuBufferBytes, gpuBufferBytes());
      const memory = await hostMemory();
      if (memory.jsHeapBytes !== null)
        peakJsHeapBytes = Math.max(peakJsHeapBytes ?? 0, memory.jsHeapBytes);
      if (memory.agentBytes !== null)
        peakAgentBytes = Math.max(peakAgentBytes ?? 0, memory.agentBytes);
    }

    const disposeStarted = performance.now();
    await decider.dispose();
    await root.device.queue.onSubmittedWorkDone();
    const releaseMs = performance.now() - disposeStarted;
    const afterDispose = [...liveBuffers()].filter((buffer) => !before.has(buffer));
    const report = {
      mode: MODE,
      branchBase: '68f3a12506bada2c9ac6cb3629426f3e06a56349',
      model: MODEL,
      revision: REVISION,
      elapsedMs: +(performance.now() - started).toFixed(2),
      loadMs: +loadMs.toFixed(2),
      loadedWeightBytes,
      bytesReceived,
      progressDoneBytes,
      progressTotalBytes,
      decisions,
      peakJsHeapBytes,
      peakAgentBytes,
      peakGpuBufferBytes,
      gpuBufferBytesAfterDispose: afterDispose.reduce((sum, buffer) => sum + buffer.size, 0),
      gpuBufferCountAfterDispose: afterDispose.length,
      releaseMs: +releaseMs.toFixed(2),
      measurements: {
        hostPeak:
          peakJsHeapBytes === null && peakAgentBytes === null
            ? 'unknown: browser exposes no host peak'
            : 'sampled browser memory; not a process-wide peak',
        gpuPeak: 'sum of live GPUBuffer.size snapshots; driver-private allocations are unknown',
        release: 'original decider release; shared executor allocations remain process-owned',
      },
    };
    console.info('D1_QUALIFICATION', JSON.stringify(report));
    await ctx.annotate('D1_QUALIFICATION', {
      contentType: 'application/json',
      body: JSON.stringify(report, null, 2),
    });
    expect(decisions).toHaveLength(corpus.cases.length);
  },
  900_000,
);

it.skipIf(MODE !== 'repaired')(
  'qualifies D1 references, resource cleanup, serialization, abort, disposal, and device recovery',
  async (ctx) => {
    const corpus = requireReferenceCorpus(vectors);
    const started = performance.now();
    const { root } = await gpu();
    let bytesReceived = 0;
    const original = await fromSafetensors(MODEL_URL, {
      onBytes: (bytes) => (bytesReceived += bytes),
    });
    const rawTensors = new Map(original.tensors);
    const stateDict = () => cloneStateDict(original, new Map(rawTensors));
    const tokenizer = makeTokenizer(corpus);
    const temps = temperatures(corpus);

    const beforeFailedLoad = liveBuffers();
    const loadAbort = new AbortController();
    await expect(
      createD1Decider(stateDict(), tokenizer, {
        signal: loadAbort.signal,
        onProgress: () => loadAbort.abort('qualification abort after first upload'),
      }),
    ).rejects.toMatchObject({ code: 'LOAD_ABORTED' });
    expect(buffersSince(beforeFailedLoad)).toEqual([]);

    let loadedWeightBytes = 0;
    let progressDoneBytes = 0;
    let progressTotalBytes = 0;
    let peakGpuBufferBytes = gpuBufferBytes();
    let peakJsHeapBytes: number | null;
    let peakAgentBytes: number | null;
    const loadStarted = performance.now();
    const decider = await createD1Decider(stateDict(), tokenizer, {
      temperatures: temps,
      onProgress: (_name, done, total) => {
        loadedWeightBytes = total;
        progressDoneBytes = done;
        progressTotalBytes = total;
        peakGpuBufferBytes = Math.max(peakGpuBufferBytes, gpuBufferBytes());
      },
    });
    const loadMs = performance.now() - loadStarted;
    const afterLoadMemory = await hostMemory();
    peakJsHeapBytes = afterLoadMemory.jsHeapBytes;
    peakAgentBytes = afterLoadMemory.agentBytes;
    peakGpuBufferBytes = Math.max(peakGpuBufferBytes, gpuBufferBytes());

    const decisions: Record<string, unknown>[] = [];
    for (const item of corpus.cases) {
      const t = performance.now();
      const result = await decider.systemOne(item.state, { answer: item.question });
      const latencyMs = performance.now() - t;
      const answer = result.answers.answer!;
      const got = probabilities(answer, item.question);
      const maxAbsError = Math.max(
        ...got.map((p, index) => Math.abs(p - item.reference.probabilities[index]!)),
      );
      expect(result.inputTokens).toBe(item.reference.inputTokens);
      expect(stableAnswer(answer, item.question)).toBe(item.reference.answer);
      expect(maxAbsError).toBeLessThanOrEqual(TOLERANCE);
      decisions.push({
        id: item.id,
        inputTokens: result.inputTokens,
        answer: stableAnswer(answer, item.question),
        probabilities: got,
        maxAbsError,
        latencyMs: +latencyMs.toFixed(2),
      });
      peakGpuBufferBytes = Math.max(peakGpuBufferBytes, gpuBufferBytes());
      const memory = await hostMemory();
      if (memory.jsHeapBytes !== null)
        peakJsHeapBytes = Math.max(peakJsHeapBytes ?? 0, memory.jsHeapBytes);
      if (memory.agentBytes !== null)
        peakAgentBytes = Math.max(peakAgentBytes ?? 0, memory.agentBytes);
    }

    const beforeReadbackFailure = liveBuffers();
    const originalMapAsync = GPUBuffer.prototype.mapAsync;
    let rejectReadback = true;
    GPUBuffer.prototype.mapAsync = function (mode, offset, size) {
      if (rejectReadback) {
        rejectReadback = false;
        return Promise.reject(new Error('qualification rejected readback'));
      }
      return originalMapAsync.call(this, mode, offset, size);
    };
    try {
      await expect(
        decider.systemOne(corpus.cases[0]!.state, { answer: corpus.cases[0]!.question }),
      ).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
    } finally {
      GPUBuffer.prototype.mapAsync = originalMapAsync;
    }
    await root.device.queue.onSubmittedWorkDone();
    expect(buffersSince(beforeReadbackFailure)).toEqual([]);

    const aborted = new AbortController();
    aborted.abort('qualification pre-abort');
    const beforeAbort = liveBuffers();
    await expect(
      decider.systemOne(corpus.cases[0]!.state, { answer: corpus.cases[0]!.question }, undefined, {
        signal: aborted.signal,
      }),
    ).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
    expect(buffersSince(beforeAbort)).toEqual([]);

    const concurrentStarted = performance.now();
    const concurrentResults = await Promise.all(
      corpus.cases
        .slice(0, 2)
        .map((item) => decider.systemOne(item.state, { answer: item.question })),
    );
    expect(
      concurrentResults.map((result, index) =>
        stableAnswer(result.answers.answer!, corpus.cases[index]!.question),
      ),
    ).toEqual(corpus.cases.slice(0, 2).map((item) => item.reference.answer));
    const serializationMs = performance.now() - concurrentStarted;

    const queuedAbort = new AbortController();
    const first = decider.systemOne(corpus.cases[0]!.state, {
      answer: corpus.cases[0]!.question,
    });
    const second = decider.systemOne(
      corpus.cases[1]!.state,
      { answer: corpus.cases[1]!.question },
      undefined,
      { signal: queuedAbort.signal },
    );
    queuedAbort.abort('qualification queued abort');
    const serialResult = await first;
    await expect(second).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
    expect(stableAnswer(serialResult.answers.answer!, corpus.cases[0]!.question)).toBe(
      corpus.cases[0]!.reference.answer,
    );

    const activeAbort = new AbortController();
    const abortWork = decider.systemOne(
      corpus.cases[0]!.state,
      { answer: corpus.cases[0]!.question },
      undefined,
      { signal: activeAbort.signal },
    );
    await Promise.resolve();
    activeAbort.abort('qualification active abort');
    await expect(abortWork).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });

    const lossCase = corpus.cases[0]!;
    const lossStarted = performance.now();
    const lostWork = decider.systemOne(lossCase.state, { answer: lossCase.question });
    await Promise.resolve();
    root.device.destroy();
    const disposalStarted = performance.now();
    const disposal = decider.dispose();
    await expect(lostWork).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
    await disposal;
    const releaseMs = performance.now() - disposalStarted;
    const deviceLossMs = performance.now() - lossStarted;
    const afterDisposeBuffers = [...liveBuffers()].filter(
      (buffer) => !beforeFailedLoad.has(buffer),
    );

    const recoveredRoot = await tgpu.init({
      device: { optionalFeatures: ['subgroups', 'shader-f16'] },
    });
    initRunntime(recoveredRoot);
    const recoveryLoadStarted = performance.now();
    const recovered = await createD1Decider(stateDict(), tokenizer, { temperatures: temps });
    const recoveryLoadMs = performance.now() - recoveryLoadStarted;
    const recoveryCase = corpus.cases[0]!;
    const recoveryStarted = performance.now();
    const recoveryResult = await recovered.systemOne(recoveryCase.state, {
      answer: recoveryCase.question,
    });
    const recoveryMs = performance.now() - recoveryStarted;
    const recoveryAnswer = recoveryResult.answers.answer!;
    const recoveryProbabilities = probabilities(recoveryAnswer, recoveryCase.question);
    expect(recoveryResult.inputTokens).toBe(recoveryCase.reference.inputTokens);
    expect(stableAnswer(recoveryAnswer, recoveryCase.question)).toBe(recoveryCase.reference.answer);
    expect(
      Math.max(
        ...recoveryProbabilities.map((p, index) =>
          Math.abs(p - recoveryCase.reference.probabilities[index]!),
        ),
      ),
    ).toBeLessThanOrEqual(TOLERANCE);
    const beforeRecoveryDispose = liveBuffers();
    const recoveryDisposeStarted = performance.now();
    await recovered.dispose();
    await recoveredRoot.device.queue.onSubmittedWorkDone();
    const recoveryReleaseMs = performance.now() - recoveryDisposeStarted;
    const recoveryAfterDispose = [...liveBuffers()].filter(
      (buffer) => !beforeRecoveryDispose.has(buffer),
    );

    const report = {
      mode: MODE,
      branchBase: '68f3a12506bada2c9ac6cb3629426f3e06a56349',
      model: MODEL,
      revision: REVISION,
      elapsedMs: +(performance.now() - started).toFixed(2),
      loadMs: +loadMs.toFixed(2),
      loadedWeightBytes,
      bytesReceived,
      progressDoneBytes,
      progressTotalBytes,
      decisions,
      serializationMs: +serializationMs.toFixed(2),
      peakJsHeapBytes,
      peakAgentBytes,
      peakGpuBufferBytes,
      gpuBufferBytesAfterDeviceLoss: afterDisposeBuffers.reduce(
        (sum, buffer) => sum + buffer.size,
        0,
      ),
      gpuBufferCountAfterDeviceLoss: afterDisposeBuffers.length,
      releaseMs: +releaseMs.toFixed(2),
      deviceLossMs: +deviceLossMs.toFixed(2),
      recoveryLoadMs: +recoveryLoadMs.toFixed(2),
      recoveryDecisionMs: +recoveryMs.toFixed(2),
      recoveryInputTokens: recoveryResult.inputTokens,
      recoveryAnswer: stableAnswer(recoveryAnswer, recoveryCase.question),
      recoveryReleaseMs: +recoveryReleaseMs.toFixed(2),
      recoveryGpuBytesAfterDispose: recoveryAfterDispose.reduce(
        (sum, buffer) => sum + buffer.size,
        0,
      ),
      recoveryGpuBufferCountAfterDispose: recoveryAfterDispose.length,
      measurements: {
        hostPeak:
          peakJsHeapBytes === null && peakAgentBytes === null
            ? 'unknown: browser exposes no host peak'
            : 'sampled browser memory; not a process-wide peak',
        gpuPeak: 'sum of live GPUBuffer.size snapshots; driver-private allocations are unknown',
        gpuAfterDeviceLoss:
          'device.destroy releases the old device allocations; this count is not decider-specific',
        release:
          'recovery decider disposed with zero new tracked buffers; persistent per-root readback staging predates disposal',
      },
    };
    console.info('D1_QUALIFICATION', JSON.stringify(report));
    await ctx.annotate('D1_QUALIFICATION', {
      contentType: 'application/json',
      body: JSON.stringify(report, null, 2),
    });
    expect(decisions).toHaveLength(corpus.cases.length);
    expect(recoveryAfterDispose.length).toBeLessThanOrEqual(1);
  },
  900_000,
);

it.skipIf(MODE !== 'correction')(
  'qualifies D1 exceptional cleanup, captured-root uploads, and healthy disposal',
  async (ctx) => {
    const corpus = requireReferenceCorpus(vectors);
    const started = performance.now();
    const { root, ex: sharedExecutor } = await gpu();
    const original = await fromSafetensors(MODEL_URL);
    const rawTensors = new Map(original.tensors);
    const stateDict = () => cloneStateDict(original, new Map(rawTensors));
    const tokenizer = makeTokenizer(corpus);
    const temps = temperatures(corpus);
    const sharedInput = tensor(root, Float32Array.of(2, 4, 6, 8, 10, 12, 14, 16), [1, 8]);
    const sharedOutput = add(sharedInput, 3);
    expectClose(await toArray(sharedOutput, sharedExecutor), [5, 7, 9, 11, 13, 15, 17, 19]);
    sharedExecutor.releaseValues([sharedOutput]);

    const beforeAbortedLoad = liveBuffers();
    const loadAbort = new AbortController();
    await expect(
      createD1Decider(stateDict(), tokenizer, {
        signal: loadAbort.signal,
        onProgress: () => loadAbort.abort('correction load abort after first upload'),
      }),
    ).rejects.toMatchObject({ code: 'LOAD_ABORTED' });
    expect(buffersSince(beforeAbortedLoad)).toEqual([]);

    const replacementRoot = await tgpu.init({
      device: { optionalFeatures: ['subgroups', 'shader-f16'] },
    });
    const beforeHealthyCreate = liveBuffers();
    const baselineStaging = [...beforeHealthyCreate].filter(
      (buffer) => (buffer.usage & GPUBufferUsage.MAP_READ) !== 0,
    );
    const baselineUniforms = [...beforeHealthyCreate].filter(
      (buffer) => (buffer.usage & GPUBufferUsage.UNIFORM) !== 0,
    );
    const baselineGpuBufferBytes = [...beforeHealthyCreate].reduce(
      (sum, buffer) => sum + buffer.size,
      0,
    );
    let replacementExecutor: ReturnType<typeof initRunntime> | undefined;
    let changedRootDuringLoad = false;
    let loadedWeightBytes = 0;
    let progressDoneBytes = 0;
    let progressTotalBytes = 0;
    let peakGpuBufferBytes = gpuBufferBytes();
    let peakJsHeapBytes: number | null = null;
    let peakAgentBytes: number | null = null;
    const loadStarted = performance.now();
    let decider: D1Decider | undefined;
    let freshDecider: D1Decider | undefined;
    let freshInput: ReturnType<typeof tensor> | undefined;
    const decisions: Record<string, unknown>[] = [];
    const verifyDecision = (
      result: Awaited<ReturnType<D1Decider['systemOne']>>,
      item: ReferenceCase,
      id = item.id,
    ) => {
      const answer = result.answers.answer!;
      const got = probabilities(answer, item.question);
      const maxAbsError = Math.max(
        ...got.map((probability, index) =>
          Math.abs(probability - item.reference.probabilities[index]!),
        ),
      );
      expect(result.inputTokens).toBe(item.reference.inputTokens);
      expect(stableAnswer(answer, item.question)).toBe(item.reference.answer);
      expect(maxAbsError).toBeLessThanOrEqual(TOLERANCE);
      decisions.push({
        id,
        inputTokens: result.inputTokens,
        answer: stableAnswer(answer, item.question),
        probabilities: got,
        maxAbsError,
      });
      peakGpuBufferBytes = Math.max(peakGpuBufferBytes, gpuBufferBytes());
    };
    const decide = async (item: ReferenceCase, id = item.id) => {
      const decisionStarted = performance.now();
      const result = await decider!.systemOne(item.state, { answer: item.question });
      verifyDecision(result, item, id);
      decisions.at(-1)!.latencyMs = +(performance.now() - decisionStarted).toFixed(2);
      const memory = await hostMemory();
      if (memory.jsHeapBytes !== null)
        peakJsHeapBytes = Math.max(peakJsHeapBytes ?? 0, memory.jsHeapBytes);
      if (memory.agentBytes !== null)
        peakAgentBytes = Math.max(peakAgentBytes ?? 0, memory.agentBytes);
    };

    try {
      decider = await createD1Decider(stateDict(), tokenizer, {
        temperatures: temps,
        onProgress: (_name, done, total) => {
          loadedWeightBytes = total;
          progressDoneBytes = done;
          progressTotalBytes = total;
          peakGpuBufferBytes = Math.max(peakGpuBufferBytes, gpuBufferBytes());
          if (!changedRootDuringLoad) {
            replacementExecutor = initRunntime(replacementRoot);
            changedRootDuringLoad = true;
          }
        },
      });
      const loadMs = performance.now() - loadStarted;
      const loadedMemory = await hostMemory();
      peakJsHeapBytes = loadedMemory.jsHeapBytes;
      peakAgentBytes = loadedMemory.agentBytes;
      peakGpuBufferBytes = Math.max(peakGpuBufferBytes, gpuBufferBytes());
      expect(changedRootDuringLoad).toBe(true);

      for (const item of corpus.cases) await decide(item);
      initRunntime(replacementRoot);
      await decide(corpus.cases[0]!, 'old-decider-after-root-reinitialization');

      const firstCase = corpus.cases[0]!;
      const expectExecutionFailure = (promise: Promise<unknown>) =>
        expect(promise).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });

      const originalDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups;
      let dispatches = 0;
      GPUComputePassEncoder.prototype.dispatchWorkgroups = function (x, y, z) {
        dispatches++;
        if (dispatches === 4) throw new Error('correction rejected command encoding');
        return originalDispatch.call(this, x, y, z);
      };
      try {
        await expectExecutionFailure(
          decider.systemOne(firstCase.state, { answer: firstCase.question }),
        );
      } finally {
        GPUComputePassEncoder.prototype.dispatchWorkgroups = originalDispatch;
      }
      expect(dispatches).toBe(4);
      await decide(firstCase, 'recovered-after-command-encoding-failure');

      const originalSubmit = GPUQueue.prototype.submit;
      let rejectedSubmission = false;
      GPUQueue.prototype.submit = function (commands) {
        if (this === root.device.queue && !rejectedSubmission) {
          rejectedSubmission = true;
          throw new Error('correction rejected queue submission');
        }
        return originalSubmit.call(this, commands);
      };
      try {
        await expectExecutionFailure(
          decider.systemOne(firstCase.state, { answer: firstCase.question }),
        );
      } finally {
        GPUQueue.prototype.submit = originalSubmit;
      }
      expect(rejectedSubmission).toBe(true);
      await decide(firstCase, 'recovered-after-queue-submission-failure');

      const originalCreateBindGroup = GPUDevice.prototype.createBindGroup;
      let bindGroups = 0;
      GPUDevice.prototype.createBindGroup = function (descriptor) {
        if (this === root.device && ++bindGroups === 9) {
          throw new Error('correction rejected partial graph evaluation');
        }
        return originalCreateBindGroup.call(this, descriptor);
      };
      try {
        await expectExecutionFailure(
          decider.systemOne(firstCase.state, { answer: firstCase.question }),
        );
      } finally {
        GPUDevice.prototype.createBindGroup = originalCreateBindGroup;
      }
      expect(bindGroups).toBe(9);
      await decide(firstCase, 'recovered-after-partial-evaluation-failure');

      const beforeReadbackFailure = liveBuffers();
      const originalMapAsync = GPUBuffer.prototype.mapAsync;
      let rejectReadback = true;
      GPUBuffer.prototype.mapAsync = function (mode, offset, size) {
        if (rejectReadback) {
          rejectReadback = false;
          return Promise.reject(new Error('correction rejected readback'));
        }
        return originalMapAsync.call(this, mode, offset, size);
      };
      try {
        await expectExecutionFailure(
          decider.systemOne(firstCase.state, { answer: firstCase.question }),
        );
      } finally {
        GPUBuffer.prototype.mapAsync = originalMapAsync;
      }
      await root.device.queue.onSubmittedWorkDone();
      expect(buffersSince(beforeReadbackFailure)).toEqual([]);
      await decide(firstCase, 'recovered-after-readback-failure');

      const preAborted = new AbortController();
      preAborted.abort('correction pre-abort');
      const beforePreAbort = liveBuffers();
      await expect(
        decider.systemOne(firstCase.state, { answer: firstCase.question }, undefined, {
          signal: preAborted.signal,
        }),
      ).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
      expect(buffersSince(beforePreAbort)).toEqual([]);

      const queuedAbort = new AbortController();
      const first = decider.systemOne(firstCase.state, { answer: firstCase.question });
      const second = decider.systemOne(
        corpus.cases[1]!.state,
        { answer: corpus.cases[1]!.question },
        undefined,
        { signal: queuedAbort.signal },
      );
      queuedAbort.abort('correction queued abort');
      verifyDecision(await first, firstCase, 'active-before-queued-abort');
      await expect(second).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });

      const activeAbort = new AbortController();
      const abortWork = decider.systemOne(
        firstCase.state,
        { answer: firstCase.question },
        undefined,
        { signal: activeAbort.signal },
      );
      await Promise.resolve();
      activeAbort.abort('correction active abort');
      await expect(abortWork).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
      await decide(firstCase, 'recovered-after-active-abort');

      const disposeCase = corpus.cases[0]!;
      const active = decider.systemOne(disposeCase.state, { answer: disposeCase.question });
      await Promise.resolve();
      const queued = decider.systemOne(corpus.cases[1]!.state, {
        answer: corpus.cases[1]!.question,
      });
      const disposeStarted = performance.now();
      const disposal = decider.dispose();
      expect(decider.dispose()).toBe(disposal);
      verifyDecision(await active, disposeCase, 'active-during-healthy-disposal');
      await expect(queued).rejects.toMatchObject({ code: 'RESOURCE_DISPOSED' });
      await expect(
        decider.systemOne(firstCase.state, { answer: firstCase.question }),
      ).rejects.toMatchObject({ code: 'RESOURCE_DISPOSED' });
      await disposal;
      const healthyDisposeMs = performance.now() - disposeStarted;
      const afterHealthyDispose = [...liveBuffers()].filter(
        (buffer) => !beforeHealthyCreate.has(buffer),
      );
      const healthyUniforms = afterHealthyDispose.filter(
        (buffer) => (buffer.usage & GPUBufferUsage.UNIFORM) !== 0,
      );
      const healthyStaging = afterHealthyDispose.filter(
        (buffer) => (buffer.usage & GPUBufferUsage.MAP_READ) !== 0,
      );
      const healthyOwned = afterHealthyDispose.filter(
        (buffer) =>
          (buffer.usage & GPUBufferUsage.UNIFORM) === 0 &&
          (buffer.usage & GPUBufferUsage.MAP_READ) === 0,
      );
      expect(healthyOwned).toHaveLength(0);

      const sharedAfterDispose = add(sharedInput, 3);
      expectClose(await toArray(sharedAfterDispose, sharedExecutor), [5, 7, 9, 11, 13, 15, 17, 19]);
      sharedExecutor.releaseValues([sharedAfterDispose]);

      initRunntime(replacementRoot);
      const currentReplacementExecutor = replacementExecutor;
      if (currentReplacementExecutor === undefined) {
        throw new Error('replacement executor was not initialized during loading');
      }
      freshInput = tensor(replacementRoot, Float32Array.of(4, 6, 8, 10, 12, 14, 16, 18), [1, 8]);
      const freshSharedOutput = add(freshInput, 2);
      expectClose(
        await toArray(freshSharedOutput, currentReplacementExecutor),
        [6, 8, 10, 12, 14, 16, 18, 20],
      );
      currentReplacementExecutor.releaseValues([freshSharedOutput]);
      const beforeFreshCreate = liveBuffers();
      const freshStaging = [...beforeFreshCreate].filter(
        (buffer) => (buffer.usage & GPUBufferUsage.MAP_READ) !== 0,
      );
      const freshUniforms = [...beforeFreshCreate].filter(
        (buffer) => (buffer.usage & GPUBufferUsage.UNIFORM) !== 0,
      );
      const freshLoadStarted = performance.now();
      freshDecider = await createD1Decider(stateDict(), tokenizer, { temperatures: temps });
      const freshLoadMs = performance.now() - freshLoadStarted;
      const freshCase = corpus.cases[0]!;
      const freshDecisionStarted = performance.now();
      const freshResult = await freshDecider.systemOne(freshCase.state, {
        answer: freshCase.question,
      });
      const freshDecisionMs = performance.now() - freshDecisionStarted;
      verifyDecision(freshResult, freshCase, 'fresh-device-reference');
      const freshDisposeStarted = performance.now();
      await freshDecider.dispose();
      await replacementRoot.device.queue.onSubmittedWorkDone();
      const freshDisposeMs = performance.now() - freshDisposeStarted;
      const afterFreshDispose = [...liveBuffers()].filter(
        (buffer) => !beforeFreshCreate.has(buffer),
      );
      const freshUniformsAfterDispose = afterFreshDispose.filter(
        (buffer) => (buffer.usage & GPUBufferUsage.UNIFORM) !== 0,
      );
      const freshStagingAfterDispose = afterFreshDispose.filter(
        (buffer) => (buffer.usage & GPUBufferUsage.MAP_READ) !== 0,
      );
      const freshOwned = afterFreshDispose.filter(
        (buffer) =>
          (buffer.usage & GPUBufferUsage.UNIFORM) === 0 &&
          (buffer.usage & GPUBufferUsage.MAP_READ) === 0,
      );
      expect(freshOwned).toHaveLength(0);

      const report = {
        mode: MODE,
        correctionBase: '735fbd94195a8c2a7c40a12aec6b86fa34ebd48c',
        model: MODEL,
        revision: REVISION,
        elapsedMs: +(performance.now() - started).toFixed(2),
        loadMs: +loadMs.toFixed(2),
        loadedWeightBytes,
        progressDoneBytes,
        progressTotalBytes,
        decisions,
        peakJsHeapBytes,
        peakAgentBytes,
        peakGpuBufferBytes,
        healthyAllocationBaseline: {
          gpuBufferCount: beforeHealthyCreate.size,
          gpuBufferBytes: baselineGpuBufferBytes,
          sharedStagingBufferCount: baselineStaging.length,
          sharedStagingBufferBytes: baselineStaging.reduce((sum, buffer) => sum + buffer.size, 0),
          sharedUniformBufferCount: baselineUniforms.length,
          sharedUniformBufferBytes: baselineUniforms.reduce((sum, buffer) => sum + buffer.size, 0),
          newSharedStagingBufferCount: healthyStaging.length,
          newSharedStagingBufferBytes: healthyStaging.reduce((sum, buffer) => sum + buffer.size, 0),
          newSharedUniformBufferCount: healthyUniforms.length,
          newSharedUniformBufferBytes: healthyUniforms.reduce(
            (sum, buffer) => sum + buffer.size,
            0,
          ),
          unownedBuffersAfterDispose: healthyOwned.length,
          disposeMs: +healthyDisposeMs.toFixed(2),
        },
        freshDevice: {
          loadMs: +freshLoadMs.toFixed(2),
          decisionMs: +freshDecisionMs.toFixed(2),
          inputTokens: freshResult.inputTokens,
          answer: stableAnswer(freshResult.answers.answer!, freshCase.question),
          sharedStagingBufferCount: freshStaging.length,
          sharedStagingBufferBytes: freshStaging.reduce((sum, buffer) => sum + buffer.size, 0),
          sharedUniformBufferCount: freshUniforms.length,
          sharedUniformBufferBytes: freshUniforms.reduce((sum, buffer) => sum + buffer.size, 0),
          newSharedStagingBufferCount: freshStagingAfterDispose.length,
          newSharedStagingBufferBytes: freshStagingAfterDispose.reduce(
            (sum, buffer) => sum + buffer.size,
            0,
          ),
          newSharedUniformBufferCount: freshUniformsAfterDispose.length,
          newSharedUniformBufferBytes: freshUniformsAfterDispose.reduce(
            (sum, buffer) => sum + buffer.size,
            0,
          ),
          unownedBuffersAfterDispose: freshOwned.length,
          disposeMs: +freshDisposeMs.toFixed(2),
        },
        measurements: {
          hostPeak:
            peakJsHeapBytes === null && peakAgentBytes === null
              ? 'unknown: browser exposes no host peak'
              : 'sampled browser memory; not a process-wide peak',
          gpuPeak: 'sum of live GPUBuffer.size snapshots; driver-private allocations are unknown',
          disposal:
            'no new D1-owned buffers after healthy disposal; per-root uniform buffers are retained by dispatchCache and MAP_READ staging is shared by readback; both are reported separately; neither device was destroyed',
        },
      };
      console.info('D1_SAFETY_CORRECTION', JSON.stringify(report));
      await ctx.annotate('D1_SAFETY_CORRECTION', {
        contentType: 'application/json',
        body: JSON.stringify(report, null, 2),
      });
      expect(decisions).toHaveLength(corpus.cases.length + 9);
      expect(sharedInput.state).toBe('materialized');
    } finally {
      await decider?.dispose();
      await freshDecider?.dispose();
      sharedExecutor.dispose();
      replacementExecutor?.dispose();
      sharedInput.buffer.destroy();
      freshInput?.buffer.destroy();
    }
  },
  900_000,
);
