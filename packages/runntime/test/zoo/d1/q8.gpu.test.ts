/** Opt-in public-loader qualification for a locally converted D1 text Q8 file. */

import { expect, it } from 'vitest';
import {
  createD1Decider,
  models,
  type Answer,
  type Question,
  type RangeSource,
} from '../../../src/zoo/index.ts';
import { buffersSince, gpu, liveBuffers } from '../../gpu.ts';
import vectors from '../../../playground/d1/reference-vectors.json';

declare global {
  interface ImportMetaEnv {
    readonly VITE_D1_Q8_QUALIFICATION_MODE?: string;
    readonly VITE_D1_Q8_MODEL_URL?: string;
    readonly VITE_D1_Q8_TOKENIZER_URL?: string;
    readonly VITE_D1_Q8_ARTIFACT_SHA256?: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const MODE = import.meta.env.VITE_D1_Q8_QUALIFICATION_MODE;
const MODEL_URL = import.meta.env.VITE_D1_Q8_MODEL_URL;
const TOKENIZER_URL = import.meta.env.VITE_D1_Q8_TOKENIZER_URL;
const ARTIFACT_SHA256 = import.meta.env.VITE_D1_Q8_ARTIFACT_SHA256;
const PROBABILITY_TOLERANCE = 0.04;
const CLEAR_MARGIN = 0.1;

interface ReferenceCase {
  id: string;
  state: string;
  question: Question;
  reference: { probabilities: number[]; answer: string; inputTokens: number };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isReferenceCase(value: unknown): value is ReferenceCase {
  if (!isRecord(value) || !isRecord(value.reference) || !isRecord(value.question)) return false;
  const question = value.question;
  const reference = value.reference;
  return (
    typeof value.id === 'string' &&
    typeof value.state === 'string' &&
    typeof question.type === 'string' &&
    typeof question.instructions === 'string' &&
    Array.isArray(reference.probabilities) &&
    reference.probabilities.every((probability) => typeof probability === 'number') &&
    typeof reference.answer === 'string' &&
    typeof reference.inputTokens === 'number'
  );
}

function casesFrom(value: unknown): ReferenceCase[] {
  if (!isRecord(value) || !Array.isArray(value.cases) || !value.cases.every(isReferenceCase)) {
    throw new Error('pinned D1 reference cases are malformed');
  }
  return value.cases;
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

function stableAnswer(answer: Answer): string {
  if (answer.type === 'noul') return answer.noul >= 0.5 ? 'yes' : 'no';
  if (answer.type === 'choice') return answer.choice;
  let best = 0;
  const scores = Object.values(answer.probabilities);
  for (let i = 1; i < scores.length; i++) if (scores[i]! > scores[best]!) best = i;
  return String(best);
}

function margin(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => b - a);
  return sorted[0]! - sorted[1]!;
}

function rangeSource(url: string): RangeSource & { bytesReceived(): number; rangeMode(): string } {
  let whole: Promise<Uint8Array> | undefined;
  let received = 0;
  let mode = 'range';
  return {
    async read(begin, end) {
      if (whole) return (await whole).slice(begin, end);
      const response = await fetch(url, { headers: { Range: `bytes=${begin}-${end - 1}` } });
      if (!response.ok) throw new Error(`D1 Q8 artifact request returned ${response.status}`);
      if (response.status === 200) {
        mode = 'whole-file fallback';
        whole = response.arrayBuffer().then((buffer) => {
          const bytes = new Uint8Array(buffer);
          received += bytes.byteLength;
          return bytes;
        });
        return (await whole).slice(begin, end);
      }
      if (response.status !== 206) throw new Error(`D1 Q8 range returned ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.byteLength !== end - begin) {
        throw new Error(`D1 Q8 range returned ${bytes.byteLength} bytes, expected ${end - begin}`);
      }
      received += bytes.byteLength;
      return bytes;
    },
    bytesReceived: () => received,
    rangeMode: () => mode,
  };
}

function gpuBufferBytes(): number {
  return [...liveBuffers()].reduce((sum, buffer) => sum + buffer.size, 0);
}

function ownedBuffersSince(before: ReadonlySet<GPUBuffer>): string[] {
  return buffersSince(before).filter((_, index) => {
    const buffer = [...liveBuffers()].filter((current) => !before.has(current))[index]!;
    return (buffer.usage & (GPUBufferUsage.MAP_READ | GPUBufferUsage.UNIFORM)) === 0;
  });
}

function sampledHostBytes(): number | null {
  const memory = Reflect.get(performance, 'memory');
  const bytes =
    typeof memory === 'object' && memory !== null ? Reflect.get(memory, 'usedJSHeapSize') : null;
  return typeof bytes === 'number' ? bytes : null;
}

it.skipIf(MODE !== 'q8' || MODEL_URL === undefined)(
  'loads the public D1 Q8 zoo entry from a caller RangeSource and meets frozen Liquid parity',
  async (ctx) => {
    const cases = casesFrom(vectors);
    const { root } = await gpu();
    const source = rangeSource(MODEL_URL!);
    const config = {
      ...models.d1.D1_OMNI_600M.Q8,
      modelPath: source,
      ...(TOKENIZER_URL === undefined ? {} : { tokenizerPath: TOKENIZER_URL }),
    };

    const malformed = {
      ...models.d1.D1_OMNI_600M.Q8,
      modelPath: { read: async () => new Uint8Array(8) },
    };
    await expect(createD1Decider(malformed, { warmUp: false })).rejects.toMatchObject({
      code: 'LOAD_FAILED',
    });

    const beforeAbort = liveBuffers();
    const abort = new AbortController();
    await expect(
      createD1Decider(config, {
        signal: abort.signal,
        warmUp: false,
        onProgress: () => abort.abort('D1 Q8 upload abort test'),
      }),
    ).rejects.toMatchObject({ code: 'LOAD_ABORTED' });
    await root.device.queue.onSubmittedWorkDone();
    expect(ownedBuffersSince(beforeAbort)).toEqual([]);

    const beforeLoad = liveBuffers();
    let loadedWeightBytes = 0;
    let progressDoneBytes = 0;
    let progressTotalBytes = 0;
    let peakGpuBufferBytes = gpuBufferBytes();
    let peakHostBytes = sampledHostBytes();
    const progressNames = new Set<string>();
    const runner = await createD1Decider(config, {
      onProgress: (name, done, total) => {
        progressNames.add(name);
        loadedWeightBytes = total;
        progressDoneBytes = done;
        progressTotalBytes = total;
        peakGpuBufferBytes = Math.max(peakGpuBufferBytes, gpuBufferBytes());
        const hostBytes = sampledHostBytes();
        if (hostBytes !== null) peakHostBytes = Math.max(peakHostBytes ?? 0, hostBytes);
      },
    });

    const decisions: Record<string, unknown>[] = [];
    try {
      await expect(
        runner.systemOne('unit '.repeat(20_000), { answer: cases[0]!.question }),
      ).rejects.toMatchObject({ code: 'INVALID_ARGUMENT' });

      for (const item of cases) {
        const result = await runner.systemOne(item.state, { answer: item.question });
        const answer = result.answers.answer!;
        const got = probabilities(answer, item.question);
        const maxAbsError = Math.max(
          ...got.map((probability, index) =>
            Math.abs(probability - item.reference.probabilities[index]!),
          ),
        );
        const referenceMargin = margin(item.reference.probabilities);
        expect(result.inputTokens).toBe(item.reference.inputTokens);
        expect(maxAbsError).toBeLessThanOrEqual(PROBABILITY_TOLERANCE);
        if (referenceMargin >= CLEAR_MARGIN) {
          expect(stableAnswer(answer)).toBe(item.reference.answer);
        }
        const decision: Record<string, unknown> = {
          id: item.id,
          inputTokens: result.inputTokens,
          answer: stableAnswer(answer),
          referenceAnswer: item.reference.answer,
          clearMargin: referenceMargin,
          probabilities: got,
          maxAbsError,
        };
        if (item.question.type === 'score' && answer.type === 'score') {
          const referenceScore = item.reference.probabilities.reduce(
            (sum, probability, index) => sum + probability * index,
            0,
          );
          const normalizedRubricError =
            Math.abs(answer.score - referenceScore) /
            Math.max(1, item.question.criteria.length - 1);
          expect(normalizedRubricError).toBeLessThanOrEqual(0.05);
          decision.normalizedRubricError = normalizedRubricError;
        }
        decisions.push(decision);
        peakGpuBufferBytes = Math.max(peakGpuBufferBytes, gpuBufferBytes());
        const hostBytes = sampledHostBytes();
        if (hostBytes !== null) peakHostBytes = Math.max(peakHostBytes ?? 0, hostBytes);
      }

      const report = {
        mode: MODE,
        model: 'LiquidAI/d1-omni-600M',
        modelRevision: '02b55d7076f15129e59ab3f94783f32c4b088674',
        artifactSha256: ARTIFACT_SHA256 ?? null,
        tokenizerRevision: '02b55d7076f15129e59ab3f94783f32c4b088674',
        quantization: 'runntime.quant v1, symmetric int8, groupSize 32',
        artifactDelivery: source.rangeMode(),
        bytesReceived: source.bytesReceived(),
        loadedWeightBytes,
        progressDoneBytes,
        progressTotalBytes,
        progressTensorCount: progressNames.size,
        peakHostBytes,
        peakGpuBufferBytes,
        decisions,
        probabilityTolerance: PROBABILITY_TOLERANCE,
        normalizedRubricTolerance: 0.05,
        gpuBuffersBeforeLoad: beforeLoad.size,
      };
      console.info('D1_Q8_QUALIFICATION', JSON.stringify(report));
      await ctx.annotate('D1_Q8_QUALIFICATION', {
        contentType: 'application/json',
        body: JSON.stringify(report, null, 2),
      });
      expect(progressDoneBytes).toBe(progressTotalBytes);
      expect(loadedWeightBytes).toBeGreaterThan(0);
      expect(decisions).toHaveLength(cases.length);
    } finally {
      await runner.dispose();
    }

    await root.device.queue.onSubmittedWorkDone();
    const disposedOwnedBuffers = ownedBuffersSince(beforeLoad);
    expect(disposedOwnedBuffers).toEqual([]);
    await expect(
      runner.systemOne('', { warmup: { type: 'noul', instructions: 'disposed' } }),
    ).rejects.toMatchObject({
      code: 'RESOURCE_DISPOSED',
    });
  },
  900_000,
);
