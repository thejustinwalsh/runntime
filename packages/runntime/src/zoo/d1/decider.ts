/** d1-omni's System One call on the initRunntime() device: named questions
 *  over a text state and optional images, one trunk pass per question, no
 *  generated tokens. */

import {
  cat,
  defaultRoot,
  evalValues,
  gpuExecutor,
  RunntimeError,
  slice,
  tensor,
  toArray,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { asLoadError, rethrowRunError } from '../errors.ts';
import { throwIfAborted } from '../load.ts';
import { preprocessImage, resizeAA, type RgbImage } from './image.ts';
import { D1, D1Model, D1TextModel } from './model.ts';
import {
  encodeQuestion,
  optionCount,
  QTYPES,
  softmaxT,
  temperatureKey,
  toAnswer,
  YES_NO,
  type Answer,
  type D1Tokenizer,
  type Question,
} from './prompt.ts';

/** Total positions the trunk was trained on: media plus text. */
const MAX_LENGTH = 16384;
/** Text positions after an image prefix, as trained. */
const IMAGE_TEXT_LENGTH = 896;

export interface D1Decider {
  /** Answers each named question; `images` makes the call an image call. */
  systemOne(
    state: string,
    questions: Record<string, Question>,
    images?: readonly RgbImage[],
    options?: { signal?: AbortSignal },
  ): Promise<{ answers: Record<string, Answer>; inputTokens: number }>;
  /** Stops accepting work and releases owned resources after queued work settles. */
  dispose(): Promise<void>;
}

interface D1DeciderOptions {
  temperatures?: Record<string, number>;
  onProgress?: (name: string, done: number, total: number) => void;
  signal?: AbortSignal;
}

/** Pair-duplicated RoPE tables [T, headDim] for positions 0..T−1. */
function ropeTables(t: number): { cos: Float32Array; sin: Float32Array } {
  const hd = D1.headDim;
  const cos = new Float32Array(t * hd);
  const sin = new Float32Array(t * hd);
  for (let i = 0; i < hd / 2; i++) {
    const invFreq = Math.fround(1 / Math.pow(D1.ropeTheta, (2 * i) / hd));
    for (let p = 0; p < t; p++) {
      const a = Math.fround(p * invFreq);
      cos[p * hd + 2 * i] = cos[p * hd + 2 * i + 1] = Math.cos(a);
      sin[p * hd + 2 * i] = sin[p * hd + 2 * i + 1] = Math.sin(a);
    }
  }
  return { cos, sin };
}

export async function createD1Decider(
  sd: LazyStateDict,
  tokenizer: D1Tokenizer,
  opts: D1DeciderOptions = {},
): Promise<D1Decider> {
  return createD1DeciderWithModel(sd, tokenizer, new D1Model(), opts);
}

/** Loads the text-only D1 model and ignores audio and vision checkpoint tensors. */
export async function createD1TextDecider(
  sd: LazyStateDict,
  tokenizer: D1Tokenizer,
  opts: D1DeciderOptions = {},
): Promise<D1Decider> {
  return createD1DeciderWithModel(sd, tokenizer, new D1TextModel(), opts);
}

async function createD1DeciderWithModel(
  sd: LazyStateDict,
  tokenizer: D1Tokenizer,
  model: D1TextModel,
  opts: D1DeciderOptions,
): Promise<D1Decider> {
  const root = defaultRoot();
  const ex = gpuExecutor(root);
  let zeroRow: Value | undefined;
  try {
    throwIfAborted(opts.signal);
    await model.loadStateDict(sd, {
      root,
      onProgress: (name, done, total) => {
        opts.onProgress?.(name, done, total);
        throwIfAborted(opts.signal);
      },
    });
    throwIfAborted(opts.signal);
    zeroRow = tensor(root, new Float32Array(D1.hidden), [1, D1.hidden]);

    const temperatures = opts.temperatures ?? {};

    let disposed = false;
    let queue: Promise<void> = Promise.resolve();
    let disposePromise: Promise<void> | undefined;

    const releaseAfterWork = async () => {
      try {
        await root.device.queue.onSubmittedWorkDone();
      } catch {
        // Device loss rejects the queue wait; owned buffers still need teardown.
      }
    };

    /** One image's crops through the tower and projector, rows concatenated. */
    const encodeImage = (img: RgbImage, uploads: Value[]): Value => {
      if (!(model instanceof D1Model)) {
        throw new RunntimeError('INVALID_ARGUMENT', 'D1 text decider does not accept images');
      }
      const posTable = model.vision.position_embedding.data;
      if (!(posTable instanceof Float32Array)) throw new Error('d1: position table must be f32');
      const parts = preprocessImage(img).map(({ ph, pw, patches }) => {
        const pos = resizeAA(posTable, D1.posGrid, D1.posGrid, D1.visionHidden, ph, pw);
        const p = tensor(root, patches, [ph * pw, 3 * 16 * 16]);
        uploads.push(p);
        const q = tensor(root, pos, [ph * pw, D1.visionHidden]);
        uploads.push(q);
        return model.vision.forward(p, q, ph, pw);
      });
      return parts.length === 1 ? parts[0]! : cat(parts, 0);
    };

    const runQuestion = async (
      prefix: Value | undefined,
      ids: number[],
      q: Question,
      markers: number[],
      signal?: AbortSignal,
    ): Promise<number[]> => {
      const uploads: Value[] = [];
      let output: Value | undefined;
      try {
        throwIfDecisionAborted(signal);
        const p = prefix ? prefix.shape.dims![0]! : 0;
        const t = p + ids.length;
        const emb = track(model.encoder.embed_tokens.forward(ids, root), uploads);
        const { cos: c, sin: s } = ropeTables(t);
        const cos = track(tensor(root, c, [t, D1.headDim]), uploads);
        const sin = track(tensor(root, s, [t, D1.headDim]), uploads);
        let segments: Value | undefined;
        let keepRight: Value | undefined;
        if (p > 0) {
          const seg = new Float32Array(2 * t);
          const keep = new Float32Array(t).fill(1);
          for (let i = 0; i < t; i++) seg[2 * i + 1] = i < p ? p : t;
          keep[p - 1] = 0;
          segments = track(tensor(root, seg, [t, 2]), uploads);
          keepRight = track(tensor(root, keep, [t, 1]), uploads);
        }
        const typeRow = track(model.head.type_emb.forward([QTYPES[q.type]], root), uploads);
        const h = model.encoder.forward(prefix ? cat([prefix, emb], 0) : emb, {
          cos,
          sin,
          segments,
          keepRight,
          zeroRow: zeroRow!,
        });
        const text = p > 0 ? slice(h, 0, p, t) : h;
        output = model.head.forward(text, typeRow, markers);
        const logits = await toArray(output, ex);
        throwIfDecisionAborted(signal);
        return Array.from(logits.subarray(0, optionCount(q)));
      } finally {
        await releaseAfterWork();
        if (output) ex.releaseValues([output]);
        for (const value of uploads) value.buffer.destroy();
        ex.trim();
      }
    };

    const systemOne: D1Decider['systemOne'] = (state, questions, images, options) => {
      if (disposed) {
        return Promise.reject(new RunntimeError('RESOURCE_DISPOSED', 'D1 decider is disposed'));
      }
      if (images && images.length > 0 && !(model instanceof D1Model)) {
        return Promise.reject(
          new RunntimeError('INVALID_ARGUMENT', 'D1 text decider does not accept images'),
        );
      }
      const job = queue.then(async () => {
        if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'D1 decider is disposed');
        const signal = options?.signal;
        let prefix: Value | undefined;
        const imageUploads: Value[] = [];
        try {
          throwIfDecisionAborted(signal);
          if (images && images.length > 0) {
            const parts = images.map((img) => encodeImage(img, imageUploads));
            prefix = parts.length === 1 ? parts[0]! : cat(parts, 0);
            evalValues([prefix], ex);
          }
          const p = prefix ? prefix.shape.dims![0]! : 0;
          const maxLen = Math.min(prefix ? IMAGE_TEXT_LENGTH : MAX_LENGTH, MAX_LENGTH - p);
          const answers: Record<string, Answer> = {};
          let inputTokens = 0;
          for (const [name, q] of Object.entries(questions)) {
            throwIfDecisionAborted(signal);
            const { ids, markers } = encodeQuestion(
              tokenizer,
              state,
              q,
              maxLen,
              prefix ? YES_NO : undefined,
            );
            inputTokens += p + ids.length;
            const logits = await runQuestion(prefix, ids, q, markers, signal);
            const temperature = prefix
              ? 1
              : (temperatures[temperatureKey(q)] ?? temperatures[q.type] ?? 1);
            const probs = softmaxT(logits, temperature);
            // The model reads a noul as [false, true]; answers want [yes, no].
            answers[name] = toAnswer(q, q.type === 'noul' ? probs.reverse() : probs);
          }
          return { answers, inputTokens };
        } finally {
          await releaseAfterWork();
          if (prefix) ex.releaseValues([prefix]);
          for (const value of imageUploads) value.buffer.destroy();
          ex.trim();
        }
      });
      const result = job.catch(rethrowRunError);
      queue = result.then(
        () => undefined,
        () => undefined,
      );
      return result;
    };

    const dispose = (): Promise<void> => {
      if (disposePromise) return disposePromise;
      disposed = true;
      disposePromise = queue.then(() => {
        try {
          model.dispose();
        } finally {
          zeroRow?.buffer.destroy();
          ex.dispose();
        }
      });
      return disposePromise;
    };

    return { systemOne, dispose };
  } catch (err) {
    try {
      model.dispose();
    } finally {
      zeroRow?.buffer.destroy();
      ex.dispose();
    }
    throw asLoadError(err, opts.signal);
  }
}

function track(value: Value, uploads: Value[]): Value {
  uploads.push(value);
  return value;
}

function throwIfDecisionAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new RunntimeError('EXECUTION_FAILED', 'D1 decision aborted', { cause: signal.reason });
  }
}
