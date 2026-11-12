/** Streaming text-generation session and its cache/disposal lifecycle. */

import { RunntimeError } from '../../core/index.ts';
import { LFM3 } from './model.ts';
import type { Lfm3Message, Lfm3Tokenizer } from './tokenizer.ts';

/** Generation options for a text chat request. */
export interface Lfm3ChatOptions {
  readonly maxNewTokens?: number;
  readonly signal?: AbortSignal;
}

/** Final text and token counts for one generation. */
export interface Lfm3GenerationResult {
  readonly text: string;
  readonly inputTokens: number;
  readonly completionTokens: number;
  readonly finishReason: 'stop' | 'length';
}

/** Stream of stable text deltas and its final generation result. */
export interface Lfm3Generation extends AsyncIterable<string> {
  readonly result: Promise<Lfm3GenerationResult>;
  cancel(): void;
}

/** Stateful, serialized text generation over the original LFM3 base model. */
export interface Lfm3ChatSession {
  generate(messages: readonly Lfm3Message[], options?: Lfm3ChatOptions): Lfm3Generation;
  /** Cancels in-flight work and drops its prompt-specific KV/convolution cache. */
  reset(): Promise<void>;
  /** Stops accepting calls, cancels queued/in-flight generation and releases weights. */
  dispose(): Promise<void>;
}

export interface Lfm3Step<Cache = unknown> {
  readonly logits: Float32Array;
  readonly cache: Cache;
}

export interface Lfm3SessionRuntime<Cache = unknown> {
  forward(
    ids: readonly number[],
    cache: Cache | undefined,
    signal: AbortSignal,
  ): Promise<Lfm3Step<Cache>>;
  release(cache: Cache): void | Promise<void>;
  trim(): void;
  dispose(): void | Promise<void>;
}

class AsyncQueue<T> implements AsyncIterableIterator<T> {
  private readonly values: T[] = [];
  private readonly waiters: {
    resolve: (result: IteratorResult<T>) => void;
    reject: (error: unknown) => void;
  }[] = [];
  private ended = false;
  private failure: unknown;

  [Symbol.asyncIterator](): AsyncIterableIterator<T> {
    return this;
  }

  next(): Promise<IteratorResult<T>> {
    const value = this.values.shift();
    if (value !== undefined) return Promise.resolve({ done: false, value });
    if (this.failure !== undefined) return Promise.reject(this.failure);
    if (this.ended) return Promise.resolve({ done: true, value: undefined });
    return new Promise((resolve, reject) => this.waiters.push({ resolve, reject }));
  }

  push(value: T): void {
    if (this.ended || this.failure !== undefined) return;
    const waiter = this.waiters.shift();
    if (waiter) waiter.resolve({ done: false, value });
    else this.values.push(value);
  }

  close(): void {
    if (this.ended || this.failure !== undefined) return;
    this.ended = true;
    for (const waiter of this.waiters.splice(0)) waiter.resolve({ done: true, value: undefined });
  }

  fail(error: unknown): void {
    if (this.ended || this.failure !== undefined) return;
    this.failure = error;
    this.values.length = 0;
    for (const waiter of this.waiters.splice(0)) waiter.reject(error);
  }
}

function aborted(signal: AbortSignal): void {
  if (signal.aborted) {
    throw new RunntimeError('EXECUTION_FAILED', 'LFM3 generation aborted', {
      cause: signal.reason,
    });
  }
}

function greedy(logits: Float32Array): number {
  if (logits.length !== LFM3.vocab) {
    throw new RunntimeError(
      'EXECUTION_FAILED',
      `LFM3 returned ${logits.length} logits; expected ${LFM3.vocab}`,
    );
  }
  let best = -Infinity;
  let token = 0;
  for (let i = 0; i < logits.length; i++) {
    const value = logits[i]!;
    if (!Number.isFinite(value)) {
      throw new RunntimeError('EXECUTION_FAILED', `LFM3 produced a non-finite logit at ${i}`);
    }
    if (value > best) {
      best = value;
      token = i;
    }
  }
  return token;
}

function validateMaxNewTokens(value: number | undefined): number {
  const maxNewTokens = value ?? 128;
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 0) {
    throw new RunntimeError('INVALID_ARGUMENT', 'maxNewTokens must be a non-negative integer');
  }
  return maxNewTokens;
}

/** Creates a serialized chat session over the supplied model runtime. */
export function createLfm3Session<Cache>(
  tokenizer: Lfm3Tokenizer,
  runtime: Lfm3SessionRuntime<Cache>,
): Lfm3ChatSession {
  let disposed = false;
  let queue: Promise<void> = Promise.resolve();
  let disposePromise: Promise<void> | undefined;
  const controllers = new Set<AbortController>();

  const runGeneration = async (
    prompt: number[],
    maxNewTokens: number,
    signal: AbortSignal,
    emit: (text: string) => void,
  ): Promise<Lfm3GenerationResult> => {
    const availableTokens = LFM3.maxContext - prompt.length;
    const limit = Math.min(maxNewTokens, availableTokens);
    let cache: Cache | undefined;
    let logits: Float32Array | undefined;
    const generated: number[] = [];
    let emitted = '';
    let finishReason: Lfm3GenerationResult['finishReason'] = 'length';
    try {
      aborted(signal);
      if (limit > 0) {
        let step = await runtime.forward(prompt, undefined, signal);
        cache = step.cache;
        logits = step.logits;
        for (let index = 0; index < limit; index++) {
          aborted(signal);
          const token = greedy(logits);
          generated.push(token);
          if (tokenizer.stopIds.has(token)) {
            finishReason = 'stop';
            break;
          }
          const text = tokenizer.decode(generated);
          const stableText = Array.from(text).slice(0, -1).join('');
          const delta = stableText.slice(emitted.length);
          emitted = stableText;
          if (delta) emit(delta);
          if (index + 1 === limit) break;
          aborted(signal);
          const previous = cache;
          step = await runtime.forward([token], previous, signal);
          cache = step.cache;
          logits = step.logits;
          if (previous !== undefined) await runtime.release(previous);
        }
      }
      const text = tokenizer.decode(generated);
      const finalDelta = text.slice(emitted.length);
      if (finalDelta) emit(finalDelta);
      return {
        text,
        inputTokens: prompt.length,
        completionTokens: generated.length,
        finishReason,
      };
    } finally {
      if (cache !== undefined) await runtime.release(cache);
      runtime.trim();
    }
  };

  const generate: Lfm3ChatSession['generate'] = (messages, options = {}) => {
    if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'LFM3 chat session is disposed');
    const prompt = tokenizer.encodeChat(messages);
    if (prompt.length > LFM3.maxContext) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `LFM3 prompt has ${prompt.length} tokens; context limit is ${LFM3.maxContext}`,
      );
    }
    const maxNewTokens = validateMaxNewTokens(options.maxNewTokens);
    const controller = new AbortController();
    controllers.add(controller);
    const forwardSignal = controller.signal;
    const abort = () => controller.abort(options.signal?.reason);
    if (options.signal?.aborted) abort();
    else options.signal?.addEventListener('abort', abort, { once: true });

    const updates = new AsyncQueue<string>();
    const job = queue.then(async () => {
      if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'LFM3 chat session is disposed');
      return runGeneration(prompt, maxNewTokens, forwardSignal, (text) => updates.push(text));
    });
    queue = job.then(
      () => undefined,
      () => undefined,
    );
    const result = job
      .then(
        (value) => {
          updates.close();
          return value;
        },
        (error: unknown) => {
          updates.fail(error);
          throw error;
        },
      )
      .finally(() => {
        controllers.delete(controller);
        options.signal?.removeEventListener('abort', abort);
      });
    result.catch(() => undefined);
    return {
      result,
      cancel: abort,
      [Symbol.asyncIterator]: () => updates,
    };
  };

  const reset = async (): Promise<void> => {
    if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'LFM3 chat session is disposed');
    for (const controller of controllers) controller.abort('session reset');
    await queue;
  };

  const dispose = (): Promise<void> => {
    if (disposePromise) return disposePromise;
    disposed = true;
    for (const controller of controllers) controller.abort('session disposed');
    disposePromise = queue.then(() => runtime.dispose());
    return disposePromise;
  };

  return { generate, reset, dispose };
}
