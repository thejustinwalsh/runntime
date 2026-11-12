/** Public text-only LFM3-Omni chat session on the initialized WebGPU device. */

import {
  defaultRoot,
  evalValues,
  gpuExecutor,
  isRunntimeError,
  RunntimeError,
  toArray,
  warmUp,
  type Value,
} from '../../../core/index.ts';
import { asLoadError, rethrowRunError } from '../../errors.ts';
import {
  fetchJson,
  openWeights,
  throwIfAborted,
  type LoadOptions,
  type ModelPath,
} from '../../load.ts';
import { models } from '../../models.ts';
import { cacheValues, Lfm3CausalLM, type Lfm3Cache } from '../../lfm3/model.ts';
import {
  createLfm3Session,
  type Lfm3ChatSession,
  type Lfm3SessionRuntime,
} from '../../lfm3/session.ts';
import { createLfm3Tokenizer, LFM3_REVISION, LFM3_TOKENIZER_URL } from '../../lfm3/tokenizer.ts';

const MODEL_URL = `https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/${LFM3_REVISION}/model.safetensors`;

export interface Lfm3ChatModel {
  /** Original BF16 safetensors checkpoint, or caller-owned RangeSource. */
  readonly modelPath?: ModelPath;
  /** Pinned Liquid tokenizer.json. */
  readonly tokenizerPath?: string;
}

export type {
  Lfm3ChatOptions,
  Lfm3ChatSession,
  Lfm3Generation,
  Lfm3GenerationResult,
} from '../../lfm3/session.ts';
export type { Lfm3Message } from '../../lfm3/tokenizer.ts';

/** Loads the pinned original floating checkpoint and returns a streaming chat session. */
export async function createLfm3Chat(
  config: Lfm3ChatModel = models.lfm3.LFM3_OMNI_2B.DEFAULT,
  opts: LoadOptions = {},
): Promise<Lfm3ChatSession> {
  const modelPath = config.modelPath ?? MODEL_URL;
  const tokenizerPath = config.tokenizerPath ?? LFM3_TOKENIZER_URL;
  const root = defaultRoot();
  const executor = gpuExecutor(root);
  const model = new Lfm3CausalLM();
  const releaseAfterWork = async (): Promise<void> => {
    try {
      await root.device.queue.onSubmittedWorkDone();
    } catch {
      // Device loss rejects the wait; owned GPU buffers still need teardown.
    }
  };
  const runtime: Lfm3SessionRuntime<Lfm3Cache> = {
    async forward(ids: readonly number[], previous: Lfm3Cache | undefined, signal: AbortSignal) {
      let step: ReturnType<Lfm3CausalLM['forward']> | undefined;
      try {
        if (signal.aborted) {
          throw new RunntimeError('EXECUTION_FAILED', 'LFM3 generation aborted', {
            cause: signal.reason,
          });
        }
        step = model.forward(ids, root, previous);
        evalValues([step.logits, ...cacheValues(step.cache)], executor);
        const logits = await toArray(step.logits, executor);
        await releaseAfterWork();
        if (signal.aborted) {
          throw new RunntimeError('EXECUTION_FAILED', 'LFM3 generation aborted', {
            cause: signal.reason,
          });
        }
        executor.releaseValues([step.logits]);
        return { logits, cache: step.cache };
      } catch (error) {
        await releaseAfterWork();
        if (step) executor.releaseValues([step.logits, ...cacheValues(step.cache)]);
        throw rethrowRunError(error);
      } finally {
        if (step) destroyUploads(step.uploads);
      }
    },
    async release(cache: Lfm3Cache) {
      executor.releaseValues(cacheValues(cache));
    },
    trim() {
      executor.trim();
    },
    async dispose() {
      await releaseAfterWork();
      try {
        model.dispose();
      } finally {
        executor.dispose();
      }
    },
  };

  let tokenizer: ReturnType<typeof createLfm3Tokenizer> | undefined;
  try {
    throwIfAborted(opts.signal);
    const weights = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? cacheKey(modelPath),
    });
    const tokenizerJson = await fetchJson<unknown>(tokenizerPath, {
      cache: opts.cache,
      cacheId: cacheKey(tokenizerPath),
      signal: opts.signal,
    });
    tokenizer = createLfm3Tokenizer(tokenizerJson);
    throwIfAborted(opts.signal);

    await model.loadStateDict(weights, {
      root,
      onProgress: (name, done, total) => {
        opts.onProgress?.(name, done, total);
        throwIfAborted(opts.signal);
      },
    });
    throwIfAborted(opts.signal);

    if (opts.warmUp !== false) {
      await warmUp(
        root.device,
        async () => {
          const prompt = tokenizer!.encodeChat([{ role: 'user', content: 'Hello.' }]);
          const result = await runtime.forward(prompt, undefined, new AbortController().signal);
          await runtime.release(result.cache);
          return result.logits;
        },
        'LFM3 model warm-up',
      );
      runtime.trim();
    }

    return createLfm3Session(tokenizer, runtime);
  } catch (error) {
    try {
      await runtime.dispose();
    } catch {
      // Preserve the actual load, tokenizer, abort or warm-up failure.
    }
    if (isRunntimeError(error)) throw error;
    throw asLoadError(error, opts.signal);
  }
}

function destroyUploads(values: readonly Value[]): void {
  for (const value of values) {
    if (value.state === 'materialized') value.buffer.destroy();
  }
}

const cacheKey = (path: ModelPath): string | undefined =>
  typeof path === 'string' ? `lfm3-base-v1/${path.replace(/^https?:\/\//, '')}` : undefined;
