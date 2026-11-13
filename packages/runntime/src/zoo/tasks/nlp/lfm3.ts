/** Public text-only LFM3-Omni chat session on the initialized WebGPU device. */

import {
  defaultRoot,
  evalValues,
  fromGguf,
  gpuExecutor,
  httpRangeSource,
  inGpuErrorScopes,
  isRunntimeError,
  RunntimeError,
  toArray,
  warmUp,
  type RangeSource,
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
import { LFM3_Q4_K_M_SOURCE } from '../../lfm3/gguf.ts';
import { cacheValues, Lfm3CausalLM, type Lfm3Cache } from '../../lfm3/model.ts';
import {
  createLfm3Session,
  type Lfm3ChatSession,
  type Lfm3SessionRuntime,
} from '../../lfm3/session.ts';
import { createLfm3Tokenizer, LFM3_REVISION, LFM3_TOKENIZER_URL } from '../../lfm3/tokenizer.ts';

const MODEL_URL = `https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/${LFM3_REVISION}/model.safetensors`;
const GGUF_CACHE_PREFIX = 'lfm3-omni-q4km-v1';

export interface Lfm3ChatModel {
  /** Original BF16 safetensors or the pinned Q4_K_M GGUF, or a caller-owned RangeSource. */
  readonly modelPath?: ModelPath;
  /** Pinned Liquid tokenizer.json. */
  readonly tokenizerPath?: string;
  /** Selects the pinned GGUF reader instead of the original safetensors loader. */
  readonly format?: 'gguf';
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
  const modelPath =
    config.modelPath ?? (config.format === 'gguf' ? LFM3_Q4_K_M_SOURCE.url : MODEL_URL);
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
      const uploads: Value[] = [];
      try {
        if (signal.aborted) {
          throw new RunntimeError('EXECUTION_FAILED', 'LFM3 generation aborted', {
            cause: signal.reason,
          });
        }
        const result = await inGpuErrorScopes(root.device, 'LFM3 generation', async () => {
          const current = model.forward(ids, root, previous, uploads);
          step = current;
          evalValues([current.logits, ...cacheValues(current.cache)], executor);
          const logits = await toArray(current.logits, executor);
          await releaseAfterWork();
          if (signal.aborted) {
            throw new RunntimeError('EXECUTION_FAILED', 'LFM3 generation aborted', {
              cause: signal.reason,
            });
          }
          if (
            logits.length === 0 ||
            !logits.every(Number.isFinite) ||
            logits.every((value) => value === 0)
          ) {
            throw new RunntimeError('EXECUTION_FAILED', 'LFM3 generation produced unusable logits');
          }
          return { logits, step: current };
        });
        executor.releaseValues([result.step.logits]);
        return { logits: result.logits, cache: result.step.cache };
      } catch (error) {
        await releaseAfterWork();
        if (step) executor.releaseValues([step.logits, ...cacheValues(step.cache)]);
        throw rethrowRunError(error);
      } finally {
        destroyUploads(uploads);
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
    const weights =
      config.format === 'gguf'
        ? await openGgufWeights(modelPath, opts)
        : await openWeights(modelPath, {
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
      try {
        await warmUp(
          root.device,
          async () => {
            throwIfAborted(opts.signal);
            const prompt = tokenizer!.encodeChat([{ role: 'user', content: 'Hello.' }]);
            const result = await runtime.forward(
              prompt,
              undefined,
              opts.signal ?? new AbortController().signal,
            );
            await runtime.release(result.cache);
            throwIfAborted(opts.signal);
            return result.logits;
          },
          'LFM3 model warm-up',
        );
      } catch (error) {
        throwIfAborted(opts.signal);
        throw error;
      }
      throwIfAborted(opts.signal);
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

async function openGgufWeights(modelPath: ModelPath, opts: LoadOptions) {
  const source: RangeSource =
    typeof modelPath === 'string'
      ? httpRangeSource(modelPath, { onBytes: opts.onBytes })
      : modelPath;
  return fromGguf(source, {
    fileBytes: LFM3_Q4_K_M_SOURCE.bytes,
    sha256: LFM3_Q4_K_M_SOURCE.sha256,
    cache: opts.cache,
    cacheId: opts.cacheId ?? ggufCacheKey(modelPath),
    signal: opts.signal,
  });
}

function destroyUploads(values: readonly Value[]): void {
  for (const value of values) {
    if (value.state === 'materialized') value.buffer.destroy();
  }
}

const cacheKey = (path: ModelPath): string | undefined =>
  typeof path === 'string' ? `lfm3-base-v1/${path.replace(/^https?:\/\//, '')}` : undefined;

const ggufCacheKey = (path: ModelPath): string | undefined =>
  typeof path === 'string' ? `${GGUF_CACHE_PREFIX}/${path.replace(/^https?:\/\//, '')}` : undefined;
