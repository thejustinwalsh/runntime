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
  tensor,
  toArray,
  warmUp,
  type RangeSource,
  type RunntimeExecutor,
  type Value,
} from '../../../core/index.ts';
import { readGgufEmbeddingRows, type GgufStateDict } from '../../../core/weights/gguf.ts';
import type { TgpuRoot } from 'typegpu';
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
import {
  cacheValues,
  LFM3,
  Lfm3CausalLM,
  type Lfm3Cache,
  type Lfm3ModelStep,
} from '../../lfm3/model.ts';
import {
  createLfm3Session,
  type Lfm3ChatSession,
  type Lfm3SessionRuntime,
} from '../../lfm3/session.ts';
import {
  createLfm3Tokenizer,
  LFM3_REVISION,
  LFM3_TOKENIZER_URL,
  type Lfm3Tokenizer,
} from '../../lfm3/tokenizer.ts';

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

/** Resources shared by public LFM3 language and audio runners. */
export interface Lfm3Backbone {
  readonly root: TgpuRoot;
  readonly executor: RunntimeExecutor;
  readonly model: Lfm3CausalLM;
  readonly tokenizer: Lfm3Tokenizer;
  readonly runtime: Lfm3SessionRuntime<Lfm3Cache>;
  forwardEmbeddings(
    embeddings: Value,
    previous: Lfm3Cache | undefined,
    signal: AbortSignal,
    uploads: Value[],
  ): Promise<{ logits: Float32Array; hidden: Float32Array; cache: Lfm3Cache }>;
  forwardTokens(
    ids: readonly number[],
    previous: Lfm3Cache | undefined,
    signal: AbortSignal,
  ): Promise<{ logits: Float32Array; hidden: Float32Array; cache: Lfm3Cache }>;
  embedTokens(ids: readonly number[], uploads: Value[], signal: AbortSignal): Promise<Value>;
}

/** Loads the pinned LFM3 language backbone for a composed task runner. */
export async function loadLfm3Backbone(
  config: Lfm3ChatModel = models.lfm3.LFM3_OMNI_2B.DEFAULT,
  opts: LoadOptions = {},
): Promise<Lfm3Backbone> {
  const modelPath =
    config.modelPath ?? (config.format === 'gguf' ? LFM3_Q4_K_M_SOURCE.url : MODEL_URL);
  const tokenizerPath = config.tokenizerPath ?? LFM3_TOKENIZER_URL;
  const root = defaultRoot();
  const executor = gpuExecutor(root);
  let gguf: GgufStateDict | undefined;
  const table = 'token_embd.weight';
  const model = new Lfm3CausalLM(
    config.format === 'gguf' ? (start, count) => gguf!.rows(table, start, count) : undefined,
  );
  const embedTokens = async (ids: readonly number[], uploads: Value[], signal: AbortSignal) => {
    const embeddings = gguf
      ? tensor(root, await readGgufEmbeddingRows(gguf, table, LFM3.hidden, ids, signal), {
          dims: [ids.length, LFM3.hidden],
          elems: ids.length * LFM3.hidden,
          dtype: 'f32',
        })
      : model.model.embed_tokens!.forward(ids, root);
    uploads.push(embeddings);
    return embeddings;
  };
  const releaseAfterWork = async (): Promise<void> => {
    try {
      await root.device.queue.onSubmittedWorkDone();
    } catch {
      // Device loss rejects the wait; owned GPU buffers still need teardown.
    }
  };
  const runStep = async (
    create: (uploads: Value[]) => Lfm3ModelStep | Promise<Lfm3ModelStep>,
    previous: Lfm3Cache | undefined,
    signal: AbortSignal,
    uploads: Value[],
    readHidden: boolean,
  ): Promise<{ logits: Float32Array; hidden?: Float32Array; cache: Lfm3Cache }> => {
    let step: Lfm3ModelStep | undefined;
    try {
      if (signal.aborted) {
        throw new RunntimeError('EXECUTION_FAILED', 'LFM3 generation aborted', {
          cause: signal.reason,
        });
      }
      const result = await inGpuErrorScopes(root.device, 'LFM3 generation', async () => {
        const current = await create(uploads);
        step = current;
        // Readbacks stay unfused: the logits (and hidden) copy runs after the
        // eval's own submit. Queuing a fused readback before evalValues leaves
        // the shared ReadbackQueue occupied when graph collection, allocation
        // or dispatch throws pre-submit, and the next request then fails with
        // 'a readback is already queued'. toArray rejects a pending readback
        // when its own eval fails, so this separate-submit path recovers.
        evalValues(
          [current.logits, ...(readHidden ? [current.hidden] : []), ...cacheValues(current.cache)],
          executor,
        );
        const logitsPromise = toArray(current.logits, executor);
        const hiddenPromise = readHidden ? toArray(current.hidden, executor) : undefined;
        const [logits, hidden] = await Promise.all([logitsPromise, hiddenPromise]);
        await releaseAfterWork();
        if (signal.aborted) {
          throw new RunntimeError('EXECUTION_FAILED', 'LFM3 generation aborted', {
            cause: signal.reason,
          });
        }
        if (
          logits.length === 0 ||
          !logits.every(Number.isFinite) ||
          logits.every((value) => value === 0) ||
          (hidden && !hidden.every(Number.isFinite))
        ) {
          throw new RunntimeError('EXECUTION_FAILED', 'LFM3 generation produced unusable output');
        }
        return { logits, hidden, step: current };
      });
      executor.releaseValues([result.step.logits, ...(readHidden ? [result.step.hidden] : [])]);
      const lastHidden = result.hidden?.slice(-2_048);
      return {
        logits: result.logits,
        ...(lastHidden ? { hidden: lastHidden } : {}),
        cache: result.step.cache,
      };
    } catch (error) {
      await releaseAfterWork();
      if (step) {
        executor.releaseValues([
          step.logits,
          ...(readHidden ? [step.hidden] : []),
          ...cacheValues(step.cache),
        ]);
      }
      throw rethrowRunError(error);
    } finally {
      destroyUploads(uploads);
    }
  };
  const runtime: Lfm3SessionRuntime<Lfm3Cache> = {
    async forward(ids: readonly number[], previous: Lfm3Cache | undefined, signal: AbortSignal) {
      return runStep(
        async (uploads) =>
          model.forwardEmbeddings(await embedTokens(ids, uploads, signal), root, previous, uploads),
        previous,
        signal,
        [],
        false,
      );
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

  let tokenizer: Lfm3Tokenizer | undefined;
  try {
    throwIfAborted(opts.signal);
    const weights =
      config.format === 'gguf'
        ? (gguf = await openGgufWeights(modelPath, opts))
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
    if (gguf) gguf.tensors.clear();
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

    return {
      root,
      executor,
      model,
      tokenizer,
      runtime,
      async forwardEmbeddings(embeddings, previous, signal, uploads) {
        const result = await runStep(
          (owned) => model.forwardEmbeddings(embeddings, root, previous, owned),
          previous,
          signal,
          uploads,
          true,
        );
        if (!result.hidden) {
          throw new RunntimeError('EXECUTION_FAILED', 'LFM3 backbone returned no hidden state');
        }
        return { logits: result.logits, hidden: result.hidden, cache: result.cache };
      },
      async forwardTokens(ids, previous, signal) {
        const result = await runStep(
          async (uploads) =>
            model.forwardEmbeddings(
              await embedTokens(ids, uploads, signal),
              root,
              previous,
              uploads,
            ),
          previous,
          signal,
          [],
          true,
        );
        if (!result.hidden) {
          throw new RunntimeError('EXECUTION_FAILED', 'LFM3 backbone returned no hidden state');
        }
        return { logits: result.logits, hidden: result.hidden, cache: result.cache };
      },
      embedTokens,
    };
  } catch (error) {
    try {
      await runtime.dispose();
    } catch {
      // Preserve the actual load, tokenizer, abort or warm-up failure.
    }
    throwIfAborted(opts.signal);
    if (isRunntimeError(error)) throw error;
    throw asLoadError(error, opts.signal);
  }
}

/** Loads the original floating checkpoint and returns a streaming chat session. */
export async function createLfm3Chat(
  config: Lfm3ChatModel = models.lfm3.LFM3_OMNI_2B.DEFAULT,
  opts: LoadOptions = {},
): Promise<Lfm3ChatSession> {
  const loaded = await loadLfm3Backbone(config, opts);
  return createLfm3Session(loaded.tokenizer, loaded.runtime);
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
