/** D1 Omni text decisions on the initRunntime() device. */

import { defaultRoot, RunntimeError, warmUp } from '../../../core/index.ts';
import { asLoadError, rethrowRunError } from '../../errors.ts';
import {
  fetchJson,
  openWeights,
  throwIfAborted,
  type LoadOptions,
  type ModelPath,
} from '../../load.ts';
import { models } from '../../models.ts';
import { createD1TextDecider, type D1Decider as D1CoreDecider } from '../../d1/decider.ts';
import { createD1Tokenizer } from '../../d1/tokenizer.ts';
import type { Answer, Question } from '../../d1/prompt.ts';

/** What to load. Q8 weights are caller-provided; no derivative weights are hosted. */
export interface D1DeciderModel {
  /** A native runntime safetensors checkpoint, URL or caller-owned RangeSource. */
  readonly modelPath?: ModelPath;
  /** The Hugging Face tokenizer.json. Defaults to the pinned D1 tokenizer. */
  readonly tokenizerPath?: string;
  /** Calibration temperatures keyed by question type and option count. */
  readonly temperatures?: Record<string, number>;
}

/** Text-only D1 Omni decision runner. */
export interface D1Decider {
  /** Answers named yes/no, choice, or rubric questions over one text state. */
  systemOne(
    state: string,
    questions: Record<string, Question>,
    options?: { signal?: AbortSignal },
  ): Promise<{ answers: Record<string, Answer>; inputTokens: number }>;
  /** Stops accepting work and releases the model's owned resources. */
  dispose(): Promise<void>;
}

/** Loads D1 text weights and tokenizer, optionally warms the model, and returns a serialized runner. */
export async function createD1Decider(
  config: D1DeciderModel = models.d1.D1_OMNI_600M.DEFAULT,
  opts: LoadOptions = {},
): Promise<D1Decider> {
  const { modelPath, tokenizerPath, temperatures } = config;
  if (modelPath === undefined) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      'createD1Decider: modelPath is required for a caller-provided D1 Q8 artifact',
    );
  }

  let decider: D1CoreDecider | undefined;
  try {
    const device = defaultRoot().device;
    throwIfAborted(opts.signal);
    const weights = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? cacheKey(modelPath),
    });
    const tokPath = tokenizerPath ?? models.d1.D1_OMNI_600M.DEFAULT.tokenizerPath!;
    const json = await fetchJson(tokPath, {
      cache: opts.cache,
      cacheId: cacheKey(tokPath),
      signal: opts.signal,
    });
    const tokenizer = createD1Tokenizer(json);

    throwIfAborted(opts.signal);
    decider = await createD1TextDecider(weights, tokenizer, {
      temperatures,
      onProgress: opts.onProgress,
      signal: opts.signal,
    });
    if (opts.warmUp !== false) {
      await warmUp(device, async () => {
        const { answers } = await decider!.systemOne('', {
          warmup: { type: 'noul', instructions: 'Is this a decision?' },
        });
        const answer = answers.warmup!;
        if (answer.type !== 'noul') {
          throw new RunntimeError('EXECUTION_FAILED', 'D1 warm-up returned a non-binary answer');
        }
        return Float32Array.of(answer.noul, 1 - answer.noul);
      });
    }

    return {
      systemOne: (state, questions, callOptions) =>
        decider!.systemOne(state, questions, undefined, callOptions).catch(rethrowRunError),
      dispose: () => decider!.dispose(),
    };
  } catch (err) {
    try {
      await decider?.dispose();
    } catch {
      // Keep the load or warm-up failure as the visible error.
    }
    throw asLoadError(err, opts.signal);
  }
}

const cacheKey = (path: ModelPath): string | undefined =>
  typeof path === 'string' ? `d1-v1/${path.replace(/^https?:\/\//, '')}` : undefined;
