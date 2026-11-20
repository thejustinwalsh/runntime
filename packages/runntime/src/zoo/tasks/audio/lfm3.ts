/** Public LFM3-Omni transcription and streamed 24 kHz speech generation. */

import {
  astype,
  cat,
  evalValues,
  fromGguf,
  httpRangeSource,
  inGpuErrorScopes,
  isRunntimeError,
  parseGgufHeader,
  RunntimeError,
  supportsF16,
  tensor,
  toArray,
  warmUp,
  type RangeSource,
  type Value,
} from '../../../core/index.ts';
import { asLoadError, rethrowRunError } from '../../errors.ts';
import { throwIfAborted, type LoadOptions, type ModelPath } from '../../load.ts';
import { models } from '../../models.ts';
import {
  LFM3_AUDIO_EOS_CODE,
  LFM3_AUDIO_CODEBOOKS,
  LFM3_AUDIO_CODEBOOK_SIZE,
  lfm3AudioTokensForSamples,
  Lfm3AudioModel as Lfm3OmniAudioModel,
  type Lfm3AudioDecoderCache,
} from '../../lfm3/audio/model.ts';
import { LFM3 } from '../../lfm3/model.ts';
import { LiquidCodecStream } from '../../lfm3/audio/codec.ts';
import { encodeLfm3SpeechPrompt, encodeLfm3TranscriptionPrompt } from '../../lfm3/audio/prompts.ts';
import {
  LFM3_AUDIO_MAX_SECONDS,
  LFM3_AUDIO_SAMPLE_RATE,
  lfm3AudioPaddedLength,
} from '../../lfm3/audio/frontend.ts';
import {
  validateLfm3AudioProjector,
  type Lfm3AudioProjectorConfig,
} from '../../lfm3/audio/projector.ts';
import {
  createLfm3DepthSampler,
  lfm3MaskgitOrder,
  lfm3MaskgitRevealSizes,
  sampleLfm3DepthCode,
  type Lfm3DepthSampler,
  type Lfm3DepthSamplingSettings,
} from '../../lfm3/audio/sampling.ts';
import { LFM3_Q4_K_M_SOURCE, LFM3_Q8_0_PROJECTOR_SOURCE } from '../../lfm3/gguf.ts';
import type { Lfm3Backbone, Lfm3ChatModel } from '../nlp/lfm3.ts';
import { loadLfm3Backbone } from '../nlp/lfm3.ts';

/** Sampling options for one audio transcription. Input is mono float PCM at 16 kHz. */
export interface Lfm3TranscriptionOptions {
  readonly maxNewTokens?: number;
  readonly signal?: AbortSignal;
}

/** Options for one streamed speech response. */
export interface Lfm3SpeechOptions {
  /** Maximum number of 80 ms speech frames. Defaults to 256, at most 1024. */
  readonly maxAudioFrames?: number;
  /** Temperature for the depth-code sampler; zero selects the highest logit. */
  readonly temperature?: number;
  /** Number of MaskGIT refinement passes, from 1 through 12. Defaults to 3. */
  readonly depthPasses?: number;
  /** Reproducible codebook order and sampling when supplied. */
  readonly seed?: number;
  readonly signal?: AbortSignal;
}

/** Final samples and generation accounting for one speech response. */
export interface Lfm3SpeechResult {
  readonly samples: Float32Array;
  readonly sampleRate: 24_000;
  readonly audioFrames: number;
  readonly finishReason: 'stop' | 'length';
}

/** Stream of 24 kHz PCM deltas and its final response. */
export interface Lfm3SpeechGeneration extends AsyncIterable<Float32Array> {
  readonly result: Promise<Lfm3SpeechResult>;
  cancel(): void;
}

/** Public exact-pair audio model configuration. */
export interface Lfm3AudioModel extends Lfm3ChatModel {
  /** Exact Liquid Q8_0 audio-projector GGUF, or a caller-owned range source. */
  readonly projectorPath?: ModelPath;
}

/** Native 16 kHz transcription and streamed 24 kHz speech runner. */
export interface Lfm3Audio {
  /** Exact source identities used by this runner; no weight conversion is performed. */
  readonly sources: {
    readonly language: Omit<typeof LFM3_Q4_K_M_SOURCE, 'url'> & { readonly url: string | null };
    readonly projector: Omit<typeof LFM3_Q8_0_PROJECTOR_SOURCE, 'url'> & {
      readonly url: string | null;
    };
  };
  /** Transcribes bounded mono PCM at 16 kHz. */
  transcribe(audio: Float32Array, options?: Lfm3TranscriptionOptions): Promise<string>;
  /** Generates 24 kHz mono PCM, yielding stable sample chunks as codec frames arrive. */
  speak(text: string, options?: Lfm3SpeechOptions): Lfm3SpeechGeneration;
  /** Stops queued and in-flight work and releases the complete model pair. */
  dispose(): Promise<void>;
}

const AUDIO_PROJECTOR_CACHE_PREFIX = 'lfm3-omni-audio-q8-v1';
const SPEECH_SAMPLE_RATE = 24_000;
const DEFAULT_TRANSCRIPTION_TOKENS = 256;
const MAX_TRANSCRIPTION_TOKENS = 4_096;
const DEFAULT_SPEECH_FRAMES = 256;
const MAX_SPEECH_FRAMES = 1_024;

interface LinkedSignal {
  readonly signal: AbortSignal;
  abort(reason?: unknown): void;
  dispose(): void;
}

interface AudioCodecStep {
  readonly spectrum: Float32Array;
  readonly cache: Lfm3AudioDecoderCache;
}

class AudioQueue implements AsyncIterableIterator<Float32Array> {
  private readonly values: Float32Array[] = [];
  private readonly waiters: {
    resolve(result: IteratorResult<Float32Array>): void;
    reject(error: unknown): void;
  }[] = [];
  private ended = false;
  private failure: unknown;

  constructor(private readonly cancel: () => void) {}

  [Symbol.asyncIterator](): AsyncIterableIterator<Float32Array> {
    return this;
  }

  next(): Promise<IteratorResult<Float32Array>> {
    const value = this.values.shift();
    if (value) return Promise.resolve({ done: false, value });
    if (this.failure !== undefined) return Promise.reject(this.failure);
    if (this.ended) return Promise.resolve({ done: true, value: undefined });
    return new Promise((resolve, reject) => this.waiters.push({ resolve, reject }));
  }

  return(): Promise<IteratorResult<Float32Array>> {
    if (!this.ended && this.failure === undefined) this.cancel();
    this.close();
    return Promise.resolve({ done: true, value: undefined });
  }

  push(value: Float32Array): void {
    if (this.ended || this.failure !== undefined || value.length === 0) return;
    const waiter = this.waiters.shift();
    if (waiter) waiter.resolve({ done: false, value });
    else this.values.push(value);
  }

  close(): void {
    if (this.ended || this.failure !== undefined) return;
    this.ended = true;
    for (const waiter of this.waiters.splice(0)) {
      waiter.resolve({ done: true, value: undefined });
    }
  }

  fail(error: unknown): void {
    if (this.ended || this.failure !== undefined) return;
    this.failure = error;
    this.values.length = 0;
    for (const waiter of this.waiters.splice(0)) waiter.reject(error);
  }
}

/** Loads the exact published Q4_K_M language and Q8_0 audio-projector pair. */
export async function createLfm3Audio(
  config: Lfm3AudioModel = models.lfm3.LFM3_OMNI_2B.Q4_K_M,
  opts: LoadOptions = {},
): Promise<Lfm3Audio> {
  let backbone: Lfm3Backbone | undefined;
  let audioModel: Lfm3OmniAudioModel | undefined;
  try {
    throwIfAborted(opts.signal);
    if (!supportsF16()) {
      throw new RunntimeError(
        'UNSUPPORTED_DEVICE',
        "createLfm3Audio requires shader-f16 for its convolutional audio encoder. Request it with tgpu.init({ device: { optionalFeatures: ['shader-f16'] } })",
      );
    }
    const projectorPath = config.projectorPath ?? LFM3_Q8_0_PROJECTOR_SOURCE.url;
    const projectorOptions = opts.cacheId
      ? { ...opts, cacheId: `${opts.cacheId}/audio-projector` }
      : opts;
    const backboneOptions = opts.cacheId
      ? { ...opts, cacheId: `${opts.cacheId}/language-backbone` }
      : opts;
    const projector = await openAudioProjector(projectorPath, projectorOptions);
    throwIfAborted(opts.signal);
    backbone = await loadLfm3Backbone({ ...config, format: 'gguf' }, backboneOptions);
    const root = backbone.root;
    const executor = backbone.executor;
    const liveBackbone = backbone;
    audioModel = new Lfm3OmniAudioModel();
    audioModel.half(root);
    await audioModel.loadStateDict(projector.weights, {
      root,
      onProgress: (name, done, total) => {
        opts.onProgress?.(name, done, total);
        throwIfAborted(opts.signal);
      },
    });
    throwIfAborted(opts.signal);

    const liveAudioModel = audioModel;
    const depthSampling = projector.config.depthDecoder.sampling;
    const controllers = new Set<AbortController>();
    let queue: Promise<void> = Promise.resolve();
    let disposed = false;
    let disposePromise: Promise<void> | undefined;

    const releaseAfterWork = async (): Promise<void> => {
      try {
        await root.device.queue.onSubmittedWorkDone();
      } catch {
        // Device loss rejects the wait; owned GPU buffers still need teardown.
      }
    };
    const guard = (): void => {
      if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'LFM3 audio runner is disposed');
    };
    const linkSignal = (signal?: AbortSignal): LinkedSignal => {
      const controller = new AbortController();
      const forwardAbort = () => controller.abort(signal?.reason);
      if (signal?.aborted) forwardAbort();
      else signal?.addEventListener('abort', forwardAbort, { once: true });
      controllers.add(controller);
      return {
        signal: controller.signal,
        abort: (reason) => controller.abort(reason),
        dispose() {
          signal?.removeEventListener('abort', forwardAbort);
          controllers.delete(controller);
        },
      };
    };
    const schedule = <T>(linked: LinkedSignal, operation: () => Promise<T>): Promise<T> => {
      const run = queue.then(async () => {
        guard();
        checkAborted(linked.signal);
        return operation();
      });
      queue = run.then(
        () => undefined,
        () => undefined,
      );
      return run.catch(rethrowRunError);
    };

    const encodeAudioPrompt = (audio: Float32Array) => {
      validateAudio(audio);
      const uploads: Value[] = [];
      try {
        const prompt = encodeLfm3TranscriptionPrompt(liveBackbone.tokenizer);
        const positions = prompt.flatMap((token, index) =>
          token === liveBackbone.tokenizer.audioId ? [index] : [],
        );
        if (positions.length !== 1) {
          throw new RunntimeError(
            'CHECKPOINT_MISMATCH',
            'LFM3 audio prompt must contain exactly one audio placeholder',
          );
        }
        const features = astype(liveAudioModel.encodeAudio(audio, root, uploads), 'f32');
        const audioTokens = features.shape.dims?.[0];
        const expectedTokens = lfm3AudioTokensForSamples(audio.length);
        if (audioTokens !== expectedTokens) {
          throw new RunntimeError(
            'EXECUTION_FAILED',
            `LFM3 audio encoder returned ${audioTokens ?? 'an unknown number of'} tokens; expected ${expectedTokens}`,
          );
        }
        const at = positions[0]!;
        const ids = [
          ...prompt.slice(0, at),
          ...Array.from({ length: expectedTokens }, () => liveBackbone.tokenizer.audioId),
          ...prompt.slice(at + 1),
        ];
        const parts: Value[] = [];
        if (at > 0) parts.push(liveBackbone.embedTokens(ids.slice(0, at), uploads));
        parts.push(features);
        if (prompt.length - at - 1 > 0) {
          parts.push(liveBackbone.embedTokens(ids.slice(at + expectedTokens), uploads));
        }
        return {
          ids,
          embeddings: parts.length === 1 ? parts[0]! : cat(parts, 0),
          uploads,
        };
      } catch (error) {
        destroyUploads(uploads);
        throw error;
      }
    };

    const transcribeNow = async (
      audio: Float32Array,
      options: Lfm3TranscriptionOptions,
      signal: AbortSignal,
    ): Promise<string> => {
      const maxNewTokens = validateTranscriptionLimit(options.maxNewTokens);
      const prompt = encodeAudioPrompt(audio);
      let uploadsHandedOff = false;
      let cache: Awaited<ReturnType<Lfm3Backbone['forwardTokens']>>['cache'] | undefined;
      try {
        checkAborted(signal);
        uploadsHandedOff = true;
        const first = await liveBackbone.forwardEmbeddings(
          prompt.embeddings,
          undefined,
          signal,
          prompt.uploads,
        );
        cache = first.cache;
        let logits = first.logits;
        const limit = Math.min(maxNewTokens, LFM3.maxContext - prompt.ids.length);
        const generated: number[] = [];
        for (let index = 0; index < limit; index++) {
          checkAborted(signal);
          const token = argmax(logits);
          if (liveBackbone.tokenizer.stopIds.has(token)) break;
          generated.push(token);
          const step = await liveBackbone.forwardTokens([token], cache, signal);
          await liveBackbone.runtime.release(cache);
          cache = step.cache;
          logits = step.logits;
        }
        return liveBackbone.tokenizer.decode(generated).trim();
      } finally {
        if (!uploadsHandedOff) destroyUploads(prompt.uploads);
        if (cache) await liveBackbone.runtime.release(cache);
      }
    };

    const depthLogits = async (
      hidden: Float32Array,
      codes: readonly number[],
      masked: readonly boolean[],
      signal: AbortSignal,
    ): Promise<Float32Array[]> => {
      let values: Value[] = [];
      const uploads: Value[] = [];
      try {
        return await inGpuErrorScopes(root.device, 'LFM3 depth decoder', async () => {
          checkAborted(signal);
          const hiddenUpload = tensor(root, hidden, [1, 2_048]);
          uploads.push(hiddenUpload);
          const hiddenValue = astype(hiddenUpload, 'f16');
          values = liveAudioModel.forwardDepthFrame(hiddenValue, codes, masked, root, uploads);
          evalValues(values, executor);
          const logits = await Promise.all(values.map((value) => toArray(value, executor)));
          await releaseAfterWork();
          checkAborted(signal);
          return logits;
        }).then((logits) => {
          executor.releaseValues(values);
          return logits;
        });
      } catch (error) {
        await releaseAfterWork();
        executor.releaseValues(values);
        throw rethrowRunError(error);
      } finally {
        destroyUploads(uploads);
      }
    };

    const codecCacheValues = (cache: Lfm3AudioDecoderCache): Value[] =>
      cache.layers.flatMap((layer) => {
        if (!layer) return [];
        return [
          ...(layer.attention ? [layer.attention.key, layer.attention.value] : []),
          ...(layer.convolution ? [layer.convolution] : []),
        ];
      });

    const decodeCodecFrame = async (
      codes: readonly number[],
      previous: Lfm3AudioDecoderCache | undefined,
      signal: AbortSignal,
    ): Promise<AudioCodecStep> => {
      let spectrum: Value | undefined;
      let cache: Lfm3AudioDecoderCache | undefined;
      const uploads: Value[] = [];
      try {
        const decoded = await inGpuErrorScopes(root.device, 'LiquidCodec decoder', async () => {
          checkAborted(signal);
          const output = liveAudioModel.decodeSpeechCodes(codes, root, uploads, previous);
          spectrum = output.spectrum;
          cache = output.cache;
          evalValues([output.spectrum, ...codecCacheValues(output.cache)], executor);
          const values = await toArray(output.spectrum, executor);
          await releaseAfterWork();
          checkAborted(signal);
          return { spectrum: values, cache: output.cache };
        });
        executor.releaseValues([spectrum!]);
        return decoded;
      } catch (error) {
        await releaseAfterWork();
        executor.releaseValues([
          ...(spectrum ? [spectrum] : []),
          ...(cache ? codecCacheValues(cache) : []),
        ]);
        throw rethrowRunError(error);
      } finally {
        destroyUploads(uploads);
      }
    };

    const warmAudioPaths = async (): Promise<Float32Array> => {
      const warmSignal = opts.signal ?? new AbortController().signal;
      const encoded = encodeAudioPrompt(new Float32Array(LFM3_AUDIO_SAMPLE_RATE));
      let transcriptionCache:
        Awaited<ReturnType<Lfm3Backbone['forwardTokens']>>['cache'] | undefined;
      let speechCache: Awaited<ReturnType<Lfm3Backbone['forwardTokens']>>['cache'] | undefined;
      let codecCache: Lfm3AudioDecoderCache | undefined;
      try {
        const language = await liveBackbone.forwardEmbeddings(
          encoded.embeddings,
          undefined,
          warmSignal,
          encoded.uploads,
        );
        transcriptionCache = language.cache;
        await liveBackbone.runtime.release(transcriptionCache);
        transcriptionCache = undefined;

        const speechPrompt = encodeLfm3SpeechPrompt(liveBackbone.tokenizer, 'Hello.');
        const speech = await liveBackbone.forwardTokens(speechPrompt, undefined, warmSignal);
        speechCache = speech.cache;
        const logits = await depthLogits(
          speech.hidden,
          Array(LFM3_AUDIO_CODEBOOKS).fill(0),
          Array(LFM3_AUDIO_CODEBOOKS).fill(true),
          warmSignal,
        );
        const codes = logits.map((book) => argmax(book, LFM3_AUDIO_EOS_CODE));
        const decoded = await decodeCodecFrame(codes, undefined, warmSignal);
        codecCache = decoded.cache;
        const uploads: Value[] = [];
        const embedding = astype(liveAudioModel.audioOutEmbedding(codes, root, uploads), 'f32');
        const speechStep = await liveBackbone.forwardEmbeddings(
          embedding,
          speechCache,
          warmSignal,
          uploads,
        );
        await liveBackbone.runtime.release(speechCache);
        speechCache = speechStep.cache;
        const stream = new LiquidCodecStream();
        const first = stream.decode(decoded.spectrum);
        const last = stream.finish();
        const samples = joinAudio(first, last);
        if (samples.length === 0) {
          throw new RunntimeError('EXECUTION_FAILED', 'LiquidCodec warm-up returned no samples');
        }
        return samples;
      } finally {
        if (transcriptionCache) await liveBackbone.runtime.release(transcriptionCache);
        if (speechCache) await liveBackbone.runtime.release(speechCache);
        if (codecCache) executor.releaseValues(codecCacheValues(codecCache));
      }
    };

    const speakNow = async (
      text: string,
      options: Lfm3SpeechOptions,
      signal: AbortSignal,
      emit: (samples: Float32Array) => void,
    ): Promise<Lfm3SpeechResult> => {
      if (!text.trim())
        throw new RunntimeError('INVALID_ARGUMENT', 'speech text must not be empty');
      const maxAudioFrames = validateSpeechFrameLimit(options.maxAudioFrames);
      const temperature = validateTemperature(options.temperature, depthSampling.temperature);
      const depthPasses = validateDepthPasses(options.depthPasses);
      const sampler = createLfm3DepthSampler(options.seed ?? randomSeed());
      const prompt = encodeLfm3SpeechPrompt(liveBackbone.tokenizer, text);
      let languageCache: Awaited<ReturnType<Lfm3Backbone['forwardTokens']>>['cache'] | undefined;
      let codecCache: Lfm3AudioDecoderCache | undefined;
      let hidden: Float32Array | undefined;
      const chunks: Float32Array[] = [];
      const decoder = new LiquidCodecStream();
      let finishReason: Lfm3SpeechResult['finishReason'] = 'length';
      let audioFrames = 0;
      try {
        checkAborted(signal);
        const first = await liveBackbone.forwardTokens(prompt, undefined, signal);
        languageCache = first.cache;
        hidden = first.hidden;
        const history: number[][] = [];
        for (let frame = 0; frame < maxAudioFrames; frame++) {
          checkAborted(signal);
          const codes = await sampleAudioFrame(
            hidden,
            history,
            frame,
            depthPasses,
            temperature,
            depthSampling,
            sampler,
            depthLogits,
            signal,
          );
          if (codes.some((code) => code === LFM3_AUDIO_EOS_CODE)) {
            finishReason = 'stop';
            break;
          }
          history.push(codes);
          audioFrames++;

          const codec = await decodeCodecFrame(codes, codecCache, signal);
          if (codecCache) executor.releaseValues(codecCacheValues(codecCache));
          codecCache = codec.cache;
          const samples = decoder.decode(codec.spectrum);
          if (samples.length > 0) {
            chunks.push(samples);
            emit(samples);
          }

          const uploads: Value[] = [];
          const embedding = astype(liveAudioModel.audioOutEmbedding(codes, root, uploads), 'f32');
          const step = await liveBackbone.forwardEmbeddings(
            embedding,
            languageCache,
            signal,
            uploads,
          );
          if (languageCache) await liveBackbone.runtime.release(languageCache);
          languageCache = step.cache;
          hidden = step.hidden;
        }
        const tail = decoder.finish();
        if (tail.length > 0) {
          chunks.push(tail);
          emit(tail);
        }
        if (audioFrames === 0 || chunks.length === 0) {
          throw new RunntimeError(
            'EXECUTION_FAILED',
            'LFM3 speech generation returned no audio frames',
          );
        }
        const samples = joinAudio(...chunks);
        if (!samples.every(Number.isFinite)) {
          throw new RunntimeError(
            'EXECUTION_FAILED',
            'LFM3 speech generation returned non-finite samples',
          );
        }
        return { samples, sampleRate: SPEECH_SAMPLE_RATE, audioFrames, finishReason };
      } finally {
        if (languageCache) await liveBackbone.runtime.release(languageCache);
        if (codecCache) executor.releaseValues(codecCacheValues(codecCache));
      }
    };

    if (opts.warmUp !== false) {
      await warmUp(root.device, warmAudioPaths, 'LFM3 audio and speech warm-up');
      throwIfAborted(opts.signal);
      executor.trim();
    }

    return {
      sources: {
        language: {
          ...LFM3_Q4_K_M_SOURCE,
          url: typeof config.modelPath === 'string' ? config.modelPath : LFM3_Q4_K_M_SOURCE.url,
        },
        projector: {
          ...LFM3_Q8_0_PROJECTOR_SOURCE,
          url: typeof projectorPath === 'string' ? projectorPath : null,
        },
      },
      transcribe(audio, options = {}) {
        guard();
        const linked = linkSignal(options.signal);
        return schedule(linked, () => transcribeNow(audio, options, linked.signal)).finally(
          linked.dispose,
        );
      },
      speak(text, options = {}) {
        guard();
        if (typeof text !== 'string' || !text.trim()) {
          throw new RunntimeError('INVALID_ARGUMENT', 'speech text must not be empty');
        }
        validateSpeechOptions(options, depthSampling);
        const linked = linkSignal(options.signal);
        const stream = new AudioQueue(() => linked.abort());
        const task = schedule(linked, () =>
          speakNow(text, options, linked.signal, (samples) => stream.push(samples)),
        );
        const result = task
          .then(
            (value) => {
              stream.close();
              return value;
            },
            (error: unknown) => {
              stream.fail(error);
              throw error;
            },
          )
          .finally(linked.dispose);
        // Stream-only and early-break consumers may never await `result`;
        // hold an internal observer so a late rejection never surfaces as an
        // unhandled rejection, mirroring the chat session. Awaiting `result`
        // still observes the original outcome.
        result.catch(() => undefined);
        return { result, cancel: () => linked.abort(), [Symbol.asyncIterator]: () => stream };
      },
      async dispose() {
        if (!disposePromise) {
          disposed = true;
          for (const controller of controllers) controller.abort();
          disposePromise = queue.then(async () => {
            await releaseAfterWork();
            try {
              liveAudioModel.dispose();
            } finally {
              await liveBackbone.runtime.dispose();
            }
          });
        }
        await disposePromise;
      },
    };
  } catch (error) {
    try {
      audioModel?.dispose();
    } catch {
      // Preserve the actual projector, backbone, abort or warm-up failure.
    } finally {
      try {
        await backbone?.runtime.dispose();
      } catch {
        // Preserve the actual projector, backbone, abort or warm-up failure.
      }
    }
    throwIfAborted(opts.signal);
    if (isRunntimeError(error)) throw error;
    throw asLoadError(error, opts.signal);
  }
}

async function openAudioProjector(path: ModelPath, opts: LoadOptions) {
  const source: RangeSource =
    typeof path === 'string' ? httpRangeSource(path, { onBytes: opts.onBytes }) : path;
  const index = await parseGgufHeader(source, LFM3_Q8_0_PROJECTOR_SOURCE.bytes);
  const config: Lfm3AudioProjectorConfig = validateLfm3AudioProjector(index);
  throwIfAborted(opts.signal);
  return {
    weights: await fromGguf(source, {
      fileBytes: LFM3_Q8_0_PROJECTOR_SOURCE.bytes,
      sha256: LFM3_Q8_0_PROJECTOR_SOURCE.sha256,
      cache: opts.cache,
      cacheId: opts.cacheId ?? projectorCacheKey(path),
      signal: opts.signal,
    }),
    config,
  };
}

async function sampleAudioFrame(
  hidden: Float32Array,
  history: readonly number[][],
  frame: number,
  passes: number,
  temperature: number,
  sampling: Lfm3DepthSamplingSettings,
  sampler: Lfm3DepthSampler,
  forward: (
    hidden: Float32Array,
    codes: readonly number[],
    masked: readonly boolean[],
    signal: AbortSignal,
  ) => Promise<Float32Array[]>,
  signal: AbortSignal,
): Promise<number[]> {
  const order = lfm3MaskgitOrder(LFM3_AUDIO_CODEBOOKS, sampler.nextRandom);
  const revealSizes = lfm3MaskgitRevealSizes(LFM3_AUDIO_CODEBOOKS, passes);
  const codes = Array<number>(LFM3_AUDIO_CODEBOOKS).fill(0);
  const revealed = Array<boolean>(LFM3_AUDIO_CODEBOOKS).fill(false);
  const previous = sampling.repetitionWindow === 0 ? [] : history.slice(-sampling.repetitionWindow);
  let start = 0;
  for (const count of revealSizes) {
    const selected = order.slice(start, start + count);
    const logits = await forward(
      hidden,
      codes,
      revealed.map((value) => !value),
      signal,
    );
    if (logits.length !== LFM3_AUDIO_CODEBOOKS) {
      throw new RunntimeError(
        'EXECUTION_FAILED',
        `LFM3 depth decoder returned ${logits.length} codebooks; expected ${LFM3_AUDIO_CODEBOOKS}`,
      );
    }
    for (const book of selected) {
      const values = logits[book]!;
      if (values.length !== LFM3_AUDIO_CODEBOOK_SIZE) {
        throw new RunntimeError(
          'EXECUTION_FAILED',
          `LFM3 depth decoder returned ${values.length} logits for codebook ${book}`,
        );
      }
      codes[book] = sampleLfm3DepthCode({
        logits: values,
        codebookSize: LFM3_AUDIO_CODEBOOK_SIZE,
        codebook: book,
        frame,
        samplerSeed: sampler.seed,
        temperature,
        repetitionPenalty: sampling.repetitionPenalty,
        previousCodes: previous.map((previousFrame) => previousFrame[book]!),
      });
      revealed[book] = true;
    }
    start += selected.length;
  }
  if (codes.some((code) => code === LFM3_AUDIO_EOS_CODE)) {
    return Array<number>(LFM3_AUDIO_CODEBOOKS).fill(LFM3_AUDIO_EOS_CODE);
  }
  return codes;
}

function argmax(values: Float32Array, limit = values.length): number {
  if (limit < 1 || values.length < limit) {
    throw new RunntimeError(
      'EXECUTION_FAILED',
      `LFM3 returned an invalid logits length ${values.length}`,
    );
  }
  let best = 0;
  for (let index = 0; index < limit; index++) {
    const value = values[index]!;
    if (!Number.isFinite(value)) {
      throw new RunntimeError('EXECUTION_FAILED', `LFM3 produced a non-finite logit at ${index}`);
    }
    if (value > values[best]!) best = index;
  }
  return best;
}

function randomSeed(): number {
  if (typeof crypto !== 'undefined' && 'getRandomValues' in crypto) {
    return crypto.getRandomValues(new Uint32Array(1))[0]!;
  }
  return Math.floor(Math.random() * 4_294_967_296);
}

function validateAudio(audio: Float32Array): void {
  if (!(audio instanceof Float32Array)) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      'LFM3 audio must be mono Float32Array PCM at 16 kHz',
    );
  }
  lfm3AudioPaddedLength(audio.length);
  if (!audio.every(Number.isFinite)) {
    throw new RunntimeError('INVALID_ARGUMENT', 'LFM3 audio samples must all be finite');
  }
  if (audio.length > LFM3_AUDIO_MAX_SECONDS * LFM3_AUDIO_SAMPLE_RATE) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `LFM3 audio is limited to ${LFM3_AUDIO_MAX_SECONDS} seconds`,
    );
  }
}

function validateTranscriptionLimit(value: number | undefined): number {
  const limit = value ?? DEFAULT_TRANSCRIPTION_TOKENS;
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_TRANSCRIPTION_TOKENS) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `maxNewTokens must be an integer in [1, ${MAX_TRANSCRIPTION_TOKENS}]`,
    );
  }
  return limit;
}

function validateSpeechFrameLimit(value: number | undefined): number {
  const limit = value ?? DEFAULT_SPEECH_FRAMES;
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_SPEECH_FRAMES) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `maxAudioFrames must be an integer in [1, ${MAX_SPEECH_FRAMES}]`,
    );
  }
  return limit;
}

function validateTemperature(value: number | undefined, defaultTemperature: number): number {
  const temperature = value ?? defaultTemperature;
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
    throw new RunntimeError('INVALID_ARGUMENT', 'temperature must be finite and in [0, 2]');
  }
  return temperature;
}

function validateDepthPasses(value: number | undefined): number {
  const passes = value ?? 3;
  if (!Number.isInteger(passes) || passes < 1 || passes > LFM3_AUDIO_CODEBOOKS) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `depthPasses must be an integer in [1, ${LFM3_AUDIO_CODEBOOKS}]`,
    );
  }
  return passes;
}

function validateSpeechOptions(
  options: Lfm3SpeechOptions,
  sampling: Lfm3DepthSamplingSettings,
): void {
  validateSpeechFrameLimit(options.maxAudioFrames);
  validateTemperature(options.temperature, sampling.temperature);
  validateDepthPasses(options.depthPasses);
  if (
    options.seed !== undefined &&
    (!Number.isSafeInteger(options.seed) || options.seed < 0 || options.seed > 0xffff_ffff)
  ) {
    throw new RunntimeError('INVALID_ARGUMENT', 'seed must be an unsigned 32-bit integer');
  }
}

function checkAborted(signal: AbortSignal): void {
  if (signal.aborted) {
    throw new RunntimeError('EXECUTION_FAILED', 'LFM3 audio operation aborted', {
      cause: signal.reason,
    });
  }
}

function joinAudio(...parts: readonly Float32Array[]): Float32Array {
  const output = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}

function destroyUploads(values: readonly Value[]): void {
  for (const value of values) {
    if (value.state === 'materialized') value.buffer.destroy();
  }
}

const projectorCacheKey = (path: ModelPath): string | undefined =>
  typeof path === 'string'
    ? `${AUDIO_PROJECTOR_CACHE_PREFIX}/${path.replace(/^https?:\/\//, '')}`
    : undefined;
