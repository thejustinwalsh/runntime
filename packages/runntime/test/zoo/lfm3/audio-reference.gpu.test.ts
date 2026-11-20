/** Compares the learned audio encoder, depth decoder and codec with pinned Liquid. */

import { expect, it } from 'vitest';
import {
  evalValues,
  fromGguf,
  gpuExecutor,
  httpRangeSource,
  tensor,
  toArray,
  type Value,
} from '../../../src/core/index.ts';
import { decodeLiquidCodecSpectrum } from '../../../src/zoo/lfm3/audio/codec.ts';
import {
  LFM3_AUDIO_CODEBOOKS,
  LFM3_AUDIO_CODEBOOK_SIZE,
  Lfm3AudioModel,
  type Lfm3AudioDecoderCache,
} from '../../../src/zoo/lfm3/audio/model.ts';
import { createLfm3DepthSampler, lfm3MaskgitOrder } from '../../../src/zoo/lfm3/audio/sampling.ts';
import { LFM3_Q8_0_PROJECTOR_SOURCE } from '../../../src/zoo/lfm3/gguf.ts';
import { gpu } from '../../gpu.ts';

interface Message {
  readonly role: 'system' | 'user';
  readonly content: string;
}

interface LiquidCache {
  get_seq_length(): number;
}

interface LiquidGenerationState {
  readonly cache: LiquidCache;
}

interface LiquidDepthSettings {
  readonly do_sample: boolean;
  readonly temperature: number;
  readonly repetition_penalty: number;
  readonly repetition_window: number;
}

interface LiquidDepthSampler {
  readonly seed: number;
  readonly settings: LiquidDepthSettings;
  windowStart(position: number): number;
}

interface LiquidLanguageModel {
  readonly runtime: {
    readTensor(value: unknown): Promise<Float32Array>;
  };
  createGenerationState(options: { maxLength: number }): LiquidGenerationState;
  disposeGenerationState(state: LiquidGenerationState): void;
  prefillTokens(cache: LiquidCache, ids: Uint32Array): Promise<void>;
  submitDecodeStep(
    cache: LiquidCache,
    position: number,
    input: { token: number; predict: boolean },
  ): Promise<{ next: Promise<number>; hidden: unknown }>;
}

interface LiquidReference {
  readonly model: {
    readonly language_model: LiquidLanguageModel;
    readonly audio_encoder: { encode(samples: Float32Array): Promise<unknown> };
    readonly depth_decoder: {
      submitFrame(
        hidden: unknown,
        order: readonly number[],
        sampler: LiquidDepthSampler,
        frame: number,
        position: number,
      ): Promise<{ codes: Promise<Int32Array>; row: unknown }>;
    };
    readonly codec: { decode(codes: Int32Array): Promise<Float32Array> };
    readonly tokens: { readonly audioStart: number };
  };
  readonly tokenizer: {
    encode(text: string, options?: { add_special_tokens?: boolean }): { ids: number[] };
    decode(ids: readonly number[]): string;
  };
  encodePrompt(messages: readonly Message[]): number[];
}

interface LiquidModel {
  readonly reference: LiquidReference;
  dispose(): Promise<void>;
}

interface LiquidEngineModule {
  readonly Lfm3OmniForConditionalGeneration: {
    from_pretrained(model: string, options: { mmproj: string }): Promise<LiquidModel>;
  };
}

declare global {
  interface ImportMetaEnv {
    readonly VITE_LFM3_QUALIFICATION_MODE?: string;
    readonly VITE_LFM3_LIQUID_ENGINE_URL?: string;
    readonly VITE_LFM3_LIQUID_MODEL_URL?: string;
    readonly VITE_LFM3_LIQUID_PROJECTOR_URL?: string;
    readonly VITE_LFM3_LIQUID_AUDIO_INPUT_BASE_URL?: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const MODE = import.meta.env.VITE_LFM3_QUALIFICATION_MODE;
const ENGINE_URL = import.meta.env.VITE_LFM3_LIQUID_ENGINE_URL;
const MODEL_URL = import.meta.env.VITE_LFM3_LIQUID_MODEL_URL;
const PROJECTOR_URL = import.meta.env.VITE_LFM3_LIQUID_PROJECTOR_URL;
const AUDIO_INPUT_BASE_URL = import.meta.env.VITE_LFM3_LIQUID_AUDIO_INPUT_BASE_URL;
const TTS_SYSTEM_PROMPT = 'Perform TTS in American English using a female voice.';
const REQUESTED_TEXT = 'The quick brown fox jumps over the lazy dog.';
const REVEAL_SIZES = [4, 4, 4] as const;

function isLiquidEngineModule(value: unknown): value is LiquidEngineModule {
  if (
    typeof value !== 'object' ||
    value === null ||
    !('Lfm3OmniForConditionalGeneration' in value)
  ) {
    return false;
  }
  const constructor = Reflect.get(value, 'Lfm3OmniForConditionalGeneration');
  return (
    (typeof constructor === 'object' || typeof constructor === 'function') &&
    constructor !== null &&
    typeof Reflect.get(constructor, 'from_pretrained') === 'function'
  );
}

function decodePcm16Wav(data: ArrayBuffer): Float32Array {
  const bytes = new Uint8Array(data);
  const view = new DataView(data);
  if (bytes.length < 12 || new TextDecoder().decode(bytes.subarray(0, 4)) !== 'RIFF') {
    throw new Error('frozen audio fixture is not a RIFF WAV');
  }
  let channels = 0;
  let sampleRate = 0;
  let format = 0;
  let bits = 0;
  let dataOffset = -1;
  let dataLength = 0;
  for (let offset = 12; offset + 8 <= bytes.length;) {
    const chunk = new TextDecoder().decode(bytes.subarray(offset, offset + 4));
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (body + size > bytes.length) throw new Error(`truncated WAV ${chunk} chunk`);
    if (chunk === 'fmt ') {
      format = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
    } else if (chunk === 'data') {
      dataOffset = body;
      dataLength = size;
    }
    offset = body + size + (size % 2);
  }
  if (format !== 1 || channels !== 1 || sampleRate !== 16_000 || bits !== 16) {
    throw new Error('frozen audio fixture is not mono 16-bit 16 kHz PCM WAV');
  }
  if (dataOffset < 0 || dataLength === 0 || dataLength % 2 !== 0) {
    throw new Error('frozen audio fixture has an invalid PCM data chunk');
  }
  const samples = new Float32Array(dataLength / 2);
  for (let index = 0; index < samples.length; index++) {
    samples[index] = view.getInt16(dataOffset + index * 2, true) / 32_768;
  }
  return samples;
}

function destroyUploads(values: readonly Value[]): void {
  for (const value of values) {
    if (value.state === 'materialized') value.buffer.destroy();
  }
}

function audioCacheValues(cache: Lfm3AudioDecoderCache): Value[] {
  const values: Value[] = [];
  for (const layer of cache.layers) {
    if (layer?.attention) values.push(layer.attention.key, layer.attention.value);
    if (layer?.convolution) values.push(layer.convolution);
  }
  return values;
}

function argmax(values: Float32Array): number {
  let best = 0;
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index]!)) {
      throw new Error(`native depth decoder returned a non-finite logit at ${index}`);
    }
    if (values[index]! > values[best]!) best = index;
  }
  return best;
}

function compare(actual: Float32Array, expected: Float32Array) {
  if (actual.length !== expected.length) {
    throw new Error(
      `reference length ${expected.length} differs from native length ${actual.length}`,
    );
  }
  let squaredError = 0;
  let dot = 0;
  let actualNorm = 0;
  let expectedNorm = 0;
  let maxAbsoluteError = 0;
  for (let index = 0; index < actual.length; index++) {
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
    values: actual.length,
    rmsError: Math.sqrt(squaredError / actual.length),
    maxAbsoluteError,
    cosineSimilarity: dot / Math.sqrt(actualNorm * expectedNorm),
  };
}

it.skipIf(MODE !== 'liquid-reference')(
  'matches native audio encoder, depth codes and codec PCM against pinned Liquid',
  async () => {
    if (!ENGINE_URL || !MODEL_URL || !PROJECTOR_URL || !AUDIO_INPUT_BASE_URL) {
      throw new Error('pinned Liquid engine, model, projector and frozen-audio URLs are required');
    }
    const engineModule: unknown = await import(/* @vite-ignore */ ENGINE_URL);
    if (!isLiquidEngineModule(engineModule)) throw new Error('Liquid engine exports are invalid');
    const inputResponse = await fetch(
      `${AUDIO_INPUT_BASE_URL.replace(/\/$/, '')}/quick-brown-fox.wav`,
    );
    if (!inputResponse.ok)
      throw new Error(`failed to load frozen audio: HTTP ${inputResponse.status}`);
    const samples = decodePcm16Wav(await inputResponse.arrayBuffer());

    let liquidModel: LiquidModel | undefined;
    let liquidState: LiquidGenerationState | undefined;
    let expectedEncoder: Float32Array | undefined;
    let expectedHidden: Float32Array | undefined;
    let expectedDepthCodes: Int32Array | undefined;
    let expectedCodecPcm: Float32Array | undefined;
    try {
      liquidModel = await engineModule.Lfm3OmniForConditionalGeneration.from_pretrained(MODEL_URL, {
        mmproj: PROJECTOR_URL,
      });
      const reference = liquidModel.reference;
      const languageModel = reference.model.language_model;
      const encoder = await reference.model.audio_encoder.encode(samples);
      expectedEncoder = await languageModel.runtime.readTensor(encoder);

      const promptIds = reference.encodePrompt([
        { role: 'system', content: TTS_SYSTEM_PROMPT },
        { role: 'user', content: REQUESTED_TEXT },
      ]);
      liquidState = languageModel.createGenerationState({ maxLength: promptIds.length + 1 });
      await languageModel.prefillTokens(liquidState.cache, Uint32Array.from(promptIds));
      const step = await languageModel.submitDecodeStep(
        liquidState.cache,
        liquidState.cache.get_seq_length(),
        { token: reference.model.tokens.audioStart, predict: false },
      );
      await step.next;
      expectedHidden = await languageModel.runtime.readTensor(step.hidden);

      const sampler = createLfm3DepthSampler(1_234);
      const order = lfm3MaskgitOrder(LFM3_AUDIO_CODEBOOKS, sampler.nextRandom);
      const liquidSampler: LiquidDepthSampler = {
        seed: sampler.seed,
        settings: {
          do_sample: false,
          temperature: 0.4,
          repetition_penalty: 1.2,
          repetition_window: 64,
        },
        windowStart: () => Infinity,
      };
      const depth = await reference.model.depth_decoder.submitFrame(
        step.hidden,
        order,
        liquidSampler,
        0,
        0,
      );
      expectedDepthCodes = await depth.codes;

      const codecCodes = Int32Array.from(
        { length: LFM3_AUDIO_CODEBOOKS },
        (_, book) => (book * 17 + 31) % (LFM3_AUDIO_CODEBOOK_SIZE - 1),
      );
      expectedCodecPcm = await reference.model.codec.decode(codecCodes);
    } finally {
      if (liquidModel && liquidState) {
        liquidModel.reference.model.language_model.disposeGenerationState(liquidState);
      }
      await liquidModel?.dispose();
    }

    if (!expectedEncoder || !expectedHidden || !expectedDepthCodes || !expectedCodecPcm) {
      throw new Error('Liquid audio reference did not return its expected vectors');
    }

    const { root } = await gpu();
    const executor = gpuExecutor(root);
    const audioModel = new Lfm3AudioModel();
    let modelLoaded = false;
    const result = {
      projector: {
        repository: 'LiquidAI/LFM3-Omni-2B-GGUF',
        revision: LFM3_Q8_0_PROJECTOR_SOURCE.revision,
        file: LFM3_Q8_0_PROJECTOR_SOURCE.file,
        sha256: LFM3_Q8_0_PROJECTOR_SOURCE.sha256,
      },
      audioEncoder: undefined as ReturnType<typeof compare> | undefined,
      depthDecoder: undefined as { order: number[]; codes: number[]; matches: boolean } | undefined,
      codec: undefined as ReturnType<typeof compare> | undefined,
    };
    try {
      audioModel.half(root);
      const weights = await fromGguf(httpRangeSource(PROJECTOR_URL), {
        fileBytes: LFM3_Q8_0_PROJECTOR_SOURCE.bytes,
        sha256: LFM3_Q8_0_PROJECTOR_SOURCE.sha256,
      });
      await audioModel.loadStateDict(weights, { root });
      modelLoaded = true;

      const encoderUploads: Value[] = [];
      const encoded = audioModel.encodeAudio(samples, root, encoderUploads);
      evalValues([encoded], executor);
      const actualEncoder = await toArray(encoded, executor);
      await root.device.queue.onSubmittedWorkDone();
      executor.releaseValues([encoded]);
      destroyUploads(encoderUploads);
      result.audioEncoder = compare(actualEncoder, expectedEncoder);
      expect(result.audioEncoder.rmsError).toBeLessThan(0.5);
      expect(result.audioEncoder.cosineSimilarity).toBeGreaterThan(0.99);

      const hiddenUploads: Value[] = [];
      const hidden = tensor(root, expectedHidden, {
        elems: expectedHidden.length,
        dtype: 'f16',
        dims: [1, expectedHidden.length],
      });
      hiddenUploads.push(hidden);
      const sampler = createLfm3DepthSampler(1_234);
      const order = lfm3MaskgitOrder(LFM3_AUDIO_CODEBOOKS, sampler.nextRandom);
      const nativeCodes = Array<number>(LFM3_AUDIO_CODEBOOKS).fill(0);
      const revealed = Array<boolean>(LFM3_AUDIO_CODEBOOKS).fill(false);
      let start = 0;
      for (const count of REVEAL_SIZES) {
        const selected = order.slice(start, start + count);
        const depthUploads: Value[] = [];
        const logitsValues = audioModel.forwardDepthFrame(
          hidden,
          nativeCodes,
          revealed.map((value) => !value),
          root,
          depthUploads,
        );
        evalValues(logitsValues, executor);
        const logits = await Promise.all(logitsValues.map((value) => toArray(value, executor)));
        await root.device.queue.onSubmittedWorkDone();
        executor.releaseValues(logitsValues);
        destroyUploads(depthUploads);
        for (const book of selected) {
          nativeCodes[book] = argmax(logits[book]!);
          revealed[book] = true;
        }
        start += selected.length;
      }
      result.depthDecoder = {
        order,
        codes: nativeCodes,
        matches: nativeCodes.every((code, book) => code === expectedDepthCodes![book]),
      };
      expect(result.depthDecoder.matches).toBe(true);

      const codecCodes = Array.from(
        { length: LFM3_AUDIO_CODEBOOKS },
        (_, book) => (book * 17 + 31) % (LFM3_AUDIO_CODEBOOK_SIZE - 1),
      );
      const codecUploads: Value[] = [];
      const decoded = audioModel.decodeSpeechCodes(codecCodes, root, codecUploads);
      const cacheValues = audioCacheValues(decoded.cache);
      evalValues([decoded.spectrum, ...cacheValues], executor);
      const spectrum = await toArray(decoded.spectrum, executor);
      await root.device.queue.onSubmittedWorkDone();
      const actualCodecPcm = decodeLiquidCodecSpectrum(spectrum, 1);
      executor.releaseValues([decoded.spectrum, ...cacheValues]);
      destroyUploads(codecUploads);
      destroyUploads(hiddenUploads);
      result.codec = compare(actualCodecPcm, expectedCodecPcm);
      expect(result.codec.rmsError).toBeLessThan(0.01);
      expect(result.codec.maxAbsoluteError).toBeLessThan(0.05);
      console.info(`LFM3_AUDIO_OPERATOR_REFERENCE ${JSON.stringify(result)}`);
    } finally {
      await root.device.queue.onSubmittedWorkDone().catch(() => undefined);
      if (modelLoaded) audioModel.dispose();
      executor.dispose();
    }
  },
  1_200_000,
);
