/** Generates teacher-forced vectors and deterministic answers with Liquid's pinned engine. */

import { expect, it } from 'vitest';
import { resampleAudio } from '../../../src/zoo/index.ts';

interface Message {
  readonly role: 'user';
  readonly content: string;
}

interface LiquidCache {
  get_seq_length(): number;
  advance(count: number): void;
}

interface LiquidGenerationState {
  readonly cache: LiquidCache;
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
  streamTokenIds(input: {
    input_ids: number[][];
    max_new_tokens: number;
    do_sample: boolean;
  }): AsyncIterable<number>;
}

interface LiquidAudioStreamOptions {
  readonly input_ids: readonly number[][];
  readonly audio?: readonly Float32Array[];
  readonly generation_mode?: 'audio';
  readonly max_new_tokens: number;
  readonly do_sample: boolean;
  readonly seed?: number;
  readonly depth_decoder_do_sample?: boolean;
  readonly depth_decoder_temperature?: number;
  readonly on_speech?: LiquidSpeechCallback;
}

interface LiquidSpeechCallback {
  readonly audio: (audio: Float32Array) => void;
  frame?(codes: Int32Array): void;
}

interface LiquidAudioModel {
  readonly tokens: { readonly audio: number };
  planAudio(samples: Float32Array): { readonly tokens: number; readonly tokenId: number };
  streamTokenIds(options: LiquidAudioStreamOptions): AsyncIterable<number>;
}

interface LiquidReferenceModel {
  readonly model: {
    readonly language_model: LiquidLanguageModel;
    readonly tokens: { readonly audio: number };
    planAudio: LiquidAudioModel['planAudio'];
    streamTokenIds: LiquidAudioModel['streamTokenIds'];
  };
  readonly tokenizer: {
    encode(text: string, options?: { add_special_tokens?: boolean }): { ids: number[] };
    decode(
      ids: readonly number[],
      options?: { skip_special_tokens?: boolean; clean_up_tokenization_spaces?: boolean },
    ): string;
  };
  encodePrompt(messages: readonly Message[]): number[];
}

interface LiquidModel {
  readonly reference: LiquidReferenceModel;
  readonly audio_sampling_rate: number;
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
    readonly VITE_LFM3_LIQUID_REFERENCE_CAPTURE?: string;
    readonly VITE_LFM3_LIQUID_MODEL_URL?: string;
    readonly VITE_LFM3_LIQUID_PROJECTOR_URL?: string;
    readonly VITE_LFM3_LIQUID_AUDIO_INPUT_BASE_URL?: string;
    readonly VITE_LFM3_LIQUID_AUDIO_CAPTURE?: string;
    readonly VITE_LFM3_NATIVE_SPEECH_URL?: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const MODE = import.meta.env.VITE_LFM3_QUALIFICATION_MODE;
const ENGINE_URL = import.meta.env.VITE_LFM3_LIQUID_ENGINE_URL;
const LIQUID_MODEL_URL = import.meta.env.VITE_LFM3_LIQUID_MODEL_URL;
const LIQUID_PROJECTOR_URL = import.meta.env.VITE_LFM3_LIQUID_PROJECTOR_URL;
const AUDIO_INPUT_BASE_URL = import.meta.env.VITE_LFM3_LIQUID_AUDIO_INPUT_BASE_URL;
const AUDIO_CAPTURE = import.meta.env.VITE_LFM3_LIQUID_AUDIO_CAPTURE === '1';
const NATIVE_SPEECH_URL = import.meta.env.VITE_LFM3_NATIVE_SPEECH_URL;
const ENGINE_REVISION = '9c2fb176f4c222da394e8037d3c0d0837f70b978';
const ENGINE_SHA256 = '77821949412f5dd6d60e37f22415f0e71454e65f6a73928529c6fbec0f2cf2cb';
const GGUF_REVISION = '3d29af2ad166f8299c2d820937e47dc0add20ffc';
const GGUF_URL =
  LIQUID_MODEL_URL ??
  `https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/${GGUF_REVISION}/LFM3-Omni-2B-Q4_K_M.gguf`;
const PROJECTOR_URL =
  LIQUID_PROJECTOR_URL ??
  `https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/${GGUF_REVISION}/mmproj-LFM3-Omni-2B-Q8_0.gguf`;
const CASES = [
  {
    id: 'arithmetic',
    messages: [{ role: 'user', content: 'What is 6 times 7? Reply with only the number.' }],
    answer: '42',
  },
  {
    id: 'capital',
    messages: [
      { role: 'user', content: 'What is the capital of France? Reply with only the city.' },
    ],
    answer: 'Paris',
  },
  {
    id: 'color',
    messages: [
      { role: 'user', content: 'What color is a clear daytime sky? Reply with one word.' },
    ],
    answer: 'Blue',
  },
] satisfies readonly { id: string; messages: readonly Message[]; answer: string }[];

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

interface AudioReferenceCase {
  readonly id: string;
  readonly file: string;
  readonly transcript: string;
}

interface AudioReferenceResult {
  readonly id: string;
  readonly input: {
    readonly file: string;
    readonly transcript: string;
    readonly sampleRate: number;
    readonly samples: number;
    readonly bytes: number;
    readonly sha256: string;
  };
  readonly liquidTranscription: {
    readonly text: string;
    readonly normalizedText: string;
    readonly promptTokens: number;
    readonly generatedTokens: number;
    readonly matchesExpected: boolean;
  };
  readonly liquidSpeech: {
    readonly file: string;
    readonly sampleRate: number;
    readonly samples: number;
    readonly seconds: number;
    readonly bytes: number;
    readonly sha256: string;
    readonly seed: number;
    readonly depthDecoderTemperature: number;
    readonly speechFrames: number;
    readonly streamChunks: number;
    readonly generatedTokens: number;
    readonly finiteNonempty: boolean;
  };
  readonly speechTranscription: {
    readonly text: string;
    readonly normalizedText: string;
    readonly sampleRate: number;
    readonly samples: number;
    readonly promptTokens: number;
    readonly generatedTokens: number;
    readonly recoversRequestedText: boolean;
  };
}

const AUDIO_CASES: readonly AudioReferenceCase[] = [
  {
    id: 'quick-brown-fox',
    file: 'quick-brown-fox.wav',
    transcript: 'The quick brown fox jumps over the lazy dog.',
  },
  {
    id: 'turn-left-next-street',
    file: 'turn-left-next-street.wav',
    transcript: 'Please turn left at the next street.',
  },
];

function decodePcm16Wav(
  data: ArrayBuffer,
  expectedSampleRate = 16_000,
): { samples: Float32Array; sampleRate: number } {
  const bytes = new Uint8Array(data);
  const view = new DataView(data);
  if (bytes.length < 44 || new TextDecoder().decode(bytes.subarray(0, 4)) !== 'RIFF') {
    throw new Error('audio fixture is not a RIFF WAV');
  }
  let sampleRate = 0;
  let channels = 0;
  let format = 0;
  let bits = 0;
  let dataOffset = -1;
  let dataBytes = 0;
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
      dataBytes = size;
    }
    offset = body + size + (size % 2);
  }
  if (format !== 1 || channels !== 1 || bits !== 16 || sampleRate !== expectedSampleRate) {
    throw new Error(
      `WAV must be mono PCM16 at ${expectedSampleRate} Hz; got format=${format}, channels=${channels}, bits=${bits}, rate=${sampleRate}`,
    );
  }
  if (dataOffset < 0 || dataBytes === 0 || dataBytes % 2 !== 0) {
    throw new Error('WAV has no whole PCM16 samples');
  }
  const samples = new Float32Array(dataBytes / 2);
  for (let index = 0; index < samples.length; index++) {
    samples[index] = view.getInt16(dataOffset + index * 2, true) / 32_768;
  }
  return { samples, sampleRate };
}

function normalizeWords(text: string): string {
  return text
    .toLocaleLowerCase('en-US')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim()
    .replace(/\s+/g, ' ');
}

async function sha256(data: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, '0')).join(
    '',
  );
}

function encodeFloat32Wav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 4);
  const view = new DataView(buffer);
  const writeTag = (offset: number, value: string): void => {
    for (let index = 0; index < value.length; index++)
      view.setUint8(offset + index, value.charCodeAt(index));
  };
  writeTag(0, 'RIFF');
  view.setUint32(4, buffer.byteLength - 8, true);
  writeTag(8, 'WAVE');
  writeTag(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 32, true);
  writeTag(36, 'data');
  view.setUint32(40, samples.length * 4, true);
  for (let index = 0; index < samples.length; index++) {
    view.setFloat32(44 + index * 4, samples[index]!, true);
  }
  return buffer;
}

function audioPromptIds(reference: LiquidReferenceModel, placeholderCount: number): number[] {
  const placeholders = '<|audio|>'.repeat(placeholderCount);
  const prompt = `<|startoftext|><|im_start|>user\nTranscribe this audio: ${placeholders}<|im_end|>\n<|im_start|>assistant\n<think></think>`;
  const ids = reference.tokenizer.encode(prompt, { add_special_tokens: false }).ids;
  const audioId = reference.model.tokens.audio;
  const audioPositions = ids.flatMap((id, index) => (id === audioId ? [index] : []));
  if (audioPositions.length !== placeholderCount) {
    throw new Error(
      `Liquid tokenizer produced ${audioPositions.length} audio placeholders, expected ${placeholderCount}`,
    );
  }
  return ids;
}

async function streamLiquidTokens(
  model: LiquidAudioModel,
  reference: LiquidReferenceModel,
  input: {
    readonly samples?: Float32Array;
    readonly prompt: string;
    readonly maxNewTokens: number;
    readonly seed?: number;
    readonly onSpeech?: LiquidSpeechCallback;
  },
): Promise<{ tokenIds: number[]; inputTokens: number }> {
  const audio = input.samples;
  let ids: number[];
  let audioInputs: readonly Float32Array[] | undefined;
  if (audio) {
    const plan = reference.model.planAudio(audio);
    ids = audioPromptIds(reference, plan.tokens);
    if (plan.tokenId !== reference.model.tokens.audio) {
      throw new Error('Liquid audio token ID does not match planAudio');
    }
    audioInputs = [audio];
  } else {
    ids = reference.encodePrompt([{ role: 'user', content: input.prompt }]);
  }
  const tokenIds: number[] = [];
  for await (const token of model.streamTokenIds({
    input_ids: [ids],
    ...(audioInputs ? { audio: audioInputs } : {}),
    ...(input.onSpeech ? { generation_mode: 'audio' as const, on_speech: input.onSpeech } : {}),
    max_new_tokens: input.maxNewTokens,
    do_sample: false,
    ...(input.seed === undefined
      ? {}
      : { seed: input.seed, depth_decoder_do_sample: true, depth_decoder_temperature: 0.4 }),
  })) {
    tokenIds.push(token);
  }
  return { tokenIds, inputTokens: ids.length };
}

async function captureAudioReferences(
  reference: LiquidReferenceModel,
  speechSampleRate: number,
): Promise<AudioReferenceResult[]> {
  if (!AUDIO_INPUT_BASE_URL) throw new Error('the local frozen audio fixture URL was not provided');
  const audioModel: LiquidAudioModel = reference.model;
  const results: AudioReferenceResult[] = [];
  for (const testCase of AUDIO_CASES) {
    const response = await fetch(`${AUDIO_INPUT_BASE_URL.replace(/\/$/, '')}/${testCase.file}`);
    if (!response.ok) throw new Error(`failed to load ${testCase.file}: HTTP ${response.status}`);
    const inputBytes = await response.arrayBuffer();
    const inputDigest = await sha256(inputBytes);
    const { samples, sampleRate } = decodePcm16Wav(inputBytes);
    const transcription = await streamLiquidTokens(audioModel, reference, {
      samples,
      prompt: '',
      maxNewTokens: 128,
    });
    const transcript = reference.tokenizer
      .decode(transcription.tokenIds, {
        skip_special_tokens: true,
        clean_up_tokenization_spaces: false,
      })
      .trim();
    const audioChunks: Float32Array[] = [];
    let speechFrames = 0;
    const onSpeech: LiquidSpeechCallback = {
      audio: (chunk) => audioChunks.push(chunk),
      frame: () => speechFrames++,
    };
    const speech = await streamLiquidTokens(audioModel, reference, {
      prompt: `Say exactly: ${testCase.transcript}`,
      maxNewTokens: 256,
      seed: 1234,
      onSpeech,
    });
    const totalSamples = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const speechSamples = new Float32Array(totalSamples);
    let cursor = 0;
    for (const chunk of audioChunks) {
      speechSamples.set(chunk, cursor);
      cursor += chunk.length;
    }
    if (speechSampleRate !== 24_000) {
      throw new Error(`Liquid speech sample rate is ${speechSampleRate}, expected 24000`);
    }
    if (speechSamples.length === 0 || !speechSamples.every(Number.isFinite)) {
      throw new Error(`Liquid speech for ${testCase.id} is empty or non-finite`);
    }
    const speechBytes = encodeFloat32Wav(speechSamples, speechSampleRate);
    const speechDigest = await sha256(speechBytes);
    const speechFilename = `liquid-speech-${testCase.id}.wav`;
    const saved = await fetch(`/__liquid-audio-output/${speechFilename}`, {
      method: 'POST',
      headers: { 'content-type': 'audio/wav' },
      body: speechBytes,
    });
    if (saved.status !== 204)
      throw new Error(`failed to save Liquid speech fixture: HTTP ${saved.status}`);

    const speechTranscriptionSamples = await resampleAudio(speechSamples, speechSampleRate);
    const speechTranscription = await streamLiquidTokens(audioModel, reference, {
      samples: speechTranscriptionSamples,
      prompt: '',
      maxNewTokens: 128,
    });
    const speechTranscript = reference.tokenizer
      .decode(speechTranscription.tokenIds, {
        skip_special_tokens: true,
        clean_up_tokenization_spaces: false,
      })
      .trim();
    results.push({
      id: testCase.id,
      input: {
        file: testCase.file,
        transcript: testCase.transcript,
        sampleRate,
        samples: samples.length,
        bytes: inputBytes.byteLength,
        sha256: inputDigest,
      },
      liquidTranscription: {
        text: transcript,
        normalizedText: normalizeWords(transcript),
        promptTokens: transcription.inputTokens,
        generatedTokens: transcription.tokenIds.length,
        matchesExpected: normalizeWords(transcript) === normalizeWords(testCase.transcript),
      },
      liquidSpeech: {
        file: speechFilename,
        sampleRate: speechSampleRate,
        samples: speechSamples.length,
        seconds: speechSamples.length / speechSampleRate,
        bytes: speechBytes.byteLength,
        sha256: speechDigest,
        seed: 1234,
        depthDecoderTemperature: 0.4,
        speechFrames,
        streamChunks: audioChunks.length,
        generatedTokens: speech.tokenIds.length,
        finiteNonempty: true,
      },
      speechTranscription: {
        text: speechTranscript,
        normalizedText: normalizeWords(speechTranscript),
        sampleRate: 16_000,
        samples: speechTranscriptionSamples.length,
        promptTokens: speechTranscription.inputTokens,
        generatedTokens: speechTranscription.tokenIds.length,
        recoversRequestedText:
          normalizeWords(speechTranscript) === normalizeWords(testCase.transcript),
      },
    });
  }
  return results;
}

async function captureNativeSpeechReference(
  audioModel: LiquidAudioModel,
  reference: LiquidReferenceModel,
) {
  if (!NATIVE_SPEECH_URL) throw new Error('the native speech WAV URL was not provided');
  const response = await fetch(NATIVE_SPEECH_URL);
  if (!response.ok) throw new Error(`failed to load native speech WAV: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  const { samples, sampleRate } = decodePcm16Wav(bytes, 24_000);
  const transcriptionInput = await resampleAudio(samples, sampleRate);
  const transcription = await streamLiquidTokens(audioModel, reference, {
    samples: transcriptionInput,
    prompt: '',
    maxNewTokens: 128,
  });
  const text = reference.tokenizer
    .decode(transcription.tokenIds, {
      skip_special_tokens: true,
      clean_up_tokenization_spaces: false,
    })
    .trim();
  return {
    sampleRate,
    samples: samples.length,
    bytes: bytes.byteLength,
    sha256: await sha256(bytes),
    transcriptionSampleRate: 16_000,
    transcriptionSamples: transcriptionInput.length,
    requestedText: AUDIO_CASES[0]!.transcript,
    liquidTranscription: {
      text,
      normalizedText: normalizeWords(text),
      matchesRequestedText: normalizeWords(text) === normalizeWords(AUDIO_CASES[0]!.transcript),
      promptTokens: transcription.inputTokens,
      generatedTokens: transcription.tokenIds.length,
    },
  };
}

async function teacherForcedVectors(
  reference: LiquidReferenceModel,
  inputIds: readonly number[],
): Promise<{
  inputTokenIds: number[];
  vectors: { tokenId: number; predictedTokenId: number; values: number[] }[];
}> {
  const languageModel = reference.model.language_model;
  const captureCount = Math.min(inputIds.length, 4);
  const captureStart = inputIds.length - captureCount;
  const state = languageModel.createGenerationState({ maxLength: inputIds.length });
  const cache = state.cache;
  const vectors: { tokenId: number; predictedTokenId: number; values: number[] }[] = [];
  try {
    await languageModel.prefillTokens(cache, Uint32Array.from(inputIds.slice(0, captureStart)));
    for (let index = captureStart; index < inputIds.length; index++) {
      const step = await languageModel.submitDecodeStep(cache, cache.get_seq_length(), {
        token: inputIds[index]!,
        predict: true,
      });
      const [predictedTokenId, hidden] = await Promise.all([
        step.next,
        languageModel.runtime.readTensor(step.hidden),
      ]);
      vectors.push({
        tokenId: inputIds[index]!,
        predictedTokenId,
        values: Array.from(hidden),
      });
      cache.advance(1);
    }
  } finally {
    languageModel.disposeGenerationState(state);
  }
  return { inputTokenIds: [...inputIds], vectors };
}

it.skipIf(MODE !== 'liquid-reference')(
  'records pinned Liquid Q4_K_M teacher-forced vectors and deterministic chat answers',
  async (ctx) => {
    if (!ENGINE_URL) throw new Error('the pinned Liquid engine URL was not provided');
    const engineModule: unknown = await import(/* @vite-ignore */ ENGINE_URL);
    if (!isLiquidEngineModule(engineModule)) throw new Error('Liquid engine exports are invalid');

    const model = await engineModule.Lfm3OmniForConditionalGeneration.from_pretrained(GGUF_URL, {
      mmproj: PROJECTOR_URL,
    });
    ctx.onTestFinished(() => model.dispose());
    const reference = model.reference;
    const languageModel = reference.model.language_model;
    const fixtures = [];
    for (const testCase of CASES) {
      const promptTokenIds = reference.encodePrompt(testCase.messages);
      const generatedTokenIds: number[] = [];
      for await (const token of languageModel.streamTokenIds({
        input_ids: [promptTokenIds],
        max_new_tokens: 32,
        do_sample: false,
      })) {
        generatedTokenIds.push(token);
      }
      const answer = reference.tokenizer.decode(generatedTokenIds, {
        skip_special_tokens: true,
        clean_up_tokenization_spaces: false,
      });
      const teacherForced = await teacherForcedVectors(reference, [
        ...promptTokenIds,
        ...generatedTokenIds.slice(0, 3),
      ]);
      expect(answer.trim()).toBe(testCase.answer);
      expect(teacherForced.vectors.length).toBeGreaterThanOrEqual(2);
      expect(teacherForced.vectors.every((vector) => vector.values.length === 2_048)).toBe(true);
      fixtures.push({
        id: testCase.id,
        messages: testCase.messages,
        promptTokenIds,
        generatedTokenIds,
        answer,
        ...teacherForced,
      });
    }
    const audioFixtures = AUDIO_CAPTURE
      ? await captureAudioReferences(reference, model.audio_sampling_rate)
      : undefined;
    const nativeSpeech =
      AUDIO_CAPTURE && NATIVE_SPEECH_URL
        ? await captureNativeSpeechReference(reference.model, reference)
        : undefined;
    const report = {
      schemaVersion: 1,
      engine: { revision: ENGINE_REVISION, sha256: ENGINE_SHA256 },
      model: {
        repository: 'LiquidAI/LFM3-Omni-2B-GGUF',
        revision: GGUF_REVISION,
        file: 'LFM3-Omni-2B-Q4_K_M.gguf',
        sha256: '9f8d30e9d4fd7a9349d586e077988229b3ca0a0e4ad730897563d1eccc9893e2',
      },
      projector: {
        file: 'mmproj-LFM3-Omni-2B-Q8_0.gguf',
        bytes: 506_842_528,
        sha256: '8a25f9d1931244429de404fccacfb7fb933970a36aa5688d2f5069f29c6b27b1',
      },
      fixtures,
      ...(audioFixtures ? { audioFixtures } : {}),
      ...(nativeSpeech ? { nativeSpeech } : {}),
    };
    const encoded = JSON.stringify(report);
    if (import.meta.env.VITE_LFM3_LIQUID_REFERENCE_CAPTURE === '1') {
      const response = await fetch('/__liquid-reference-fixture', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: encoded,
      });
      expect(response.status).toBe(204);
    }
    if (nativeSpeech) {
      console.info(`LFM3_LIQUID_NATIVE_SPEECH_REFERENCE ${JSON.stringify(nativeSpeech)}`);
    }
    if (audioFixtures) {
      console.info(`LFM3_LIQUID_AUDIO_REFERENCE ${JSON.stringify(audioFixtures)}`);
      expect(audioFixtures).toHaveLength(AUDIO_CASES.length);
      for (const fixture of audioFixtures) {
        expect(fixture.input.sampleRate).toBe(16_000);
        expect(fixture.liquidTranscription.matchesExpected).toBe(true);
        expect(fixture.liquidSpeech.sampleRate).toBe(24_000);
        expect(fixture.liquidSpeech.finiteNonempty).toBe(true);
        expect(fixture.liquidSpeech.speechFrames).toBeGreaterThan(0);
        expect(fixture.speechTranscription.sampleRate).toBe(16_000);
      }
      expect(
        audioFixtures.every((fixture) => fixture.speechTranscription.recoversRequestedText),
        'pinned Liquid speech should recover the requested words through same-file transcription',
      ).toBe(true);
    }
    if (nativeSpeech) {
      expect(nativeSpeech.sampleRate).toBe(24_000);
      expect(nativeSpeech.liquidTranscription.matchesRequestedText).toBe(true);
    }
    expect(fixtures).toHaveLength(CASES.length);
  },
  1_200_000,
);
