/** Generates teacher-forced vectors and deterministic answers with Liquid's pinned engine. */

import { expect, it } from 'vitest';

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

interface LiquidReferenceModel {
  readonly model: { readonly language_model: LiquidLanguageModel };
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
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const MODE = import.meta.env.VITE_LFM3_QUALIFICATION_MODE;
const ENGINE_URL = import.meta.env.VITE_LFM3_LIQUID_ENGINE_URL;
const ENGINE_REVISION = '9c2fb176f4c222da394e8037d3c0d0837f70b978';
const ENGINE_SHA256 = '77821949412f5dd6d60e37f22415f0e71454e65f6a73928529c6fbec0f2cf2cb';
const GGUF_REVISION = '3d29af2ad166f8299c2d820937e47dc0add20ffc';
const GGUF_URL = `https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/${GGUF_REVISION}/LFM3-Omni-2B-Q4_K_M.gguf`;
const PROJECTOR_URL = `https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/${GGUF_REVISION}/mmproj-LFM3-Omni-2B-Q8_0.gguf`;
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
    expect(fixtures).toHaveLength(CASES.length);
  },
  1_200_000,
);
