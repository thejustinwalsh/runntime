/** Browser-side harness shared by the audio qualification modes: local-only loading,
 *  a network guard, the fixed sentence and voice sets and the machine-readable report line. */

import {
  createLfm3Omni,
  models,
  type Lfm3Omni,
  type Lfm3SpeechVoice,
} from '../../../../../src/zoo/index.ts';
import { gpu, liveBuffers } from '../../../../gpu.ts';

declare global {
  interface ImportMetaEnv {
    readonly VITE_LFM3_QUALIFICATION_MODE?: string;
    readonly VITE_LFM3_MODEL_URL?: string;
    readonly VITE_LFM3_PROJECTOR_URL?: string;
    readonly VITE_LFM3_TOKENIZER_URL?: string;
    readonly VITE_LFM3_AUDIO_SEEDS?: string;
    readonly VITE_LFM3_AUDIO_VOICES?: string;
    readonly VITE_LFM3_GUARDED_TEST?: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

export const MODE = import.meta.env.VITE_LFM3_QUALIFICATION_MODE;

/** Fixed sentences: plain words only. No digits (ASR writes "9" for "nine") and no dialect spellings
 *  ("harbor" and "harbour" are one word to a listener), so the rate counts recognition errors only. */
export const SENTENCES = [
  'The quick brown fox jumps over the lazy dog.',
  'Please turn left at the next street.',
  'She sells seashells by the seashore.',
  'The meeting has been moved to Thursday afternoon.',
  'A gentle breeze carried the scent of rain across the quiet valley.',
  'Remember to save your work before closing the window.',
  'The library opens early and closes late in the evening.',
  'Our train to the coast leaves from the old station.',
] as const;

/** The four voices the product exposes: a language and a gender in the TTS system prompt. */
export const VOICES: readonly { readonly id: string; readonly voice: Lfm3SpeechVoice }[] = [
  { id: 'american-female', voice: { language: 'American English', gender: 'female' } },
  { id: 'american-male', voice: { language: 'American English', gender: 'male' } },
  { id: 'british-female', voice: { language: 'British English', gender: 'female' } },
  { id: 'british-male', voice: { language: 'British English', gender: 'male' } },
];

/** The voices to run: all four unless VITE_LFM3_AUDIO_VOICES lists ids, so a long run can be sharded. */
export function selectedVoices(): typeof VOICES {
  const raw = import.meta.env.VITE_LFM3_AUDIO_VOICES;
  if (!raw) return VOICES;
  return raw.split(',').map((id) => {
    const found = VOICES.find((voice) => voice.id === id);
    if (!found) throw new Error(`unknown voice ${id}`);
    return found;
  });
}

/** Seeds for the round trip; the first is the canonical one the thresholds are stated for. */
export function seeds(): number[] {
  const raw = import.meta.env.VITE_LFM3_AUDIO_SEEDS ?? '1234';
  return raw.split(',').map((seed) => {
    const value = Number(seed);
    if (!Number.isSafeInteger(value) || value < 0) throw new Error(`invalid seed ${seed}`);
    return value;
  });
}

function requiredEnv(name: string, value: string | undefined): string {
  if (!value) throw new Error(`${name} is required: run the qualification through test:audio`);
  return value;
}

/** Makes any request that leaves this origin fail the run, and records the ones that do not. */
export function installNetworkGuard(): { readonly requests: string[]; readonly blocked: string[] } {
  const requests: string[] = [];
  const blocked: string[] = [];
  const check = (input: string | URL): string => {
    const url = new URL(input, location.href);
    if (url.origin !== location.origin) {
      blocked.push(url.href);
      throw new Error(`audio qualification is local only; refused ${url.href}`);
    }
    requests.push(url.pathname);
    return url.href;
  };
  const nativeFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = (input, init) =>
    nativeFetch(check(input instanceof Request ? input.url : input), init);
  const nativeOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (method: string, url: string | URL, ...rest: unknown[]) {
    return Reflect.apply(nativeOpen, this, [method, check(url), ...rest]);
  };
  return { requests, blocked };
}

export const gpuBufferBytes = (): number =>
  [...liveBuffers()].reduce((sum, buffer) => sum + buffer.size, 0);

export interface LoadedAudio {
  readonly audio: Lfm3Omni;
  readonly loadMs: number;
  readonly sourceBytes: number;
  readonly loadedGpuBytes: number;
  readonly guard: ReturnType<typeof installNetworkGuard>;
  readonly before: ReadonlySet<GPUBuffer>;
}

/** Loads the pinned Q4_K_M and Q8_0 pair from the local weight URLs, with the network guard on. */
export async function loadAudio(): Promise<LoadedAudio> {
  const guard = installNetworkGuard();
  await gpu();
  const before = liveBuffers();
  let sourceBytes = 0;
  const started = performance.now();
  const audio = await createLfm3Omni(
    {
      ...models.lfm3.LFM3_OMNI_2B.Q4_K_M,
      modelPath: requiredEnv('VITE_LFM3_MODEL_URL', import.meta.env.VITE_LFM3_MODEL_URL),
      projectorPath: requiredEnv(
        'VITE_LFM3_PROJECTOR_URL',
        import.meta.env.VITE_LFM3_PROJECTOR_URL,
      ),
      tokenizerPath: requiredEnv(
        'VITE_LFM3_TOKENIZER_URL',
        import.meta.env.VITE_LFM3_TOKENIZER_URL,
      ),
    },
    {
      onBytes(bytes) {
        sourceBytes += bytes;
      },
    },
  );
  return {
    audio,
    loadMs: performance.now() - started,
    sourceBytes,
    loadedGpuBytes: gpuBufferBytes(),
    guard,
    before,
  };
}

/** Adapter facts for the report. */
export async function adapterInfo(): Promise<Record<string, unknown>> {
  const { root } = await gpu();
  const info = root.device.adapterInfo;
  return {
    label: root.device.label,
    vendor: info?.vendor,
    architecture: info?.architecture,
    device: info?.device,
    description: info?.description,
    features: [...root.device.features].sort(),
    userAgent: navigator.userAgent,
  };
}

/** One JSON line the qualification script collects from the console. */
export function report(name: string, payload: Record<string, unknown>): void {
  console.info(`LFM3_AUDIO_QUALIFICATION ${JSON.stringify({ name, ...payload })}`);
}

export const round = (value: number, digits = 4): number => +value.toFixed(digits);
