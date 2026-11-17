import type { Lfm3Message, Lfm3Tokenizer } from '../tokenizer.ts';

export const LFM3_ASR_SYSTEM_PROMPT = 'Perform ASR in English.';
export const LFM3_TTS_SYSTEM_PROMPT = 'Perform TTS in American English using a female voice.';
export const LFM3_AUDIO_PLACEHOLDER = '<|audio|>';

export function lfm3TranscriptionMessages(): readonly Lfm3Message[] {
  return [
    { role: 'system', content: LFM3_ASR_SYSTEM_PROMPT },
    { role: 'user', content: LFM3_AUDIO_PLACEHOLDER },
  ];
}

/** Typed voice selection for LFM3 text-to-speech. Omitted fields use the default voice. */
export interface Lfm3SpeechVoice {
  /** Spoken language or dialect as plain English, for example "British English". Defaults to "American English". */
  readonly language?: string;
  /** Voice gender. Defaults to "female". */
  readonly gender?: 'female' | 'male';
}

/** Selection of the TTS system instruction: an explicit prompt wins over a typed voice. */
export interface Lfm3SpeechPromptOptions {
  readonly voice?: Lfm3SpeechVoice;
  readonly systemPrompt?: string;
}

/** Builds the TTS system instruction, e.g. "Perform TTS in British English using a female voice." */
export function lfm3SpeechSystemPrompt(options: Lfm3SpeechPromptOptions = {}): string {
  if (options.systemPrompt !== undefined) return options.systemPrompt;
  const { voice } = options;
  if (!voice || (voice.language === undefined && voice.gender === undefined)) {
    return LFM3_TTS_SYSTEM_PROMPT;
  }
  return `Perform TTS in ${voice.language ?? 'American English'} using a ${voice.gender ?? 'female'} voice.`;
}

export function lfm3SpeechMessages(
  text: string,
  options: Lfm3SpeechPromptOptions = {},
): readonly Lfm3Message[] {
  return [
    { role: 'system', content: lfm3SpeechSystemPrompt(options) },
    { role: 'user', content: text },
  ];
}

export function encodeLfm3TranscriptionPrompt(tokenizer: Lfm3Tokenizer): number[] {
  return tokenizer.encodeChat(lfm3TranscriptionMessages());
}

export function encodeLfm3SpeechPrompt(
  tokenizer: Lfm3Tokenizer,
  text: string,
  options: Lfm3SpeechPromptOptions = {},
): number[] {
  const prompt = tokenizer.encodeChat(lfm3SpeechMessages(text, options));
  const audioStartPositions = prompt.flatMap((token, index) =>
    token === tokenizer.audioStartId ? [index] : [],
  );
  if (audioStartPositions.length === 1 && audioStartPositions[0] === prompt.length - 1) {
    return prompt;
  }
  if (audioStartPositions.length > 0) {
    throw new Error('LFM3 speech prompt contains a misplaced audio-start token');
  }
  return [...prompt, tokenizer.audioStartId];
}
