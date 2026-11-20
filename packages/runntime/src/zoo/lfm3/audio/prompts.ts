/** Chat prompts and their token ids for LFM3-Omni transcription and speech. */

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

export function lfm3SpeechMessages(text: string): readonly Lfm3Message[] {
  return [
    { role: 'system', content: LFM3_TTS_SYSTEM_PROMPT },
    { role: 'user', content: text },
  ];
}

export function encodeLfm3TranscriptionPrompt(tokenizer: Lfm3Tokenizer): number[] {
  return tokenizer.encodeChat(lfm3TranscriptionMessages());
}

export function encodeLfm3SpeechPrompt(tokenizer: Lfm3Tokenizer, text: string): number[] {
  const prompt = tokenizer.encodeChat(lfm3SpeechMessages(text));
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
