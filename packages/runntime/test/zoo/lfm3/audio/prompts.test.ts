import { describe, expect, it } from 'vitest';
import { createLfm3Tokenizer } from '../../../../src/zoo/lfm3/tokenizer.ts';
import {
  encodeLfm3SpeechPrompt,
  encodeLfm3TranscriptionPrompt,
} from '../../../../src/zoo/lfm3/audio/prompts.ts';

const SPECIAL_TOKENS = [
  ['<|startoftext|>', 124_894],
  ['<|im_start|>', 124_899],
  ['<|im_end|>', 124_900],
  ['<|audio|>', 125_031],
  ['<|audio_out|>', 125_032],
  ['<|audio_start|>', 125_012],
  ['<|text_end|>', 125_014],
  ['Perform ASR in English.', 500],
  ['Perform TTS in American English using a female voice.', 501],
  ['The quick brown fox jumps over the lazy dog.', 502],
  ['<think></think>', 503],
  ['system\n', 504],
  ['user\n', 505],
  ['assistant\n', 506],
] as const;

function tokenizerFixture() {
  return createLfm3Tokenizer({
    added_tokens: SPECIAL_TOKENS.map(([content, id]) => ({
      id,
      content,
      single_word: false,
      lstrip: false,
      rstrip: false,
      normalized: false,
      special: true,
    })),
    normalizer: null,
    pre_tokenizer: { type: 'Whitespace' },
    post_processor: null,
    decoder: null,
    model: {
      type: 'BPE',
      dropout: null,
      unk_token: 'unused_0',
      continuing_subword_prefix: null,
      end_of_word_suffix: null,
      fuse_unk: false,
      byte_fallback: false,
      vocab: Object.fromEntries(Array.from({ length: 32 }, (_, id) => [`unused_${id}`, id])),
      merges: [],
    },
  });
}

describe('LFM3 audio prompt framing', () => {
  it('serializes the documented ASR system instruction and audio-only user content', () => {
    const tokenizer = tokenizerFixture();

    expect(encodeLfm3TranscriptionPrompt(tokenizer)).toEqual([
      124_894, 124_899, 504, 500, 124_900, 124_899, 505, 125_031, 124_900, 124_899, 506, 503,
    ]);
  });

  it('serializes the documented TTS instruction, raw user phrase, assistant suffix, and one audio start', () => {
    const tokenizer = tokenizerFixture();

    expect(
      encodeLfm3SpeechPrompt(tokenizer, 'The quick brown fox jumps over the lazy dog.'),
    ).toEqual([
      124_894, 124_899, 504, 501, 124_900, 124_899, 505, 502, 124_900, 124_899, 506, 503, 125_012,
    ]);
  });

  it('does not append or accept duplicated or misplaced audio-start markers', () => {
    const tokenizer = tokenizerFixture();
    const alreadyFramed = {
      ...tokenizer,
      encodeChat: () => [1, 2, 125_012],
    };
    const duplicated = {
      ...tokenizer,
      encodeChat: () => [1, 2, 125_012, 3, 125_012],
    };

    expect(encodeLfm3SpeechPrompt(alreadyFramed, 'Hello.')).toEqual([1, 2, 125_012]);
    expect(() => encodeLfm3SpeechPrompt(duplicated, 'Hello.')).toThrow(
      /misplaced audio-start token/,
    );
  });
});
