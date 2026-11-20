import { describe, expect, it } from 'vitest';
import { createD1Tokenizer } from '../../../src/zoo/d1/tokenizer.ts';

const specialTokens = {
  '<|startoftext|>': 1,
  '<|mask|>': 16,
  '<|reserved_7|>': 17,
  '<|reserved_8|>': 18,
  '<|reserved_9|>': 19,
  '<|reserved_10|>': 20,
  '<|reserved_11|>': 21,
};

const vocabulary = Object.fromEntries(
  Array.from({ length: 22 }, (_, id) => [
    Object.entries(specialTokens).find(([, tokenId]) => tokenId === id)?.[0] ?? `unused_${id}`,
    id,
  ]),
);
const pieces = Object.fromEntries(
  [...new Set('heloworld')].map((piece, index) => [piece, index + 22]),
);

const json = {
  added_tokens: Object.entries(specialTokens).map(([content, id]) => ({
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
    vocab: { ...vocabulary, ...pieces },
    merges: [],
  },
};

describe('createD1Tokenizer', () => {
  it('uses the pinned marker ids and BPE encoding without adding special tokens', () => {
    const tokenizer = createD1Tokenizer(json);

    expect(tokenizer.bosId).toBe(1);
    expect(tokenizer.tokenId('<|mask|>')).toBe(16);
    expect(tokenizer.tokenId('<|reserved_11|>')).toBe(21);
    expect(tokenizer.encode('hello world')).toEqual([22, 23, 24, 24, 25, 26, 25, 27, 24, 28]);
  });

  it('rejects another tokenizer type and mismatched D1 marker ids', () => {
    expect(() =>
      createD1Tokenizer({ ...json, model: { ...json.model, type: 'WordPiece' } }),
    ).toThrow(expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }));
    expect(() =>
      createD1Tokenizer({
        ...json,
        added_tokens: json.added_tokens.map((token) =>
          token.content === '<|mask|>' ? { ...token, id: 15 } : token,
        ),
      }),
    ).toThrow(expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }));
  });
});
