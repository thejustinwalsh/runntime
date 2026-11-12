import { describe, expect, it } from 'vitest';
import { createLfm3Tokenizer } from '../../../src/zoo/lfm3/tokenizer.ts';

const specialTokens = [
  ['<|startoftext|>', 124_894],
  ['<|im_start|>', 124_899],
  ['<|im_end|>', 124_900],
  ['<|text_end|>', 125_014],
] as const;

const json = {
  added_tokens: specialTokens.map(([content, id]) => ({
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
};

describe('LFM3 tokenizer and chat template', () => {
  it('pins the original special ids and renders the empty-think generation prefix', () => {
    const tokenizer = createLfm3Tokenizer(json);
    const ids = tokenizer.encodeChat([
      { role: 'system', content: 'Be concise.' },
      { role: 'user', content: 'Hello.' },
    ]);
    expect(tokenizer.bosId).toBe(124_894);
    expect(ids[0]).toBe(124_894);
    expect(ids.filter((id) => id === 124_899)).toHaveLength(3);
    expect(ids.filter((id) => id === 124_900)).toHaveLength(2);
    expect(tokenizer.stopIds).toEqual(new Set([124_900, 125_014]));
  });

  it('rejects a tokenizer whose pinned stop token has moved', () => {
    expect(() =>
      createLfm3Tokenizer({
        ...json,
        added_tokens: json.added_tokens.map((token) =>
          token.content === '<|text_end|>' ? { ...token, id: 125_013 } : token,
        ),
      }),
    ).toThrow(expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }));
  });
});
