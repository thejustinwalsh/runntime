/** D1's BPE tokenizer, with the model's fixed decision markers. */

import { Tokenizer } from '@huggingface/tokenizers';
import { checkpointMismatch } from '../errors.ts';
import type { D1Tokenizer } from './prompt.ts';

const SPECIAL_TOKENS = {
  '<|startoftext|>': 1,
  '<|mask|>': 16,
  '<|reserved_7|>': 17,
  '<|reserved_8|>': 18,
  '<|reserved_9|>': 19,
  '<|reserved_10|>': 20,
  '<|reserved_11|>': 21,
} as const;

const isBpeTokenizerJson = (json: unknown): json is object =>
  typeof json === 'object' &&
  json !== null &&
  'model' in json &&
  typeof json.model === 'object' &&
  json.model !== null &&
  'type' in json.model &&
  json.model.type === 'BPE';

/** Builds the D1 tokenizer from its pinned Hugging Face tokenizer.json. */
export function createD1Tokenizer(json: unknown): D1Tokenizer {
  const bad = (why: string) => checkpointMismatch('d1 tokenizer', why);
  if (!isBpeTokenizerJson(json)) throw bad('expected a BPE tokenizer.json');

  const tokenizer = new Tokenizer(json, {});
  const ids = new Map<string, number>();
  for (const [token, expected] of Object.entries(SPECIAL_TOKENS)) {
    const id = tokenizer.token_to_id(token);
    if (id === undefined) throw bad(`vocab has no ${token}`);
    if (id !== expected) throw bad(`${token} has id ${id}, expected ${expected}`);
    ids.set(token, id);
  }

  return {
    bosId: ids.get('<|startoftext|>')!,
    encode: (text) => tokenizer.encode(text, { add_special_tokens: false }).ids,
    tokenId(token) {
      const id = ids.get(token);
      if (id === undefined) throw bad(`unknown special token ${token}`);
      return id;
    },
  };
}
