/** Pinned LFM3-Omni BPE tokenizer and text-only rendering of its chat template. */

import { Tokenizer } from '@huggingface/tokenizers';
import { checkpointMismatch } from '../errors.ts';

/** Original Liquid LFM3-Omni-2B checkpoint revision. */
export const LFM3_REVISION = '21d75123f86c1f7ffb4bef1ea202710c4e7ebe67';
/** Pinned tokenizer asset from the original checkpoint. */
export const LFM3_TOKENIZER_URL = `https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/${LFM3_REVISION}/tokenizer.json`;

const EXPECTED_TOKENS = {
  '<|startoftext|>': 124_894,
  '<|im_start|>': 124_899,
  '<|im_end|>': 124_900,
  '<|text_end|>': 125_014,
} as const;

/** Text-only chat message supported by the public LFM3 generator. */
export interface Lfm3Message {
  readonly role: 'system' | 'user' | 'assistant';
  readonly content: string;
}

/** Tokenization and text chat-template operations used by LFM3 generation. */
export interface Lfm3Tokenizer {
  readonly bosId: number;
  readonly stopIds: ReadonlySet<number>;
  encodeChat(messages: readonly Lfm3Message[]): number[];
  decode(ids: readonly number[]): string;
}

const isBpeTokenizerJson = (json: unknown): json is object & { model: { type: string } } =>
  typeof json === 'object' &&
  json !== null &&
  'model' in json &&
  typeof json.model === 'object' &&
  json.model !== null &&
  'type' in json.model &&
  json.model.type === 'BPE';

/** Validates pinned special-token ids and creates LFM3 text chat tokenization. */
export function createLfm3Tokenizer(json: unknown): Lfm3Tokenizer {
  const bad = (why: string) => checkpointMismatch('LFM3 tokenizer', why);
  if (!isBpeTokenizerJson(json)) throw bad('expected the pinned BPE tokenizer.json');
  const tokenizer = new Tokenizer(json, {});
  const ids = new Map<string, number>();
  for (const [token, expected] of Object.entries(EXPECTED_TOKENS)) {
    const id = tokenizer.token_to_id(token);
    if (id === undefined) throw bad(`vocabulary has no ${token}`);
    if (id !== expected) throw bad(`${token} has id ${id}, expected ${expected}`);
    ids.set(token, id);
  }

  const encode = (text: string): number[] =>
    tokenizer.encode(text, { add_special_tokens: false }).ids;
  const stopIds = new Set([ids.get('<|im_end|>')!, ids.get('<|text_end|>')!]);

  return {
    bosId: ids.get('<|startoftext|>')!,
    stopIds,
    encodeChat(messages) {
      if (messages.length === 0) throw bad('a chat prompt needs at least one message');
      for (const message of messages) {
        if (!['system', 'user', 'assistant'].includes(message.role)) {
          throw bad(`unsupported message role '${message.role}'`);
        }
        if (typeof message.content !== 'string') throw bad('chat message content must be text');
      }
      let conversation = '<|startoftext|>';
      let remaining = [...messages];
      const first = remaining[0];
      if (first?.role === 'system') {
        if (first.content) {
          conversation += `<|im_start|>system\n${first.content}<|im_end|>\n`;
        }
        remaining = remaining.slice(1);
      }
      let lastUser = -1;
      remaining.forEach((message, index) => {
        if (message.role === 'user') lastUser = index;
      });
      for (const [index, message] of remaining.entries()) {
        conversation += `<|im_start|>${message.role}\n`;
        if (message.role === 'assistant') {
          let content = message.content;
          const keepThinking = index > lastUser;
          if (!keepThinking) {
            const close = content.lastIndexOf('</think>');
            if (close !== -1) content = content.slice(close + '</think>'.length).trim();
          }
          if (!content.includes('</think>')) content = `<think></think>${content}`;
          conversation += content;
        } else {
          conversation += message.content;
        }
        conversation += '<|im_end|>\n';
      }
      conversation += '<|im_start|>assistant\n<think></think>';
      return encode(conversation);
    },
    decode(ids) {
      return tokenizer.decode([...ids], {
        skip_special_tokens: true,
        clean_up_tokenization_spaces: false,
      });
    },
  };
}
