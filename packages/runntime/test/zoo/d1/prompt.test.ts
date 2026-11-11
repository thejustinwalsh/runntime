import { describe, expect, it } from 'vitest';
import { encodeQuestion, type D1Tokenizer } from '../../../src/zoo/d1/prompt.ts';

const tokenIds = new Map<string, number>();
const tokenizer: D1Tokenizer = {
  encode(text) {
    return text
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .map((token) => tokenId(token));
  },
  tokenId,
  bosId: 1,
};

function tokenId(token: string): number {
  let id = tokenIds.get(token);
  if (id === undefined) {
    id = tokenIds.size + 2;
    tokenIds.set(token, id);
  }
  return id;
}

describe('D1 prompt packing', () => {
  it('keeps complete state, instruction, and option text when it fits', () => {
    const result = encodeQuestion(
      tokenizer,
      'customer was charged twice',
      {
        type: 'choice',
        instructions: 'select a team',
        criteria: { billing: 'charges refunds invoices', technical: 'app site faults' },
      },
      40,
    );

    for (const word of [
      'customer',
      'charged',
      'twice',
      'select',
      'team',
      'billing:',
      'invoices',
      'technical:',
      'faults',
    ]) {
      expect(result.ids).toContain(tokenId(word));
    }
    expect(result.markers).toHaveLength(2);
    expect(result.ids[result.markers[0]!]).toBe(tokenId('<|mask|>'));
    expect(result.ids[result.markers[1]!]).toBe(tokenId('<|mask|>'));
  });

  it('rejects an overlong state instead of truncating it', () => {
    expect(() =>
      encodeQuestion(
        tokenizer,
        'one two three four five six seven eight nine ten',
        { type: 'noul', instructions: 'is it true?' },
        16,
      ),
    ).toThrow('D1 prompt exceeds its 16-token context');
  });

  it('rejects overlong instructions and options instead of truncating them', () => {
    expect(() =>
      encodeQuestion(
        tokenizer,
        '',
        {
          type: 'choice',
          instructions: 'choose the best matching option from this long instruction',
          criteria: { first: 'a lengthy option description that cannot be cut' },
        },
        24,
      ),
    ).toThrow('D1 prompt exceeds its 24-token context');
  });
});
