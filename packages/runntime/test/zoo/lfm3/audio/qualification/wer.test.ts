/** Unit tests for the word error rate used by the audio round-trip qualification. */

import { describe, expect, it } from 'vitest';
import { normalizeWords, wordEditDistance, wordErrorRate } from './wer.ts';

describe('word error rate', () => {
  it('ignores case, punctuation and apostrophes', () => {
    expect(normalizeWords("Don't stop, MARY!  (really)")).toEqual([
      'dont',
      'stop',
      'mary',
      'really',
    ]);
  });

  it('is zero for equal words and counts each edit once', () => {
    expect(wordErrorRate('the cat sat', 'The cat sat.')).toBe(0);
    expect(wordErrorRate('the cat sat', 'the dog sat')).toBeCloseTo(1 / 3);
    expect(wordErrorRate('the cat sat', 'the cat')).toBeCloseTo(1 / 3);
    expect(wordErrorRate('the cat sat', 'the big cat sat')).toBeCloseTo(1 / 3);
  });

  it('is one for an empty hypothesis and can exceed one for insertions', () => {
    expect(wordErrorRate('a b', '')).toBe(1);
    expect(wordErrorRate('a', 'a b c')).toBe(2);
  });

  it('matches the Levenshtein distance of the classic example', () => {
    expect(wordEditDistance([...'kitten'], [...'sitting'])).toBe(3);
  });

  it('refuses an empty reference', () => {
    expect(() => wordErrorRate('...', 'anything')).toThrow(/non-empty reference/);
  });
});
