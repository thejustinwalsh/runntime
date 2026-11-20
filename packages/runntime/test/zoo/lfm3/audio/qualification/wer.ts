/** Word error rate for the audio round-trip qualification. */

/** Lowercases, drops punctuation and collapses whitespace so only the words are compared. */
export function normalizeWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, ' ')
    .replace(/'/g, '')
    .split(/\s+/)
    .filter((word) => word.length > 0);
}

/** Word-level edit distance between the reference and the hypothesis. */
export function wordEditDistance(
  reference: readonly string[],
  hypothesis: readonly string[],
): number {
  let previous = Array.from({ length: hypothesis.length + 1 }, (_, index) => index);
  for (let row = 1; row <= reference.length; row++) {
    const current = [row];
    for (let column = 1; column <= hypothesis.length; column++) {
      const substitution =
        previous[column - 1]! + (reference[row - 1] === hypothesis[column - 1] ? 0 : 1);
      current.push(Math.min(substitution, previous[column]! + 1, current[column - 1]! + 1));
    }
    previous = current;
  }
  return previous[hypothesis.length]!;
}

/** Edit distance over the reference word count: 0 is exact, 1 is every word wrong. */
export function wordErrorRate(reference: string, hypothesis: string): number {
  const expected = normalizeWords(reference);
  if (expected.length === 0) throw new Error('word error rate needs a non-empty reference');
  return wordEditDistance(expected, normalizeWords(hypothesis)) / expected.length;
}
