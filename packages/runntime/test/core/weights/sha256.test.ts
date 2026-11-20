import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { Sha256 } from '../../../src/core/weights/sha256.ts';

describe('incremental SHA-256', () => {
  it.each([
    { input: '', digest: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855' },
    { input: 'abc', digest: 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad' },
    {
      input: 'The quick brown fox jumps over the lazy dog',
      digest: 'd7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592',
    },
  ])('hashes $input', ({ input, digest }) => {
    const hash = new Sha256();
    hash.update(new TextEncoder().encode(input));
    expect(hash.hex()).toBe(digest);
    expect(hash.hex()).toBe(digest);
  });

  it.each([55, 56, 63, 64, 65, 4_097])('keeps byte counts across chunks (%i bytes)', (length) => {
    const bytes = Uint8Array.from({ length }, (_, index) => index & 0xff);
    const hash = new Sha256();
    for (let offset = 0; offset < bytes.length; offset += 31) {
      hash.update(bytes.subarray(offset, Math.min(offset + 31, bytes.length)));
    }
    const expected = createHash('sha256').update(bytes).digest('hex');
    expect(hash.hex()).toBe(expected);
  });
});
