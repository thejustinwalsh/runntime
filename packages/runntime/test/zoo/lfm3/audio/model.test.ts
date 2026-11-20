import { describe, expect, it } from 'vitest';
import {
  derive,
  quantTensor,
  type LazyStateDict,
  type LazyTensor,
} from '../../../../src/core/weights/safetensors.ts';
import { Lfm3AudioModel } from '../../../../src/zoo/lfm3/audio/model.ts';

const BOOKS = 12;
const BOOK_ROWS = 2_017;
const HIDDEN = 1_024;

/** The 12 depth codebooks, one [2017, 1024] matrix each, in groups of 32, symmetric or with
 *  offsets. */
function codebooks(withOffsets: boolean): LazyTensor {
  const groups = (BOOKS * BOOK_ROWS * HIDDEN) / 32;
  return quantTensor({
    name: 'a.gen.code.embd.weight',
    shape: [BOOKS, BOOK_ROWS, HIDDEN],
    bits: 8,
    groupSize: 32,
    byteLength: 0,
    sources: [],
    ints: async () => new Int8Array(0),
    scales: async () => new Float32Array(groups),
    offsets: withOffsets ? async () => new Float32Array(groups) : undefined,
  });
}

function checkpoint(withOffsets: boolean): LazyStateDict {
  const norm = derive([BOOKS, HIDDEN], [], async () => new Float32Array(BOOKS * HIDDEN));
  return {
    metadata: {},
    tensors: new Map([
      ['a.gen.code.embd.weight', codebooks(withOffsets)],
      ['a.gen.code.head_norm.weight', norm],
    ]),
    totalBytes: () => 0,
  };
}

describe('Lfm3AudioModel.transformStateDict', () => {
  it('pads each depth codebook into a head projection', () => {
    const sd = checkpoint(false);
    new Lfm3AudioModel().transformStateDict(sd);
    const head = sd.tensors.get('depthDecoder.books.0.head.weight');
    // 2017 rows rounded up to a multiple of 4
    expect(head?.shape).toEqual([2_020, HIDDEN]);
    expect(sd.tensors.get('depthDecoder.books.0.embedding.weight')?.shape).toEqual([
      BOOK_ROWS,
      HIDDEN,
    ]);
  });

  it('refuses a codebook with per-group offsets rather than drop them in the padding', () => {
    expect(() => new Lfm3AudioModel().transformStateDict(checkpoint(true))).toThrow(
      'cannot pad quantized projection',
    );
  });
});
