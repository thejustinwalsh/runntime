/** Checks upload ownership when public-model graph construction fails. */

import { expect, it } from 'vitest';
import { tensor } from '../../../src/core/gpu/buffers.ts';
import { matrix, type Value } from '../../../src/core/graph/value.ts';
import { Lfm3CausalLM } from '../../../src/zoo/lfm3/model.ts';
import { buffersSince, gpu, liveBuffers } from '../../gpu.ts';

it('exposes every upload to its caller when graph construction throws', async () => {
  const { root } = await gpu();
  const before = liveBuffers();
  const model = new Lfm3CausalLM();
  const embedding = tensor(root, Float32Array.of(1), matrix(1, 1));
  model.model.embed_tokens.forward = () => embedding;
  model.model.layers.items[0]!.forward = () => {
    throw new Error('injected synchronous graph failure');
  };
  const uploads: Value[] = [];

  try {
    expect(() => model.forward([1], root, undefined, uploads)).toThrow(
      'injected synchronous graph failure',
    );
    expect(uploads).toHaveLength(5);
    expect(uploads).toContain(embedding);
  } finally {
    for (const value of uploads) {
      if (value.state === 'materialized') value.buffer.destroy();
    }
    model.dispose();
  }

  expect(buffersSince(before)).toEqual([]);
});
