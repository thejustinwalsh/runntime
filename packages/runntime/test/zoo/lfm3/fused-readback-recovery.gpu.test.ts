/** The retained separate-submit readback path recovers after an eval failure.
 *
 *  runStep queues no fused readback before its eval: when graph collection,
 *  allocation or dispatch throws pre-submit, no shared ReadbackQueue entry is
 *  left behind to poison the next request with 'a readback is already
 *  queued'. toArrayFused is the only fused entry point kept, and it rejects
 *  its own pending readback when its eval fails. This locks both halves. */

import { expect, it } from 'vitest';
import { tensor } from '../../../src/core/gpu/buffers.ts';
import { matrix } from '../../../src/core/graph/value.ts';
import { toArray } from '../../../src/core/gpu/eval.ts';
import { gpu } from '../../gpu.ts';

it('toArrayFused rejects its pending readback and the next readback proceeds', async () => {
  const { root, ex } = await gpu();
  const good = tensor(root, new Float32Array([1, 2, 3, 4]), matrix(2, 2));
  try {
    // A released target makes the fused eval throw before any submit.
    const released = tensor(root, new Float32Array([5, 6, 7, 8]), matrix(2, 2));
    ex.releaseValues([released]);
    await expect(toArray(released, ex)).rejects.toThrow(/evalValues/);
    // The queue is clean: the next fused readback is not poisoned.
    await expect(toArray(good, ex)).resolves.toEqual(new Float32Array([1, 2, 3, 4]));
  } finally {
    if (good.state === 'materialized') good.buffer.destroy();
  }
  // The shared staging lease and memoized dims uniform persist per root by
  // design; the assertion above (second readback resolves) proves the queue
  // itself was cleared.
});
