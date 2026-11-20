/** The retained separate-submit readback path recovers after an eval failure.
 *
 *  runStep queues no fused readback before its eval: when graph collection,
 *  allocation or dispatch throws pre-submit, no shared ReadbackQueue entry is
 *  left behind to poison the next request with 'a readback is already
 *  queued'. toArrayFused is the only fused entry point kept, and it rejects
 *  its own pending readback when its eval fails. This locks both halves: the
 *  failing leg goes through the fused queue, and so does the recovering leg. */

import { expect, it } from 'vitest';
import { add } from '../../../src/core/graph/ops.ts';
import { tensor } from '../../../src/core/gpu/buffers.ts';
import { matrix } from '../../../src/core/graph/value.ts';
import { toArray } from '../../../src/core/gpu/eval.ts';
import { gpu } from '../../gpu.ts';

it('toArrayFused rejects its pending readback and a pending successor proceeds', async () => {
  const { root, ex } = await gpu();
  const left = tensor(root, new Float32Array([1, 2, 3, 4]), matrix(2, 2));
  const right = tensor(root, new Float32Array([10, 20, 30, 40]), matrix(2, 2));
  try {
    // A released target makes the fused eval throw before any submit, while
    // the pending readback is queued: rejectPending must fire.
    const released = tensor(root, new Float32Array([5, 6, 7, 8]), matrix(2, 2));
    ex.releaseValues([released]);
    await expect(toArray(released, ex)).rejects.toThrow(/evalValues/);
    // The queue is clean, and the successor is genuinely pending, so this
    // second readback exercises queue rejection followed by a fused requeue.
    const pendingSum = add(left, right);
    await expect(toArray(pendingSum, ex)).resolves.toEqual(new Float32Array([11, 22, 33, 44]));
  } finally {
    for (const value of [left, right]) {
      if (value.state === 'materialized') value.buffer.destroy();
    }
  }
  // The shared staging lease and memoized dims uniform persist per root by
  // design; the resolving successor above proves the queue itself was cleared.
});
