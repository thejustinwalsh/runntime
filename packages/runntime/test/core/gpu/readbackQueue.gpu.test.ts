import { describe, expect, it, vi } from 'vitest';
import { leaseStaging } from '../../../src/core/gpu/buffers.ts';
import { ReadbackQueue } from '../../../src/core/gpu/readbackQueue.ts';
import { expectClose, gpu, randomF32, upload } from '../../gpu.ts';

const { root } = await gpu();

/** Whether the shared staging buffer is free: a lease taken and cancelled is
 *  leased again. A busy one would push both leases onto transient buffers. */
function sharedStagingIsFree(): boolean {
  const first = leaseStaging(root, 16);
  first.cancel();
  const second = leaseStaging(root, 16);
  second.cancel();
  return second.staging === first.staging;
}

describe('ReadbackQueue', () => {
  it('copies the requested values out in the submit that follows', async () => {
    const queue = new ReadbackQueue(root);
    const a = randomF32(6, 1);
    const b = randomF32(4, 2);
    const result = queue.request([upload(root, a, [2, 3]), upload(root, b, [2, 2])]);
    queue.submit(() => undefined);
    const [gotA, gotB] = await result;
    expectClose(gotA!, a, { atol: 0, rtol: 0 });
    expectClose(gotB!, b, { atol: 0, rtol: 0 });
  });

  it('takes one request at a time', () => {
    const queue = new ReadbackQueue(root);
    const first = queue.request([upload(root, randomF32(4, 3), [2, 2])]);
    expect(() => queue.request([upload(root, randomF32(4, 4), [2, 2])])).toThrow('already queued');
    queue.rejectPending(new Error('dropped'));
    return expect(first).rejects.toThrow('dropped');
  });

  it('rejects the request and rethrows when the build step throws', async () => {
    const queue = new ReadbackQueue(root);
    const result = queue.request([upload(root, randomF32(4, 5), [2, 2])]);
    expect(() =>
      queue.submit(() => {
        throw new Error('build failed');
      }),
    ).toThrow('build failed');
    await expect(result).rejects.toThrow('build failed');
  });

  it('rejects a request whose value is stored in another layout, leasing nothing', async () => {
    const queue = new ReadbackQueue(root);
    const value = upload(root, randomF32(4, 6), { elems: 4, dtype: 'f32', layout: 'hwc4' });
    const result = queue.request([value]);
    expect(() => queue.submit(() => undefined)).toThrow('toChw');
    await expect(result).rejects.toThrow('toChw');
    expect(sharedStagingIsFree()).toBe(true);
  });

  it('rejects the request and frees the staging buffer when the queue submit throws', async () => {
    const queue = new ReadbackQueue(root);
    const result = queue.request([upload(root, randomF32(4, 7), [2, 2])]);
    const submit = vi.spyOn(root.device.queue, 'submit').mockImplementation(() => {
      throw new Error('submit failed');
    });
    try {
      expect(() => queue.submit(() => undefined)).toThrow('submit failed');
    } finally {
      submit.mockRestore();
    }
    await expect(result).rejects.toThrow('submit failed');
    expect(sharedStagingIsFree()).toBe(true);
  });
});
