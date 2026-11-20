import { describe, expect, it } from 'vitest';
import { leaseStaging } from '../../../src/core/gpu/buffers.ts';
import { buffersSince, gpu, liveBuffers } from '../../gpu.ts';

const { root } = await gpu();

describe('leaseStaging', () => {
  it('hands the shared staging buffer to the next lease once finish returns', async () => {
    const first = leaseStaging(root, 16);
    expect(await first.finish((mapped) => mapped.byteLength)).toBe(16);
    const second = leaseStaging(root, 16);
    expect(second.staging).toBe(first.staging);
    second.cancel();
  });

  it('leases a transient buffer while the shared one is busy', () => {
    const shared = leaseStaging(root, 16);
    const transient = leaseStaging(root, 16);
    expect(transient.staging).not.toBe(shared.staging);
    shared.cancel();
    transient.cancel();
  });

  it('cancel frees the shared buffer for the next lease', () => {
    const first = leaseStaging(root, 16);
    first.cancel();
    const second = leaseStaging(root, 16);
    expect(second.staging).toBe(first.staging);
    second.cancel();
  });

  it('cancel destroys a transient buffer and does nothing the second time', () => {
    const shared = leaseStaging(root, 16);
    const before = liveBuffers();
    const transient = leaseStaging(root, 16);
    expect(buffersSince(before)).toHaveLength(1);
    transient.cancel();
    expect(buffersSince(before)).toEqual([]);
    transient.cancel();
    // A repeated cancel must not mark the busy shared buffer free.
    expect(leaseStaging(root, 16).staging).not.toBe(shared.staging);
    shared.cancel();
  });
});
