import { describe, expect, it } from 'vitest';
import { bufferLimits } from '../../../src/core/gpu/limits.ts';

const adapterLimits = { maxStorageBufferBindingSize: 4_294_967_292, maxBufferSize: 4_294_967_296 };
const gpu = { requestAdapter: async () => ({ limits: adapterLimits }) } as unknown as Pick<
  GPU,
  'requestAdapter'
>;

describe('bufferLimits', () => {
  it("returns the adapter's limits when no size is given", async () => {
    expect(await bufferLimits(undefined, gpu)).toEqual(adapterLimits);
  });

  it('returns the requested size between the default and the adapter limit', async () => {
    expect(await bufferLimits(500_000_000, gpu)).toEqual({
      maxStorageBufferBindingSize: 500_000_000,
      maxBufferSize: 500_000_000,
    });
  });

  it('never goes below the WebGPU defaults or above the adapter', async () => {
    expect(await bufferLimits(1, gpu)).toEqual({
      maxStorageBufferBindingSize: 134_217_728,
      maxBufferSize: 268_435_456,
    });
    expect(await bufferLimits(2 ** 40, gpu)).toEqual(adapterLimits);
  });

  it('returns undefined with no adapter', async () => {
    const none = { requestAdapter: async () => null } as unknown as Pick<GPU, 'requestAdapter'>;
    expect(await bufferLimits(undefined, none)).toBeUndefined();
    expect(await bufferLimits(undefined, undefined)).toBeUndefined();
  });
});
