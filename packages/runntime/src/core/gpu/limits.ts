/** The two buffer limits a device needs raised to hold large tensors. */
export type BufferLimits = {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
};

/** Buffer limits to pass as `requiredLimits` when creating the device.
 *  WebGPU gives a device the default limits unless larger ones are requested:
 *  a 128 MiB storage buffer binding and a 256 MiB buffer, whatever the adapter
 *  supports. A tensor past the binding limit cannot be bound by any shader
 *  (see checkBindable), so a long prompt or a large f32 weight is refused on
 *  hardware that could hold it.
 *
 *  With no `bytes`, returns the adapter's own limits. With `bytes`, the
 *  largest single buffer the app expects, returns that size, never below the
 *  WebGPU default and never above what the adapter supports. Returns
 *  undefined when there is no adapter. */
export async function bufferLimits(
  bytes?: number,
  gpu: Pick<GPU, 'requestAdapter'> | undefined = globalThis.navigator?.gpu,
): Promise<BufferLimits | undefined> {
  const adapter = await gpu?.requestAdapter();
  if (!adapter) {
    return undefined;
  }
  const { maxStorageBufferBindingSize: binding, maxBufferSize: buffer } = adapter.limits;
  if (bytes === undefined) {
    return { maxStorageBufferBindingSize: binding, maxBufferSize: buffer };
  }
  return {
    maxStorageBufferBindingSize: Math.min(binding, Math.max(bytes, DEFAULT_BINDING)),
    maxBufferSize: Math.min(buffer, Math.max(bytes, DEFAULT_BUFFER)),
  };
}

/** WebGPU's default `maxStorageBufferBindingSize`: 128 MiB. */
const DEFAULT_BINDING = 134_217_728;
/** WebGPU's default `maxBufferSize`: 256 MiB. */
const DEFAULT_BUFFER = 268_435_456;
