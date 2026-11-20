import tgpu from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { bufferLimits, initRunntime } from '../../core/index.ts';
import { registerRunntimeBackend, type RegisterRunntimeBackendOpts } from './registry.ts';

/** One-call plugin setup: creates the WebGPU device, runs the models on it
 *  and patches transformers.js. Requests `subgroups`, which makes matmul and
 *  attention faster, and `shader-f16`, which the YOLO26 and DepthART models
 *  need and the others use when present. A device without them still loads
 *  every model that runs in f32. Also requests the adapter's buffer limits,
 *  since the WebGPU defaults refuse a tensor past 128 MiB. Options go straight to
 *  registerRunntimeBackend(), so `models` and `fallbackToOnnx` work here too.
 *  Returns the root for apps that also use TypeGPU directly. */
export async function initRunntimeBackend(
  opts: RegisterRunntimeBackendOpts = {},
): Promise<TgpuRoot> {
  const requiredLimits = await bufferLimits();
  const root = await tgpu.init({
    device: { optionalFeatures: ['subgroups', 'shader-f16'], requiredLimits },
  });
  initRunntime(root);
  registerRunntimeBackend(opts);
  return root;
}
