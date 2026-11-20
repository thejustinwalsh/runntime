import tgpu, { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import {
  type FloatBuffer,
  flatIndex,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** One output per time/channel pair, accumulating depthwise taps in f32. */
const Dims = d.struct({ rowsIn: d.u32, rowsOut: d.u32, padLeft: d.u32 });
const Config = d.struct({ channels: d.u32, kernel: d.u32 });
const config = tgpu.accessor(Config, { channels: 1, kernel: 1 });

function makeVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    w: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    dims: { uniform: Dims },
  });
  const store = elem.scalar;
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const C = config.$;
    const D = layout.$.dims;
    const idx = flatIndex(gid);
    if (idx >= D.rowsOut * C.channels) return;
    const row = d.u32(idx / C.channels);
    const channel = idx % C.channels;
    let acc = d.f32(0);
    for (let tap = d.u32(0); tap < C.kernel; tap++) {
      if (row + tap >= D.padLeft) {
        const sourceRow = row + tap - D.padLeft;
        if (sourceRow < D.rowsIn) {
          acc +=
            d.f32(layout.$.x[sourceRow * C.channels + channel]!) *
            d.f32(layout.$.w[channel * C.kernel + tap]!);
        }
      }
    }
    layout.$.out[idx] = store(acc);
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeVariant>>();
function variantFor(elem: Elem): ReturnType<typeof makeVariant> {
  let variant = variants.get(elem.key);
  if (!variant) {
    variant = makeVariant(elem);
    variants.set(elem.key, variant);
  }
  return variant;
}

export function createDepthwiseConv1dPipeline(
  root: TgpuRoot,
  cfg: { channels: number; kernel: number },
  elem: Elem = F32_ELEM,
) {
  return root.with(config, cfg).createComputePipeline({ compute: variantFor(elem).kernel });
}

export function depthwiseConv1dHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createDepthwiseConv1dPipeline>,
  args: { rowsIn: number; rowsOut: number; channels: number; padLeft: number },
  buffers: { x: FloatBuffer; w: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = cachedUniform(root, Dims, {
    rowsIn: args.rowsIn,
    rowsOut: args.rowsOut,
    padLeft: args.padLeft,
  });
  const bindGroup = cachedBindGroup(root, variantFor(elem).layout, { ...buffers, dims });
  return makeHandle(
    pipeline,
    'depthwiseConv1d',
    bindGroup,
    Math.ceil((args.rowsOut * args.channels) / WORKGROUP_SIZE),
  );
}
