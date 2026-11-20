import { describe, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { depthwiseConv1d } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

interface Case {
  rows: number;
  channels: number;
  kernel: number;
  padLeft?: number;
  padRight?: number;
  dtype: 'f32' | 'f16';
}

function reference(x: Float32Array, w: Float32Array, c: Case): Float64Array {
  const left = c.padLeft ?? 0;
  const right = c.padRight ?? 0;
  const rows = c.rows + left + right - c.kernel + 1;
  const out = new Float64Array(rows * c.channels);
  for (let t = 0; t < rows; t++) {
    for (let channel = 0; channel < c.channels; channel++) {
      let sum = 0;
      for (let tap = 0; tap < c.kernel; tap++) {
        const source = t + tap - left;
        if (source >= 0 && source < c.rows) {
          sum += x[source * c.channels + channel]! * w[channel * c.kernel + tap]!;
        }
      }
      out[t * c.channels + channel] = sum;
    }
  }
  return out;
}

async function check(c: Case): Promise<void> {
  const round = c.dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
  const x = round(randomF32(c.rows * c.channels, 31));
  const w = round(randomF32(c.channels * c.kernel, 47));
  const meta = (rows: number, cols: number) => ({ ...matrix(rows, cols), dtype: c.dtype });
  const got = await toArray(
    depthwiseConv1d(
      upload(root, x, meta(c.rows, c.channels)),
      upload(root, w, meta(c.channels, c.kernel)),
      { kernelSize: c.kernel, padLeft: c.padLeft, padRight: c.padRight },
    ),
    ex,
  );
  expectClose(
    got,
    reference(x, w, c),
    c.dtype === 'f16' ? { atol: 1e-3, rtol: 2e-3 } : { atol: 1e-6, rtol: 1e-5 },
  );
}

describe('depthwiseConv1d kernel', () => {
  it.for<Case>([
    { rows: 9, channels: 5, kernel: 3, padLeft: 2, dtype: 'f32' },
    { rows: 7, channels: 4, kernel: 5, padRight: 4, dtype: 'f32' },
    { rows: 1, channels: 1, kernel: 1, dtype: 'f32' },
    { rows: 12, channels: 8, kernel: 7, dtype: 'f16' },
  ])(
    'matches the channel-wise convolution definition: $dtype $rows×$channels K=$kernel',
    async (c, ctx) => {
      if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
      await check(c);
    },
  );
});
