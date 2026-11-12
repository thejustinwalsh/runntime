import { describe, expect, it } from 'vitest';
import { depthwiseConv1d } from '../../../../../src/core/graph/ops/conv/conv1d.ts';
import { input, meta } from '../../values.ts';

const x = input([9, 3]);
const weight = input([3, 3]);

describe('depthwiseConv1d shape checks', () => {
  it('keeps the valid output width and rejects mismatched kernels and padding', () => {
    expect(depthwiseConv1d(x, weight, { kernelSize: 3, padLeft: 2 }).shape).toEqual(meta([9, 3]));
    expect(() => depthwiseConv1d(x, input([3, 4]), { kernelSize: 3 })).toThrow('weight shape');
    expect(() => depthwiseConv1d(x, weight, { kernelSize: 3, padLeft: 3 })).toThrow('padLeft');
    expect(() => depthwiseConv1d(input([2, 3]), weight, { kernelSize: 3 })).toThrow(
      'padded frames',
    );
  });
});
