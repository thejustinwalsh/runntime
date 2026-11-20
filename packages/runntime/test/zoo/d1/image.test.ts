import { describe, expect, it } from 'vitest';
import { layout, PATCH, preprocessImage, resizeAA } from '../../../src/zoo/d1/image.ts';

describe('layout', () => {
  // Sizes are multiples of 32. A crop holds 64 to 256 projected tokens of 32x32 px, that is
  // 65_536 to 262_144 px, and an image over 524_288 rounded px is tiled in 512 px tiles.
  it.each([
    // already inside the budget: kept as is
    { name: '512x512', width: 512, height: 512, thumbnail: [512, 512], grid: [1, 1], tiled: false },
    // 1_024 px is under the 65_536 minimum; beta = sqrt(65_536 / 1_024) = 8, so 32 · 8 = 256
    { name: '32x32', width: 32, height: 32, thumbnail: [256, 256], grid: [1, 1], tiled: false },
    // 1_048_576 px is over the 262_144 maximum; beta = 2, so 1_024 / 2 = 512, and the
    // image is tiled: ratio 1 first matches the 2x2 grid, 3x3 needs more than 1_179_648 px
    {
      name: '1024x1024',
      width: 1024,
      height: 1024,
      thumbnail: [512, 512],
      grid: [2, 2],
      tiled: true,
    },
    // 6_144 rounded px is under the minimum; beta = sqrt(65_536 / 5_000) = 3.62, so
    // h = ceil(50 · 3.62 / 32) · 32 = 192 and w = ceil(100 · 3.62 / 32) · 32 = 384
    { name: '100x50', width: 100, height: 50, thumbnail: [192, 384], grid: [1, 1], tiled: false },
  ])('plans $name', ({ width, height, thumbnail, grid, tiled }) => {
    expect(layout(width, height)).toEqual({ thumbnail, grid, tiled });
  });
});

describe('resizeAA', () => {
  it('keeps an image at its own size', () => {
    const src = Float32Array.from({ length: 2 * 3 * 3 }, (_, i) => i);
    expect(resizeAA(src, 2, 3, 3, 2, 3)).toEqual(src);
  });

  it('averages two pixels into one with the triangle filter', () => {
    // scale 2, support 2: both taps weigh 0.75 before normalizing, so 0.5 each
    expect(resizeAA(Float32Array.from([0, 10]), 1, 2, 1, 1, 1)[0]).toBeCloseTo(5, 5);
  });

  it('widens the filter by the downscale factor', () => {
    // out 0: taps 0..2 weigh 0.75, 0.75, 0.25 of 1.75; out 1: taps 1..3 weigh 0.25, 0.75, 0.75
    const got = resizeAA(Float32Array.from([0, 10, 20, 30]), 1, 4, 1, 1, 2);
    expect(got[0]).toBeCloseTo((10 * 0.75 + 20 * 0.25) / 1.75, 4);
    expect(got[1]).toBeCloseTo((10 * 0.25 + 20 * 0.75 + 30 * 0.75) / 1.75, 4);
  });

  it('resizes height after width and keeps channels apart', () => {
    // [2, 1, 2]: channel 0 grows down the rows, channel 1 is constant
    const src = Float32Array.from([0, 7, 0, 7, 10, 7, 10, 7]);
    const got = resizeAA(src, 2, 2, 2, 1, 1);
    expect(got[0]).toBeCloseTo(5, 5);
    expect(got[1]).toBeCloseTo(7, 5);
  });
});

describe('preprocessImage', () => {
  const flat = (width: number, height: number, value: number) => ({
    width,
    height,
    data: new Uint8Array(width * height * 3).fill(value),
  });

  it.each([
    { value: 0, want: -1 },
    { value: 255, want: 1 },
  ])('maps byte $value to $want in 16 px patch rows', ({ value, want }) => {
    const crops = preprocessImage(flat(32, 32, value));
    expect(crops).toHaveLength(1);
    const [crop] = crops;
    // the 32x32 image is resized to a 256x256 thumbnail: 16 x 16 patches of 3 · 16 · 16 values
    expect([crop!.ph, crop!.pw]).toEqual([256 / PATCH, 256 / PATCH]);
    expect(crop!.patches).toHaveLength(16 * 16 * 3 * PATCH * PATCH);
    expect(crop!.patches.every((v) => v === want)).toBe(true);
  });

  it('puts the tiles of a large image before its thumbnail', () => {
    const crops = preprocessImage(flat(1024, 1024, 255));
    // a 2x2 grid of 512 px tiles, then the 512 px thumbnail
    expect(crops.map((c) => [c.ph, c.pw])).toEqual(Array(5).fill([32, 32]));
  });
});
