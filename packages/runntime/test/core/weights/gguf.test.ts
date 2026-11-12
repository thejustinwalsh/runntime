import { describe, expect, it } from 'vitest';
import { fromGguf, parseGgufHeader } from '../../../src/core/weights/gguf.ts';
import type { RangeSource } from '../../../src/core/weights/safetensors.ts';
import { Sha256 } from '../../../src/core/weights/sha256.ts';

interface TensorFixture {
  name: string;
  dimensions: number[];
  type: 0 | 12 | 14;
  payload: Uint8Array;
}

function fixture(tensors: readonly TensorFixture[], alignment = 32): Uint8Array {
  const payloadOffsets: number[] = [];
  let payloadBytes = 0;
  for (const tensor of tensors) {
    payloadOffsets.push(payloadBytes);
    payloadBytes = Math.ceil((payloadBytes + tensor.payload.byteLength) / alignment) * alignment;
  }
  const header: number[] = [];
  const u32 = (value: number) => {
    for (let shift = 0; shift < 32; shift += 8) header.push((value >>> shift) & 0xff);
  };
  const u64 = (value: number) => {
    let rest = BigInt(value);
    for (let index = 0; index < 8; index++) {
      header.push(Number(rest & 0xffn));
      rest >>= 8n;
    }
  };
  const string = (value: string) => {
    const bytes = new TextEncoder().encode(value);
    u64(bytes.byteLength);
    header.push(...bytes);
  };
  header.push(...new TextEncoder().encode('GGUF'));
  u32(3);
  u64(tensors.length);
  u64(1);
  string('general.alignment');
  u32(4);
  u32(alignment);
  tensors.forEach((tensor, index) => {
    string(tensor.name);
    u32(tensor.dimensions.length);
    tensor.dimensions.forEach(u64);
    u32(tensor.type);
    u64(payloadOffsets[index]!);
  });
  const dataStart = Math.ceil(header.length / alignment) * alignment;
  const file = new Uint8Array(dataStart + payloadBytes);
  file.set(header);
  tensors.forEach((tensor, index) => file.set(tensor.payload, dataStart + payloadOffsets[index]!));
  return file;
}

const source = (bytes: Uint8Array): RangeSource => ({
  async read(begin, end) {
    if (begin < 0 || end > bytes.byteLength || end < begin) throw new Error('range out of bounds');
    return bytes.slice(begin, end);
  },
});

const sha256 = (bytes: Uint8Array): string => {
  const hash = new Sha256();
  hash.update(bytes);
  return hash.hex();
};

const half = (bytes: Uint8Array, offset: number, value: number): void => {
  new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).setUint16(offset, value, true);
};

function q4Block(): { payload: Uint8Array; ints: number[]; scales: number[]; offsets: number[] } {
  const payload = new Uint8Array(144);
  half(payload, 0, 0x3800); // d = 0.5
  half(payload, 2, 0x3400); // dmin = 0.25
  const groupScales = [1, 2, 3, 4, 5, 6, 7, 8];
  const minima = [0, 1, 2, 3, 4, 5, 6, 7];
  for (let group = 0; group < 4; group++) {
    payload[4 + group] = groupScales[group]!;
    payload[8 + group] = minima[group]!;
  }
  for (let group = 4; group < 8; group++) {
    payload[8 + group] = groupScales[group]! | (minima[group]! << 4);
    payload[group - 4] = payload[group - 4]! | ((groupScales[group]! >> 4) << 6);
    payload[group] = payload[group]! | ((minima[group]! >> 4) << 6);
  }
  const ints = new Array<number>(256);
  const scales: number[] = [];
  const offsets: number[] = [];
  for (let group = 0; group < 8; group++) {
    const scale = 0.5 * groupScales[group]!;
    scales.push(scale);
    offsets.push(scale * 8 - 0.25 * minima[group]!);
    const qOffset = 16 + (group >> 1) * 32;
    for (let index = 0; index < 32; index++) {
      const quant = (group + (index % 16)) % 16;
      const at = qOffset + index;
      payload[at] =
        group % 2 === 0 ? (payload[at]! & 0xf0) | quant : (payload[at]! & 0x0f) | (quant << 4);
      ints[group * 32 + index] = quant - 8;
    }
  }
  return { payload, ints, scales, offsets };
}

function q6Block(): { payload: Uint8Array; ints: number[]; scales: number[] } {
  const payload = new Uint8Array(210);
  half(payload, 208, 0x3c00); // d = 1
  const groupScales = Array.from({ length: 16 }, (_, index) => index - 8);
  const ints = new Array<number>(256);
  for (let group = 0; group < 16; group++) payload[192 + group] = groupScales[group]! & 0xff;
  for (let output = 0; output < 256; output++) {
    const halfBlock = Math.floor(output / 128);
    const withinHalf = output % 128;
    const sub = Math.floor(withinHalf / 32);
    const index = withinHalf % 32;
    const quant = (output * 13) % 64;
    const lowAt = halfBlock * 64 + (sub % 2) * 32 + index;
    const highAt = 128 + halfBlock * 32 + index;
    const lowShift = sub >= 2 ? 4 : 0;
    payload[lowAt] = payload[lowAt]! | ((quant & 15) << lowShift);
    payload[highAt] = payload[highAt]! | (((quant >> 4) & 3) << (sub * 2));
    ints[output] = quant - 32;
  }
  return { payload, ints, scales: groupScales };
}

describe('GGUF v3 weights', () => {
  it('uses one final aligned payload start for every tensor entry', async () => {
    const bytes = fixture([
      { name: 'first', dimensions: [2], type: 0, payload: new Uint8Array(8) },
      { name: 'second', dimensions: [2], type: 0, payload: new Uint8Array(8) },
    ]);
    const index = await parseGgufHeader(source(bytes), bytes.byteLength);
    expect(index.tensors.get('first')).toMatchObject({
      shape: [2],
      type: 'F32',
      begin: index.dataStart,
      end: index.dataStart + 8,
    });
    expect(index.tensors.get('second')).toMatchObject({
      begin: index.dataStart + 32,
      end: index.dataStart + 40,
    });
  });

  it('decodes all Q4_K groups and preserves their affine minima', async () => {
    const block = q4Block();
    const bytes = fixture([{ name: 'q4', dimensions: [256], type: 12, payload: block.payload }]);
    const state = await fromGguf(source(bytes), {
      fileBytes: bytes.byteLength,
      sha256: sha256(bytes),
    });
    const tensor = state.tensors.get('q4');
    if (tensor?.kind !== 'quant') throw new Error('expected Q4_K tensor');
    expect(tensor.bits).toBe(4);
    expect(tensor.groupSize).toBe(32);
    expect(await tensor.ints()).toEqual(new Int8Array(block.ints));
    expect(await tensor.scales()).toEqual(new Float32Array(block.scales));
    expect(await tensor.offsets?.()).toEqual(new Float32Array(block.offsets));
    for (let index = 0; index < 256; index++) {
      const group = Math.floor(index / 32);
      expect(block.ints[index]! * block.scales[group]! + block.offsets[group]!).toBeCloseTo(
        (((index % 32) + group) % 16) * block.scales[group]! - 0.25 * group,
        5,
      );
    }
  });

  it('unpacks Q6_K codes losslessly into signed int8 with signed group scales', async () => {
    const block = q6Block();
    const bytes = fixture([{ name: 'q6', dimensions: [256], type: 14, payload: block.payload }]);
    const state = await fromGguf(source(bytes), {
      fileBytes: bytes.byteLength,
      sha256: sha256(bytes),
    });
    const tensor = state.tensors.get('q6');
    if (tensor?.kind !== 'quant') throw new Error('expected Q6_K tensor');
    expect(tensor.bits).toBe(8);
    expect(tensor.groupSize).toBe(16);
    expect(await tensor.ints()).toEqual(new Int8Array(block.ints));
    expect(await tensor.scales()).toEqual(new Float32Array(block.scales));
    expect(tensor.offsets).toBeUndefined();
  });

  it('rejects a mismatched source digest before exposing tensors', async () => {
    const bytes = fixture([
      { name: 'first', dimensions: [1], type: 0, payload: new Uint8Array(4) },
    ]);
    await expect(
      fromGguf(source(bytes), { fileBytes: bytes.byteLength, sha256: '0'.repeat(64) }),
    ).rejects.toMatchObject({ code: 'CHECKPOINT_MISMATCH' });
  });
});
