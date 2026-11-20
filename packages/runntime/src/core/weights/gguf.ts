/** GGUF v3 header parsing and lazy F16/F32/Q4_K/Q6_K/Q8_0 tensor access. */

import { RunntimeError } from '../error.ts';
import { f16ToF32 } from './convert.ts';
import { cachedRangeSource } from './cachedSource.ts';
import type { WeightCache } from './cache.ts';
import { Sha256 } from './sha256.ts';
import {
  chunkedSource,
  deriveFromTensors,
  quantTensor,
  type FileSource,
  type LazyStateDict,
  type RangeSource,
} from './safetensors.ts';

const HEADER_LIMIT = 64 * 1024 * 1024;
const CHUNK_BYTES = 2 * 1024 * 1024;
const QK_BLOCK = 256;
const Q4_K_BYTES = 144;
const Q6_K_BYTES = 210;
const Q8_0_BLOCK = 32;
const Q8_0_BYTES = 34;

export type GgufTensorType = 'F16' | 'F32' | 'Q4_K' | 'Q6_K' | 'Q8_0';

export interface GgufTensorInfo {
  readonly name: string;
  /** Logical dimensions in row-major order. */
  readonly shape: number[];
  readonly type: GgufTensorType;
  readonly begin: number;
  readonly end: number;
}

export interface GgufIndex {
  readonly version: number;
  readonly metadata: Record<string, unknown>;
  readonly tensors: Map<string, GgufTensorInfo>;
  readonly dataStart: number;
  readonly fileBytes: number;
}

class Reader {
  private offset = 0;
  private readonly view: DataView;
  private readonly decoder = new TextDecoder('utf-8', { fatal: true });

  constructor(private readonly bytes: Uint8Array) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  get position(): number {
    return this.offset;
  }

  private require(size: number): void {
    if (!Number.isSafeInteger(size) || size < 0 || this.offset + size > this.bytes.byteLength) {
      throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF header is truncated');
    }
  }

  u8(): number {
    this.require(1);
    return this.view.getUint8(this.offset++);
  }

  i8(): number {
    this.require(1);
    return this.view.getInt8(this.offset++);
  }

  u16(): number {
    this.require(2);
    const value = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return value;
  }

  i16(): number {
    this.require(2);
    const value = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return value;
  }

  u32(): number {
    this.require(4);
    const value = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return value;
  }

  i32(): number {
    this.require(4);
    const value = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return value;
  }

  u64(): number {
    this.require(8);
    const value = this.view.getBigUint64(this.offset, true);
    this.offset += 8;
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF header contains an unsafe integer');
    }
    return Number(value);
  }

  i64(): number {
    this.require(8);
    const value = this.view.getBigInt64(this.offset, true);
    this.offset += 8;
    if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) {
      throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF header contains an unsafe integer');
    }
    return Number(value);
  }

  f32(): number {
    this.require(4);
    const value = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return value;
  }

  f64(): number {
    this.require(8);
    const value = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return value;
  }

  string(): string {
    const length = this.u64();
    this.require(length);
    const value = this.decoder.decode(this.bytes.subarray(this.offset, this.offset + length));
    this.offset += length;
    return value;
  }
}

type MetadataType = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;

function metadataValue(reader: Reader, type: MetadataType): unknown {
  switch (type) {
    case 0:
      return reader.u8();
    case 1:
      return reader.i8();
    case 2:
      return reader.u16();
    case 3:
      return reader.i16();
    case 4:
      return reader.u32();
    case 5:
      return reader.i32();
    case 6:
      return reader.f32();
    case 7:
      return reader.u8() !== 0;
    case 8:
      return reader.string();
    case 9: {
      const elementType = reader.u32();
      const length = reader.u64();
      if (elementType > 12 || length > 1_000_000) {
        throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF metadata array is unsupported');
      }
      return Array.from({ length }, () => metadataValue(reader, elementType as MetadataType));
    }
    case 10:
      return reader.u64();
    case 11:
      return reader.i64();
    case 12:
      return reader.f64();
  }
}

function tensorBytes(type: GgufTensorType, shape: readonly number[]): number {
  const elements = shape.reduce((product, dimension) => {
    const next = product * dimension;
    if (!Number.isSafeInteger(next)) {
      throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF tensor has an unsafe element count');
    }
    return next;
  }, 1);
  if (type === 'F16') return elements * 2;
  if (type === 'F32') return elements * 4;
  if (type === 'Q8_0') {
    if (elements % Q8_0_BLOCK !== 0) {
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `GGUF ${type} tensor has ${elements} elements, not a multiple of ${Q8_0_BLOCK}`,
      );
    }
    return (elements / Q8_0_BLOCK) * Q8_0_BYTES;
  }
  if (elements % QK_BLOCK !== 0) {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      `GGUF ${type} tensor has ${elements} elements, not a multiple of ${QK_BLOCK}`,
    );
  }
  return (elements / QK_BLOCK) * (type === 'Q4_K' ? Q4_K_BYTES : Q6_K_BYTES);
}

/** Parses a bounded GGUF v3 header without reading its tensor payloads. */
export async function parseGgufHeader(source: RangeSource, fileBytes: number): Promise<GgufIndex> {
  if (!Number.isSafeInteger(fileBytes) || fileBytes < 24) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      'GGUF file size must be a safe integer of at least 24 bytes',
    );
  }
  const prefix = await source.read(0, 24);
  if (prefix.byteLength !== 24 || new TextDecoder().decode(prefix.subarray(0, 4)) !== 'GGUF') {
    throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF file has an invalid magic header');
  }
  const prefixView = new DataView(prefix.buffer, prefix.byteOffset, prefix.byteLength);
  const version = prefixView.getUint32(4, true);
  const tensorCount = Number(prefixView.getBigUint64(8, true));
  const metadataCount = Number(prefixView.getBigUint64(16, true));
  if (version !== 3) {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      `GGUF version ${version} is unsupported; expected 3`,
    );
  }
  if (
    !Number.isSafeInteger(tensorCount) ||
    tensorCount <= 0 ||
    tensorCount > 100_000 ||
    !Number.isSafeInteger(metadataCount) ||
    metadataCount > 100_000
  ) {
    throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF header counts are invalid');
  }

  const headerEnd = Math.min(fileBytes, HEADER_LIMIT);
  const headerBytes = await source.read(0, headerEnd);
  if (headerBytes.byteLength !== headerEnd) {
    throw new RunntimeError('LOAD_FAILED', 'GGUF header range returned a short read');
  }
  const reader = new Reader(headerBytes);
  if (reader.u32() !== 0x46554747) {
    throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF file has an invalid magic header');
  }
  if (reader.u32() !== version) {
    throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF header changed while reading');
  }
  const readCount = (value: number): number => {
    const parsed = reader.u64();
    if (parsed !== value)
      throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF header changed while reading');
    return parsed;
  };
  readCount(tensorCount);
  readCount(metadataCount);

  const metadata: Record<string, unknown> = {};
  for (let index = 0; index < metadataCount; index++) {
    const key = reader.string();
    const type = reader.u32();
    if (type > 12 || Object.hasOwn(metadata, key)) {
      throw new RunntimeError('CHECKPOINT_MISMATCH', `GGUF metadata entry '${key}' is invalid`);
    }
    metadata[key] = metadataValue(reader, type as MetadataType);
  }

  const alignmentValue = metadata['general.alignment'] ?? 32;
  if (
    typeof alignmentValue !== 'number' ||
    !Number.isSafeInteger(alignmentValue) ||
    alignmentValue <= 0 ||
    Math.log2(alignmentValue) % 1 !== 0
  ) {
    throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF tensor alignment is invalid');
  }
  const alignment = alignmentValue;
  const tensorEntries: { name: string; shape: number[]; type: GgufTensorType; offset: number }[] =
    [];
  const names = new Set<string>();
  for (let index = 0; index < tensorCount; index++) {
    const name = reader.string();
    const dimensions = reader.u32();
    if (names.has(name) || dimensions < 1 || dimensions > 4) {
      throw new RunntimeError('CHECKPOINT_MISMATCH', `GGUF tensor '${name}' has invalid metadata`);
    }
    names.add(name);
    const shape = Array.from({ length: dimensions }, () => reader.u64()).reverse();
    if (shape.some((dimension) => dimension <= 0)) {
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `GGUF tensor '${name}' has an empty dimension`,
      );
    }
    const rawType = reader.u32();
    const type: GgufTensorType | undefined =
      rawType === 0
        ? 'F32'
        : rawType === 1
          ? 'F16'
          : rawType === 8
            ? 'Q8_0'
            : rawType === 12
              ? 'Q4_K'
              : rawType === 14
                ? 'Q6_K'
                : undefined;
    if (!type) {
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `GGUF tensor '${name}' has unsupported type ${rawType}`,
      );
    }
    const offset = reader.u64();
    if (offset % alignment !== 0) {
      throw new RunntimeError('CHECKPOINT_MISMATCH', `GGUF tensor '${name}' offset is not aligned`);
    }
    tensorEntries.push({ name, shape, type, offset });
  }

  const dataStart = Math.ceil(reader.position / alignment) * alignment;
  const entries: GgufTensorInfo[] = tensorEntries.map(({ name, shape, type, offset }) => {
    const begin = dataStart + offset;
    const end = begin + tensorBytes(type, shape);
    if (!Number.isSafeInteger(begin) || !Number.isSafeInteger(end) || end > fileBytes) {
      throw new RunntimeError('CHECKPOINT_MISMATCH', `GGUF tensor '${name}' exceeds the file`);
    }
    return { name, shape, type, begin, end };
  });
  entries.sort((left, right) => left.begin - right.begin);
  for (let index = 1; index < entries.length; index++) {
    if (entries[index]!.begin < entries[index - 1]!.end) {
      throw new RunntimeError('CHECKPOINT_MISMATCH', 'GGUF tensor payloads overlap');
    }
  }
  return {
    version,
    metadata,
    tensors: new Map(entries.map((entry) => [entry.name, entry])),
    dataStart,
    fileBytes,
  };
}

function fp16(view: DataView, offset: number): number {
  const bits = new Uint16Array([view.getUint16(offset, true)]);
  return f16ToF32(bits)[0]!;
}

function sharedTensorParts<T>(
  info: GgufTensorInfo,
  source: RangeSource,
  count: number,
  decode: (bytes: Uint8Array) => T,
): (part: number) => Promise<T> {
  let pending: Promise<T> | undefined;
  const consumed = new Set<number>();
  return async (part) => {
    let current = pending;
    if (!current) {
      current = source.read(info.begin, info.end).then((bytes) => {
        if (bytes.byteLength !== info.end - info.begin) {
          throw new RunntimeError(
            'LOAD_FAILED',
            `GGUF tensor '${info.name}' returned a short read`,
          );
        }
        return decode(bytes);
      });
      pending = current;
      current.catch(() => {
        if (pending === current) pending = undefined;
      });
    }
    const result = await current;
    consumed.add(part);
    if (consumed.size === count) {
      pending = undefined;
      consumed.clear();
    }
    return result;
  };
}

/** The file tensor `info` is read from, for the binding order. */
function sourcesOf(info: GgufTensorInfo): FileSource[] {
  return [{ name: info.name, position: info.begin, byteLength: info.end - info.begin }];
}

function q4Tensor(info: GgufTensorInfo, source: RangeSource): ReturnType<typeof quantTensor> {
  const elements = info.shape.reduce((product, dimension) => product * dimension, 1);
  const blocks = elements / QK_BLOCK;
  const readPart = sharedTensorParts(info, source, 3, (data) => {
    const groups = elements / 32;
    const scales = new Float32Array(groups);
    const offsets = new Float32Array(groups);
    const ints = new Int8Array(elements);
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    for (let block = 0; block < blocks; block++) {
      const base = block * Q4_K_BYTES;
      const scaleOffset = base + 4;
      const d = fp16(view, base);
      const dmin = fp16(view, base + 2);
      for (let group = 0; group < 8; group++) {
        let scale: number;
        let minimum: number;
        if (group < 4) {
          scale = data[scaleOffset + group]! & 63;
          minimum = data[scaleOffset + group + 4]! & 63;
        } else {
          scale =
            (data[scaleOffset + group + 4]! & 15) | ((data[scaleOffset + group - 4]! >> 6) << 4);
          minimum =
            (data[scaleOffset + group + 4]! >> 4) | ((data[scaleOffset + group]! >> 6) << 4);
        }
        const index = block * 8 + group;
        scales[index] = d * scale;
        offsets[index] = d * scale * 8 - dmin * minimum;
        const qOffset = base + 16 + (group >> 1) * 32;
        const high = group % 2 === 1;
        for (let value = 0; value < 32; value++) {
          const byte = data[qOffset + value]!;
          ints[block * QK_BLOCK + group * 32 + value] = (high ? byte >> 4 : byte & 15) - 8;
        }
      }
    }
    return { ints, scales, offsets };
  });
  return quantTensor({
    name: info.name,
    shape: info.shape,
    bits: 4,
    groupSize: 32,
    byteLength: info.end - info.begin,
    sources: sourcesOf(info),
    async ints() {
      return (await readPart(0)).ints;
    },
    async scales() {
      return (await readPart(1)).scales;
    },
    async offsets() {
      return (await readPart(2)).offsets;
    },
  });
}

function q6Tensor(info: GgufTensorInfo, source: RangeSource): ReturnType<typeof quantTensor> {
  const elements = info.shape.reduce((product, dimension) => product * dimension, 1);
  const blocks = elements / QK_BLOCK;
  const groupSize = 16;
  const readPart = sharedTensorParts(info, source, 2, (data) => {
    const groups = new Float32Array(elements / groupSize);
    const ints = new Int8Array(elements);
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    for (let block = 0; block < blocks; block++) {
      const base = block * Q6_K_BYTES;
      const ql = base;
      const qh = base + 128;
      const d = fp16(view, base + 208);
      for (let group = 0; group < 16; group++) {
        groups[block * 16 + group] = d * view.getInt8(base + 192 + group);
      }
      for (let half = 0; half < 2; half++) {
        for (let sub = 0; sub < 4; sub++) {
          const lowOffset = ql + half * 64 + (sub % 2) * 32;
          const highOffset = qh + half * 32;
          const highNibble = sub >= 2;
          const highShift = sub * 2;
          for (let index = 0; index < 32; index++) {
            const low = data[lowOffset + index]!;
            const high = data[highOffset + index]!;
            const nibble = highNibble ? low >> 4 : low & 15;
            const value = nibble | (((high >> highShift) & 3) << 4);
            const output = block * QK_BLOCK + half * 128 + sub * 32 + index;
            ints[output] = value - 32;
          }
        }
      }
    }
    return { ints, scales: groups };
  });
  return quantTensor({
    name: info.name,
    shape: info.shape,
    bits: 8,
    groupSize,
    byteLength: info.end - info.begin,
    sources: sourcesOf(info),
    async ints() {
      return (await readPart(0)).ints;
    },
    async scales() {
      return (await readPart(1)).scales;
    },
  });
}

function q8_0Tensor(info: GgufTensorInfo, source: RangeSource): ReturnType<typeof quantTensor> {
  const elements = info.shape.reduce((product, dimension) => product * dimension, 1);
  const blocks = elements / Q8_0_BLOCK;
  const readPart = sharedTensorParts(info, source, 2, (data) => {
    const scales = new Float32Array(blocks);
    const ints = new Int8Array(elements);
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    for (let block = 0; block < blocks; block++) {
      const base = block * Q8_0_BYTES;
      scales[block] = fp16(view, base);
      for (let index = 0; index < Q8_0_BLOCK; index++) {
        ints[block * Q8_0_BLOCK + index] = view.getInt8(base + 2 + index);
      }
    }
    return { ints, scales };
  });
  return quantTensor({
    name: info.name,
    shape: info.shape,
    bits: 8,
    groupSize: Q8_0_BLOCK,
    byteLength: info.end - info.begin,
    sources: sourcesOf(info),
    async ints() {
      return (await readPart(0)).ints;
    },
    async scales() {
      return (await readPart(1)).scales;
    },
  });
}

function ggufTensor(info: GgufTensorInfo, source: RangeSource) {
  if (info.type === 'Q4_K') return q4Tensor(info, source);
  if (info.type === 'Q6_K') return q6Tensor(info, source);
  if (info.type === 'Q8_0') return q8_0Tensor(info, source);
  return {
    ...deriveFromTensors(
      info.shape,
      [],
      async () => {
        const bytes = await source.read(info.begin, info.end);
        if (bytes.byteLength !== info.end - info.begin) {
          throw new RunntimeError(
            'LOAD_FAILED',
            `GGUF tensor '${info.name}' returned a short read`,
          );
        }
        if (info.type === 'F16') {
          return f16ToF32(
            new Uint16Array(
              bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
            ),
          );
        }
        return new Float32Array(
          bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
        );
      },
      info.end - info.begin,
    ),
    sources: sourcesOf(info),
  };
}

function abortIfNeeded(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new RunntimeError('LOAD_ABORTED', 'model load aborted', { cause: signal.reason });
  }
}

export interface GgufStateDict extends LazyStateDict {
  rows(name: string, start: number, count: number): ReturnType<typeof ggufTensor>;
}

/** Decode exactly the requested embedding rows. Keep no decoded checkpoint table. */
export async function readGgufEmbeddingRows(
  file: GgufStateDict,
  name: string,
  width: number,
  ids: readonly number[],
  signal: AbortSignal,
): Promise<Float32Array> {
  const output = new Float32Array(ids.length * width);
  const positions = new Map<number, number[]>();
  ids.forEach((id, index) => {
    const repeats = positions.get(id) ?? [];
    repeats.push(index);
    positions.set(id, repeats);
  });
  const entries = [...positions];
  // Bound concurrent IPC reads without serializing prompt lookups.
  for (let start = 0; start < entries.length; start += 16) {
    await Promise.all(
      entries.slice(start, start + 16).map(async ([id, repeats]) => {
        signal.throwIfAborted();
        const row = file.rows(name, id, 1);
        const values =
          row.kind === 'quant'
            ? await (async () => {
                const ints = await row.ints();
                const scales = await row.scales();
                const offsets = await row.offsets?.();
                return Float32Array.from(ints, (value, col) => {
                  const group = Math.floor(col / row.groupSize);
                  return value * scales[group]! + (offsets?.[group] ?? 0);
                });
              })()
            : await row.f32();
        signal.throwIfAborted();
        for (const index of repeats) output.set(values, index * width);
      }),
    );
  }
  return output;
}

/** Verifies exact source bytes, then exposes their tensors as a lazy state dict. */
export async function fromGguf(
  source: RangeSource,
  opts: {
    fileBytes: number;
    sha256: string;
    cache?: WeightCache;
    cacheId?: string;
    signal?: AbortSignal;
  },
): Promise<GgufStateDict> {
  if (!Number.isSafeInteger(opts.fileBytes) || opts.fileBytes < 24) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      'GGUF file size must be a safe integer of at least 24 bytes',
    );
  }
  if (!/^[a-f0-9]{64}$/.test(opts.sha256)) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      'GGUF SHA-256 must be 64 lowercase hexadecimal digits',
    );
  }
  if (opts.cache && !opts.cacheId) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      'GGUF cache with a RangeSource needs an explicit cacheId',
    );
  }
  const cached = opts.cache ? cachedRangeSource(source, opts.cache, opts.cacheId!) : source;
  const hash = new Sha256();
  for (let begin = 0; begin < opts.fileBytes; begin += CHUNK_BYTES) {
    abortIfNeeded(opts.signal);
    const end = Math.min(begin + CHUNK_BYTES, opts.fileBytes);
    let bytes: Uint8Array;
    try {
      bytes = await cached.read(begin, end);
    } catch (error) {
      if (opts.signal?.aborted) abortIfNeeded(opts.signal);
      throw error;
    }
    if (bytes.byteLength !== end - begin) {
      throw new RunntimeError('LOAD_FAILED', `GGUF range [${begin},${end}) returned a short read`);
    }
    hash.update(bytes);
  }
  abortIfNeeded(opts.signal);
  if (hash.hex() !== opts.sha256) {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      'GGUF source SHA-256 does not match the pinned model',
    );
  }

  const index = await parseGgufHeader(cached, opts.fileBytes);
  const data = chunkedSource(cached, opts.fileBytes, CHUNK_BYTES);
  const tensors = new Map([...index.tensors].map(([name, info]) => [name, ggufTensor(info, data)]));
  return {
    metadata: {},
    ggufMetadata: index.metadata,
    rows(name, start, count) {
      const info = index.tensors.get(name);
      if (
        !info ||
        info.shape.length !== 2 ||
        !Number.isSafeInteger(start) ||
        !Number.isSafeInteger(count) ||
        start < 0 ||
        count < 1 ||
        start + count > info.shape[0]!
      ) {
        throw new RunntimeError('INVALID_ARGUMENT', `Invalid GGUF row range for ${name}`);
      }
      const shape = [count, info.shape[1]!];
      const rowBytes = tensorBytes(info.type, [info.shape[1]!]);
      const begin = info.begin + start * rowBytes;
      return ggufTensor({ ...info, shape, begin, end: begin + count * rowBytes }, cached);
    },
    tensors,
    totalBytes: () => [...tensors.values()].reduce((sum, tensor) => sum + tensor.byteLength, 0),
  };
}
