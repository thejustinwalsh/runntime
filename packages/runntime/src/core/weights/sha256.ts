/** Incremental SHA-256 for verifying large range-readable weight files. */

const ROUND = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const INITIAL = [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const rotateRight = (value: number, count: number): number =>
  (value >>> count) | (value << (32 - count));

/** Hashes consecutive byte ranges without retaining the complete file. */
export class Sha256 {
  private readonly state = Uint32Array.from(INITIAL);
  private readonly pending = new Uint8Array(64);
  private readonly words = new Uint32Array(64);
  private pendingBytes = 0;
  private byteLength = 0n;
  private finalized = false;
  private result: string | undefined;

  /** Adds bytes to this hash in source order. */
  update(bytes: Uint8Array): void {
    if (this.finalized) throw new Error('SHA-256 has already been finalized');
    this.byteLength += BigInt(bytes.byteLength);
    let offset = 0;

    if (this.pendingBytes > 0) {
      const count = Math.min(64 - this.pendingBytes, bytes.byteLength);
      this.pending.set(bytes.subarray(0, count), this.pendingBytes);
      this.pendingBytes += count;
      offset += count;
      if (this.pendingBytes === 64) {
        this.compress(new DataView(this.pending.buffer));
        this.pendingBytes = 0;
      }
    }

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    while (offset + 64 <= bytes.byteLength) {
      this.compress(view, offset);
      offset += 64;
    }
    if (offset < bytes.byteLength) {
      this.pending.set(bytes.subarray(offset), 0);
      this.pendingBytes = bytes.byteLength - offset;
    }
  }

  /** Finalizes the hash and returns 64 lowercase hexadecimal digits. */
  hex(): string {
    if (this.result) return this.result;
    const final = new Uint8Array(this.pendingBytes < 56 ? 64 : 128);
    final.set(this.pending.subarray(0, this.pendingBytes));
    final[this.pendingBytes] = 0x80;
    new DataView(final.buffer).setBigUint64(final.byteLength - 8, this.byteLength * 8n, false);
    const view = new DataView(final.buffer);
    for (let offset = 0; offset < final.byteLength; offset += 64) this.compress(view, offset);
    this.finalized = true;
    this.result = Array.from(this.state, (word) => word.toString(16).padStart(8, '0')).join('');
    return this.result;
  }

  private compress(view: DataView, offset = 0): void {
    const w = this.words;
    for (let index = 0; index < 16; index++) w[index] = view.getUint32(offset + index * 4, false);
    for (let index = 16; index < 64; index++) {
      const previous15 = w[index - 15]!;
      const previous2 = w[index - 2]!;
      const sigma0 = rotateRight(previous15, 7) ^ rotateRight(previous15, 18) ^ (previous15 >>> 3);
      const sigma1 = rotateRight(previous2, 17) ^ rotateRight(previous2, 19) ^ (previous2 >>> 10);
      w[index] = (w[index - 16]! + sigma0 + w[index - 7]! + sigma1) >>> 0;
    }

    let a = this.state[0]!;
    let b = this.state[1]!;
    let c = this.state[2]!;
    let d = this.state[3]!;
    let e = this.state[4]!;
    let f = this.state[5]!;
    let g = this.state[6]!;
    let h = this.state[7]!;
    for (let index = 0; index < 64; index++) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const choose = (e & f) ^ (~e & g);
      const first = (h + sum1 + choose + ROUND[index]! + w[index]!) >>> 0;
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const second = (sum0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + first) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (first + second) >>> 0;
    }
    this.state[0] = (this.state[0]! + a) >>> 0;
    this.state[1] = (this.state[1]! + b) >>> 0;
    this.state[2] = (this.state[2]! + c) >>> 0;
    this.state[3] = (this.state[3]! + d) >>> 0;
    this.state[4] = (this.state[4]! + e) >>> 0;
    this.state[5] = (this.state[5]! + f) >>> 0;
    this.state[6] = (this.state[6]! + g) >>> 0;
    this.state[7] = (this.state[7]! + h) >>> 0;
  }
}
