import { describe, expect, it } from 'vitest';
import { LFM3_Q4_K_M_SOURCE } from '../../../src/zoo/lfm3/gguf.ts';
import liquidReference from './liquid-reference.json';

describe('LFM3_Q4_K_M_SOURCE', () => {
  it('names a file at one pinned revision, addressed by that revision', () => {
    expect(LFM3_Q4_K_M_SOURCE.revision).toMatch(/^[a-f0-9]{40}$/);
    expect(LFM3_Q4_K_M_SOURCE.url).toBe(
      `https://huggingface.co/${LFM3_Q4_K_M_SOURCE.repository}/resolve/${LFM3_Q4_K_M_SOURCE.revision}/${LFM3_Q4_K_M_SOURCE.file}`,
    );
  });

  it('carries the 64-digit lowercase digest and exact size fromGguf checks', () => {
    expect(LFM3_Q4_K_M_SOURCE.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(Number.isSafeInteger(LFM3_Q4_K_M_SOURCE.bytes)).toBe(true);
    expect(LFM3_Q4_K_M_SOURCE.bytes).toBeGreaterThan(24);
  });

  it('is the file the Liquid reference vectors were produced from', () => {
    expect(liquidReference.model).toMatchObject({
      repository: LFM3_Q4_K_M_SOURCE.repository,
      revision: LFM3_Q4_K_M_SOURCE.revision,
      file: LFM3_Q4_K_M_SOURCE.file,
      sha256: LFM3_Q4_K_M_SOURCE.sha256,
    });
  });
});
