import { describe, expect, it } from 'vitest';
import { createLfm3Audio, type RangeSource } from '../../../../src/zoo/index.ts';

describe('public LFM3-Omni audio loader', () => {
  it('honors a pre-aborted load before reading either pinned model source', async () => {
    let reads = 0;
    const source: RangeSource = {
      async read(begin, end) {
        reads++;
        return new Uint8Array(end - begin);
      },
    };
    const controller = new AbortController();
    controller.abort('cancel before load');

    await expect(
      createLfm3Audio(
        {
          format: 'gguf',
          modelPath: source,
          projectorPath: source,
          tokenizerPath: 'unused',
        },
        { signal: controller.signal },
      ),
    ).rejects.toMatchObject({ code: 'LOAD_ABORTED' });
    expect(reads).toBe(0);
  });
});
