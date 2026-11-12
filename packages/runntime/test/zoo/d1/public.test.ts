import { describe, expect, it } from 'vitest';
import { createD1Decider, models } from '../../../src/zoo/index.ts';

describe('public D1 zoo entry', () => {
  it('requires caller-provided weights for the unhosted Q8 model', async () => {
    expect('modelPath' in models.d1.D1_OMNI_600M.Q8).toBe(false);
    await expect(createD1Decider(models.d1.D1_OMNI_600M.Q8)).rejects.toMatchObject({
      code: 'INVALID_ARGUMENT',
    });
  });
});
