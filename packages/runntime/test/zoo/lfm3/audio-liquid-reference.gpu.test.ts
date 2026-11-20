/** Runs the pinned Liquid engine comparisons (audio encoder, depth codes, codec PCM) from local
 *  files only: the engine bundle, the GGUF pair and the regenerated frozen clips are all same-origin. */

import { afterAll, it } from 'vitest';
import { MODE, installNetworkGuard, report } from './audio/qualification/harness.ts';

if (MODE === 'liquid-reference') {
  const guard = installNetworkGuard();
  afterAll(() => {
    report('network', {
      mode: 'audio-liquid-reference',
      sameOriginRequests: guard.requests.length,
      blocked: guard.blocked,
    });
  });
  await import('./audio-reference.gpu.test.ts');
} else {
  it.skip('needs VITE_LFM3_QUALIFICATION_MODE=liquid-reference');
}
