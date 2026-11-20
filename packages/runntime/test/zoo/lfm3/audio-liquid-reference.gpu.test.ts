/** Runs the pinned Liquid engine comparisons from local files only: the engine bundle, the GGUF pair
 *  and the regenerated frozen clips are all same-origin. VITE_LFM3_LIQUID_TEST picks the comparison:
 *  `audio-reference` (encoder, depth codes and codec PCM against this implementation) or
 *  `liquid-reference` (Liquid's own transcription and speech round trip on the frozen clips). */

import { afterAll, it } from 'vitest';
import { MODE, installNetworkGuard, report } from './audio/qualification/harness.ts';

const TEST = import.meta.env.VITE_LFM3_LIQUID_TEST;

if (MODE === 'liquid-reference' && (TEST === 'audio-reference' || TEST === 'liquid-reference')) {
  const guard = installNetworkGuard();
  afterAll(() => {
    report('network', {
      mode: `audio-${TEST}`,
      sameOriginRequests: guard.requests.length,
      blocked: guard.blocked,
    });
  });
  if (TEST === 'audio-reference') await import('./audio-reference.gpu.test.ts');
  else await import('./liquid-reference.gpu.test.ts');
} else {
  it.skip('needs VITE_LFM3_QUALIFICATION_MODE=liquid-reference and VITE_LFM3_LIQUID_TEST');
}
