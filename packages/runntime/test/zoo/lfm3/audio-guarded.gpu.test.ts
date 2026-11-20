/** Runs an existing opt-in audio test with the network guard on, so it can only reach this origin:
 *  the pinned Liquid engine bundle, the GGUF pair, the tokenizer and the regenerated frozen clips
 *  are all served locally. VITE_LFM3_GUARDED_TEST picks the test:
 *  `audio-reference`  encoder, depth codes and codec PCM against Liquid's engine (mode liquid-reference),
 *  `liquid-reference` Liquid's own transcription and speech round trip on the frozen clips (mode liquid-reference),
 *  `public-audio`     the public runner on the frozen clips, depth 1 and 3, abort recovery (mode audio). */

import { afterAll, it } from 'vitest';
import { MODE, installNetworkGuard, report } from './audio/qualification/harness.ts';

const TEST = import.meta.env.VITE_LFM3_GUARDED_TEST;
const ENABLED =
  (MODE === 'liquid-reference' && (TEST === 'audio-reference' || TEST === 'liquid-reference')) ||
  (MODE === 'audio' && TEST === 'public-audio');

if (ENABLED) {
  const guard = installNetworkGuard();
  afterAll(() => {
    report('network', {
      mode: TEST,
      sameOriginRequests: guard.requests.length,
      blocked: guard.blocked,
    });
  });
  if (TEST === 'audio-reference') await import('./audio-reference.gpu.test.ts');
  else if (TEST === 'liquid-reference') await import('./liquid-reference.gpu.test.ts');
  else await import('./public-audio.gpu.test.ts');
} else {
  it.skip('needs VITE_LFM3_QUALIFICATION_MODE and VITE_LFM3_GUARDED_TEST');
}
