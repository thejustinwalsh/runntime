/** Opt-in full-checkpoint device-loss and fresh-session recovery on WebGPU. */

import tgpu from 'typegpu';
import { expect, it } from 'vitest';
import { createLfm3Chat, initRunntime, models } from '../../../src/zoo/index.ts';
import { gpu, liveBuffers } from '../../gpu.ts';

declare global {
  interface ImportMetaEnv {
    readonly VITE_LFM3_QUALIFICATION_MODE?: string;
    readonly VITE_LFM3_MODEL_URL?: string;
    readonly VITE_LFM3_TOKENIZER_URL?: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const MODE = import.meta.env.VITE_LFM3_QUALIFICATION_MODE;
const CONFIG = {
  ...models.lfm3.LFM3_OMNI_2B.DEFAULT,
  ...(import.meta.env.VITE_LFM3_MODEL_URL
    ? { modelPath: import.meta.env.VITE_LFM3_MODEL_URL }
    : {}),
  ...(import.meta.env.VITE_LFM3_TOKENIZER_URL
    ? { tokenizerPath: import.meta.env.VITE_LFM3_TOKENIZER_URL }
    : {}),
};

it.skipIf(MODE !== 'recovery')(
  'recovers with a new public LFM3 session after real WebGPU device loss',
  async (ctx) => {
    const { root: lostRoot } = await gpu();
    const before = liveBuffers();
    const firstLoadStarted = performance.now();
    const session = await createLfm3Chat(CONFIG);
    const firstLoadMs = performance.now() - firstLoadStarted;
    ctx.onTestFinished(() => session.dispose());

    const lost = lostRoot.device.lost;
    lostRoot.device.destroy();
    const lostInfo = await lost;
    expect(lostInfo.reason).toBe('destroyed');

    const failed = session.generate([{ role: 'user', content: 'Continue after device loss.' }], {
      maxNewTokens: 1,
    });
    await expect(failed.result).rejects.toBeDefined();
    await session.dispose();
    const afterLoss = [...liveBuffers()].filter((buffer) => !before.has(buffer));
    const lostStorage = afterLoss.filter((buffer) => (buffer.usage & GPUBufferUsage.STORAGE) !== 0);
    expect(lostStorage).toEqual([]);

    const freshRoot = await tgpu.init({
      device: { optionalFeatures: ['subgroups', 'shader-f16'] },
    });
    ctx.onTestFinished(() => freshRoot.device.destroy());
    initRunntime(freshRoot);
    const freshBefore = liveBuffers();
    const freshLoadStarted = performance.now();
    const freshSession = await createLfm3Chat(CONFIG, { warmUp: false });
    const freshLoadMs = performance.now() - freshLoadStarted;
    const generationStarted = performance.now();
    const generation = freshSession.generate([{ role: 'user', content: 'Say hello briefly.' }], {
      maxNewTokens: 4,
    });
    let streamed = '';
    for await (const delta of generation) streamed += delta;
    const result = await generation.result;
    const generationMs = performance.now() - generationStarted;
    expect(result.text.trim()).toBeTruthy();
    expect(streamed).toBe(result.text);
    await freshSession.dispose();
    const freshRemaining = [...liveBuffers()].filter((buffer) => !freshBefore.has(buffer));
    const freshStorage = freshRemaining.filter(
      (buffer) => (buffer.usage & GPUBufferUsage.STORAGE) !== 0,
    );
    expect(freshStorage).toEqual([]);
    freshRoot.device.destroy();

    console.info(
      `LFM3_DEVICE_RECOVERY ${JSON.stringify({
        checkpointRevision: '21d75123f86c1f7ffb4bef1ea202710c4e7ebe67',
        deviceLossReason: lostInfo.reason,
        firstLoadMs: +firstLoadMs.toFixed(2),
        failedSessionError: 'generation rejected after GPUDevice.destroy()',
        storageBuffersAfterLostSessionDispose: lostStorage.length,
        rootCacheBuffersAfterLostSessionDispose: afterLoss.length - lostStorage.length,
        freshLoadMs: +freshLoadMs.toFixed(2),
        freshGenerationMs: +generationMs.toFixed(2),
        freshText: result.text,
        freshStorageBuffersAfterDispose: freshStorage.length,
        freshRootCacheBuffersAfterDispose: freshRemaining.length - freshStorage.length,
      })}`,
    );
  },
  1_200_000,
);
