import { describe, expect, it } from 'vitest';
import { RunntimeError } from '../../../src/core/index.ts';
import { LFM3 } from '../../../src/zoo/lfm3/model.ts';
import { createLfm3Session, type Lfm3SessionRuntime } from '../../../src/zoo/lfm3/session.ts';
import type { Lfm3Message, Lfm3Tokenizer } from '../../../src/zoo/lfm3/tokenizer.ts';

function fixture(
  options: {
    promptLength?: number;
    tokens?: number[];
    decode?: (ids: readonly number[]) => string;
  } = {},
) {
  const tokens = options.tokens ?? [7, 8, 124_900];
  let calls = 0;
  const released: number[] = [];
  let disposed = false;
  const tokenizer: Lfm3Tokenizer = {
    bosId: 1,
    stopIds: new Set([124_900, 125_014]),
    encodeChat(_messages: readonly Lfm3Message[]) {
      return Array.from({ length: options.promptLength ?? 2 }, (_, i) => i + 1);
    },
    decode(ids) {
      return (
        options.decode?.(ids) ??
        ids
          .filter((id) => ![124_900, 125_014].includes(id))
          .map((id) => (id === 7 ? 'A' : 'B'))
          .join('')
      );
    },
  };
  const runtime: Lfm3SessionRuntime<{ id: number }> = {
    async forward(_ids, _cache, signal) {
      if (signal.aborted) throw new Error('unexpected call after abort');
      const target = tokens[calls++] ?? 0;
      const logits = new Float32Array(LFM3.vocab).fill(-1);
      logits[target] = 1;
      return { logits, cache: { id: calls } };
    },
    release(cache) {
      released.push(cache.id);
    },
    trim() {},
    dispose() {
      disposed = true;
    },
  };
  return {
    session: createLfm3Session(tokenizer, runtime),
    released,
    get calls() {
      return calls;
    },
    get disposed() {
      return disposed;
    },
  };
}

describe('LFM3 streaming session', () => {
  it('streams decoded text in order, stops on EOS and releases replaced cache state', async () => {
    const f = fixture();
    const generation = f.session.generate([{ role: 'user', content: 'hello' }]);
    const updates: string[] = [];
    for await (const update of generation) updates.push(update);
    await expect(generation.result).resolves.toEqual({
      text: 'AB',
      inputTokens: 2,
      completionTokens: 3,
      finishReason: 'stop',
    });
    expect(updates).toEqual(['A', 'B']);
    expect(f.calls).toBe(3);
    expect(f.released).toEqual([1, 2, 3]);
  });

  it('reports an explicit length finish at the generation and context bounds', async () => {
    const limited = fixture({ tokens: [7, 8] });
    const result = limited.session.generate([{ role: 'user', content: 'hi' }], {
      maxNewTokens: 1,
    });
    await expect(result.result).resolves.toMatchObject({
      text: 'A',
      completionTokens: 1,
      finishReason: 'length',
    });
    await limited.session.dispose();

    const exhausted = fixture({ promptLength: LFM3.maxContext });
    const atLimit = exhausted.session.generate([{ role: 'user', content: 'long' }]);
    await expect(atLimit.result).resolves.toMatchObject({
      inputTokens: LFM3.maxContext,
      completionTokens: 0,
      finishReason: 'length',
    });
    expect(exhausted.calls).toBe(0);
    const oversized = fixture({ promptLength: LFM3.maxContext + 1 });
    expect(() => oversized.session.generate([{ role: 'user', content: 'too long' }])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
    await exhausted.session.dispose();
    await oversized.session.dispose();
  });

  it('holds an incomplete trailing character until decoding makes it stable', async () => {
    const f = fixture({
      tokens: [10, 11, 124_900],
      decode(ids) {
        return ids.length === 1 ? 'A�' : 'Aø';
      },
    });
    const generation = f.session.generate([{ role: 'user', content: 'utf8' }]);
    const updates: string[] = [];
    for await (const update of generation) updates.push(update);
    await expect(generation.result).resolves.toMatchObject({ text: 'Aø' });
    expect(updates.join('')).toBe('Aø');
    await f.session.dispose();
  });

  it('cancels and resets queued work before it starts, then remains usable', async () => {
    const f = fixture();
    const cancelled = f.session.generate([{ role: 'user', content: 'first' }]);
    cancelled.cancel();
    await expect(cancelled.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
    await f.session.reset();
    expect(f.calls).toBe(0);
    const fresh = f.session.generate([{ role: 'user', content: 'second' }], { maxNewTokens: 1 });
    await expect(fresh.result).resolves.toMatchObject({ text: 'A' });
    await f.session.dispose();
    expect(f.disposed).toBe(true);
    expect(() => f.session.generate([{ role: 'user', content: 'after disposal' }])).toThrow(
      expect.objectContaining({ code: 'RESOURCE_DISPOSED' }),
    );
  });

  it('resets an in-flight generation and serializes the next request after cancellation', async () => {
    let started!: () => void;
    const entered = new Promise<void>((resolve) => {
      started = resolve;
    });
    let calls = 0;
    const runtime: Lfm3SessionRuntime<{ id: number }> = {
      async forward(_ids, _cache, signal) {
        calls++;
        if (calls === 1) {
          started();
          await new Promise<never>((_resolve, reject) => {
            signal.addEventListener(
              'abort',
              () => reject(new RunntimeError('EXECUTION_FAILED', 'LFM3 generation aborted')),
              { once: true },
            );
          });
        }
        const logits = new Float32Array(LFM3.vocab).fill(-1);
        logits[7] = 1;
        return { logits, cache: { id: calls } };
      },
      release() {},
      trim() {},
      dispose() {},
    };
    const session = createLfm3Session(
      {
        bosId: 1,
        stopIds: new Set([124_900]),
        encodeChat: () => [1],
        decode: (ids) => (ids.includes(7) ? 'A' : ''),
      },
      runtime,
    );
    const interrupted = session.generate([{ role: 'user', content: 'first' }]);
    await entered;
    await session.reset();
    await expect(interrupted.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
    const next = session.generate([{ role: 'user', content: 'after reset' }], { maxNewTokens: 1 });
    await expect(next.result).resolves.toMatchObject({ text: 'A', completionTokens: 1 });
    await session.dispose();
  });

  it('propagates runtime failures to both the stream and result promise', async () => {
    const runtime = {
      forward: async () => {
        throw new Error('GPU readback failed');
      },
      release: () => {},
      trim: () => {},
      dispose: () => {},
    } satisfies Lfm3SessionRuntime<never>;
    const session = createLfm3Session(
      {
        bosId: 1,
        stopIds: new Set([124_900]),
        encodeChat: () => [1],
        decode: () => '',
      },
      runtime,
    );
    const generation = session.generate([{ role: 'user', content: 'hello' }]);
    await expect(generation.result).rejects.toThrow('GPU readback failed');
    await expect(generation[Symbol.asyncIterator]().next()).rejects.toThrow('GPU readback failed');
    await session.dispose();
  });

  it.each(['invalid logits', 'GPU out-of-memory'] as const)(
    'propagates %s to the stream and result without poisoning an independent session',
    async (failure) => {
      const runtime = {
        async forward() {
          if (failure === 'invalid logits') {
            return { logits: new Float32Array(1), cache: { id: 1 } };
          }
          throw new RunntimeError('EXECUTION_FAILED', 'GPU out-of-memory');
        },
        release() {},
        trim() {},
        dispose() {},
      } satisfies Lfm3SessionRuntime<{ id: number }>;
      const session = createLfm3Session(
        {
          bosId: 1,
          stopIds: new Set([124_900]),
          encodeChat: () => [1],
          decode: () => '',
        },
        runtime,
      );
      const failed = session.generate([{ role: 'user', content: 'fail' }]);
      await expect(failed.result).rejects.toMatchObject({ code: 'EXECUTION_FAILED' });
      await expect(failed[Symbol.asyncIterator]().next()).rejects.toMatchObject({
        code: 'EXECUTION_FAILED',
      });
      await session.dispose();

      const independent = fixture({ tokens: [7, 124_900] });
      const fresh = independent.session.generate([{ role: 'user', content: 'remain usable' }]);
      await expect(fresh.result).resolves.toMatchObject({ text: 'A', finishReason: 'stop' });
      await independent.session.dispose();
    },
  );
});
