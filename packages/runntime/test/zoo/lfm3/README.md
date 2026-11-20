# LFM3-Omni audio qualification

GPU tests for speech in and speech out, run from local weights with no network. They are not part
of CI: they need a WebGPU device with `subgroups` and `shader-f16`, about 2 GB of GPU memory, and the
published Q4_K_M and Q8_0 files.

## Setup

Once per machine: `pnpm --filter runntime exec playwright install chromium`. Put these in
`packages/runntime/test/.weights/` (git-excluded; symlinks are fine):

| File                                                        | Source                                                                                                   |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `LFM3-Omni-2B-Q4_K_M.gguf`, `mmproj-LFM3-Omni-2B-Q8_0.gguf` | `LiquidAI/LFM3-Omni-2B-GGUF` at `3d29af2ad166f8299c2d820937e47dc0add20ffc`                               |
| `lfm3-tokenizer.json`                                       | `LiquidAI/LFM3-Omni-2B` at `21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`                                    |
| `lfm3-engine.js` (only for `audio-liquid-reference`)        | Liquid's WebGPU engine bundle, sha256 `77821949412f5dd6d60e37f22415f0e71454e65f6a73928529c6fbec0f2cf2cb` |

## Run

From `packages/runntime`:

```sh
pnpm test:audio                                   # the four modes below, one JSON result on stdout
pnpm test:audio --mode audio-roundtrip            # one mode
pnpm test:audio --mode audio-roundtrip --voices american-female --seeds 1234,7 --out result.json
pnpm test:audio --mode audio-liquid-reference     # against the pinned Liquid engine, named only
pnpm test:audio --mode audio-liquid-roundtrip     # Liquid's own speech round trip, the baseline for ours
```

The script hashes the weights, runs each mode in headless Chromium, and prints JSON: per-test status
and duration, the measurements each test reports (word error rate, timings, GPU bytes), adapter
info, the commit and the weight checksums. It exits non-zero when a test fails or when any request
left the page's origin (the page's `fetch` and `XMLHttpRequest` refuse other origins).

| Mode                     | What it checks                                                                                                                                                                                                   |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `audio-roundtrip`        | `speak` in four voices over eight fixed sentences, resampled to 16 kHz and `transcribe`d back; word error rate against the sentence.                                                                             |
| `audio-streaming`        | Chunks add up to the result, one seed repeats exactly, `finishReason: 'length'`, early `break`, `cancel()`, `AbortSignal`, a failed queued request, invalid options, `dispose` mid-stream.                       |
| `audio-input`            | Padded-graph boundaries, the 32 s maximum and one sample more, silence and very short clips, `resampleAudio` and `decodeAudio` at 24 and 48 kHz, a spoken question answered through `transcribe` and `generate`. |
| `audio-memory`           | GPU bytes over twelve repeated speak and transcribe cycles, and no storage buffer left after `dispose`.                                                                                                          |
| `audio-liquid-reference` | The audio encoder, the depth decoder's codes and one codec frame against Liquid's engine on the same inputs.                                                                                                     |
| `audio-liquid-roundtrip` | Liquid's own transcription of the frozen clips and its speech recovered by its own transcriber: the baseline the native round trip is read against.                                                              |

`speak` is seeded, so the speech inputs are produced by the model itself: no recorded audio is stored.
`audio-liquid-reference` needs two frozen 16 kHz clips; the script regenerates them on macOS with
`say -v Samantha -r 175` and `afconvert` and refuses a clip whose hash differs. Its engine is the
pinned bundle with one getter added (`get reference(){return this.#e}`) to reach the inner model.

Frontend tests on synthesised signals (`audio/qualification/frontendSignals.test.ts`) and the word
error rate itself (`wer.test.ts`) are Node tests and run with `pnpm --filter runntime test:unit`.

Limits live in `audio/qualification/thresholds.ts`, each next to the measurement it came from.
