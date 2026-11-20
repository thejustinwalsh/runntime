/** Runs the audio qualification from local weights and prints one JSON result.
 *  Usage: node test/zoo/lfm3/audio-qualification.mjs [--mode <name>]... [--voices a,b] [--seeds 1,2] [--out file]
 *  Modes: audio-roundtrip, audio-streaming, audio-input, audio-memory (default: these four) and
 *  audio-liquid-reference, which compares against the pinned Liquid engine and runs only when named.
 *  Run from packages/runntime. Exits non-zero when any mode fails or a request would leave the machine. */

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  createReadStream,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';

const DEFAULT_MODES = ['audio-roundtrip', 'audio-streaming', 'audio-input', 'audio-memory'];
const MODES = [...DEFAULT_MODES, 'audio-liquid-reference'];
const LIQUID_ENGINE = {
  file: 'lfm3-engine.js',
  instrumentedFile: 'lfm3-engine-instrumented.js',
  // The comparison needs the engine's inner model; this getter is the only change made to the bundle.
  accessor: 'get generation_config(){return this.#e.generation_config}',
  instrumentation: 'get reference(){return this.#e}',
  sha256: '77821949412f5dd6d60e37f22415f0e71454e65f6a73928529c6fbec0f2cf2cb',
};
// The frozen clips: macOS Samantha at 175 wpm, converted to 16 kHz mono PCM16. The same macOS
// build regenerates them byte for byte, so the hashes pin them without storing audio.
const FROZEN_CLIPS = [
  {
    file: 'quick-brown-fox.wav',
    text: 'The quick brown fox jumps over the lazy dog.',
    sha256: '0497e70d431e9d827c10bbaff120c9cbabf0834a747107f83cb161dc1f759880',
  },
  {
    file: 'turn-left-next-street.wav',
    text: 'Please turn left at the next street.',
    sha256: 'f9ce57f4e548e339e944ba87860cd35a545db46e5eafd69f3692ede19f6cb054',
  },
];
const WEIGHTS = {
  model: 'LFM3-Omni-2B-Q4_K_M.gguf',
  projector: 'mmproj-LFM3-Omni-2B-Q8_0.gguf',
  tokenizer: 'lfm3-tokenizer.json',
};
const MARKER = 'LFM3_AUDIO_QUALIFICATION ';
// The Liquid comparison test prints its own line; it is carried under the name `operator-reference`.
const OPERATOR_MARKER = 'LFM3_AUDIO_OPERATOR_REFERENCE ';

const { values } = parseArgs({
  options: {
    mode: { type: 'string', multiple: true },
    voices: { type: 'string' },
    seeds: { type: 'string' },
    out: { type: 'string' },
  },
});
const modes = values.mode?.length ? values.mode : DEFAULT_MODES;
for (const mode of modes) {
  if (!MODES.includes(mode))
    throw new Error(`unknown mode ${mode}; expected one of ${MODES.join(', ')}`);
}

function sha256(path) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    createReadStream(path)
      .on('data', (chunk) => hash.update(chunk))
      .on('end', () => resolve(hash.digest('hex')))
      .on('error', reject);
  });
}

const checksums = {};
for (const [role, file] of Object.entries(WEIGHTS)) {
  checksums[role] = { file, sha256: await sha256(join('test/.weights', file)) };
}
const git = (...args) => spawnSync('git', args, { encoding: 'utf8' }).stdout.trim();

const FIXTURES = 'test/.weights/audio-fixtures';
async function prepareLiquid() {
  const engine = await sha256(join('test/.weights', LIQUID_ENGINE.file));
  if (engine !== LIQUID_ENGINE.sha256) {
    throw new Error(
      `${LIQUID_ENGINE.file} is ${engine}, expected the pinned ${LIQUID_ENGINE.sha256}`,
    );
  }
  const source = readFileSync(join('test/.weights', LIQUID_ENGINE.file), 'utf8');
  if (source.split(LIQUID_ENGINE.accessor).length !== 2) {
    throw new Error(
      'the pinned Liquid engine does not contain the reference accessor location once',
    );
  }
  writeFileSync(
    join('test/.weights', LIQUID_ENGINE.instrumentedFile),
    source.replace(
      LIQUID_ENGINE.accessor,
      `${LIQUID_ENGINE.instrumentation}${LIQUID_ENGINE.accessor}`,
    ),
  );
  mkdirSync(FIXTURES, { recursive: true });
  for (const clip of FROZEN_CLIPS) {
    const path = join(FIXTURES, clip.file);
    if (!existsSync(path)) {
      const aiff = join(FIXTURES, `${clip.file}.aiff`);
      for (const [command, args] of [
        ['say', ['-v', 'Samantha', '-r', '175', '-o', aiff, clip.text]],
        ['afconvert', ['-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1', aiff, path]],
      ]) {
        const run = spawnSync(command, args, { encoding: 'utf8' });
        if (run.status !== 0) throw new Error(`${command} failed: ${run.stderr}`);
      }
      rmSync(aiff);
    }
    const actual = await sha256(path);
    if (actual !== clip.sha256) {
      throw new Error(
        `${clip.file} is ${actual}; the frozen clip is ${clip.sha256}. Regenerate it on the macOS build that produced it.`,
      );
    }
  }
  return {
    engine: {
      file: LIQUID_ENGINE.file,
      sha256: engine,
      instrumentation: LIQUID_ENGINE.instrumentation,
      instrumentationSha256: createHash('sha256')
        .update(LIQUID_ENGINE.instrumentation)
        .digest('hex'),
    },
    clips: FROZEN_CLIPS,
  };
}
const liquidInputs = modes.includes('audio-liquid-reference') ? await prepareLiquid() : undefined;

const results = [];
for (const mode of modes) {
  const dir = mkdtempSync(join(tmpdir(), 'lfm3-audio-'));
  const outputFile = join(dir, 'vitest.json');
  const started = Date.now();
  const run = spawnSync(
    'pnpm',
    [
      'exec',
      'vitest',
      'run',
      '--project',
      'gpu',
      mode === 'audio-liquid-reference'
        ? 'test/zoo/lfm3/audio-liquid-reference.gpu.test.ts'
        : 'test/zoo/lfm3/audio-qualification.gpu.test.ts',
      '--reporter=default',
      '--reporter=json',
      `--outputFile.json=${outputFile}`,
    ],
    {
      encoding: 'utf8',
      maxBuffer: 1 << 28,
      env: {
        ...process.env,
        VITE_LFM3_QUALIFICATION_MODE: mode === 'audio-liquid-reference' ? 'liquid-reference' : mode,
        VITE_LFM3_LIQUID_ENGINE_URL: `/test/.weights/${LIQUID_ENGINE.instrumentedFile}`,
        VITE_LFM3_LIQUID_MODEL_URL: `/test/.weights/${WEIGHTS.model}`,
        VITE_LFM3_LIQUID_PROJECTOR_URL: `/test/.weights/${WEIGHTS.projector}`,
        VITE_LFM3_LIQUID_AUDIO_INPUT_BASE_URL: '/test/.weights/audio-fixtures',
        VITE_LFM3_MODEL_URL: `/test/.weights/${WEIGHTS.model}`,
        VITE_LFM3_PROJECTOR_URL: `/test/.weights/${WEIGHTS.projector}`,
        VITE_LFM3_TOKENIZER_URL: `/test/.weights/${WEIGHTS.tokenizer}`,
        ...(values.voices ? { VITE_LFM3_AUDIO_VOICES: values.voices } : {}),
        ...(values.seeds ? { VITE_LFM3_AUDIO_SEEDS: values.seeds } : {}),
      },
    },
  );
  const log = `${run.stdout}\n${run.stderr}`;
  const reports = [];
  for (const line of log.split('\n')) {
    const at = line.indexOf(MARKER);
    if (at >= 0) reports.push(JSON.parse(line.slice(at + MARKER.length)));
    const operator = line.indexOf(OPERATOR_MARKER);
    if (operator >= 0) {
      reports.push({
        name: 'operator-reference',
        ...JSON.parse(line.slice(operator + OPERATOR_MARKER.length)),
      });
    }
  }
  let tests = [];
  try {
    const json = JSON.parse(readFileSync(outputFile, 'utf8'));
    tests = json.testResults.flatMap((file) =>
      file.assertionResults
        .filter((test) => test.status !== 'skipped')
        .map((test) => ({
          name: test.fullName,
          status: test.status,
          durationMs: test.duration,
          failure: test.failureMessages?.join('\n') || undefined,
        })),
    );
  } catch {
    // No reporter output means vitest never started a test; the exit code and log tail say why.
  }
  const blocked = reports
    .filter((entry) => entry.name === 'network')
    .flatMap((entry) => entry.blocked ?? []);
  const passed =
    tests.some((test) => test.status === 'passed') &&
    tests.every((test) => test.status !== 'failed') &&
    blocked.length === 0 &&
    run.status === 0;
  results.push({
    mode,
    passed,
    exitCode: run.status,
    durationMs: Date.now() - started,
    tests,
    reports,
    blockedRequests: blocked,
    logTail: passed ? undefined : log.split('\n').slice(-60).join('\n'),
  });
  writeFileSync(join(dir, 'console.log'), log);
}

const summary = {
  schemaVersion: 1,
  finishedAt: new Date().toISOString(),
  commit: git('rev-parse', 'HEAD'),
  branch: git('rev-parse', '--abbrev-ref', 'HEAD'),
  dirty: git('status', '--porcelain') !== '',
  node: process.version,
  weights: checksums,
  liquid: liquidInputs,
  selection: { voices: values.voices ?? 'all', seeds: values.seeds ?? '1234' },
  ok: results.every((result) => result.passed),
  modes: results,
};
const text = JSON.stringify(summary, null, 2);
if (values.out) writeFileSync(values.out, text);
console.log(text);
process.exit(summary.ok ? 0 : 1);
