/** Builds the package tarball and runs its public LFM3 loader on real WebGPU. */

import {
  copyFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const packageDirectory = resolve(scriptDirectory, '../../..');
const repositoryDirectory = resolve(packageDirectory, '../..');
const modelUrl =
  process.env.LFM3_MODEL_URL ?? process.env.LFM3_BASE_MODEL_URL ?? process.env.LFM3_Q4_MODEL_URL;
const tokenizerUrl = process.env.LFM3_BASE_TOKENIZER_URL;
const reportPath = process.env.LFM3_BASE_VITEST_REPORT;
const qualificationMode = process.env.LFM3_BASE_QUALIFICATION_MODE ?? 'base';
const liquidEnginePath = process.env.LFM3_LIQUID_ENGINE_PATH;
const liquidReferenceOutput = process.env.LFM3_LIQUID_REFERENCE_OUTPUT;
const liquidModelUrl = process.env.LFM3_LIQUID_MODEL_URL ?? modelUrl;
const liquidProjectorUrl = process.env.LFM3_LIQUID_PROJECTOR_URL;
const liquidAudioInputBaseUrl = process.env.LFM3_LIQUID_AUDIO_INPUT_BASE_URL;
const liquidAudioOutputDirectory = process.env.LFM3_LIQUID_AUDIO_OUTPUT_DIR;
const packedTestFile = process.env.LFM3_PACKED_TEST_FILE;
const keepTemporaryDirectory = process.env.LFM3_KEEP_PACKED_TEMP === '1';

const temporaryDirectory = mkdtempSync(join(tmpdir(), 'runntime-packed-lfm3-'));

function run(command, args, cwd, env = process.env) {
  const result = spawnSync(command, args, { cwd, env, stdio: 'inherit' });
  if (result.status !== 0)
    throw new Error(`${command} ${args.join(' ')} failed (${result.status})`);
}

function linkDependencies(source, destination) {
  for (const name of readdirSync(source)) {
    if (name.startsWith('.')) continue;
    const from = join(source, name);
    const to = join(destination, name);
    if (name.startsWith('@') && lstatSync(from).isDirectory()) {
      mkdirSync(to);
      for (const scopedName of readdirSync(from)) {
        symlinkSync(resolve(from, scopedName), join(to, scopedName), 'dir');
      }
    } else {
      symlinkSync(resolve(from), to, 'dir');
    }
  }
}

try {
  const packDirectory = join(temporaryDirectory, 'pack');
  const consumerDirectory = join(temporaryDirectory, 'consumer');
  mkdirSync(packDirectory);
  mkdirSync(consumerDirectory);
  run('pnpm', ['pack', '--pack-destination', packDirectory], packageDirectory);

  const tarballs = readdirSync(packDirectory).filter((name) => name.endsWith('.tgz'));
  if (tarballs.length !== 1)
    throw new Error(`expected one package tarball, found ${tarballs.length}`);
  run(
    'tar',
    ['-xzf', join(packDirectory, tarballs[0]), '-C', consumerDirectory],
    consumerDirectory,
  );

  const nodeModules = join(consumerDirectory, 'node_modules');
  mkdirSync(nodeModules);
  renameSync(join(consumerDirectory, 'package'), join(nodeModules, 'runntime'));
  linkDependencies(join(packageDirectory, 'node_modules'), nodeModules);

  const testDirectory = join(consumerDirectory, 'test', 'zoo', 'lfm3');
  mkdirSync(testDirectory, { recursive: true });
  copyFileSync(
    join(packageDirectory, 'test/zoo/lfm3/liquid-reference.json'),
    join(testDirectory, 'liquid-reference.json'),
  );
  const qualificationSource = readFileSync(
    join(packageDirectory, 'test/zoo/lfm3/public.gpu.test.ts'),
    'utf8',
  ).replace("from '../../../src/zoo/index.ts'", "from 'runntime/zoo'");
  if (!qualificationSource.includes("from 'runntime/zoo'")) {
    throw new Error('LFM3 consumer did not switch to the packed public entry point');
  }
  writeFileSync(join(testDirectory, 'public.gpu.test.ts'), qualificationSource);
  const recoverySource = readFileSync(
    join(packageDirectory, 'test/zoo/lfm3/recovery.gpu.test.ts'),
    'utf8',
  ).replace("from '../../../src/zoo/index.ts'", "from 'runntime/zoo'");
  writeFileSync(join(testDirectory, 'recovery.gpu.test.ts'), recoverySource);
  const referenceSource = readFileSync(
    join(packageDirectory, 'test/zoo/lfm3/liquid-reference.gpu.test.ts'),
    'utf8',
  );
  writeFileSync(join(testDirectory, 'liquid-reference.gpu.test.ts'), referenceSource);
  if (qualificationMode === 'liquid-reference') {
    if (!liquidEnginePath)
      throw new Error('set LFM3_LIQUID_ENGINE_PATH to the pinned Liquid engine bundle');
    const engineBytes = readFileSync(liquidEnginePath);
    const engineDigest = createHash('sha256').update(engineBytes).digest('hex');
    const expectedEngineDigest = '77821949412f5dd6d60e37f22415f0e71454e65f6a73928529c6fbec0f2cf2cb';
    if (engineDigest !== expectedEngineDigest) {
      throw new Error(
        `Liquid engine SHA-256 ${engineDigest} does not match ${expectedEngineDigest}`,
      );
    }
    const engineSource = engineBytes.toString('utf8');
    const accessor = 'get generation_config(){return this.#e.generation_config}';
    if (engineSource.split(accessor).length !== 2) {
      throw new Error('pinned Liquid engine wrapper did not match the reference accessor location');
    }
    const instrumentation = 'get reference(){return this.#e}';
    const instrumented = engineSource.replace(accessor, `${instrumentation}${accessor}`);
    writeFileSync(join(testDirectory, 'liquid-reference-engine.js'), instrumented);
    console.info(`LFM3_LIQUID_ENGINE_SHA256 ${engineDigest}`);
    console.info(
      `LFM3_LIQUID_ENGINE_INSTRUMENTATION_SHA256 ${createHash('sha256').update(instrumentation).digest('hex')}`,
    );
  }
  writeFileSync(
    join(consumerDirectory, 'test/gpu.ts'),
    `/** Real WebGPU setup and buffer-lifetime observation for the packed consumer. */
import tgpu from 'typegpu';
import { initRunntime } from 'runntime/zoo';

const live = new Set<GPUBuffer>();
const buffersPerDevice = new WeakMap<GPUDevice, Set<GPUBuffer>>();
const devicePerBuffer = new WeakMap<GPUBuffer, GPUDevice>();
const createBuffer = GPUDevice.prototype.createBuffer;
const destroyDevice = GPUDevice.prototype.destroy;
const destroyBuffer = GPUBuffer.prototype.destroy;
GPUDevice.prototype.createBuffer = function (descriptor) {
  const buffer = createBuffer.call(this, descriptor);
  live.add(buffer);
  let buffers = buffersPerDevice.get(this);
  if (buffers === undefined) {
    buffers = new Set();
    buffersPerDevice.set(this, buffers);
  }
  buffers.add(buffer);
  devicePerBuffer.set(buffer, this);
  return buffer;
};
GPUBuffer.prototype.destroy = function () {
  live.delete(this);
  const device = devicePerBuffer.get(this);
  if (device !== undefined) buffersPerDevice.get(device)?.delete(this);
  destroyBuffer.call(this);
};
GPUDevice.prototype.destroy = function () {
  for (const buffer of buffersPerDevice.get(this) ?? []) live.delete(buffer);
  buffersPerDevice.delete(this);
  destroyDevice.call(this);
};

export function liveBuffers() { return new Set(live); }
export function buffersSince(before) {
  return [...live].filter((buffer) => !before.has(buffer)).map((buffer) => buffer.size + ' B');
}

let rootPromise;
export function gpu() {
  rootPromise ??= tgpu.init({ device: { optionalFeatures: ['subgroups'] } }).then((root) => {
    initRunntime(root);
    return { root };
  });
  return rootPromise;
}
`,
  );

  writeFileSync(
    join(consumerDirectory, 'vitest.config.ts'),
    `import { playwright } from '@vitest/browser-playwright';
import typegpu from 'unplugin-typegpu/vite';
import { defineConfig } from 'vitest/config';
import { mkdirSync, writeFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';

export default defineConfig({
  plugins: [typegpu(), {
    name: 'liquid-reference-capture',
    configureServer(server) {
      server.middlewares.use('/__liquid-reference-fixture', (request, response) => {
        if (request.method !== 'POST') { response.statusCode = 405; response.end(); return; }
        const chunks = [];
        request.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
        request.on('end', () => {
          writeFileSync(${JSON.stringify(join(temporaryDirectory, 'liquid-reference-fixture.json'))}, Buffer.concat(chunks));
          response.statusCode = 204;
          response.end();
        });
      });
      server.middlewares.use('/__liquid-audio-output', (request, response) => {
        if (request.method !== 'POST') { response.statusCode = 405; response.end(); return; }
        const filename = basename(new URL(request.url ?? '/', 'http://localhost').pathname);
        if (!/^liquid-speech-[a-z0-9-]+\\.wav$/.test(filename)) { response.statusCode = 400; response.end(); return; }
        const chunks = [];
        request.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
        request.on('end', () => {
          const output = join(${JSON.stringify(join(temporaryDirectory, 'liquid-audio'))}, filename);
          mkdirSync(dirname(output), { recursive: true });
          writeFileSync(output, Buffer.concat(chunks));
          response.statusCode = 204;
          response.end();
        });
      });
    },
  }],
  optimizeDeps: { include: ['typegpu'] },
  server: { fs: { allow: [${JSON.stringify(repositoryDirectory)}, ${JSON.stringify(temporaryDirectory)}] } },
  test: {
    include: ['test/**/*.gpu.test.ts'],
    browser: {
      enabled: true,
      headless: true,
      provider: playwright({ launchOptions: { channel: 'chromium', args: ['--enable-unsafe-webgpu'] } }),
      instances: [{ browser: 'chromium' }],
    },
  },
});
`,
  );
  writeFileSync(join(consumerDirectory, 'package.json'), JSON.stringify({ type: 'module' }));

  const env = {
    ...process.env,
    VITE_LFM3_QUALIFICATION_MODE: qualificationMode,
    ...(modelUrl ? { VITE_LFM3_MODEL_URL: modelUrl } : {}),
    ...(tokenizerUrl
      ? { VITE_LFM3_TOKENIZER_URL: tokenizerUrl }
      : {
          VITE_LFM3_TOKENIZER_URL:
            'https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/21d75123f86c1f7ffb4bef1ea202710c4e7ebe67/tokenizer.json',
        }),
    ...(qualificationMode === 'liquid-reference'
      ? {
          VITE_LFM3_LIQUID_ENGINE_URL: '/test/zoo/lfm3/liquid-reference-engine.js',
          ...(liquidModelUrl ? { VITE_LFM3_LIQUID_MODEL_URL: liquidModelUrl } : {}),
          ...(liquidProjectorUrl ? { VITE_LFM3_LIQUID_PROJECTOR_URL: liquidProjectorUrl } : {}),
          ...(liquidAudioInputBaseUrl
            ? { VITE_LFM3_LIQUID_AUDIO_INPUT_BASE_URL: liquidAudioInputBaseUrl }
            : {}),
          ...(liquidAudioOutputDirectory ? { VITE_LFM3_LIQUID_AUDIO_CAPTURE: '1' } : {}),
        }
      : {}),
    ...(liquidReferenceOutput ? { VITE_LFM3_LIQUID_REFERENCE_CAPTURE: '1' } : {}),
  };
  const vitestArgs = [
    join(nodeModules, 'vitest/vitest.mjs'),
    'run',
    '--config',
    join(consumerDirectory, 'vitest.config.ts'),
  ];
  if (reportPath) {
    vitestArgs.push(
      '--reporter=verbose',
      '--reporter=json',
      `--outputFile.json=${resolve(reportPath)}`,
    );
  }
  if (packedTestFile) vitestArgs.push(packedTestFile);
  run(process.execPath, vitestArgs, consumerDirectory, env);
  if (liquidReferenceOutput) {
    const capturedFixture = join(temporaryDirectory, 'liquid-reference-fixture.json');
    if (!existsSync(capturedFixture)) throw new Error('Liquid reference fixture was not captured');
    copyFileSync(capturedFixture, resolve(liquidReferenceOutput));
  }
  if (liquidAudioOutputDirectory) {
    const capturedAudio = join(temporaryDirectory, 'liquid-audio');
    if (!existsSync(capturedAudio)) throw new Error('Liquid speech audio was not captured');
    mkdirSync(resolve(liquidAudioOutputDirectory), { recursive: true });
    const files = readdirSync(capturedAudio).filter((name) => name.endsWith('.wav'));
    if (files.length === 0) throw new Error('Liquid speech audio capture is empty');
    for (const name of files) {
      copyFileSync(join(capturedAudio, name), join(resolve(liquidAudioOutputDirectory), name));
    }
  }
} finally {
  if (keepTemporaryDirectory) console.info(`LFM3_PACKED_TEMP ${temporaryDirectory}`);
  else rmSync(temporaryDirectory, { recursive: true, force: true });
}
