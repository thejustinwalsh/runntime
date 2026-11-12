/** Builds the package tarball and runs its public LFM3 loader on real WebGPU. */

import {
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
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const packageDirectory = resolve(scriptDirectory, '../../..');
const repositoryDirectory = resolve(packageDirectory, '../..');
const modelUrl = process.env.LFM3_BASE_MODEL_URL;
const tokenizerUrl = process.env.LFM3_BASE_TOKENIZER_URL;
const reportPath = process.env.LFM3_BASE_VITEST_REPORT;
const qualificationMode = process.env.LFM3_BASE_QUALIFICATION_MODE ?? 'base';

if (!modelUrl) {
  throw new Error('set LFM3_BASE_MODEL_URL to the pinned original model.safetensors URL');
}

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

export default defineConfig({
  plugins: [typegpu()],
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
    VITE_LFM3_MODEL_URL: modelUrl,
    VITE_LFM3_TOKENIZER_URL:
      tokenizerUrl ??
      'https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/21d75123f86c1f7ffb4bef1ea202710c4e7ebe67/tokenizer.json',
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
  run(process.execPath, vitestArgs, consumerDirectory, env);
} finally {
  rmSync(temporaryDirectory, { recursive: true, force: true });
}
