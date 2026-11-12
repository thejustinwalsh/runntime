/** Packs ruNNtime, then runs the public D1 Q8 fixture through that tarball. */

import {
  copyFileSync,
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
const modelPath = process.env.D1_Q8_MODEL_PATH;
const tokenizerPath = process.env.D1_Q8_TOKENIZER_PATH;
const reportPath = process.env.D1_Q8_VITEST_REPORT;

if (!modelPath || !tokenizerPath) {
  throw new Error('set D1_Q8_MODEL_PATH and D1_Q8_TOKENIZER_PATH to local artifacts');
}

const temporaryDirectory = mkdtempSync(join(tmpdir(), 'runntime-packed-d1-'));

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

  const publicDirectory = join(consumerDirectory, 'public');
  const testDirectory = join(consumerDirectory, 'test', 'zoo', 'd1');
  mkdirSync(publicDirectory);
  mkdirSync(testDirectory, { recursive: true });
  symlinkSync(resolve(modelPath), join(publicDirectory, 'model.safetensors'), 'file');
  copyFileSync(tokenizerPath, join(publicDirectory, 'tokenizer.json'));
  copyFileSync(
    join(packageDirectory, 'playground/d1/reference-vectors.json'),
    join(testDirectory, 'reference-vectors.json'),
  );

  const qualificationSource = readFileSync(
    join(packageDirectory, 'test/zoo/d1/q8.gpu.test.ts'),
    'utf8',
  )
    .replace("from '../../../src/zoo/index.ts'", "from 'runntime/zoo'")
    .replace(
      "from '../../../playground/d1/reference-vectors.json'",
      "from './reference-vectors.json'",
    );
  if (!qualificationSource.includes("from 'runntime/zoo'")) {
    throw new Error('Q8 consumer test did not switch to the packed public entry point');
  }
  writeFileSync(join(testDirectory, 'q8.gpu.test.ts'), qualificationSource);
  writeFileSync(
    join(consumerDirectory, 'test/gpu.ts'),
    `/** Minimal WebGPU setup for the packed public-zoo consumer. */
import tgpu from 'typegpu';
import { initRunntime } from 'runntime/zoo';

const live = new Set<GPUBuffer>();
if (typeof GPUDevice !== 'undefined') {
  const createBuffer = GPUDevice.prototype.createBuffer;
  const destroyBuffer = GPUBuffer.prototype.destroy;
  GPUDevice.prototype.createBuffer = function (descriptor) {
    const buffer = createBuffer.call(this, descriptor);
    live.add(buffer);
    return buffer;
  };
  GPUBuffer.prototype.destroy = function () {
    live.delete(this);
    destroyBuffer.call(this);
  };
}

export function liveBuffers() {
  return new Set(live);
}

export function buffersSince(before) {
  return [...live]
    .filter((buffer) => !before.has(buffer))
    .map((buffer) => buffer.size + ' B, usage ' + buffer.usage);
}

let rootPromise;
export function gpu() {
  rootPromise ??= tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } }).then((root) => {
    initRunntime(root);
    return { root };
  });
  return rootPromise;
}
`,
  );

  const config = `import { playwright } from '@vitest/browser-playwright';
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
`;
  writeFileSync(join(consumerDirectory, 'vitest.config.ts'), config);
  writeFileSync(join(consumerDirectory, 'package.json'), JSON.stringify({ type: 'module' }));

  const env = {
    ...process.env,
    VITE_D1_Q8_QUALIFICATION_MODE: 'q8',
    VITE_D1_Q8_MODEL_URL: '/model.safetensors',
    VITE_D1_Q8_TOKENIZER_URL: '/tokenizer.json',
    VITE_D1_Q8_ARTIFACT_SHA256: process.env.D1_Q8_ARTIFACT_SHA256 ?? '',
  };
  if (!env.VITE_D1_Q8_ARTIFACT_SHA256) {
    throw new Error('set D1_Q8_ARTIFACT_SHA256 to the converted artifact SHA-256');
  }

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
