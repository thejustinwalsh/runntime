/** Text-only streaming chat against Liquid's pinned LFM3-Omni-2B base model. */

import tgpu from 'typegpu';
import {
  createLfm3Chat,
  initRunntime,
  models,
  type Lfm3ChatSession,
  type Lfm3Generation,
  type Lfm3Message,
} from '../../src/zoo/index.ts';

const loadButton = document.querySelector<HTMLButtonElement>('#load')!;
const sendButton = document.querySelector<HTMLButtonElement>('#send')!;
const cancelButton = document.querySelector<HTMLButtonElement>('#cancel')!;
const resetButton = document.querySelector<HTMLButtonElement>('#reset')!;
const prompt = document.querySelector<HTMLTextAreaElement>('#prompt')!;
const progress = document.querySelector<HTMLProgressElement>('#progress')!;
const status = document.querySelector<HTMLParagraphElement>('#status')!;
const output = document.querySelector<HTMLPreElement>('#chat')!;

let session: Lfm3ChatSession | undefined;
let generation: Lfm3Generation | undefined;
let history: Lfm3Message[] = [];

loadButton.addEventListener('click', () => void loadModel());
sendButton.addEventListener('click', () => void sendMessage());
cancelButton.addEventListener('click', () => generation?.cancel());
resetButton.addEventListener('click', () => void resetConversation());
window.addEventListener('pagehide', () => void session?.dispose().catch(() => undefined));

async function loadModel(): Promise<void> {
  loadButton.disabled = true;
  status.textContent = 'Requesting WebGPU and loading weights…';
  const started = performance.now();
  try {
    initRunntime(await tgpu.init({ device: { optionalFeatures: ['subgroups'] } }));
    session = await createLfm3Chat(models.lfm3.LFM3_OMNI_2B.DEFAULT, {
      onProgress: (name, done, total) => {
        progress.value = total === 0 ? 0 : done / total;
        status.textContent = `Loading ${name}: ${Math.floor(progress.value * 100)}%`;
      },
    });
    status.textContent = `Ready in ${((performance.now() - started) / 1000).toFixed(1)} s`;
    sendButton.disabled = false;
    resetButton.disabled = false;
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error);
    loadButton.disabled = false;
  }
}

async function sendMessage(): Promise<void> {
  if (!session || generation || !prompt.value.trim()) return;
  const content = prompt.value.trim();
  prompt.value = '';
  history = [...history, { role: 'user', content }];
  const prior = output.textContent ?? '';
  const promptText = `${prior}You: ${content}\nLFM3: `;
  output.textContent = promptText;
  generation = session.generate(history, { maxNewTokens: 128 });
  sendButton.disabled = true;
  cancelButton.disabled = false;
  let answer = '';
  try {
    for await (const delta of generation) {
      answer += delta;
      output.textContent = `${promptText}${answer}`;
    }
    const result = await generation.result;
    history = [...history, { role: 'assistant', content: result.text }];
    output.textContent += `\n[${result.finishReason}, ${result.completionTokens} tokens]\n\n`;
    status.textContent = 'Ready';
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    generation = undefined;
    sendButton.disabled = false;
    cancelButton.disabled = true;
  }
}

async function resetConversation(): Promise<void> {
  if (!session) return;
  generation?.cancel();
  await session.reset();
  generation = undefined;
  history = [];
  output.textContent = '';
  status.textContent = 'Conversation reset';
  sendButton.disabled = false;
  cancelButton.disabled = true;
}
