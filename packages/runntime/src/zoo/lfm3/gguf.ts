/** Identity of Liquid AI's published Q4_K_M LFM3-Omni language file. */

const revision = '3d29af2ad166f8299c2d820937e47dc0add20ffc';
const repository = 'LiquidAI/LFM3-Omni-2B-GGUF';
const file = 'LFM3-Omni-2B-Q4_K_M.gguf';

/** Repository, pinned revision, file, size, SHA-256 and download URL. `fromGguf`
 *  checks the size and digest before it reads any tensor. */
export const LFM3_Q4_K_M_SOURCE = {
  repository,
  revision,
  file,
  bytes: 1_099_630_080,
  sha256: '9f8d30e9d4fd7a9349d586e077988229b3ca0a0e4ad730897563d1eccc9893e2',
  url: `https://huggingface.co/${repository}/resolve/${revision}/${file}`,
} as const;
