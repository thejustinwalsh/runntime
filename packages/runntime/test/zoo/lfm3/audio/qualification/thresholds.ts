/** Limits for the audio qualification. Each is stated with the measurement it came from.
 *  Measured on an Apple M2 Pro (Metal 3), Chromium 156 headless, Q4_K_M and Q8_0 weights. */

/** 96 round trips (3 seeds x 4 voices x 8 sentences): mean 0.0015, max 0.1429 (one word of the
 *  seven in "Please turn left at the next street", heard as "treat"), 95 exact. Seed 1234 alone: 64 of
 *  64 exact in two runs. The mean limit allows about three wrong words in the 32 items of one seed;
 *  the item limit is two wrong words in the shortest sentence. */
export const WER_LIMITS = {
  meanWer: 0.01,
  itemWer: 2 / 7,
  /** One 8-sentence clip of 29 s: 0 of 71 words wrong. Same per-item limit. */
  longWer: 2 / 7,
} as const;
