/**
 * Seeded Pseudo-Random Number Generator (Mulberry32)
 * Ensures 100% deterministic mutations given the same seed.
 */
export function createPRNG(seed) {
  let s = Math.abs(Math.floor(seed)) || 42;
  return function random() {
    let t = (s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Helper to pick a random item from an array deterministically.
 */
export function pickRandom(arr, rng) {
  if (!arr || arr.length === 0) return null;
  const index = Math.floor(rng() * arr.length);
  return arr[index];
}

/**
 * Helper to generate a deterministic random hexadecimal suffix.
 */
export function randomHexSuffix(rng, length = 3) {
  const chars = '0123456789abcdef';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars[Math.floor(rng() * chars.length)];
  }
  return result;
}
