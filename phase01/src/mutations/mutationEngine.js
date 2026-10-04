import { createPRNG, pickRandom, randomHexSuffix } from './prng.js';
import { MUTATION_LEVELS, SEMANTIC_DICTIONARY, resolvePreset } from './mutationPresets.js';
import { mutationLogger } from './mutationLogger.js';

/**
 * Core Mutation Engine
 *
 * Deterministically computes mutated element attributes based on
 * (level, seed, semanticRole).
 *
 * For Level 5 and Level 6, activeMutations are resolved per-seed
 * via resolvePreset() rather than from the fixed MUTATION_LEVELS table.
 * This enables the explicit seed→preset mapping required for the
 * train/test experimental design.
 *
 * CRITICAL GROUND-TRUTH INVARIANT:
 *   data-semantic-role MUST ALWAYS BE PRESERVED EXACTLY AS SUPPLIED.
 *   It is evaluator-only metadata and must never be used as an RL
 *   agent observation feature.
 */

function stringHash(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

export function mutateProps(semanticRole, originalProps, level = 0, seed = 42) {
  // Resolve active mutations — for L5/L6 this is seed-dependent
  const { activeMutations } = resolvePreset(level, seed);

  // Level 0 or no active mutations — identity transform + semantic tag
  if (level === 0 || activeMutations.length === 0) {
    return {
      ...originalProps,
      'data-semantic-role': semanticRole
    };
  }

  // Deterministic RNG seeded per (seed, semanticRole)
  const roleSeed = Number(seed) * 10007 + stringHash(semanticRole);
  const rng = createPRNG(roleSeed);

  const mutatedProps = { ...originalProps };
  // INVARIANT: semantic role always preserved for evaluator ground truth
  mutatedProps['data-semantic-role'] = semanticRole;

  // ── M1: ID Change ─────────────────────────────────────────────────
  // Replaces the original id with a randomised hex-suffixed variant.
  // This directly breaks brittle #id selectors used in Phase 2 tests.
  if (activeMutations.includes('id') && originalProps.id) {
    const baseName = originalProps.id.split('-')[0] || 'element';
    const hex = randomHexSuffix(rng, 3);
    mutatedProps.id = `${baseName}-x${hex}`;
  }

  // ── M2: Class Change ──────────────────────────────────────────────
  if (activeMutations.includes('class') && originalProps.className) {
    const hex = randomHexSuffix(rng, 2);
    const existingClasses = originalProps.className.split(' ');
    const primaryClass = existingClasses[0] || 'control';
    mutatedProps.className = `${primaryClass} ctrl-x${hex} ${existingClasses.slice(1).join(' ')}`.trim();
  }

  // ── M3: Text / Label / Placeholder Change ─────────────────────────
  if (activeMutations.includes('text')) {
    const dict = SEMANTIC_DICTIONARY[semanticRole];
    if (dict) {
      if (originalProps.placeholder && dict.placeholder) {
        const alt = pickRandom(dict.placeholder, rng);
        if (alt) mutatedProps.placeholder = alt;
      }
      if (originalProps.text && dict.text) {
        const alt = pickRandom(dict.text, rng);
        if (alt) mutatedProps.text = alt;
      }
      if (originalProps.label && dict.label) {
        const alt = pickRandom(dict.label, rng);
        if (alt) mutatedProps.label = alt;
      }
    }
  }

  // ── M7: Element Type / Component Change ───────────────────────────
  // Only applied to button elements; preserves interactive semantics.
  if (activeMutations.includes('type') && originalProps.type === 'button') {
    const roll = rng();
    if (roll > 0.5) {
      mutatedProps.renderAs = 'a';
      mutatedProps.role = 'button';
    }
  }

  // Log changes for evaluator manifest (not exposed to RL agent)
  mutationLogger.logMutation(semanticRole, originalProps, mutatedProps);

  return mutatedProps;
}
