import React, { createContext, useMemo } from 'react';
import { useLocation } from 'react-router-dom';
import { mutateProps } from './mutationEngine.js';
import { MUTATION_LEVELS, DISTRACTOR_TEMPLATES, resolvePreset } from './mutationPresets.js';
import { mutationLogger } from './mutationLogger.js';
import { createPRNG } from './prng.js';

// GROUND-TRUTH METADATA ONLY.
//
// MutationContext provides mutation state to all page components.
// It MUST NOT expose data-semantic-role or mutation manifest to any
// future RL agent observation. These are evaluator-only ground truths.

export const MutationContext = createContext({
  mutationLevel: 0,
  seed: 42,
  levelConfig: null,
  mutate: (role, props) => ({ ...props, 'data-semantic-role': role }),
  getDistractors: () => [],
  shouldWrap: () => false,
  shouldReorder: () => false
});

export function MutationProvider({ children }) {
  const location = useLocation();

  const { level, seed } = useMemo(() => {
    const params = new URLSearchParams(location.search);
    const paramLevel = params.get('mutationLevel');
    const paramSeed = params.get('seed');

    const envLevel = typeof window !== 'undefined' && window.__MUTATION_LEVEL__ !== undefined
      ? window.__MUTATION_LEVEL__
      : null;
    const envSeed = typeof window !== 'undefined' && window.__MUTATION_SEED__ !== undefined
      ? window.__MUTATION_SEED__
      : null;

    const finalLevel = Math.max(0, Math.min(6, Number(paramLevel ?? envLevel ?? 0)));
    const finalSeed = Number(paramSeed ?? envSeed ?? 42);

    mutationLogger.init(finalLevel, finalSeed);
    return { level: finalLevel, seed: finalSeed };
  }, [location.search]);

  const value = useMemo(() => {
    const levelConfig = MUTATION_LEVELS[level] || MUTATION_LEVELS[0];
    const { activeMutations, name: presetName } = resolvePreset(level, seed);


    return {
      mutationLevel: level,
      seed,
      levelConfig,
      presetName,
      activeMutations,
      mutate: (semanticRole, originalProps) => mutateProps(semanticRole, originalProps, level, seed),


      // M6: Distractor elements generator
      getDistractors: (workflowName) => {
        if (!activeMutations.includes('distractor')) return [];
        const templates = DISTRACTOR_TEMPLATES[workflowName] || [];
        const rng = createPRNG(seed * 7001 + level);
        return templates.filter(() => rng() > 0.3).map((item) => ({
          ...item,
          id: `${item.idPrefix}-distractor-x${Math.floor(rng() * 900 + 100)}`
        }));
      },

      // M5: DOM structure nesting check
      shouldWrap: (semanticRole) => {
        if (!activeMutations.includes('dom')) return false;
        const rng = createPRNG(seed * 3001 + semanticRole.length);
        return rng() > 0.4;
      },

      // M4: Position/order change check
      shouldReorder: () => {
        if (!activeMutations.includes('position')) return false;
        const rng = createPRNG(seed * 5003);
        return rng() > 0.5;
      }
    };
  }, [level, seed]);

  return (
    <MutationContext.Provider value={value}>
      {children}
    </MutationContext.Provider>
  );
}
