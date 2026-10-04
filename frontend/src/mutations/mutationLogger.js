/**
 * Mutation Manifest Logger
 * Records deterministic mutation logs for evaluation and debugging.
 *
 * CRITICAL SECURITY WARNING:
 * Mutation manifests contain ground-truth target mappings.
 * THEY MUST NEVER BE SUPPLIED TO THE RL AGENT.
 */

class MutationLogger {
  constructor() {
    this.currentManifest = {
      seed: 0,
      level: 0,
      timestamp: new Date().toISOString(),
      mutations: []
    };
  }

  init(level, seed) {
    this.currentManifest = {
      seed: Number(seed) || 0,
      level: Number(level) || 0,
      timestamp: new Date().toISOString(),
      mutations: []
    };
    if (typeof window !== 'undefined') {
      window.__RL_MUTATION_MANIFEST__ = this.currentManifest;
    }
  }

  logMutation(semanticRole, originalProps, mutatedProps) {
    const changes = {};

    if (originalProps.id !== mutatedProps.id) {
      changes.id = { from: originalProps.id, to: mutatedProps.id };
    }
    if (originalProps.placeholder !== mutatedProps.placeholder) {
      changes.placeholder = { from: originalProps.placeholder, to: mutatedProps.placeholder };
    }
    if (originalProps.text !== mutatedProps.text) {
      changes.text = { from: originalProps.text, to: mutatedProps.text };
    }
    if (originalProps.type !== mutatedProps.type) {
      changes.type = { from: originalProps.type, to: mutatedProps.type };
    }
    if (originalProps.className !== mutatedProps.className) {
      changes.className = { from: originalProps.className, to: mutatedProps.className };
    }

    if (Object.keys(changes).length > 0) {
      const existing = this.currentManifest.mutations.find((m) => m.semanticRole === semanticRole);
      if (existing) {
        existing.changes = { ...existing.changes, ...changes };
      } else {
        this.currentManifest.mutations.push({
          semanticRole,
          changes
        });
      }
    }
  }

  getManifest() {
    return this.currentManifest;
  }
}

export const mutationLogger = new MutationLogger();
