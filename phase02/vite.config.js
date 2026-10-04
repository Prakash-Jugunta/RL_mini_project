import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * MUTATION INJECTION PLUGIN
 *
 * Mechanism:
 *   1. The evaluation runner calls:
 *        GET /api/set-mutation?level=X&seed=Y
 *      which stores currentLevel / currentSeed in the dev-server process.
 *
 *   2. Every HTML response served by Vite has an inline <script> injected
 *      in the <head> that sets:
 *        window.__MUTATION_LEVEL__ = X
 *        window.__MUTATION_SEED__  = Y
 *      BEFORE any React/application JavaScript runs.
 *
 *   3. MutationProvider reads window.__MUTATION_LEVEL__ and
 *      window.__MUTATION_SEED__ as its fallback when URL query params
 *      are absent — which they always are during Playwright navigation
 *      (page.goto('/login') never carries query params).
 *
 * This approach requires:
 *   - NO modification to Phase 2 test files
 *   - NO query params in any test navigation
 *   - NO Vite server restart between evaluation runs
 *
 * GROUND-TRUTH / RL BOUNDARY:
 *   window.__MUTATION_LEVEL__ and window.__MUTATION_SEED__ are
 *   EVALUATOR-ONLY configuration. They must never be used as RL
 *   agent observation features.
 */
function mutationInjectionPlugin() {
  // Mutable state held in the dev-server Node.js process.
  // Mutated only by the /api/set-mutation endpoint.
  let currentLevel = 0;
  let currentSeed = 42;

  return {
    name: 'mutation-injection',

    // Only active during Vite dev server (not production build)
    apply: 'serve',

    configureServer(server) {
      // Evaluation runner calls this endpoint before each Playwright run.
      // Example: GET /api/set-mutation?level=1&seed=11
      server.middlewares.use('/api/set-mutation', (req, res) => {
        try {
          const url = new URL(req.url, 'http://localhost');
          const level = Number(url.searchParams.get('level') ?? 0);
          const seed  = Number(url.searchParams.get('seed')  ?? 42);

          if (isNaN(level) || level < 0 || level > 6) {
            res.statusCode = 400;
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ error: 'level must be 0–6' }));
            return;
          }

          currentLevel = level;
          currentSeed  = seed;

          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ ok: true, level: currentLevel, seed: currentSeed }));
        } catch (e) {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: String(e) }));
        }
      });

      // Read-only status endpoint — used by activation verification.
      server.middlewares.use('/api/get-mutation', (req, res) => {
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ level: currentLevel, seed: currentSeed }));
      });
    },

    // Called by Vite for every HTML response in dev mode.
    // Runs AFTER the endpoint has set currentLevel/currentSeed.
    transformIndexHtml(html) {
      const injection = `<script>
// EVALUATOR INJECTION — NOT AN RL OBSERVATION
window.__MUTATION_LEVEL__ = ${currentLevel};
window.__MUTATION_SEED__  = ${currentSeed};
</script>`;
      return html.replace('<head>', `<head>\n    ${injection}`);
    }
  };
}

export default defineConfig({
  plugins: [react(), mutationInjectionPlugin()],
  server: {
    port: 3000,
    open: false
  }
});
