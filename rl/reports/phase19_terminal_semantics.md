# Phase 19 — Terminal vs Truncation Semantics & Reproducibility Documentation

## 1. Episode Boundary & Terminal Semantics in `UIRecoveryEnv`

In `UIRecoveryEnv`, every transition $s_t \to s_{t+1}$ produces two boolean status flags according to standard Gymnasium contracts:

### A. `terminated = True`
- **Definition**: The active workflow completed successfully (`workflow_completed = True`).
- **MDP Nature**: True absorbing state. The target recovery goal has been achieved.
- **Value Bootstrap Policy**: Value estimate $V(s_{t+1}) = 0.0$.
- **GAE Masking**:
  $$\text{term\_mask} = 1.0 - \text{float}(\text{dones}[t]) = 0.0$$
  $$\delta_t = r_t - V(s_t)$$
  $$\text{cont\_mask} = 1.0 - \text{float}(\text{dones}[t] \lor \text{truncations}[t]) = 0.0$$
  $$A_t = \delta_t$$

### B. `truncated = True`
- **Definition**: The decision budget limit ($20$ steps) was reached without completing the workflow (`not workflow_completed and episode_steps >= 20`).
- **MDP Nature**: Step-budget cutoff penalty. The environment applies a failure penalty ($-3.0$) via `RewardCalculator`.
- **Value Bootstrap Policy**: The step limit is an artificial environment boundary, so $V(s_{t+1})$ is bootstrapped to estimate expected continuing return from $s_{t+1}$. However, because step $t$ is the final step of the episode, transitions from the *next* episode must not leak.
- **GAE Masking**:
  $$\text{term\_mask} = 1.0 - \text{float}(\text{dones}[t]) = 1.0$$
  $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
  $$\text{cont\_mask} = 1.0 - \text{float}(\text{dones}[t] \lor \text{truncations}[t]) = 0.0$$
  $$A_t = \delta_t$$

---

## 2. Multi-Episode Rollout Buffer GAE Isolation

When a rollout buffer contains transitions spanning multiple episodes:
- Transition $t$ ending an episode (either via `done=True` or `truncated=True`) has $\text{cont\_mask}_t = 0.0$.
- Thus, $A_{t+1}$ from the subsequent episode is multiplied by $\text{cont\_mask}_t = 0.0$ and **NEVER** propagates into $A_t$ or earlier transitions of the preceding episode.

---

## 3. Seed Reproducibility & RNG Controls

Training seed `--training-seed` explicitly seeds:
1. `random.seed(seed)`
2. `np.random.seed(seed)`
3. `torch.manual_seed(seed)`
4. `torch.cuda.manual_seed_all(seed)` (if CUDA is enabled)
5. `PPOAgent.set_seed(seed)` for reproducible NumPy RandomState minibatch shuffling
6. `Categorical(logits=masked_logits)` sampling via PyTorch RNG

### Playwright / Real-Browser Nondeterminism Note
While PyTorch policy weights, initializations, and state encodings are 100% deterministic given `--training-seed`, Playwright real-browser DOM rendering operates over asynchronous HTTP/DOM event dispatch on `http://localhost:3000`. Minor sub-millisecond OS thread scheduling variations may cause rare asynchronous DOM element extraction timing differences.
