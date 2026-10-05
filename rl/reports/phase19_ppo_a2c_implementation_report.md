# Phase 19 — PPO & A2C Implementation Report

## 1. Executive Summary & Objective
Phase 19 implements candidate-aware **Proximal Policy Optimization (PPO)** and **Synchronous Advantage Actor-Critic (A2C)** algorithms on the exact frozen `UIRecoveryEnv` environment. This enables a fair baseline comparison against the Masked Double DQN while preserving state dimensions, action space, reward functions, candidate extraction, and split definitions.

---

## 2. Frozen Environment Contract

Both PPO and A2C reuse the canonical Gymnasium environment without modification:

- **Observation Space**:
  - `objective`: shape `(64,)` (BERT objective embedding)
  - `context`: shape `(30,)` (previous action & DOM history)
  - `candidates`: shape `(20, 91)` (public candidate features)
  - `candidate_mask`: shape `(20,)` (validity mask)
- **Action Space**: `Discrete(20)` (select candidate slot $i \in [0, 19]$)
- **Candidate Validity Masking**: Padded slots ($m_i \le 0$) masked with $-\infty$ logit bias.

---

## 3. Candidate-Aware Actor-Critic Architecture

PPO and A2C share a permutation-equivariant actor and permutation-invariant critic architecture ([actor_critic_network.py](file:///e:/sem7/RL/RL_replay/rl/agents/common/actor_critic_network.py)):

### Submodule Encoders
- **Objective Encoder**: $64 \to 128 \to 64$
- **Context Encoder**: $30 \to 64 \to 32$
- **Shared Candidate Encoder**: $91 \to 128 \to 64$ (applied to each slot)

### Actor Scorer (Permutation Equivariant)
- Concatenates objective ($64$), context ($32$), and candidate ($64$) embeddings $\to 160$ dimensions per slot.
- Applies shared linear head $160 \to 128 \to 64 \to 1$ across all 20 candidate slots.
- Output: raw policy logits $\mathbf{z} \in \mathbb{R}^{20}$.
- Applies candidate mask bias before constructing `torch.distributions.Categorical(logits=masked_logits)`.

### Critic Scorer (Permutation Invariant)
- Performs masked mean pooling over valid candidate embeddings:
  $$\mathbf{e}_{\text{pooled}} = \frac{\sum_{i=1}^K m_i \mathbf{e}_i}{\max(\sum_{i=1}^K m_i, 1)}$$
- Concatenates objective ($64$), context ($32$), and pooled candidate ($64$) embeddings $\to 160$ dimensions.
- Applies value head $160 \to 128 \to 64 \to 1 \to V(s) \in \mathbb{R}$.

---

## 4. Parameter Complexity

- **Actor Parameters**: $65,569$
- **Critic Parameters**: $65,569$
- **Total Shared Network Parameters**: **$98,530$ trainable parameters** ($394.12 \text{ KB}$ / $384.88 \text{ KiB}$)

---

## 5. Algorithmic Formulations

### Proximal Policy Optimization (PPO)
- **Clipped Surrogate Objective**:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$
- **GAE Advantage Estimation**:
  $$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$
- **Exploration**: Stochastic Categorical policy sampling during training + entropy bonus $S[\pi_\theta]$. Deterministic `argmax` during evaluation.

### Synchronous Advantage Actor-Critic (A2C)
- **Actor Loss**:
  $$L^{\text{actor}}(\theta) = -\hat{\mathbb{E}}_t \left[ \log \pi_\theta(a_t | s_t) \hat{A}_t \right]$$
- **Critic Loss**:
  $$L^{\text{critic}}(\phi) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ (V_\phi(s_t) - R_t)^2 \right]$$

---

## 6. Verification & Test Suite Results

- **Unit Tests**: `13/13 passed` ([test_actor_critic_network.py](file:///e:/sem7/RL/RL_replay/rl/tests/test_actor_critic_network.py), [test_ppo_agent.py](file:///e:/sem7/RL/RL_replay/rl/tests/test_ppo_agent.py), [test_a2c_agent.py](file:///e:/sem7/RL/RL_replay/rl/tests/test_a2c_agent.py))
- **Permutation Equivariance & Invariance**: `PASSED`
- **PPO & A2C Mock Environment Smoke Test**: `PASSED` (200 steps)
- **PPO & A2C Real Playwright Browser Smoke Test**: `PASSED` (Real browser integration on `http://localhost:3000`)
- **Full Test Suite**: `285/285 passed` (`python -m pytest rl/tests -q`)

---

## 7. Execution Commands for Full Training

To execute full 15,000-episode training runs manually, execute:

### PPO Full Training
```bash
python -u -m rl.training.train_ppo \
  --curriculum-seed 1301 \
  --split-seed 1201 \
  --training-seed 2026 \
  --output-dir artifacts/ppo/phase19-v1 \
  --max-episodes 15000 \
  --confirm-training
```

### A2C Full Training
```bash
python -u -m rl.training.train_a2c \
  --curriculum-seed 1301 \
  --split-seed 1201 \
  --training-seed 2026 \
  --output-dir artifacts/a2c/phase19-v1 \
  --max-episodes 15000 \
  --confirm-training
```
