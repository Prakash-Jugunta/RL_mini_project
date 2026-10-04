# Phase 4 — Gymnasium RL Environment Architecture

## 1. Environment Overview

The **UI Recovery Environment** (`UIRecoveryEnv`) is a Gymnasium-compatible environment representing sequential UI test recovery as a Markov Decision Process (MDP).

When conventional fixed CSS selectors break under DOM/text/ID mutations, the RL agent policy is tasked with identifying and selecting the correct UI candidate element to advance the workflow to completion.

---

## 2. Conceptual MDP Loop

```text
               s_t (Observation: workflow, step, candidates, history)
                                 │
                                 ▼
                     Agent Policy Chooses a_t (Discrete Candidate Index)
                                 │
                                 ▼
  Environment Executes Action & Evaluates Outcome (Private Evaluator Check)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Reward r_t (+1/-1/+2)           Next State s_(t+1)
                                                 │
                                                 ▼
                                     Terminated / Truncated?
```

---

## 3. Core Environment Concepts

### A. Episode
An episode represents a single execution attempt of a specific sequential workflow (e.g. `LOGIN`, `SEARCH`, `PROFILE`, `CHECKOUT`) under a designated UI mutation configuration (`level` 0–6, `seed`).

### B. State / Observation (`s_t`)
The agent policy observes the current environment state.

**TEMPORARY PHASE 4 OBSERVATION CONTRACT**:
For Phase 4 architectural validation, the observation vector is a placeholder 7-dimensional numeric vector:
1. `workflow_index` (0=LOGIN, 1=SEARCH, 2=PROFILE, 3=CHECKOUT)
2. `current_step_index` (0..num_steps)
3. `num_candidates` (1..max_candidates)
4. `previous_action` (-1.0 if none, else action index)
5. `previous_success` (1.0 if last action succeeded, else 0.0)
6. `remaining_steps` (num_steps - current_step_index)
7. `mutation_level` (0..6)

*Note: This temporary observation will be replaced by the rich State Encoder in Phase 6.*

### C. Action Space (`a_t`)
- **Space**: `gymnasium.spaces.Discrete(max_candidates)`
- **Semantics**: Choice of candidate UI element index (`0 <= action < max_candidates`).
- **Candidate Ordering**: Seed-dependently permuted using NumPy random generator. Target elements are NEVER fixed at index 0.

### D. Transitions & Reward Contract
- **Correct Step Advancement**: Reward = `+1.0`, step index advances.
- **Incorrect Candidate Choice**: Reward = `-1.0`, step index does not advance.
- **Invalid Action (Out of bounds)**: Reward = `-1.0`, step index does not advance.
- **Workflow Completion Bonus**: Reward = `+2.0`, `terminated = True`.

*Note: Reward function values are temporary placeholders for Phase 4 architecture and will be tuned in Phase 8.*

### E. Termination vs Truncation
- `terminated = True`: Workflow is successfully completed (`current_step_index >= total_steps`).
- `truncated = True`: Maximum episode steps limit (`max_episode_steps = 20`) reached without workflow completion.

---

## 4. Critical Integrity Guards

### A. Ground-Truth Leakage Protection (`data-semantic-role`)
- `data-semantic-role` is **EVALUATOR GROUND TRUTH ONLY**.
- It is strictly **EXCLUDED** from:
  - Agent observation space (`obs`)
  - Agent-visible candidate structures (`UICandidate`)
  - Info dictionaries (`info`)
- The agent policy receives element attributes (tag, text, placeholder, role, bounding box, class names) but NEVER the ground-truth semantic role.

### B. Held-Out Mutation Level 6 Protection
- Mutation Level 6 (`id+dom+distractor`, `text+type+position`, etc.) is **HELD OUT** for evaluation only.
- In `mode == 'train'`, attempting to instantiate or reset the environment with `mutation_level == 6` raises a `ValueError` exception.
- Level 6 is permitted strictly in `mode == 'test'` or `mode == 'validation'`.

---

## 5. Workflows Supported

1. **`LOGIN`**: `ENTER_USERNAME` $\rightarrow$ `ENTER_PASSWORD` $\rightarrow$ `CLICK_LOGIN` $\rightarrow$ `VERIFY_DASHBOARD`
2. **`SEARCH`**: `ENTER_SEARCH_QUERY` $\rightarrow$ `CLICK_SEARCH` $\rightarrow$ `SELECT_PRODUCT` $\rightarrow$ `VERIFY_PRODUCT_PAGE`
3. **`PROFILE`**: `ENTER_NAME` $\rightarrow$ `ENTER_EMAIL` $\rightarrow$ `CLICK_SAVE` $\rightarrow$ `VERIFY_SUCCESS`
4. **`CHECKOUT`**: `ADD_TO_CART` $\rightarrow$ `CLICK_CART` $\rightarrow$ `PROCEED_CHECKOUT` $\rightarrow$ `CONFIRM_ORDER` $\rightarrow$ `VERIFY_ORDER_SUCCESS`

---

## 6. Project Roadmap & Future Phases

- **Phase 4 (CURRENT)**: Environment Architecture, Interfaces, Contracts, Gymnasium Compliance, Leakage Guards.
- **Phase 5 (NEXT)**: Real DOM Candidate Extraction & Playwright Integration.
- **Phase 6**: Feature Vectorization & State Encoder (SentenceTransformers / Embeddings).
- **Phase 7**: Action Space & Candidate Ranking Design.
- **Phase 8**: Reward Function Engineering & Penalty Tuning.
- **Phase 9–11**: Policy Training (DQN/PPO/A2C) & Self-Healing Evaluation.
