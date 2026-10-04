# PHASE 10 — MASKED DOUBLE DQN IMPLEMENTATION REPORT

## 1. Objective
The goal of Phase 10 is to build a technically correct, fully tested, reproducible candidate-aware **Masked Double DQN** architecture and learning machinery for self-healing UI test recovery. The implementation interacts with the frozen Gymnasium environment (`UIRecoveryEnv`) without altering any underlying contracts, reward mechanisms, candidate extractors, or state representations, and without starting full experimental training or hyperparameter tuning.

---

## 2. Frozen Environment Contract
All previous phases (Phases 1–8) remain strictly frozen and unchanged:
- **Observation Space**: Gymnasium `Dict` with keys `objective` (64,), `candidates` (20, 91), `candidate_mask` (20,), `context` (30,).
- **Action Space**: `Discrete(20)` representing candidate index selection \(i \in \{0, \dots, 19\}\).
- **Reward Function**: Step cost \(-0.05\), progress reward \(+1.00\), wrong action penalty \(-0.50\), invalid action penalty \(-1.00\), completion bonus \(+5.00\), failure penalty \(-2.00\).
- **Page-State Verification**: Automatic URL assertions (`VERIFY_PRODUCT_PAGE`) consume 0 RL actions and 0 independent rewards.

---

## 3. Observation Structure
The observation space represents structured UI element candidates alongside global step context and test objective embeddings:
- `objective`: (64,) float32 array encoding the high-level intent (e.g. login, search, profile, checkout).
- `context`: (30,) float32 array encoding progress, step indices, and page metadata.
- `candidates`: (20, 91) float32 array encoding features of up to 20 candidate DOM elements.
- `candidate_mask`: (20,) int8/float32 binary mask where 1 indicates a valid extracted candidate and 0 indicates padding.

---

## 4. Why Conventional Fixed-Output Interpretation is Inappropriate
In a traditional DQN for video games (e.g., Atari), output neuron \(i\) maps to a persistent semantic action (e.g. Action 0 = UP, Action 1 = DOWN). 

In UI recovery with dynamic DOM candidate extraction:
- Candidate slot #3 in state A might be the Login Button.
- Candidate slot #3 in state B might be a Username Input field.
- Candidate slot #3 in state C might be padding or an irrelevant footer link.

Candidate ordering is deliberately randomized by the candidate extractor. A standard 1934 \(\rightarrow\) 20 MLP would incorrectly try to learn permanent semantic meanings for output indices. Instead, the Q-network must evaluate **\(Q(s, \text{candidate}_i)\)** based on the content and features of candidate \(i\).

---

## 5. Candidate-Aware Network Architecture
The `CandidateAwareQNetwork` processes the observation via shared encoders:

```text
Objective (64)   ──────►  Objective Encoder (64 ──► 128 ──► 64) ──────────┐
                                                                          │
Context (30)     ──────►  Context Encoder (30 ──► 64 ──► 32)   ──────────┼──► Concatenated Representation (160) ──► Q-Scorer (160 ──► 128 ──► 64 ──► 1) ──► Q(s, candidate_i)
                                                                          │
Candidate_i (91) ──────►  Candidate Encoder (91 ──► 128 ──► 64) ──────────┘   [Shared across all 20 candidate slots]
```

### Mathematical Formalization
For candidate \(i \in \{0, \dots, 19\}\):
\[
z_o = f_{\text{obj}}(\text{objective})
\]
\[
z_c = f_{\text{ctx}}(\text{context})
\]
\[
z_i = f_{\text{cand}}(\text{candidate}_i)
\]
\[
Q(s, i) = f_{Q}([z_o, z_c, z_i])
\]

The candidate encoder \(f_{\text{cand}}\) and Q-scorer \(f_{Q}\) are shared across all 20 candidate slots, ensuring **permutation equivariance** across candidate permutations.

---

## 6. Network Dimensions & PyTorch Layer Specifications
- **Objective Encoder**: `Linear(64, 128)` \(\rightarrow\) `ReLU()` \(\rightarrow\) `Linear(128, 64)` \(\rightarrow\) `ReLU()`
- **Context Encoder**: `Linear(30, 64)` \(\rightarrow\) `ReLU()` \(\rightarrow\) `Linear(64, 32)` \(\rightarrow\) `ReLU()`
- **Candidate Encoder (Shared)**: `Linear(91, 128)` \(\rightarrow\) `ReLU()` \(\rightarrow\) `Linear(128, 64)` \(\rightarrow\) `ReLU()`
- **Q-Scorer (Shared)**: `Linear(160, 128)` \(\rightarrow\) `ReLU()` \(\rightarrow\) `Linear(128, 64)` \(\rightarrow\) `ReLU()` \(\rightarrow\) `Linear(64, 1)`
- **Input Batched Shape**: `(B, 20, 91)`, `(B, 64)`, `(B, 30)`
- **Output Batched Shape**: `(B, 20)`

---

## 7. Action Masking
Padding candidate slots must never be selected by the agent during exploitation or target calculation.
- **Masking Mechanism**: `masked_q = raw_q.masked_fill(candidate_mask <= 0, -torch.inf)`
- **Empty Mask Policy**: If `candidate_mask` contains zero valid candidates (all zeros), the agent raises a `NoValidActionError` instead of falling back to action 0.

---

## 8. \(\epsilon\)-Greedy Exploration
Exploration samples uniformly from valid candidate slots only:
- **Valid Action Indices**: `valid_indices = np.flatnonzero(candidate_mask == 1)`
- **Action Selection**: `action = np.random.choice(valid_indices)` using a dedicated `np.random.Generator`.
- Padding slots are strictly excluded from exploration.

---

## 9. Replay Buffer
`DictReplayBuffer` stores structured Gymnasium Dict observations using preallocated NumPy arrays:
- **Stored Fields**: `objective`, `candidates`, `candidate_mask`, `context`, `action`, `reward`, `next_objective`, `next_candidates`, `next_candidate_mask`, `next_context`, `terminated`, `truncated`.
- **Exclusions**: Ground truth locators, `expected_role`, `semantic_role`, `mutation_level`, `mutation_seed`, and evaluator hints are strictly absent from replay storage.

---

## 10. Double DQN Target Calculation
To prevent Q-value overestimation, Double DQN uses the **online network** to select the greedy action in the next state using the **next state's candidate mask**, and evaluates that action with the **target network**:
\[
a^* = \arg\max_{a \in \text{valid}(s')} Q_{\text{online}}(s', a)
\]
\[
y = \begin{cases} 
r & \text{if } \text{terminated or truncated} \\
r + \gamma \cdot Q_{\text{target}}(s', a^*) & \text{otherwise}
\end{cases}
\]

### Termination & Truncation Bootstrapping Contract
Per the project budget contract, budget truncation represents episode failure. Thus `done = terminated or truncated`. No bootstrapping occurs on either true episode termination or budget truncation.

---

## 11. Target Network Updates
- Target network weights are initialized to match the online network: `target_net.load_state_dict(online_net.state_dict())`.
- Target network gradients are disabled (`requires_grad = False`).
- Hard target updates occur every `target_update_frequency` environment steps.

---

## 12. Loss Function & Optimizer
- **Loss**: Smooth L1 (Huber) loss evaluated strictly on the executed action's Q-value:
  \[
  \mathcal{L} = \text{SmoothL1Loss}(Q_{\text{online}}(s, a), y)
  \]
- **Optimizer**: AdamW optimizer with configurable learning rate (`1e-3` default for smoke training) and gradient norm clipping (`max_norm = 1.0`).

---

## 13. Reproducibility
Random seed configuration explicitly seeds:
- Python `random`
- NumPy global RNG & `np.random.default_rng(seed)`
- PyTorch CPU & CUDA generators
- Gymnasium environment seeds

---

## 14. Leakage Prevention & Audit
A strict source audit was conducted across `rl/agents` and `rl/training`.
- Prohibited private evaluator metadata (`expected_role`, `semantic_role`, `ground_truth`, `is_correct`, `target_role`, `private_meta`) is 100% absent from Q-network inputs, action selection logic, and replay buffer storage.
- Mutation metadata (`mutation_level`, `mutation_seed`) is restricted exclusively to experiment logging and background episode tracking. Counterfactual tests verified that changing private metadata while holding public observations constant produces bitwise identical Q-values.

---

## 15. Unit Tests
15 comprehensive unit tests were implemented and verified across 6 test modules:
1. `test_network_output_shapes`: Verifies batched Q-value tensor shapes `(B, 20)` for batch sizes 1 and >1.
2. `test_candidate_permutation_equivariance`: Proves candidate scoring equivariance when candidate feature order is permuted.
3. `test_masked_action_selection_excludes_padding`: Confirms greedy action selection never selects padded elements.
4. `test_random_exploration_never_selects_padding`: Validates 1000 exploratory steps never select padding slots.
5. `test_empty_mask_raises_no_valid_action_error`: Ensures zero-valid candidate masks raise `NoValidActionError`.
6. `test_replay_buffer_capacity_and_rollover`: Validates circular FIFO rollover and length bounds.
7. `test_replay_buffer_sampling_shapes_and_types`: Verifies sampled batch dictionary keys, shapes, and dtypes.
8. `test_replay_buffer_private_metadata_absence_audit`: Asserts replay buffer schema contains no private metadata keys.
9. `test_double_dqn_target_uses_next_state_mask`: Confirms next-state action selection uses `next_candidate_mask`.
10. `test_terminal_and_truncated_target_no_bootstrap`: Verifies zero bootstrapping when `terminated=True` or `truncated=True`.
11. `test_target_network_isolation`: Validates online parameter updates leave target network untouched until hard update.
12. `test_checkpoint_round_trip`: Confirms perfect Q-value reproduction after save/load cycle.
13. `test_private_and_mutation_metadata_counterfactual_safety`: Asserts policy outputs are invariant to private metadata changes.
14. `test_mock_smoke_training`: Runs 300-step smoke training with `MockBrowserAdapter`, verifying loss finiteness and target updates.
15. `test_real_browser_integration_smoke`: Executes end-to-end real browser Playwright execution with live RL step, reward extraction, and replay insertion.

---

## 16. Mock Smoke Test
`run_mock_smoke_training()` executed 300 interactions on `MockBrowserAdapter`:
- Target network updated successfully every 50 steps.
- Loss remained finite (non-zero, non-NaN, non-Inf).
- Checkpoints saved and reloaded cleanly.

---

## 17. Real-Browser Integration Smoke Test
`run_real_browser_integration_test()` executed a live Playwright session on `http://localhost:3000`:
- Reset environment to `LOGIN` workflow.
- Q-network evaluated DOM candidates extracted from real browser page.
- Action selected valid DOM element, Playwright performed real click/fill.
- Step reward and next state observation successfully stored in `DictReplayBuffer`.

---

## 18. Known Limitations
- Network architecture parameters (encoder layer sizes) are preliminary defaults for machinery verification.
- Replay capacity (10,000) and learning rate (1e-3) are smoke defaults, not hyperparameter-tuned values.

---

## 19. What Phase 10 Deliberately Does NOT Do
- Does **NOT** run full experimental training or multi-episode training curves.
- Does **NOT** implement PPO, A2C, or policy gradient methods.
- Does **NOT** perform hyperparameter searches or grid tuning.
- Does **NOT** touch or evaluate held-out Level 6 mutations.
- Does **NOT** implement multi-level curriculum schedules (reserved for Phase 11–13).

---

## 20. Readiness for Phase 11
Phase 10 machinery is complete, fully tested, and verified. The codebase is ready for **Phase 11 (Training Episode Generation & Multi-Level Workflows)** upon user review and freeze approval.
