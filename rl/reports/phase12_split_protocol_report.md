# PHASE 12 — TRAIN / VALIDATION / TEST SPLIT PROTOCOL REPORT

## 1. Objective
The goal of Phase 12 is to define and implement a leakage-safe, reproducible train / validation / test split strategy for the RL-based self-healing UI automation project. Phase 12 ensures that DQN model selection, checkpoint evaluation, and hyperparameter decisions are strictly separated from final generalization testing.

---

## 2. Why a Split Protocol is Required
Reinforcement learning agents risk overfitting to specific DOM mutations or candidate element orderings if development decisions are made on training episodes. Establishing an explicit partition separates:
- **Weight Updates**: Policy network optimization during training.
- **Development & Model Selection**: Hyperparameter tuning, checkpoint selection, early stopping, and curriculum decisions.
- **Out-of-Distribution Generalization**: Unbiased final performance evaluation on unseen environment realizations and held-out mutation combinations.

---

## 3. Existing L6 Holdout Contract
Level 6 (`L6`) represents complex held-out combined DOM mutations explicitly designed as the project's final generalization benchmark.
- `L6` is **strictly prohibited** from `TRAIN` and `VALIDATION` splits.
- Any attempt to assign an `L6` episode to `train` or `validation` raises a hard `ValueError("EXPERIMENTAL INTEGRITY VIOLATION")`.

---

## 4. Split Strategy
Phase 12 employs a **two-axis split strategy**:
1. **Mutation-Level Axis**: Separates `L0–L5` (train-eligible mutations) from `L6` (held-out combined mutations).
2. **Environment-Seed Axis**: Partitioning environment candidate ordering seeds deterministically using SHA-256 stable hashing.

---

## 5. Train Definition
- **Content**: `L0–L5` mutation levels across all 4 workflows (`LOGIN`, `SEARCH`, `PROFILE`, `CHECKOUT`) and all 5 mutation seeds (`11`, `22`, `33`, `44`, `55`).
- **Partition**: 80% ratio of environment candidate-ordering seeds per base stratum.
- **Canonical Count**: 960 EpisodeSpecs (120 strata \(\times\) 8 environment realizations).
- **Gym Mode**: `env.reset(options={"mode": "train"})`.

---

## 6. Validation Definition
- **Content**: `L0–L5` mutation levels.
- **Partition**: 10% held-out environment-seed realizations per base stratum.
- **Canonical Count**: 120 EpisodeSpecs (120 strata \(\times\) 1 environment realization).
- **Purpose**: Hyperparameter comparison, checkpoint selection, early stopping, and curriculum progression.
- **Gym Mode**: `env.reset(options={"mode": "validation"})`.

---

## 7. Test-ID Definition (In-Distribution Test)
- **Content**: `L0–L5` mutation levels.
- **Partition**: 10% held-out environment-seed realizations per base stratum.
- **Canonical Count**: 120 EpisodeSpecs (120 strata \(\times\) 1 environment realization).
- **Purpose**: Evaluates candidate-order robustness on unseen realizations of known mutation levels.
- **Gym Mode**: `env.reset(options={"mode": "test"})`.

---

## 8. Test-L6 Definition (Held-Out Generalization Test)
- **Content**: `L6` mutation level exclusively.
- **Partition**: All 4 workflows \(\times\) 5 mutation seeds = 20 base configurations, each evaluated over 5 distinct environment realizations.
- **Canonical Count**: 100 EpisodeSpecs (20 base configs \(\times\) 5 environment realizations).
- **Purpose**: Unbiased final benchmark on unseen, held-out DOM mutation combinations.
- **Gym Mode**: `env.reset(options={"mode": "test"})`.

---

## 9. Seed Separation
The experiment protocol maintains strict seed independence:
- `mutation_seed`: Playwright DOM mutation framework seed (11, 22, 33, 44, 55).
- `environment_seed`: Gymnasium candidate ordering seed.
- `schedule_seed`: Episode order shuffling seed.
- `split_seed`: Base seed controlling SHA-256 split partitioning (default 1201).
- `training_seed`: Network parameter initialization seed.

---

## 10. Stratification
Each of the 120 base strata (\(4 \text{ workflows} \times 6 \text{ levels} \times 5 \text{ mutation seeds}\)) is independently stratified across environment-seed realizations:
- **Stratum Realization Allocation**: 8 TRAIN, 1 VALIDATION, 1 TEST-ID per 10-repeat stratum.
- **Guarantees**: Uniform workflow, mutation level, and seed proportions across all three L0–L5 splits.

---

## 11. Canonical Materialization Summary
| Split Category | Mutation Levels | Episode Count | SHA-256 Fingerprint Hash |
| :--- | :--- | :---: | :--- |
| **TRAIN Reference** | L0–L5 | 960 | `07ab62a4ce64b3a65dc5e80368bcb3747ea2175c001f1a734f4adc1c3be9bbb2` |
| **VALIDATION** | L0–L5 | 120 | `c2434ed75835078d010313e0b4396c62e65d0b49028c613d3a85ff1b41c38234` |
| **TEST-ID** | L0–L5 | 120 | `4d92a5248eb992f570225ac2bf0ad3e47f80bc021b5106df7a1fdc72d8f5d7fd` |
| **TEST-L6** | L6 | 100 | `4210cf0cbd276f96a28015b53b4c8732aef782890a6c9c04c270e501bd3e04ed` |

---

## 12. Disjointness Guarantees
Unit test `test_disjointness_and_l6_isolation` verifies:
- \(\text{TRAIN} \cap \text{VALIDATION} = \emptyset\)
- \(\text{TRAIN} \cap \text{TEST-ID} = \emptyset\)
- \(\text{TRAIN} \cap \text{TEST-L6} = \emptyset\)
- \(\text{VALIDATION} \cap \text{TEST-ID} = \emptyset\)
- \(\text{VALIDATION} \cap \text{TEST-L6} = \emptyset\)
- \(\text{TEST-ID} \cap \text{TEST-L6} = \emptyset\)
- Environment seed sets within each stratum across splits are strictly disjoint.

---

## 13. Replay & Optimizer Leakage Prevention
- **Replay Buffer Isolation**: Unit test `test_validation_and_test_never_enter_replay_buffer` proves transitions from `validation`, `test_id`, and `test_l6` episodes are never stored in the training replay buffer.
- **Optimizer Isolation**: Unit test `test_validation_and_test_never_update_optimizer` proves executing validation or test steps never alters Q-network weights or triggers optimizer steps.

---

## 14. Model-Selection Policy
- Checkpoint selection, early stopping, and hyperparameter tuning must be decided strictly using **VALIDATION** metrics (e.g. validation workflow success rate).
- **TEST-ID** and **TEST-L6** metrics must remain hidden from training control flow.

---

## 15. L6 Isolation
- Hard guards in `assign_split()`, `validate_episode_spec()`, and `load_schedule_json()` reject `L6` from training or validation splits.
- `L6` appears exclusively in `test_l6`.

---

## 16. Reproducibility
- Pure, platform-independent SHA-256 hash partitioning.
- Independent of Python's built-in `hash()` or episode generation sequence length (`schedule-length independence`).
- Re-running materialization with `split_seed=1201` reproduces bitwise identical schedules and manifest hashes.

---

## 17. Unit Tests
11 Phase 12 unit tests passed in 1.44s:
- `test_splitter.py`: Split assignment, L6 routing, env mode mapping, schedule-length independence.
- `test_split_integrity.py`: Canonical counts, set disjointness, stratum stratification, metadata leak audit.
- `test_split_integration.py`: Replay buffer isolation, optimizer isolation, Gym reset mode mapping.

Total project tests: 68/68 passed.

---

## 18. Limitations
- Canonical 960 `TRAIN` reference specs define a baseline dataset; infinite training generation from the training-eligible seed partition is supported.
- Ratios are protocol defaults (80/10/10) for machinery validation.

---

## 19. What Phase 12 Deliberately Does NOT Decide
- Does **NOT** decide curriculum difficulty ordering (e.g. L0 first vs L1 vs mixed) — reserved for Phase 13.
- Does **NOT** execute actual multi-episode DQN training runs.
- Does **NOT** alter network architectures, learning rates, or reward parameters.

---

## 20. Readiness for Phase 13
Phase 12 split protocol is complete, tested, and ready for **Phase 13 (Curriculum & Difficulty Progression)** upon user review and freeze approval.
