# PHASE 11 — TRAINING EPISODE GENERATION REPORT

## 1. Objective
The goal of Phase 11 is to design and implement a deterministic, leakage-safe, reproducible episode-generation system (`EpisodeGenerator`) that decides which workflow, mutation level, mutation seed, and environment seed are used for each training episode. Phase 11 provides the infrastructure for episode specification without initiating actual DQN multi-episode training or hyperparameter tuning.

---

## 2. EpisodeSpec Definition
Each training or evaluation episode is specified by an immutable `EpisodeSpec` dataclass:
```python
@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: int
    workflow: str
    mutation_level: str
    mutation_seed: int
    environment_seed: int
    generation_version: str = "phase11-v1"
```

---

## 3. Eligible Training Configuration Universe
The training-eligible base configuration universe comprises:
- **Primary Workflows**: `LOGIN`, `SEARCH`, `PROFILE`, `CHECKOUT` (4 workflows)
- **Training-Eligible Mutation Levels**: `L0`, `L1`, `L2`, `L3`, `L4`, `L5` (6 levels)
- **Mutation Seeds**: `11`, `22`, `33`, `44`, `55` (5 seeds)

**Total Base Universe**: \(4 \times 6 \times 5 = 120\) unique base configurations.

---

## 4. Why Level 6 (L6) Is Excluded
Level 6 represents complex held-out combined DOM mutations used exclusively for final out-of-distribution evaluation.
- **Hard Guard**: Attempting to include `L6` in a training schedule or `EpisodeGeneratorConfig` with `mode="train"` raises a explicit `ValueError("EXPERIMENTAL INTEGRITY VIOLATION: Level L6 is HELD OUT and CANNOT be generated or used in training mode.")`.
- **Validation**: `validate_episode_spec(spec, mode="train")` and `EpisodeGenerator.validate_schedule()` strictly enforce zero `L6` presence.

---

## 5. Seed Separation
To prevent conflating distinct random processes:
- `mutation_seed`: Controls deterministic UI mutations in the Playwright DOM framework.
- `environment_seed`: Controls Gymnasium environment candidate ordering and DOM element extraction randomness.
- `schedule_seed`: Base seed controlling deterministic episode schedule generation and block shuffling.
- `training_seed`: Controls policy network initialization and exploratory action selection (stored separately in training CLI).

---

## 6. Balanced-Block Generator
`EpisodeGenerator` implements a balanced 120-configuration block strategy:
- Each block of 120 episodes contains exactly 1 occurrence of all 120 base configurations.
- Within each block, the configuration sequence is deterministically shuffled using a sub-generator derived from `schedule_seed` and `block_index`.
- Block generation guarantees uniform coverage across workflows (30 each per block), levels (20 each per block), and seeds (24 each per block).

---

## 7. Environment-Seed Strategy
Candidate ordering in observations is randomized to ensure permutation-invariant learning.
- For each episode `episode_id`, `environment_seed` is derived deterministically using a SHA-256 hash digest over `f"{schedule_seed}:environment_seed:{episode_id}"`.
- Across repeated blocks (e.g. episode 0 vs episode 120), identical base configurations receive distinct, deterministically generated `environment_seed` values.

---

## 8. Determinism & Random Access
- **Random Access**: `generator.get_episode(episode_id)` calculates block index and position in block, returning the exact `EpisodeSpec` statelessly without state mutations.
- **Reproducibility**: Generators initialized with identical `schedule_seed` produce bitwise identical `EpisodeSpec` sequences.

---

## 9. Prefix Stability
Schedule generation exhibits strict prefix stability:
\[
\text{generate}(50) \equiv \text{generate}(500)[:50]
\]
This property enables exact checkpoint resumption and experiment re-runs.

---

## 10. Serialization & I/O
Schedules are fully serializable:
- **JSON**: `save_schedule_json()` and `load_schedule_json()` preserve `EpisodeSpec` fields and include an embedded manifest.
- **CSV**: `save_schedule_csv()` and `load_schedule_csv()` support tabular inspection.

---

## 11. Schedule SHA-256 Fingerprint Hash
`compute_schedule_hash(schedule)` calculates a 64-character SHA-256 hex digest over the canonical JSON string of the schedule list. Any modification to episode order, level, or seed alters the fingerprint hash.

---

## 12. Coverage Reporting & Inspection
`generate_coverage_report()` and `inspect_episode_schedule.py` compute:
- Workflow, level, and seed count distributions.
- Duplicate episode ID detection.
- Held-out `L6` exclusion verification (must be 0).
- Unique environment seed counts.

---

## 13. Environment Integration
`reset_from_episode_spec(env, spec, mode="train")` bridges `EpisodeSpec` to `UIRecoveryEnv.reset(seed=spec.environment_seed, options=spec.to_reset_options(mode=mode))`.
- Post-reset assertions verify `workflow_id`, `mutation_level`, and `mutation_seed` match `spec`.

---

## 14. Leakage Prevention & Audit
A source code audit confirmed that private evaluator metadata (`expected_role`, `semantic_role`, `ground_truth`, `correct_candidate`, `target_role`, `private_meta`) is 100% absent from `EpisodeSpec`, JSON/CSV outputs, manifests, and schedule reports.

---

## 15. Actual DQN Training Entry Point
**ACTUAL DQN TRAINING ENTRY POINT: NOT ENABLED YET**

**Reason**: Phase 12 train/validation/test protocol and Phase 13 curriculum/difficulty strategy must be frozen first.

No actual training command should be run yet. `rl/training/train_dqn.py` remains reserved and strictly exits cleanly without running any training loop.

---

## 16. Unit Tests
20 unit tests were added across 4 test modules:
- `test_episode_generator.py`: Configuration universe (120), L6 guard, workflow/level/seed coverage, shuffling, determinism, prefix stability, random access, env seed uniqueness.
- `test_episode_schedule.py`: JSON/CSV round-trip, SHA-256 hash stability, manifest generation, coverage reports, leakage audit.
- `test_episode_integration.py`: Gym environment reset from spec, candidate ordering diversity across env seeds, single episode runner.
- `test_training_guard.py`: CLI guard activation without `--confirm-training`.

All 20 Phase 11 tests passed in 1.48s. Total project tests: 53/53 passed.

---

## 17. What Phase 11 Deliberately Does NOT Decide
- Does **NOT** decide train/validation/test split ratios (reserved for Phase 12).
- Does **NOT** implement curriculum difficulty progression or adaptive sampling (reserved for Phase 13).
- Does **NOT** execute actual multi-episode DQN training runs.

---

## 18. Readiness for Phase 12
Phase 11 episode generation infrastructure is complete, tested, and ready for **Phase 12 (Train/Validation/Test Split)** upon user review and freeze approval.
