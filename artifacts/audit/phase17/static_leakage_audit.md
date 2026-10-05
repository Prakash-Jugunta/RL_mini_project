# Static Leakage Audit Report

## Policy Observation Pipeline Trace
```text
Browser DOM -> Candidate Extractor -> UICandidate -> StateEncoder -> Dict Observation -> Masked DQNAgent
```

| Field Symbol | Source File | Function / Location | Classification | Accessible to Policy Network? |
|---|---|---|---|---|
| `data-semantic-role` | `frontend/src/mutations/mutationEngine.js` | `applyMutations` | **EVALUATOR-ONLY** | NO — Stripped by sanitize_attributes() before candidate creation. |
| `semantic_role` | `rl/env/types.py` | `sanitize_attributes()` | **EVALUATOR-ONLY** | NO — Contained in PrivateEvaluatorMetadata, strictly prohibited from UICandidate. |
| `expected_role` | `rl/env/workflow.py` | `WorkflowStep` | **EVALUATOR-ONLY** | NO — Used solely by BrowserAdapter.validate_step() for private evaluator step check. |
| `target_role` | `rl/env/browser_adapter.py` | `get_candidates()` | **EVALUATOR-ONLY** | NO — Filtered out during agent candidate extraction. |
| `correct_candidate` | `rl/reward/calculator.py` | `RewardCalculator` | **TRAINING TARGET** | NO — Evaluated inside environment step() to calculate reward, not in observation. |
| `oracle` | `rl/splits/validator.py` | `validate_split_integrity()` | **TEST-ONLY** | NO — Evaluator sanity validator. |
| `ground_truth` | `rl/env/types.py` | `PROHIBITED_ATTRIBUTE_KEYS` | **EVALUATOR-ONLY** | NO — Prohibited key list in UICandidate post-init guard. |
| `expected_locator` | `rl/evaluation/evaluate_baselines.py` | `BRITTLE_WORKFLOW_MAP` | **TEST-ONLY** | NO — Used only by brittle baseline for fixed Playwright selectors. |
| `target_index` | `rl/evaluation/evaluate_candidate_recall.py` | `evaluate_recall()` | **DEBUG-ONLY** | NO — Diagnostic recall measurement script. |
| `mutation_seed` | `rl/training/episode_spec.py` | `EpisodeSpec` | **EVALUATOR-ONLY** | NO — Passed to reset(options={...}) for browser mutation setup, excluded from StateEncoder. |
| `mutation_level` | `rl/training/episode_spec.py` | `EpisodeSpec` | **EVALUATOR-ONLY** | NO — Passed to reset(options={...}) for browser mutation setup, excluded from StateEncoder. |
| `workflow_answer` | `rl/env/workflow.py` | `WorkflowDefinition` | **EVALUATOR-ONLY** | NO — Page-state success assertion value for step validation. |
| `private_metadata` | `rl/env/browser_adapter.py` | `get_candidates()` | **EVALUATOR-ONLY** | NO — Second element of return tuple (agent_candidates, private_metadata) kept strictly in env. |

## Ground-Truth Protection Audit Summary
- **`UICandidate.__post_init__` Guard**: `sanitize_attributes()` automatically strips all keys matching `PROHIBITED_ATTRIBUTE_KEYS` (`semantic_role`, `data-semantic-role`, `expected_role`, `is_correct`, `ground_truth`, `target_role`).
- **`PrivateEvaluatorMetadata` Separation**: Returned in a separate tuple element during extraction and kept strictly inside `UIRecoveryEnv`.
- **`StateEncoder` Inspection**: Uses strictly public fields (`tag`, `element_type`, `text`, `placeholder`, `aria_label`, `visible`, `enabled`, `x`, `y`, `width`, `height`, `attributes`).

**CONCLUSION**: Direct private label leakage is **NOT OBSERVED** (PASS).