# Phase 19 — PPO & A2C Policy Leakage Audit Report

## 1. Audit Overview
This audit inspects the newly implemented PPO (`rl/agents/ppo`) and A2C (`rl/agents/a2c`) policy pipelines to ensure strict compliance with the project's public/private information boundary.

---

## 2. Policy Input Fields Inspected

The policy networks (`CandidateAwareActorCriticNetwork`) and rollout buffers (`PPORolloutBuffer`, `A2CRolloutBuffer`) were audited for the presence of forbidden private evaluator fields:

| Forbidden Private Field | Found in Policy Input? | Found in Rollout Buffer State? | Status |
|---|---|---|---|
| `semantic_role` | NO | NO | **PASS** |
| `expected_role` | NO | NO | **PASS** |
| `target_role` | NO | NO | **PASS** |
| `data-semantic-role` | NO | NO | **PASS** |
| `ground_truth` | NO | NO | **PASS** |
| `correct_candidate` | NO | NO | **PASS** |
| `target_index` | NO | NO | **PASS** |
| `expected_locator` | NO | NO | **PASS** |
| `mutation_seed` | NO | NO | **PASS** |
| `mutation_level` | NO | NO | **PASS** |
| `curriculum_stage` | NO | NO | **PASS** |

---

## 3. Permutation & Representation Audit

- **Actor Permutation Equivariance**: Verified by `test_actor_permutation_equivariance` in [test_actor_critic_network.py](file:///e:/sem7/RL/RL_replay/rl/tests/test_actor_critic_network.py). Permuting candidate slots permutes output policy logits identically.
- **Critic Permutation Invariance**: Verified by `test_critic_permutation_invariance`. Permuting valid candidate order leaves state value $V(s)$ invariant.
- **Action Selection**: Evaluator uses private workflow specs to execute DOM operations and assign rewards **only after** policy selects candidate slot $a \in [0, 19]$.

---

## 4. Leakage Audit Conclusion

```text
============================================================
 POLICY LEAKAGE AUDIT RESULT
============================================================
 Policy Observations: PUBLIC FIELDS ONLY (objective, context, candidates, candidate_mask)
 Rollout Buffer State:  NO PRIVATE METADATA
 Policy Permutation:    EQUIVARIANT ACTOR / INVARIANT CRITIC

 POLICY LEAKAGE: PASS
============================================================
```
