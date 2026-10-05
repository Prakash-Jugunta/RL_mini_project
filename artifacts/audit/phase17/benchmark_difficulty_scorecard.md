# Benchmark Difficulty & Saturation Scorecard

## Evidence Assessment Matrix

| Heading | Evidence Finding | Concern Level |
|---|---|---|
| **A. Direct Leakage** | `sanitize_attributes()` strips all private roles; `PrivateEvaluatorMetadata` kept strictly in env | **NOT OBSERVED (PASS)** |
| **B. Candidate Recall** | Target element present in extracted candidate list for 100% of Test-L6 decisions | **NOT OBSERVED (PASS)** |
| **C. Number of Alternatives** | Mean candidates = 11.2 per step; operation type filtering leaves ~2–4 valid options | **MODERATE CONCERN** |
| **D. Q-Value Separation** | DQN ranks target at #1 with large mean Q margin (+1.45) across decisions | **LOW CONCERN** |
| **E. Simple Heuristic Separability** | Public feature heuristic achieves 100% success without RL weights | **HIGH CONCERN** |
| **F. Feature Ablation Sensitivity** | Zeroing text features drops success to 0%; structural/visual removal has <5% impact | **MODERATE CONCERN** |
| **G. L6 Primitive Novelty** | L6 combines 3–4 primitive mutations whose individual operators were observed in L0–L5 | **MODERATE CONCERN** |
| **H. Adversarial Robustness** | Agent maintains 100% success under AUDIT-A distractors | **LOW CONCERN** |
| **I. Cross-Workflow Diversity** | Evaluated across 4 distinct web app workflows (LOGIN, SEARCH, PROFILE, CHECKOUT) | **LOW CONCERN** |

---

## Final Scientific Conclusion

**CONCLUSION: 2. No direct leakage, but benchmark appears structurally easy / saturated.**

### Rationale
1. **Zero Direct Leakage**: The static leakage audit proves that private semantic roles and evaluator metadata are strictly excluded from agent observations.
2. **Benchmark Saturation**: The 100% success rate across all methods (DQN and Public Heuristic) occurs because web application workflows feature distinct element operation types (`input` vs `button`) and unambiguous text cues that make the target candidates trivially separable.
