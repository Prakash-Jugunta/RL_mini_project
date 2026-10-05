# Empirical Performance Stabilization Report

## Operational Convergence Criterion
In this project, **convergence** refers to **empirical performance stabilization**—defined as reaching and maintaining a rolling 100-episode success rate above a defined threshold across progressive curriculum stages.

| Target Threshold | First Episode Reaching | Step at First | Stable After Episode | Step at Stable | Evidence |
|---|---|---|---|---|---|
| `50% rolling success` | **135** | 1929 | **135** | 1929 | Rolling 100-episode window success rate reached 50% |
| `80% rolling success` | **422** | 4351 | **422** | 4351 | Rolling 100-episode window success rate reached 80% |
| `90% rolling success` | **463** | 4557 | **463** | 4557 | Rolling 100-episode window success rate reached 90% |
| `95% rolling success` | **483** | 4640 | **813** | 6272 | Rolling 100-episode window success rate reached 95% |
| `99% rolling success` | **634** | 5391 | **14994** | 68435 | Rolling 100-episode window success rate reached 99% |

## Stage Transition Dynamics & Adaptation
- **Stage 0 (L0)**: Initial exploration; rolling success rate begins at 0% and stabilizes as Q-network learns basic navigation.
- **Stage 1 (L0-L1)**: Introduction of ID & Class attribute mutations causes temporary adaptation dips before stabilizing.
- **Stage 2 to 5 (L0-L5)**: Progressive mixed curriculum introduces text, structural, distractor, and compound mutations. Rolling success rate remains high with brief adaptation plateaus.

**NOTE**: Phase 17 identified that the canonical benchmark is structurally easy. Therefore, fast stabilization must be interpreted in the context of high candidate separability rather than mathematical optimality on unbounded domains.