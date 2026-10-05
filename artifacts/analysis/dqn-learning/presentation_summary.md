# CO4 Summary — Learning, Complexity & Resource Analysis

## Learning Behaviour
The Masked Double DQN agent was trained over 15,000 curriculum episodes (68,459 total environment transitions). Training return steadily increased from negative exploration values to maximum optimal returns across 6 progressive curriculum stages.

## Convergence / Empirical Stabilization
- **First Reached 90% Rolling Success**: Episode 463 (Step 4,557)
- **Stable Above 90% Thereafter**: Episode 463 (Step 4,557)

## Sample Efficiency Assessment
The DQN reached the defined 90% rolling-success threshold after 463 episodes (4,557 environment interactions). Phase 17 found the canonical benchmark to be structurally easy/saturated, therefore the observed learning speed should not be interpreted as evidence of general sample efficiency.

## Computational Complexity
Action candidate scoring scales **linearly $\mathcal{O}(K)$** with candidate count $K=20$. Model forward pass latency is **0.6526 ms** (CPU), adding minimal overhead to Playwright browser execution.

## Resource Requirements
- **Trainable Parameters**: 69,601 float32 parameters (278.4 KB / 271.88 KiB).
- **Checkpoint Size**: 1.14 MB on disk.
- **Replay Buffer Memory**: 154.86 MB / 147.69 MiB at 10,000 transition capacity.

## Important Limitation
Phase 17 audit revealed that the canonical benchmark is structurally easy / saturated. Fast empirical stabilization reflects strong public candidate separability rather than optimal policy learning on unbounded domains.
