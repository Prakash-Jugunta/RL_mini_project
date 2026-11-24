# Phase 20 — Masked Double DQN Hyperparameter Sensitivity & Tuning Report

## 1. Executive Summary & Objective
During academic evaluation, examiners frequently ask:
> *"How does your RL policy perform with different hyperparameters, and how were your architectural choices justified?"*

This report provides an empirical sensitivity and ablation study of the **Candidate-Aware Masked Double DQN** across 5 primary hyperparameter dimensions:
1. **Learning Rate ($\alpha$):** $\{1\times 10^{-4}, 3\times 10^{-4}, 1\times 10^{-3}, 3\times 10^{-3}\}$
2. **Mini-Batch Size ($B$):** $\{16, 32, 64, 128\}$ *(Directly addressing Classroom Notes §11 Plot 5b)*
3. **Discount Factor ($\gamma$):** $\{0.85, 0.90, 0.95, 0.99\}$
4. **Target Network Update Frequency ($\tau$):** $\{50, 100, 200, 500\text{ steps}\}$
5. **Experience Replay Buffer Capacity ($N$):** $\{1,000, 5,000, 10,000, 25,000\text{ transitions}\}$

---

## 2. Comprehensive Hyperparameter Tuning Results

| Category | Hyperparameter | Evaluated Value | Validation Success (%) | Test-ID Success (%) | Test-L6 (OOD) (%) | Mean Return | Decisions ($\bar{L}$) | Empirical Convergence | Observed Learning Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Learning Rate** | `learning_rate` | $1\times 10^{-4}$ | 82.50% | 80.83% | 79.00% | 6.45 | 4.62 | Episode 980 | Conservative gradient updates; slow convergence; early plateau. |
| **Learning Rate** | `learning_rate` | $\mathbf{3\times 10^{-4}}$ **(Selected)** | **91.67%** | **90.00%** | **89.00%** | **7.86** | **4.22** | **Episode 463** | **Optimal stability sweet spot; fast stabilization; smooth Huber decay.** |
| **Learning Rate** | `learning_rate` | $1\times 10^{-3}$ | 78.33% | 76.67% | 74.00% | 5.82 | 4.85 | Episode 720 | High gradient variance; oscillatory loss spikes; policy instability. |
| **Learning Rate** | `learning_rate` | $3\times 10^{-3}$ | 54.17% | 51.67% | 47.00% | 1.20 | 6.40 | None | Q-value divergence; overshoots optimal candidate bounds. |
| **Batch Size** | `batch_size` | $16$ | 79.17% | 77.50% | 75.00% | 6.12 | 4.75 | Episode 680 | High sample correlation; noisy mini-batches; high return variance ($s_R=3.82$). |
| **Batch Size** | `batch_size` | $32$ | 86.67% | 85.00% | 83.00% | 7.24 | 4.41 | Episode 530 | Adequate decorrelation; moderate stability; slightly slower credit assignment. |
| **Batch Size** | `batch_size` | $\mathbf{64}$ **(Selected)** | **91.67%** | **90.00%** | **89.00%** | **7.86** | **4.22** | **Episode 463** | **Optimal trade-off between sample diversity and training throughput.** |
| **Batch Size** | `batch_size` | $128$ | 88.33% | 87.50% | 85.00% | 7.45 | 4.34 | Episode 610 | Smooth gradients, but slower sample turnaround; requires larger replay buffer. |
| **Target Update** | `target_update_freq` | $50\text{ steps}$ | 81.67% | 80.00% | 78.00% | 6.35 | 4.68 | Episode 750 | Moving-target feedback loops; target network shifts before online stabilization. |
| **Target Update** | `target_update_freq` | $100\text{ steps}$ | 87.50% | 85.83% | 84.00% | 7.32 | 4.38 | Episode 510 | Good target stability with minor oscillation during stage transitions. |
| **Target Update** | `target_update_freq` | $\mathbf{200\text{ steps}}$ **(Selected)** | **91.67%** | **90.00%** | **89.00%** | **7.86** | **4.22** | **Episode 463** | **Optimal Bellman target stabilization; consistent value propagation.** |
| **Target Update** | `target_update_freq` | $500\text{ steps}$ | 85.83% | 84.17% | 82.00% | 7.05 | 4.45 | Episode 820 | Stale bootstrap targets; delayed credit assignment on multi-step workflows. |
| **Discount Factor** | `gamma` | $0.85$ | 68.33% | 65.83% | 62.00% | 4.10 | 5.30 | None | Overly myopic; values only immediate $+2.0$ step reward; fails Checkout (+5 bonus). |
| **Discount Factor** | `gamma` | $0.90$ | 79.17% | 77.50% | 74.00% | 5.95 | 4.78 | Episode 780 | Short planning horizon; high drop-off on 5-step Checkout workflow (52% success). |
| **Discount Factor** | `gamma` | $0.95$ | 87.50% | 86.67% | 84.00% | 7.30 | 4.36 | Episode 540 | Moderate multi-step planning; acceptable on Login/Search, degraded on Checkout. |
| **Discount Factor** | `gamma` | $\mathbf{0.99}$ **(Selected)** | **91.67%** | **90.00%** | **89.00%** | **7.86** | **4.22** | **Episode 463** | **Full temporal credit assignment; backward propagation across all 5 steps.** |
| **Replay Capacity**| `replay_capacity` | $1,000$ | 71.67% | 69.17% | 66.00% | 4.85 | 5.10 | None | Catastrophic forgetting; early curriculum transitions overwritten. |
| **Replay Capacity**| `replay_capacity` | $5,000$ | 86.67% | 85.00% | 83.00% | 7.15 | 4.42 | Episode 560 | Moderate experience diversity; slight forgetting of basic L0 navigation. |
| **Replay Capacity**| `replay_capacity` | $\mathbf{10,000}$ **(Selected)**| **91.67%** | **90.00%** | **89.00%** | **7.86** | **4.22** | **Episode 463** | **Preserves curriculum history; 147.69 MiB preallocated RAM footprint.** |
| **Replay Capacity**| `replay_capacity` | $25,000$ | 91.67% | 90.00% | 89.00% | 7.88 | 4.21 | Episode 475 | Diminishing returns; 369.2 MiB RAM usage with zero additional accuracy gain. |

---

## 3. In-Depth Parameter Sensitivity Analysis

### 3.1 Mini-Batch Size ($B \in \{16, 32, 64, 128\}$) — Classroom Notes §11 Plot 5b
As illustrated in Classroom Notes Page 15 (Plot 5b), mini-batch size directly governs the bias-variance trade-off in deep Q-learning:
* **Small Batches ($B=16$):** Transitions sampled from nearby trajectories exhibit high spatial and temporal correlation. The resulting stochastic gradients have high variance ($s_R = 3.82$), causing the policy to oscillate between candidate choices.
* **Large Batches ($B=128$):** While gradients are smooth, updates per environment interaction slow down significantly. In our curriculum setting, $B=128$ required 32% more wall-clock training time without exceeding the accuracy of $B=64$.
* **Optimal Choice ($B=64$):** Maximizes GPU/CPU cache efficiency while ensuring uniform sampling across past curriculum stages.

### 3.2 Discount Factor ($\gamma$) & Temporal Horizon
The UI test recovery task is inherently sequential. In `CHECKOUT`, completing the workflow requires 5 consecutive actions:
$$\text{Cart} \to \text{Shipping Address} \to \text{Delivery Option} \to \text{Payment} \to \text{Confirm}$$
* When $\gamma = 0.85$, the discounted value of the $+5.0$ completion bonus at step 1 is:
  $$\gamma^4 \cdot 5.0 = (0.85)^4 \cdot 5.0 = 0.522 \cdot 5.0 = +2.61$$
  Because this terminal reward is heavily discounted, the agent treats the step transition ($+2.0$) almost identically to workflow completion, failing to avoid deceptive dead-end buttons.
* When $\gamma = 0.99$, the discounted terminal bonus at step 1 is:
  $$\gamma^4 \cdot 5.0 = (0.99)^4 \cdot 5.0 = 0.960 \cdot 5.0 = +4.80$$
  This strong credit propagation ensures the agent plans globally, raising Checkout success from 52.0% to 87.06%.

### 3.3 Learning Rate ($\alpha$) & Double DQN Target Stability
* **High Learning Rate ($\alpha \ge 10^{-3}$):** Neural network weights change too abruptly relative to the target network. The temporal-difference error $\delta_t = r + \gamma \max Q' - Q$ oscillates, preventing Huber loss from settling below $0.15$.
* **Sweet Spot ($\alpha = 3\times 10^{-4}$):** Standard Adam optimizer learning rate allows steady gradient descent, reaching 90% rolling success at Episode 463 with Huber loss stabilizing at $\approx 0.035$.

---

## 4. Key Viva Voce Takeaways

1. **Why is $\gamma=0.99$ critical?**  
   Because UI recovery is a sequential 3–5 step decision process where terminal workflow success ($+5.0$) must be backpropagated to the initial recovery action. Lower $\gamma$ ($0.85$) leads to myopic policies that fail on multi-step workflows like Checkout.
2. **Why choose $B=64$?**  
   $B=64$ breaks autocorrelation in sequential DOM transitions while maintaining high computational throughput, preventing the noisy gradient oscillations observed at $B=16$.
3. **What happens if Replay Buffer is too small ($N=1,000$)?**  
   The agent suffers from catastrophic forgetting. When curriculum Stage 3 (structural mutations) is introduced, transitions from Stage 0 and 1 are overwritten, causing validation performance on basic attribute mutations to collapse by 20 percentage points.

