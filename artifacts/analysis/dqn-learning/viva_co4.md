# Viva Defence Notes — CO4 Learning, Convergence & Complexity

### 1. How do you know the DQN learned?
Rolling 100-episode returns increased from negative exploration values to maximum optimal returns, while Huber loss decreased and stabilized.

### 2. What does convergence mean in this project?
It means **empirical performance stabilization**—defined as reaching and maintaining a rolling 100-episode success rate $\ge 90\%$ across progressive curriculum stages.

### 3. Did the loss monotonically decrease?
No. Huber loss exhibited brief spikes at curriculum stage transitions when new mutation levels were introduced, before stabilizing.

### 4. Why can reward fall when a new curriculum level is introduced?
New mutation levels alter DOM element attributes, text, or structure, temporarily confusing policy feature representations until replay memory incorporates updated transitions.

### 5. How many episodes were required?
A total of 15,000 curriculum training episodes (68,459 environment transitions) were executed.

### 6. What is sample efficiency?
Sample efficiency measures the amount of environment interactions required by an RL algorithm to reach a target performance threshold.

### 7. Is your DQN sample-efficient?
The agent reached 90% rolling success at Episode 463 (4,557 transitions). However, because Phase 17 found the benchmark structurally easy, this speed should not be interpreted as general sample efficiency.

### 8. What is the computational cost of action selection?
Action selection requires a forward pass through the CandidateAwareQNetwork (69,601 parameters), taking 0.6526 ms on CPU.

### 9. How does cost change with candidate count?
Candidate scoring scales linearly $\mathcal{O}(K)$ with candidate count $K$ because the candidate encoder and Q-head are shared across candidate slots.

### 10. How much memory does replay require?
The 10,000-transition replay buffer preallocates exactly 154.86 MB / 147.69 MiB of RAM (15,486 bytes per transition).

### 11. Why separate DQN inference latency from browser latency?
DQN forward pass (0.6526 ms) measures Q-network efficiency, whereas browser latency is dominated by Playwright DOM rendering and event dispatch.

### 12. Does 100% final success prove generalization?
No. Phase 17 audit demonstrated that the canonical benchmark is structurally easy, so 100% success reflects high public candidate separability.

### 13. How does Phase 17 affect interpretation of convergence?
Phase 17 shows that candidate operation filtering narrows options to 1–4 candidates, explaining why empirical stabilization occurred rapidly.

### 14. Why is empirical stabilization different from mathematical convergence?
Empirical stabilization measures trajectory success rate on finite tasks, whereas mathematical convergence guarantees optimal Q-values under infinite exploration.

### 15. What are the main resource bottlenecks?
The primary bottleneck is Playwright DOM rendering and element extraction latency, rather than neural network computation or memory storage.
