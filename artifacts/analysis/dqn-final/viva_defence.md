# Viva Defence Notes — Self-Healing UI Test Automation via Masked Double DQN

### 1. Why was DQN chosen?
DQN (specifically Masked Double DQN) was chosen because UI element selection in self-healing test automation is a discrete decision problem over a candidate set of variable size. Double DQN mitigates Q-value overestimation, and action masking strictly prevents invalid candidate selection.

### 2. What exactly does the DQN predict?
The DQN predicts state-action Q-values \(Q(s, a_i)\) for each candidate element \(i \in \{1, \dots, K\}\) in the DOM observation, representing the expected cumulative discounted return of selecting candidate \(a_i\) to achieve the target workflow step.

### 3. What constitutes a state?
State \(s\) is represented as a multimodal embedding combining:
1. Candidate feature vectors (element tag, type, text/placeholder embeddings, bounding box, visibility),
2. Workflow objective embedding (current workflow + step ID),
3. Previous action context embedding (last executed candidate features).

### 4. What constitutes an action?
An action \(a \in \{0, \dots, K-1\}\) represents selecting the candidate element at index \(a\) from the extracted candidate array to perform the required action (`fill` or `click`).

### 5. What constitutes a reward?
The step reward function provides:
- Step completion: \(+2.0\) for executing the correct semantic candidate,
- Workflow completion: \(+5.0\) final bonus upon workflow success,
- Wrong candidate penalty: \(-1.0\) for choosing an incorrect valid candidate,
- Step budget penalty: \(-0.05\) per step decision to encourage efficiency.

### 6. What does episode success mean?
Episode success means the agent completed every required step of the target workflow (`LOGIN`, `SEARCH`, `PROFILE`, `CHECKOUT`) within the maximum decision budget without unrecoverable page or execution failures.

### 7. Why is Test-L6 important?
Test-L6 represents complex, 3-to-4 compound DOM mutations (`id+dom+distractor`, `text+type+position`, `id+text+dom+distractor`) that were strictly held out during training and validation. It evaluates true Out-of-Distribution (OOD) policy robustness.

### 8. What is Test-ID vs Test-L6?
- **Test-ID**: In-Distribution test split using mutation levels L0–L5 with environment seeds distinct from training.
- **Test-L6**: Out-of-Distribution test split using held-out Level 6 compound mutations.

### 9. How did you prevent leakage?
1. Split assignment was fixed via deterministic SHA-256 hashing on episode specifications.
2. Private evaluator metadata (`data-semantic-role`, ground-truth expected roles) was stripped from candidate observations.
3. Level 6 episodes were strictly prohibited from training and hyperparameter tuning.

### 10. Why can candidate recall limit DQN?
DQN selects actions from the candidate set provided by the candidate extractor. If the correct UI element is not included in the extracted candidate array, the agent cannot select it, capping policy performance regardless of Q-network quality.

### 11. Why is success rate more meaningful than "accuracy" here?
Success rate measures end-to-end task completion across sequential multi-step decision trajectories, whereas point accuracy ignores sequential compounding errors and state transitions.

### 12. Why can browser performance be lower than simulator performance?
Real browser execution introduces DOM rendering delays, dynamic event timing, layout reflows, and real Playwright interaction constraints not present in simplified deterministic simulators.

### 13. How was the final checkpoint selected?
The final checkpoint (`artifacts/dqn/phase13-v1/latest_checkpoint.pt`) was selected based on peak validation success rate during Stage 5 curriculum training without accessing Test-L6.

### 14. Was L6 used for training or hyperparameter tuning?
No. Level 6 was strictly isolated and evaluated once during final post-training evaluation.

### 15. What are the main limitations of the project?
1. Dependency on the DOM candidate extractor's recall.
2. Action space capped at \(K=20\) candidates per step.
3. Focused on single-page / multi-step web application workflows rather than unbounded web crawling.
