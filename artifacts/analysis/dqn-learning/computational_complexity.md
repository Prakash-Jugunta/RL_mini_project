# Computational Complexity & Architectural Scaling Report

## 1. Candidate-Aware Q-Network Forward Pass Scaling
The `CandidateAwareQNetwork` scores $K$ candidate UI elements in parallel using shared linear projections:

1. **Objective Encoding**: $\mathcal{O}(D_{\text{obj}} \cdot H_{\text{obj}}) = 64 \times 128 + 128 \times 64 = 16,384$ FLOPs (computed **once** per state).
2. **Context Encoding**: $\mathcal{O}(D_{\text{ctx}} \cdot H_{\text{ctx}}) = 30 \times 64 + 64 \times 32 = 3,968$ FLOPs (computed **once** per state).
3. **Candidate Encoding**: Shared candidate encoder maps each candidate slot:
   $$\mathcal{O}(K \cdot (91 \times 128 + 128 \times 64)) = K \cdot 19,840 \text{ FLOPs}$$
4. **Q-Value Scoring**: Shared Q-head maps concatenated features $(64 + 32 + 64 = 160)$:
   $$\mathcal{O}(K \cdot (160 \times 128 + 128 \times 64 + 64 \times 1)) = K \cdot 28,736 \text{ FLOPs}$$
5. **Action Selection**: Masked argmax over $K$ valid candidates: $\mathcal{O}(K)$.

---

## 2. Overall Time Complexity
For $K$ candidates (max capacity $K=20$):
$$\text{Time Complexity} = \mathcal{O}(K \cdot D_{\text{cand}} + D_{\text{state}})$$

Candidate scoring scales **linearly $\mathcal{O}(K)$** with the number of extracted candidates because the shared Candidate Encoder and Q-Scorer networks are evaluated per candidate slot.

---

## 3. Space Complexity
- Observation Tensor Space: $\mathcal{O}(K \cdot D_{\text{cand}} + D_{\text{obj}} + D_{\text{ctx}}) = \mathcal{O}(20 \times 91 + 64 + 30) = 1,914 \text{ float32 values}$.
- Model Parameters: $69,601$ trainable parameters ($pprox 278.40 \text{ KB} / 271.88 \text{ KiB}$).
