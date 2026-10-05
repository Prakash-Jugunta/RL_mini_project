# Phase 17 Q-Value Margin Reconciliation Report

## 1. Executive Summary
During Phase 17 audit execution, two values were observed for the mean Q-value margin:
- **`0.1141`**: Empirical mean Q-margin calculated across all 80 decision steps evaluated by the frozen Masked Double DQN (`latest_checkpoint.pt`) in `q_value_decisions.csv`.
- **`1.4500`**: Diagnostic fallback value present in early script drafts prior to empirical CSV aggregation.

---

## 2. Mathematical Definition
For each valid candidate decision step $t$, let $Q(s, a_i)$ represent the action Q-value computed by the online Q-network under candidate mask $M(a) = 1$.
$$a^* = \arg\max_{a: M(a)=1} Q(s, a)$$
$$a_{\text{runner\_up}} = \arg\max_{a: M(a)=1, a \neq a^*} Q(s, a)$$
$$\text{Q Margin} = Q(s, a^*) - Q(s, a_{\text{runner\_up}})$$

---

## 3. Empirical Discrepancy Breakdown

| Metric Value | Origin / Source | Calculation Method | Status |
|---|---|---|---|
| **`0.1141`** | `artifacts/audit/phase17/q_value_decisions.csv` | Mean of $(Q_{\text{top}} - Q_{\text{runner\_up}})$ across all 80 evaluated decisions | **AUTHORITATIVE EMPIRICAL VALUE** |
| **`1.4500`** | Early CLI fallback string in `audit_benchmark.py` | Static fallback constant before CSV generation completed | **REPLACED BY EMPIRICAL DATA** |

---

## 4. Summary Statistics for Authoritative Q Margin (`0.1141`)
- **Mean Q Margin**: `0.1141`
- **Median Q Margin**: `0.0541`
- **Min Q Margin**: `0.0016`
- **P10 Q Margin**: `0.0099`
- **P90 Q Margin**: `0.3420`

---

## 5. Conclusion
The authoritative, empirically measured mean Q-value margin for the frozen Masked Double DQN checkpoint on Test-L6 is **`0.1141`**. Historical artifacts have been reconciled without altering underlying raw CSV decision data.
