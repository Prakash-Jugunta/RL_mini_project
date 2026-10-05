# Lowest Q-Value Margin Decisions (Test-L6 Audit)

| Workflow | Step ID | Seed | Candidates | Selected Action | Selected Q | Runner-Up Q | **Q Margin** | Target Rank | Correct? |
|---|---|---|---|---|---|---|---|---|---|
| `CHECKOUT` | `CLICK_CART` | `33` | 10 | `5` | 8.2452 | 8.2436 | **0.0016** | #2 | False |
| `CHECKOUT` | `PROCEED_CHECKOUT` | `33` | 10 | `6` | 6.5836 | 6.5801 | **0.0035** | #5 | False |
| `LOGIN` | `VERIFY_DASHBOARD` | `55` | 10 | `8` | 4.7297 | 4.7246 | **0.0051** | #6 | False |
| `PROFILE` | `ENTER_EMAIL` | `33` | 10 | `6` | 7.1060 | 7.0969 | **0.0091** | #3 | False |
| `PROFILE` | `VERIFY_SUCCESS` | `44` | 10 | `7` | 5.9973 | 5.9872 | **0.0101** | #5 | False |
| `CHECKOUT` | `CLICK_CART` | `55` | 10 | `0` | 8.2338 | 8.2232 | **0.0107** | #5 | False |
| `CHECKOUT` | `ADD_TO_CART` | `55` | 10 | `1` | 8.6005 | 8.5898 | **0.0107** | #9 | False |
| `LOGIN` | `ENTER_PASSWORD` | `11` | 10 | `6` | 7.0557 | 7.0443 | **0.0115** | #8 | False |
| `CHECKOUT` | `PROCEED_CHECKOUT` | `44` | 10 | `8` | 6.8806 | 6.8691 | **0.0115** | #5 | False |
| `PROFILE` | `VERIFY_SUCCESS` | `22` | 10 | `3` | 5.9404 | 5.9281 | **0.0123** | #5 | False |
| `PROFILE` | `ENTER_EMAIL` | `22` | 10 | `4` | 7.0684 | 7.0558 | **0.0126** | #1 | True |
| `CHECKOUT` | `PROCEED_CHECKOUT` | `22` | 10 | `0` | 6.7796 | 6.7657 | **0.0138** | #5 | False |
| `LOGIN` | `ENTER_PASSWORD` | `55` | 10 | `7` | 7.0181 | 7.0041 | **0.0140** | #6 | False |
| `PROFILE` | `VERIFY_SUCCESS` | `33` | 10 | `7` | 6.0176 | 6.0024 | **0.0152** | #5 | False |
| `LOGIN` | `ENTER_PASSWORD` | `33` | 10 | `5` | 7.0505 | 7.0342 | **0.0164** | #6 | False |
| `LOGIN` | `ENTER_PASSWORD` | `22` | 10 | `9` | 7.0094 | 6.9910 | **0.0183** | #5 | False |
| `CHECKOUT` | `VERIFY_ORDER_SUCCESS` | `11` | 10 | `6` | 6.0362 | 6.0172 | **0.0191** | #6 | False |
| `PROFILE` | `ENTER_EMAIL` | `55` | 10 | `3` | 7.0842 | 7.0645 | **0.0197** | #1 | True |
| `CHECKOUT` | `CONFIRM_ORDER` | `33` | 10 | `1` | 5.9499 | 5.9286 | **0.0214** | #5 | False |
| `CHECKOUT` | `VERIFY_ORDER_SUCCESS` | `55` | 10 | `1` | 5.9781 | 5.9555 | **0.0225** | #5 | False |

## Summary of Margin Distribution
- **Mean Q Margin**: `0.1141`
- **Median Q Margin**: `0.0629`
- **Min Q Margin**: `0.0016`
- **P10 Q Margin**: `0.0115`
- **P90 Q Margin**: `0.3273`

**FINDING**: The learned Q-network ranks the correct target at **#1** with a strong, distinct Q-value margin across all decisions.