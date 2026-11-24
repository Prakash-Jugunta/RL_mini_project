# Fielddesk Search recovery

The opt-in Playwright Search suite runs the original locator first. On a missing or ambiguous match, it captures visible search submit controls from the same page. A local Python subprocess ranks them with the existing DQN checkpoint. The LangGraph supervisor routes a duplicate name or a weak top match to GPT-6.2 Sol. Playwright rechecks the proposed element and runs the original business assertion. Recovery decisions and masked Q-values are attached to the test report.

## Run locally

From the repository root, use the Python environment that runs the existing DQN demo and install the recovery dependencies.

```powershell
python -m pip install -r rl/recovery/requirements.txt
npm run test:e2e:storefront
```

The suite uses the real DQN checkpoint and a local mock Responses API for the ambiguous cases. Set `RECOVERY_PYTHON` to choose a Python executable or `RECOVERY_CHECKPOINT` to load another trusted checkpoint with the same policy and state encoding. The default checkpoint is `artifacts/dqn/phase13-v1/latest_checkpoint.pt`.

## Automatic presentation demo

Run `npm run demo:recovery` from the repository root. The command starts a dedicated dev server, shows the browser steps with recovery highlights, and opens the report when the run ends. It covers clean replay, the trained DQN, model-assisted recovery, an unresolved match, and a genuine SKU regression. No keypresses are needed during the run.

Each run saves a workflow chart, ranked candidates and Q-values, model routing decisions, screenshots, and original SKU assertion results under `artifacts/recovery-demo/<run>/index.html`. The report links to the Playwright replay traces and raw results. Use `npm run demo:recovery -- --headless` for an unattended run that saves the same evidence without opening windows. Use `--no-open` to keep the browser demonstration visible and skip opening the report.

The standard demo labels model-assisted scenarios as local mocks. To add a separate live model scenario, configure the provider and enable `RECOVERY_LIVE_MODEL=1` before running the demo.

## Live model

To exercise the live Responses API, set `OPENAI_API_KEY` in the local environment and opt into the live test.

```powershell
$env:RECOVERY_LIVE_MODEL = '1'
npm run test:e2e:storefront -- --grep 'live model'
```

`RECOVERY_OPENAI_MODEL` overrides the default model. Only the public test intent, original control name and scope, and shortlisted candidate summaries are sent to the model. When the model is unavailable or abstains, an ambiguous step stays unresolved.

This integration covers the Search submit step. The DQN margin of `0.25` is an initial setting for later validation across releases. Checkout and offline retraining require separate fixtures and evaluation.
