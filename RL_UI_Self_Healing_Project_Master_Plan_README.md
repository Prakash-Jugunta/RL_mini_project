# RL-Based Self-Healing UI Test Automation

## 1. Project Goal

Build a complete reinforcement-learning-based UI test recovery system.

Normal Playwright tests use fixed selectors. When a UI mutation breaks a
selector, the recovery system observes the current DOM, constructs a
numerical state, asks an RL policy to select the most appropriate
candidate element, executes that action through Playwright, validates
whether the workflow progressed, and continues the test.

The project compares:

-   Fixed-selector Playwright baseline
-   Random-agent baseline
-   Heuristic self-healing baseline
-   DQN
-   PPO
-   A2C

The final system must evaluate both element recovery and complete
workflow success on mutations that were not seen during training.

------------------------------------------------------------------------

## 2. Core Research Question

**Can an RL agent learn a recovery policy that allows automated UI test
workflows to continue after DOM/UI mutations break their original
selectors?**

The RL problem is deliberately sequential. The agent is not merely asked
to find the element most similar to an old selector. It must choose
actions that maximize successful completion of the entire test workflow.

------------------------------------------------------------------------

## 3. System Architecture

``` text
Test Specification
       |
       v
Normal Playwright Test
       |
       v
Original selector works?
   |              |
  YES             NO
   |              |
   v              v
Continue       RL Recovery
                  |
                  v
          Extract current DOM
                  |
                  v
          Generate candidates
                  |
                  v
            Encode state
                  |
                  v
       DQN / PPO / A2C policy
                  |
                  v
          Select candidate
                  |
                  v
        Playwright executes
                  |
                  v
         Validate transition
                  |
                  v
        Reward + next state
                  |
             +----+----+
             |         |
          Continue   Retry/Fail
```

------------------------------------------------------------------------

# PHASE 1 --- Demo Web Application

## Objective

Create a small deterministic website with multiple sequential workflows.

## Recommended Stack

-   React + Vite
-   Playwright
-   Python
-   Gymnasium
-   Stable-Baselines3 where appropriate
-   PyTorch
-   A lightweight pretrained text embedding model for semantic features

## Required Workflows

### Workflow A --- Login

``` text
Username
   ->
Password
   ->
Login
   ->
Dashboard
```

### Workflow B --- Search

``` text
Search input
   ->
Search
   ->
Select result
   ->
Product page
```

### Workflow C --- Profile Update

``` text
Name
   ->
Email
   ->
Save
   ->
Profile updated
```

### Workflow D --- Checkout

``` text
Product
   ->
Add to cart
   ->
Cart
   ->
Checkout
   ->
Confirm
   ->
Order success
```

## Completion Criteria

-   All pages work manually.
-   Every workflow has an unambiguous success state.
-   Important elements have stable original IDs, labels, ARIA
    attributes, and test metadata.
-   The original application contains no mutations.

------------------------------------------------------------------------

# PHASE 2 --- Normal Playwright Tests

## Objective

Create conventional automated tests before adding RL.

Example login test:

``` python
page.locator("#username").fill("testuser")
page.locator("#password").fill("password")
page.locator("#login-btn").click()
expect(page).to_have_url("/dashboard")
```

## Required Tests

-   `test_login`
-   `test_search`
-   `test_profile`
-   `test_checkout`

## Baseline Requirement

On the original application:

``` text
Login       PASS
Search      PASS
Profile     PASS
Checkout    PASS
```

Save these results as the fixed-selector baseline.

------------------------------------------------------------------------

# PHASE 3 --- Mutation Engine

## Objective

Programmatically generate realistic UI changes without manually building
hundreds of pages.

## Mutation Types

### M1 --- ID Mutation

``` text
username -> user-x381
login-btn -> submit-a72
```

### M2 --- Class Mutation

``` text
.login-button -> .primary-action-x91
```

### M3 --- Text / Label Mutation

``` text
Username -> Account
Password -> Passcode
Login -> Continue
Add to Cart -> Add to Bag
```

### M4 --- Position / Order Mutation

Reorder elements or move them to another region of the page.

### M5 --- DOM Structure Mutation

Example:

``` html
<form>
    <input id="username">
</form>
```

becomes:

``` html
<form>
    <div>
        <section>
            <input id="account-x12">
        </section>
    </div>
</form>
```

### M6 --- Distractor Mutation

Add plausible but incorrect elements.

Example:

``` text
Newsletter Email
Account
Passcode
Continue
Forgot Account
```

### M7 --- Element-Type Mutation

Examples:

``` text
<input type="submit"> -> <button>
<select> -> custom dropdown
<a role="button"> -> <button>
```

### M8 --- Combined Mutation

Apply several mutations simultaneously.

## Mutation Configuration

The mutation engine should support deterministic seeds.

Example:

``` python
mutation = {
    "id_change": True,
    "class_change": False,
    "text_change": True,
    "position_change": False,
    "dom_change": True,
    "add_distractor": True,
    "element_type_change": False,
}
```

## Critical Requirement

The engine must preserve the intended business functionality. A mutation
should break automation assumptions, not intentionally break the
application itself.

------------------------------------------------------------------------

# PHASE 4 --- RL Environment

## Objective

Implement the UI recovery problem as a Gymnasium-compatible environment.

Each interaction follows:

``` text
State -> Action -> Environment -> Reward -> Next State
```

## Episode Definition

One episode represents an attempt to complete one complete UI workflow
under a generated mutation configuration.

Example:

``` text
Need username
   ->
select candidate
   ->
Need password
   ->
select candidate
   ->
Need login
   ->
select candidate
   ->
Dashboard
```

## Terminal Conditions

Success:

-   Required workflow success state is reached.

Failure:

-   Maximum step count reached.
-   Unrecoverable invalid state.
-   Critical application failure.

------------------------------------------------------------------------

# PHASE 5 --- Candidate Extraction

## Objective

Playwright observes the browser and produces candidate UI elements.

For each candidate collect useful information such as:

``` text
tag
type
id
class
text
placeholder
aria-label
role
x position
y position
width
height
visible
enabled
parent tag
nearby text
```

Example:

``` text
Candidate 0
tag         = input
type        = email
placeholder = Newsletter Email

Candidate 1
tag         = input
type        = text
placeholder = Account

Candidate 2
tag         = input
type        = password
placeholder = Passcode

Candidate 3
tag         = button
text        = Continue
```

Do not allow hidden answer labels or evaluator-only identities to enter
the observation.

------------------------------------------------------------------------

# PHASE 6 --- State Representation

## Objective

Convert the current browser situation into numerical features usable by
the RL policy.

A state should represent:

``` text
Current objective
+
Candidate elements
+
Workflow progress
+
Previous action/result
```

Conceptually:

``` text
STATE

Objective:
USERNAME_INPUT

Workflow progress:
0 / 3

Candidate 0:
[semantic features + structural features]

Candidate 1:
[semantic features + structural features]

...

Previous action:
NONE
```

## Structural Features

Possible features:

-   Tag encoding
-   Input type encoding
-   Visibility
-   Enabled state
-   Normalized x/y position
-   Normalized width/height
-   Parent element information
-   DOM depth
-   Role

## Semantic Features

Textual information can include:

``` text
id
text
placeholder
aria-label
nearby label text
```

Use meaningful text representations rather than arbitrary integer IDs.

Example:

``` text
"Account Name"
      |
      v
Text Encoder
      |
      v
[0.13, -0.52, 0.82, ...]
```

The semantic encoder provides perception features. The RL algorithm
still makes the sequential decision.

## Important Rule

The observation used during training and the observation extracted from
the real browser must have the same structure and meaning.

Avoid a simulation-to-browser representation gap.

------------------------------------------------------------------------

# PHASE 7 --- Action Space

## Objective

Allow the agent to select a current candidate.

Basic version:

``` text
Action 0 -> candidate 0
Action 1 -> candidate 1
Action 2 -> candidate 2
...
Action N -> candidate N
```

Use a fixed maximum candidate count with masking/padding if required by
the selected RL implementation.

Possible future actions:

``` text
WAIT
SCROLL
BACK
```

These are optional for the first complete version.

## Critical Anti-Shortcut Rule

Randomize candidate ordering.

Otherwise the model may learn:

``` text
username = candidate 1
password = candidate 2
login = candidate 3
```

instead of learning from candidate properties.

------------------------------------------------------------------------

# PHASE 8 --- Reward Function

## Objective

Reward workflow progress rather than simple visual similarity.

Initial reward proposal:

``` text
Correct candidate             +2
Workflow advances             +3
Wrong candidate               -2
Invalid candidate             -3
Repeated useless action       -1
Complete workflow             +10
Episode failure               -10
```

These values are starting points and may be tuned.

## Important Principle

The terminal workflow reward matters because RL should optimize
successful completion of the full test, not merely individual element
matching.

------------------------------------------------------------------------

# PHASE 9 --- Baselines

## Fixed Selector Baseline

Normal Playwright selectors.

Expected behavior:

-   Strong on original UI.
-   Fragile when relevant selectors mutate.

## Random Agent

Randomly selects a valid candidate.

Purpose:

-   Establish a minimum learning baseline.

## Heuristic Recovery Baseline

Build a deterministic candidate-ranking system.

Possible scoring inputs:

``` text
Semantic similarity
Tag match
Type match
ARIA/role compatibility
Position similarity
DOM/context similarity
```

Conceptually:

``` text
Score =
w1 * semantic_similarity
+ w2 * tag_match
+ w3 * type_match
+ w4 * context_similarity
+ w5 * position_similarity
```

The heuristic is important because RL should be compared against a
simpler recovery method.

------------------------------------------------------------------------

# PHASE 10 --- DQN Implementation

## Objective

Get one RL algorithm working before adding others.

DQN learns:

``` text
Q(state, action)
```

At inference:

``` text
Q(candidate 0) = 1.21
Q(candidate 1) = 7.82
Q(candidate 2) = 0.93
Q(candidate 3) = -0.42

Selected = candidate 1
```

During training, use exploration such as epsilon-greedy behavior.

## DQN Completion Criteria

-   Training reward improves.
-   Workflow success exceeds random baseline.
-   Evaluation uses unseen episodes.
-   No evaluator-only information leaks into the observation.
-   Candidate order is randomized.
-   Saved model can be loaded independently for evaluation.

Do not add PPO/A2C until these checks pass.

------------------------------------------------------------------------

# PHASE 11 --- Training Episode Generation

For every training episode:

``` text
1. Select workflow.
2. Reset application/environment.
3. Generate mutation configuration.
4. Apply mutations.
5. Extract candidates.
6. Construct state.
7. Agent selects action.
8. Execute action.
9. Validate workflow progress.
10. Calculate reward.
11. Construct next state.
12. Repeat until success/failure.
```

Training should cover many mutation seeds and combinations.

Generated experiences are synthetic RL interactions, not independent
manually collected real-world examples.

------------------------------------------------------------------------

# PHASE 12 --- Train / Validation / Test Strategy

## Critical Rule

Do not evaluate only on mutations sampled identically from the training
distribution.

### Example Training Mutations

``` text
ID
class
text
position
DOM nesting
distractor

ID + text
text + position
ID + position
```

### Held-Out Test Combinations

Examples:

``` text
ID + DOM + distractor
text + type + position
ID + text + DOM
position + DOM + distractor
ID + text + DOM + distractor
```

The exact held-out combinations must be fixed before final evaluation.

## Goal

Test whether the policy learned useful recovery behavior rather than
memorizing generated patterns.

------------------------------------------------------------------------

# PHASE 13 --- Difficulty Levels

Create explicit evaluation levels.

``` text
Level 0 — Original UI
Level 1 — Single attribute mutation
Level 2 — Semantic/text mutation
Level 3 — Position/DOM mutation
Level 4 — Distractor elements
Level 5 — Multiple known mutation types
Level 6 — Held-out mutation combinations
```

Each level should contain multiple seeds/episodes.

------------------------------------------------------------------------

# PHASE 14 --- PPO and A2C

Only after the environment is frozen and DQN works.

Train:

-   DQN
-   PPO
-   A2C

All algorithms must use:

-   Same state representation
-   Same action semantics
-   Same mutation distribution
-   Same reward function
-   Same train/test split
-   Same evaluation seeds where possible

This makes the comparison meaningful.

------------------------------------------------------------------------

# PHASE 15 --- Browser Recovery Integration

## Objective

Use RL as a fallback for normal test automation.

Pseudo-flow:

``` python
try:
    execute_original_selector()
except ElementNotFound:
    state = build_state(
        objective=current_test_objective,
        dom=current_dom,
        workflow_progress=progress
    )

    action = rl_agent.predict(state)

    candidate = candidates[action]

    execute_candidate(candidate)

    validate_progress()
```

## Important

The RL model does not directly control the browser.

Responsibilities:

``` text
Playwright
    -> observes DOM
    -> extracts candidates
    -> executes selected action

State Encoder
    -> converts browser information to numerical state

RL Agent
    -> selects an action

Environment
    -> evaluates transition and reward
```

------------------------------------------------------------------------

# PHASE 16 --- Live Demo Interface

Build a visual interface showing both the website and agent decisions.

Example:

``` text
+------------------------------------------------+
| UI SELF-HEALING DEMO                           |
+----------------------+-------------------------+
| WEBSITE              | AGENT                   |
|                      |                         |
| Account [________]   | Objective               |
| Passcode[________]   | USERNAME_INPUT          |
|                      |                         |
| [Continue]           | Candidate values        |
|                      |                         |
|                      | Newsletter     1.21      |
|                      | Account        7.82 <-   |
|                      | Passcode       0.93      |
|                      | Continue      -0.42      |
+----------------------+-------------------------+
| Original selector: #username       FAILED      |
| RL recovery: Account input         SELECTED    |
| Workflow progress: 1/3                         |
+------------------------------------------------+
```

## Controls

Include:

``` text
Original UI / Mutated UI

Mutation toggles:
[ ] ID
[ ] Class
[ ] Text
[ ] Position
[ ] DOM
[ ] Distractor
[ ] Type

Algorithm:
DQN / PPO / A2C / Heuristic

Run Normal Test
Run Recovery
Reset
```

Optional:

-   Step mode
-   Auto mode
-   Highlight selected candidate
-   Show reward
-   Show current state/objective
-   Show episode return

------------------------------------------------------------------------

# PHASE 17 --- Evaluation Metrics

## 1. Element Recovery Accuracy

``` text
correct recovered elements
-------------------------- x 100
total recovery attempts
```

## 2. Workflow Success Rate

``` text
successful complete workflows
----------------------------- x 100
total evaluation workflows
```

This is a primary metric because the task is sequential.

## 3. Average Episode Return

Average cumulative reward.

## 4. Average Recovery Steps

Number of RL decisions required to complete/recover a workflow.

## 5. Wrong Actions

Average incorrect actions per episode.

## 6. Performance by Mutation Level

Evaluate separately for Levels 0--6.

## 7. Held-Out Generalization

Report performance specifically on mutation combinations excluded from
training.

## 8. Recovery Latency

Optional but useful:

``` text
time from selector failure
to successful recovery
```

------------------------------------------------------------------------

# PHASE 18 --- Final Comparison

Produce a result table using actual measured results only.

  --------------------------------------------------------------------------------
  Mutation           Fixed   Heuristic     Random        DQN        PPO        A2C
  Level           Selector                                              
  ------------- ---------- ----------- ---------- ---------- ---------- ----------
  L0 Original          TBD         TBD        TBD        TBD        TBD        TBD

  L1 Attribute         TBD         TBD        TBD        TBD        TBD        TBD

  L2 Semantic          TBD         TBD        TBD        TBD        TBD        TBD

  L3 Structural        TBD         TBD        TBD        TBD        TBD        TBD

  L4                   TBD         TBD        TBD        TBD        TBD        TBD
  Distractors                                                           

  L5 Combined          TBD         TBD        TBD        TBD        TBD        TBD

  L6 Held-out          TBD         TBD        TBD        TBD        TBD        TBD
  --------------------------------------------------------------------------------

Never replace weak measurements with assumed values.

------------------------------------------------------------------------

# PHASE 19 --- Required Graphs

Generate at least:

1.  Training reward vs episodes/timesteps
2.  Workflow success rate by algorithm
3.  Success rate by mutation level
4.  Element recovery accuracy
5.  Average recovery steps
6.  Held-out generalization comparison

Optional:

7.  Recovery latency
8.  Wrong-action count
9.  Training stability across multiple random seeds

------------------------------------------------------------------------

# PHASE 20 --- Ablation Experiments

If time permits, test what happens when important state information is
removed.

Examples:

``` text
Full state
vs
No semantic embedding
vs
No position
vs
No DOM context
```

This helps answer:

**What information is actually helping the RL agent recover elements?**

A particularly useful comparison is:

``` text
Structural features only
vs
Semantic + structural features
```

------------------------------------------------------------------------

# PHASE 21 --- Validity Checks

Before trusting results, verify all of the following.

## Observation Integrity

-   No hidden target identity.
-   No correct-candidate index.
-   No evaluator-only field.
-   No mutation generator answer leaked into state.

## Candidate Integrity

-   Candidate ordering randomized.
-   Distractors genuinely plausible.
-   Correct candidate not always at the same index.

## Evaluation Integrity

-   Test seeds excluded from training.
-   Held-out mutation combinations excluded from training.
-   Model evaluated without learning/updating.
-   Same environment used for all algorithms.
-   Same evaluation cases used for algorithm comparison.

## Browser Integrity

-   Browser state representation matches training representation.
-   Recovery operates on real extracted DOM candidates.
-   Success is validated from actual page state.

------------------------------------------------------------------------

# PHASE 22 --- Failure Analysis

Do not report only successful cases.

For every failure, log:

``` text
Workflow
Mutation seed
Mutation types
Current objective
Candidate list
Selected candidate
Correct candidate
Agent values/probabilities
Reward
Terminal reason
```

Group failures into categories such as:

``` text
Semantic ambiguity
Too many similar distractors
Unseen structural mutation
Candidate extraction failure
State encoding problem
Action-space limitation
Workflow transition failure
```

This makes the final analysis much stronger.

------------------------------------------------------------------------

# PHASE 23 --- Final Demo Sequence

Use this exact narrative for the live demonstration.

## Step 1 --- Original Application

Show the normal login page.

Run normal Playwright test.

``` text
PASS
```

## Step 2 --- Apply Mutations

Example:

``` text
username -> account-x719
Username -> Account Name

password -> secure-x812
Password -> Passcode

login-btn -> continue-x13
Login -> Continue

+ add Newsletter Email distractor
+ modify DOM nesting
```

## Step 3 --- Run Original Test

Show:

``` text
#username NOT FOUND

TEST FAILED
```

## Step 4 --- Activate RL Recovery

Display:

``` text
Objective:
USERNAME_INPUT

Candidates:

Newsletter Email      Q = ...
Account Name          Q = ... <- selected
Passcode              Q = ...
Continue              Q = ...
```

Highlight the selected element.

## Step 5 --- Continue Workflow

Show recovery of:

``` text
Username
   ->
Password
   ->
Login
   ->
Dashboard
```

## Step 6 --- Show Evaluation

Show results for:

``` text
Fixed selector
Heuristic
Random
DQN
PPO
A2C
```

including held-out mutation performance.

------------------------------------------------------------------------

# PHASE 24 --- Final Project Claims

## Safe Claim

> The project investigates whether reinforcement learning can learn a
> sequential UI-test recovery policy that selects candidate elements and
> completes automated workflows under DOM and interface mutations.

## Stronger Claim Only If Supported by Results

> The learned policy generalized to held-out combinations of UI
> mutations within the controlled application environment.

## Do Not Claim

``` text
The agent understands every website.

The agent can recover any DOM mutation.

The system works universally on unseen websites.

RL understands HTML automatically.

The RL model directly reads the screen.

The model learned semantic meaning entirely through RL.
```

Unless separately demonstrated, these claims are not justified.

------------------------------------------------------------------------

# PHASE 25 --- Repository Structure

``` text
rl-ui-self-healing/
|
|-- webapp/
|   |-- src/
|   |-- pages/
|   |-- components/
|   `-- mutation-config/
|
|-- tests/
|   |-- playwright/
|   `-- specifications/
|
|-- mutations/
|   |-- generator.py
|   |-- attribute.py
|   |-- semantic.py
|   |-- structural.py
|   `-- distractors.py
|
|-- environment/
|   |-- ui_env.py
|   |-- candidate_extractor.py
|   |-- state_encoder.py
|   |-- actions.py
|   `-- rewards.py
|
|-- baselines/
|   |-- random_agent.py
|   `-- heuristic_agent.py
|
|-- agents/
|   |-- dqn.py
|   |-- ppo.py
|   `-- a2c.py
|
|-- training/
|   |-- train_dqn.py
|   |-- train_ppo.py
|   `-- train_a2c.py
|
|-- evaluation/
|   |-- evaluate.py
|   |-- mutation_levels.py
|   |-- heldout_mutations.py
|   `-- metrics.py
|
|-- demo/
|   `-- ...
|
|-- results/
|   |-- raw/
|   |-- plots/
|   `-- reports/
|
|-- models/
|
|-- requirements.txt
`-- README.md
```

------------------------------------------------------------------------

# TODAY --- BUILD ORDER

The goal is a complete working project today, so prioritize a working
end-to-end pipeline before optional sophistication.

## Milestone 1 --- Application

-   [ ] Create project repository
-   [ ] Build demo website
-   [ ] Implement Login workflow
-   [ ] Implement Search workflow
-   [ ] Implement Profile workflow
-   [ ] Implement Checkout workflow
-   [ ] Verify all manually

## Milestone 2 --- Conventional Testing

-   [ ] Install/configure Playwright
-   [ ] Write original tests
-   [ ] Verify 100% original workflow pass
-   [ ] Save baseline output

## Milestone 3 --- Mutations

-   [ ] ID mutation
-   [ ] Class mutation
-   [ ] Text mutation
-   [ ] Position mutation
-   [ ] DOM mutation
-   [ ] Distractor mutation
-   [ ] Type mutation
-   [ ] Combined mutations
-   [ ] Deterministic seeds

## Milestone 4 --- RL Environment

-   [ ] Candidate extractor
-   [ ] Objective representation
-   [ ] State encoder
-   [ ] Action space
-   [ ] Reward function
-   [ ] Episode reset
-   [ ] Step transition
-   [ ] Success/failure termination
-   [ ] Environment contract tests

## Milestone 5 --- Baselines

-   [ ] Random agent
-   [ ] Fixed-selector baseline
-   [ ] Heuristic recovery baseline

## Milestone 6 --- DQN

-   [ ] Training script
-   [ ] Model saving/loading
-   [ ] Learning logs
-   [ ] Evaluation script
-   [ ] Verify improvement over random
-   [ ] Check for observation leakage

## Milestone 7 --- Generalization

-   [ ] Define train mutation combinations
-   [ ] Define held-out combinations
-   [ ] Freeze evaluation seeds
-   [ ] Evaluate DQN
-   [ ] Inspect failure cases

## Milestone 8 --- Browser Recovery

-   [ ] Trigger recovery after selector failure
-   [ ] Extract live candidates
-   [ ] Encode browser state
-   [ ] Run policy
-   [ ] Execute selected candidate
-   [ ] Validate workflow transition
-   [ ] Complete at least Login end-to-end

## Milestone 9 --- PPO + A2C

-   [ ] Freeze environment
-   [ ] Train PPO
-   [ ] Train A2C
-   [ ] Evaluate using identical held-out cases

## Milestone 10 --- Results

-   [ ] Element recovery accuracy
-   [ ] Workflow success rate
-   [ ] Average return
-   [ ] Average steps
-   [ ] Wrong actions
-   [ ] Mutation-level results
-   [ ] Held-out results
-   [ ] Generate graphs

## Milestone 11 --- Demo UI

-   [ ] Mutation controls
-   [ ] Algorithm selector
-   [ ] Normal-test button
-   [ ] Recovery button
-   [ ] Current objective
-   [ ] Candidate list
-   [ ] Agent values/probabilities
-   [ ] Selected-element highlight
-   [ ] Reward/progress display
-   [ ] Reset control

## Milestone 12 --- Documentation

-   [ ] Architecture diagram
-   [ ] RL formulation
-   [ ] State definition
-   [ ] Action definition
-   [ ] Reward definition
-   [ ] Mutation definitions
-   [ ] Algorithm descriptions
-   [ ] Experimental protocol
-   [ ] Results
-   [ ] Failure analysis
-   [ ] Limitations
-   [ ] Reproduction instructions

------------------------------------------------------------------------

# Minimum Viable Complete Version

If time becomes limited, the minimum defensible complete system is:

``` text
Login + one additional workflow
        +
5 mutation types
        +
combined mutations
        +
Playwright baseline
        +
random baseline
        +
heuristic baseline
        +
DQN
        +
held-out evaluation
        +
real browser recovery
        +
visual decision display
```

Get this working before spending time on PPO/A2C, extra pages, or
cosmetic improvements.

------------------------------------------------------------------------

# Definition of Done

The project is complete when the following experiment works end-to-end:

``` text
1. Launch original website.

2. Run Playwright test.
   -> PASS

3. Apply a held-out UI mutation.

4. Run original fixed selector.
   -> FAIL

5. Recovery system extracts real DOM candidates.

6. State encoder creates the RL observation.

7. Trained agent selects a candidate.

8. Playwright executes the selected action.

9. Environment validates workflow progress.

10. Agent continues sequential decisions.

11. Entire workflow reaches its success state.

12. Evaluation script reports the result honestly.

13. The same held-out cases can be run against:
    fixed selectors,
    heuristic,
    random,
    DQN,
    PPO,
    A2C.
```

------------------------------------------------------------------------

# Core Principle

``` text
Good mutation environment
        +
Good state representation
        +
Good reward design
        +
Strict held-out evaluation
        ↓
Meaningful RL experiment
```

The objective is not to manufacture high accuracy. The objective is to
determine whether the learned recovery policy actually generalizes to UI
changes that were not directly encountered during training.
