import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from rl.env.types import (
    UICandidate,
    PrivateEvaluatorMetadata,
    TRAIN_MUTATION_LEVELS,
    HELD_OUT_MUTATION_LEVEL,
    VALID_MODES
)
from rl.env.workflow import WorkflowDefinition, WORKFLOW_REGISTRY, WORKFLOW_LOGIN
from rl.env.browser_adapter import BrowserAdapter, MockBrowserAdapter
from rl.state.encoder import StateEncoder
from rl.action.types import ActionExecutionResult
from rl.reward import RewardConfig, RewardCalculator, REWARD_VERSION


class UIRecoveryEnv(gym.Env):
    """
    Gymnasium Environment representing UI Test Recovery as a Sequential Decision-Making Problem.

    PHASE 7 CONTRACT:
      - Action Space: spaces.Discrete(20) where action index i selects candidate row i.
      - Operation (fill, click, verify) is determined by active workflow step, NOT by RL agent.
      - candidate_mask encodes candidate existence ONLY (1=real candidate, 0=padding).
      - Wrong real candidates remain valid (mask=1). Invalid padded or out-of-space actions
        do NOT call Playwright, do NOT advance workflow, and do NOT crash.
      - Page-state verification steps (url_equals) consume 0 RL candidate decisions.
      - Leak-free: no data-semantic-role, expected_role, or mutation metadata in state/mask/info.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        workflow_id: str = "LOGIN",
        mutation_level: int = 0,
        mutation_seed: int = 42,
        mode: str = "train",
        max_candidates: int = 20,
        max_episode_steps: int = 20,
        browser_adapter: Optional[BrowserAdapter] = None,
        reward_config: Optional[RewardConfig] = None
    ):
        super().__init__()

        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {VALID_MODES}")

        if mutation_level < 0 or mutation_level > 6:
            raise ValueError(f"Invalid mutation_level '{mutation_level}'. Must be 0–6.")

        if workflow_id not in WORKFLOW_REGISTRY:
            raise ValueError(f"Invalid workflow_id '{workflow_id}'. Available: {list(WORKFLOW_REGISTRY.keys())}")

        self.mode = mode
        self.mutation_level = mutation_level
        self.mutation_seed = mutation_seed
        self.max_candidates = max_candidates
        self.max_episode_steps = max_episode_steps

        # Held-out protection
        self._enforce_held_out_protection(self.mode, self.mutation_level)

        self.workflow_id = workflow_id
        self.workflow: WorkflowDefinition = WORKFLOW_REGISTRY[self.workflow_id]

        # Action Space: Discrete candidate index choice
        self.action_space = spaces.Discrete(self.max_candidates)

        # PHASE 8 REWARD CALCULATOR & CONTRACT BOUNDS
        self.reward_config = reward_config or RewardConfig()
        self.reward_calculator = RewardCalculator(self.reward_config)
        self.reward_range = (-3.05, 5.95)

        # PHASE 6 STATE ENCODER
        self.state_encoder = StateEncoder(max_candidates=self.max_candidates)
        self.observation_space = self.state_encoder.get_observation_space()

        # Browser Adapter (Mock for Phase 4 / Real for Phase 5+)
        self.browser_adapter = browser_adapter or MockBrowserAdapter(max_candidates=self.max_candidates)

        # Episode State Variables
        self.current_step_index: int = 0
        self.episode_steps: int = 0
        self.wrong_actions: int = 0
        self.previous_action: int = -1
        self.previous_success: float = 0.0
        self.has_previous_action: bool = False

        # Phase 7 Diagnostic Metrics
        self.total_decisions: int = 0
        self.valid_candidate_actions: int = 0
        self.padded_actions: int = 0
        self.out_of_space_actions: int = 0
        self.execution_failures: int = 0
        self.successful_step_actions: int = 0
        self.wrong_step_actions: int = 0

        self.agent_candidates: List[UICandidate] = []
        self._private_meta: List[PrivateEvaluatorMetadata] = []

    def _enforce_held_out_protection(self, mode: str, level: int) -> None:
        """Enforces that Level 6 (held-out) CANNOT be used during training."""
        if mode == "train" and level == HELD_OUT_MUTATION_LEVEL:
            raise ValueError(
                f"EXPERIMENTAL INTEGRITY VIOLATION: Level {HELD_OUT_MUTATION_LEVEL} "
                f"is HELD OUT and CANNOT be used when mode == 'train'."
            )

    def get_action_mask(self) -> np.ndarray:
        """
        Returns the action validity mask for current active candidates.
        Shape: (20,)
        Values: 1 for valid real candidate index, 0 for padded/nonexistent index.

        Identical to observation["candidate_mask"].
        Does NOT encode candidate correctness or task compatibility.
        """
        mask = np.zeros(self.max_candidates, dtype=np.int8)
        num_valid = min(len(self.agent_candidates), self.max_candidates)
        mask[:num_valid] = 1
        return mask

    def valid_action_mask(self) -> np.ndarray:
        """Alias for get_action_mask() for standard RL interface compliance."""
        return self.get_action_mask()

    def _get_workflow_idx(self) -> float:
        workflow_keys = list(WORKFLOW_REGISTRY.keys())
        return float(workflow_keys.index(self.workflow_id)) if self.workflow_id in workflow_keys else 0.0

    def _construct_observation(self) -> Dict[str, np.ndarray]:
        """
        Constructs the Phase 6 RL observation dictionary via StateEncoder.
        """
        if self.current_step_index < len(self.workflow.steps):
            step = self.workflow.steps[self.current_step_index]
        else:
            step = self.workflow.steps[-1]

        return self.state_encoder.encode(
            workflow_id=self.workflow_id,
            step=step,
            current_step_index=self.current_step_index,
            total_steps=len(self.workflow.steps),
            candidates=self.agent_candidates,
            previous_action=self.previous_action,
            previous_success=self.previous_success,
            has_previous_action=self.has_previous_action,
            previous_candidate=self.previous_candidate,
        )

    def _is_page_state_step(self, step: Any) -> bool:
        """Page-state steps (url_equals) are page-state assertions, requiring zero RL candidate decisions."""
        return getattr(step, "success_condition", None) == "url_equals"

    def _process_page_state_verifications(self) -> bool:
        """
        Automatically evaluates any consecutive page-state verification steps starting
        at self.current_step_index.

        Returns True if all encountered page-state steps passed, False if any failed.
        """
        while self.current_step_index < len(self.workflow.steps):
            step = self.workflow.steps[self.current_step_index]
            if self._is_page_state_step(step):
                passed = self.browser_adapter.validate_page_state(
                    success_condition=step.success_condition or "url_equals",
                    expected_value=step.value or ""
                )
                if passed:
                    self.current_step_index += 1
                else:
                    return False
            else:
                break
        return True

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment for a new episode following Gymnasium API contract.
        Strictly validates reset options.
        """
        super().reset(seed=seed)

        if options:
            if "workflow_id" in options:
                w_id = options["workflow_id"]
                if w_id not in WORKFLOW_REGISTRY:
                    raise ValueError(f"Invalid workflow_id '{w_id}'. Available: {list(WORKFLOW_REGISTRY.keys())}")
                self.workflow_id = w_id
                self.workflow = WORKFLOW_REGISTRY[self.workflow_id]

            if "mode" in options:
                m = options["mode"]
                if m not in VALID_MODES:
                    raise ValueError(f"Invalid mode '{m}'. Must be one of {VALID_MODES}")
                self.mode = m

            if "mutation_level" in options:
                lvl = int(options["mutation_level"])
                if lvl < 0 or lvl > 6:
                    raise ValueError(f"Invalid mutation_level '{lvl}'. Must be 0–6.")
                self.mutation_level = lvl

            if "mutation_seed" in options:
                self.mutation_seed = int(options["mutation_seed"])

        # Re-enforce protection if options updated level/mode
        self._enforce_held_out_protection(self.mode, self.mutation_level)

        # Seed adapter RNG deterministically from Gymnasium np_random.
        if hasattr(self.browser_adapter, "set_rng"):
            self.browser_adapter.set_rng(self.np_random)

        self.current_step_index = 0
        self.episode_steps = 0
        self.wrong_actions = 0
        self.previous_action = -1
        self.previous_candidate = None
        self.previous_success = 0.0
        self.has_previous_action = False

        # Reset diagnostic metrics
        self.total_decisions = 0
        self.valid_candidate_actions = 0
        self.padded_actions = 0
        self.out_of_space_actions = 0
        self.execution_failures = 0
        self.successful_step_actions = 0
        self.wrong_step_actions = 0

        # Reset browser
        self.browser_adapter.reset(
            start_path=self.workflow.start_path,
            level=self.mutation_level,
            seed=self.mutation_seed
        )

        # Automatically advance past any initial page-state verification steps
        self._process_page_state_verifications()

        # Fetch initial candidates for the first active step (if not completed)
        if self.current_step_index < len(self.workflow.steps):
            current_step = self.workflow.steps[self.current_step_index]
            self.agent_candidates, self._private_meta = self.browser_adapter.get_candidates(
                step_id=current_step.step_id,
                expected_role=current_step.expected_role,
                action_type=current_step.action_type,
            )
            initial_step_id = current_step.step_id
        else:
            self.agent_candidates = []
            self._private_meta = []
            initial_step_id = None

        obs = self._construct_observation()

        # AGENT-SAFE INFO DICTIONARY — NO GROUND TRUTH LEAKAGE
        info = {
            "workflow_id": self.workflow_id,
            "step_index": self.current_step_index,
            "step_id": initial_step_id,
            "episode_steps": self.episode_steps,
            "total_decisions": self.total_decisions,
            "valid_candidate_actions": self.valid_candidate_actions,
            "padded_actions": self.padded_actions,
            "out_of_space_actions": self.out_of_space_actions,
            "execution_failures": self.execution_failures,
            "successful_step_actions": self.successful_step_actions,
            "wrong_step_actions": self.wrong_step_actions,
            "wrong_actions": self.wrong_actions,
            "candidate_count": len(self.agent_candidates),
            "mutation_level": self.mutation_level,
            "mutation_seed": self.mutation_seed,
            "mode": self.mode,
            "workflow_completed": (self.current_step_index >= len(self.workflow.steps))
        }

        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Executes an action in the environment following Gymnasium API contract.
        """
        t_step_start = time.time()
        self.episode_steps += 1
        self.total_decisions += 1
        self.has_previous_action = True
        action_idx = int(action)

        current_step = (
            self.workflow.steps[self.current_step_index]
            if self.current_step_index < len(self.workflow.steps)
            else self.workflow.steps[-1]
        )
        op_type = getattr(current_step, "action_type", "none")

        # Classify Action Space Range vs Valid Candidate Count
        is_out_of_space = (action_idx < 0 or action_idx >= self.max_candidates)
        is_padded = (0 <= action_idx < self.max_candidates and action_idx >= len(self.agent_candidates))
        is_valid_action = (0 <= action_idx < len(self.agent_candidates) and not is_out_of_space)

        attempted = False
        executed = False
        execution_success = False
        step_success = False
        invalid_reason: Optional[str] = None
        candidate_id: Optional[Any] = None
        reward = 0.0

        if is_out_of_space:
            self.out_of_space_actions += 1
            self.wrong_actions += 1
            self.previous_action = action_idx
            self.previous_candidate = None
            self.previous_success = 0.0
            invalid_reason = "out_of_space"
        elif is_padded:
            self.padded_actions += 1
            self.wrong_actions += 1
            self.previous_action = action_idx
            self.previous_candidate = None
            self.previous_success = 0.0
            invalid_reason = "padded_candidate"
        else:
            # Valid real candidate selection
            self.valid_candidate_actions += 1
            attempted = True
            candidate = self.agent_candidates[action_idx]
            candidate_id = candidate.candidate_id
            self.previous_candidate = candidate

            try:
                exec_ok = self.browser_adapter.execute_action(
                    candidate_id=candidate.candidate_id,
                    action_type=current_step.action_type,
                    value=current_step.value
                )
                executed = True
                execution_success = True if (exec_ok is None or exec_ok is True) else False
            except Exception as e:
                executed = False
                execution_success = False
                invalid_reason = f"execution_error: {str(e)}"

            self.previous_action = action_idx

            if execution_success:
                # Private evaluator check
                step_success = self.browser_adapter.validate_step(current_step.expected_role)

                if step_success:
                    saved_step_index = self.current_step_index
                    self.current_step_index += 1
                    page_state_ok = self._process_page_state_verifications()
                    if not page_state_ok:
                        step_success = False
                        self.wrong_actions += 1
                        self.wrong_step_actions += 1
                        self.previous_success = 0.0
                        invalid_reason = "verification_failed"

                        # Reset browser to initial workflow start path and restart workflow at step 0
                        self.browser_adapter.reset(
                            start_path=self.workflow.start_path,
                            level=self.mutation_level,
                            seed=self.mutation_seed
                        )
                        self.current_step_index = 0
                    else:
                        self.successful_step_actions += 1
                        self.previous_success = 1.0
                        invalid_reason = None
                else:
                    self.wrong_step_actions += 1
                    self.wrong_actions += 1
                    self.previous_success = 0.0
                    invalid_reason = "incorrect_candidate"
            else:
                self.execution_failures += 1
                self.wrong_actions += 1
                self.previous_success = 0.0
                if invalid_reason is None:
                    invalid_reason = "execution_failed"

        # Check termination (workflow completed)
        terminated = False
        if self.current_step_index >= len(self.workflow.steps):
            terminated = True

        # Check truncation (max decision budget reached)
        truncated = False
        if not terminated and self.episode_steps >= self.max_episode_steps:
            truncated = True

        # PHASE 8 CENTRALIZED REWARD CALCULATION
        reward_breakdown = self.reward_calculator.calculate(
            action_valid=is_valid_action,
            step_success=step_success,
            workflow_completed=terminated,
            truncated=truncated
        )
        reward = reward_breakdown.total

        # Fetch candidates for next step if state refreshed & active
        if not terminated and not truncated:
            next_step = self.workflow.steps[self.current_step_index]
            self.agent_candidates, self._private_meta = self.browser_adapter.get_candidates(
                step_id=next_step.step_id,
                expected_role=next_step.expected_role,
                action_type=next_step.action_type,
            )

        obs = self._construct_observation()

        # Terminal transition handling: if completed, step_id is None
        next_step_id = None
        if not terminated and self.current_step_index < len(self.workflow.steps):
            next_step_id = self.workflow.steps[self.current_step_index].step_id

        action_result = ActionExecutionResult(
            attempted=attempted,
            executed=executed,
            candidate_index=action_idx if (0 <= action_idx < self.max_candidates) else None,
            candidate_id=candidate_id,
            operation_type=op_type,
            execution_success=execution_success,
            step_success=step_success,
            failure_reason=invalid_reason
        )

        t_step_total = time.time() - t_step_start
        timing_diag = getattr(self.browser_adapter, "last_timing_diagnostics", {})
        self.last_timing_diagnostics = {
            "execute_action_sec": float(timing_diag.get("execute_action_sec", 0.0)),
            "validate_step_sec": float(timing_diag.get("validate_step_sec", 0.0)),
            "candidate_refresh_sec": float(timing_diag.get("candidate_refresh_sec", 0.0)),
            "total_step_sec": float(t_step_total),
        }

        # AGENT-SAFE INFO DICTIONARY — NO GROUND TRUTH LEAKAGE
        info = {
            "workflow_id": self.workflow_id,
            "step_index": self.current_step_index,
            "step_id": next_step_id,
            "episode_steps": self.episode_steps,
            "total_decisions": self.total_decisions,
            "valid_candidate_actions": self.valid_candidate_actions,
            "padded_actions": self.padded_actions,
            "out_of_space_actions": self.out_of_space_actions,
            "execution_failures": self.execution_failures,
            "successful_step_actions": self.successful_step_actions,
            "wrong_step_actions": self.wrong_step_actions,
            "wrong_actions": self.wrong_actions,
            "action_index": action_idx,
            "action_valid": is_valid_action,
            "invalid_reason": invalid_reason,
            "candidate_count": len(self.agent_candidates),
            "candidate_id": candidate_id,
            "operation_type": op_type,
            "execution_success": execution_success,
            "step_success": step_success,
            "action_success": step_success,  # backward compatibility alias
            "workflow_completed": terminated,
            "workflow_success": terminated,
            "mutation_level": self.mutation_level,
            "mutation_seed": self.mutation_seed,
            "mode": self.mode,
            "action_result": action_result,
            "reward_total": float(reward),
            "reward_breakdown": reward_breakdown.to_dict(),
            "reward_version": REWARD_VERSION,
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        pass

    def close(self) -> None:
        if self.browser_adapter:
            self.browser_adapter.close()
