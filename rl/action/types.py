from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ActionExecutionResult:
    """
    Structured container capturing the result of an attempted candidate action in Phase 7.

    Fields:
    -------
    attempted : bool
        True if an action was attempted (valid index, not padded or out-of-bounds).
    executed : bool
        True if the browser adapter actually executed the action.
    candidate_index : Optional[int]
        The 0-based candidate selection index chosen by the agent (0..19).
    candidate_id : Optional[Union[int, str]]
        The candidate identifier associated with candidate_index.
    operation_type : str
        The workflow-determined browser operation ("fill", "click", "verify").
    execution_success : bool
        True if Playwright / BrowserAdapter performed the DOM action without error.
    step_success : bool
        True if the executed action satisfied the private evaluator ground truth.
    failure_reason : Optional[str]
        Structured diagnostic string if execution or validation failed
        (e.g., "padded_candidate", "out_of_space", "incorrect_candidate", "not_executable").
    """

    attempted: bool
    executed: bool
    candidate_index: Optional[int]
    candidate_id: Optional[Union[int, str]]
    operation_type: str
    execution_success: bool
    step_success: bool
    failure_reason: Optional[str] = None
