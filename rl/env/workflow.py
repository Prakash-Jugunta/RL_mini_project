from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class WorkflowStep:
    """
    Representation of a single step in a UI workflow.

    expected_role is PRIVATE EVALUATOR GROUND TRUTH.
    It is used by the environment to evaluate action correctness.
    It MUST NEVER be exposed in agent observations or policy-visible candidate structures.

    agent_intent is the PUBLIC AGENT-VISIBLE task objective description.
    """
    step_id: str
    action_type: str  # "fill", "click", "verify"
    value: Optional[str] = None
    expected_role: Optional[str] = None
    success_condition: Optional[str] = None
    agent_intent: str = ""


@dataclass
class WorkflowDefinition:
    workflow_id: str
    start_path: str
    steps: List[WorkflowStep]


# ─────────────────────────────────────────────────────────────────────────────
# Pre-defined Workflows matching frozen Phase 1/2 application routes & roles
# ─────────────────────────────────────────────────────────────────────────────

WORKFLOW_LOGIN = WorkflowDefinition(
    workflow_id="LOGIN",
    start_path="/login",
    steps=[
        WorkflowStep(
            step_id="ENTER_USERNAME",
            action_type="fill",
            value="testuser",
            expected_role="username-input",
            agent_intent="enter username"
        ),
        WorkflowStep(
            step_id="ENTER_PASSWORD",
            action_type="fill",
            value="password123",
            expected_role="password-input",
            agent_intent="enter password"
        ),
        WorkflowStep(
            step_id="CLICK_LOGIN",
            action_type="click",
            value=None,
            expected_role="login-action",
            agent_intent="login"
        ),
        WorkflowStep(
            step_id="VERIFY_DASHBOARD",
            action_type="verify",
            value="Welcome, testuser",
            expected_role="dashboard-welcome",
            success_condition="text_contains",
            agent_intent="verify welcome message"
        ),
    ]
)

WORKFLOW_SEARCH = WorkflowDefinition(
    workflow_id="SEARCH",
    start_path="/products",
    steps=[
        WorkflowStep(
            step_id="ENTER_SEARCH_QUERY",
            action_type="fill",
            value="Wireless Mouse",
            expected_role="search-input",
            agent_intent="search Wireless Mouse"
        ),
        WorkflowStep(
            step_id="CLICK_SEARCH",
            action_type="click",
            value=None,
            expected_role="search-action",
            agent_intent="search"
        ),
        WorkflowStep(
            step_id="SELECT_PRODUCT",
            action_type="click",
            value=None,
            expected_role="product-result",
            agent_intent="select Wireless Mouse product"
        ),
        WorkflowStep(
            step_id="VERIFY_PRODUCT_PAGE",
            action_type="verify",
            value="/products/wireless-mouse",
            expected_role="add-cart-action",
            success_condition="url_equals",
            agent_intent="verify product page URL"
        ),
    ]
)

WORKFLOW_PROFILE = WorkflowDefinition(
    workflow_id="PROFILE",
    start_path="/profile",
    steps=[
        WorkflowStep(
            step_id="ENTER_NAME",
            action_type="fill",
            value="Test User",
            expected_role="profile-name-input",
            agent_intent="enter profile name"
        ),
        WorkflowStep(
            step_id="ENTER_EMAIL",
            action_type="fill",
            value="test@example.com",
            expected_role="profile-email-input",
            agent_intent="enter profile email"
        ),
        WorkflowStep(
            step_id="CLICK_SAVE",
            action_type="click",
            value=None,
            expected_role="profile-save-action",
            agent_intent="save profile"
        ),
        WorkflowStep(
            step_id="VERIFY_SUCCESS",
            action_type="verify",
            value="Profile updated successfully",
            expected_role="profile-success-message",
            success_condition="text_contains",
            agent_intent="verify profile updated"
        ),
    ]
)

WORKFLOW_CHECKOUT = WorkflowDefinition(
    workflow_id="CHECKOUT",
    start_path="/products/wireless-mouse",
    steps=[
        WorkflowStep(
            step_id="ADD_TO_CART",
            action_type="click",
            value=None,
            expected_role="add-cart-action",
            agent_intent="add product to cart"
        ),
        WorkflowStep(
            step_id="CLICK_CART",
            action_type="click",
            value=None,
            expected_role="nav-cart",
            agent_intent="open cart"
        ),
        WorkflowStep(
            step_id="PROCEED_CHECKOUT",
            action_type="click",
            value=None,
            expected_role="checkout-action",
            agent_intent="proceed to checkout"
        ),
        WorkflowStep(
            step_id="CONFIRM_ORDER",
            action_type="click",
            value=None,
            expected_role="confirm-order-action",
            agent_intent="confirm order"
        ),
        WorkflowStep(
            step_id="VERIFY_ORDER_SUCCESS",
            action_type="verify",
            value="Order placed successfully",
            expected_role="order-success-message",
            success_condition="text_contains",
            agent_intent="verify order placed"
        ),
    ]
)

WORKFLOW_REGISTRY: Dict[str, WorkflowDefinition] = {
    "LOGIN": WORKFLOW_LOGIN,
    "SEARCH": WORKFLOW_SEARCH,
    "PROFILE": WORKFLOW_PROFILE,
    "CHECKOUT": WORKFLOW_CHECKOUT,
}
