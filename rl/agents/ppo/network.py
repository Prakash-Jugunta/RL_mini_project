"""
PPO Network re-exports CandidateAwareActorCriticNetwork from rl.agents.common.actor_critic_network.
"""

from rl.agents.common.actor_critic_network import (
    CandidateAwareActorCriticNetwork,
    ActorCriticOutput,
    apply_action_mask,
)

__all__ = [
    "CandidateAwareActorCriticNetwork",
    "ActorCriticOutput",
    "apply_action_mask",
]
