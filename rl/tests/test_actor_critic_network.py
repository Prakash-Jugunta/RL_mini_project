"""
Unit tests for CandidateAwareActorCriticNetwork.
Verifies network dimensions, gradient flow, action masking, actor permutation equivariance,
critic permutation invariance, parameter breakdown, and NoValidActionError protection.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pytest

from rl.agents.common.actor_critic_network import (
    CandidateAwareActorCriticNetwork,
    apply_action_mask,
    ActorCriticOutput,
    NoValidActionError,
)


def create_dummy_observation_batch(batch_size: int = 4, num_candidates: int = 20):
    np.random.seed(42)
    torch.manual_seed(42)

    objective = torch.randn((batch_size, 64), dtype=torch.float32)
    context = torch.randn((batch_size, 30), dtype=torch.float32)
    candidates = torch.randn((batch_size, num_candidates, 91), dtype=torch.float32)

    mask = torch.zeros((batch_size, num_candidates), dtype=torch.float32)
    for b in range(batch_size):
        num_valid = np.random.randint(1, num_candidates + 1)
        mask[b, :num_valid] = 1.0

    return {
        "objective": objective,
        "context": context,
        "candidates": candidates,
        "candidate_mask": mask,
    }


def test_network_shapes_and_outputs():
    net = CandidateAwareActorCriticNetwork()
    obs = create_dummy_observation_batch(batch_size=4)

    out = net(obs)
    assert isinstance(out, ActorCriticOutput)
    assert out.logits.shape == (4, 20)
    assert out.value.shape == (4, 1)

    logits_only = net.forward_actor(obs)
    assert logits_only.shape == (4, 20)

    val_only = net.forward_critic(obs)
    assert val_only.shape == (4, 1)


def test_parameter_count_breakdown():
    """Verify programmatic parameter breakdown distinguishes shared encoders from actor/critic heads."""
    net = CandidateAwareActorCriticNetwork()
    bd = net.get_parameter_count_breakdown()

    assert bd["total_unique_trainable_parameters"] == 98530
    assert bd["shared_encoders"] == 40672
    assert bd["actor_head"] == 28929
    assert bd["critic_head"] == 28929


def test_gradient_flow():
    net = CandidateAwareActorCriticNetwork()
    obs = create_dummy_observation_batch(batch_size=2)

    out = net(obs)
    loss = out.logits.sum() + out.value.sum()
    loss.backward()

    for name, param in net.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_action_masking_and_zero_valid_guard():
    net = CandidateAwareActorCriticNetwork()
    obs = create_dummy_observation_batch(batch_size=2)

    logits = net.forward_actor(obs)
    masked_logits = apply_action_mask(logits, obs["candidate_mask"], mask_value=-1e9)

    for b in range(2):
        valid_indices = torch.where(obs["candidate_mask"][b] > 0)[0]
        invalid_indices = torch.where(obs["candidate_mask"][b] <= 0)[0]

        for idx in valid_indices:
            assert masked_logits[b, idx].item() > -1e8
        for idx in invalid_indices:
            assert masked_logits[b, idx].item() <= -1e8

    # Zero valid action mask guard
    zero_mask = torch.zeros((1, 20), dtype=torch.float32)
    with pytest.raises(NoValidActionError):
        apply_action_mask(logits[:1], zero_mask)


def test_actor_permutation_equivariance():
    """
    CRITICAL STEP 20 TEST:
    Permuting candidate order permutes actor logits in exactly the same order.
    """
    net = CandidateAwareActorCriticNetwork()
    net.eval()

    obs = create_dummy_observation_batch(batch_size=1)
    cand_orig = obs["candidates"].clone()
    mask_orig = obs["candidate_mask"].clone()

    perm = torch.randperm(20)

    obs_perm = {
        "objective": obs["objective"].clone(),
        "context": obs["context"].clone(),
        "candidates": cand_orig[:, perm, :],
        "candidate_mask": mask_orig[:, perm],
    }

    with torch.no_grad():
        logits_orig = net.forward_actor(obs)
        logits_perm = net.forward_actor(obs_perm)

    logits_reordered = logits_orig[:, perm]
    torch.testing.assert_close(logits_perm, logits_reordered, atol=1e-5, rtol=1e-5)


def test_critic_permutation_invariance():
    """
    CRITICAL STEP 20 TEST:
    Permuting valid candidate order leaves critic state value V(s) invariant.
    """
    net = CandidateAwareActorCriticNetwork()
    net.eval()

    obs = create_dummy_observation_batch(batch_size=1)
    cand_orig = obs["candidates"].clone()
    mask_orig = obs["candidate_mask"].clone()

    perm = torch.randperm(20)

    obs_perm = {
        "objective": obs["objective"].clone(),
        "context": obs["context"].clone(),
        "candidates": cand_orig[:, perm, :],
        "candidate_mask": mask_orig[:, perm],
    }

    with torch.no_grad():
        val_orig = net.forward_critic(obs)
        val_perm = net.forward_critic(obs_perm)

    torch.testing.assert_close(val_orig, val_perm, atol=1e-5, rtol=1e-5)
