"""Circular replay buffer for AMP single-window trajectories.

Stores single trajectory windows ``(B, amp_obs_dim)`` produced by the policy
during rollout — NOT (s, s') pairs.  This matches legged_lab / IsaacLab
G1AmpEnv's discriminator input scheme.
"""

from __future__ import annotations

import torch
from collections.abc import Generator


class AMPReplayBuffer:
    """Fixed-size circular replay buffer for AMP single-window observations."""

    def __init__(self, buffer_size: int, obs_dim: int, device: str = "cpu") -> None:
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.device = device

        self.states = torch.zeros(buffer_size, obs_dim, device=device)
        self.step = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor) -> None:
        """Batch-insert single states, handling wrap-around."""
        batch_size = states.shape[0]
        if batch_size == 0:
            return

        end = self.step + batch_size
        if end <= self.buffer_size:
            self.states[self.step:end] = states
        else:
            first_part = self.buffer_size - self.step
            self.states[self.step:] = states[:first_part]
            second_part = batch_size - first_part
            self.states[:second_part] = states[first_part:]

        self.step = end % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + batch_size)

    def feed_forward_generator(
        self, num_mini_batches: int, mini_batch_size: int
    ) -> Generator[torch.Tensor, None, None]:
        """Yield mini-batches of single states."""
        total_needed = num_mini_batches * mini_batch_size
        if total_needed <= self.num_samples:
            indices = torch.randperm(self.num_samples, device=self.device)[:total_needed]
        else:
            indices = torch.randint(0, self.num_samples, (total_needed,), device=self.device)

        for i in range(num_mini_batches):
            start = i * mini_batch_size
            batch_idx = indices[start:start + mini_batch_size]
            yield self.states[batch_idx]
