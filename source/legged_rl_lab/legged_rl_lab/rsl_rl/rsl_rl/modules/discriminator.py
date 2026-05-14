"""AMP Discriminator network.

Distinguishes between expert (reference motion) and policy-generated transitions.
Input: concatenation of (state, next_state) AMP observation pairs.
Output: single raw logit (no squashing).

Design follows the reference implementation from TienKung-Lab / legged_lab:
- Pure LSGAN loss: MSE(D(expert), +1) + MSE(D(policy), -1)
- R1 gradient penalty on expert (λ=10)
- No tanh squashing, no logit regularization
- External (numpy running) normalizer, updated from both policy + expert
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .normalization import EmpiricalNormalization
from .mlp import MLP
from rsl_rl.utils import resolve_nn_activation


class AMPDiscriminator(nn.Module):
    """Discriminator for Adversarial Motion Priors (AMP).

    Takes concatenated (current_amp_obs, next_amp_obs) as input and outputs
    a raw logit indicating whether the transition is from the expert dataset.
    """

    def __init__(
        self,
        amp_obs_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (1024, 512, 256),
        activation: str = "relu",
        reward_scale: float = 0.3,
        gradient_penalty_coef: float = 10.0,
        logit_reg_coef: float = 0.0,  # kept for API compat; TienKung uses 0
        weight_decay: float = 1e-4,
        obs_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.amp_obs_dim = amp_obs_dim
        self.input_dim = amp_obs_dim * 2  # concatenation of (s, s')
        self.reward_scale = reward_scale
        self.gradient_penalty_coef = gradient_penalty_coef
        self.logit_reg_coef = logit_reg_coef
        self.weight_decay = weight_decay

        # Trunk MLP (without final activation)
        self.trunk = MLP(
            input_dim=self.input_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=list(hidden_dims[:-1]),
            activation=activation,
            last_activation=activation,
        )
        # Linear head
        self.head = nn.Linear(hidden_dims[-1], 1)

        # Observation normalizer (normalizes each half independently)
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(amp_obs_dim)
        else:
            self.obs_normalizer = None

    def forward(self, amp_obs_pair: torch.Tensor) -> torch.Tensor:
        """Forward pass → raw logit.

        Args:
            amp_obs_pair: Concatenated (state, next_state) of shape ``(B, input_dim)``.

        Returns:
            Raw logit of shape ``(B, 1)``.
        """
        if self.obs_normalizer is not None:
            pair_dim = amp_obs_pair.shape[-1]
            if pair_dim != self.input_dim:
                raise ValueError(
                    f"AMPDiscriminator.forward: expected input dim {self.input_dim} "
                    f"(= 2 × amp_obs_dim {self.amp_obs_dim}), but got {pair_dim}. "
                    f"Check that env amp obs dim ({pair_dim // 2} per step) matches "
                    f"motion_loader obs dim ({self.amp_obs_dim // 2} per frame)."
                )
            s, s_next = amp_obs_pair.chunk(2, dim=-1)
            s = self.obs_normalizer(s)
            s_next = self.obs_normalizer(s_next)
            amp_obs_pair = torch.cat([s, s_next], dim=-1)

        features = self.trunk(amp_obs_pair)
        return self.head(features)

    def predict_reward(self, amp_obs: torch.Tensor, amp_obs_next: torch.Tensor) -> torch.Tensor:
        """Style reward computed from raw logit (TienKung-style).

        r = reward_scale × clamp(1 - 0.25 * (D - 1)^2, min=0)

        At D = +1 (expert-like):   r = reward_scale × 1.0  (maximum)
        At D =  0 (boundary):      r = reward_scale × 0.75
        At D = -1 (policy-like):   r = reward_scale × 0.0  (minimum)
        At |D - 1| ≥ 2 (very far): r = 0

        Note: no tanh — the raw D is already bounded in practice by the grad
        penalty and LSGAN targets at ±1.  This matches Peng et al. 2021 and
        gives the disc room to learn meaningful score gradients instead of
        saturating against a tanh asymptote.
        """
        with torch.no_grad():
            pair = torch.cat([amp_obs, amp_obs_next], dim=-1)
            d = self.forward(pair)
            reward = self.reward_scale * torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)
        return reward

    def compute_loss(
        self,
        policy_pair: torch.Tensor,
        expert_pair: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """LSGAN loss with R1 gradient penalty.

        Expert target = +1; policy target = -1.  Raw-logit MSE, no tanh.

        Returns:
            (amp_loss, grad_pen_loss, logit_reg_loss)
        """
        all_pairs = torch.cat([policy_pair, expert_pair], dim=0)
        all_logits = self.forward(all_pairs)
        policy_logits, expert_logits = all_logits.split(policy_pair.shape[0], dim=0)

        expert_loss = torch.nn.functional.mse_loss(
            expert_logits, torch.ones_like(expert_logits)
        )
        policy_loss = torch.nn.functional.mse_loss(
            policy_logits, -torch.ones_like(policy_logits)
        )
        amp_loss = 0.5 * (expert_loss + policy_loss)

        # R1 gradient penalty on expert data (λ * ||grad||^2)
        grad_pen_loss = self._compute_gradient_penalty(expert_pair)

        # Logit regularization — kept for API compat but disabled by default.
        if self.logit_reg_coef > 0.0:
            logit_reg_loss = self.logit_reg_coef * torch.mean(
                expert_logits**2 + policy_logits**2
            )
        else:
            logit_reg_loss = torch.zeros((), device=amp_loss.device)

        return amp_loss, grad_pen_loss, logit_reg_loss

    def _compute_gradient_penalty(self, expert_pair: torch.Tensor) -> torch.Tensor:
        """R1 grad penalty: λ * E[||grad_x D(x)||^2] on expert data."""
        expert_pair = expert_pair.detach().requires_grad_(True)

        if self.obs_normalizer is not None:
            s, s_next = expert_pair.chunk(2, dim=-1)
            s_norm = self.obs_normalizer(s)
            s_next_norm = self.obs_normalizer(s_next)
            pair_norm = torch.cat([s_norm, s_next_norm], dim=-1)
        else:
            pair_norm = expert_pair

        features = self.trunk(pair_norm)
        logits = self.head(features)

        grad = torch.autograd.grad(
            outputs=logits,
            inputs=expert_pair,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
        )[0]

        # TienKung form: λ * mean(||grad||^2)
        grad_penalty = self.gradient_penalty_coef * grad.pow(2).sum(dim=1).mean()
        return grad_penalty

    def update_normalizer(self, amp_obs: torch.Tensor) -> None:
        """Update the observation normalizer with new data."""
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(amp_obs)
