"""AMP Discriminator network — single-window (not pair) input.

Following IsaacLab G1AmpEnv / legged_lab design:
- Disc input is a flat ``(B, hl * obs_per_frame)`` tensor (one trajectory
  window), NOT a concatenated ``(s, s')`` pair.  This avoids the redundant
  9-frame overlap between consecutive windows that previously kept disc_acc
  saturated at 1.0.
- Pure LSGAN loss: MSE(D(expert), +1) + MSE(D(policy), -1)
- R1 gradient penalty on expert (λ=10)
- No tanh squashing, no logit regularization
- Reward is multiplied by ``dt`` at predict time so ``reward_scale`` has
  units of "style reward / second".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .normalization import EmpiricalNormalization
from .mlp import MLP


class AMPDiscriminator(nn.Module):
    """Discriminator for Adversarial Motion Priors (AMP).

    Forward input: a single trajectory window flattened to
    ``(B, amp_obs_dim)`` where ``amp_obs_dim = history_length * per_frame_dim``.
    """

    def __init__(
        self,
        amp_obs_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (1024, 512),
        activation: str = "relu",
        reward_scale: float = 5.0,
        gradient_penalty_coef: float = 10.0,
        logit_reg_coef: float = 0.0,
        weight_decay: float = 1e-4,
        obs_normalization: bool = True,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.amp_obs_dim = amp_obs_dim
        self.input_dim = amp_obs_dim  # single window, no pair concat
        self.reward_scale = reward_scale
        self.gradient_penalty_coef = gradient_penalty_coef
        self.logit_reg_coef = logit_reg_coef
        self.weight_decay = weight_decay
        # Label smoothing — train disc to predict ±(1 - smoothing) instead of ±1.
        # Caps disc accuracy at ~95% in practice and prevents the saturation
        # mode where policy logits go to -∞ and style reward collapses to 0.
        self.label_smoothing = label_smoothing

        self.trunk = MLP(
            input_dim=self.input_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=list(hidden_dims[:-1]),
            activation=activation,
            last_activation=activation,
        )
        self.head = nn.Linear(hidden_dims[-1], 1)

        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(amp_obs_dim)
        else:
            self.obs_normalizer = None

    def forward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass → raw logit.

        Args:
            amp_obs: ``(B, amp_obs_dim)`` flat single window.
        Returns:
            Raw logit of shape ``(B, 1)``.
        """
        if amp_obs.shape[-1] != self.input_dim:
            raise ValueError(
                f"AMPDiscriminator.forward: expected input dim {self.input_dim}, "
                f"got {amp_obs.shape[-1]}.  Check that env amp_obs_dim "
                f"(={self.amp_obs_dim}) matches motion_loader output."
            )
        if self.obs_normalizer is not None:
            amp_obs = self.obs_normalizer(amp_obs)
        return self.head(self.trunk(amp_obs))

    def predict_reward(self, amp_obs: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Style reward from a single trajectory window.

        ``r = reward_scale * dt * clamp(1 - 0.25 * (D - 1)^2, min=0)``

        - At D = +1 (expert-like): r = reward_scale × dt × 1.0 (max)
        - At D =  0 (boundary):    r = reward_scale × dt × 0.75
        - At D = -1 (policy-like): r = reward_scale × dt × 0.0 (min)

        Multiplying by ``dt`` (env step seconds) gives the reward an
        intuitive "per-second" interpretation matching legged_lab's design.
        """
        with torch.no_grad():
            d = self.forward(amp_obs)
            reward = (
                self.reward_scale
                * dt
                * torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)
            )
        return reward

    def compute_loss(
        self,
        policy_obs: torch.Tensor,
        expert_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """LSGAN loss + R1 gradient penalty.

        Args:
            policy_obs: ``(B, amp_obs_dim)`` policy trajectory windows.
            expert_obs: ``(B, amp_obs_dim)`` expert trajectory windows.
        """
        all_obs = torch.cat([policy_obs, expert_obs], dim=0)
        all_logits = self.forward(all_obs)
        policy_logits, expert_logits = all_logits.split(policy_obs.shape[0], dim=0)

        pos_target = 1.0 - self.label_smoothing
        neg_target = -1.0 + self.label_smoothing
        expert_loss = torch.nn.functional.mse_loss(
            expert_logits, torch.full_like(expert_logits, pos_target)
        )
        policy_loss = torch.nn.functional.mse_loss(
            policy_logits, torch.full_like(policy_logits, neg_target)
        )
        amp_loss = 0.5 * (expert_loss + policy_loss)

        grad_pen_loss = self._compute_gradient_penalty(expert_obs)

        if self.logit_reg_coef > 0.0:
            logit_reg_loss = self.logit_reg_coef * torch.mean(
                expert_logits**2 + policy_logits**2
            )
        else:
            logit_reg_loss = torch.zeros((), device=amp_loss.device)

        return amp_loss, grad_pen_loss, logit_reg_loss

    def _compute_gradient_penalty(self, expert_obs: torch.Tensor) -> torch.Tensor:
        """R1 grad penalty: λ * E[||grad_x D(x)||^2] on expert data."""
        expert_obs = expert_obs.detach().requires_grad_(True)

        if self.obs_normalizer is not None:
            obs_norm = self.obs_normalizer(expert_obs)
        else:
            obs_norm = expert_obs

        logits = self.head(self.trunk(obs_norm))

        grad = torch.autograd.grad(
            outputs=logits,
            inputs=expert_obs,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
        )[0]

        return self.gradient_penalty_coef * grad.pow(2).sum(dim=1).mean()

    def update_normalizer(self, amp_obs: torch.Tensor) -> None:
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(amp_obs)
