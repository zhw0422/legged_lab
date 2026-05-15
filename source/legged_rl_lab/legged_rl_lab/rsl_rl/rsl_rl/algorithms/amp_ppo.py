# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP-PPO algorithm.

Extends the standard PPO algorithm with:
- An AMP discriminator for style rewards
- A replay buffer for policy AMP transitions
- Joint optimization of actor-critic and discriminator
"""

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules.discriminator import AMPDiscriminator
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import AMPReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_nn_activation, resolve_obs_groups, resolve_optimizer


class AMPPPO(PPO):
    """PPO with Adversarial Motion Priors (AMP).

    The AMP discriminator provides style rewards that encourage the policy to
    produce motions similar to reference data. The total reward is a weighted
    blend of task reward and style reward.

    Reference:
        Peng et al. "AMP: Adversarial Motion Priors for Stylized Physics-Based
        Character Animation." ACM Trans. Graphics, 2021.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        # PPO params
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND/Symmetry/Multi-GPU (passed through to parent)
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        # AMP params
        amp_cfg: dict | None = None,
    ) -> None:
        # Initialize parent PPO (handles actor, critic, optimizer, storage, etc.)
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        # AMP components (initialized lazily via set_reference_data)
        self._amp_cfg = amp_cfg or {}
        self.discriminator: AMPDiscriminator | None = None
        self.amp_replay_buffer: AMPReplayBuffer | None = None
        self.reference_data: torch.Tensor | None = None
        self.amp_history_length: int = 2
        self._amp_transition_state: torch.Tensor | None = None

        # AMP hyperparameters
        self.amp_task_reward_lerp = self._amp_cfg.get("amp_task_reward_lerp", 0.5)

    def set_reference_data(self, reference_data: torch.Tensor, history_length: int = 2) -> None:
        """Set expert reference data and initialize discriminator + replay buffer.

        Args:
            reference_data: Expert AMP features ``(N, obs_dim_per_frame)``.
            history_length: Number of consecutive frames in AMP observation.
        """
        self.reference_data = reference_data.to(self.device)
        self.amp_history_length = history_length
        amp_obs_dim = reference_data.shape[1] * history_length

        # Create discriminator
        hidden_dims = self._amp_cfg.get("amp_discriminator_hidden_dims", [1024, 512])
        activation = self._amp_cfg.get("amp_discriminator_activation", "relu")
        reward_scale = self._amp_cfg.get("amp_reward_scale", 1.0)
        gradient_penalty_coef = self._amp_cfg.get("amp_disc_gradient_penalty_coef", 5.0)
        logit_reg_coef = self._amp_cfg.get("amp_disc_logit_reg_coef", 0.05)
        disc_weight_decay = self._amp_cfg.get("amp_disc_weight_decay", 1e-4)
        obs_norm = self._amp_cfg.get("amp_disc_obs_normalization", True)
        label_smoothing = self._amp_cfg.get("amp_disc_label_smoothing", 0.0)

        self.discriminator = AMPDiscriminator(
            amp_obs_dim=amp_obs_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            reward_scale=reward_scale,
            gradient_penalty_coef=gradient_penalty_coef,
            logit_reg_coef=logit_reg_coef,
            weight_decay=disc_weight_decay,
            obs_normalization=obs_norm,
            label_smoothing=label_smoothing,
        ).to(self.device)

        # Create replay buffer
        buffer_size = self._amp_cfg.get("amp_replay_buffer_size", 1_000_000)
        self.amp_replay_buffer = AMPReplayBuffer(buffer_size, amp_obs_dim, self.device)

        # Add discriminator params to optimizer
        amp_lr = self._amp_cfg.get("amp_learning_rate", 1e-4)
        self.optimizer.add_param_group({
            "params": self.discriminator.trunk.parameters(),
            "lr": amp_lr,
            "weight_decay": disc_weight_decay,
        })
        self.optimizer.add_param_group({
            "params": self.discriminator.head.parameters(),
            "lr": amp_lr,
            "weight_decay": disc_weight_decay * 10,
        })

        print(f"[AMPPPO] Discriminator initialized: amp_obs_dim={amp_obs_dim}, "
              f"hidden_dims={hidden_dims}, activation={activation}")
        print(f"[AMPPPO] Replay buffer size: {buffer_size}")
        print(f"[AMPPPO] Task/Style reward blend: {self.amp_task_reward_lerp:.2f} / "
              f"{1.0 - self.amp_task_reward_lerp:.2f}")

    def record_amp_obs(self, amp_obs: torch.Tensor) -> None:
        """Record current AMP single-window observation for replay insertion.

        Args:
            amp_obs: Current AMP observation ``(num_envs, amp_obs_dim)``.
        """
        self._amp_transition_state = amp_obs.clone()

    def process_amp_transition(self, current_amp_obs: torch.Tensor) -> None:
        """Insert the current AMP single-window obs into the replay buffer.

        Single-window approach: we store one trajectory window per env step,
        not (s, s') pairs.  ``current_amp_obs`` is the observation BEFORE
        env reset (passed in from the runner via ``extras["amp_obs"]``).
        """
        if self.amp_replay_buffer is not None:
            self.amp_replay_buffer.insert(current_amp_obs)

    def compute_style_reward(self, amp_obs: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute style reward from a single AMP window.

        Args:
            amp_obs: AMP observation ``(B, amp_obs_dim)``.
            dt: env step_dt for "per-second" reward scaling.
        Returns:
            Style reward ``(B, 1)``.
        """
        if self.discriminator is None:
            return torch.zeros(amp_obs.shape[0], 1, device=self.device)
        return self.discriminator.predict_reward(amp_obs, dt=dt)

    def blend_rewards(self, task_rewards: torch.Tensor, style_rewards: torch.Tensor) -> torch.Tensor:
        """Blend task and style rewards as a convex combination."""
        task_r = task_rewards.view(-1)
        style_r = style_rewards.view(-1)
        lerp = self.amp_task_reward_lerp
        return lerp * task_r + (1.0 - lerp) * style_r

    def _sample_expert_obs(self, batch_size: int) -> torch.Tensor:
        """Sample expert single-window observations from reference data.

        Returns ``(batch_size, hl * obs_dim_per_frame)`` flat tensor —
        consecutive ``hl`` frames concatenated, matching the env's flat
        AMP observation layout.
        """
        num_frames = self.reference_data.shape[0]
        hl = self.amp_history_length

        if hl <= 1:
            indices = torch.randint(0, num_frames, (batch_size,), device=self.device)
            return self.reference_data[indices]

        max_start = max(1, num_frames - hl + 1)
        start = torch.randint(0, max_start, (batch_size,), device=self.device)
        offsets = torch.arange(hl, device=self.device)
        frame_idx = start.unsqueeze(1) + offsets.unsqueeze(0)
        frame_idx = frame_idx.clamp(max=num_frames - 1)
        clips = self.reference_data[frame_idx]                 # (B, hl, D)
        return clips.reshape(batch_size, -1)                   # (B, hl*D)

    def update(self) -> dict[str, float]:
        """Run PPO + AMP discriminator update."""
        # If no discriminator, fall back to standard PPO
        if self.discriminator is None or self.reference_data is None:
            return super().update()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_logit_reg_loss = 0.0
        mean_disc_accuracy_policy = 0.0
        mean_disc_accuracy_expert = 0.0
        actual_updates = 0

        # Get PPO mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # ---- PPO part (same as parent) ----
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            # Recompute
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Adaptive LR
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    # Only update LR for actor-critic param groups (first group)
                    for pg in self.optimizer.param_groups[:1]:
                        pg["lr"] = self.learning_rate

            # Surrogate loss
            # Clamp log-ratio to [-10, 10] to prevent exp() overflow when action std
            # is large and log_prob_old approaches -inf at the tanh boundary.
            log_ratio = actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
            log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
            ratio = torch.exp(log_ratio)
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # ---- AMP discriminator part (single-window) ----
            mini_batch_size = original_batch_size
            # Get policy single-window observations from replay buffer
            amp_policy_gen = self.amp_replay_buffer.feed_forward_generator(1, mini_batch_size)
            policy_obs_amp = next(amp_policy_gen)               # (B, amp_obs_dim)

            # Get expert single-window observations
            expert_obs_amp = self._sample_expert_obs(mini_batch_size)

            # Discriminator loss
            amp_loss, grad_pen_loss, logit_reg_loss = self.discriminator.compute_loss(
                policy_obs_amp, expert_obs_amp
            )

            # Total loss
            loss = ppo_loss + amp_loss + grad_pen_loss + logit_reg_loss

            # NaN guard: skip this mini-batch if any loss term is non-finite
            if not torch.isfinite(loss):
                import warnings
                warnings.warn(
                    f"[AMPPPO] Non-finite loss detected (ppo={ppo_loss.item():.4f}, "
                    f"amp={amp_loss.item():.4f}, gp={grad_pen_loss.item():.4f}). "
                    f"Skipping update for this mini-batch.",
                    RuntimeWarning,
                )
                continue

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Synchronize actor/critic gradients across all GPUs (discriminator stays local per GPU)
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Clip gradients only for actor-critic
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Update discriminator normalizer from BOTH policy AND expert data.
            # Bug fix: if normalizer only trains on policy data, expert obs get
            # normalized by policy statistics → out-of-distribution values give
            # the discriminator a trivial "free" cue (normalization leak).
            if self.amp_replay_buffer.num_samples > 0:
                # Policy data
                norm_indices = torch.randint(
                    0, self.amp_replay_buffer.num_samples,
                    (min(512, self.amp_replay_buffer.num_samples),),
                    device=self.device,
                )
                self.discriminator.update_normalizer(self.amp_replay_buffer.states[norm_indices])

                # Expert data: construct history-stacked obs to match amp_obs_dim
                num_frames = self.reference_data.shape[0]
                hl = self.amp_history_length
                max_start = max(1, num_frames - hl)
                n_exp = min(512, max_start)
                exp_idx = torch.randint(0, max_start, (n_exp,), device=self.device)
                if hl > 1:
                    offsets = torch.arange(hl, device=self.device)
                    frame_indices = exp_idx.unsqueeze(1) + offsets.unsqueeze(0)
                    expert_obs_hist = self.reference_data[frame_indices].reshape(n_exp, -1)
                else:
                    expert_obs_hist = self.reference_data[exp_idx]
                self.discriminator.update_normalizer(expert_obs_hist)

            # Compute discriminator accuracy for logging.
            # Raw-logit disc (no tanh): threshold at 0 — policy should produce
            # negative logits, expert should produce positive logits.
            with torch.no_grad():
                all_obs = torch.cat([policy_obs_amp, expert_obs_amp], dim=0)
                all_logits = self.discriminator(all_obs)
                p_logits, e_logits = all_logits.split(mini_batch_size, dim=0)
                acc_policy = (p_logits < 0.0).float().mean().item()
                acc_expert = (e_logits > 0.0).float().mean().item()

            # Accumulate
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_logit_reg_loss += logit_reg_loss.item()
            mean_disc_accuracy_policy += acc_policy
            mean_disc_accuracy_expert += acc_expert
            actual_updates += 1

        # Average over actually executed updates (skipped batches contribute nothing)
        num_updates = max(1, actual_updates)
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_logit_reg_loss /= num_updates
        mean_disc_accuracy_policy /= num_updates
        mean_disc_accuracy_expert /= num_updates

        self.storage.clear()

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp_loss": mean_amp_loss,
            "grad_pen_loss": mean_grad_pen_loss,
            "logit_reg_loss": mean_logit_reg_loss,
            "disc_accuracy_policy": mean_disc_accuracy_policy,
            "disc_accuracy_expert": mean_disc_accuracy_expert,
        }

    def broadcast_parameters(self) -> None:
        """Broadcast actor, critic, AND discriminator from rank 0 to all GPUs."""
        super().broadcast_parameters()
        if self.discriminator is not None:
            disc_params = [self.discriminator.state_dict()]
            torch.distributed.broadcast_object_list(disc_params, src=0)
            self.discriminator.load_state_dict(disc_params[0])

    def train_mode(self) -> None:
        super().train_mode()
        if self.discriminator is not None:
            self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        if self.discriminator is not None:
            self.discriminator.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        if self.discriminator is not None:
            saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        # During play the discriminator is not initialized, so the optimizer only has 1
        # param group (policy) while the checkpoint has 3 (policy + 2 disc groups).
        # Skip optimizer loading when discriminator is absent to avoid ValueError.
        if self.discriminator is None and load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": True,
            }
        result = super().load(loaded_dict, load_cfg, strict)
        if self.discriminator is not None and "discriminator_state_dict" in loaded_dict:
            try:
                self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=False)
            except RuntimeError:
                # Architecture mismatch (e.g. hidden_dims changed) — start discriminator fresh.
                print("[AMPPPO] Discriminator architecture changed; re-initializing from scratch.")
            # Clear stale optimizer state for discriminator params.
            # When discriminator architecture changes between checkpoint and current
            # code, load_state_dict silently loads wrong-shaped momentum/variance
            # tensors that would crash at the first optimizer.step().  Deleting the
            # state here forces Adam to start fresh for disc params only.
            for p in self.discriminator.parameters():
                self.optimizer.state.pop(p, None)
        return result
