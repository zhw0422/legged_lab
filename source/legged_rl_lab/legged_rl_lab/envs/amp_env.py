# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP runtime environment helpers.

This module keeps the two AMP-specific runtime pieces together:

- ``AMPManagerBasedRLEnv``: captures pre-reset AMP observations from IsaacLab.
- ``AmpRslRlVecEnvWrapper``: adapts the IsaacLab env to the RSL-RL ``VecEnv`` interface.

Keeping them in one module makes the AMP data flow easier to follow end-to-end.
"""

from __future__ import annotations

import os

import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from rsl_rl.env import VecEnv


class AMPManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv extended for AMP training.

    Key difference from the base class: AMP observations are computed **before** the reset
    and stored in ``extras["amp_obs"]``, so the AMP (state, next_state) transition pairs
    reflect the actual physics transition rather than a post-reset artifact.

    The ``amp_obs_group`` parameter specifies which observation group to treat as the AMP
    observation (default: ``"amp"``).

    Exposes ``amp_observation_space`` and ``collect_reference_motions`` for skrl AMP support.
    """

    cfg: ManagerBasedRLEnvCfg
    """The configuration object for the environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._amp_obs_group = "amp"

        # Motion data for AMP (used by both RSL-RL and skrl)
        self._amp_motion_data: torch.Tensor | None = None
        """Loaded reference motion AMP features, shape (N, obs_dim)."""
        self._amp_motion_loader = None
        """MotionLoader instance for dynamic sampling."""
        self._load_motion_data()

    # ------------------------------------------------------------------
    # skrl AMP compatibility
    # ------------------------------------------------------------------

    @property
    def amp_observation_space(self) -> spaces.Box:
        """AMP observation space for skrl agent instantiation.

        Returns a gymnasium Box space matching the AMP observation group's
        output dimensionality.
        """
        # Compute one dummy observation to get the shape
        amp_obs = self.observation_manager.compute_group(
            self._amp_obs_group, update_history=False
        )
        obs_dim = amp_obs.shape[-1]
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        """Collect reference motion AMP features for skrl AMP training.

        Samples random frames from the loaded motion data.

        Args:
            num_samples: Number of reference motion frames to collect.

        Returns:
            Tensor of shape ``(num_samples, amp_obs_dim)``.
        """
        if self._amp_motion_data is None:
            raise RuntimeError(
                "No reference motion data loaded. Set amp_motion_files in the env config."
            )
        N = self._amp_motion_data.shape[0]
        idx = torch.randint(0, N, (num_samples,), device=self._amp_motion_data.device)
        return self._amp_motion_data[idx]

    # ------------------------------------------------------------------
    # Motion data loading
    # ------------------------------------------------------------------

    def _load_motion_data(self) -> None:
        """Load reference motion data for AMP training.

        Reads ``amp_motion_files`` from the config (a file path or directory)
        and stores the AMP feature tensor for use by ``collect_reference_motions``.
        """
        motion_path = getattr(self.cfg, "amp_motion_files", None)
        if not motion_path or not os.path.exists(motion_path):
            return

        from legged_rl_lab.managers.motion_loader import MotionLoader

        robot_type = getattr(self.cfg, "robot_type", "g1")
        loader = MotionLoader(device=str(self.device), robot=robot_type)
        self._amp_motion_data = loader.load(motion_path)
        self._amp_motion_loader = loader

    # ------------------------------------------------------------------
    # Step (AMP-aware)
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        """Override step to capture AMP observations before reset.

        Follows the exact flow of ``ManagerBasedRLEnv.step()`` in IsaacLab 2.3.0, with the
        single addition of computing AMP observations after rewards/terminations but before
        environments are reset.

        Flow:
        1. Process actions and record pre-step
        2. Physics decimation loop (apply_action → write → sim.step → render → update)
        3. Post-step counters + terminations + rewards
        4. Recorder post-step (conditional)
        5. **[AMP] Compute AMP obs before reset** → store in ``extras["amp_obs"]``
        6. Reset terminated envs (with recorder pre/post-reset calls)
        7. Commands + interval events
        8. Final observations (post-reset, with history update)
        9. Return
        """
        # -- 1. Process actions
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        # -- 2. Physics decimation loop (matches ManagerBasedRLEnv exactly)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # apply actions into actuators
            self.action_manager.apply_action()
            # write actions to sim
            self.scene.write_data_to_sim()
            # simulate one physics step
            self.sim.step(render=False)
            # recorder hook for each decimation step
            self.recorder_manager.record_post_physics_decimation_step()
            # render at configured interval
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update scene buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # -- 3. Post-step counters, terminations, rewards
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # -- 4. Recorder post-step (only when recorder terms are active)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- 5. [AMP SPECIFIC] Capture AMP observations BEFORE any env reset.
        # This ensures the discriminator sees genuine physics transitions, not
        # post-reset states for terminated environments.
        # Use `active_terms` (IsaacLab 2.x) with fallback to `_group_obs_term_names`.
        _obs_groups = getattr(
            self.observation_manager, "active_terms",
            getattr(self.observation_manager, "_group_obs_term_names", {})
        )
        if self._amp_obs_group in _obs_groups:
            self.extras["amp_obs"] = self.observation_manager.compute_group(
                self._amp_obs_group, update_history=False
            )

        # -- 6. Reset terminated environments
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            # Re-render after reset if RTX sensors are present
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- 7. Commands and interval events
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # -- 8. Final observations (post-reset, history updated for all groups)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # -- 9. Add time_outs for reward bootstrapping in RL algorithms
        self.extras["time_outs"] = self.reset_time_outs

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


class AmpRslRlVecEnvWrapper(VecEnv):
    """Wrap a Gymnasium IsaacLab env as an RSL-RL VecEnv with AMP extras."""

    def __init__(
        self,
        env: gym.Env,
        clip_actions: float | None = None,
        amp_obs_group: str = "amp",
    ) -> None:
        super().__init__()
        self.env = env
        self.unwrapped_env = env.unwrapped
        self._clip_actions = clip_actions
        self._amp_obs_group = amp_obs_group

        self.num_envs = self.unwrapped_env.num_envs
        self.num_actions = self.unwrapped_env.action_manager.action.shape[1]
        self.max_episode_length = int(
            self.unwrapped_env.max_episode_length
            if hasattr(self.unwrapped_env, "max_episode_length")
            else 1000
        )
        self.device = self.unwrapped_env.device
        self.cfg = self.unwrapped_env.cfg

        obs_dict, _ = self.env.reset()
        self._obs = self._build_obs_tensordict(obs_dict)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped_env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped_env.episode_length_buf = value

    def get_observations(self):
        return self._obs

    def step(self, actions: torch.Tensor):
        if self._clip_actions is not None:
            actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        obs_dict, rewards, terminated, time_outs, extras = self.env.step(actions)
        dones = (terminated | time_outs).to(dtype=torch.long)
        extras["time_outs"] = time_outs

        obs = self._build_obs_tensordict(obs_dict)
        self._obs = obs
        return obs, rewards, dones, extras

    def reset(self):
        obs_dict, extras = self.env.reset()
        obs = self._build_obs_tensordict(obs_dict)
        self._obs = obs
        return obs, extras

    def close(self) -> None:
        self.env.close()

    def _build_obs_tensordict(self, obs_dict):
        from tensordict import TensorDict

        result = {}
        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor):
                result[key] = value
            elif hasattr(value, "shape") and hasattr(value, "to"):
                result[key] = value
            elif hasattr(value, "items"):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        result[f"{key}/{subkey}"] = subvalue

        if not result:
            raise RuntimeError(
                f"No tensors found in obs_dict. "
                f"Keys: {list(obs_dict.keys())}. "
                f"Value types: {[type(v).__name__ for v in obs_dict.values()]}"
            )

        if "policy" in result:
            batch_size = result["policy"].shape[0]
        else:
            batch_size = next(iter(result.values())).shape[0]

        return TensorDict(result, batch_size=[batch_size], device=self.device)
