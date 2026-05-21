# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Gait-periodic reward functions matching TienKung-Lab design.

These rewards use the gait phase clock to shape foot contact patterns —
rewarding low contact force during the air phase and high force during the
stance phase.  Together they give the policy an explicit stepping cadence
prior so the feet don't just slide along the ground.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _gait_phase(
    env: ManagerBasedRLEnv,
    offset: float = 0.0,
    cycle: float = 0.85,
) -> torch.Tensor:
    """Compute normalised gait phase for a single foot.

    Returns:
        phase in [0, 1) — 0 = heel strike, 1 = next heel strike.
    """
    step_dt = env.cfg.sim.dt * env.cfg.decimation
    t = env.episode_length_buf.float() * step_dt / cycle
    return (t + offset) % 1.0


def gait_feet_frc_perio(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    cycle: float = 0.85,
    offset_l: float = 0.38,
    offset_r: float = 0.88,
    air_ratio: float = 0.38,
    threshold: float = 500.0,
) -> torch.Tensor:
    """Reward periodic foot-force pattern: low force during air, high during stance.

    For each foot, the phase clock determines whether it should be in the air
    (phase < air_ratio) or on the ground (phase >= air_ratio).  The reward is:

      - air:   1.0 − clip(f / threshold, 0, 1)   (want force ≈ 0)
      - stance:     clip(f / threshold, 0, 1)    (want force high)

    Returns:
        Scalar reward per environment (averaged across feet).
    """
    sensor = env.scene[sensor_cfg.name]
    forces = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3], dim=-1)
    # forces: (num_envs, num_feet), typically 2 feet

    phase_l = _gait_phase(env, offset=offset_l, cycle=cycle)
    phase_r = _gait_phase(env, offset=offset_r, cycle=cycle)
    in_air_l = (phase_l < air_ratio).float()
    in_air_r = (phase_r < air_ratio).float()

    norm_force = (forces / threshold).clamp(0.0, 1.0)  # (N, F)
    air_reward = 1.0 - norm_force
    stance_reward = norm_force

    reward_l = in_air_l * air_reward[:, 0] + (1.0 - in_air_l) * stance_reward[:, 0]
    reward_r = in_air_r * air_reward[:, 1] + (1.0 - in_air_r) * stance_reward[:, 1]

    return (reward_l + reward_r) * 0.5


def gait_feet_spd_perio(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cycle: float = 0.85,
    offset_l: float = 0.38,
    offset_r: float = 0.88,
    air_ratio: float = 0.38,
    speed_threshold: float = 1.5,
) -> torch.Tensor:
    """Reward periodic foot-speed pattern: fast feet during air, slow during stance.

    During the air phase the foot should move (high speed); during stance it
    should stay planted (low speed).  The reward is:

      - air:       clip(s / speed_threshold, 0, 1)
      - stance:    1.0 − clip(s / speed_threshold, 0, 1)

    Returns:
        Scalar reward per environment (averaged across feet).
    """
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
    foot_speeds = torch.norm(asset.data.body_lin_vel_w[:, body_ids, :], dim=-1)

    phase_l = _gait_phase(env, offset=offset_l, cycle=cycle)
    phase_r = _gait_phase(env, offset=offset_r, cycle=cycle)
    in_air_l = (phase_l < air_ratio).float()
    in_air_r = (phase_r < air_ratio).float()

    norm_speed = (foot_speeds / speed_threshold).clamp(0.0, 1.0)  # (N, F)
    air_reward = norm_speed
    stance_reward = 1.0 - norm_speed

    reward_l = in_air_l * air_reward[:, 0] + (1.0 - in_air_l) * stance_reward[:, 0]
    reward_r = in_air_r * air_reward[:, 1] + (1.0 - in_air_r) * stance_reward[:, 1]

    return (reward_l + reward_r) * 0.5


def gait_feet_frc_support_perio(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    cycle: float = 0.85,
    offset_l: float = 0.38,
    offset_r: float = 0.88,
    air_ratio: float = 0.38,
    threshold: float = 500.0,
) -> torch.Tensor:
    """Reward double-support: both feet on the ground during stance overlaps.

    The reward is active when **both** feet are in their stance phase
    (i.e. the double-support portion of the gait cycle).  It rewards high
    contact force on both feet simultaneously.

    Returns:
        Scalar reward per environment.
    """
    sensor = env.scene[sensor_cfg.name]
    forces = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3], dim=-1)

    phase_l = _gait_phase(env, offset=offset_l, cycle=cycle)
    phase_r = _gait_phase(env, offset=offset_r, cycle=cycle)
    in_stance_l = (phase_l >= air_ratio).float()
    in_stance_r = (phase_r >= air_ratio).float()
    double_support = in_stance_l * in_stance_r  # 1.0 only when both in stance

    norm_force_l = (forces[:, 0] / threshold).clamp(0.0, 1.0)
    norm_force_r = (forces[:, 1] / threshold).clamp(0.0, 1.0)

    return double_support * torch.min(norm_force_l, norm_force_r)