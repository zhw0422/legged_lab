# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP task reward functions."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command_gate(
    env: "ManagerBasedRLEnv",
    command_name: str | None,
    command_threshold: float,
) -> torch.Tensor:
    """Return per-env mask: 1 when command is above threshold, 0 otherwise."""
    if command_name is None:
        return torch.ones(env.num_envs, device=env.device)
    cmd = env.command_manager.get_command(command_name)
    return (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2]) > command_threshold).float()


def track_anchor_linear_velocity(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 0.0,
) -> torch.Tensor:
    """Track commanded xy velocity at an anchor body."""
    del mask_delay, delay_env_rew_ratio
    asset = env.scene[anchor_cfg.name]
    body_ids, _ = asset.find_bodies(anchor_cfg.body_names, preserve_order=True)
    anchor_id = body_ids[0]
    lin_vel_w = asset.data.body_lin_vel_w[:, anchor_id, :]
    anchor_quat_w = asset.data.body_quat_w[:, anchor_id, :]
    lin_vel_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(anchor_quat_w), lin_vel_w)
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - lin_vel_yaw[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_anchor_angular_velocity(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 0.0,
) -> torch.Tensor:
    """Track commanded yaw velocity at an anchor body."""
    del mask_delay, delay_env_rew_ratio
    asset = env.scene[anchor_cfg.name]
    body_ids, _ = asset.find_bodies(anchor_cfg.body_names, preserve_order=True)
    anchor_id = body_ids[0]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.body_ang_vel_w[:, anchor_id, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def track_root_height(
    env: "ManagerBasedRLEnv",
    std: float,
    target_height: float = 0.78,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 0.0,
) -> torch.Tensor:
    """Track a nominal root height."""
    del mask_delay, delay_env_rew_ratio
    asset = env.scene[asset_cfg.name]
    height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error / std**2)


def body_ang_vel_xy_l2(
    env: "ManagerBasedRLEnv",
    std: float,
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 0.0,
) -> torch.Tensor:
    """Reward small roll/pitch angular velocity for selected bodies."""
    del mask_delay, delay_env_rew_ratio
    asset = env.scene[body_cfg.name]
    body_ids, _ = asset.find_bodies(body_cfg.body_names, preserve_order=True)
    ang_vel_error = torch.sum(torch.square(asset.data.body_ang_vel_w[:, body_ids, :2]), dim=(1, 2))
    return torch.exp(-ang_vel_error / std**2)


def body_upright_orientation(
    env: "ManagerBasedRLEnv",
    std: float,
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 0.0,
) -> torch.Tensor:
    """Reward selected bodies for staying upright in world frame."""
    del mask_delay, delay_env_rew_ratio
    asset = env.scene[body_cfg.name]
    body_ids, _ = asset.find_bodies(body_cfg.body_names, preserve_order=True)

    body_quat_w = asset.data.body_quat_w[:, body_ids, :]
    gravity_w = asset.data.GRAVITY_VEC_W[:, None, :].expand(-1, len(body_ids), -1)
    gravity_b = math_utils.quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        gravity_w.reshape(-1, 3),
    ).reshape(env.num_envs, len(body_ids), 3)

    tilt_error = torch.sum(torch.square(gravity_b[..., :2]), dim=(1, 2))
    return torch.exp(-tilt_error / std**2)


def body_pair_distance_violation(
    env: "ManagerBasedRLEnv",
    source_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    min_distance: float,
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 0.0,
) -> torch.Tensor:
    """Penalize selected body pairs when their origins get too close."""
    del mask_delay, delay_env_rew_ratio
    asset = env.scene[source_cfg.name]
    source_ids, _ = asset.find_bodies(source_cfg.body_names, preserve_order=True)
    target_ids, _ = asset.find_bodies(target_cfg.body_names, preserve_order=True)

    source_pos = asset.data.body_pos_w[:, source_ids, :]
    target_pos = asset.data.body_pos_w[:, target_ids, :]
    pair_dist = torch.linalg.norm(source_pos[:, :, None, :] - target_pos[:, None, :, :], dim=-1)
    violation = torch.clamp(min_distance - pair_dist, min=0.0)
    return torch.sum(torch.square(violation), dim=(1, 2))


def feet_slide(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty for foot horizontal speed while in contact.

    ``Σ_feet  ‖v_xy‖ * 1[is_in_contact]``.  This complements the periodic
    spd reward by directly killing any horizontal velocity the moment a foot
    is detected to be on the ground (regardless of phase).
    """
    sensor = env.scene[sensor_cfg.name]
    in_contact = (
        torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3], dim=-1)
        > contact_force_threshold
    )

    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
    foot_xy_speed = torch.norm(asset.data.body_lin_vel_w[:, body_ids, :2], dim=-1)

    return torch.sum(foot_xy_speed * in_contact.float(), dim=1)


def feet_slip(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty for foot slip while a velocity command is active."""
    slip = feet_slide(env, sensor_cfg, asset_cfg, contact_force_threshold)
    return slip * _command_gate(env, command_name, command_threshold)
