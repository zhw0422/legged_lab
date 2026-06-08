"""Reward terms for parkour attention tasks."""

from __future__ import annotations

import math

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat


def alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float)


def fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return (torch.sum(is_contact, dim=-1) < 0.5).float()


def body_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_gravity = quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :],
        asset.data.GRAVITY_VEC_W,
    )
    return torch.sum(torch.square(body_gravity[:, :2]), dim=1)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, ratio: float = 4.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    lateral_t = torch.norm(history[..., :2], dim=-1)
    vertical_t = torch.abs(history[..., 2])
    lateral_max = torch.max(lateral_t, dim=1)[0]
    vertical_max = torch.max(vertical_t, dim=1)[0]
    return torch.any(lateral_max > ratio * vertical_max, dim=1).float()


def feet_too_near(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.2,
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2, "feet_too_near requires exactly 2 body_ids"
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def joint_deviation_l1_always(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def leg_ref_joint_pos(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    period: float = 0.8,
    scales: tuple[float, ...] = (-0.2, 0.4, -0.2),
    double_support_threshold: float = 0.1,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    asset: Articulation = env.scene[left_cfg.name]
    t = env.episode_length_buf.float() * env.step_dt
    sin_pos = torch.sin(2 * math.pi * t / period)

    left_pos = asset.data.joint_pos[:, left_cfg.joint_ids]
    left_default = asset.data.default_joint_pos[:, left_cfg.joint_ids]
    right_pos = asset.data.joint_pos[:, right_cfg.joint_ids]
    right_default = asset.data.default_joint_pos[:, right_cfg.joint_ids]

    num_per_leg = left_pos.shape[1]
    scales_t = torch.tensor(list(scales[:num_per_leg]), device=left_pos.device, dtype=left_pos.dtype)
    swing_l = (-sin_pos).clamp(min=0)
    swing_r = sin_pos.clamp(min=0)
    ds_mask = torch.abs(sin_pos) < double_support_threshold
    swing_l = swing_l.masked_fill(ds_mask, 0.0)
    swing_r = swing_r.masked_fill(ds_mask, 0.0)

    left_ref = left_default + swing_l.unsqueeze(1) * scales_t.unsqueeze(0)
    right_ref = right_default + swing_r.unsqueeze(1) * scales_t.unsqueeze(0)
    diff = torch.cat([left_pos - left_ref, right_pos - right_ref], dim=-1)
    diff_norm = torch.norm(diff, dim=-1)
    reward = torch.exp(-2.0 * diff_norm) - 0.2 * diff_norm.clamp(0, 0.5)
    cmd = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.linalg.norm(cmd, dim=-1) > cmd_threshold).float()
    return reward * moving


class action_smoothness_l2(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_prev_action = torch.zeros(
            env.num_envs,
            env.action_manager.total_action_dim,
            device=env.device,
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action = env.action_manager.action
        prev_action = env.action_manager.prev_action
        smoothness = torch.sum(torch.square(action - 2.0 * prev_action + self._prev_prev_action), dim=1)
        self._prev_prev_action = prev_action.clone()
        return smoothness


def track_lin_vel_xy_yaw_frame_heading_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    y_error_weight: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_term = env.command_manager.get_term(command_name)

    if hasattr(cmd_term, "heading_target"):
        heading_err = wrap_to_pi(cmd_term.heading_target - asset.data.heading_w)
        heading_coef = (1.0 + torch.cos(heading_err)) * 0.5
    else:
        heading_coef = torch.ones(env.num_envs, device=env.device)

    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    err_x = torch.square(cmd[:, 0] - vel_yaw[:, 0])
    err_y = y_error_weight * torch.square(cmd[:, 1] - vel_yaw[:, 1])
    return torch.exp(-(err_x + err_y) / std**2) * heading_coef


def foot_clearance_target(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.08,
    foot_offset: float = 0.022,
    sigma: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    feet_vel_norm = torch.linalg.norm(feet_vel_xy, dim=-1)
    terrain_z_max = sensor.data.ray_hits_w[..., 2].max(dim=-1).values
    height_above = feet_z - terrain_z_max.unsqueeze(-1) - target_height - foot_offset
    err = torch.sum(feet_vel_norm * torch.square(height_above), dim=-1)
    return torch.exp(-err / sigma)


def feet_contact_stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    cmd_threshold: float = 0.2,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    feet_max_force_z = torch.max(torch.abs(history[..., 2]), dim=1)[0]
    in_contact = feet_max_force_z > force_threshold
    all_in_contact = torch.all(in_contact, dim=-1)
    cmd = env.command_manager.get_command(command_name)[:, :3]
    standing = torch.linalg.norm(cmd, dim=-1) < cmd_threshold
    return (standing & all_in_contact).float()


def hip_pos_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(err), dim=-1)


def dof_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    omega = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(tau * omega), dim=-1)
