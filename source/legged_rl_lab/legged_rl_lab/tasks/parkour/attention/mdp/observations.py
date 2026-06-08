"""Observation terms for parkour attention tasks."""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def elevation_map(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    noise: bool = False,
    height_clip: tuple[float, float] = (-1.2, 0.0),
    height_noise_std: float = 0.03,
    height_offset_range: tuple[float, float] = (-0.05, 0.05),
) -> torch.Tensor:
    """Return ray-hit coordinates in the sensor/yaw frame as a flattened AME map."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w
    num_envs, num_rays, _ = relative_pos_w.shape

    if getattr(sensor.cfg, "ray_alignment", "base") == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat = sensor_quat.unsqueeze(1).expand(num_envs, num_rays, 4)
    sensor_quat = sensor_quat.reshape(num_envs * num_rays, 4)
    sensor_coords = quat_apply_inverse(
        sensor_quat.to(torch.float),
        relative_pos_w.reshape(num_envs * num_rays, 3),
    )
    sensor_coords = sensor_coords.reshape(num_envs, num_rays, 3)
    sensor_coords = torch.nan_to_num(sensor_coords)

    if noise:
        needs_offset = (
            getattr(env, "_attention_map_height_offset", None) is None
            or env._attention_map_height_offset.shape != (num_envs, 1)
        )
        if needs_offset:
            env._attention_map_height_offset = torch.zeros((num_envs, 1), device=env.device)

        if hasattr(env, "reset_buf"):
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if reset_env_ids.numel() > 0:
                lo, hi = height_offset_range
                env._attention_map_height_offset[reset_env_ids] = (
                    torch.rand((reset_env_ids.numel(), 1), device=env.device) * (hi - lo) + lo
                )

        sensor_coords[..., 2] += torch.randn_like(sensor_coords[..., 2]) * height_noise_std
    sensor_coords[..., 2] += env._attention_map_height_offset

    sensor_coords[..., 2] = sensor_coords[..., 2].clamp(min=height_clip[0], max=height_clip[1])
    return sensor_coords.reshape(num_envs, num_rays * 3)


def height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    clip: tuple[float, float] | None = None,
) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    if clip is not None:
        height = height.clamp(min=clip[0], max=clip[1])
    return height


def gait_phase_sin_cos(
    env: ManagerBasedRLEnv,
    period: float = 0.8,
    offset: list[float] | tuple[float, ...] = (0.0, 0.5),
) -> torch.Tensor:
    time = env.episode_length_buf.float() * env.step_dt
    global_phase = time % period / period
    phases = torch.stack([(global_phase + phase_offset) % 1.0 for phase_offset in offset], dim=-1)
    sin_phase = torch.sin(2 * math.pi * phases)
    cos_phase = torch.cos(2 * math.pi * phases)
    return torch.cat([sin_phase, cos_phase], dim=-1)


def feet_pos_body_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    rel_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    parts = [quat_apply_inverse(asset.data.root_quat_w, rel_pos_w[:, index]) for index in range(rel_pos_w.shape[1])]
    return torch.cat(parts, dim=-1)


def feet_vel_body_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    rel_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w.unsqueeze(1)
    parts = [quat_apply_inverse(asset.data.root_quat_w, rel_vel_w[:, index]) for index in range(rel_vel_w.shape[1])]
    return torch.cat(parts, dim=-1)


def feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return forces.reshape(forces.shape[0], -1)


def scalar_rigid_friction_mean(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties().to(env.device)
    static_friction = materials[:, :, 0].mean(dim=1)
    dynamic_friction = materials[:, :, 1].mean(dim=1)
    return ((static_friction + dynamic_friction) * 0.5).unsqueeze(-1)


def body_mass_scale(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    mass = asset.root_physx_view.get_masses().to(env.device)
    default_mass = asset.data.default_mass.to(env.device)
    ratio = mass[:, asset_cfg.body_ids] / (default_mass[:, asset_cfg.body_ids] + 1e-8)
    return ratio.mean(dim=-1, keepdim=True)


def last_push_delta_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "_attention_push_xy"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env._attention_push_xy.to(device=env.device)


def joint_stiffness_scale(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*"]),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    stiffness = asset.data.joint_stiffness
    default_stiffness = asset.data.default_joint_stiffness
    if stiffness is None or default_stiffness is None:
        return torch.ones(env.num_envs, asset.num_joints, device=env.device)
    return stiffness[:, asset_cfg.joint_ids] / (default_stiffness[:, asset_cfg.joint_ids] + 1e-8)


def joint_damping_scale(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*"]),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    damping = asset.data.joint_damping
    default_damping = asset.data.default_joint_damping
    if damping is None or default_damping is None:
        return torch.ones(env.num_envs, asset.num_joints, device=env.device)
    return damping[:, asset_cfg.joint_ids] / (default_damping[:, asset_cfg.joint_ids] + 1e-8)


def links_contact_binary(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return is_contact.float()


def height_relative_to_feet(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: tuple[float, float] | None = (-1.0, 1.0),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    parts = []
    for index, name in enumerate(sensor_names):
        sensor: RayCaster = env.scene.sensors[name]
        rel = foot_z[:, index : index + 1] - sensor.data.ray_hits_w[..., 2]
        if clip is not None:
            rel = rel.clamp(min=clip[0], max=clip[1])
        parts.append(rel)
    return torch.cat(parts, dim=-1)


def normal_vector_around_feet(env: ManagerBasedRLEnv, sensor_names: list[str]) -> torch.Tensor:
    parts = []
    for name in sensor_names:
        sensor: RayCaster = env.scene.sensors[name]
        hits = sensor.data.ray_hits_w
        point_0 = hits[:, 0]
        point_1 = hits[:, 2]
        point_2 = hits[:, 6]
        valid = torch.isfinite(point_0).all(dim=-1)
        valid &= torch.isfinite(point_1).all(dim=-1)
        valid &= torch.isfinite(point_2).all(dim=-1)

        normal = torch.cross(point_1 - point_0, point_2 - point_0, dim=-1)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True).clamp_min(1e-6)
        default = torch.zeros_like(normal)
        default[..., 2] = 1.0
        parts.append(torch.where(valid.unsqueeze(-1), normal, default))
    return torch.cat(parts, dim=-1)


def body_com_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    com_b = asset.data.body_com_pos_b[:, asset_cfg.body_ids, :]
    return com_b.reshape(com_b.shape[0], -1)
