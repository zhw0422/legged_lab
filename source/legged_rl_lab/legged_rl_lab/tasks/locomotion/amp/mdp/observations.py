# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP-specific observation functions.

These observations are used as input to the AMP discriminator.
Following IsaacLab's official G1AmpEnv (Design 3) — per-frame AMP obs is::

    joint_pos(num_dof) + joint_vel(num_dof) +
    root_height(1) + root_tan_norm(6) + root_lin_vel_b(3) + root_ang_vel_b(3) +
    key_body_pos_b(num_keys * 3)

For G1 with 29 DOF and 6 key bodies (feet + wrists + shoulders): 89 dims/frame.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Policy observations (yaw-removed for invariance to heading)
# ---------------------------------------------------------------------------

def root_local_rot_tan_norm(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Yaw-removed base rotation as 6D vector (tan + normal columns).

    Used in the *policy* and *critic* obs to give heading invariance.
    """
    robot = env.scene[asset_cfg.name]
    root_quat = robot.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)
    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)
    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    tan_vec = root_rotm_local[:, :, 0]
    norm_vec = root_rotm_local[:, :, 2]
    return torch.cat([tan_vec, norm_vec], dim=-1)


# ---------------------------------------------------------------------------
# AMP observation terms
# ---------------------------------------------------------------------------

def amp_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Raw joint positions (NOT relative to default)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def amp_joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to default (kept for backwards-compat)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos


def amp_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def amp_root_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root body z-coordinate (world frame).

    Captures gait rhythm — pelvis bobs up/down with each step.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]


def amp_root_tan_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root rotation as tangent + normal vectors (Design 3, no yaw removal).

    Matches IsaacLab G1AmpEnv exactly: take ref tangent (1,0,0) and normal
    (0,0,1) in body frame, rotate to world frame.  Captures full base
    orientation including yaw — important because expert mocap data has
    heading variations.
    """
    asset = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w  # (N, 4)
    ref_tan = torch.zeros_like(q[..., :3])
    ref_tan[..., 0] = 1.0
    ref_norm = torch.zeros_like(q[..., :3])
    ref_norm[..., -1] = 1.0
    tan = quat_apply(q, ref_tan)
    norm = quat_apply(q, ref_norm)
    return torch.cat([tan, norm], dim=-1)  # (N, 6)


def amp_base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base linear velocity in base frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def amp_base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base angular velocity in base frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def amp_key_body_pos_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Positions of key bodies relative to the root, in base frame.

    asset_cfg.body_names should specify the key bodies in a fixed order
    (e.g. left+right ankle/foot, wrist, shoulder).  Output shape:
    ``(num_envs, num_keys * 3)``.
    """
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
    body_pos_w = asset.data.body_pos_w[:, body_ids, :]               # (N, K, 3)
    rel_w = body_pos_w - asset.data.root_pos_w.unsqueeze(1)          # (N, K, 3)
    base_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(body_ids), -1)
    pos_b = quat_apply_inverse(base_quat, rel_w)                     # (N, K, 3)
    return pos_b.reshape(pos_b.shape[0], -1)


def amp_full_features(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """All AMP per-frame features concatenated in one term.

    Layout (must match motion_loader's per-frame layout exactly)::

        joint_pos(num_dof) | joint_vel(num_dof) | root_height(1) |
        root_tan_norm(6) | key_body_pos_b(num_keys * 3)

    For G1 with 29 DOF + 6 key bodies → 83 dims/frame.

    .. important::
       This is a *single* observation term so that when the AMP
       ObservationGroup applies ``history_length=2`` with
       ``flatten_history_dim=True``, the resulting flat layout is::

           [features_{t-1} (83) , features_t (83)]

       which matches motion_loader's expert sampling
       ``[features_t (83) , features_{t+1} (83)]``.

       If we used five separate terms, IsaacLab would buffer each term's
       history *independently* and the flat output becomes
       ``[jpos_{t-1}, jpos_t, jvel_{t-1}, jvel_t, ...]`` — a feature-major
       layout that does NOT match the expert's frame-major layout.  The
       discriminator then trivially distinguishes the two and saturates
       at 100% accuracy, providing no useful style gradient.
    """
    jpos = amp_joint_pos(env)                                # (N, num_dof)
    jvel = amp_joint_vel(env)                                # (N, num_dof)
    rh = amp_root_height(env)                                # (N, 1)
    rtn = amp_root_tan_norm(env)                             # (N, 6)
    kbp = amp_key_body_pos_b(env, asset_cfg=asset_cfg)       # (N, num_keys * 3)
    return torch.cat([jpos, jvel, rh, rtn, kbp], dim=-1)


# ---------------------------------------------------------------------------
# Legacy: foot positions only (kept for backwards-compat, not used now)
# ---------------------------------------------------------------------------

def amp_foot_positions_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
) -> torch.Tensor:
    """Foot positions in base frame.  Subset of :func:`amp_key_body_pos_b`."""
    return amp_key_body_pos_b(env, asset_cfg)


# ---------------------------------------------------------------------------
# Gait phase clock (TienKung-Lab inspired)
# ---------------------------------------------------------------------------

def gait_phase_obs(
    env: ManagerBasedRLEnv,
    gait_cycle: float = 0.85,
    phase_offset_l: float = 0.0,
    phase_offset_r: float = 0.5,
    air_ratio_l: float = 0.38,
    air_ratio_r: float = 0.38,
) -> torch.Tensor:
    """Periodic gait clock: sin/cos of left/right phase + duty ratios.

    Returns 6 dims: ``sin(2π·φ_L), cos(2π·φ_L), sin(2π·φ_R), cos(2π·φ_R),
    air_ratio_L, air_ratio_R``.

    Following TienKung-Lab — gives policy an explicit step-timing prior so
    it doesn't have to learn the gait cycle from scratch.  AMP shapes
    *style*; the clock shapes *cadence*.
    """
    # episode_length_buf is in env-step counts.  step_dt seconds per step.
    step_dt = env.cfg.sim.dt * env.cfg.decimation
    t = env.episode_length_buf.float() * step_dt / gait_cycle  # (num_envs,)
    phase_l = (t + phase_offset_l) % 1.0
    phase_r = (t + phase_offset_r) % 1.0
    two_pi = 2.0 * torch.pi
    sin_l = torch.sin(two_pi * phase_l).unsqueeze(-1)
    cos_l = torch.cos(two_pi * phase_l).unsqueeze(-1)
    sin_r = torch.sin(two_pi * phase_r).unsqueeze(-1)
    cos_r = torch.cos(two_pi * phase_r).unsqueeze(-1)
    air_l = torch.full_like(sin_l, air_ratio_l)
    air_r = torch.full_like(sin_r, air_ratio_r)
    return torch.cat([sin_l, cos_l, sin_r, cos_r, air_l, air_r], dim=-1)
