# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP-specific observation functions.

These observations are used as input to the AMP discriminator.
The discriminator needs motion-related features (without command/gravity info)
to distinguish between reference and policy-generated motions.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import quat_apply_inverse


def root_local_rot_tan_norm(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Base rotation (yaw removed) as tan + normal vectors from rotation matrix.

    Richer than ``projected_gravity`` — encodes full base pose (roll + pitch
    combined) as a 6D vector instead of a 3D gravity projection.  Follows
    legged_lab's observation design (see
    ``legged_lab/tasks/locomotion/amp/mdp/observations.py``).
    """
    robot = env.scene[asset_cfg.name]
    root_quat = robot.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)
    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)
    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    # First and last column = tangent (forward) and normal (up) axes in the
    # yaw-removed local frame.
    tan_vec = root_rotm_local[:, :, 0]   # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    return torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)


def amp_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Raw joint positions for AMP observations (NOT relative to default).

    legged_lab / TienKung-Lab both feed raw joint_pos into the discriminator.
    The policy also learns on raw joint_pos for consistency.  Removes the
    default_joint_pos offset that would otherwise need to be matched between
    env and motion data.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos


def amp_joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to default for AMP observations.

    Returns joint positions offset by their default values.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos


def amp_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities for AMP observations."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def amp_base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base linear velocity in base frame for AMP observations."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def amp_base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base angular velocity in base frame for AMP observations."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def amp_foot_positions_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
) -> torch.Tensor:
    """Computes foot positions relative to the base frame for AMP."""
    asset = env.scene[asset_cfg.name]

    # Get foot positions in world frame: (N, num_feet, 3)
    body_ids, _ = asset.find_bodies(asset_cfg.body_names)
    foot_pos_w = asset.data.body_pos_w[:, body_ids, :]

    # Compute relative offset in world frame: (N, num_feet, 3)
    rel_pos_w = foot_pos_w - asset.data.root_pos_w.unsqueeze(1)

    # Transform to base frame using broadcasting: (N, 1, 4) rotates (N, num_feet, 3)
    # Note: quat_apply_inverse flattens tensors internally, so we must expand the quat first.
    base_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(body_ids), -1)
    foot_pos_b = quat_apply_inverse(base_quat_w, rel_pos_w)

    # Flatten output to (N, num_feet * 3)
    return foot_pos_b.reshape(foot_pos_b.shape[0], -1)


