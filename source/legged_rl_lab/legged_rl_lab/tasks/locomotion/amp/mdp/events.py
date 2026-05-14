# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP-specific event + termination functions — Reference State Initialization (RSI)
and reference-deviation early termination."""

from __future__ import annotations

import os
import warnings

import torch

from isaaclab.managers import SceneEntityCfg

from legged_rl_lab.managers import MotionLoader


# ---------------------------------------------------------------------------
# Lazy MotionLoader cache on env
# ---------------------------------------------------------------------------

def _get_loader(env) -> MotionLoader | None:
    """Lazy-init MotionLoader on env instance; cached at ``env._amp_rsi_loader``.

    Reads ``env.cfg.amp_motion_files`` and ``env.cfg.robot_type``.  On any
    failure the cache is set to ``None`` and a warning is emitted once.
    """
    if hasattr(env, "_amp_rsi_loader"):
        return env._amp_rsi_loader  # type: ignore[attr-defined]

    loader: MotionLoader | None = None
    motion_path: str = getattr(env.cfg, "amp_motion_files", "")
    robot_type: str = getattr(env.cfg, "robot_type", "g1")

    if motion_path and os.path.exists(motion_path):
        try:
            loader = MotionLoader(device=env.device, robot=robot_type)
            loader.load(motion_path)
            if not loader.has_state_data:
                warnings.warn(
                    "[RSI] MotionLoader loaded data but found no raw state tensors "
                    "(pre-processed .npy/.pt files do not carry state). "
                    "RSI + reference-deviation termination will be no-ops. "
                    "Use .npz or .csv source files.",
                    stacklevel=1,
                )
                loader = None
        except Exception as exc:
            warnings.warn(
                f"[RSI] Failed to load motion data from '{motion_path}': {exc}. "
                "RSI + reference-deviation termination will be no-ops.",
                stacklevel=1,
            )
            loader = None
    else:
        warnings.warn(
            f"[RSI] Motion file path not found or not set: '{motion_path}'. "
            "RSI + reference-deviation termination will be no-ops.",
            stacklevel=1,
        )

    env._amp_rsi_loader = loader  # type: ignore[attr-defined]
    return loader


def reset_from_reference_motion(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.1,
) -> None:
    """Reset robot state by sampling a random reference motion frame (RSI).

    Also stores the sampled frame index per env at ``env._ref_frame_start`` so
    that :func:`joint_deviation_from_reference` can phase-advance through the
    same reference clip.

    Adds a small ``height_offset`` (legged_lab uses 0.1m) so the robot doesn't
    spawn with its feet clipping into the ground due to mocap floor-height
    inconsistencies.  Without this, many RSI samples spawn below the kinematic
    ground plane and get catapulted on the first physics step → bad_orientation
    termination fires immediately.

    No-op if motion data is unavailable or carries no raw state.

    Args:
        env: The RL environment instance.
        env_ids: Indices of environments to reset.
        asset_cfg: Scene entity config for the robot articulation.
        height_offset: Extra Z added to the motion-supplied root height.
    """
    loader = _get_loader(env)
    if loader is None:
        return

    n = len(env_ids)
    if n == 0:
        return

    # Sample indices directly so we can record them on env for termination.
    num_frames = loader.num_frames
    idx = torch.randint(0, num_frames, (n,), device=env.device)

    # Persistent per-env buffer: remember which reference frame each env started
    # from, so the termination term can advance phase with episode_length_buf.
    if not hasattr(env, "_ref_frame_start"):
        env._ref_frame_start = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._ref_frame_start[env_ids] = idx

    asset = env.scene[asset_cfg.name]

    # Root position:
    #   - x, y: from env origin so each env stays in its own patch
    #   - z: reference motion height + env origin z + height_offset safety
    env_origins = env.scene.env_origins[env_ids]  # (n, 3)
    root_pos = env_origins.clone()
    root_pos[:, 2] = (
        env_origins[:, 2] + loader.state_root_pos_w[idx, 2] + height_offset
    )

    # Root orientation: directly from reference motion [w, x, y, z]
    root_quat = loader.state_root_quat[idx]  # (n, 4)
    root_pose = torch.cat([root_pos, root_quat], dim=-1)

    # Root velocity: world frame (n, 6) = [lin_vel(3), ang_vel(3)]
    root_vel = torch.cat(
        [loader.state_root_lin_vel_w[idx], loader.state_root_ang_vel_w[idx]], dim=-1
    )

    # Joint state (absolute positions, BFS order)
    joint_pos = loader.state_joint_pos[idx]  # (n, num_dof)
    joint_vel = loader.state_joint_vel[idx]  # (n, num_dof)

    # Write to simulation
    asset.write_root_pose_to_sim(root_pose, env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


# ---------------------------------------------------------------------------
# Termination: joint pose deviation from phase-advanced reference
# ---------------------------------------------------------------------------

def joint_deviation_from_reference(
    env,
    threshold: float = 3.5,
    motion_fps: float = 30.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate episodes whose joint pose drifts too far from the reference.

    For each env, the reference frame index at time ``t`` is::

        frame(t) = (start_frame + t * motion_fps / env_fps) mod num_frames

    where ``start_frame`` was recorded at RSI (``env._ref_frame_start``).
    The L2 norm of the per-joint pose error is compared to ``threshold``.

    Purpose: prevents AMP discriminator saturation by keeping the policy's
    trajectory close to the reference motion manifold.  Without this the
    policy finds a locally-stable but gait-wise alien behaviour (e.g. shuffling
    without proper stride) and the discriminator trivially separates it from
    the LAFAN1 walking data from iter-5 onward.

    Args:
        env: RL environment instance.
        threshold: L2 norm of joint pose error above which to terminate.
            Default 3.5 rad ≈ 0.65 rad (~37°) average per joint over 29 DOF.
        motion_fps: Frame rate of the reference motion data (LAFAN1 = 30).
        asset_cfg: Scene entity config for the robot articulation.

    Returns:
        Bool tensor of shape ``(num_envs,)``; ``True`` where the episode
        should terminate.
    """
    loader = getattr(env, "_amp_rsi_loader", None)
    if loader is None or loader.state_joint_pos is None:
        # Fail-safe: never terminate when reference data is unavailable.
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if not hasattr(env, "_ref_frame_start"):
        # RSI hasn't initialised indices yet (pre-first-reset).  No termination.
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    ref_joint = loader.state_joint_pos  # (N_frames, num_dof)
    num_frames = ref_joint.shape[0]

    # Phase advance: frame_offset = init + t * (motion_fps / env_fps)
    env_fps = 1.0 / env.step_dt
    rate = motion_fps / env_fps
    phase = (
        env._ref_frame_start.float()
        + env.episode_length_buf.float() * rate
    ).long() % num_frames

    target = ref_joint[phase]  # (num_envs, num_dof)
    current = env.scene[asset_cfg.name].data.joint_pos
    err = torch.linalg.vector_norm(current - target, dim=-1)
    return err > threshold
