# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Motion data loader for AMP reference motions.

Supports multiple source formats:

1. **Pre-processed AMP features** (.npy / .pt / .pth)
   - numpy array or torch tensor of shape ``(N, obs_dim)``
   - dict with key ``"amp_obs"`` containing such an array

2. **AMASS-retargeted NPZ** (.npz) — e.g. AMASS_Retargeted_for_G1
   Keys: ``fps``, ``dof_names``, ``dof_positions``, ``dof_velocities``,
   ``body_positions``, ``body_rotations``, ``body_linear_velocities``,
   ``body_angular_velocities``.
   The loader extracts AMP features:
     joint_pos_rel | joint_vel | base_lin_vel_b | base_ang_vel_b | foot_pos_b

3. **LAFAN1 Retargeting Dataset CSV** (.csv) — 30 FPS, per-robot column layout
   Each row: ``[x, y, z, qx, qy, qz, qw,  joint_0 … joint_N-1]``
   Velocities are computed via finite difference.
   Foot positions are unavailable (set to zero) without running FK.

4. **Directory** — loads all supported files recursively and concatenates.

AMP observation layout (matches ``AMPCfg`` in amp_env_cfg.py):
  joint_pos_rel (num_dof) | joint_vel (num_dof) |
  base_lin_vel (3) | base_ang_vel (3) | foot_positions_base (num_foot_links * 3)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known robot profiles
# ---------------------------------------------------------------------------
# foot_body_indices: indices into body_names array (from .npz)
# default_joint_pos: per-joint default positions matching dof_names order
# csv_quat_format: "xyzw" (LAFAN1) or "wxyz"
# joint_reorder_map: MuJoCo/AMASS order → IsaacLab BFS order (for .npz files)
_ROBOT_PROFILES: dict[str, dict] = {
    "g1": {
        "num_dof": 29,
        # Substring patterns matched against ``body_names`` loaded from the NPZ
        # to resolve foot body indices at runtime.  Order = [left, right].
        "foot_body_names": ("left_ankle_roll_link", "right_ankle_roll_link"),
        # Fallback indices if NPZ has no body_names (e.g. old files).
        "foot_body_indices": [],
        # G1 29-DOF default standing posture (matches UNITREE_G1_29DOF_CFG BFS order)
        "default_joint_pos": np.array([
            -0.1,  # [0] left_hip_pitch_joint
            -0.1,  # [1] right_hip_pitch_joint
             0.0,  # [2] waist_yaw_joint
             0.0,  # [3] left_hip_roll_joint
             0.0,  # [4] right_hip_roll_joint
             0.0,  # [5] waist_roll_joint
             0.0,  # [6] left_hip_yaw_joint
             0.0,  # [7] right_hip_yaw_joint
             0.0,  # [8] waist_pitch_joint
             0.3,  # [9] left_knee_joint
             0.3,  # [10] right_knee_joint
             0.3,  # [11] left_shoulder_pitch_joint
             0.3,  # [12] right_shoulder_pitch_joint
            -0.2,  # [13] left_ankle_pitch_joint
            -0.2,  # [14] right_ankle_pitch_joint
             0.25, # [15] left_shoulder_roll_joint
            -0.25, # [16] right_shoulder_roll_joint
             0.0,  # [17] left_ankle_roll_joint
             0.0,  # [18] right_ankle_roll_joint
             0.0,  # [19] left_shoulder_yaw_joint
             0.0,  # [20] right_shoulder_yaw_joint
             0.97, # [21] left_elbow_joint
             0.97, # [22] right_elbow_joint
             0.15, # [23] left_wrist_roll_joint
            -0.15, # [24] right_wrist_roll_joint
             0.0,  # [25] left_wrist_pitch_joint
             0.0,  # [26] right_wrist_pitch_joint
             0.0,  # [27] left_wrist_yaw_joint
             0.0,  # [28] right_wrist_yaw_joint
        ], dtype=np.float32),
        "csv_quat_format": "xyzw",
        # MuJoCo/AMASS joint order → IsaacLab BFS order (gather map).
        # Given data in AMASS order, result[i] = data[gather_map[i]] produces BFS order.
        # AMASS order: left_hip_pitch(0), left_hip_roll(1), left_hip_yaw(2), left_knee(3), ...
        # IsaacLab BFS: left_hip_pitch(0), right_hip_pitch(6), waist_yaw(12), left_hip_roll(1), ...
        "joint_reorder_map": np.array([
            0, 6, 12, 1, 7, 13, 2, 8, 14,    # hips + waist (p/r/y interleaved)
            3, 9,                              # knees
            15, 22,                            # shoulder_pitch
            4, 10,                             # ankle_pitch
            16, 23,                            # shoulder_roll
            5, 11,                             # ankle_roll
            17, 24,                            # shoulder_yaw
            18, 25,                            # elbow
            19, 26,                            # wrist_roll
            20, 27,                            # wrist_pitch
            21, 28,                            # wrist_yaw
        ], dtype=np.int32),
    },
    "go2": {
        "num_dof": 12,
        "foot_body_indices": [3, 6, 9, 12],  # FL, FR, RL, RR foot
        "default_joint_pos": np.zeros(12, dtype=np.float32),
        "csv_quat_format": "wxyz",
        "joint_reorder_map": None,  # Go2 uses same order (or verify if needed)
    },
}


# ---------------------------------------------------------------------------
# Quaternion math (numpy, no IsaacLab dependency)
# Convention: [w, x, y, z]
# ---------------------------------------------------------------------------

def _quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by the *inverse* of quaternion ``q``.

    Args:
        q: ``(N, 4)`` quaternions in ``[w, x, y, z]`` order.
        v: ``(N, 3)`` vectors.

    Returns:
        ``(N, 3)`` rotated vectors.
    """
    q_w = q[:, 0:1]
    q_xyz = -q[:, 1:4]          # conjugate = inverse for unit quat
    uv = np.cross(q_xyz, v)     # (N, 3)
    uuv = np.cross(q_xyz, uv)   # (N, 3)
    return v + 2.0 * (q_w * uv + uuv)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q`` (forward rotation, not inverse).

    Used to convert body-frame angular velocity to world-frame.

    Args:
        q: ``(N, 4)`` quaternions in ``[w, x, y, z]`` order.
        v: ``(N, 3)`` vectors.

    Returns:
        ``(N, 3)`` rotated vectors.
    """
    q_w = q[:, 0:1]
    q_xyz = q[:, 1:4]            # not negated — forward rotation
    uv = np.cross(q_xyz, v)
    uuv = np.cross(q_xyz, uv)
    return v + 2.0 * (q_w * uv + uuv)


def _finite_diff(x: np.ndarray, fps: float) -> np.ndarray:
    """Central/forward finite difference along axis 0."""
    vel = np.empty_like(x)
    vel[:-1] = (x[1:] - x[:-1]) * fps
    vel[-1] = vel[-2]           # repeat last frame
    return vel


def _ang_vel_from_quats(q: np.ndarray, fps: float) -> np.ndarray:
    """Estimate angular velocity (world frame) from consecutive quaternions.

    Uses: ω ≈ 2 * q_conj ⊗ dq/dt  (imaginary part only)
    Args:
        q: ``(N, 4)`` quaternions ``[w, x, y, z]``.
    Returns:
        ``(N, 3)`` angular velocity in *local* frame.
    """
    dq = np.empty_like(q)
    dq[:-1] = (q[1:] - q[:-1]) * fps
    dq[-1] = dq[-2]
    # q_conj ⊗ dq: imaginary part → angular velocity in body frame
    q_w = q[:, 0:1]
    q_xyz = q[:, 1:4]
    dq_w = dq[:, 0:1]
    dq_xyz = dq[:, 1:4]
    # imaginary part of (q_conj ⊗ dq): 2*(q_w*dq_xyz - dq_w*q_xyz - q_xyz×dq_xyz)
    ang_vel = 2.0 * (q_w * dq_xyz - dq_w * q_xyz + np.cross(-q_xyz, dq_xyz))
    return ang_vel.astype(np.float32)


# ---------------------------------------------------------------------------
# MotionLoader
# ---------------------------------------------------------------------------

class MotionLoader:
    """Loads and stores reference motion data for AMP training.

    Parameters
    ----------
    device:
        Torch device for the output tensor.
    robot:
        Robot identifier for built-in profiles (``"g1"``, ``"go2"``).
        Used when loading raw motion files (.npz / .csv) that need conversion
        to AMP feature vectors.  Ignored for pre-processed ``.npy`` / ``.pt``
        files that already contain an ``"amp_obs"`` array.
    default_joint_pos:
        Override the default joint positions used for ``joint_pos_rel``
        computation.  Shape: ``(num_dof,)``.
    """

    def __init__(
        self,
        device: str = "cpu",
        robot: str = "g1",
        default_joint_pos: Optional[np.ndarray] = None,
    ) -> None:
        self.device = device
        self.robot = robot.lower()
        profile = _ROBOT_PROFILES.get(self.robot, {})
        self._profile = profile
        if default_joint_pos is not None:
            self._default_jpos = np.asarray(default_joint_pos, dtype=np.float32)
        else:
            self._default_jpos = profile.get("default_joint_pos", None)
        self.data: torch.Tensor | None = None
        # Raw state tensors for Reference State Initialization (RSI).
        # Set by _load_npz / _load_csv; None for pre-processed formats.
        self.state_root_pos_w: torch.Tensor | None = None     # (N, 3) world frame
        self.state_root_quat: torch.Tensor | None = None      # (N, 4) [w, x, y, z]
        self.state_root_lin_vel_w: torch.Tensor | None = None # (N, 3) world frame
        self.state_root_ang_vel_w: torch.Tensor | None = None # (N, 3) world frame
        self.state_joint_pos: torch.Tensor | None = None      # (N, num_dof) BFS order
        self.state_joint_vel: torch.Tensor | None = None      # (N, num_dof) BFS order

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str) -> torch.Tensor:
        """Load motion data from a file or directory.

        Dispatches to the appropriate sub-loader based on file extension.

        Args:
            path: Path to a motion file or directory.

        Returns:
            AMP observation tensor. Shape: ``(total_frames, obs_dim)``
        """
        if os.path.isdir(path):
            return self._load_directory(path)
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".npy":
            return self._load_npy(path)
        elif ext in (".pt", ".pth"):
            return self._load_pt(path)
        elif ext == ".npz":
            return self._load_npz(path)
        elif ext == ".csv":
            return self._load_csv(path)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}': {path}. "
                "Supported: .npy, .pt, .pth, .npz, .csv, or a directory."
            )

    # ------------------------------------------------------------------
    # Per-format loaders
    # ------------------------------------------------------------------

    def _load_npy(self, path: str) -> torch.Tensor:
        """Load pre-processed AMP features from a .npy file."""
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        if isinstance(data, dict):
            if "amp_obs" in data:
                data = data["amp_obs"]
            else:
                raise KeyError(f"Expected key 'amp_obs' in dict, got: {list(data.keys())}")
        tensor = torch.from_numpy(np.array(data, dtype=np.float32)).to(self.device)
        self.data = tensor
        # Pre-processed files have no raw state — RSI not available
        self.state_root_pos_w = None
        self.state_root_quat = None
        self.state_root_lin_vel_w = None
        self.state_root_ang_vel_w = None
        self.state_joint_pos = None
        self.state_joint_vel = None
        return tensor

    def _load_pt(self, path: str) -> torch.Tensor:
        """Load pre-processed AMP features from a .pt/.pth file."""
        data = torch.load(path, weights_only=True, map_location=self.device)
        if isinstance(data, dict):
            if "amp_obs" in data:
                data = data["amp_obs"]
            else:
                raise KeyError(f"Expected key 'amp_obs' in dict, got: {list(data.keys())}")
        tensor = data.float().to(self.device)
        self.data = tensor
        # Pre-processed files have no raw state — RSI not available
        self.state_root_pos_w = None
        self.state_root_quat = None
        self.state_root_lin_vel_w = None
        self.state_root_ang_vel_w = None
        self.state_joint_pos = None
        self.state_joint_vel = None
        return tensor

    def _load_npz(self, path: str) -> torch.Tensor:
        """Load a retargeted motion ``.npz`` and convert to flat AMP feature vectors.

        Two schemas are auto-detected from the key set:

        1. **AMASS-retargeted** (keys ``dof_positions``, ``dof_velocities``,
           ``body_positions``, ``body_rotations``, ``body_linear_velocities``,
           ``body_angular_velocities``).  DOFs are in MuJoCo/AMASS order and
           will be reordered to IsaacLab BFS via ``joint_reorder_map``.
           Quaternion convention: ``[w, x, y, z]``.
        2. **IsaacLab-replayed** (keys ``joint_pos``, ``joint_vel``,
           ``body_pos_w``, ``body_quat_w``, ``body_lin_vel_w``,
           ``body_ang_vel_w``, scalar ``fps``).  Produced by
           ``scripts/csv_to_npz.py`` which kinematically replays a CSV through
           IsaacSim — the velocities come from PhysX rather than finite
           difference, matching the distribution the policy sees at training
           time.  DOFs are already in BFS order (no reorder applied).

        AMP feature layout::

          joint_pos_rel(num_dof) | joint_vel(num_dof) |
          base_lin_vel_b(3) | base_ang_vel_b(3) | foot_pos_b(num_feet*3)
        """
        d = np.load(path, allow_pickle=True)
        keys = set(d.files)

        if {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w"}.issubset(keys):
            # IsaacLab-replayed schema — already in BFS order, PhysX velocities.
            dof_pos = d["joint_pos"].astype(np.float32)       # (N, J)
            dof_vel = d["joint_vel"].astype(np.float32)       # (N, J)
            body_pos = d["body_pos_w"].astype(np.float32)     # (N, B, 3)
            body_rot = d["body_quat_w"].astype(np.float32)    # (N, B, 4) [w,x,y,z]
            body_lin = d["body_lin_vel_w"].astype(np.float32) # (N, B, 3)
            body_ang = d["body_ang_vel_w"].astype(np.float32) # (N, B, 3)
            already_bfs = True

            # Resolve foot body indices from body_names (if present).
            # TienKung-style AMP features require foot positions in base frame,
            # not base linear/angular velocity, to avoid the PhysX-vs-mocap
            # distribution mismatch that saturates the discriminator.
            if "body_names" in keys:
                body_names = [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in d["body_names"]
                ]
                foot_patterns = self._profile.get("foot_body_names", ())
                foot_body_indices = []
                for pat in foot_patterns:
                    for i, name in enumerate(body_names):
                        if pat in name:  # substring match
                            foot_body_indices.append(i)
                            break
                # Runtime-resolved foot indices win over the hardcoded profile.
                self._runtime_foot_body_indices = foot_body_indices
                logger.info(
                    f"[MotionLoader] Resolved foot body indices from NPZ: "
                    f"{foot_body_indices} (from names {foot_patterns})"
                )
            else:
                self._runtime_foot_body_indices = None
        else:
            # AMASS schema
            dof_pos = d["dof_positions"].astype(np.float32)
            dof_vel = d["dof_velocities"].astype(np.float32)
            body_pos = d["body_positions"].astype(np.float32)
            body_rot = d["body_rotations"].astype(np.float32)
            body_lin = d["body_linear_velocities"].astype(np.float32)
            body_ang = d["body_angular_velocities"].astype(np.float32)
            already_bfs = False

        N = dof_pos.shape[0]

        # Reorder joints only when source is in MuJoCo/AMASS order.
        if not already_bfs:
            reorder_map = self._profile.get("joint_reorder_map", None)
            if reorder_map is not None:
                dof_pos = dof_pos[:, reorder_map]
                dof_vel = dof_vel[:, reorder_map]
                logger.debug(f"Reordered joints from MuJoCo to IsaacLab BFS order")

        # 1. joint_pos (RAW — not relative to default).
        #    legged_lab uses raw joint_pos in all obs groups (policy, critic,
        #    disc, disc_demo); the policy learns the absolute gait trajectory
        #    rather than a delta.  This also simplifies disc feature matching:
        #    env's ``joint_pos`` = motion data's ``dof_pos`` directly, no
        #    alignment with default_joint_pos needed.
        jpos = dof_pos

        # 2. joint_vel (finite-diff at motion fps, matches env at same fps)
        jvel = dof_vel

        # 3. base angular velocity in base frame.
        #    Feature set mirrors legged_lab's ``disc`` obs group:
        #      base_ang_vel(3) + joint_pos(29) + joint_vel(29) = 61 per frame.
        root_quat = body_rot[:, 0, :]  # (N, 4) [w,x,y,z]
        base_ang_vel = _quat_rotate_inverse_np(root_quat, body_ang[:, 0, :])

        feature_blocks = [base_ang_vel, jpos, jvel]
        amp_obs_np = np.concatenate(feature_blocks, axis=1)
        tensor = torch.from_numpy(amp_obs_np).to(self.device)
        self.data = tensor

        # Keep foot_body_indices resolved for RSI or auxiliary observations.
        root_pos = body_pos[:, 0, :]

        # Store raw state for RSI (Reference State Initialization)
        self.state_root_pos_w = torch.from_numpy(root_pos).to(self.device)
        self.state_root_quat = torch.from_numpy(body_rot[:, 0, :]).to(self.device)  # [w,x,y,z]
        self.state_root_lin_vel_w = torch.from_numpy(body_lin[:, 0, :]).to(self.device)
        self.state_root_ang_vel_w = torch.from_numpy(body_ang[:, 0, :]).to(self.device)
        self.state_joint_pos = torch.from_numpy(dof_pos).to(self.device)  # BFS order
        self.state_joint_vel = torch.from_numpy(dof_vel).to(self.device)  # BFS order

        return tensor

    def _load_csv(self, path: str) -> torch.Tensor:
        """Load LAFAN1-style CSV and convert to flat AMP feature vectors.

        Column layout (G1, 30 FPS):
          ``[x, y, z, qx, qy, qz, qw,  joint_0 … joint_N-1]``

        Velocities are computed via finite difference at the file's native FPS
        (default 30 Hz).

        .. warning::
            Foot positions require FK and are **not** available from CSV alone.
            They are set to **zero** in the output.  If foot position features
            are important for your discriminator, convert to .npz format first
            or use the AMASS dataset instead.
        """
        data_np = np.loadtxt(path, delimiter=",", dtype=np.float32)  # (N, 36+)
        fps = self._profile.get("csv_fps", 30.0)

        root_pos = data_np[:, :3]   # (N, 3)  x, y, z
        quat_csv = data_np[:, 3:7]  # (N, 4)  format depends on robot

        # Convert quaternion to [w, x, y, z]
        csv_fmt = self._profile.get("csv_quat_format", "xyzw")
        if csv_fmt == "xyzw":
            # input: qx, qy, qz, qw → output: qw, qx, qy, qz
            root_quat = np.concatenate([quat_csv[:, 3:4], quat_csv[:, :3]], axis=1)
        else:
            root_quat = quat_csv  # already [w,x,y,z]

        num_dof = self._profile.get("num_dof", data_np.shape[1] - 7)
        joint_pos = data_np[:, 7: 7 + num_dof]  # (N, J) in AMASS/DFS order

        # Reorder joints from AMASS/DFS order to IsaacLab BFS order
        reorder_map = self._profile.get("joint_reorder_map", None)
        if reorder_map is not None:
            joint_pos = joint_pos[:, reorder_map]
            logger.debug("CSV: Reordered joints from AMASS to IsaacLab BFS order")

        # joint_pos (RAW, matches env's joint_pos in base obs)
        jpos = joint_pos

        # velocities via finite difference at native CSV fps
        jvel = _finite_diff(joint_pos, fps)                          # (N, J)
        # Base angular velocity in body frame (from quat finite-diff)
        base_ang_vel = _ang_vel_from_quats(root_quat, fps)            # (N, 3)

        # Feature layout matches legged_lab disc obs:
        #   base_ang_vel(3) + joint_pos(num_dof) + joint_vel(num_dof)
        amp_obs_np = np.concatenate([base_ang_vel, jpos, jvel], axis=1)
        tensor = torch.from_numpy(amp_obs_np).to(self.device)
        self.data = tensor

        # Below we still store raw state vectors for RSI, but base velocities
        # are kept out of AMP features above — this only affects RSI init.
        root_lin_vel_w = _finite_diff(root_pos, fps)  # (N, 3) world frame
        base_ang_vel = _ang_vel_from_quats(root_quat, fps)  # body frame

        # base_ang_vel is body-frame → rotate to world-frame for write_root_velocity_to_sim
        ang_vel_w_np = _quat_rotate_np(root_quat, base_ang_vel)
        self.state_root_pos_w = torch.from_numpy(root_pos).to(self.device)
        self.state_root_quat = torch.from_numpy(root_quat).to(self.device)  # [w,x,y,z]
        self.state_root_lin_vel_w = torch.from_numpy(root_lin_vel_w).to(self.device)
        self.state_root_ang_vel_w = torch.from_numpy(ang_vel_w_np).to(self.device)
        self.state_joint_pos = torch.from_numpy(joint_pos).to(self.device)  # BFS order
        self.state_joint_vel = torch.from_numpy(jvel).to(self.device)

        return tensor

    def _load_directory(self, path: str) -> torch.Tensor:
        """Recursively load all supported motion files from a directory."""
        all_data: list[torch.Tensor] = []
        # Accumulate per-file state chunks for RSI concatenation
        state_chunks: dict[str, list[torch.Tensor]] = {
            "root_pos_w": [], "root_quat": [], "root_lin_vel_w": [],
            "root_ang_vel_w": [], "joint_pos": [], "joint_vel": [],
        }
        supported = (".npy", ".pt", ".pth", ".npz", ".csv")
        for root, _, files in os.walk(path):
            for filename in sorted(files):
                if not filename.lower().endswith(supported):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    chunk = self.load(filepath)
                    all_data.append(chunk)
                    # Collect state data if this file provided it
                    if self.state_root_pos_w is not None:
                        state_chunks["root_pos_w"].append(self.state_root_pos_w)
                        state_chunks["root_quat"].append(self.state_root_quat)
                        state_chunks["root_lin_vel_w"].append(self.state_root_lin_vel_w)
                        state_chunks["root_ang_vel_w"].append(self.state_root_ang_vel_w)
                        state_chunks["joint_pos"].append(self.state_joint_pos)
                        state_chunks["joint_vel"].append(self.state_joint_vel)
                except Exception as e:
                    logger.warning(f"[MotionLoader] Skipping {filepath}: {e}")

        if not all_data:
            raise FileNotFoundError(f"No supported motion files found under {path}")

        tensor = torch.cat(all_data, dim=0)
        self.data = tensor

        # Concatenate accumulated state tensors (or set to None if none available)
        if state_chunks["root_pos_w"]:
            self.state_root_pos_w = torch.cat(state_chunks["root_pos_w"], dim=0)
            self.state_root_quat = torch.cat(state_chunks["root_quat"], dim=0)
            self.state_root_lin_vel_w = torch.cat(state_chunks["root_lin_vel_w"], dim=0)
            self.state_root_ang_vel_w = torch.cat(state_chunks["root_ang_vel_w"], dim=0)
            self.state_joint_pos = torch.cat(state_chunks["joint_pos"], dim=0)
            self.state_joint_vel = torch.cat(state_chunks["joint_vel"], dim=0)
        else:
            self.state_root_pos_w = None
            self.state_root_quat = None
            self.state_root_lin_vel_w = None
            self.state_root_ang_vel_w = None
            self.state_joint_pos = None
            self.state_joint_vel = None

        return tensor

    def sample(self, batch_size: int, history_length: int = 2) -> torch.Tensor:
        """Sample random consecutive-frame clips from the loaded motion data.

        When ``history_length > 1`` the sampler draws pairs (or longer clips) of
        **consecutive** frames from the same motion clip and concatenates them
        along the feature dimension, so the output matches the AMP observation
        that the environment produces with ``ObservationGroupCfg.history_length``
        set to the same value (``flatten_history_dim=True`` is the IsaacLab
        default).

        Args:
            batch_size: Number of clips to sample.
            history_length: Number of consecutive frames per clip.  Must match
                the ``history_length`` set on the AMP observation group in the
                environment config (default 2).

        Returns:
            Sampled clips.
            Shape: ``(batch_size, history_length * obs_dim_per_frame)``
        """
        if self.data is None:
            raise RuntimeError("No motion data loaded. Call load() first.")
        num_frames = self.data.shape[0]
        if history_length <= 1:
            indices = torch.randint(0, num_frames, (batch_size,), device=self.device)
            return self.data[indices]
        # Sample start indices such that start + history_length - 1 < num_frames
        indices = torch.randint(0, num_frames - history_length + 1, (batch_size,), device=self.device)
        # Gather consecutive frames: shape (batch_size, history_length, obs_dim)
        offsets = torch.arange(history_length, device=self.device)  # (history_length,)
        frame_indices = indices.unsqueeze(1) + offsets.unsqueeze(0)  # (batch_size, history_length)
        clips = self.data[frame_indices]  # (batch_size, history_length, obs_dim)
        # Flatten history dim to match env output: (batch_size, history_length * obs_dim)
        return clips.reshape(batch_size, -1)

    @property
    def num_frames(self) -> int:
        """Number of frames in the loaded data."""
        return 0 if self.data is None else self.data.shape[0]

    @property
    def obs_dim(self) -> int:
        """Observation dimension of the loaded data."""
        return 0 if self.data is None else self.data.shape[1]

    @property
    def has_state_data(self) -> bool:
        """True if raw state tensors for RSI are available."""
        return self.state_root_pos_w is not None

    def get_random_state(self, n: int) -> dict[str, torch.Tensor]:
        """Sample ``n`` random frames and return their raw robot state.

        Returns a dict with keys:
          ``root_pos``       — (n, 3) root position in world frame
          ``root_quat``      — (n, 4) root quaternion [w, x, y, z]
          ``root_lin_vel_w`` — (n, 3) root linear velocity, world frame
          ``root_ang_vel_w`` — (n, 3) root angular velocity, world frame
          ``joint_pos``      — (n, num_dof) absolute joint positions (BFS order)
          ``joint_vel``      — (n, num_dof) joint velocities (BFS order)

        Args:
            n: Number of frames to sample.

        Raises:
            RuntimeError: If raw state data is not available.
        """
        if not self.has_state_data:
            raise RuntimeError(
                "MotionLoader has no raw state data for RSI. "
                "Load a .npz or .csv motion file (not pre-processed .npy/.pt)."
            )
        num_frames = self.data.shape[0]
        idx = torch.randint(0, num_frames, (n,), device=self.device)
        return {
            "root_pos":       self.state_root_pos_w[idx],
            "root_quat":      self.state_root_quat[idx],
            "root_lin_vel_w": self.state_root_lin_vel_w[idx],
            "root_ang_vel_w": self.state_root_ang_vel_w[idx],
            "joint_pos":      self.state_joint_pos[idx],
            "joint_vel":      self.state_joint_vel[idx],
        }
