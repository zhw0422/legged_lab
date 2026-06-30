#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Sim2Sim for Motion Tracking (MuJoCo)

Replays a trained tracking policy that follows reference motion trajectories.
The ONNX model embeds the full motion clip; at each step it takes (obs, time_step)
and returns (actions, ref_joint_pos, ref_joint_vel, ref_body_pos_w, ref_body_quat_w,
ref_body_lin_vel_w, ref_body_ang_vel_w).

Usage:
    python deploy/g1_deploy/g1_python/sim2sim_mimic.py --model policy.onnx --config g1_mimic.yaml
"""

import time
import argparse
import os
import sys
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
G1_DEPLOY_DIR = SCRIPT_DIR.parent
DEPLOY_DIR = G1_DEPLOY_DIR.parent
sys.path.insert(0, str(DEPLOY_DIR / "utils"))
from joystick import create_gamepad_controller
from sim2sim_walk import KeyboardController


# ==================== Math Helpers ====================

def quat_rotate_inverse_np(q_wxyz, v):
    """Rotate vector v from world frame to body frame. q is (w, x, y, z)."""
    r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])  # scipy uses xyzw
    return r.apply(v, inverse=True).astype(np.float32)


def quat_mul_np(q1, q2):
    """Hamilton quaternion multiplication (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def quat_inv_np(q):
    """Inverse of unit quaternion (w, x, y, z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_apply_np(q_wxyz, v):
    """Apply quaternion rotation to vector. q is (w, x, y, z)."""
    r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return r.apply(v).astype(np.float32)


def subtract_frame_transforms_np(pos_a, quat_a_wxyz, pos_b, quat_b_wxyz):
    """Compute relative transform of frame B in frame A's local frame.
    Returns (pos_b_in_a, quat_b_in_a) both in (w,x,y,z) convention.
    """
    q_inv = quat_inv_np(quat_a_wxyz)
    rel_pos = quat_apply_np(q_inv, pos_b - pos_a)
    rel_quat = quat_mul_np(q_inv, quat_b_wxyz)
    return rel_pos, rel_quat


def matrix_from_quat_np(q_wxyz):
    """Convert (w,x,y,z) quaternion to 3x3 rotation matrix."""
    r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return r.as_matrix().astype(np.float32)


# ==================== Config Parser ====================

STRING_LIST_KEYS = {'joint_names_mujoco', 'actuator_names_mujoco', 'sdk_joint_order', 'body_names'}


def parse_config(raw_cfg):
    """Parse YAML dict into SimpleNamespace with auto numpy conversion."""
    cfg = SimpleNamespace(**raw_cfg)
    for k, v in raw_cfg.items():
        if isinstance(v, list) and k not in STRING_LIST_KEYS:
            try:
                v = np.array(v, dtype=np.int32 if 'map' in k else np.float32)
            except (ValueError, TypeError):
                pass
        setattr(cfg, k, v)
    return cfg


# ==================== Observation Builder ====================

def build_tracking_obs(
    ref_joint_pos, ref_joint_vel,
    motion_anchor_pos_b, motion_anchor_ori_b_6,
    base_lin_vel_b, base_ang_vel_b,
    joint_pos_rel, joint_vel,
    last_action,
    include_state_estimation=True,
):
    """
    Build observation vector matching TrackingEnvCfg.ObservationsCfg.PolicyCfg:
      command          = [ref_joint_pos(29), ref_joint_vel(29)]  -> 58
      anchor_pos_b     = 3  (absent in Wo-State-Estimation variant)
      anchor_ori_b     = 6
      base_lin_vel     = 3  (absent in Wo-State-Estimation variant)
      base_ang_vel     = 3
      joint_pos_rel    = 29
      joint_vel        = 29
      last_action      = 29
    Total: 160 (with SE) or 154 (without SE)
    """
    parts = [ref_joint_pos, ref_joint_vel]
    if include_state_estimation:
        parts.append(motion_anchor_pos_b)
    parts.append(motion_anchor_ori_b_6)
    if include_state_estimation:
        parts.append(base_lin_vel_b)
    parts.extend([base_ang_vel_b, joint_pos_rel, joint_vel, last_action])
    return np.concatenate(parts).astype(np.float32)


def build_walk_obs(base_ang_vel_w, projected_gravity, commands, joint_pos_rel, joint_vel, last_action, cfg):
    """
    Build 96-dim walk/flat observation matching IsaacLab flat velocity policy:
      base_ang_vel  (world frame, 3)  * ang_vel_scale
      projected_gravity (body frame, 3)
      commands [vx, vy, vyaw]  (3)
      joint_pos_rel  (29) * dof_pos_scale
      joint_vel      (29) * dof_vel_scale
      last_action    (29)
    Total: 96
    """
    return np.concatenate([
        base_ang_vel_w * cfg.ang_vel_scale,
        projected_gravity,
        commands,
        joint_pos_rel * cfg.dof_pos_scale,
        joint_vel * cfg.dof_vel_scale,
        last_action,
    ]).astype(np.float32)


# ==================== Sim2Sim Controller ====================

class MimicSim2SimController:
    """MuJoCo sim2sim controller for ONNX motion-tracking policies."""

    def __init__(self, config_path, model_name, debug_policy=False, debug_interval=1.0):
        self._base_dir = str(G1_DEPLOY_DIR)
        self._active_policy_idx = 0
        self.debug_policy = debug_policy
        self.debug_interval = max(float(debug_interval), 1.0e-6)
        self._last_debug_print_time = -np.inf

        # ---- Load config ----
        with open(config_path, 'r') as f:
            raw_cfg = yaml.safe_load(f)
        self.config = parse_config(raw_cfg)
        cfg = self.config

        # ---- MuJoCo model ----
        xml_filename = os.path.basename(cfg.xml_path)
        self.xml_path = os.path.join(self._base_dir, "assets", xml_filename)
        print(f"Loading MuJoCo model: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.policy_decimation = int(cfg.control_dt / cfg.sim_dt)
        self.sim_dt = cfg.sim_dt

        # ---- Joint / actuator mapping (MuJoCo order) ----
        self.joint_qpos_addrs = [self.model.jnt_qposadr[self.model.joint(n).id] for n in cfg.joint_names_mujoco]
        self.joint_qvel_addrs = [self.model.jnt_dofadr[self.model.joint(n).id] for n in cfg.joint_names_mujoco]
        self.actuator_ids = [self.model.actuator(n).id for n in cfg.actuator_names_mujoco]
        self.num_joints = len(cfg.joint_names_mujoco)

        # ---- Anchor body ID in MuJoCo (tracking mode only) ----
        self.anchor_body_id = (
            self.model.body(cfg.anchor_body_name).id
            if hasattr(cfg, 'anchor_body_name') else None
        )

        # ---- Load ONNX policy ----
        self.policy_path = os.path.join(self._base_dir, "exported_policy", model_name)
        print(f"Loading ONNX policy: {self.policy_path}")
        self._load_onnx(self.policy_path)

        # ---- PD gains (Isaac order -> MuJoCo order) ----
        self.kp = cfg.kps[cfg.isaac_to_mujoco_map]
        self.kd = cfg.kds[cfg.isaac_to_mujoco_map]

        # ---- Default pose (Isaac & MuJoCo order) ----
        self.default_qpos_mj = cfg.default_joint_pos[cfg.isaac_to_mujoco_map]
        self.target_qpos_mj = self.default_qpos_mj.copy()

        # ---- Buffers ----
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self.time_step = 0

        # ---- Policy type (flat=walk stabilization / tracking=mimic) ----
        self._policy_type = getattr(cfg, 'policy_type', 'tracking')
        history_len = getattr(cfg, 'history_length', 1)
        self._obs_history = deque(
            [np.zeros(cfg.num_obs, dtype=np.float32)] * history_len,
            maxlen=history_len,
        )

        # ---- Set initial MuJoCo pose ----
        self.data.qpos[self.joint_qpos_addrs] = self.default_qpos_mj
        self.data.qpos[2] = getattr(cfg, 'init_height', 0.76)
        mujoco.mj_step(self.model, self.data)

        print(f"MimicSim2Sim initialized (num_obs={cfg.num_obs}, "
              f"state_estimation={getattr(cfg, 'include_state_estimation', True)})")

    def _load_onnx(self, path):
        self.ort_session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.ort_input_names = [inp.name for inp in self.ort_session.get_inputs()]
        self.ort_output_names = [out.name for out in self.ort_session.get_outputs()]

    # -------- Policy hot-swap --------

    def load_policy(self, config_path, model_name, policy_idx):
        """Hot-swap config + ONNX policy without restarting MuJoCo."""
        print(f"[PolicySwitch] Loading policy {policy_idx}: {config_path} / {model_name}")
        with open(config_path, 'r') as f:
            raw_cfg = yaml.safe_load(f)
        new_cfg = parse_config(raw_cfg)

        policy_path = os.path.join(self._base_dir, "exported_policy", model_name)
        if not os.path.exists(policy_path):
            print(f"[PolicySwitch] Policy not found: {policy_path}, skipping.")
            return False

        self.config = new_cfg
        cfg = new_cfg
        self._load_onnx(policy_path)
        self.policy_decimation = int(cfg.control_dt / cfg.sim_dt)
        self.kp = cfg.kps[cfg.isaac_to_mujoco_map]
        self.kd = cfg.kds[cfg.isaac_to_mujoco_map]
        self.default_qpos_mj = cfg.default_joint_pos[cfg.isaac_to_mujoco_map]
        self.target_qpos_mj = self.default_qpos_mj.copy()
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self.time_step = 0
        self._policy_type = getattr(cfg, 'policy_type', 'tracking')
        history_len = getattr(cfg, 'history_length', 1)
        self._obs_history = deque(
            [np.zeros(cfg.num_obs, dtype=np.float32)] * history_len,
            maxlen=history_len,
        )
        if hasattr(cfg, 'anchor_body_name'):
            self.anchor_body_id = self.model.body(cfg.anchor_body_name).id
        self._active_policy_idx = policy_idx
        print(f"[PolicySwitch] Switched to policy {policy_idx} (type={self._policy_type})")
        return True

    # -------- PD control --------

    def pd_controller(self, target_pos_mj, target_vel_mj):
        tau = self.kp * (target_pos_mj - self.data.qpos[self.joint_qpos_addrs]) \
            + self.kd * (target_vel_mj - self.data.qvel[self.joint_qvel_addrs])
        self.data.ctrl[self.actuator_ids] = tau

    def run_onnx_flat(self, obs):
        """Run flat walk ONNX policy (single obs input -> actions)."""
        obs_batch = obs[np.newaxis, :].astype(np.float32)
        outputs = self.ort_session.run(
            self.ort_output_names,
            {self.ort_input_names[0]: obs_batch},
        )
        return outputs[0].flatten().astype(np.float32)

    # -------- ONNX inference --------

    def run_onnx(self, obs):
        """Run ONNX inference.
        Returns: (actions, ref_joint_pos, ref_joint_vel, ref_body_pos_w,
                  ref_body_quat_w, ref_body_lin_vel_w, ref_body_ang_vel_w)
        """
        obs_batch = obs[np.newaxis, :].astype(np.float32)
        ts_batch = np.array([[self.time_step]], dtype=np.float32)
        feed = {"obs": obs_batch, "time_step": ts_batch}
        outputs = self.ort_session.run(self.ort_output_names, feed)
        return [o.squeeze(0) for o in outputs]

    # -------- Anchor body state from MuJoCo --------

    def _get_anchor_state(self):
        """Get anchor body position (3,) and quat (w,x,y,z) from MuJoCo."""
        pos = self.data.xpos[self.anchor_body_id].copy().astype(np.float32)
        quat_wxyz = self.data.xquat[self.anchor_body_id].copy().astype(np.float32)
        return pos, quat_wxyz

    def _joint_names_policy_order(self, cfg):
        return [cfg.joint_names_mujoco[int(i)] for i in cfg.mujoco_to_isaac_map]

    def _print_policy_debug(
        self,
        sim_time,
        obs_input,
        action,
        target_isaac,
        curr_qpos_isaac,
        curr_qvel_isaac,
        joint_pos_rel,
        base_lin_vel_w,
        base_ang_vel_w,
        extra=None,
    ):
        cfg = self.config
        names = self._joint_names_policy_order(cfg)
        obs_input = np.asarray(obs_input, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        target_isaac = np.asarray(target_isaac, dtype=np.float32).reshape(-1)
        target_err = target_isaac - curr_qpos_isaac

        print("\n" + "=" * 120, flush=True)
        print(
            f"[MimicDebug/PY] sim_t={sim_time:.3f}s policy={self._policy_type} "
            f"time_step={self.time_step} obs_dim={obs_input.size} expected={cfg.num_obs}",
            flush=True,
        )
        print(
            f"[root] xyz={np.array2string(self.data.qpos[:3], precision=4)} "
            f"quat_wxyz={np.array2string(self.data.qpos[3:7], precision=4)} "
            f"lin_w={np.array2string(base_lin_vel_w, precision=4)} "
            f"ang={np.array2string(base_ang_vel_w, precision=4)}",
            flush=True,
        )
        print(
            f"[obs] min={obs_input.min():+.5f} max={obs_input.max():+.5f} "
            f"mean={obs_input.mean():+.5f} l2={np.linalg.norm(obs_input):.5f} "
            f"nan={np.isnan(obs_input).any()} inf={np.isinf(obs_input).any()}",
            flush=True,
        )
        if extra:
            for key, value in extra.items():
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                print(
                    f"[obs:{key}] shape={arr.shape} min={arr.min():+.5f} max={arr.max():+.5f} "
                    f"mean={arr.mean():+.5f} l2={np.linalg.norm(arr):.5f}",
                    flush=True,
                )
        print(
            f"[action] min={action.min():+.5f} max={action.max():+.5f} "
            f"mean={action.mean():+.5f} l2={np.linalg.norm(action):.5f}",
            flush=True,
        )
        print(
            f"[qrel] min={joint_pos_rel.min():+.5f} max={joint_pos_rel.max():+.5f} "
            f"mean={joint_pos_rel.mean():+.5f} l2={np.linalg.norm(joint_pos_rel):.5f}",
            flush=True,
        )
        print("[Joint table in policy order]", flush=True)
        print(" idx name                              q_abs    q_rel       qd   action   target      err", flush=True)
        for i, name in enumerate(names):
            print(
                f"{i:>3d} {name:<30s} "
                f"{curr_qpos_isaac[i]:+7.3f} {joint_pos_rel[i]:+8.3f} {curr_qvel_isaac[i]:+8.3f} "
                f"{action[i]:+8.3f} {target_isaac[i]:+8.3f} {target_err[i]:+8.3f}",
                flush=True,
            )
        print("=" * 120 + "\n", flush=True)

    # -------- Main loop --------

    def run(self, gamepad, policy_registry=None):
        motiontime = 0
        cfg = self.config
        if policy_registry is None:
            policy_registry = {}

        include_se = getattr(cfg, 'include_state_estimation', True)
        self.model.opt.timestep = self.sim_dt

        # Initialize reference data for tracking mode
        ref_joint_pos = ref_joint_vel = None
        ref_body_pos_w = ref_body_quat_w = None
        anchor_body_idx = None

        def _init_tracking_refs(cfg):
            nonlocal ref_joint_pos, ref_joint_vel, ref_body_pos_w, ref_body_quat_w, anchor_body_idx
            dummy_obs = np.zeros(cfg.num_obs, dtype=np.float32)
            onnx_out = self.run_onnx(dummy_obs)
            ref_joint_pos = onnx_out[1].astype(np.float32)
            ref_joint_vel = onnx_out[2].astype(np.float32)
            ref_body_pos_w = onnx_out[3].astype(np.float32)
            ref_body_quat_w = onnx_out[4].astype(np.float32)
            anchor_body_idx = cfg.body_names.index(cfg.anchor_body_name)

        if self._policy_type == 'tracking':
            _init_tracking_refs(cfg)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.lookat[:] = self.data.qpos[:3]
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20

            start_time = time.time()

            while viewer.is_running():
                if gamepad.exit_requested:
                    print("\nExit requested, ending simulation...")
                    break

                # ---- Policy switch ----
                requested = gamepad.active_policy
                if requested != 0 and requested != self._active_policy_idx:
                    if requested in policy_registry:
                        cfg_path, mdl_name = policy_registry[requested]
                        print()  # newline to keep progress bar intact
                        if self.load_policy(cfg_path, mdl_name, requested):
                            cfg = self.config
                            include_se = getattr(cfg, 'include_state_estimation', True)
                            if self._policy_type == 'tracking':
                                _init_tracking_refs(cfg)
                            print(f"[PolicySwitch] Active policy: {requested} ({self._policy_type})")
                    else:
                        print(f"\n[PolicySwitch] Policy {requested} not registered.")

                # ---- Apply perturbation ----
                self.data.xfrc_applied[:] = 0
                mujoco.mjv_applyPerturbForce(self.model, self.data, viewer.perturb)

                # ---- PD control ----
                self.pd_controller(self.target_qpos_mj, np.zeros_like(self.kd))
                mujoco.mj_step(self.model, self.data)
                motiontime += 1

                # ---- Policy step (at control frequency) ----
                if motiontime % self.policy_decimation == 0:
                    # Current robot state (MuJoCo -> Isaac order)
                    curr_qpos_mj = self.data.qpos[self.joint_qpos_addrs]
                    curr_qvel_mj = self.data.qvel[self.joint_qvel_addrs]
                    curr_qpos_isaac = curr_qpos_mj[cfg.mujoco_to_isaac_map]
                    curr_qvel_isaac = curr_qvel_mj[cfg.mujoco_to_isaac_map]

                    joint_pos_rel = (curr_qpos_isaac - cfg.default_joint_pos).astype(np.float32)
                    joint_vel = curr_qvel_isaac.astype(np.float32)

                    base_quat_wxyz = self.data.qpos[3:7].copy().astype(np.float32)
                    base_lin_vel_w = self.data.qvel[0:3].copy().astype(np.float32)
                    base_ang_vel_w = self.data.qvel[3:6].copy().astype(np.float32)

                    if self._policy_type == 'flat':
                        # ---- Flat walk policy (stabilization) ----
                        quat_xyzw = base_quat_wxyz[[1, 2, 3, 0]]
                        proj_grav = R.from_quat(quat_xyzw).apply([0, 0, -1], inverse=True).astype(np.float32)
                        commands = np.zeros(3, dtype=np.float32)
                        obs = build_walk_obs(
                            base_ang_vel_w, proj_grav, commands,
                            joint_pos_rel, joint_vel, self.last_action, cfg,
                        )
                        self._obs_history.append(obs)
                        obs_arr = np.array(list(self._obs_history))
                        n = self.num_joints
                        feature_indices = [
                            (0, 3), (3, 6), (6, 9),
                            (9, 9 + n), (9 + n, 9 + 2 * n), (9 + 2 * n, 9 + 3 * n),
                        ]
                        obs_input = np.concatenate([
                            obs_arr[:, s:e].ravel() for s, e in feature_indices
                        ])
                        action = self.run_onnx_flat(obs_input)
                        debug_extra = {
                            "flat_frame": obs,
                        }

                    else:
                        # ---- Tracking policy ----
                        base_lin_vel_b = quat_rotate_inverse_np(base_quat_wxyz, base_lin_vel_w)
                        # MuJoCo qvel[3:6] is angular velocity already in body frame;
                        # do NOT apply quat_rotate_inverse again.
                        base_ang_vel_b = base_ang_vel_w

                        # Anchor transforms
                        robot_anchor_pos, robot_anchor_quat = self._get_anchor_state()
                        motion_anchor_pos = ref_body_pos_w[anchor_body_idx].astype(np.float32)
                        motion_anchor_quat = ref_body_quat_w[anchor_body_idx].astype(np.float32)

                        anchor_pos_b, anchor_quat_b = subtract_frame_transforms_np(
                            robot_anchor_pos, robot_anchor_quat,
                            motion_anchor_pos, motion_anchor_quat,
                        )
                        # Extract first 2 columns of rotation matrix -> 6 values (R6)
                        # Must match Isaac Lab: mat[..., :2].reshape(N, -1) which is row-major
                        # i.e. [m00, m01, m10, m11, m20, m21]
                        anchor_mat = matrix_from_quat_np(anchor_quat_b)
                        anchor_ori_b_6 = anchor_mat[:, :2].reshape(-1)  # row-major: [m00,m01,m10,m11,m20,m21]

                        # Build observation
                        obs = build_tracking_obs(
                            ref_joint_pos, ref_joint_vel,
                            anchor_pos_b, anchor_ori_b_6,
                            base_lin_vel_b, base_ang_vel_b,
                            joint_pos_rel, joint_vel,
                            self.last_action,
                            include_state_estimation=include_se,
                        )
                        obs_input = obs
                        debug_extra = {
                            "ref_joint_pos": ref_joint_pos,
                            "ref_joint_vel": ref_joint_vel,
                            "anchor_pos_b": anchor_pos_b,
                            "anchor_ori_b_6": anchor_ori_b_6,
                            "base_lin_vel_b": base_lin_vel_b,
                            "base_ang_vel_b": base_ang_vel_b,
                        }

                        # Run ONNX policy
                        onnx_out = self.run_onnx(obs)
                        action = onnx_out[0].flatten().astype(np.float32)

                        # Update reference data for next step
                        ref_joint_pos = onnx_out[1].astype(np.float32)
                        ref_joint_vel = onnx_out[2].astype(np.float32)
                        ref_body_pos_w = onnx_out[3].astype(np.float32)
                        ref_body_quat_w = onnx_out[4].astype(np.float32)

                        self.time_step += 1

                        # Loop motion clip if finished
                        total_steps = getattr(cfg, 'motion_total_steps', None)
                        if total_steps and self.time_step >= total_steps:
                            print("\n[Motion] Clip finished, looping from start.")
                            self.time_step = 0

                    # ---- Shared: update last_action and target pose ----
                    self.last_action = action.copy()

                    # Compute target joint positions (Isaac order -> MuJoCo order)
                    target_isaac = action * cfg.action_scale + cfg.default_joint_pos
                    self.target_qpos_mj = target_isaac[cfg.isaac_to_mujoco_map]

                    policy_sim_t = motiontime * self.sim_dt
                    if self.debug_policy and policy_sim_t - self._last_debug_print_time >= self.debug_interval:
                        self._print_policy_debug(
                            sim_time=policy_sim_t,
                            obs_input=obs_input,
                            action=action,
                            target_isaac=target_isaac,
                            curr_qpos_isaac=curr_qpos_isaac,
                            curr_qvel_isaac=curr_qvel_isaac,
                            joint_pos_rel=joint_pos_rel,
                            base_lin_vel_w=base_lin_vel_w,
                            base_ang_vel_w=base_ang_vel_w,
                            extra=debug_extra,
                        )
                        self._last_debug_print_time = policy_sim_t

                # ---- Camera follow ----
                viewer.cam.lookat[:] = self.data.qpos[:3]
                viewer.sync()

                # ---- Real-time sync ----
                expected = start_time + (motiontime * self.sim_dt)
                sleep_time = expected - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # ---- Telemetry (every 1s) ----
                if motiontime % int(1.0 / self.sim_dt) == 0:
                    real_t = time.time() - start_time
                    hz = motiontime / real_t if real_t > 0 else 0
                    sim_t = motiontime * self.sim_dt
                    total = getattr(cfg, 'motion_total_steps', None)
                    if self._policy_type == 'tracking' and total:
                        bar_width = 40
                        filled = int(bar_width * self.time_step / total)
                        bar = '█' * filled + '░' * (bar_width - filled)
                        pct = self.time_step / total * 100
                        print(f"\r[Motion] [{bar}] {pct:5.1f}%  step={self.time_step}/{total} | "
                              f"t={sim_t:.1f}s | height={self.data.qpos[2]:.3f}m | {hz:.0f} Hz",
                              end='', flush=True)
                    elif self._policy_type == 'tracking':
                        print(f"\r[Sim] t={sim_t:.1f}s | step={self.time_step} | "
                              f"height={self.data.qpos[2]:.3f}m | {hz:.0f} Hz | mode=tracking",
                              end='', flush=True)
                    else:
                        print(f"\r[Sim] t={sim_t:.1f}s | height={self.data.qpos[2]:.3f}m | "
                              f"{hz:.0f} Hz | mode=flat-stabilize",
                              end='', flush=True)


# ==================== Main ====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sim2Sim for G1 Motion Tracking")
    parser.add_argument('--model', type=str, default='g1_dance.onnx', help='ONNX model filename')
    parser.add_argument('--config', type=str, default='g1_mimic.yaml', help='Config YAML filename')
    parser.add_argument('--input', choices=['gamepad', 'keyboard'], default='gamepad',
                        help='Control input device.')
    parser.add_argument('--debug_policy', action='store_true', help='Print policy obs/action/joint debug info.')
    parser.add_argument('--debug_interval', type=float, default=1.0, help='Seconds between --debug_policy prints.')
    args = parser.parse_args()

    script_dir = str(G1_DEPLOY_DIR)

    def resolve_config(name):
        if os.path.exists(name):
            return name
        path = Path(name)
        if path.parts and path.parts[0] == "config":
            return str(G1_DEPLOY_DIR / path)
        return os.path.join(script_dir, 'config', name)

    mimic_config = resolve_config(args.config)
    flat_config = resolve_config('g1_walk.yaml')

    # ---- Policy registry (keyboard: 1/2/3/4, gamepad: A/B/X/Y) ----
    # Slot 1 (startup): flat walk policy for standing stabilization
    # Slot 2: main tracking/mimic policy
    policy_registry = {
        1: (flat_config, 'g1_flat_1.onnx'),                            # 1 / A: flat stabilize
        2: (mimic_config, args.model),                                 # 2 / B: main mimic policy
        3: (resolve_config('g1_mimic.yaml'), 'g1_jump.onnx'),           # 3 / X: jump
        4: (resolve_config('g1_mimic.yaml'), 'g1_dance.onnx'),          # 4 / Y: dance
    }

    # 1. Initialize controller with flat policy (stable standing first)
    controller = MimicSim2SimController(
        flat_config,
        'g1_flat_1.onnx',
        debug_policy=args.debug_policy,
        debug_interval=args.debug_interval,
    )
    controller._active_policy_idx = 1

    # 2. Initialize input device
    cfg = controller.config
    if args.input == 'keyboard':
        gamepad = KeyboardController(
            vx_range=[-1.0, 1.0],
            vy_range=[-0.5, 0.5],
            vyaw_range=[-1.0, 1.0],
        )
    else:
        gamepad = create_gamepad_controller(
            getattr(cfg, 'gamepad_type_sim2sim', 'gamesir'),
            vx_range=[-1.0, 1.0],
            vy_range=[-0.5, 0.5],
            vyaw_range=[-1.0, 1.0],
            policy_switch_mode='face_buttons',
            exit_button='select',
        )
    gamepad.start()
    gamepad.active_policy = 1

    print("\n" + "=" * 70)
    print("  Motion Tracking Sim2Sim (MuJoCo)")
    print("  Starts with flat walk policy (g1_flat_1.onnx) for stabilization")
    if args.input == 'keyboard':
        print("  1 : Flat walk policy (stand / stabilize)")
        print("  2 : Main mimic / tracking policy")
        print("  3 : g1_jump.onnx")
        print("  4 : g1_dance.onnx")
        print("  X or Esc : Exit")
    else:
        print("  A      : Flat walk policy (stand / stabilize)")
        print("  B      : Main mimic / tracking policy")
        print("  X      : g1_jump.onnx")
        print("  Y      : g1_dance.onnx")
        print("  Select : Exit")
    print("=" * 70 + "\n")

    # 3. Run
    controller.run(gamepad, policy_registry)
    gamepad.stop()
    print("\nProgram ended.")
