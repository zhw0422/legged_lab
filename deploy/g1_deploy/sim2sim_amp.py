#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Sim2Sim (MuJoCo)

Shared configuration parameters and policy inference logic.

Usage:
    Simulation: python deploy/g1_deploy/sim2sim_amp.py 
"""

import time
import numpy as np
import argparse
import onnxruntime as ort
import sys
import os
import select
import termios
import threading
import tty
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from joystick import create_gamepad_controller
import mujoco
import mujoco.viewer
import yaml
from pathlib import Path
from types import SimpleNamespace

LEGGED_RL_LAB_ROOT_DIR = str(Path(__file__).resolve().parent.parent)

# ========== Compute Projected Gravity ==========

def quat_rotate_inverse(q, v):
    """Rotate a vector from world frame to body frame using the inverse quaternion."""
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def compute_projected_gravity(quat):
    """Compute the projected gravity vector in the body frame."""
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    projected_gravity = quat_rotate_inverse(quat, gravity_world)
    return projected_gravity


def quat_wxyz_to_euler_xyz(q):
    """Convert MuJoCo root quaternion [w, x, y, z] to roll/pitch/yaw."""
    w, x, y, z = q / np.linalg.norm(q)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def rotate_world_to_yaw_frame(yaw, vec):
    """Rotate a world-frame vector into the root yaw frame."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        c * vec[0] + s * vec[1],
        -s * vec[0] + c * vec[1],
        vec[2],
    ], dtype=np.float32)

def build_obs(base_ang_vel, projected_gravity, commands, dof_pos_rel, dof_vel, last_action, config):
    """
    Build one AMP policy observation frame for the current G1 AMP policy.

    Layout:
      base_ang_vel(3), projected_gravity(3), commands(3),
      joint_pos_rel(29), joint_vel(29), last_action(29)
    Total: 96 dims/frame. With history_length=4, the ONNX input is 384 dims.
    """
    obs = []
    
    # 1-3: Base angular velocity (scaled)
    base_ang_vel_scaled = base_ang_vel * config.ang_vel_scale
    obs.extend(list(base_ang_vel_scaled))
    
    # 7-9: Projected gravity
    obs.extend(list(projected_gravity))
    
    # 10-12: Commands [vx, vy, vyaw]
    obs.extend(list(commands))
    
    # 13-24: joint position relative to default
    dof_pos_rel_scaled = dof_pos_rel * config.dof_pos_scale
    obs.extend(list(dof_pos_rel_scaled))
    
    # 25-36: joint velocities (scaled)
    dof_vel_scaled = dof_vel * config.dof_vel_scale
    obs.extend(list(dof_vel_scaled))
    
    # 37-48: Last action
    obs.extend(list(last_action))
    
    return np.array(obs, dtype=np.float32)


class KeyboardController:
    """Terminal keyboard controller with the same interface as GamepadController."""

    def __init__(
        self,
        vx_range=(-1.0, 1.0),
        vy_range=(-0.5, 0.5),
        vyaw_range=(-1.0, 1.0),
        vx_step=0.1,
        vy_step=0.05,
        vyaw_step=0.1,
    ):
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vyaw_range = vyaw_range
        self.vx_step = vx_step
        self.vy_step = vy_step
        self.vyaw_step = vyaw_step
        self.active_policy = 1
        self.exit_requested = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self._old_termios = None
        self._raw_enabled = False

    def get_velocity(self):
        with self.lock:
            return self.vx, self.vy, self.vyaw

    def _clip(self):
        self.vx = float(np.clip(self.vx, self.vx_range[0], self.vx_range[1]))
        self.vy = float(np.clip(self.vy, self.vy_range[0], self.vy_range[1]))
        self.vyaw = float(np.clip(self.vyaw, self.vyaw_range[0], self.vyaw_range[1]))

    def _apply_key(self, key):
        with self.lock:
            if key in ("w", "\x1b[A"):
                self.vx += self.vx_step
            elif key in ("s", "\x1b[B"):
                self.vx -= self.vx_step
            elif key == "a":
                self.vy += self.vy_step
            elif key == "d":
                self.vy -= self.vy_step
            elif key in ("q", "\x1b[D"):
                self.vyaw += self.vyaw_step
            elif key in ("e", "\x1b[C"):
                self.vyaw -= self.vyaw_step
            elif key in (" ", "0"):
                self.vx = 0.0
                self.vy = 0.0
                self.vyaw = 0.0
            elif key in ("1", "2", "3", "4"):
                self.active_policy = int(key)
            elif key in ("x", "\x1b"):
                self.exit_requested = True
                self.running = False
            self._clip()

    def _read_key(self):
        ch = sys.stdin.read(1)
        if ch == "\x1b" and select.select([sys.stdin], [], [], 0.001)[0]:
            ch += sys.stdin.read(1)
            if select.select([sys.stdin], [], [], 0.001)[0]:
                ch += sys.stdin.read(1)
        return ch

    def _control_thread(self):
        while self.running and not self.exit_requested:
            try:
                if select.select([sys.stdin], [], [], 0.02)[0]:
                    key = self._read_key()
                    self._apply_key(key.lower() if len(key) == 1 else key)
            except (OSError, ValueError):
                time.sleep(0.02)

    def start(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Keyboard input requires an interactive terminal.")
        self._old_termios = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._raw_enabled = True
        self.running = True
        self.thread = threading.Thread(target=self._control_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self._raw_enabled and self._old_termios is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            self._raw_enabled = False


# ========== Sim2Sim (MuJoCo) controller ==========

class Sim2SimController:
    """MuJoCo simulation controller."""
    
    def __init__(
        self,
        config_path,
        model_name,
        debug_policy=False,
        debug_interval=1.0,
        debug_joints="all",
    ):
        self._base_dir = _base_dir = os.path.dirname(os.path.abspath(__file__))
        self._active_policy_idx = 0  # 0=idle, 1=AMP, 2/3/4=other policies
        self.debug_policy = debug_policy
        self.debug_interval = max(float(debug_interval), 1.0e-6)
        self.debug_joints = debug_joints
        self._last_debug_print_time = -np.inf
        # 1. 直接加载并解析配置
        self.config = self._load_config(config_path)

        # 3. 基础变量提取
        c = self.config
        xml_filename = os.path.basename(c.xml_path) 
        self.xml_path = os.path.join(_base_dir, "assets", xml_filename)
        self.policy_path = os.path.join(_base_dir, "exported_policy", model_name)

        # 4. 加载 MuJoCo 模型
        print(f"Loading MuJoCo model: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.policy_decimation = int(round(c.control_dt / c.sim_dt))
        self.sim_dt = c.sim_dt
        
        # 5. 映射关节与执行器索引 (MuJoCo 顺序)
        # 使用列表推导式精简获取 ID
        self.joint_qpos_addrs = [self.model.jnt_qposadr[self.model.joint(n).id] for n in c.joint_names_mujoco]
        self.joint_qvel_addrs  = [self.model.jnt_dofadr[self.model.joint(n).id] for n in c.joint_names_mujoco]
        self.actuator_ids     = [self.model.actuator(n).id for n in c.actuator_names_mujoco]
        self.num_joints       = len(c.joint_names_mujoco)

        # 6. 加载 Policy
        print(f"Loading policy: {self.policy_path}")
        self._load_onnx(self.policy_path)
        self._validate_dimensions(c)
        
        # 7. 初始化缓冲区与增益 (直接从 config 读取)
        self.obs_history = self._make_obs_history(c)
        # kps/kds are stored in policy order; pd_controller works in MuJoCo order
        self.kp = c.kps[c.isaac_to_mujoco_map]
        self.kd = c.kds[c.isaac_to_mujoco_map]
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self._last_tau_mj = np.zeros(self.num_joints, dtype=np.float32)
        self._last_tau_isaac = np.zeros(self.num_joints, dtype=np.float32)

        # 8. 初始姿态对齐 (policy order -> MuJoCo order)
        # 使用配置里的 map 数组进行重排
        self.default_qpos_mj = c.default_joint_pos[c.isaac_to_mujoco_map]
        self.target_qpos_mj = self.default_qpos_mj.copy()

        self.data.qpos[self.joint_qpos_addrs] = self.default_qpos_mj
        self.data.qpos[2] = getattr(c, 'init_height', 0.90)  # 优先从配置读高度
        mujoco.mj_step(self.model, self.data)

        print("Sim2Sim controller initialized")

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            raw_cfg = yaml.safe_load(f)

        cfg = SimpleNamespace(**raw_cfg)
        string_list_keys = {'joint_names_mujoco', 'actuator_names_mujoco', 'sdk_joint_order'}
        for k, v in raw_cfg.items():
            if isinstance(v, list) and k not in string_list_keys:
                try:
                    v = np.array(v, dtype=np.int32 if 'map' in k or 'idx' in k else np.float32)
                except (ValueError, TypeError):
                    pass
            setattr(cfg, k, v)
        return cfg

    def _load_onnx(self, path):
        self.ort_session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.ort_input_names = [inp.name for inp in self.ort_session.get_inputs()]
        self.ort_output_names = [out.name for out in self.ort_session.get_outputs()]
        input_shape = self.ort_session.get_inputs()[0].shape
        output_shape = self.ort_session.get_outputs()[0].shape
        self.policy_input_dim = int(input_shape[-1]) if isinstance(input_shape[-1], int) else None
        self.policy_output_dim = int(output_shape[-1]) if isinstance(output_shape[-1], int) else None

    def _frame_obs_dim(self, config):
        return 3 + 3 + 3 + 3 * self.num_joints

    def _make_obs_history(self, config):
        return deque(
            [np.zeros(self._frame_obs_dim(config), dtype=np.float32)] * int(config.history_length),
            maxlen=int(config.history_length),
        )

    def _validate_dimensions(self, config):
        frame_dim = self._frame_obs_dim(config)
        expected_obs_dim = frame_dim * int(config.history_length)

        checks = {
            "kps": len(config.kps),
            "kds": len(config.kds),
            "default_joint_pos": len(config.default_joint_pos),
            "mujoco_to_isaac_map": len(config.mujoco_to_isaac_map),
            "isaac_to_mujoco_map": len(config.isaac_to_mujoco_map),
        }
        for name, size in checks.items():
            if size != self.num_joints:
                raise ValueError(f"{name} has {size} entries, expected {self.num_joints}.")

        if int(config.num_actions) != self.num_joints:
            raise ValueError(f"num_actions={config.num_actions}, expected {self.num_joints}.")
        if int(config.num_obs) != expected_obs_dim:
            raise ValueError(f"num_obs={config.num_obs}, expected {expected_obs_dim}.")

        action_scale = np.asarray(config.action_scale, dtype=np.float32)
        if action_scale.ndim == 0:
            config.action_scale = np.full(self.num_joints, float(action_scale), dtype=np.float32)
        elif action_scale.shape == (self.num_joints,):
            config.action_scale = action_scale
        else:
            raise ValueError(f"action_scale has shape {action_scale.shape}, expected scalar or ({self.num_joints},).")

        if self.policy_input_dim is not None and self.policy_input_dim != expected_obs_dim:
            raise ValueError(f"ONNX input dim={self.policy_input_dim}, expected {expected_obs_dim}.")
        if self.policy_output_dim is not None and self.policy_output_dim != self.num_joints:
            raise ValueError(f"ONNX output dim={self.policy_output_dim}, expected {self.num_joints}.")

        if not np.array_equal(config.mujoco_to_isaac_map[config.isaac_to_mujoco_map], np.arange(self.num_joints)):
            raise ValueError("mujoco_to_isaac_map and isaac_to_mujoco_map are not inverse mappings.")

        for joint_name, actuator_name in zip(config.joint_names_mujoco, config.actuator_names_mujoco):
            joint_id = self.model.joint(joint_name).id
            actuator_id = self.model.actuator(actuator_name).id
            actuator_joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if actuator_joint_id != joint_id:
                raise ValueError(f"Actuator {actuator_name} does not drive joint {joint_name}.")

        print(
            f"[Check] AMP obs: frame_dim={frame_dim}, history={config.history_length}, "
            f"onnx_input={self.policy_input_dim}, actions={self.policy_output_dim}"
        )
        print(
            f"[Check] Joint map/gains: {self.num_joints} joints, MuJoCo actuators aligned, maps invertible. "
            f"action_scale=[{config.action_scale.min():.3f}, {config.action_scale.max():.3f}]"
        )

    # -------- 策略热切换 --------

    def load_policy(self, config_path, model_name, policy_idx):
        """Hot-swap config + policy network without restarting MuJoCo."""
        print(f"[PolicySwitch] Loading policy {policy_idx}: {config_path} / {model_name}")
        _base_dir = self._base_dir

        new_cfg = self._load_config(config_path)

        policy_path = os.path.join(_base_dir, "exported_policy", model_name)
        if not os.path.exists(policy_path):
            print(f"[PolicySwitch] ❗ Policy file not found: {policy_path}, skipping.")
            return False

        self.config = new_cfg
        c = new_cfg
        self._load_onnx(policy_path)
        self._validate_dimensions(c)
        self.policy_decimation = int(round(c.control_dt / c.sim_dt))
        self.kp = c.kps[c.isaac_to_mujoco_map]
        self.kd = c.kds[c.isaac_to_mujoco_map]
        self.default_qpos_mj = c.default_joint_pos[c.isaac_to_mujoco_map]
        self.target_qpos_mj = self.default_qpos_mj.copy()
        # Reset observation buffers
        self.obs_history = self._make_obs_history(c)
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self._active_policy_idx = policy_idx
        print(f"[PolicySwitch] ✅ Switched to policy {policy_idx}")
        return True

    def pd_controller(self, target_pos_mj, target_vel_mj):
        """Compute torques via PD control and send to MuJoCo actuators.
        tau = kp * (target_q - current_q) - kd * current_v
        """         
        tau = self.kp * (target_pos_mj - self.data.qpos[self.joint_qpos_addrs]) + self.kd * (target_vel_mj - self.data.qvel[self.joint_qvel_addrs])
        self._last_tau_mj = tau.astype(np.float32).copy()
        self._last_tau_isaac = self._last_tau_mj[self.config.mujoco_to_isaac_map]
        self.data.ctrl[self.actuator_ids] = tau

    def _joint_names_isaac(self):
        return [self.config.joint_names_mujoco[int(i)] for i in self.config.mujoco_to_isaac_map]

    def _debug_joint_indices(self):
        names = self._joint_names_isaac()
        if self.debug_joints == "all":
            return list(range(self.num_joints))
        if self.debug_joints == "arms":
            keys = ("shoulder", "elbow", "wrist")
        elif self.debug_joints == "legs":
            keys = ("hip", "knee", "ankle")
        else:
            keys = ("hip", "knee", "ankle", "waist", "shoulder")
        return [i for i, name in enumerate(names) if any(k in name for k in keys)]

    def _print_policy_debug(
        self,
        sim_time,
        motiontime,
        commands,
        base_ang,
        proj_grav,
        curr_qpos_isaac,
        curr_qvel_isaac,
        obs,
        obs_input,
        action,
        target_qpos_isaac,
    ):
        root_pos = self.data.qpos[:3].copy()
        root_quat = self.data.qpos[3:7].copy()
        root_rpy = quat_wxyz_to_euler_xyz(root_quat)
        root_lin_vel = self.data.qvel[:3].copy()
        root_ang_vel = self.data.qvel[3:6].copy()
        root_lin_vel_yaw = rotate_world_to_yaw_frame(root_rpy[2], root_lin_vel)

        target_err = target_qpos_isaac - curr_qpos_isaac
        names = self._joint_names_isaac()
        indices = self._debug_joint_indices()

        print("\n" + "=" * 118, flush=True)
        print(f"[PolicyDebug] sim_t={sim_time:.3f}s step={motiontime} policy_dt={self.config.control_dt:.4f}s", flush=True)
        print(
            "[PolicyDebug] command[vx vy yaw]="
            f"{np.array2string(commands, precision=3, suppress_small=False)}",
            flush=True,
        )
        print(
            "[MuJoCo root] pos(xyz)="
            f"{np.array2string(root_pos, precision=4, suppress_small=False)} "
            "quat(wxyz)="
            f"{np.array2string(root_quat, precision=4, suppress_small=False)}",
            flush=True,
        )
        print(
            "[MuJoCo root] rpy(rad)="
            f"{np.array2string(root_rpy, precision=4, suppress_small=False)} "
            "lin_vel_raw="
            f"{np.array2string(root_lin_vel, precision=4, suppress_small=False)} "
            "lin_vel_yaw="
            f"{np.array2string(root_lin_vel_yaw, precision=4, suppress_small=False)} "
            "ang_vel_raw="
            f"{np.array2string(root_ang_vel, precision=4, suppress_small=False)}",
            flush=True,
        )
        print(
            "[Obs newest] base_ang_raw="
            f"{np.array2string(base_ang, precision=4, suppress_small=False)} "
            "projected_gravity="
            f"{np.array2string(proj_grav, precision=4, suppress_small=False)}",
            flush=True,
        )
        print(
            "[Obs input] shape="
            f"{obs_input.shape} min={obs_input.min():+.4f} max={obs_input.max():+.4f} "
            f"mean={obs_input.mean():+.4f} l2={np.linalg.norm(obs_input):.4f} "
            f"nan={np.isnan(obs_input).any()} inf={np.isinf(obs_input).any()}",
            flush=True,
        )
        print(
            "[Policy action policy-order] min="
            f"{action.min():+.4f} max={action.max():+.4f} mean={action.mean():+.4f} "
            f"l2={np.linalg.norm(action):.4f}",
            flush=True,
        )
        print(
            "[Policy action policy-order raw]\n"
            f"{np.array2string(action, precision=4, suppress_small=False, max_line_width=160)}",
            flush=True,
        )
        print(
            "[Policy target_qpos policy-order]\n"
            f"{np.array2string(target_qpos_isaac, precision=4, suppress_small=False, max_line_width=160)}",
            flush=True,
        )
        print("[Joint table in policy order]", flush=True)
        print(" idx name                              q       qd     action   target      err      tau", flush=True)
        for i in indices:
            print(
                f"{i:>3d} {names[i]:<30s} "
                f"{curr_qpos_isaac[i]:+7.3f} {curr_qvel_isaac[i]:+7.3f} "
                f"{action[i]:+8.3f} {target_qpos_isaac[i]:+8.3f} "
                f"{target_err[i]:+8.3f} {self._last_tau_isaac[i]:+8.2f}",
                flush=True,
            )
        print("=" * 118 + "\n", flush=True)
    
    def run(self, gamepad, policy_registry=None):
        """
        Main simulation loop with Absolute Time Sync, Camera Follow, and Policy Switching.
        policy_registry: dict {policy_idx: (config_path, model_name)}
        """
        motiontime = 0 # simulation step counter
        c = self.config
        if policy_registry is None:
            policy_registry = {}

        self.model.opt.timestep = self.sim_dt  # Physics timestep (e.g., 0.005s)

        # Launch viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Initial Camera setup
            viewer.cam.lookat[:] = self.data.qpos[:3]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20

            # --- Establish Absolute Time Reference ---
            start_time = time.time()

            while viewer.is_running():
                if gamepad.exit_requested:
                    print("\nExit request detected, ending simulation...")
                    break

                # --- 策略切换检测 ---
                requested = gamepad.active_policy
                if requested != 0 and requested != self._active_policy_idx:
                    if requested in policy_registry:
                        cfg_path, mdl_name = policy_registry[requested]
                        self.load_policy(cfg_path, mdl_name, requested)
                        c = self.config  # refresh local ref
                    else:
                        print(f"[PolicySwitch] Policy {requested} not registered, ignoring.")

                # Get MuJoCo internal simulation time
                sim_time = self.data.time

                # Apply mouse perturbation force from viewer UI:
                #   Double-click a body to select it, then Ctrl + Right-drag to push
                self.data.xfrc_applied[:] = 0
                mujoco.mjv_applyPerturbForce(self.model, self.data, viewer.perturb)

                self.pd_controller(self.target_qpos_mj, np.zeros_like(self.kd))
                
                mujoco.mj_step(self.model, self.data)
                motiontime += 1
                
                if motiontime % self.policy_decimation == 0:
                        
                    quat_wxyz = self.data.qpos[3:7]
                    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
                    proj_grav = compute_projected_gravity(quat_xyzw)
                    
                    base_ang = self.data.qvel[3:6]
                    
                    curr_qpos_mj = self.data.qpos[self.joint_qpos_addrs]
                    curr_qvel_mj = self.data.qvel[self.joint_qvel_addrs]
                    
                    curr_qpos_isaac = curr_qpos_mj[c.mujoco_to_isaac_map]
                    curr_qpos_rel_isaac = curr_qpos_isaac - c.default_joint_pos
                    curr_qvel_isaac = curr_qvel_mj[c.mujoco_to_isaac_map]
                    # Get user input from gamepad
                    cmd_vx, cmd_vy, cmd_vyaw = gamepad.get_velocity()
                    # cmd_vx, cmd_vy, cmd_vyaw = [0.0, 0.0, 0.4]
                    commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
                        
                    # Prepare observation for the policy
                    obs = build_obs(base_ang, proj_grav, commands, 
                                        curr_qpos_rel_isaac, curr_qvel_isaac, self.last_action, self.config)
                    self.obs_history.append(obs)
                    # Group-major reorganization (matches training format):
                    # [omega x H, gravity x H, cmd x H, pos x H, vel x H, action x H]
                    # --- Dynamic Observation Stacking (Compatible with H=1 to N) ---
                    n = self.num_joints
                    obs_arr = np.array(list(self.obs_history))

                    # Define feature slices [start, end]
                    feature_indices = [
                        (0, 3),         # Angular velocity
                        (3, 6),         # Projected gravity
                        (6, 9),         # Commands
                        (9, 9 + n),     # Joint positions
                        (9 + n, 9 + 2 * n),   # Joint velocities
                        (9 + 2 * n, 9 + 3 * n), # Last actions
                    ]

                    # Extract each feature across all history frames and flatten
                    # Result format: [Feature1_t0...tN, Feature2_t0...tN, ...]
                    obs_input = np.concatenate([
                        obs_arr[:, start:end].ravel() for start, end in feature_indices
                    ])

                    # Final batch preparation for policy inference
                    obs_batch = obs_input[np.newaxis, :].astype(np.float32)

                    # ONNX inference
                    outputs = self.ort_session.run(
                        self.ort_output_names,
                        {self.ort_input_names[0]: obs_batch},
                    )
                    action = outputs[0].flatten().astype(np.float32)
                        
                    # Parse 29D action: position offsets
                    self.last_action = action.copy()
                        
                    # Position action: 29 dims (relative to default), scale
                    # Policy order -> MuJoCo order.
                    self.target_qpos_isaac = action * self.config.action_scale + c.default_joint_pos
                    self.target_qpos_mj = self.target_qpos_isaac[self.config.isaac_to_mujoco_map]

                    if self.debug_policy and sim_time - self._last_debug_print_time >= self.debug_interval:
                        tau_preview_mj = self.kp * (self.target_qpos_mj - curr_qpos_mj) - self.kd * curr_qvel_mj
                        self._last_tau_isaac = tau_preview_mj[c.mujoco_to_isaac_map].astype(np.float32)
                        self._print_policy_debug(
                            sim_time=sim_time,
                            motiontime=motiontime,
                            commands=commands,
                            base_ang=base_ang,
                            proj_grav=proj_grav,
                            curr_qpos_isaac=curr_qpos_isaac,
                            curr_qvel_isaac=curr_qvel_isaac,
                            obs=obs,
                            obs_input=obs_input,
                            action=action,
                            target_qpos_isaac=self.target_qpos_isaac,
                        )
                        self._last_debug_print_time = sim_time
                
                # --- 摄像头跟随机器人 ---
                viewer.cam.lookat[:] = self.data.qpos[:3]

                viewer.sync() # Sync rendering to the simulation loop (decimated by render_skip)
                    
                 # Rudimentary time keeping, will drift relative to wall clock.
                expected_real_time = start_time + (motiontime * self.sim_dt)
                time_to_sleep = expected_real_time - time.time()
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)   
            
            
                
                # --- Status telemetry ---
                if motiontime % int(1.0 / self.config.sim_dt) == 0:
                    real_time_now = time.time() - start_time
                    actual_hz = motiontime / real_time_now if real_time_now > 0 else 0
                    elapsed_sim_time = motiontime * self.sim_dt  # use motiontime, not data.time (immune to viewer reset)
                    
                    vx_cur, vy_cur, vyaw_cur = gamepad.get_velocity()
                    
                    print(f"[Gamepad] vx={vx_cur:+.2f} m/s | vy={vy_cur:+.2f} | yaw={vyaw_cur:+.2f} rad/s")
                    print(f"[Sim Time]: t={elapsed_sim_time:.1f}s, Base height: {self.data.qpos[2]:.3f}m")
                    print(f"[Real Time]: t={real_time_now:.1f}s, Actual Hz: {actual_hz:.2f} Hz")


# ========== Main ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='g1_amp.onnx')
    parser.add_argument('--config', type=str, default='g1_amp.yaml')
    parser.add_argument('--input', choices=['gamepad', 'keyboard'], default='gamepad', help='Control input device.')
    parser.add_argument('--gamepad_type', type=str, default=None, help='Override gamepad type from YAML.')
    parser.add_argument('--check', action='store_true', help='Validate config/model dimensions and exit before viewer.')
    parser.add_argument('--debug_policy', action='store_true', help='Print policy outputs and MuJoCo state periodically.')
    parser.add_argument('--debug_interval', type=float, default=1.0, help='Seconds between --debug_policy prints.')
    parser.add_argument('--debug_gamepad', action='store_true', help='Print raw gamepad axes and mapped velocity commands.')
    parser.add_argument('--gamepad_debug_interval', type=float, default=None, help='Seconds between --debug_gamepad prints.')
    parser.add_argument(
        '--debug_joints',
        choices=['all', 'core', 'arms', 'legs'],
        default='all',
        help='Joint subset printed by --debug_policy.',
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve_config(name):
        """Resolve config yaml path relative to script's config/ directory."""
        if os.path.exists(name):
            return name
        return os.path.join(script_dir, 'config', name)

    amp_config = resolve_config(args.config)

    # ---- 策略注册表 ----
    # key: gamepad combo index (1=RB+A, 2=RB+B, 3=RB+X, 4=RB+Y)
    # value: (config_yaml_path, policy_model_name)
    # policy1/policy2/policy3 配置文件尚未创建，切换时会自动跳过
    policy_registry = {
        1: (amp_config,                               args.model),           # RB+A → AMP 策略
        2: (resolve_config('policy1.yaml'),           'policy1.onnx'),       # RB+B → 策略 1 (占位符)
        3: (resolve_config('policy2.yaml'),           'policy2.onnx'),       # RB+X → 策略 2 (占位符)
        4: (resolve_config('policy3.yaml'),           'policy3.onnx'),       # RB+Y → 策略 3 (占位符)
    }

    # 1. 初始化 Controller，默认加载 AMP 策略
    controller = Sim2SimController(
        amp_config,
        args.model,
        debug_policy=args.debug_policy,
        debug_interval=args.debug_interval,
        debug_joints=args.debug_joints,
    )
    controller._active_policy_idx = 1
    if args.check:
        print("[Check] Config/model validation passed.")
        sys.exit(0)

    # 2. 初始化输入设备
    cfg = controller.config
    if args.input == 'keyboard':
        gamepad = KeyboardController(
            vx_range=cfg.command_range['lin_vel_x'],
            vy_range=cfg.command_range['lin_vel_y'],
            vyaw_range=cfg.command_range['ang_vel_z'],
        )
    else:
        gamepad_type = args.gamepad_type or getattr(cfg, 'gamepad_type_sim2sim', getattr(cfg, 'gamepad_type', 'gamesir'))
        gamepad = create_gamepad_controller(
            gamepad_type,
            vx_range=cfg.command_range['lin_vel_x'],
            vy_range=cfg.command_range['lin_vel_y'],
            vyaw_range=cfg.command_range['ang_vel_z'],
            btn_start=getattr(cfg, 'gamepad_btn_start', None),
            btn_rb=getattr(cfg, 'gamepad_btn_rb', None),
            btn_a=getattr(cfg, 'gamepad_btn_a', None),
            axis_left_x=getattr(cfg, 'gamepad_axis_left_x', None),
            axis_left_y=getattr(cfg, 'gamepad_axis_left_y', None),
            axis_right_x=getattr(cfg, 'gamepad_axis_right_x', None),
            deadzone=getattr(cfg, 'gamepad_deadzone', 0.05),
            command_slew_rate=getattr(cfg, 'gamepad_slew_rate', (2.0, 4.0, 3.0)),
            debug=args.debug_gamepad or bool(getattr(cfg, 'gamepad_debug', False)),
            debug_interval=(
                args.gamepad_debug_interval
                if args.gamepad_debug_interval is not None
                else getattr(cfg, 'gamepad_debug_interval', 0.5)
            ),
        )
    try:
        gamepad.start()
    except Exception:
        gamepad.stop()
        raise
    # 初始化时同步活跋索引，避免第一帧就触发切换
    gamepad.active_policy = 1

    print("\n" + "="*70)
    if args.input == 'keyboard':
        print("  W/S or Up/Down     : vx +/-")
        print("  A/D                : vy +/-")
        print("  Q/E or Left/Right  : vyaw +/-")
        print("  Space or 0         : zero command")
        print("  1/2/3/4            : policy switch")
        print("  X or Esc           : Exit")
    else:
        if cfg.command_range['lin_vel_x'][0] < 0.0:
            print("  Left Joystick Up/Down : vx (forward/back)")
        else:
            print("  Left Joystick Up      : vx (forward; back disabled by config)")
        print("  Left Joystick L/R     : vy (strafe)")
        print("  Right Joystick L/R    : vyaw (turn)")
        print("  RB + A  : AMP policy   (g1_amp)")
        print("  RB + B  : Policy 1     (policy1 - placeholder)")
        print("  RB + X  : Policy 2     (policy2 - placeholder)")
        print("  RB + Y  : Policy 3     (policy3 - placeholder)")
        print("  Start   : Exit")
        print("  Optional: --debug_gamepad prints raw axes/mapped commands")
    print("="*70 + "\n")

    try:
        # 3. 运行
        controller.run(gamepad, policy_registry)
    finally:
        gamepad.stop()
    print("\nProgram ended.")
