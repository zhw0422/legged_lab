#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MuJoCo sim2sim deployment for the G1 attention terrain policy."""

from __future__ import annotations

import argparse
import importlib
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


def _load_mujoco():
    module = importlib.import_module("mujoco")
    if hasattr(module, "MjModel") and hasattr(module, "mj_multiRay"):
        return module

    sys.modules.pop("mujoco", None)
    repo_root = Path(__file__).resolve().parents[3]
    filtered = []
    for entry in sys.path:
        try:
            resolved = Path.cwd().resolve() if entry == "" else Path(entry).resolve()
        except OSError:
            filtered.append(entry)
            continue
        if resolved == repo_root:
            continue
        filtered.append(entry)
    sys.path[:] = filtered

    module = importlib.import_module("mujoco")
    if not hasattr(module, "MjModel") or not hasattr(module, "mj_multiRay"):
        raise ImportError(
            "The imported mujoco module does not expose MjModel/mj_multiRay. "
            "Install the official MuJoCo Python package or run from an environment where it is available."
        )
    return module


mujoco = _load_mujoco()
import mujoco.viewer

SCRIPT_DIR = Path(__file__).resolve().parent
G1_DEPLOY_DIR = SCRIPT_DIR.parent
DEPLOY_DIR = G1_DEPLOY_DIR.parent
sys.path.insert(0, str(DEPLOY_DIR / "utils"))
from joystick import create_gamepad_controller


def quat_rotate_inverse(q, v):
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def quat_apply(q_xyzw, v):
    """Forward quaternion rotation. q is (x, y, z, w)."""
    q_w = q_xyzw[3]
    q_vec = np.asarray(q_xyzw[:3], dtype=np.float64)
    a = np.asarray(v, dtype=np.float64) * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * (2.0 * np.dot(q_vec, v))
    return a + b + c


def compute_projected_gravity(quat_xyzw):
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return quat_rotate_inverse(quat_xyzw, gravity_world)


def quat_wxyz_to_euler_xyz(q):
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


def yaw_from_xmat(xmat):
    mat = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def yaw_rotation_matrix(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_proprio_obs(base_ang_vel, projected_gravity, commands, dof_pos_rel, dof_vel, last_action, config):
    command_scale = np.asarray(config.command_scale, dtype=np.float32)
    obs = np.concatenate(
        [
            np.asarray(base_ang_vel, dtype=np.float32) * float(config.ang_vel_scale),
            np.asarray(projected_gravity, dtype=np.float32),
            np.asarray(commands, dtype=np.float32) * command_scale,
            np.asarray(dof_pos_rel, dtype=np.float32) * float(config.dof_pos_scale),
            np.asarray(dof_vel, dtype=np.float32) * float(config.dof_vel_scale),
            np.asarray(last_action, dtype=np.float32) * float(config.last_action_scale),
        ]
    )
    return obs.astype(np.float32)


class KeyboardController:
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


class ConstantController:
    """Headless drop-in replacement for the gamepad/keyboard controllers.

    Holds zero command for ``warmup_s`` real-time seconds (so the robot can
    settle from the spawn pose), then linearly ramps to the configured
    (vx, vy, vyaw) target over ``ramp_s`` seconds. Used by ``--input const``
    when the terminal cannot deliver keyboard events to the running process.
    """

    def __init__(self, vx=0.0, vy=0.0, vyaw=0.0, warmup_s=2.0, ramp_s=1.0):
        self.target = (float(vx), float(vy), float(vyaw))
        self.warmup_s = float(warmup_s)
        self.ramp_s = max(float(ramp_s), 1e-3)
        self.active_policy = 1
        self.exit_requested = False
        self._start = None

    def start(self):
        self._start = time.time()

    def stop(self):
        pass

    def get_velocity(self):
        if self._start is None:
            return 0.0, 0.0, 0.0
        elapsed = time.time() - self._start
        if elapsed < self.warmup_s:
            return 0.0, 0.0, 0.0
        alpha = min((elapsed - self.warmup_s) / self.ramp_s, 1.0)
        return tuple(alpha * v for v in self.target)


class Sim2SimAttentionController:
    def __init__(
        self,
        config_path,
        model_name=None,
        debug_policy=False,
        debug_interval=1.0,
        show_rays=False,
        show_ray_lines=False,
        ray_subsample=6,
    ):
        self._base_dir = G1_DEPLOY_DIR
        self.debug_policy = debug_policy
        self.debug_interval = max(float(debug_interval), 1.0e-6)
        self.show_rays = bool(show_rays)
        self.show_ray_lines = bool(show_ray_lines)
        self.ray_subsample = max(1, int(ray_subsample))
        self._last_debug_print_time = -np.inf
        self._last_scan_cache = None

        self.config = self._load_config(config_path)
        c = self.config
        self.xml_path = self._resolve_path(c.xml_path, self._base_dir)
        self.policy_path = self._resolve_policy_path(model_name or c.policy_path)

        print(f"Loading MuJoCo model: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(c.sim_dt)
        self.sim_dt = float(c.sim_dt)
        self.policy_decimation = int(round(float(c.control_dt) / float(c.sim_dt)))

        self.joint_qpos_addrs = [self.model.jnt_qposadr[self.model.joint(n).id] for n in c.joint_names_mujoco]
        self.joint_qvel_addrs = [self.model.jnt_dofadr[self.model.joint(n).id] for n in c.joint_names_mujoco]
        self.actuator_ids = [self.model.actuator(n).id for n in c.actuator_names_mujoco]
        self.num_joints = len(c.joint_names_mujoco)

        self.terrain_body_id = self._body_id(c.terrain_sensor_body)
        self._configure_ray_geom_groups()
        self._build_scan_grid()

        print(f"Loading policy: {self.policy_path}")
        self._load_onnx(self.policy_path)
        self._validate_dimensions(c)

        self.kp = c.kps[c.isaac_to_mujoco_map]
        self.kd = c.kds[c.isaac_to_mujoco_map]
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self.target_qpos_isaac = c.default_joint_pos.copy()
        self.default_qpos_mj = c.default_joint_pos[c.isaac_to_mujoco_map]
        self.target_qpos_mj = self.default_qpos_mj.copy()
        self._last_tau_mj = np.zeros(self.num_joints, dtype=np.float32)
        self._last_tau_isaac = np.zeros(self.num_joints, dtype=np.float32)

        self.data.qpos[self.joint_qpos_addrs] = self.default_qpos_mj
        self.data.qpos[2] = float(getattr(c, "init_height", 0.90))
        mujoco.mj_forward(self.model, self.data)

        print("Sim2Sim attention controller initialized")

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)

        cfg = SimpleNamespace(**raw_cfg)
        string_list_keys = {"joint_names_mujoco", "actuator_names_mujoco", "sdk_joint_order"}
        int_list_keys = {
            "mujoco_to_isaac_map",
            "isaac_to_mujoco_map",
            "sdk2isaac_idx",
        }
        for k, v in raw_cfg.items():
            if isinstance(v, list) and k not in string_list_keys:
                try:
                    dtype = np.int32 if k in int_list_keys else np.float32
                    v = np.array(v, dtype=dtype)
                except (ValueError, TypeError):
                    pass
            setattr(cfg, k, v)
        return cfg

    def _resolve_path(self, path, base_dir):
        candidate = Path(path)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        direct = base_dir / candidate
        if direct.exists():
            return direct
        asset = base_dir / "assets" / candidate.name
        if asset.exists():
            return asset
        return direct

    def _resolve_policy_path(self, path):
        candidate = Path(path)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        direct = self._base_dir / candidate
        if direct.exists():
            return direct
        policy = self._base_dir / "exported_policy" / candidate.name
        if policy.exists():
            return policy
        return policy

    def _load_onnx(self, path):
        import onnxruntime as ort

        self.ort_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.ort_input_names = [inp.name for inp in self.ort_session.get_inputs()]
        self.ort_output_names = [out.name for out in self.ort_session.get_outputs()]
        input_shape = self.ort_session.get_inputs()[0].shape
        output_shape = self.ort_session.get_outputs()[0].shape
        self.policy_input_dim = int(input_shape[-1]) if isinstance(input_shape[-1], int) else None
        self.policy_output_dim = int(output_shape[-1]) if isinstance(output_shape[-1], int) else None

    def _body_id(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found in MuJoCo model.")
        return body_id

    def _configure_ray_geom_groups(self):
        self._ray_geomgroup = np.zeros(getattr(mujoco, "mjNGROUP", 6), dtype=np.uint8)
        self._ray_geomgroup[0] = 1
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) != 0:
                self.model.geom_group[geom_id] = 1

    def _build_scan_grid(self):
        c = self.config
        length = int(c.terrain_map_length)
        width = int(c.terrain_map_width)
        coord_dim = int(c.terrain_map_coord_dim)
        if coord_dim != 3:
            raise ValueError(f"terrain_map_coord_dim={coord_dim}, expected 3.")

        size = np.asarray(c.terrain_map_size, dtype=np.float64)
        xs = np.linspace(-0.5 * size[0], 0.5 * size[0], length, dtype=np.float64)
        ys = np.linspace(-0.5 * size[1], 0.5 * size[1], width, dtype=np.float64)
        ordering = str(getattr(c, "terrain_ordering", "xy")).lower()
        if ordering == "xy":
            # IsaacLab "xy" → numpy "xy" meshgrid: grid shape (width, length), order varies y-fastest.
            grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        elif ordering == "yx":
            grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        else:
            raise ValueError(f"Unsupported terrain_ordering='{ordering}', expected 'xy' or 'yx'.")

        grid = np.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), np.zeros(grid_x.size, dtype=np.float64)],
            axis=1,
        )

        offset = np.asarray(c.terrain_sensor_offset, dtype=np.float64)
        self._scan_grid_local = grid
        self._scan_offset_local = offset
        self.nray = int(grid.shape[0])

    def _proprio_dim(self):
        return 3 + 3 + 3 + 3 * self.num_joints

    def _terrain_dim(self):
        return int(self.config.terrain_map_length) * int(self.config.terrain_map_width) * int(self.config.terrain_map_coord_dim)

    def _validate_dimensions(self, config):
        expected_proprio_dim = self._proprio_dim()
        expected_terrain_dim = self._terrain_dim()
        expected_obs_dim = expected_proprio_dim + expected_terrain_dim

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
        if self.nray * 3 != expected_terrain_dim:
            raise ValueError(f"scan rays produce {self.nray * 3} values, expected {expected_terrain_dim}.")

        config.command_scale = np.asarray(config.command_scale, dtype=np.float32)
        if config.command_scale.shape != (3,):
            raise ValueError(f"command_scale has shape {config.command_scale.shape}, expected (3,).")

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
            f"[Check] attention obs: proprio={expected_proprio_dim}, terrain={expected_terrain_dim}, "
            f"total={expected_obs_dim}, rays={self.nray}, onnx_input={self.policy_input_dim}, "
            f"actions={self.policy_output_dim}"
        )
        print(
            f"[Check] terrain sensor: body={config.terrain_sensor_body} id={self.terrain_body_id}, "
            f"ordering={config.terrain_ordering}, action_scale=[{config.action_scale.min():.3f}, {config.action_scale.max():.3f}]"
        )

    def pd_controller(self, target_pos_mj, target_vel_mj):
        tau = self.kp * (target_pos_mj - self.data.qpos[self.joint_qpos_addrs]) + self.kd * (
            target_vel_mj - self.data.qvel[self.joint_qvel_addrs]
        )
        self._last_tau_mj = tau.astype(np.float32).copy()
        self._last_tau_isaac = self._last_tau_mj[self.config.mujoco_to_isaac_map]
        self.data.ctrl[self.actuator_ids] = tau

    def compute_terrain_scan(self):
        mujoco.mj_kinematics(self.model, self.data)
        body_pos_w = np.asarray(self.data.xpos[self.terrain_body_id], dtype=np.float64).copy()
        yaw = yaw_from_xmat(self.data.xmat[self.terrain_body_id])
        rot = yaw_rotation_matrix(yaw)

        sensor_pos_w = body_pos_w + self._scan_offset_local
        grid_world_xy = self._scan_grid_local @ rot.T
        ray_starts_w = grid_world_xy + sensor_pos_w.reshape(1, 3)
        ray_dirs_w = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (self.nray, 1))

        geomid = np.full(self.nray, -1, dtype=np.int32)
        dist = np.full(self.nray, -1.0, dtype=np.float64)

        # mj_multiRay assumes all rays share one origin, so we cast each ray with mj_ray.
        for i in range(self.nray):
            geom_id_buf = np.array([-1], dtype=np.int32)
            d = mujoco.mj_ray(
                self.model,
                self.data,
                ray_starts_w[i],
                ray_dirs_w[i],
                self._ray_geomgroup,
                int(self.config.terrain_flg_static),
                int(self.config.terrain_bodyexclude),
                geom_id_buf,
                None,
            )
            dist[i] = d
            geomid[i] = int(geom_id_buf[0])

        valid = (dist >= 0.0) & np.isfinite(dist)
        hit_world = ray_starts_w.copy()
        hit_world[valid, 2] = ray_starts_w[valid, 2] - dist[valid]
        # Misses: fall back to ray start (zero relative z after subtraction, clipped later).
        hit_world[~valid, 2] = body_pos_w[2]

        # `sensor.data.pos_w` in IsaacLab equals the sensor body position (offset is only baked
        # into `ray_starts`), so the relative term subtracts body_pos_w, not body_pos_w + offset.
        relative_w = hit_world - body_pos_w.reshape(1, 3)
        # Yaw inverse: world xy → yaw-aligned local xy. rot is the yaw rotation, so use rot[:2,:2].T.
        rot_xy_inv = rot[:2, :2].T
        local_xy = relative_w[:, :2] @ rot_xy_inv.T
        local_xyz = np.concatenate([local_xy, relative_w[:, 2:3]], axis=1)
        local_xyz = np.nan_to_num(local_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        z_clip = np.asarray(self.config.terrain_map_z_clip, dtype=np.float64)
        local_xyz[:, 2] = np.clip(local_xyz[:, 2], z_clip[0], z_clip[1])
        terrain_map = local_xyz.reshape(-1).astype(np.float32)

        self._last_scan_cache = {
            "ray_starts": ray_starts_w.copy(),
            "sensor_pos": sensor_pos_w.copy(),
            "body_pos": body_pos_w.copy(),
            "hit_world": hit_world.copy(),
            "local_xyz": local_xyz.copy(),
            "valid": valid.copy(),
            "geomid": geomid.copy(),
        }
        return terrain_map

    def build_attention_obs(self, base_ang, proj_grav, commands, curr_qpos_rel_isaac, curr_qvel_isaac):
        proprio = build_proprio_obs(
            base_ang,
            proj_grav,
            commands,
            curr_qpos_rel_isaac,
            curr_qvel_isaac,
            self.last_action,
            self.config,
        )
        terrain_map = self.compute_terrain_scan()
        obs = np.concatenate([proprio, terrain_map]).astype(np.float32)
        expected = int(self.config.num_obs)
        if obs.shape != (expected,):
            raise ValueError(f"attention obs shape={obs.shape}, expected ({expected},).")
        return proprio, terrain_map, obs

    def draw_scan_debug(self, viewer):
        if not self.show_rays or viewer.user_scn is None or self._last_scan_cache is None:
            return
        user_scn = viewer.user_scn
        user_scn.ngeom = 0
        hits = self._last_scan_cache["hit_world"]
        valid = self._last_scan_cache["valid"]
        starts = self._last_scan_cache["ray_starts"]
        mat = np.eye(3, dtype=np.float64).reshape(-1)
        dot_size = np.array([0.018, 0.018, 0.018], dtype=np.float64)
        zero_size = np.zeros(3, dtype=np.float64)
        valid_rgba = np.array([0.0, 0.9, 0.2, 0.85], dtype=np.float32)
        miss_rgba = np.array([1.0, 0.15, 0.0, 0.85], dtype=np.float32)
        line_rgba = np.array([0.0, 0.6, 1.0, 0.35], dtype=np.float32)

        for idx in range(0, self.nray, self.ray_subsample):
            if user_scn.ngeom >= user_scn.maxgeom:
                break
            rgba = valid_rgba if valid[idx] else miss_rgba
            geom = user_scn.geoms[user_scn.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, dot_size, hits[idx], mat, rgba)
            user_scn.ngeom += 1

            if self.show_ray_lines and user_scn.ngeom < user_scn.maxgeom:
                geom = user_scn.geoms[user_scn.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, zero_size, np.zeros(3), mat, line_rgba)
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, starts[idx], hits[idx])
                user_scn.ngeom += 1

    def _print_policy_debug(self, sim_time, motiontime, commands, proprio, terrain_map, obs_input, action,
                             base_ang_body, base_ang_world, base_lin_world, proj_grav,
                             curr_qpos_rel_isaac, curr_qvel_isaac,
                             onnx_call_ms, ctrl_step_ms, ctrl_step_ms_ema, total_step_ms):
        c = self.config
        scan = self._last_scan_cache
        valid_ratio = float(np.mean(scan["valid"])) if scan is not None else 0.0
        terrain_xyz = terrain_map.reshape(-1, 3)
        num_joints = self.num_joints

        # Slice proprio block by training term order
        ang_part   = proprio[0:3]
        grav_part  = proprio[3:6]
        cmd_part   = proprio[6:9]
        qpos_part  = proprio[9:9 + num_joints]
        qvel_part  = proprio[9 + num_joints:9 + 2 * num_joints]
        last_part  = proprio[9 + 2 * num_joints:9 + 3 * num_joints]

        root_quat_wxyz = self.data.qpos[3:7].copy()
        root_rpy = quat_wxyz_to_euler_xyz(root_quat_wxyz)
        base_pos = self.data.qpos[:3].copy()
        torso_pos = scan["body_pos"] if scan is not None else np.zeros(3)
        torso_yaw = yaw_from_xmat(self.data.xmat[self.terrain_body_id])

        # Forward / lateral / backward scan slices: take rows of the (W,L,3) reshape
        # Training uses reshape(-1, width=21, length=33, coord_dim=3)
        try:
            tm = terrain_xyz.reshape(int(c.terrain_map_width), int(c.terrain_map_length), 3)
            mid_w = int(c.terrain_map_width) // 2
            mid_l = int(c.terrain_map_length) // 2
            scan_center_z   = float(tm[mid_w, mid_l, 2])
            scan_forward_z  = tm[mid_w, mid_l:, 2]
            scan_back_z     = tm[mid_w, :mid_l, 2]
            scan_left_z     = tm[mid_w:, mid_l, 2]
            scan_right_z    = tm[:mid_w, mid_l, 2]
        except ValueError:
            scan_center_z = float("nan")
            scan_forward_z = scan_back_z = scan_left_z = scan_right_z = np.array([np.nan])

        print("\n" + "=" * 110, flush=True)
        print(
            f"[time] sim_t={sim_time:8.3f}s | step={motiontime:6d} | "
            f"sim_dt={self.sim_dt * 1000:.2f}ms | ctrl_dt_target={self.sim_dt * self.policy_decimation * 1000:.2f}ms | "
            f"ctrl_step={ctrl_step_ms:6.2f}ms (ema={ctrl_step_ms_ema:6.2f}ms) | "
            f"step_total={total_step_ms:6.2f}ms | onnx={onnx_call_ms:5.2f}ms",
            flush=True,
        )
        print(
            f"[pose] base_xyz=[{base_pos[0]:+.3f},{base_pos[1]:+.3f},{base_pos[2]:+.3f}] "
            f"torso_xyz=[{torso_pos[0]:+.3f},{torso_pos[1]:+.3f},{torso_pos[2]:+.3f}] "
            f"rpy=[{root_rpy[0]:+.3f},{root_rpy[1]:+.3f},{root_rpy[2]:+.3f}] "
            f"yaw_torso={torso_yaw:+.3f}",
            flush=True,
        )
        print(
            f"[cmd raw   ] {np.array2string(commands, precision=3, suppress_small=False)}    "
            f"command_scale={np.array2string(np.asarray(c.command_scale), precision=2)}    "
            f"-> cmd*scale={np.array2string(cmd_part, precision=3)}",
            flush=True,
        )
        print(
            f"[base_ang  ] world={np.array2string(base_ang_world, precision=3)} "
            f"body={np.array2string(base_ang_body, precision=3)} "
            f"obs(scaled*0.25)={np.array2string(ang_part, precision=3)}",
            flush=True,
        )
        print(
            f"[base_lin  ] world={np.array2string(base_lin_world, precision=3)}  "
            f"proj_grav={np.array2string(grav_part, precision=3)}",
            flush=True,
        )
        # Pretty-print joints (Isaac order). Show a compact summary plus per-joint table only every N seconds.
        print(
            f"[qpos rel  ] min={qpos_part.min():+.3f} max={qpos_part.max():+.3f} "
            f"mean={qpos_part.mean():+.3f}  l2={np.linalg.norm(qpos_part):.3f}",
            flush=True,
        )
        print(
            f"[qvel*0.05 ] min={qvel_part.min():+.3f} max={qvel_part.max():+.3f} "
            f"mean={qvel_part.mean():+.3f}  l2={np.linalg.norm(qvel_part):.3f}",
            flush=True,
        )
        print(
            f"[last_act*0.1] min={last_part.min():+.3f} max={last_part.max():+.3f} "
            f"mean={last_part.mean():+.3f}  l2={np.linalg.norm(last_part):.3f}",
            flush=True,
        )
        print(
            f"[joints abs] qpos[isaac]={np.array2string(curr_qpos_rel_isaac + c.default_joint_pos, precision=3, suppress_small=True)}",
            flush=True,
        )
        print(
            f"[joints rel] qpos_rel  ={np.array2string(curr_qpos_rel_isaac, precision=3, suppress_small=True)}",
            flush=True,
        )
        print(
            f"[joints vel] qvel[isaac]={np.array2string(curr_qvel_isaac, precision=3, suppress_small=True)}",
            flush=True,
        )
        print(
            f"[scan stat ] valid={valid_ratio * 100:.1f}%   "
            f"z[min={terrain_xyz[:, 2].min():+.3f} max={terrain_xyz[:, 2].max():+.3f} mean={terrain_xyz[:, 2].mean():+.3f}]   "
            f"x[min={terrain_xyz[:, 0].min():+.3f} max={terrain_xyz[:, 0].max():+.3f}]   "
            f"y[min={terrain_xyz[:, 1].min():+.3f} max={terrain_xyz[:, 1].max():+.3f}]",
            flush=True,
        )
        print(
            f"[scan z slc] center={scan_center_z:+.3f}  "
            f"fwd(min/mean/max)=({scan_forward_z.min():+.3f}/{scan_forward_z.mean():+.3f}/{scan_forward_z.max():+.3f})  "
            f"back=({scan_back_z.min():+.3f}/{scan_back_z.mean():+.3f}/{scan_back_z.max():+.3f})  "
            f"L=({scan_left_z.min():+.3f}/{scan_left_z.mean():+.3f}/{scan_left_z.max():+.3f})  "
            f"R=({scan_right_z.min():+.3f}/{scan_right_z.mean():+.3f}/{scan_right_z.max():+.3f})",
            flush=True,
        )
        print(
            f"[obs total ] shape={obs_input.shape}  min={obs_input.min():+.4f} max={obs_input.max():+.4f} "
            f"mean={obs_input.mean():+.4f}  nan={np.isnan(obs_input).any()}  inf={np.isinf(obs_input).any()}",
            flush=True,
        )
        print(
            f"[action raw] min={action.min():+.4f} max={action.max():+.4f} mean={action.mean():+.4f} l2={np.linalg.norm(action):.4f}",
            flush=True,
        )
        clipped_now = np.any(np.abs(action) >= float(getattr(c, "action_clip", 1e9)) - 1e-3)
        print(
            f"[action set] action[isaac]={np.array2string(action, precision=3, suppress_small=True)}  saturated={clipped_now}",
            flush=True,
        )
        print(
            f"[target dq ] target_qpos_isaac={np.array2string(self.target_qpos_isaac.astype(np.float32), precision=3, suppress_small=True)}",
            flush=True,
        )
        print(
            f"[tau mj    ] min={self._last_tau_mj.min():+.2f} max={self._last_tau_mj.max():+.2f} "
            f"mean={self._last_tau_mj.mean():+.2f} l2={np.linalg.norm(self._last_tau_mj):.2f}  "
            f"tau[isaac]={np.array2string(self._last_tau_isaac, precision=1, suppress_small=True)}",
            flush=True,
        )
        print("=" * 110 + "\n", flush=True)

    def run(self, gamepad):
        motiontime = 0
        c = self.config
        last_ctrl_wall = time.time()
        ctrl_dt_ema = 0.0
        ctrl_dt_count = 0
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.lookat[:] = self.data.qpos[:3]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            start_time = time.time()

            while viewer.is_running():
                if gamepad.exit_requested:
                    print("\nExit request detected, ending simulation...")
                    break

                step_t0 = time.time()
                sim_time = self.data.time
                self.data.xfrc_applied[:] = 0
                mujoco.mjv_applyPerturbForce(self.model, self.data, viewer.perturb)

                self.pd_controller(self.target_qpos_mj, np.zeros_like(self.kd))
                mujoco.mj_step(self.model, self.data)
                motiontime += 1

                if motiontime % self.policy_decimation == 0:
                    quat_wxyz = self.data.qpos[3:7]
                    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
                    proj_grav = compute_projected_gravity(quat_xyzw)
                    # MuJoCo free-joint `qvel[3:6]` is angular velocity expressed in the
                    # *body frame*, which matches IsaacLab's `articulation.data.root_ang_vel_b`.
                    # The world-frame value is only used for diagnostics.
                    base_ang_body = self.data.qvel[3:6].astype(np.float32).copy()
                    base_ang_world = quat_apply(quat_xyzw, base_ang_body).astype(np.float32)
                    base_lin_world = self.data.qvel[0:3].astype(np.float32).copy()

                    curr_qpos_mj = self.data.qpos[self.joint_qpos_addrs]
                    curr_qvel_mj = self.data.qvel[self.joint_qvel_addrs]
                    curr_qpos_isaac = curr_qpos_mj[c.mujoco_to_isaac_map]
                    curr_qvel_isaac = curr_qvel_mj[c.mujoco_to_isaac_map]
                    curr_qpos_rel_isaac = curr_qpos_isaac - c.default_joint_pos

                    cmd_vx, cmd_vy, cmd_vyaw = gamepad.get_velocity()
                    commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
                    proprio, terrain_map, obs_input = self.build_attention_obs(
                        base_ang_body, proj_grav, commands, curr_qpos_rel_isaac, curr_qvel_isaac
                    )
                    obs_batch = obs_input[np.newaxis, :].astype(np.float32)

                    onnx_t0 = time.time()
                    outputs = self.ort_session.run(self.ort_output_names, {self.ort_input_names[0]: obs_batch})
                    onnx_call_ms = (time.time() - onnx_t0) * 1000.0
                    action = outputs[0].flatten().astype(np.float32)
                    action_clip = getattr(self.config, "action_clip", None)
                    if action_clip is not None:
                        action = np.clip(action, -float(action_clip), float(action_clip)).astype(np.float32)
                    self.last_action = action.copy()

                    self.target_qpos_isaac = action * self.config.action_scale + c.default_joint_pos
                    self.target_qpos_mj = self.target_qpos_isaac[self.config.isaac_to_mujoco_map]

                    # Track wall-clock duration of every single control step (50 Hz target).
                    now_ctrl = time.time()
                    ctrl_step_dt = now_ctrl - last_ctrl_wall
                    last_ctrl_wall = now_ctrl
                    if ctrl_dt_count == 0:
                        ctrl_dt_ema = ctrl_step_dt
                    else:
                        ctrl_dt_ema = 0.9 * ctrl_dt_ema + 0.1 * ctrl_step_dt
                    ctrl_dt_count += 1

                    if self.debug_policy and sim_time - self._last_debug_print_time >= self.debug_interval:
                        total_step_ms = (now_ctrl - step_t0) * 1000.0
                        self._print_policy_debug(
                            sim_time, motiontime, commands, proprio, terrain_map, obs_input, action,
                            base_ang_body, base_ang_world, base_lin_world, proj_grav,
                            curr_qpos_rel_isaac, curr_qvel_isaac,
                            onnx_call_ms, ctrl_step_dt * 1000.0, ctrl_dt_ema * 1000.0, total_step_ms,
                        )
                        self._last_debug_print_time = sim_time

                self.draw_scan_debug(viewer)
                viewer.cam.lookat[:] = self.data.qpos[:3]
                viewer.sync()

                expected_real_time = start_time + motiontime * self.sim_dt
                time_to_sleep = expected_real_time - time.time()
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

                if motiontime % int(1.0 / self.config.sim_dt) == 0:
                    real_time_now = time.time() - start_time
                    actual_hz = motiontime / real_time_now if real_time_now > 0 else 0.0
                    vx_cur, vy_cur, vyaw_cur = gamepad.get_velocity()
                    print(f"[Gamepad] vx={vx_cur:+.2f} m/s | vy={vy_cur:+.2f} | yaw={vyaw_cur:+.2f} rad/s")
                    print(f"[Sim Time]: t={motiontime * self.sim_dt:.1f}s, Base height: {self.data.qpos[2]:.3f}m")
                    print(f"[Real Time]: t={real_time_now:.1f}s, Actual Hz: {actual_hz:.2f} Hz")


def resolve_config(script_dir, name):
    if os.path.exists(name):
        return name
    path = Path(name)
    if path.parts and path.parts[0] == "config":
        return str(G1_DEPLOY_DIR / path)
    return os.path.join(script_dir, "config", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="g1_attention.onnx")
    parser.add_argument("--config", type=str, default="g1_attention.yaml")
    parser.add_argument("--input", choices=["gamepad", "keyboard", "const"], default="gamepad")
    parser.add_argument("--gamepad_type", type=str, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--debug_policy", action="store_true")
    parser.add_argument("--debug_interval", type=float, default=1.0)
    parser.add_argument("--debug_gamepad", action="store_true")
    parser.add_argument("--gamepad_debug_interval", type=float, default=None)
    parser.add_argument("--show_rays", action="store_true")
    parser.add_argument("--show_ray_lines", action="store_true")
    parser.add_argument("--ray_subsample", type=int, default=6)
    parser.add_argument("--const_vx", type=float, default=0.0,
                        help="With --input const: fixed forward velocity command (m/s).")
    parser.add_argument("--const_vy", type=float, default=0.0,
                        help="With --input const: fixed lateral velocity command (m/s).")
    parser.add_argument("--const_vyaw", type=float, default=0.0,
                        help="With --input const: fixed yaw-rate command (rad/s).")
    parser.add_argument("--const_warmup", type=float, default=2.0,
                        help="With --input const: seconds to hold zero command before ramping up.")
    args = parser.parse_args()

    script_dir = str(G1_DEPLOY_DIR)
    config_path = resolve_config(script_dir, args.config)
    controller = Sim2SimAttentionController(
        config_path,
        args.model,
        debug_policy=args.debug_policy,
        debug_interval=args.debug_interval,
        show_rays=args.show_rays,
        show_ray_lines=args.show_ray_lines,
        ray_subsample=args.ray_subsample,
    )

    if args.check:
        terrain_map = controller.compute_terrain_scan()
        print(
            f"[Check] Config/model validation passed. Initial terrain_map={terrain_map.shape}, "
            f"valid_hits={np.mean(controller._last_scan_cache['valid']) * 100.0:.1f}%"
        )
        return

    cfg = controller.config
    if args.input == "keyboard":
        gamepad = KeyboardController(
            vx_range=cfg.command_range["lin_vel_x"],
            vy_range=cfg.command_range["lin_vel_y"],
            vyaw_range=cfg.command_range["ang_vel_z"],
        )
    elif args.input == "const":
        vx_lo, vx_hi = cfg.command_range["lin_vel_x"]
        vy_lo, vy_hi = cfg.command_range["lin_vel_y"]
        vyaw_lo, vyaw_hi = cfg.command_range["ang_vel_z"]
        # Reject OOD vx=0 silently — clamp to the trained min so the user can
        # still pass --const_vx 0 without exiting the training distribution.
        gamepad = ConstantController(
            vx=float(np.clip(args.const_vx, vx_lo, vx_hi)),
            vy=float(np.clip(args.const_vy, vy_lo, vy_hi)),
            vyaw=float(np.clip(args.const_vyaw, vyaw_lo, vyaw_hi)),
            warmup_s=args.const_warmup,
        )
        print(
            f"[const input] vx={gamepad.target[0]:+.2f} vy={gamepad.target[1]:+.2f} "
            f"vyaw={gamepad.target[2]:+.2f}  warmup={gamepad.warmup_s:.1f}s ramp={gamepad.ramp_s:.1f}s"
        )
    else:
        gamepad_type = args.gamepad_type or getattr(cfg, "gamepad_type_sim2sim", getattr(cfg, "gamepad_type", "gamesir"))
        gamepad = create_gamepad_controller(
            gamepad_type,
            vx_range=cfg.command_range["lin_vel_x"],
            vy_range=cfg.command_range["lin_vel_y"],
            vyaw_range=cfg.command_range["ang_vel_z"],
            btn_start=getattr(cfg, "gamepad_btn_start", None),
            btn_rb=getattr(cfg, "gamepad_btn_rb", None),
            btn_a=getattr(cfg, "gamepad_btn_a", None),
            axis_left_x=getattr(cfg, "gamepad_axis_left_x", None),
            axis_left_y=getattr(cfg, "gamepad_axis_left_y", None),
            axis_right_x=getattr(cfg, "gamepad_axis_right_x", None),
            deadzone=getattr(cfg, "gamepad_deadzone", 0.05),
            command_slew_rate=getattr(cfg, "gamepad_slew_rate", (2.0, 4.0, 3.0)),
            debug=args.debug_gamepad or bool(getattr(cfg, "gamepad_debug", False)),
            debug_interval=(
                args.gamepad_debug_interval
                if args.gamepad_debug_interval is not None
                else getattr(cfg, "gamepad_debug_interval", 0.5)
            ),
        )

    try:
        gamepad.start()
        controller.run(gamepad)
    finally:
        gamepad.stop()


if __name__ == "__main__":
    main()
