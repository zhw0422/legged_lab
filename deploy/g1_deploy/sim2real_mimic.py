#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common.command_helper import create_damping_cmd
from common.remote_controller import KeyMap
from config import Config
from sim2real_walk import Controller
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.utils.crc import CRC


THIS_DIR = Path(__file__).resolve().parent


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_inv_np(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_apply_np(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_vec = q_wxyz[1:4]
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return (v + 2.0 * (q_wxyz[0] * uv + uuv)).astype(np.float32)


def quat_rotate_inverse_np(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply_np(quat_inv_np(q_wxyz), v)


def subtract_frame_transforms_np(
    pos_a: np.ndarray,
    quat_a_wxyz: np.ndarray,
    pos_b: np.ndarray,
    quat_b_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_inv = quat_inv_np(quat_a_wxyz)
    rel_pos = quat_apply_np(q_inv, pos_b - pos_a)
    rel_quat = quat_mul_np(q_inv, quat_b_wxyz)
    return rel_pos.astype(np.float32), rel_quat.astype(np.float32)


def matrix_from_quat_np(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def build_tracking_obs(
    ref_joint_pos: np.ndarray,
    ref_joint_vel: np.ndarray,
    motion_anchor_pos_b: np.ndarray,
    motion_anchor_ori_b_6: np.ndarray,
    base_lin_vel_b: np.ndarray,
    base_ang_vel_b: np.ndarray,
    joint_pos_rel: np.ndarray,
    joint_vel: np.ndarray,
    last_action: np.ndarray,
    include_state_estimation: bool,
) -> np.ndarray:
    parts = [ref_joint_pos, ref_joint_vel]
    if include_state_estimation:
        parts.append(motion_anchor_pos_b)
    parts.append(motion_anchor_ori_b_6)
    if include_state_estimation:
        parts.append(base_lin_vel_b)
    parts.extend([base_ang_vel_b, joint_pos_rel, joint_vel, last_action])
    return np.concatenate(parts).astype(np.float32)


def resolve_config(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    return THIS_DIR / "config" / path


def resolve_model(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    return THIS_DIR / "exported_policy" / name


class MimicController(Controller):
    def __init__(
        self,
        flat_config: Config,
        mimic_config_path: Path,
        mimic_model_path: Path,
        net: str,
        domain_id: int = 0,
        debug_policy: bool = False,
    ) -> None:
        self.flat_config = flat_config
        self.mimic_config_path = mimic_config_path
        self.mimic_model_path = mimic_model_path
        self.active_policy = "flat"
        self._last_a = 0
        self._last_b = 0
        self.time_step = 0
        self.ref_joint_pos = None
        self.ref_joint_vel = None
        self.ref_body_pos_w = None
        self.ref_body_quat_w = None
        self.sport_state = unitree_go_msg_dds__SportModeState_()
        self.sport_state_received = False
        super().__init__(flat_config, net, domain_id=domain_id, debug_policy=debug_policy)

    def setup_extra_subscribers(self):
        self.sport_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sport_subscriber.Init(self.SportStateHandler, 10)

    def SportStateHandler(self, msg: SportModeState_):
        self.sport_state = msg
        self.sport_state_received = True

    def _load_policy_session(self, config: Config, model_path: Path) -> None:
        config.set_policy_path(str(model_path))
        self.config = config
        self.policy_path = Path(config.policy_path)
        self.policy_type = "onnx"
        self.policy = ort.InferenceSession(str(self.policy_path), providers=["CPUExecutionProvider"])
        self.policy_input_names = [inp.name for inp in self.policy.get_inputs()]
        self.policy_output_names = [out.name for out in self.policy.get_outputs()]

    def _apply_policy_gains(self) -> None:
        with self.cmd_lock:
            for i, motor_idx in enumerate(self.config.sdk2isaac_idx):
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[i])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[i])
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0

    def switch_to_flat(self) -> None:
        if self.active_policy == "flat":
            return
        print("[PolicySwitch] A pressed: switching to flat stabilize policy.")
        self._load_policy_session(self.flat_config, Path(self.flat_config.policy_path))
        self._apply_policy_gains()
        self.active_policy = "flat"
        self.first_run = True
        self.action[:] = 0.0
        self.current_obs[:] = 0.0
        self.current_obs_history[:] = 0.0
        self.control_start_time = time.time()

    def switch_to_mimic(self) -> None:
        if self.active_policy == "mimic":
            return
        print("[PolicySwitch] B pressed: switching to mimic / tracking policy.")
        mimic_config = Config(self.mimic_config_path)
        self._load_policy_session(mimic_config, self.mimic_model_path)
        self._apply_policy_gains()
        self.active_policy = "mimic"
        self.time_step = 0
        self.ref_joint_pos = None
        self.ref_joint_vel = None
        self.ref_body_pos_w = None
        self.ref_body_quat_w = None
        self.action[:] = 0.0
        self.control_start_time = time.time()
        self._infer_tracking_policy(np.zeros((1, self.config.num_obs), dtype=np.float32))

    def _update_policy_switch(self) -> None:
        a = self.remote_controller.button[KeyMap.A]
        b = self.remote_controller.button[KeyMap.B]
        if a == 1 and self._last_a == 0:
            self.switch_to_flat()
        if b == 1 and self._last_b == 0:
            self.switch_to_mimic()
        self._last_a = a
        self._last_b = b

    def _infer_tracking_policy(self, obs_batch):
        obs = obs_batch.reshape(1, -1).astype(np.float32)
        if obs.shape[1] != self.config.num_obs:
            obs = np.zeros((1, self.config.num_obs), dtype=np.float32)
        ts = np.array([[self.time_step]], dtype=np.float32)
        outputs = self.policy.run(self.policy_output_names, {"obs": obs, "time_step": ts})
        action = outputs[0].reshape(-1).astype(np.float32)
        if len(outputs) >= 5:
            self.ref_joint_pos = outputs[1].squeeze(0).astype(np.float32)
            self.ref_joint_vel = outputs[2].squeeze(0).astype(np.float32)
            self.ref_body_pos_w = outputs[3].squeeze(0).astype(np.float32)
            self.ref_body_quat_w = outputs[4].squeeze(0).astype(np.float32)
        return action

    def run(self):
        self._update_policy_switch()
        if self.active_policy == "flat":
            super().run()
            return

        for i in range(len(self.config.sdk2isaac_idx)):
            self.joint_pos[i] = self.low_state.motor_state[self.config.sdk2isaac_idx[i]].q
            self.joint_vel[i] = self.low_state.motor_state[self.config.sdk2isaac_idx[i]].dq

        if self.ref_joint_pos is None:
            self._infer_tracking_policy(np.zeros((1, self.config.num_obs), dtype=np.float32))

        joint_pos_rel = (self.joint_pos - self.config.default_joint_pos).astype(np.float32)
        joint_vel = self.joint_vel.astype(np.float32)
        last_action = self.action.astype(np.float32)

        base_quat_wxyz = np.asarray(self.low_state.imu_state.quaternion, dtype=np.float32)
        base_ang_vel_b = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        base_lin_vel_w = (
            np.asarray(self.sport_state.velocity, dtype=np.float32)
            if self.sport_state_received
            else np.zeros(3, dtype=np.float32)
        )
        base_lin_vel_b = quat_rotate_inverse_np(base_quat_wxyz, base_lin_vel_w)
        robot_anchor_pos = (
            np.asarray(self.sport_state.position, dtype=np.float32)
            if self.sport_state_received
            else np.zeros(3, dtype=np.float32)
        )
        robot_anchor_quat = (
            np.asarray(self.sport_state.imu_state.quaternion, dtype=np.float32)
            if self.sport_state_received
            else base_quat_wxyz
        )

        anchor_idx = self.config.body_names.index(self.config.anchor_body_name)
        motion_anchor_pos = self.ref_body_pos_w[anchor_idx].astype(np.float32)
        motion_anchor_quat = self.ref_body_quat_w[anchor_idx].astype(np.float32)
        anchor_pos_b, anchor_quat_b = subtract_frame_transforms_np(
            robot_anchor_pos, robot_anchor_quat, motion_anchor_pos, motion_anchor_quat
        )
        anchor_mat = matrix_from_quat_np(anchor_quat_b)
        anchor_ori_b_6 = anchor_mat[:, :2].reshape(-1).astype(np.float32)

        obs = build_tracking_obs(
            self.ref_joint_pos.astype(np.float32),
            self.ref_joint_vel.astype(np.float32),
            anchor_pos_b.astype(np.float32),
            anchor_ori_b_6,
            base_lin_vel_b,
            base_ang_vel_b,
            joint_pos_rel,
            joint_vel,
            last_action,
            include_state_estimation=bool(self.config.include_state_estimation),
        )

        raw_action = self._infer_tracking_policy(obs.reshape(1, -1))
        self.action = np.clip(raw_action, -self.config.action_clip, self.config.action_clip)
        target_dof_pos = self.config.default_joint_pos + self.action * self.config.action_scale

        ramp = 1.0
        self.print_policy_debug(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), base_ang_vel_b, joint_pos_rel, joint_vel, target_dof_pos, raw_action, ramp)
        with self.cmd_lock:
            for i, motor_idx in enumerate(self.config.sdk2isaac_idx):
                self.low_cmd.motor_cmd[motor_idx].q = float(target_dof_pos[i])

        self.time_step += 1
        total_steps = getattr(self.config, "motion_total_steps", None)
        if total_steps and self.time_step >= total_steps:
            self.time_step = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="SDK2 sim2real controller for G1 mimic/tracking policies.")
    parser.add_argument("--net", type=str, default="enp108s0", help="network interface")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain id, use 1 for local SDK2 MuJoCo bridge")
    parser.add_argument("--config_path", type=str, default="config/g1_mimic.yaml", help="configuration file path")
    parser.add_argument("--model", type=str, default="g1_dance.onnx", help="ONNX model filename or path")
    parser.add_argument("--flat_config_path", type=str, default="config/g1_walk.yaml", help="flat stabilization config")
    parser.add_argument("--flat_model", type=str, default="g1_flat_1.onnx", help="flat stabilization ONNX model")
    parser.add_argument("--debug_policy", action="store_true", help="Print policy observation/action ranges.")
    args = parser.parse_args()

    flat_config = Config(resolve_config(args.flat_config_path))
    flat_config.set_policy_path(str(resolve_model(args.flat_model)))
    mimic_config_path = resolve_config(args.config_path)
    mimic_model_path = resolve_model(args.model)
    controller = MimicController(
        flat_config,
        mimic_config_path,
        mimic_model_path,
        args.net,
        args.domain_id,
        debug_policy=args.debug_policy,
    )
    print("SDK2 mimic starts in flat stabilize policy. Press B to enter mimic / tracking mode; press A to return flat.")

    try:
        while True:
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("Select Button detected, Exit!")
                break
            time.sleep(0.01)
    finally:
        controller.run_thread.Wait()
        controller.publish_thread.Wait()
        with controller.cmd_lock:
            controller.low_cmd = create_damping_cmd(controller.low_cmd)
            controller.low_cmd.crc = CRC().Crc(controller.low_cmd)
            controller.lowcmd_publisher_.Write(controller.low_cmd)
        time.sleep(0.2)
        print("Exit")


if __name__ == "__main__":
    main()
