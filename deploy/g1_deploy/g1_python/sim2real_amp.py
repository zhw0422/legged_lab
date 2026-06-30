#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort


THIS_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = THIS_DIR.parent
SDK2_PYTHON_DIR = THIS_DIR / "unitree_sdk2_python"
if str(SDK2_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(SDK2_PYTHON_DIR))


def _bootstrap_cyclonedds_home() -> None:
    env_home = os.environ.get("CYCLONEDDS_HOME", "")
    if env_home and (Path(env_home) / "lib" / "libddsc.so").exists():
        return

    candidates = [
        THIS_DIR / "cyclonedds" / "install",
        DEPLOY_DIR / "cyclonedds" / "install",
    ]
    for candidate in candidates:
        if (candidate / "lib" / "libddsc.so").exists():
            os.environ["CYCLONEDDS_HOME"] = str(candidate)
            print(f"[CycloneDDS] Using CYCLONEDDS_HOME={candidate}")
            return


_bootstrap_cyclonedds_home()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from common.command_helper import MotorMode, create_damping_cmd, init_cmd_go, init_cmd_hg
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from config import Config


def resolve_config(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if candidate.parts and candidate.parts[0] == "config":
        return DEPLOY_DIR / candidate
    return DEPLOY_DIR / "config" / path


def resolve_model(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    if candidate.parts and candidate.parts[0] == "exported_policy":
        return DEPLOY_DIR / candidate
    return DEPLOY_DIR / "exported_policy" / name


def build_amp_obs(
    base_ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    commands: np.ndarray,
    joint_pos_rel: np.ndarray,
    joint_vel: np.ndarray,
    last_action: np.ndarray,
    config: Config,
) -> np.ndarray:
    return np.concatenate(
        [
            base_ang_vel * config.ang_vel_scale,
            projected_gravity,
            commands,
            joint_pos_rel * config.dof_pos_scale,
            joint_vel * config.dof_vel_scale,
            last_action,
        ]
    ).astype(np.float32)


class AmpController:
    def __init__(self, config: Config, net: str, domain_id: int = 0, debug_policy: bool = False) -> None:
        ChannelFactoryInitialize(domain_id, net)

        self.config = config
        self.debug_policy = debug_policy
        self._last_debug_time = 0.0
        self.control_start_time = 0.0
        self.remote_controller = RemoteController()

        self.policy = ort.InferenceSession(str(config.policy_path), providers=["CPUExecutionProvider"])
        self.policy_input_names = [inp.name for inp in self.policy.get_inputs()]
        self.policy_output_names = [out.name for out in self.policy.get_outputs()]

        self.run_thread = RecurrentThread(interval=self.config.control_dt, target=self.run)
        self.publish_thread = RecurrentThread(interval=1 / 500, target=self.publish)
        self.cmd_lock = Lock()

        self.joint_pos = np.zeros(config.num_actions, dtype=np.float32)
        self.joint_vel = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)

        self.frame_obs_dim = 9 + 3 * config.num_actions
        self.current_obs = np.zeros(self.frame_obs_dim, dtype=np.float32)
        self.current_obs_history = np.zeros((int(config.history_length), self.frame_obs_dim), dtype=np.float32)
        self.first_run = True

        self.command_min = np.array(
            [
                config.command_range["lin_vel_x"][0],
                config.command_range["lin_vel_y"][0],
                config.command_range["ang_vel_z"][0],
            ],
            dtype=np.float32,
        )
        self.command_max = np.array(
            [
                config.command_range["lin_vel_x"][1],
                config.command_range["lin_vel_y"][1],
                config.command_range["ang_vel_z"][1],
            ],
            dtype=np.float32,
        )
        self.command_slew_rate = np.asarray(
            getattr(config, "gamepad_slew_rate", config.command_rate_limit),
            dtype=np.float32,
        )

        self._validate_dimensions()

        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        else:
            raise ValueError("Invalid msg_type")

        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        self.wait_for_low_state()
        if config.msg_type == "hg":
            self.low_cmd = init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        else:
            self.low_cmd = init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.publish_thread.Start()
        self.wait_for_start()
        self.move_to_default_pos()
        self.wait_for_control()

        print("Start AMP Control!")
        self.control_start_time = time.time()
        self.run_thread.Start()

    def _validate_dimensions(self) -> None:
        expected_obs_dim = self.frame_obs_dim * int(self.config.history_length)
        input_shape = self.policy.get_inputs()[0].shape
        output_shape = self.policy.get_outputs()[0].shape
        input_dim = input_shape[-1] if isinstance(input_shape[-1], int) else None
        output_dim = output_shape[-1] if isinstance(output_shape[-1], int) else None
        if self.config.num_obs != expected_obs_dim:
            raise ValueError(f"num_obs={self.config.num_obs}, expected {expected_obs_dim}.")
        if input_dim is not None and input_dim != expected_obs_dim:
            raise ValueError(f"ONNX input dim={input_dim}, expected {expected_obs_dim}.")
        if output_dim is not None and output_dim != self.config.num_actions:
            raise ValueError(f"ONNX output dim={output_dim}, expected {self.config.num_actions}.")

    def LowStateHandler(self, msg):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def publish(self):
        with self.cmd_lock:
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)

    def stop(self):
        print("Select Button detected, Exit!")
        self.publish_thread.Wait()
        with self.cmd_lock:
            self.low_cmd = create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)
        time.sleep(0.2)
        sys.exit(0)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        self.mode_machine_ = self.low_state.mode_machine
        print("Successfully connected to the robot.")

    def wait_for_start(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal to move to default pos...")
        while self.remote_controller.button[KeyMap.start] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)
        init_dof_pos = np.zeros(self.config.num_actions, dtype=np.float32)
        for i, motor_idx in enumerate(self.config.sdk2isaac_idx):
            init_dof_pos[i] = self.low_state.motor_state[motor_idx].q

        for i in range(num_step):
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            alpha = i / num_step
            with self.cmd_lock:
                for j, motor_idx in enumerate(self.config.sdk2isaac_idx):
                    self.low_cmd.motor_cmd[motor_idx].q = float(
                        init_dof_pos[j] * (1.0 - alpha) + self.config.default_joint_pos[j] * alpha
                    )
                    self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                    self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[j])
                    self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[j])
                    self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            time.sleep(self.config.control_dt)

    def wait_for_control(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal to Start AMP Control...")
        while self.remote_controller.button[KeyMap.A] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)

    def _remote_axes_to_command(self) -> np.ndarray:
        axes = np.array(
            [self.remote_controller.ly, -self.remote_controller.lx, -self.remote_controller.rx],
            dtype=np.float32,
        )
        if self.config.command_deadband > 0.0:
            axes[np.abs(axes) < self.config.command_deadband] = 0.0

        command = np.where(axes >= 0.0, axes * self.command_max, -(-axes) * np.abs(self.command_min))
        command *= self.config.command_scale
        return np.clip(command, self.command_min, self.command_max).astype(np.float32)

    def update_command(self, command_raw: np.ndarray) -> np.ndarray:
        max_delta = self.command_slew_rate * self.config.control_dt
        delta = np.clip(command_raw - self.command, -max_delta, max_delta)
        self.command = self.command + delta
        if self.config.command_smoothing_tau > 0.0:
            alpha = self.config.control_dt / (self.config.command_smoothing_tau + self.config.control_dt)
            self.command = self.command + alpha * (command_raw - self.command)
        return self.command.copy()

    def build_policy_input(self) -> np.ndarray:
        obs_arr = self.current_obs_history
        n = self.config.num_actions
        return np.concatenate(
            [
                obs_arr[:, 0:3].reshape(-1),
                obs_arr[:, 3:6].reshape(-1),
                obs_arr[:, 6:9].reshape(-1),
                obs_arr[:, 9 : 9 + n].reshape(-1),
                obs_arr[:, 9 + n : 9 + 2 * n].reshape(-1),
                obs_arr[:, 9 + 2 * n : 9 + 3 * n].reshape(-1),
            ]
        ).astype(np.float32)

    def infer_policy(self, obs_batch: np.ndarray) -> np.ndarray:
        outputs = self.policy.run(
            self.policy_output_names,
            {self.policy_input_names[0]: obs_batch.astype(np.float32)},
        )
        return outputs[0].flatten().astype(np.float32)

    def run(self):
        for i, motor_idx in enumerate(self.config.sdk2isaac_idx):
            self.joint_pos[i] = self.low_state.motor_state[motor_idx].q
            self.joint_vel[i] = self.low_state.motor_state[motor_idx].dq

        quat = self.low_state.imu_state.quaternion
        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.torso_idx].q
            waist_yaw_omega = self.low_state.motor_state[self.config.torso_idx].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel.reshape(1, 3),
            )
            ang_vel = np.asarray(ang_vel, dtype=np.float32).reshape(3)

        projected_gravity = get_gravity_orientation(quat)
        joint_pos_rel = self.joint_pos - self.config.default_joint_pos
        command = self.update_command(self._remote_axes_to_command())

        self.current_obs[:] = build_amp_obs(
            ang_vel,
            projected_gravity,
            command,
            joint_pos_rel,
            self.joint_vel,
            self.action,
            self.config,
        )
        if self.first_run:
            self.current_obs_history[:] = self.current_obs.reshape(1, -1)
            self.first_run = False
        else:
            self.current_obs_history = np.concatenate(
                (self.current_obs_history[1:], self.current_obs.reshape(1, -1)), axis=0
            )

        obs_batch = self.build_policy_input()[np.newaxis, :].astype(np.float32)
        raw_action = self.infer_policy(obs_batch)
        self.action = np.clip(raw_action, -self.config.action_clip, self.config.action_clip)
        target_dof_pos = self.config.default_joint_pos + self.action * self.config.action_scale

        self.print_policy_debug(command, projected_gravity, ang_vel, joint_pos_rel, self.joint_vel, target_dof_pos, raw_action)
        with self.cmd_lock:
            for i, motor_idx in enumerate(self.config.sdk2isaac_idx):
                self.low_cmd.motor_cmd[motor_idx].q = float(target_dof_pos[i])

    def print_policy_debug(self, command, projected_gravity, ang_vel, joint_pos_rel, joint_vel, target_dof_pos, raw_action):
        if not self.debug_policy or time.time() - self._last_debug_time < 0.5:
            return

        clipped = int(np.count_nonzero(np.abs(raw_action - self.action) > 1.0e-5))
        axes = np.array(
            [self.remote_controller.ly, -self.remote_controller.lx, -self.remote_controller.rx],
            dtype=np.float32,
        )
        print(
            "[AmpDebug] "
            f"axis=[{axes[0]:+.2f},{axes[1]:+.2f},{axes[2]:+.2f}] "
            f"cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}] "
            f"range_vx=[{self.command_min[0]:+.1f},{self.command_max[0]:+.1f}] "
            f"grav=[{projected_gravity[0]:+.2f},{projected_gravity[1]:+.2f},{projected_gravity[2]:+.2f}] "
            f"ang=[{ang_vel[0]:+.2f},{ang_vel[1]:+.2f},{ang_vel[2]:+.2f}] "
            f"q_rel=[{joint_pos_rel.min():+.2f},{joint_pos_rel.max():+.2f}] "
            f"dq=[{joint_vel.min():+.2f},{joint_vel.max():+.2f}] "
            f"action=[{self.action.min():+.2f},{self.action.max():+.2f}] clipped={clipped}/{self.config.num_actions} "
            f"target=[{target_dof_pos.min():+.2f},{target_dof_pos.max():+.2f}]"
        )
        self._last_debug_time = time.time()


def main() -> None:
    parser = argparse.ArgumentParser(description="SDK2 sim2real controller for G1 AMP velocity policies.")
    parser.add_argument("--net", type=str, default="enp108s0", help="network interface")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain id, use 1 for local SDK2 MuJoCo bridge")
    parser.add_argument("--config_path", type=str, default="config/g1_amp.yaml", help="configuration file path")
    parser.add_argument("--model", type=str, default="g1_walk.onnx", help="ONNX model filename or path")
    parser.add_argument("--debug_policy", action="store_true", help="Print policy observation/action ranges.")
    args = parser.parse_args()

    config = Config(resolve_config(args.config_path))
    config.set_policy_path(str(resolve_model(args.model)))
    controller = AmpController(config, args.net, args.domain_id, debug_policy=args.debug_policy)

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
