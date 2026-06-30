# Copyright (c) 2022-2025, The unitree_rl_gym Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from unitree_rl_gym Project (BSD-3-Clause license)

import sys
import time
import os
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import onnxruntime as ort


THIS_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = THIS_DIR.parent
SDK2_PYTHON_DIR = THIS_DIR / "unitree_sdk2_python"
if str(SDK2_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(SDK2_PYTHON_DIR))


def _bootstrap_cyclonedds_home() -> None:
    """Ensure CYCLONEDDS_HOME points to a valid local install before SDK import."""
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

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
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

from common.command_helper import (
    MotorMode,
    create_damping_cmd,
    init_cmd_go,
    init_cmd_hg,
)
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


class Controller:
    def __init__(self, config: Config, net: str, domain_id: int = 0, debug_policy: bool = False) -> None:

        ChannelFactoryInitialize(domain_id, net)

        self.first_run = True
        self.config = config
        self.debug_policy = debug_policy
        self._last_debug_time = 0.0
        self.control_start_time = 0.0
        self.remote_controller = RemoteController()

        self.policy_path = Path(config.policy_path)
        if self.policy_path.suffix == ".onnx":
            self.policy_type = "onnx"
            self.policy = ort.InferenceSession(str(self.policy_path), providers=["CPUExecutionProvider"])
            self.policy_input_names = [inp.name for inp in self.policy.get_inputs()]
            self.policy_output_names = [out.name for out in self.policy.get_outputs()]
        else:
            self.policy_type = "torchscript"
            self.policy = torch.jit.load(config.policy_path).eval()
        self.run_thread = RecurrentThread(interval=self.config.control_dt, target=self.run)  # 100Hz/50Hz
        self.publish_thread = RecurrentThread(interval=1 / 500, target=self.publish)
        self.cmd_lock = Lock()

        self.joint_pos = np.zeros(config.num_actions, dtype=np.float32)
        self.joint_vel = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)

        self.frame_obs_dim = 9 + 3 * config.num_actions
        self.policy_obs_dim = self.frame_obs_dim * int(config.history_length)
        self.current_obs = np.zeros(self.frame_obs_dim, dtype=np.float32)
        self.current_obs_history = np.zeros((config.history_length, self.frame_obs_dim), dtype=np.float32)

        self.clip_min_command = np.array(
            [
                self.config.command_range["lin_vel_x"][0],
                self.config.command_range["lin_vel_y"][0],
                self.config.command_range["ang_vel_z"][0],
            ],
            dtype=np.float32,
        )
        self.clip_max_command = np.array(
            [
                self.config.command_range["lin_vel_x"][1],
                self.config.command_range["lin_vel_y"][1],
                self.config.command_range["ang_vel_z"][1],
            ],
            dtype=np.float32,
        )

        for _ in range(50):
            obs = self.build_policy_input().reshape(1, -1).astype(np.float32)
            self.infer_policy(obs)

        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR  #用串联PR模式

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        else:
            raise ValueError("Invalid msg_type")

        self.setup_extra_subscribers()
        self.wait_for_low_state()

        if config.msg_type == "hg":
            self.low_cmd = init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            self.low_cmd = init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.publish_thread.Start()
        self.wait_for_start()

        self.move_to_default_pos()
        self.wait_for_control()

        print("Start Control!")
        self.control_start_time = time.time()
        self.run_thread.Start()

    def LowStateHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def setup_extra_subscribers(self):
        pass

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
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.sdk2isaac_idx
        dof_size = len(dof_idx)

        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        for i in range(num_step):
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            alpha = i / num_step
            with self.cmd_lock:
                for j in range(dof_size):
                    motor_idx = dof_idx[j]
                    target_pos = self.config.default_joint_pos[j]
                    self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                    self.low_cmd.motor_cmd[motor_idx].dq = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[j]
                    self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[j]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
            time.sleep(self.config.control_dt)

    def wait_for_control(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal to Start Control...")
        while self.remote_controller.button[KeyMap.A] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)

    def run(self):
        for i in range(len(self.config.sdk2isaac_idx)):
            self.joint_pos[i] = self.low_state.motor_state[self.config.sdk2isaac_idx[i]].q
            self.joint_vel[i] = self.low_state.motor_state[self.config.sdk2isaac_idx[i]].dq

        quat = self.low_state.imu_state.quaternion
        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)

        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.torso_idx].q
            waist_yaw_omega = self.low_state.motor_state[self.config.torso_idx].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel.reshape(1, 3)
            )
            ang_vel = np.asarray(ang_vel, dtype=np.float32).reshape(3)

        gravity_orientation = get_gravity_orientation(quat)
        joint_pos = (self.joint_pos - self.config.default_joint_pos) * self.config.dof_pos_scale
        joint_vel = self.joint_vel * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        command_raw = np.array(
            [self.remote_controller.ly, -self.remote_controller.lx, -self.remote_controller.rx], dtype=np.float32
        )
        command_raw *= self.config.command_scale
        command_raw = np.clip(command_raw, self.clip_min_command, self.clip_max_command)
        command = self.update_command(command_raw)

        num_actions = self.config.num_actions
        self.current_obs[:3] = ang_vel
        self.current_obs[3:6] = gravity_orientation
        self.current_obs[6:9] = command
        self.current_obs[9 : 9 + num_actions] = joint_pos
        self.current_obs[9 + num_actions : 9 + num_actions * 2] = joint_vel
        self.current_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action

        if self.first_run:
            self.current_obs_history[:] = self.current_obs.reshape(1, -1)
            self.first_run = False
        else:
            self.current_obs_history = np.concatenate(
                (self.current_obs_history[1:], self.current_obs.reshape(1, -1)), axis=0
            )

        # Group-major reorganization
        # [omega×H, gravity×H, cmd×H, pos×H, vel×H, action×H], H=history_length
        obs_input = self.build_policy_input()
        obs_batch = obs_input[np.newaxis, :].astype(np.float32)

        raw_action = self.infer_policy(obs_batch)
        self.action = np.clip(raw_action, -self.config.action_clip, self.config.action_clip)

        target_dof_pos = self.config.default_joint_pos + self.action * self.config.action_scale
        ramp = 1.0
        if self.config.policy_ramp_time > 0.0:
            ramp = np.clip((time.time() - self.control_start_time) / self.config.policy_ramp_time, 0.0, 1.0)
            target_dof_pos = self.config.default_joint_pos + ramp * (target_dof_pos - self.config.default_joint_pos)
        self.print_policy_debug(command, gravity_orientation, ang_vel, joint_pos, joint_vel, target_dof_pos, raw_action, ramp)
        with self.cmd_lock:
            for i in range(len(self.config.sdk2isaac_idx)):
                self.low_cmd.motor_cmd[self.config.sdk2isaac_idx[i]].q = target_dof_pos[i]

    def build_policy_input(self) -> np.ndarray:
        obs_arr = self.current_obs_history
        n = self.config.num_actions
        return np.concatenate([
            obs_arr[:, 0:3].reshape(-1),          # omega x H frames
            obs_arr[:, 3:6].reshape(-1),          # gravity x H frames
            obs_arr[:, 6:9].reshape(-1),          # cmd x H frames
            obs_arr[:, 9:9+n].reshape(-1),        # joint pos x H frames
            obs_arr[:, 9+n:9+2*n].reshape(-1),    # joint vel x H frames
            obs_arr[:, 9+2*n:9+3*n].reshape(-1),  # last action x H frames
        ]).astype(np.float32)

    def print_policy_debug(self, command, gravity_orientation, ang_vel, joint_pos, joint_vel, target_dof_pos, raw_action, ramp):
        if not self.debug_policy or time.time() - self._last_debug_time < 0.5:
            return

        action_abs = np.abs(self.action)
        max_action_idx = int(np.argmax(action_abs))
        clipped = int(np.count_nonzero(np.abs(raw_action - self.action) > 1e-5))
        target_abs = np.abs(target_dof_pos - self.config.default_joint_pos)
        max_target_idx = int(np.argmax(target_abs))
        print(
            "[PolicyDebug] "
            f"cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}] "
            f"grav=[{gravity_orientation[0]:+.2f},{gravity_orientation[1]:+.2f},{gravity_orientation[2]:+.2f}] "
            f"ang=[{ang_vel[0]:+.2f},{ang_vel[1]:+.2f},{ang_vel[2]:+.2f}] "
            f"q_rel=[{joint_pos.min():+.2f},{joint_pos.max():+.2f}] "
            f"dq=[{joint_vel.min():+.2f},{joint_vel.max():+.2f}] "
            f"ramp={ramp:.2f} "
            f"action=[{self.action.min():+.2f},{self.action.max():+.2f}] clipped={clipped}/{self.config.num_actions} "
            f"max_action_idx={max_action_idx}:{self.action[max_action_idx]:+.2f} "
            f"target_delta_max={max_target_idx}:{target_dof_pos[max_target_idx] - self.config.default_joint_pos[max_target_idx]:+.2f}"
        )
        self._last_debug_time = time.time()

    def update_command(self, command_raw: np.ndarray) -> np.ndarray:
        command_target = command_raw.copy()
        if self.config.command_deadband > 0.0:
            command_target[np.abs(command_target) < self.config.command_deadband] = 0.0

        max_delta = self.config.command_rate_limit * self.config.control_dt
        delta = np.clip(command_target - self.command, -max_delta, max_delta)
        command_limited = self.command + delta

        if self.config.command_smoothing_tau > 0.0:
            alpha = self.config.control_dt / (self.config.command_smoothing_tau + self.config.control_dt)
            self.command = self.command + alpha * (command_limited - self.command)
        else:
            self.command = command_limited
        return self.command.copy()

    def infer_policy(self, obs_batch):
        if self.policy_type == "onnx":
            outputs = self.policy.run(
                self.policy_output_names,
                {self.policy_input_names[0]: obs_batch.astype(np.float32)},
            )
            return outputs[0].flatten().astype(np.float32)

        with torch.no_grad():
            action_tensor = self.policy(torch.from_numpy(obs_batch.astype(np.float32)))
            if isinstance(action_tensor, tuple):
                action_tensor = action_tensor[0]
            return action_tensor.cpu().numpy().flatten().astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="enp108s0", help="network interface")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain id, use 1 for local unitree_mujoco simulation")
    parser.add_argument("--config_path", type=str, default="config/g1_walk.yaml", help="configuration file path")
    parser.add_argument("--debug_policy", action="store_true", help="Print policy observation/action ranges during control.")
    args = parser.parse_args()

    config = Config(resolve_config(args.config_path))
    controller = Controller(config, args.net, args.domain_id, debug_policy=args.debug_policy)

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
