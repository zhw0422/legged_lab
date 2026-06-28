# Copyright (c) 2022-2025, The unitree_rl_gym Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from unitree_rl_gym Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"]

            self.sdk2isaac_idx = config["sdk2isaac_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_joint_pos = np.array(config["default_joint_pos"], dtype=np.float32)

            if "torso_idx" in config:
                self.torso_idx = config["torso_idx"]

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = np.array(config["action_scale"], dtype=np.float32)
            self.action_clip = config.get("action_clip", 100.0)
            self.policy_ramp_time = config.get("policy_ramp_time", 0.0)
            self.command_scale = np.array(config["command_scale"], dtype=np.float32)
            self.command_deadband = config.get("command_deadband", 0.0)
            self.command_smoothing_tau = config.get("command_smoothing_tau", 0.0)
            self.command_rate_limit = np.array(config.get("command_rate_limit", [100.0, 100.0, 100.0]), dtype=np.float32)
            self.gamepad_slew_rate = np.array(config.get("gamepad_slew_rate", self.command_rate_limit), dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            self.history_length = config["history_length"]
            self.command_range = config["command_range"]

            self.policy_type = config.get("policy_type", "flat")
            self.include_state_estimation = config.get("include_state_estimation", False)
            self.anchor_body_name = config.get("anchor_body_name", "")
            self.body_names = config.get("body_names", [])
            self.motion_total_steps = config.get("motion_total_steps", None)

    def set_policy_path(self, policy_path: str) -> None:
        self.policy_path = policy_path
