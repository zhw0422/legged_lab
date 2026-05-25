# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from legged_rl_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAMPRoughEnvCfg
from legged_rl_lab import LEGGED_RL_LAB_ROOT_DIR

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

import os


@configclass
class UnitreeGo2AMPRoughEnvCfg(LocomotionAMPRoughEnvCfg):
    """Unitree Go2 AMP environment on rough terrain."""

    foot_link_name = ".*_foot"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # observations
        self.observations.policy.history_length = 5
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.critic.history_length = 5

        # action
        self.actions.joint_pos.scale = 0.25

        # ------------------------------- Event -------------------------------
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_com.params["asset_cfg"].body_names = "base"

        # ==================== Rewards Configuration ====================
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.joint_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.joint_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "base"

        # AMP: reference motion data path
        self.amp_motion_files = os.path.join(
            LEGGED_RL_LAB_ROOT_DIR, "data", "motions", "go2"
        )


@configclass
class UnitreeGo2AMPFlatEnvCfg(UnitreeGo2AMPRoughEnvCfg):
    """Unitree Go2 AMP environment on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        # No terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # No height scanner
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        # No terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class UnitreeGo2AMPRoughEnvCfg_PLAY(UnitreeGo2AMPRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable randomization
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
