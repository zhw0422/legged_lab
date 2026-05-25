# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP environment configurations for Unitree G1 humanoid robot."""

import math
import os

from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from legged_rl_lab import LEGGED_RL_LAB_ROOT_DIR
import legged_rl_lab.tasks.locomotion.amp.mdp as mdp
from legged_rl_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAMPRoughEnvCfg

# Pre-defined configs
from legged_rl_lab.assets.unitree import UNITREE_G1_29DOF_CFG  # isort: skip


@configclass
class UnitreeG1AMPFlatEnvCfg(LocomotionAMPRoughEnvCfg):
    """Unitree G1 humanoid AMP environment on flat terrain."""

    base_link_name = "torso_link"
    foot_link_name = ".*_ankle_roll_link"

    # AMP key bodies
    key_body_names = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )

    def __post_init__(self):
        super().__post_init__()

        # Control
        self.sim.dt = 1.0 / 150.0
        self.decimation = 5
        self.sim.render_interval = self.decimation

        # Scene
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        # Observations
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.5, n_max=0.5)
        self.observations.policy.history_length = 4
        self.observations.critic.history_length = 4

        anchor_body_cfg = SceneEntityCfg(
            "robot", body_names=[self.base_link_name], preserve_order=True
        )
        amp_body_cfg = SceneEntityCfg(
            "robot", body_names=list(self.key_body_names), preserve_order=True
        )

        # observations: mdp.robot_body_pos_b / mdp.robot_body_ori_b
        self.observations.critic.body_pos_b.params = {
            "anchor_cfg": anchor_body_cfg,
            "body_cfg": amp_body_cfg,
        }
        self.observations.critic.body_ori_b.params = {
            "anchor_cfg": anchor_body_cfg,
            "body_cfg": amp_body_cfg,
        }

        # observations: mdp.robot_body_*_b
        self.observations.amp.body_pos_b.params = {
            "anchor_cfg": anchor_body_cfg,
            "body_cfg": amp_body_cfg,
        }
        self.observations.amp.body_ori_b.params = {
            "anchor_cfg": anchor_body_cfg,
            "body_cfg": amp_body_cfg,
        }
        self.observations.amp.body_lin_vel_b.params = {
            "anchor_cfg": anchor_body_cfg,
            "body_cfg": amp_body_cfg,
        }
        self.observations.amp.body_ang_vel_b.params = {
            "anchor_cfg": anchor_body_cfg,
            "body_cfg": amp_body_cfg,
        }

        # Actions
        self.actions.joint_pos.scale = 0.25

        # Events
        self.events.add_base_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.base_com.params["asset_cfg"].body_names = [self.base_link_name]

        # events: mdp.reset_from_reference_motion (G1-specific tuning of base RSI ranges)
        self.events.reset_from_motion.params.update({
            "max_lin_vel_xy": 1.5,
            "max_ang_vel": 1.5,
            "min_root_height": 0.60,
        })

        # Rewards
        self.rewards.is_alive.weight = 0.0

        # rewards: mdp.track_anchor_linear_velocity / mdp.track_anchor_angular_velocity
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.track_anchor_linear_velocity = RewTerm(
            func=mdp.track_anchor_linear_velocity,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "std": 1.0,
                "anchor_cfg": anchor_body_cfg,
            },
        )
        self.rewards.track_anchor_angular_velocity = RewTerm(
            func=mdp.track_anchor_angular_velocity,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "std": math.pi,
                "anchor_cfg": anchor_body_cfg,
            },
        )

        # rewards: mdp.track_root_height / mdp.body_ang_vel_xy_l2
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.track_root_height = RewTerm(
            func=mdp.track_root_height,
            weight=1.0,
            params={"std": 0.3, "target_height": 0.78},
        )
        self.rewards.body_ang_vel_xy_l2 = RewTerm(
            func=mdp.body_ang_vel_xy_l2,
            weight=0.5,
            params={
                "std": math.pi,
                "body_cfg": SceneEntityCfg("robot", body_names=["pelvis"]),
            },
        )

        # rewards: mdp.joint_acc_l2 / mdp.joint_pos_limits / mdp.action_rate_l2
        self.rewards.joint_torques_l2 = None
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
        self.rewards.action_rate_l2.weight = -0.01

        _foot_sensor = SceneEntityCfg("contact_forces", body_names=self.foot_link_name)
        _foot_asset = SceneEntityCfg("robot", body_names=self.foot_link_name)

        # rewards: mdp.feet_slip
        self.rewards.feet_air_time = None
        self.rewards.feet_slip = RewTerm(
            func=mdp.feet_slip,
            weight=-0.25,
            params={
                "sensor_cfg": _foot_sensor,
                "asset_cfg": _foot_asset,
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "contact_force_threshold": 1.0,
            },
        )

        # rewards: mdp.is_terminated
        self.rewards.is_terminated = RewTerm(
            func=mdp.is_terminated, weight=-200.0
        )

        # Terminations
        self.terminations.illegal_contact = None
        self.terminations.reference_deviation = None

        # terminations: mdp.root_height_below_minimum
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.2},
        )

        # terminations: mdp.bad_orientation
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(75.0)},
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # AMP motion data
        self.robot_type = "g1"
        self.amp_motion_files = os.path.join(
            LEGGED_RL_LAB_ROOT_DIR,
            "data", "motion", "LAFAN1_Retargeting_Dataset", "g1", "walk1_subject1.npz",
        )


@configclass
class UnitreeG1AMPFlatEnvCfg_PLAY(UnitreeG1AMPFlatEnvCfg):
    """Unitree G1 AMP environment for play."""

    def __post_init__(self):
        super().__post_init__()

        # Scene
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Randomization
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum.lin_vel_cmd_levels = None
