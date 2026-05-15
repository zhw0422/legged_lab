# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""
Procedural Humanoid Environment Configuration.

Flat and rough terrain environment configurations for procedurally generated
humanoid (biped) robots with varying torso/leg dimensions.
All classes inherit from ``CrossEmbodiedLocomotionEnvCfg`` which provides the
cross-embodied specific defaults (events, rewards, observations, actions).
"""

import numpy as np

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns

import isaaclab.sim as sim_utils

import legged_rl_lab.tasks.locomotion.cross_emboided.mdp as mdp

from legged_rl_lab.tasks.locomotion.cross_emboided.cross_emboided_env_cfg import (
    CrossEmbodiedLocomotionEnvCfg,
)
from legged_rl_lab.tasks.locomotion.cross_emboided.mdp import morphology_params

from metamorphosis.asset_cfg import ProceduralBipedCfg  # isort: skip


PROCEDURAL_HUMANOID_CFG = ArticulationCfg(
    spawn=ProceduralBipedCfg(
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        torso_link_length_range=(0.10, 0.16),
        torso_link_width_range=(0.18, 0.26),
        torso_link_height_range=(0.2, 0.8),
        pelvis_height_range=(0.05, 0.08),
        hip_spacing_range=(0.16, 0.24),
        hip_pitch_link_length_range=(0.03, 0.06),
        hip_pitch_link_radius_range=(0.02, 0.04),
        hip_roll_link_length_range=(0.03, 0.06),
        hip_roll_link_radius_range=(0.02, 0.04),
        hip_pitch_link_initroll_range=(0.0, 0.2),  # ~0-11.5°, keeps hip_pitch axis near world-Y
        leg_length_range=(0.5, 0.7),
        shin_ratio_range=(0.85, 1.15),
        head_radius_range=(0.08, 0.12),
        arm_length_range=(0.22, 0.36),
        forearm_ratio_range=(0.85, 1.10),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_pitch_joint": -0.2,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.4,
            ".*_ankle_pitch_joint": -0.2,
            ".*_ankle_roll_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0,
            ".*_wrist_roll_joint": 0.0,
            ".*_wrist_pitch_joint": 0.0,
            ".*_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=300.0,
            velocity_limit=30.0,
            stiffness=80.0,
            damping=2.0,
            armature=0.01,
            friction=0.05,
        ),
    },
)


@configclass
class ProceduralHumanoidFlatEnvCfg(CrossEmbodiedLocomotionEnvCfg):
    """Flat terrain environment for procedurally generated humanoids."""

    def __post_init__(self):
        super().__post_init__()

        # ====Scene Cfg====
        self.scene.robot = PROCEDURAL_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ====Terrain Cfg====
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        # Morphology parameters observation (procedural-specific)
        self.observations.policy.morphology_params = ObsTerm(func=morphology_params)

        # ====Event Cfg====
        self.events.randomize_push_robot = None
        self.events.randomize_com_positions = None
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)

        # ====Rewards: velocity tracking (match G1 yaw-frame functions)====
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_world_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )

        # ====Rewards: joint penalties (scope and tune to match G1)====
        self.rewards.joint_torques_l2.weight = -2.0e-6
        self.rewards.joint_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.rewards.joint_vel_l2.weight = -0.001
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.joint_pos_limits.params = {
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            )
        }

        # ====Rewards: humanoid-specific body names====
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(hip|knee|pelvis).*"
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time_positive_biped,
            weight=1.5,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "threshold": 0.5,
            },
        )
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.feet_slide.params["asset_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.feet_slide.weight = -1.0
        self.rewards.feet_clearance = RewTerm(
            func=mdp.foot_clearance_reward_humanoid,
            weight=2.0,
            params={
                "std": 0.05,
                "tanh_mult": 2.0,
                "target_height": 0.10,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
                "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            },
        )

        # ====Rewards: anti-hop penalties====
        self.rewards.double_flight = RewTerm(
            func=mdp.double_flight_penalty,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "command_name": "base_velocity",
            },
        )
        self.rewards.lin_vel_z_l2 = RewTerm(
            func=mdp.lin_vel_z_l2,
            weight=-1.5,
        )

        # ====Rewards: posture / deviation====
        # hip_roll 劈叉专项强惩罚：防止机器人靠髋外展走"侧向劈叉步"
        self.rewards.hip_roll_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-15.0,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_roll_joint"],
            )},
        )
        # hip_pitch 前后劈叉惩罚：防止两腿前后永久劈叉; 正常迈步偏差约0.3rad可接受
        self.rewards.hip_pitch_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch_joint"],
            )},
        )
        self.rewards.joint_deviation_legs = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_yaw_joint", ".*_knee_joint"],
            )},
        )
        self.rewards.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.3,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
            )},
        )
        # base 高度下限惩罚：防止深蹲/劈叉导致躯干过低
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-5.0,
            params={"target_height": 0.65},
        )

        # ====Rewards: gait (enforce alternating foot contact)====
        self.rewards.gait = RewTerm(
            func=mdp.feet_gait,
            weight=2.0,
            params={
                "period": 0.8,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            },
        )

        # ====Termination Cfg====
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "(torso_link|pelvis)"


@configclass
class ProceduralHumanoidFlatEnvCfg_PLAY(ProceduralHumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        self.observations.policy.enable_corruption = False
        self.events.randomize_push_robot = None


@configclass
class ProceduralHumanoidRoughEnvCfg(CrossEmbodiedLocomotionEnvCfg):
    """Rough terrain environment for procedurally generated humanoids."""

    def __post_init__(self):
        super().__post_init__()

        # ====Scene Cfg====
        self.scene.robot = PROCEDURAL_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Foot scanners (single-ray, straight down): used by feet_clearance reward
        _foot_scanner_cfg = RayCasterCfg(
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            ray_alignment="world",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.0, 0.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.foot_scanner_l = _foot_scanner_cfg.replace(
            prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link"
        )
        self.scene.foot_scanner_r = _foot_scanner_cfg.replace(
            prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link"
        )
        self.scene.foot_scanner_l.update_period = self.decimation * self.sim.dt
        self.scene.foot_scanner_r.update_period = self.decimation * self.sim.dt

        # Morphology parameters observation (procedural-specific)
        self.observations.policy.morphology_params = ObsTerm(func=morphology_params)
        self.observations.policy.height_scan.scale = 1.0

        # ====Event Cfg====
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)

        # ====Rewards: velocity tracking (match G1 yaw-frame functions)====
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_world_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.5},
        )

        # ====Rewards: joint penalties (scope and tune to match G1)====
        self.rewards.joint_torques_l2.weight = -2.0e-6
        self.rewards.joint_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.rewards.joint_vel_l2.weight = -0.001
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.joint_pos_limits.params = {
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            )
        }

        # ====Rewards: humanoid-specific body names====
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(hip|knee|pelvis).*"
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time_positive_biped,
            weight=1.5,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "threshold": 0.5,
            },
        )
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.feet_slide.params["asset_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.feet_slide.weight = -1.0
        self.rewards.feet_clearance = RewTerm(
            func=mdp.foot_clearance_reward_humanoid,
            weight=2.0,
            params={
                "std": 0.05,
                "tanh_mult": 2.0,
                "target_height": 0.10,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
                "foot_scanner_cfgs": [
                    SceneEntityCfg("foot_scanner_l"),
                    SceneEntityCfg("foot_scanner_r"),
                ],
                "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            },
        )

        # ====Rewards: anti-hop penalties====
        self.rewards.double_flight = RewTerm(
            func=mdp.double_flight_penalty,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "command_name": "base_velocity",
            },
        )
        self.rewards.lin_vel_z_l2 = RewTerm(
            func=mdp.lin_vel_z_l2,
            weight=-1.5,
        )

        # ====Rewards: posture / deviation====
        # hip_roll 劈叉专项强惩罚：防止机器人靠髋外展走"侧向劈叉步"
        self.rewards.hip_roll_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-15.0,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_roll_joint"],
            )},
        )
        # hip_pitch 前后劈叉惩罚：防止两腿前后永久劈叉; 正常迈步偏差约0.3rad可接受
        self.rewards.hip_pitch_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch_joint"],
            )},
        )
        self.rewards.joint_deviation_legs = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_yaw_joint", ".*_knee_joint"],
            )},
        )
        self.rewards.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.3,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
            )},
        )
        # base 高度下限惩罚：防止深蹲/劈叉导致躯干过低
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-5.0,
            params={"target_height": 0.65},
        )

        # ====Rewards: gait (enforce alternating foot contact)====
        self.rewards.gait = RewTerm(
            func=mdp.feet_gait,
            weight=2.0,
            params={
                "period": 0.8,
                "offset": [0.0, 0.5],
                "threshold": 0.55,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            },
        )

        # ====Rewards: symmetry====
        self.rewards.joint_symmetry_l2.weight = -0.1
        self.rewards.joint_symmetry_l2.params["mirror_joints"] = [["left_.*_joint", "right_.*_joint"]]
        self.rewards.action_symmetry_l2.weight = -0.05
        self.rewards.action_symmetry_l2.params["mirror_joints"] = [["left_.*_joint", "right_.*_joint"]]

        # ====Termination Cfg====
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "(torso_link|pelvis)"


@configclass
class ProceduralHumanoidRoughEnvCfg_PLAY(ProceduralHumanoidRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        self.observations.policy.enable_corruption = False
        self.events.randomize_push_robot = None
