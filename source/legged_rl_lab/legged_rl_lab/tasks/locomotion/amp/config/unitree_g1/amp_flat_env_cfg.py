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

    # AMP discriminator body set. Keep this order aligned with the motion loader profile.
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

        # ---------------------------------------------------------------------
        # MDP: Simulation / control timing
        # ---------------------------------------------------------------------
        self.sim.dt = 1.0 / 150.0
        self.decimation = 5
        self.sim.render_interval = self.decimation

        # ---------------------------------------------------------------------
        # MDP: Scene
        # ---------------------------------------------------------------------
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        # ---------------------------------------------------------------------
        # MDP: Observations
        # Policy obs:
        #   base_ang_vel, projected_gravity, command, joint_pos, joint_vel, action
        # Critic/AMP obs:
        #   body-level kinematic features in torso_link frame for imitation.
        # ---------------------------------------------------------------------
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

        # Critic privileged body pose terms: mdp.robot_body_pos_b / robot_body_ori_b.
        self.observations.critic.body_pos_b.params = {
            "anchor_cfg": anchor_body_cfg,
            "body_cfg": amp_body_cfg,
        }
        self.observations.critic.body_ori_b.params = {
            "anchor_cfg": anchor_body_cfg,
            "body_cfg": amp_body_cfg,
        }

        # AMP discriminator terms: mdp.robot_body_*_b.
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

        # ---------------------------------------------------------------------
        # MDP: Actions
        # JointPositionAction maps policy output to default_joint_pos + scale * action.
        # Arms use smaller scales to prevent shoulder_roll saturation and hip strikes.
        # ---------------------------------------------------------------------
        self.actions.joint_pos.scale = {
            ".*_hip_.*_joint": 0.25,
            ".*_knee_joint": 0.25,
            ".*_ankle_.*_joint": 0.25,
            "waist_.*_joint": 0.25,
            ".*_shoulder_.*_joint": 0.08,
            ".*_elbow_joint": 0.08,
            ".*_wrist_.*_joint": 0.05,
        }

        # ---------------------------------------------------------------------
        # MDP: Events
        # Startup: randomize torso mass/COM.
        # Reset: reset_from_reference_motion samples valid AMP reference frames.
        # Interval: push_robot is inherited from the base AMP config.
        # ---------------------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.base_com.params["asset_cfg"].body_names = [self.base_link_name]

        # Event term: mdp.reset_from_reference_motion.
        self.events.reset_from_motion.params.update({
            "max_lin_vel_xy": 2.5,
            "max_ang_vel": 1.5,
            "min_root_height": 0.60,
        })

        # ---------------------------------------------------------------------
        # MDP: Rewards
        # AMP style reward is computed by the discriminator; the terms below keep
        # command tracking, posture, contacts, and regularization well-behaved.
        # ---------------------------------------------------------------------
        self.rewards.is_alive.weight = 0.0

        # Reward group: command tracking at torso_link.
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.track_anchor_linear_velocity = RewTerm(
            func=mdp.track_anchor_linear_velocity,
            weight=3.0,
            params={
                "command_name": "base_velocity",
                "std": 0.5,
                "anchor_cfg": anchor_body_cfg,
            },
        )
        self.rewards.track_anchor_angular_velocity = RewTerm(
            func=mdp.track_anchor_angular_velocity,
            weight=1.5,
            params={
                "command_name": "base_velocity",
                "std": 0.5,
                "anchor_cfg": anchor_body_cfg,
            },
        )

        # Reward group: torso height, uprightness, and roll/pitch stability.
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
        self.rewards.torso_upright_orientation = RewTerm(
            func=mdp.body_upright_orientation,
            weight=2.0,
            params={
                "std": 0.25,
                "body_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
            },
        )

        # Reward group: action and joint regularization.
        self.rewards.joint_torques_l2 = None
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)

        _foot_sensor = SceneEntityCfg("contact_forces", body_names=self.foot_link_name)
        _foot_asset = SceneEntityCfg("robot", body_names=self.foot_link_name)

        # Reward group: foot contacts and anti-slip.
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time_positive_biped,
            weight=0.5,
            params={
                "sensor_cfg": _foot_sensor,
                "command_name": "base_velocity",
                "threshold": 0.45,
            },
        )
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

        # Reward group: foot swing clearance and double-flight suppression.
        self.rewards.feet_clearance = RewTerm(
            func=mdp.foot_clearance_reward_humanoid,
            weight=5.0,
            params={
                "std": 0.035,
                "tanh_mult": 4.0,
                "target_height": 0.12,
                "asset_cfg": _foot_asset,
            },
        )
        self.rewards.double_flight_penalty = RewTerm(
            func=mdp.double_flight_penalty,
            weight=-0.5,
            params={
                "sensor_cfg": _foot_sensor,
                "command_name": "base_velocity",
            },
        )

        # Reward group: joint posture shaping.
        # Arms are kept near default because shoulder_roll saturation drives hands into hips.
        self.rewards.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-1.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_shoulder_.*_joint",
                        ".*_elbow_joint",
                        ".*_wrist_.*",
                    ],
                )
            },
        )

        # Reward group: soft self-collision avoidance.
        # Penalize elbows/wrists before they collide with pelvis/hips/thighs.
        self.rewards.arm_lower_body_collision_avoidance = RewTerm(
            func=mdp.body_pair_distance_violation,
            weight=-20.0,
            params={
                "source_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        ".*_elbow_link",
                        ".*_wrist_.*_link",
                    ],
                ),
                "target_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "pelvis",
                        ".*_hip_roll_link",
                        ".*_hip_yaw_link",
                        ".*_knee_link",
                    ],
                ),
                "min_distance": 0.18,
            },
        )

        # Waist stays close to default to reduce forward torso lean.
        self.rewards.joint_deviation_waist = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-3.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["waist.*"],
                )
            },
        )

        # Hip roll/yaw shaping prevents excessive leg splay.
        self.rewards.joint_deviation_legs = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"],
                )
            },
        )

        # Stand-still shaping only activates near zero command.
        self.rewards.stand_still = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            },
        )

        # Termination penalty is kept large because AMP recovery data can otherwise
        # make falls look stylistically acceptable.
        self.rewards.is_terminated = RewTerm(
            func=mdp.is_terminated, weight=-200.0
        )

        # ---------------------------------------------------------------------
        # MDP: Terminations
        # Disable inherited illegal/reference terminations, then use broad
        # height/orientation guards so recovery motions still have room to learn.
        # ---------------------------------------------------------------------
        self.terminations.illegal_contact = None
        self.terminations.reference_deviation = None

        # Termination term: mdp.root_height_below_minimum.
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.2},
        )

        # Termination term: mdp.bad_orientation.
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(75.0)},
        )

        # ---------------------------------------------------------------------
        # MDP: Commands
        # Forward-only x velocity is used unless backward reference motions are added.
        # More standing envs help zero-command stability under AMP style pressure.
        # ---------------------------------------------------------------------
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-3.081, 2.902)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ---------------------------------------------------------------------
        # MDP: AMP motion data
        # Can point to either one .npz file or a directory containing many .npz clips.
        # ---------------------------------------------------------------------
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

        # MDP: Scene overrides for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # MDP: Disable training-only noise/randomization for play.
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum.lin_vel_cmd_levels = None
