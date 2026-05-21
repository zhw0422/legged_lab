# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP environment configurations for Unitree G1 humanoid robot."""

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_rl_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAMPRoughEnvCfg
from legged_rl_lab import LEGGED_RL_LAB_ROOT_DIR
import legged_rl_lab.tasks.locomotion.amp.mdp as mdp

##
# Pre-defined configs
##
from legged_rl_lab.assets.unitree import UNITREE_G1_29DOF_CFG  # isort: skip

@configclass
class UnitreeG1AMPFlatEnvCfg(LocomotionAMPRoughEnvCfg):
    """Unitree G1 humanoid AMP environment on flat terrain."""

    base_link_name = "torso_link"
    foot_link_name = ".*_ankle_roll_link"

    # Key bodies for AMP / critic obs.  Order matters — must match the
    # body indices in motion_loader's G1 profile and the reference AMP design.
    key_body_names = (
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_rubber_hand",
        "right_rubber_hand",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ----------------------------- Control / Physics Rate -----------------------------
        self.sim.dt = 1.0 / 150.0
        self.decimation = 5
        self.sim.render_interval = self.decimation

        # ----------------------------- Scene -----------------------------
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # No height scanner
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None

        # No terrain curriculum
        self.curriculum.terrain_levels = None

        # ----------------------------- Observations -----------------------------
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # Wire up key body names for AMP + critic.
        key_body_cfg = SceneEntityCfg(
            "robot", body_names=list(self.key_body_names), preserve_order=True
        )
        self.observations.amp.features.params["asset_cfg"] = key_body_cfg
        self.observations.critic.key_body_pos_b.params["asset_cfg"] = key_body_cfg

        # ----------------------------- Actions -----------------------------
        self.actions.joint_pos.scale = 0.25

        # ----------------------------- Events -----------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.push_robot = None

        # Reference State Initialization (RSI).  Spawns the robot from a
        # near-stable frame of the LAFAN1 walks (filtered by base velocity
        # and root height) so the policy has to learn to *move while not
        # falling* instead of converging on the trivial "stand in default
        # pose" local optimum.  legged_lab/TienKung-Lab both rely on RSI to
        # break this local optimum.
        #
        # Critical: RSI must run AFTER ``reset_base`` and
        # ``reset_robot_joints`` so its absolute joint state and root pose
        # overwrite their uniform-noise initialization.  IsaacLab's
        # EventManager applies events in the order they appear on the cfg
        # class — assigning here (after super().__post_init__()) appends to
        # the end.  We also disable the uniform joint scaling so the RSI
        # joint pose isn't perturbed by 0.5–1.5× scaling.
        from isaaclab.managers import EventTermCfg as EventTerm
        self.events.reset_robot_joints = None  # RSI provides absolute joint state
        self.events.reset_from_ref = EventTerm(
            func=mdp.reset_from_reference_motion,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "height_offset": 0.05,
                # ~12% of LAFAN1 walk frames satisfy these — about 5500
                # near-stable frames, plenty for RSI sampling diversity
                # without spawning mid-stride with one foot airborne.
                "max_lin_vel_xy": 1.5,
                "max_ang_vel": 1.5,
                "min_root_height": 0.60,
            },
        )

        # ----------------------------- Rewards (legged_lab G1 style) -----------------------------
        # Disable old "is_alive" — replaced by negative termination_penalty
        # which is much stronger (legged_lab uses -50 for terminations).
        self.rewards.is_alive.weight = 0.0

        # Velocity-tracking rewards.  Doubled from legged_lab's 1.0 — the
        # previous run (500 iter) converged to a "stand in place" local minimum
        # where track_lin_vel ≈ 0.34 but error_vel_xy ≈ 1.17 (i.e. policy just
        # ignores the command).  Stronger task gradient is needed to break out.
        self.rewards.track_lin_vel_xy_exp.weight = 8.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        # Root penalties (legged_lab values)
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        # legged_lab uses -1.0; we keep -1.0 too — combined with the strong
        # termination penalty, the policy learns to stay upright instead of
        # using flat_orientation as the only "don't fall" signal.
        self.rewards.flat_orientation_l2.weight = -2.0

        # Joint penalties (legged_lab values)
        self.rewards.joint_torques_l2.weight = -2.0e-6
        self.rewards.joint_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.rewards.joint_acc_l2.weight = -1.0e-7

        self.rewards.action_rate_l2.weight = -0.005

        # Contact rewards
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=self.foot_link_name
        )

        # Gait periodic rewards (TienKung-style) — explicit stepping cadence
        # prior to prevent the robot from just sliding its feet along the floor.
        # The phase offset (L: 0.38, R: 0.88) encodes an alternating
        # left/right walking pattern.
        from isaaclab.managers import RewardTermCfg as RewTerm
        self.rewards.gait_frc_perio = RewTerm(
            func=mdp.gait_feet_frc_perio,
            weight=1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
                "cycle": 0.85,
                "offset_l": 0.38,
                "offset_r": 0.88,
                "air_ratio": 0.38,
                "threshold": 500.0,
            },
        )
        self.rewards.gait_spd_perio = RewTerm(
            func=mdp.gait_feet_spd_perio,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=self.foot_link_name),
                "cycle": 0.85,
                "offset_l": 0.38,
                "offset_r": 0.88,
                "air_ratio": 0.38,
                "speed_threshold": 1.5,
            },
        )
        self.rewards.gait_support_perio = RewTerm(
            func=mdp.gait_feet_frc_support_perio,
            weight=0.6,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
                "cycle": 0.85,
                "offset_l": 0.38,
                "offset_r": 0.88,
                "air_ratio": 0.38,
                "threshold": 500.0,
            },
        )

        # Termination penalty (legged_lab signature reward — single biggest
        # reason their policies don't fall over).  Heavy negative reward each
        # time the episode terminates from anything except time_out.
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated, weight=-200.0
        )

        # Joint deviation penalties (TienKung-style) — keeps arms in natural
        # position (prevents drooping) and hips/waist from flailing.
        self.rewards.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_joint",
                        ".*_wrist_.*",
                    ],
                )
            },
        )
        self.rewards.joint_deviation_hip = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_hip_yaw_joint",
                        ".*_hip_roll_joint",
                        "waist_.*",
                    ],
                )
            },
        )

        # ----------------------------- Terminations (legged_lab G1 style) -----------------------------
        # legged_lab disables base contact for G1 — humanoids fall in many ways
        # not always involving torso contact.  Replace with height + orientation
        # checks which are more reliable for early termination.
        self.terminations.illegal_contact = None

        from isaaclab.managers import TerminationTermCfg as DoneTerm
        # Drop reference_deviation — legged_lab doesn't use it; the strong
        # termination_penalty + height/orientation checks already do this job.
        self.terminations.reference_deviation = None

        # Height-based termination: episode dies if base falls below 0.2m
        # (matches legged_lab default).  Note this is the **pelvis** height
        # for G1, which sits much lower than the head; standing height ≈ 0.8m,
        # so 0.2m means clearly fallen.
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.2},
        )
        # Orientation termination: episode dies if base tilts > 75° from
        # upright.  legged_lab uses 60° but their RSI + height_offset combo
        # gets the robot into much better starting poses than ours; until we
        # match that we keep this looser.
        import math as _math
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": _math.radians(75.0)},
        )

        # ----------------------------- Commands -----------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ----------------------------- AMP Motion Data -----------------------------
        # Default to one validated walk clip.  A directory path will load every
        # supported motion file under it recursively, so we keep the built-in
        # default narrow and predictable here.  For broader training, override
        # this from the CLI with ``--motion_file <walk_only_dir_or_npz>``.
        self.robot_type = "g1"
        self.amp_motion_files = os.path.join(
            LEGGED_RL_LAB_ROOT_DIR,
            "data", "motion", "LAFAN1_Retargeting_Dataset", "g1", "walk1_subject1.npz",
        )

@configclass
class UnitreeG1AMPFlatEnvCfg_PLAY(UnitreeG1AMPFlatEnvCfg):
    """Unitree G1 AMP environment for visualisation / play."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.lin_vel_cmd_levels = None
