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

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ----------------------------- Control / Physics Rate -----------------------------
        # Align env step to LAFAN1 motion fps (30 Hz) so AMP (s, s') transitions
        # are time-matched between policy and expert.  With env at 50 Hz (base
        # default) vs motion at 30 Hz, the discriminator saw a 1.67× temporal
        # mismatch in joint/base velocities — a trivial free signal that kept
        # disc_accuracy≈1.0 regardless of pose quality.
        #
        # sim.dt=1/150, decimation=5  →  step_dt = 1/30  (exact 30 Hz)
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
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # AMP discriminator obs (legged_lab style):
        #   base_ang_vel(3) + joint_pos(29) + joint_vel(29) = 61 per frame,
        #   history_length=10 ⇒ disc input dim 610 per state, 1220 per pair.
        # No foot_positions, no base_lin_vel — see amp_env_cfg.AMPCfg docstring.
        # All three terms already point at the robot via default SceneEntityCfg.

        # ----------------------------- Actions -----------------------------
        self.actions.joint_pos.scale = 0.25

        # ----------------------------- Events -----------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.push_robot = None

        # RSI disabled for LAFAN1 walk1_subject1 — the clip contains no
        # standing frames, so RSI always spawns the robot mid-stride with one
        # foot airborne, and the still-random policy can't hold it up.  The
        # base_height / bad_orientation terminations then fire within ~1s,
        # leaving no room to learn.
        #
        # legged_lab uses RSI with datasets that include "Stand_to_Walk"
        # clips so the robot starts from an upright stance.  Re-enable once
        # we have such clips preprocessed; meanwhile the motion data is only
        # used as the AMP discriminator's expert source.
        self.events.reset_from_ref = None

        # ----------------------------- Rewards (legged_lab G1 style) -----------------------------
        # Disable old "is_alive" — replaced by negative termination_penalty
        # which is much stronger (legged_lab uses -50 for terminations).
        self.rewards.is_alive.weight = 0.0

        # Velocity-tracking rewards (legged_lab uses 1.0 + 1.0)
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        # Root penalties (legged_lab values)
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        # legged_lab uses -1.0; we keep -1.0 too — combined with the strong
        # termination penalty, the policy learns to stay upright instead of
        # using flat_orientation as the only "don't fall" signal.
        self.rewards.flat_orientation_l2.weight = -1.0

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

        # Termination penalty (legged_lab signature reward — single biggest
        # reason their policies don't fall over).  Heavy negative reward each
        # time the episode terminates from anything except time_out.
        from isaaclab.managers import RewardTermCfg as RewTerm
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated, weight=-50.0
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
        # Default: IsaacLab-replayed NPZ (walk1_subject1.npz) generated by
        # scripts/csv_to_npz.py.  The NPZ schema (keys: joint_pos, joint_vel,
        # body_pos_w, body_quat_w, body_names, …) provides foot body positions
        # in world frame, which MotionLoader transforms to base frame to match
        # the env's AMP obs layout.
        #
        # Must use NPZ (not raw CSV) — AMP features now include foot_positions
        # in base frame, which require FK data that CSV doesn't carry.  To
        # preprocess additional clips:
        #   python scripts/csv_to_npz.py -f path/to/walkN.csv \
        #       --input_fps 30 --output_fps 30 --headless
        self.robot_type = "g1"
        self.amp_motion_files = os.path.join(
            LEGGED_RL_LAB_ROOT_DIR,
            "data", "motion", "LAFAN1_Retargeting_Dataset", "g1_walks_npz", "walk1_subject1.npz",
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
