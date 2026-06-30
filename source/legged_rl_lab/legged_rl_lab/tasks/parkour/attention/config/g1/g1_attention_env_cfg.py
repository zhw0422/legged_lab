from __future__ import annotations

import math
from typing import Any

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from legged_rl_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_rl_lab.tasks.parkour.attention.attention_env_cfg import (
    AttentionBaseEnvCfg,
    AttentionCommandsCfg,
    AttentionCurriculumCfg,
    AttentionEventCfg,
    AttentionEnvCfgMixin,
    AttentionObservationsCfg,
    AttentionPolicyCfg,
    AttentionRewardsCfg,
    AttentionSceneCfg,
    AttentionTerminationsCfg,
    attention_height_scanner_cfg,
)
from legged_rl_lab.terrains import (
    HfAlternateColumnStakesTerrainCfg,
    HfConcentricGapTerrainCfg,
    HfDoubleColumnStakesTerrainCfg,
    HfStonesBridgeTerrainCfg,
    HfRadialPlankBridgeTerrainCfg
)
import legged_rl_lab.tasks.parkour.attention.mdp as mdp


# =============================================================================
# FINETUNE toggle — set True to switch to AME-style finetune (stake-heavy
# terrain mix + stronger gait-shaping rewards + no randomization).
# =============================================================================
FINETUNE = False


# =============================================================================
# Robot body/contact constants
# =============================================================================
G1_CONTACT_LINKS: tuple[str, ...] = (
    "torso_link",
    "left_hip_roll_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "left_ankle_pitch_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "right_ankle_pitch_link",
    "waist_yaw_link",
)
G1_FOOT_SENSORS: tuple[str, ...] = ("foot_height_scanner_L", "foot_height_scanner_R")
G1_FOOT_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
G1_HEIGHT_SCANNER_PRIM_PATH = "{ENV_REGEX_NS}/Robot/torso_link"
G1_FOOT_GRID_PATTERN = patterns.GridPatternCfg(resolution=0.1, size=(0.2, 0.2), ordering="xy")
G1_FOOT_RAY_OFFSET = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0))


# =============================================================================
# Config classes — Scene, Observations, Commands, Events, Rewards, etc.
# =============================================================================


@configclass
class G1AttentionSceneCfg(AttentionSceneCfg):
    height_scanner = attention_height_scanner_cfg(G1_HEIGHT_SCANNER_PRIM_PATH)
    foot_height_scanner_L = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=G1_FOOT_RAY_OFFSET,
        ray_alignment="yaw",
        pattern_cfg=G1_FOOT_GRID_PATTERN,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    foot_height_scanner_R = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=G1_FOOT_RAY_OFFSET,
        ray_alignment="yaw",
        pattern_cfg=G1_FOOT_GRID_PATTERN,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class G1AttentionCriticCfg(ObsGroup):
    """Critic proprio group — mirrors AME's CriticCfg.

    Same six terms as the actor `AttentionPolicyCfg` (without per-term Unoise),
    plus `base_lin_vel` as the single privileged signal. All hand-engineered
    DR / contact / foot-state observations have been removed so the critic
    matches the AME convention 1-to-1; the actor stays identical so the ONNX
    export is unaffected.
    """

    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        scale=(1.0, 1.0, 0.25),
    )
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    actions = ObsTerm(func=mdp.last_action, scale=0.1)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    # Must stay last — AttentionTerrainModel slices the flattened obs by
    # treating the trailing `length*width*coord_dim` entries as the map scan.
    terrain_map = ObsTerm(
        func=mdp.elevation_map,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
    )

    def __post_init__(self):
        self.history_length = 1
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1AttentionObservationsCfg(AttentionObservationsCfg):
    policy: AttentionPolicyCfg = AttentionPolicyCfg()
    critic: G1AttentionCriticCfg = G1AttentionCriticCfg()


@configclass
class G1AttentionCommandsCfg(AttentionCommandsCfg):
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class G1AttentionEventCfg(AttentionEventCfg):
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class G1AttentionRewardsCfg(AttentionRewardsCfg):
    # Disable conflicting/redundant terms from base class
    lin_vel_z = None              # Penalty would prevent going up stairs
    ang_vel_xy = None             # Re-defined below
    orientation = None            # Re-defined below as flat_orientation_l2
    dof_power = None              # Replaced with dof_torques_l2
    dof_acc = None                # Re-defined below
    action_rate = None            # Re-defined below
    action_smoothness = None      # AME doesn't use this
    dof_pos_limits = None         # Re-defined below

    # -- task
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    alive = RewTerm(func=mdp.alive, weight=1.0)
    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,       # AME weight (was 1.5)
        params={"command_name": "base_velocity", "std": 0.25},
    )
    tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    # -- penalties
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)       # AME weight (was -0.1)
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_ankle_roll_link).*"),
            "threshold": 1.0,
        },
    )
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    dof_pos_limits_pen = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    dof_torques_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)   # AME weight (was -1.0)
    # -- style
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.6,
        },
    )
    feet_air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.1,       # AME weight (was -0.7)
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,       # AME weight (was -2.0)
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-1.0,       # AME weight (was -5.0)
        params={"asset_cfg": SceneEntityCfg("robot", body_names=list(G1_FOOT_BODIES)), "threshold": 0.2},
    )
    # -- coordination
    joint_coordination = RewTerm(
        func=mdp.joint_coordination_rel,
        weight=-0.2,       # AME weight (was -0.1)
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "coord_joints": [
                ["left_hip_pitch_joint", "right_shoulder_pitch_joint"],
                ["right_hip_pitch_joint", "left_shoulder_pitch_joint"],
            ],
            "coord_signs": [
                [1.0, 1.0],
                [1.0, 1.0],
            ],
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,       # AME weight (was -0.1)
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
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"]),
        },
    )
    # Pull ankles back to the default pose to discourage the heel-strike / toe-up
    # gait that emerges when only `feet_slide` + `feet_air_time` constrain the
    # ankle. Start at -0.3; raise to -0.5 if the policy still curls toes upward,
    # lower to -0.15 if ankles become too stiff to clear stairs.
    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            ),
        },
    )


@configclass
class G1AttentionTerminationsCfg(AttentionTerminationsCfg):
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        },
    )
    gravity_tilt = DoneTerm(func=mdp.gravity_too_horizontal, params={"threshold": -0.1})


@configclass
class G1AttentionCurriculumCfg(AttentionCurriculumCfg):
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# =============================================================================
# Terrain configuration helpers
# =============================================================================

def _configure_terrain_common(terrain_generator: Any) -> None:
    """Apply grid/scale settings shared across train / finetune / play."""
    terrain_generator.size = (8.0, 8.0)
    terrain_generator.horizontal_scale = 0.05
    terrain_generator.vertical_scale = 0.005
    terrain_generator.slope_threshold = 0.75
    terrain_generator.use_cache = False


def configure_g1_attention_train_terrain(terrain_generator: Any) -> None:
    """Regular training terrain — matches AME ROUGH_TERRAINS_CFG exactly."""
    _configure_terrain_common(terrain_generator)
    terrain_generator.border_width = 50.0
    terrain_generator.num_rows = 10
    terrain_generator.num_cols = 20
    terrain_generator.sub_terrains = {
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.2),
            step_width=0.3, platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.2),
            step_width=0.3, platform_width=3.0, border_width=1.0, holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02,
            downsampled_scale=0.1, border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
        ),
        "hf_steppingstones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2, stone_height_max=0.05, stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.05, 0.25), platform_width=2.0,
            holes_depth=-2.0, border_width=0.25,
        ),
        "hf_gaps": HfConcentricGapTerrainCfg(
            proportion=0.2, gap_width_range=(0.1, 0.5), platform_width=2.0, border_width=0.25,
            gap_depth=-2.0, ground_width_range=(0.5, 0.5), ground_height_max=0.025,
        ),
    }


def configure_g1_attention_finetune_terrain(terrain_generator: Any) -> None:
    """AME-style finetune terrain — matches FINETUNE_ROUGH_TERRAINS_CFG exactly."""
    _configure_terrain_common(terrain_generator)
    terrain_generator.border_width = 50.0
    terrain_generator.num_rows = 10
    terrain_generator.num_cols = 20
    terrain_generator.sub_terrains = {
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.25),
            step_width=0.3, platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.05, 0.25),
            step_width=0.3, platform_width=3.0, border_width=1.0, holes=False,
        ),
        "stakes1": HfDoubleColumnStakesTerrainCfg(
            proportion=0.1, stake_height_max=0.03, stake_side_range=(0.20, 0.40),
            stake_gap_range=(0.1, 0.3), column_gap_range=(0.1, 0.1), column_jitter=0.0,
            holes_depth=-2.0, platform_width=2.0, border_width=0.25,
        ),
        "stakes2": HfAlternateColumnStakesTerrainCfg(
            proportion=0.2, stake_height_max=0.03, stake_side_range=(0.20, 0.40),
            stake_gap_range=(0.05, 0.15), column_gap_range=(0.0, 0.2), column_jitter=0.0,
            holes_depth=-2.0, platform_width=2.0, border_width=0.25,
        ),
        "stakes3": HfAlternateColumnStakesTerrainCfg(
            proportion=0.2, stake_height_max=0.03, stake_side_range=(0.20, 0.40),
            stake_gap_range=(0.05, 0.25), column_gap_range=(0.3, 0.2), column_jitter=0.0,
            holes_depth=-2.0, platform_width=2.0, border_width=0.25,
        ),
        "hf_gaps": HfConcentricGapTerrainCfg(
            proportion=0.1, gap_width_range=(0.2, 0.6), platform_width=2.0, border_width=0.25,
            gap_depth=-2.0, ground_width_range=(0.5, 0.5), ground_height_max=0.03,
        ),
        "stones_bridge": HfStonesBridgeTerrainCfg(
            proportion=0.1, platform_width=2.0, border_width=0.25, holes_depth=-2.0,
            stone_height_max=0.03, stone_width_range=(0.25, 0.35), stone_distance_range=(0.3, 0.5),
            stone_length_range=(0.6, 1.0), stone_lateral_distance_range=(0.0, 0.0),
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.1, rail_height_range=(0.25, 0.05), rail_thickness_range=(0.1, 0.3),
            platform_width=2.0,
        ),
    }


def configure_g1_attention_play_terrain(terrain_generator: Any) -> None:
    """Play/evaluation terrain: single tile, fixed difficulty, pick one preset."""
    _configure_terrain_common(terrain_generator)
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1
    terrain_generator.curriculum = False
    terrain_generator.difficulty_range = (1.0, 1.0)
    terrain_generator.border_width = 3.0

    # ── Pick ONE terrain preset below ──────────────────────────────────────

    # [A] Alternate column stakes (current default, fixed params)
    # terrain_generator.sub_terrains = {
    #     "stakes": HfAlternateColumnStakesTerrainCfg(
    #         proportion=0.5, stake_height_max=0.0, stake_side_range=(0.2, 0.2),
    #         stake_gap_range=(0.3, 0.3), column_gap_range=(0.3, 0.3), column_jitter=0.0,
    #         holes_depth=-2.0, platform_width=2.0, border_width=0.25,
    #     ),
    # }

    # [B] Double column stakes
    # terrain_generator.sub_terrains = {
    #     "double_column_stakes": HfDoubleColumnStakesTerrainCfg(
    #         proportion=0.5, stake_height_max=0.0, stake_side_range=(0.2, 0.2),
    #         stake_gap_range=(0.3, 0.3), column_gap_range=(0.3, 0.3), column_jitter=0.0,
    #         holes_depth=-2.0, platform_width=2.0, border_width=0.25,
    #     ),
    # }

    # [C] Stepping stones
    # terrain_generator.sub_terrains = {
    #     "steppingstones": terrain_gen.HfSteppingStonesTerrainCfg(
    #         proportion=1.0, stone_height_max=0.05, stone_width_range=(0.25, 0.5),
    #         stone_distance_range=(0.05, 0.25), platform_width=2.0,
    #         holes_depth=-2.0, border_width=0.25,
    #     ),
    # }

    # [D] Stone bridge
    # terrain_generator.sub_terrains = {
    #     "stones_bridge": HfStonesBridgeTerrainCfg(
    #         proportion=0.5, stone_height_max=0.05, stone_width_range=(0.3, 0.5),
    #         stone_length_range=(0.3, 0.5), stone_distance_range=(0.1, 0.25),
    #         stone_lateral_distance_range=(0.05, 0.15), holes_depth=-2.0,
    #         platform_width=2.0, border_width=0.25,
    #     ),
    # }

    # [E] Concentric gaps
    terrain_generator.sub_terrains = {
        "concentric_gaps": HfConcentricGapTerrainCfg(
            proportion=0.25, gap_width_range=(0.5, 0.5), ground_width_range=(0.5, 0.5),
            ground_height_max=0.025, gap_depth=-2.0, platform_width=2.0, border_width=0.25,
        ),
    }

    # [F] Stairs up
    # terrain_generator.sub_terrains = {
    #     "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
    #         proportion=0.125, step_height_range=(0.10, 0.16), step_width=0.30,
    #         platform_width=2.0, border_width=0.4, holes=False,
    #     ),
    # }

    # [G] Stairs down
    # terrain_generator.sub_terrains = {
    #     "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
    #         proportion=0.125, step_height_range=(0.10, 0.25), step_width=0.30,
    #         platform_width=2.0, border_width=0.4, holes=False,
    #     ),
    # }
    # [H] Radial plank bridge — center platform with narrow planks radiating outward like spokes
    # terrain_generator.sub_terrains = {
    #     "radial_plank_bridge": HfRadialPlankBridgeTerrainCfg(
    #         proportion=1.0,
    #         plank_width_range=(0.19, 0.19),
    #         plank_height_max=0.03,
    #         num_arms=4,
    #         arm_length_range=None,
    #         holes_depth=-2.0,
    #         platform_width=1.0,
    #         platform_shape="square",
    #         border_width=0.25,
    #     ),
    # }


def configure_g1_attention_sensors(scene: Any, update_period: float) -> None:
    for sensor_name in G1_FOOT_SENSORS:
        sensor = getattr(scene, sensor_name, None)
        if sensor is not None:
            sensor.update_period = update_period


# =============================================================================
# Main environment config
# =============================================================================


@configclass
class G1AttentionEnvCfg(AttentionEnvCfgMixin, AttentionBaseEnvCfg):
    scene: G1AttentionSceneCfg = G1AttentionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1AttentionObservationsCfg = G1AttentionObservationsCfg()
    rewards: G1AttentionRewardsCfg = G1AttentionRewardsCfg()
    commands: G1AttentionCommandsCfg = G1AttentionCommandsCfg()
    events: G1AttentionEventCfg = G1AttentionEventCfg()
    terminations: G1AttentionTerminationsCfg = G1AttentionTerminationsCfg()
    curriculum: G1AttentionCurriculumCfg = G1AttentionCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        configure_g1_attention_sensors(self.scene, self.decimation * self.sim.dt)
        self.configure_attention_train(G1_HEIGHT_SCANNER_PRIM_PATH)

        # Two-stage curriculum, mirrors AME `velocity_env_cfg_29dof.py`:
        #   * Backbone  (FINETUNE=False) — open command range, soft gait rewards,
        #     domain randomization OFF + observation noise OFF.
        #   * Finetune  (FINETUNE=True)  — locked forward heading, stronger
        #     gait shaping + harder terrain, domain randomization ON + obs noise ON.
        #
        # NB. Default values declared on AttentionPolicyCfg / G1AttentionEventCfg
        # already correspond to the FINETUNE setting (push/mass/COM enabled,
        # Unoise enabled). _apply_backbone() strips those down; _apply_finetune()
        # only tweaks reward weights and terrain mix.
        if FINETUNE:
            self._apply_finetune()
        else:
            self._apply_backbone()

    def _apply_backbone(self) -> None:
        """Backbone stage — disable domain randomization and observation noise.

        Mirrors the ``FINETUNE=False`` branch of AME's velocity_env_cfg_29dof.
        Use this stage to learn a robust velocity-tracking policy first; the
        gait-shaping rewards stay at their declared default weights.
        """
        # ── Disable interval / startup randomization ────────────────────────
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None

        # ── Disable observation noise on every actor obs term ───────────────
        policy_cfg = self.observations.policy
        policy_cfg.enable_corruption = False
        for term_name in ("base_ang_vel", "projected_gravity", "velocity_commands",
                          "joint_pos", "joint_vel", "actions"):
            term = getattr(policy_cfg, term_name, None)
            if term is not None:
                term.noise = None

        # Terrain map now lives at the tail of the actor proprio group.
        if hasattr(policy_cfg, "terrain_map"):
            policy_cfg.terrain_map.params["noise"] = False

    def _apply_finetune(self) -> None:
        """Apply AME-style finetune overrides.

        Mirrors the ``FINETUNE=True`` branch of AME's velocity_env_cfg_29dof:
            1. Stronger gait-shaping reward weights
            2. Stake-heavy terrain mix (set in configure_attention_train)
            3. Lock forward heading + zero out spawn pose/velocity noise

        Domain randomization (push/mass/COM) and obs noise stay **on**, matching
        AME finetune. Use ``_apply_backbone()`` instead if you want them off.
        """
        # ── Reward weights ──────────────────────────
        self.rewards.tracking_lin_vel.weight = 2.0
        self.rewards.tracking_ang_vel.weight = 3.0
        self.rewards.dof_torques_limits.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.flat_orientation.weight = -5.0
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time_variance.weight = -2.0
        self.rewards.feet_slide.weight = -0.3
        self.rewards.feet_stumble.weight = -5.0
        self.rewards.feet_too_near.weight = -5.0
        self.rewards.joint_coordination.weight = -0.5
        self.rewards.joint_deviation_arms.weight = -0.3
        self.rewards.joint_deviation_waists.weight = -1.0
        self.rewards.joint_deviation_ankle.weight = -0.5

        # Fix reset position to terrain center (AME finetune zeroes spawn noise)
        if hasattr(self.events, "reset_base") and self.events.reset_base is not None:
            self.events.reset_base.params = {
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
            }

        # Fix heading to forward-only
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

    def configure_attention_train(self, height_scanner_prim_path: str) -> None:
        AttentionEnvCfgMixin.configure_attention_train(self, height_scanner_prim_path)

        self.scene.terrain.max_init_terrain_level = 5
        if self.scene.terrain.terrain_generator is not None:
            if FINETUNE:
                configure_g1_attention_finetune_terrain(self.scene.terrain.terrain_generator)
            else:
                configure_g1_attention_train_terrain(self.scene.terrain.terrain_generator)

    def configure_attention_play(self) -> None:
        AttentionEnvCfgMixin.configure_attention_play(self)

        self.scene.num_envs = 14
        self.scene.terrain.max_init_terrain_level = None

        # Spawn at terrain center with no positional jitter
        if hasattr(self.events, "reset_base") and self.events.reset_base is not None:
            self.events.reset_base.params["pose_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
            }

        # Fixed forward velocity with heading lock — mirrors AME play behavior
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        if self.scene.terrain.terrain_generator is not None:
            configure_g1_attention_play_terrain(self.scene.terrain.terrain_generator)


@configclass
class G1AttentionEnvCfg_PLAY(G1AttentionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.configure_attention_play()