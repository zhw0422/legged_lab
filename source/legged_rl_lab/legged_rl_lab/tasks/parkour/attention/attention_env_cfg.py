from __future__ import annotations

import copy
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from legged_rl_lab.terrains import AME_PARKOUR_TERRAINS_CFG, HfAlternateColumnStakesTerrainCfg
import legged_rl_lab.tasks.parkour.attention.mdp as mdp


ATTENTION_GRID_RESOLUTION = 0.05
ATTENTION_GRID_SIZE = (1.6, 1.0)
ATTENTION_MAP_SCAN_DIM = (33, 21, 3)
# AME-style single ObsGroup per role: the height-scan tensor is appended to
# the end of the actor / critic proprio group, and `AttentionTerrainModel`
# carves it back out using ``obs[:, -map_scan_size:]``.
ATTENTION_OBS_GROUPS = {
    "actor": ["policy"],
    "critic": ["critic"],
}


def attention_height_scanner_cfg(prim_path: str) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=prim_path,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=ATTENTION_GRID_RESOLUTION, size=ATTENTION_GRID_SIZE),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class AttentionSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=AME_PARKOUR_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    height_scanner = attention_height_scanner_cfg("{ENV_REGEX_NS}/Robot/base")
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class AttentionCommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.2, 1.2),
            heading=(-math.pi, math.pi),
        ),
        limit_ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.2, 1.2),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class AttentionActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class AttentionEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class AttentionTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class AttentionCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={"reward_term_name": "tracking_lin_vel"},
    )


@configclass
class AttentionRewardsCfg:
    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_heading_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.2), "y_error_weight": 2.0},
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_power = RewTerm(
        func=mdp.dof_power_l1,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)


@configclass
class AttentionPolicyCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        scale=(1.0, 1.0, 0.25),
    )
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), scale=0.05)
    actions = ObsTerm(func=mdp.last_action, scale=0.1)
    # Must stay last — AttentionTerrainModel splits the flattened obs by
    # treating the trailing `length*width*coord_dim` entries as the map scan.
    terrain_map = ObsTerm(
        func=mdp.elevation_map,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": True},
    )

    def __post_init__(self):
        self.history_length = 1
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class AttentionObservationsCfg:
    policy: AttentionPolicyCfg = AttentionPolicyCfg()


@configclass
class AttentionBaseEnvCfg(ManagerBasedRLEnvCfg):
    scene: AttentionSceneCfg = AttentionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: AttentionObservationsCfg = AttentionObservationsCfg()
    rewards: AttentionRewardsCfg = AttentionRewardsCfg()
    actions: AttentionActionsCfg = AttentionActionsCfg()
    commands: AttentionCommandsCfg = AttentionCommandsCfg()
    terminations: AttentionTerminationsCfg = AttentionTerminationsCfg()
    events: AttentionEventCfg = AttentionEventCfg()
    curriculum: AttentionCurriculumCfg = AttentionCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        elif self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


class AttentionEnvCfgMixin:
    def configure_attention_train(self, height_scanner_prim_path: str) -> None:
        self.scene.terrain.terrain_generator = copy.deepcopy(AME_PARKOUR_TERRAINS_CFG)
        self.scene.terrain.max_init_terrain_level = 5

        self.scene.height_scanner.prim_path = height_scanner_prim_path
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=ATTENTION_GRID_RESOLUTION,
            size=ATTENTION_GRID_SIZE,
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

    def configure_attention_play(self) -> None:
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)

        self.observations.policy.enable_corruption = False
        self.observations.policy.terrain_map.params["noise"] = False
        self.scene.height_scanner.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.size = (8.0, 8.0)
            self.scene.terrain.terrain_generator.sub_terrains = {
                "stakes": HfAlternateColumnStakesTerrainCfg(
                    proportion=0.5,
                    stake_height_max=0.0,
                    stake_side_range=(0.2, 0.2),
                    stake_gap_range=(0.3, 0.3),
                    column_gap_range=(0.3, 0.3),
                    column_jitter=0.0,
                    holes_depth=-2.0,
                    platform_width=2.0,
                    border_width=0.25,
                ),
            }

        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
        if hasattr(self.events, "actuator_gains"):
            self.events.actuator_gains = None
