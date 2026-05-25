from __future__ import annotations
import math
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster import RayCasterCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import legged_rl_lab.tasks.parkour.depth.mdp as mdp
from legged_rl_lab.terrains import DWAQ_TERRAINS_CFG
_DEPTH_PITCH_DEG = 45.0
_DEPTH_ROT_Y = math.radians(_DEPTH_PITCH_DEG) - math.pi / 2.0
_DEPTH_IMAGE_HEIGHT = 30
_DEPTH_IMAGE_WIDTH = 40
_DEPTH_FOCAL_LENGTH = 24.0
_DEPTH_HFOV_DEG = 75.0
_DEPTH_HORIZONTAL_APERTURE = 2.0 * _DEPTH_FOCAL_LENGTH * math.tan(math.radians(_DEPTH_HFOV_DEG / 2.0))
_DEPTH_MAX_DEPTH = 3.0

@configclass
class TSDepthSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path='/World/ground', terrain_type='generator', terrain_generator=DWAQ_TERRAINS_CFG, max_init_terrain_level=5, collision_group=-1, physics_material=sim_utils.RigidBodyMaterialCfg(friction_combine_mode='multiply', restitution_combine_mode='multiply', static_friction=1.0, dynamic_friction=1.0), visual_material=sim_utils.MdlFileCfg(mdl_path=f'{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl', project_uvw=True, texture_scale=(0.25, 0.25)), debug_vis=False)
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*', history_length=3, track_air_time=True)
    depth_scanner = RayCasterCameraCfg(prim_path='{ENV_REGEX_NS}/Robot/base', offset=RayCasterCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.45), rot=(math.cos(_DEPTH_ROT_Y / 2.0), 0.0, math.sin(_DEPTH_ROT_Y / 2.0), 0.0), convention='opengl'), pattern_cfg=patterns.PinholeCameraPatternCfg(focal_length=_DEPTH_FOCAL_LENGTH, horizontal_aperture=_DEPTH_HORIZONTAL_APERTURE, vertical_aperture=None, height=_DEPTH_IMAGE_HEIGHT, width=_DEPTH_IMAGE_WIDTH), data_types=['distance_to_image_plane'], depth_clipping_behavior='max', max_distance=_DEPTH_MAX_DEPTH, debug_vis=False, mesh_prim_paths=['/World/ground'])
    height_scanner = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/base', offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), ray_alignment='yaw', pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)), debug_vis=False, mesh_prim_paths=['/World/ground'])
    sky_light = AssetBaseCfg(prim_path='/World/skyLight', spawn=sim_utils.DomeLightCfg(intensity=750.0, texture_file=f'{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr'))

@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(asset_name='robot', resampling_time_range=(10.0, 10.0), rel_standing_envs=0.2, rel_heading_envs=1.0, heading_command=True, heading_control_stiffness=0.5, debug_vis=True, ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 1.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.2, 1.2), heading=(-math.pi, math.pi)), limit_ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 1.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.2, 1.2), heading=(-math.pi, math.pi)))

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name='robot', joint_names=['.*'], scale=0.25, use_default_offset=True)

@configclass
class EventCfg:
    physics_material = EventTerm(func=mdp.randomize_rigid_body_material, mode='startup', params={'asset_cfg': SceneEntityCfg('robot', body_names='.*'), 'static_friction_range': (0.8, 0.8), 'dynamic_friction_range': (0.6, 0.6), 'restitution_range': (0.0, 0.0), 'num_buckets': 64})
    add_base_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode='startup', params={'asset_cfg': SceneEntityCfg('robot', body_names='base'), 'mass_distribution_params': (-1.0, 1.0), 'operation': 'add'})
    base_com = EventTerm(func=mdp.randomize_rigid_body_com, mode='startup', params={'asset_cfg': SceneEntityCfg('robot', body_names='base'), 'com_range': {'x': (-0.05, 0.05), 'y': (-0.05, 0.05), 'z': (-0.01, 0.01)}})
    base_external_force_torque = EventTerm(func=mdp.apply_external_force_torque, mode='reset', params={'asset_cfg': SceneEntityCfg('robot', body_names='base'), 'force_range': (0.0, 0.0), 'torque_range': (-0.0, 0.0)})
    reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode='reset', params={'pose_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'yaw': (-3.14, 3.14)}, 'velocity_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'z': (-0.5, 0.5), 'roll': (-0.5, 0.5), 'pitch': (-0.5, 0.5), 'yaw': (-0.5, 0.5)}})
    reset_robot_joints = EventTerm(func=mdp.reset_joints_by_scale, mode='reset', params={'position_range': (0.5, 1.5), 'velocity_range': (0.0, 0.0)})
    push_robot = EventTerm(func=mdp.push_by_setting_velocity, mode='interval', interval_range_s=(10.0, 15.0), params={'velocity_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5)}})

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='base'), 'threshold': 1.0})

@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels, params={'reward_term_name': 'tracking_lin_vel'})

@configclass
class LocomotionTSDepthEnvCfg(ManagerBasedRLEnvCfg):
    scene: TSDepthSceneCfg = TSDepthSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.depth_scanner is not None:
            self.scene.depth_scanner.update_period = 5 * self.decimation * self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if getattr(self.curriculum, 'terrain_levels', None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        elif self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False