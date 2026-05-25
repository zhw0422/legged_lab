from __future__ import annotations
import math
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors.ray_caster import RayCasterCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from legged_rl_lab.assets.unitree import UNITREE_GO2_CFG
from legged_rl_lab.tasks.parkour.depth.ts_depth_env_cfg import EventCfg as TSDepthEventCfg, LocomotionTSDepthEnvCfg, TerminationsCfg as TSDepthTerminationsCfg, TSDepthSceneCfg
import legged_rl_lab.tasks.parkour.depth.mdp as mdp
_DEPTH_PITCH_DEG = 45.0
_DEPTH_ROT_Y = math.radians(_DEPTH_PITCH_DEG) - math.pi / 2.0
_DEPTH_IMAGE_HEIGHT = 30
_DEPTH_IMAGE_WIDTH = 40
_DEPTH_FOCAL_LENGTH = 24.0
_DEPTH_HFOV_DEG = 75.0
_DEPTH_HORIZONTAL_APERTURE = 2.0 * _DEPTH_FOCAL_LENGTH * math.tan(math.radians(_DEPTH_HFOV_DEG / 2.0))
_DEPTH_MAX_DEPTH = 2.0
_GO2_CONTACT_LINKS: tuple[str, ...] = ('.*_thigh', '.*_calf', '.*_foot')
_GO2_FOOT_SENSORS: tuple[str, ...] = ('foot_height_scanner_FL', 'foot_height_scanner_FR', 'foot_height_scanner_RL', 'foot_height_scanner_RR')
_GO2_FOOT_BODIES: tuple[str, ...] = ('FL_foot', 'FR_foot', 'RL_foot', 'RR_foot')
_FOOT_GRID_PATTERN = patterns.GridPatternCfg(resolution=0.1, size=(0.2, 0.2), ordering='xy')
_FOOT_RAY_OFFSET = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0))

@configclass
class Go2TSDepthSceneCfg(TSDepthSceneCfg):
    depth_scanner = RayCasterCameraCfg(prim_path='{ENV_REGEX_NS}/Robot/base', offset=RayCasterCameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.1), rot=(math.cos(_DEPTH_ROT_Y / 2.0), 0.0, math.sin(_DEPTH_ROT_Y / 2.0), 0.0), convention='opengl'), pattern_cfg=patterns.PinholeCameraPatternCfg(focal_length=_DEPTH_FOCAL_LENGTH, horizontal_aperture=_DEPTH_HORIZONTAL_APERTURE, vertical_aperture=None, height=_DEPTH_IMAGE_HEIGHT, width=_DEPTH_IMAGE_WIDTH), data_types=['distance_to_image_plane'], depth_clipping_behavior='max', max_distance=_DEPTH_MAX_DEPTH, debug_vis=False, mesh_prim_paths=['/World/ground'])
    height_scanner = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/base', offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), ray_alignment='yaw', pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.8, 0.8)), debug_vis=False, mesh_prim_paths=['/World/ground'])
    foot_height_scanner_FL = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/FL_foot', offset=_FOOT_RAY_OFFSET, ray_alignment='yaw', pattern_cfg=_FOOT_GRID_PATTERN, debug_vis=False, mesh_prim_paths=['/World/ground'])
    foot_height_scanner_FR = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/FR_foot', offset=_FOOT_RAY_OFFSET, ray_alignment='yaw', pattern_cfg=_FOOT_GRID_PATTERN, debug_vis=False, mesh_prim_paths=['/World/ground'])
    foot_height_scanner_RL = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/RL_foot', offset=_FOOT_RAY_OFFSET, ray_alignment='yaw', pattern_cfg=_FOOT_GRID_PATTERN, debug_vis=False, mesh_prim_paths=['/World/ground'])
    foot_height_scanner_RR = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/RR_foot', offset=_FOOT_RAY_OFFSET, ray_alignment='yaw', pattern_cfg=_FOOT_GRID_PATTERN, debug_vis=False, mesh_prim_paths=['/World/ground'])

@configclass
class Go2TSDepthObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={'command_name': 'base_velocity'}, scale=(1.0, 1.0, 0.25))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action, scale=0.1)

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        dr_friction = ObsTerm(func=mdp.scalar_rigid_friction_mean, params={'asset_cfg': SceneEntityCfg('robot')})
        dr_mass_scale = ObsTerm(func=mdp.body_mass_scale, params={'asset_cfg': SceneEntityCfg('robot', body_names='base')})
        dr_com_b = ObsTerm(func=mdp.body_com_pos_b, params={'asset_cfg': SceneEntityCfg('robot', body_names='base')})
        dr_push_xy = ObsTerm(func=mdp.last_push_delta_xy)
        dr_kp_scale = ObsTerm(func=mdp.joint_stiffness_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        dr_kd_scale = ObsTerm(func=mdp.joint_damping_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_relative_to_feet = ObsTerm(func=mdp.height_relative_to_feet, params={'sensor_names': list(_GO2_FOOT_SENSORS), 'asset_cfg': SceneEntityCfg('robot', body_names=list(_GO2_FOOT_BODIES)), 'clip': (-1.0, 1.0)})
        normal_vector_around_feet = ObsTerm(func=mdp.normal_vector_around_feet, params={'sensor_names': list(_GO2_FOOT_SENSORS)})
        link_contact_states = ObsTerm(func=mdp.links_contact_binary, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=list(_GO2_CONTACT_LINKS)), 'threshold': 1.0})
        height_scan = ObsTerm(func=mdp.height_scan, params={'sensor_cfg': SceneEntityCfg('height_scanner'), 'offset': 0.4, 'clip': (-1.0, 1.0)}, scale=2.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={'command_name': 'base_velocity'}, scale=(1.0, 1.0, 0.25))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action, scale=0.1)
        dr_friction = ObsTerm(func=mdp.scalar_rigid_friction_mean, params={'asset_cfg': SceneEntityCfg('robot')})
        dr_mass_scale = ObsTerm(func=mdp.body_mass_scale, params={'asset_cfg': SceneEntityCfg('robot', body_names='base')})
        dr_com_b = ObsTerm(func=mdp.body_com_pos_b, params={'asset_cfg': SceneEntityCfg('robot', body_names='base')})
        dr_push_xy = ObsTerm(func=mdp.last_push_delta_xy)
        dr_kp_scale = ObsTerm(func=mdp.joint_stiffness_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        dr_kd_scale = ObsTerm(func=mdp.joint_damping_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        link_contact_states = ObsTerm(func=mdp.links_contact_binary, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=list(_GO2_CONTACT_LINKS)), 'threshold': 1.0})
        height_relative_to_feet = ObsTerm(func=mdp.height_relative_to_feet, params={'sensor_names': list(_GO2_FOOT_SENSORS), 'asset_cfg': SceneEntityCfg('robot', body_names=list(_GO2_FOOT_BODIES)), 'clip': (-1.0, 1.0)})
        normal_vector_around_feet = ObsTerm(func=mdp.normal_vector_around_feet, params={'sensor_names': list(_GO2_FOOT_SENSORS)})
        height_scan = ObsTerm(func=mdp.height_scan, params={'sensor_cfg': SceneEntityCfg('height_scanner'), 'offset': 0.4, 'clip': (-1.0, 1.0)}, scale=2.0)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DepthCfg(ObsGroup):
        depth_image = ObsTerm(func=mdp.depth_image_camera, params={'sensor_cfg': SceneEntityCfg('depth_scanner'), 'max_depth': _DEPTH_MAX_DEPTH}, noise=Unoise(n_min=-0.02, n_max=0.02))

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    critic: CriticCfg = CriticCfg()
    depth: DepthCfg = DepthCfg()

@configclass
class Go2TSDepthEventCfg(TSDepthEventCfg):
    reset_hip_joints = EventTerm(func=mdp.reset_joints_by_offset, mode='reset', params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*_hip_joint'), 'position_range': (-0.2, 0.2), 'velocity_range': (0.0, 0.0)})
    reset_thigh_joints = EventTerm(func=mdp.reset_joints_by_offset, mode='reset', params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*_thigh_joint'), 'position_range': (-0.4, 0.4), 'velocity_range': (0.0, 0.0)})
    reset_calf_joints = EventTerm(func=mdp.reset_joints_by_offset, mode='reset', params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*_calf_joint'), 'position_range': (-0.4, 0.4), 'velocity_range': (0.0, 0.0)})
    push_robot = EventTerm(func=mdp.push_by_setting_velocity_record_xy, mode='interval', interval_range_s=(3.0, 3.0), params={'velocity_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5)}, 'asset_cfg': SceneEntityCfg('robot')})
    actuator_gains = EventTerm(func=mdp.randomize_actuator_gains, mode='startup', params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*']), 'stiffness_distribution_params': (0.8, 1.2), 'damping_distribution_params': (0.8, 1.2), 'operation': 'scale', 'distribution': 'uniform'})

@configclass
class Go2TSDepthRewardsCfg:
    tracking_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_heading_exp, weight=1.5, params={'command_name': 'base_velocity', 'std': math.sqrt(0.2), 'y_error_weight': 2.0})
    tracking_ang_vel = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={'command_name': 'base_velocity', 'std': math.sqrt(0.2)})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_power = RewTerm(func=mdp.dof_power_l1, weight=-2e-05, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2e-07, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    foot_clearance = RewTerm(func=mdp.foot_clearance_target, weight=0.2, params={'sensor_cfg': SceneEntityCfg('height_scanner'), 'asset_cfg': SceneEntityCfg('robot', body_names='.*_foot'), 'target_height': 0.08, 'foot_offset': 0.022, 'sigma': 0.01})
    feet_contact_stand_still = RewTerm(func=mdp.feet_contact_stand_still, weight=0.1, params={'command_name': 'base_velocity', 'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_foot'), 'cmd_threshold': 0.2, 'force_threshold': 10.0})
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-1.0, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_foot')})
    hip_pos = RewTerm(func=mdp.hip_pos_deviation, weight=-0.15, params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*_hip_joint')})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    collision = RewTerm(func=mdp.undesired_contacts, weight=-10.0, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=['base', '.*_thigh', '.*_calf']), 'threshold': 1.0})

@configclass
class Go2TSDepthTerminationsCfg(TSDepthTerminationsCfg):
    gravity_tilt = DoneTerm(func=mdp.gravity_too_horizontal, params={'threshold': -0.1})

@configclass
class Go2TSDepthEnvCfg(LocomotionTSDepthEnvCfg):
    scene: Go2TSDepthSceneCfg = Go2TSDepthSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: Go2TSDepthObservationsCfg = Go2TSDepthObservationsCfg()
    rewards: Go2TSDepthRewardsCfg = Go2TSDepthRewardsCfg()
    events: Go2TSDepthEventCfg = Go2TSDepthEventCfg()
    terminations: Go2TSDepthTerminationsCfg = Go2TSDepthTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        for sensor_name in ('foot_height_scanner_FL', 'foot_height_scanner_FR', 'foot_height_scanner_RL', 'foot_height_scanner_RR'):
            sensor = getattr(self.scene, sensor_name, None)
            if sensor is not None:
                sensor.update_period = self.decimation * self.sim.dt
        robot = UNITREE_GO2_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        robot.init_state.pos = (0.0, 0.0, 0.4)
        robot.init_state.joint_pos = {'.*_hip_joint': 0.0, '.*_thigh_joint': 0.8, '.*_calf_joint': -1.5}
        if 'GO2HV' in robot.actuators:
            robot.actuators['GO2HV'].stiffness = 30.0
            robot.actuators['GO2HV'].damping = 0.75
        self.scene.robot = robot
        self.events.physics_material.params['static_friction_range'] = (0.2, 1.7)
        self.events.physics_material.params['dynamic_friction_range'] = (0.2, 1.7)
        self.events.add_base_mass.params['mass_distribution_params'] = (-1.0, 2.0)
        self.events.base_com.params['com_range'] = {'x': (-0.03, 0.03), 'y': (-0.03, 0.03), 'z': (-0.03, 0.03)}
        self.events.reset_robot_joints = None
        self.events.reset_base.params['pose_range']['yaw'] = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.curriculum.lin_vel_cmd_levels.params['reward_term_name'] = 'tracking_lin_vel'
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 1.5)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)

@configclass
class Go2TSDepthEnvCfg_PLAY(Go2TSDepthEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.observations.policy.enable_corruption = False
        self.observations.depth.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains = None
        self.scene.depth_scanner.debug_vis = True