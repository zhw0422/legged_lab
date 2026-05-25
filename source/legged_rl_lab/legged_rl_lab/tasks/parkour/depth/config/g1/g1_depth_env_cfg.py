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
from legged_rl_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_rl_lab.tasks.parkour.depth.ts_depth_env_cfg import EventCfg as TSDepthEventCfg
from legged_rl_lab.tasks.parkour.depth.ts_depth_env_cfg import LocomotionTSDepthEnvCfg
from legged_rl_lab.tasks.parkour.depth.ts_depth_env_cfg import TerminationsCfg as TSDepthTerminationsCfg
from legged_rl_lab.tasks.parkour.depth.ts_depth_env_cfg import TSDepthSceneCfg
import legged_rl_lab.tasks.parkour.depth.mdp as mdp
_DEPTH_PITCH_DEG = 45.0
_DEPTH_ROT_Y = math.radians(_DEPTH_PITCH_DEG) - math.pi / 2.0
_DEPTH_IMAGE_HEIGHT = 30
_DEPTH_IMAGE_WIDTH = 40
_DEPTH_FOCAL_LENGTH = 24.0
_DEPTH_HFOV_DEG = 75.0
_DEPTH_HORIZONTAL_APERTURE = 2.0 * _DEPTH_FOCAL_LENGTH * math.tan(math.radians(_DEPTH_HFOV_DEG / 2.0))
_DEPTH_MAX_DEPTH = 2.0
_G1_TS_DEPTH_CONTACT_LINKS = ('torso_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_roll_link', 'left_ankle_pitch_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_roll_link', 'right_ankle_pitch_link', 'waist_yaw_link')
_G1_FOOT_SENSORS: tuple[str, ...] = ('foot_height_scanner_L', 'foot_height_scanner_R')
_G1_FOOT_BODIES: tuple[str, ...] = ('left_ankle_roll_link', 'right_ankle_roll_link')
_FOOT_GRID_PATTERN = patterns.GridPatternCfg(resolution=0.1, size=(0.2, 0.2), ordering='xy')
_FOOT_RAY_OFFSET = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0))

@configclass
class G1TSDepthSceneCfg(TSDepthSceneCfg):
    depth_scanner = RayCasterCameraCfg(prim_path='{ENV_REGEX_NS}/Robot/torso_link', offset=RayCasterCameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.1), rot=(math.cos(_DEPTH_ROT_Y / 2.0), 0.0, math.sin(_DEPTH_ROT_Y / 2.0), 0.0), convention='opengl'), pattern_cfg=patterns.PinholeCameraPatternCfg(focal_length=_DEPTH_FOCAL_LENGTH, horizontal_aperture=_DEPTH_HORIZONTAL_APERTURE, vertical_aperture=None, height=_DEPTH_IMAGE_HEIGHT, width=_DEPTH_IMAGE_WIDTH), data_types=['distance_to_image_plane'], depth_clipping_behavior='max', max_distance=_DEPTH_MAX_DEPTH, debug_vis=False, mesh_prim_paths=['/World/ground'])
    foot_height_scanner_L = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/left_ankle_roll_link', offset=_FOOT_RAY_OFFSET, ray_alignment='yaw', pattern_cfg=_FOOT_GRID_PATTERN, debug_vis=False, mesh_prim_paths=['/World/ground'])
    foot_height_scanner_R = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/right_ankle_roll_link', offset=_FOOT_RAY_OFFSET, ray_alignment='yaw', pattern_cfg=_FOOT_GRID_PATTERN, debug_vis=False, mesh_prim_paths=['/World/ground'])

@configclass
class G1TSDepthEventCfg(TSDepthEventCfg):
    push_robot = EventTerm(func=mdp.push_by_setting_velocity_record_xy, mode='interval', interval_range_s=(3.0, 3.0), params={'velocity_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5)}, 'asset_cfg': SceneEntityCfg('robot')})
    actuator_gains = EventTerm(func=mdp.randomize_actuator_gains, mode='startup', params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*']), 'stiffness_distribution_params': (0.8, 1.2), 'damping_distribution_params': (0.8, 1.2), 'operation': 'scale', 'distribution': 'uniform'})
    reset_robot_joints = EventTerm(func=mdp.reset_joints_by_offset, mode='reset', params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*']), 'position_range': (-0.2, 0.2), 'velocity_range': (0.0, 0.0)})

@configclass
class G1TSDepthObservationsCfg:

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
        dr_mass_scale = ObsTerm(func=mdp.body_mass_scale, params={'asset_cfg': SceneEntityCfg('robot', body_names='torso_link')})
        dr_com_b = ObsTerm(func=mdp.body_com_pos_b, params={'asset_cfg': SceneEntityCfg('robot', body_names='torso_link')})
        dr_push_xy = ObsTerm(func=mdp.last_push_delta_xy)
        dr_kp_scale = ObsTerm(func=mdp.joint_stiffness_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        dr_kd_scale = ObsTerm(func=mdp.joint_damping_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_relative_to_feet = ObsTerm(func=mdp.height_relative_to_feet, params={'sensor_names': list(_G1_FOOT_SENSORS), 'asset_cfg': SceneEntityCfg('robot', body_names=list(_G1_FOOT_BODIES)), 'clip': (-1.0, 1.0)})
        normal_vector_around_feet = ObsTerm(func=mdp.normal_vector_around_feet, params={'sensor_names': list(_G1_FOOT_SENSORS)})
        link_contact_states = ObsTerm(func=mdp.links_contact_binary, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=list(_G1_TS_DEPTH_CONTACT_LINKS)), 'threshold': 1.0})
        height_scan = ObsTerm(func=mdp.height_scan, params={'sensor_cfg': SceneEntityCfg('height_scanner'), 'offset': 0.8, 'clip': (-1.0, 1.0)}, scale=2.0)

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
        dr_mass_scale = ObsTerm(func=mdp.body_mass_scale, params={'asset_cfg': SceneEntityCfg('robot', body_names='torso_link')})
        dr_com_b = ObsTerm(func=mdp.body_com_pos_b, params={'asset_cfg': SceneEntityCfg('robot', body_names='torso_link')})
        dr_push_xy = ObsTerm(func=mdp.last_push_delta_xy)
        dr_kp_scale = ObsTerm(func=mdp.joint_stiffness_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        dr_kd_scale = ObsTerm(func=mdp.joint_damping_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        link_contact_states = ObsTerm(func=mdp.links_contact_binary, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=list(_G1_TS_DEPTH_CONTACT_LINKS)), 'threshold': 1.0})
        height_relative_to_feet = ObsTerm(func=mdp.height_relative_to_feet, params={'sensor_names': list(_G1_FOOT_SENSORS), 'asset_cfg': SceneEntityCfg('robot', body_names=list(_G1_FOOT_BODIES)), 'clip': (-1.0, 1.0)})
        normal_vector_around_feet = ObsTerm(func=mdp.normal_vector_around_feet, params={'sensor_names': list(_G1_FOOT_SENSORS)})
        height_scan = ObsTerm(func=mdp.height_scan, params={'sensor_cfg': SceneEntityCfg('height_scanner'), 'offset': 0.8, 'clip': (-1.0, 1.0)}, scale=2.0)
        gait_phase = ObsTerm(func=mdp.gait_phase_sin_cos, params={'period': 0.8, 'offset': [0.0, 0.5]})
        feet_pos = ObsTerm(func=mdp.feet_pos_body_frame, params={'asset_cfg': SceneEntityCfg('robot', body_names=list(_G1_FOOT_BODIES))})
        feet_vel = ObsTerm(func=mdp.feet_vel_body_frame, params={'asset_cfg': SceneEntityCfg('robot', body_names=list(_G1_FOOT_BODIES))})
        feet_force = ObsTerm(func=mdp.feet_contact_force, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_ankle_roll_link')})
        root_height = ObsTerm(func=mdp.base_pos_z)

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
class G1TSDepthRewardsCfg:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    alive = RewTerm(func=mdp.alive, weight=2.0)
    tracking_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_heading_exp, weight=1.5, params={'command_name': 'base_velocity', 'std': math.sqrt(0.2), 'y_error_weight': 2.0})
    tracking_ang_vel = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={'command_name': 'base_velocity', 'std': 0.5})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    pelvis_orientation = RewTerm(func=mdp.body_orientation_l2, weight=-1.0, params={'asset_cfg': SceneEntityCfg('robot', body_names='torso_link')})
    dof_power = RewTerm(func=mdp.dof_power_l1, weight=-2e-05, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2e-07, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    foot_clearance = RewTerm(func=mdp.foot_clearance_target, weight=0.2, params={'sensor_cfg': SceneEntityCfg('height_scanner'), 'asset_cfg': SceneEntityCfg('robot', body_names=['left_ankle_roll_link', 'right_ankle_roll_link']), 'target_height': 0.08, 'foot_offset': 0.022, 'sigma': 0.01})
    feet_contact_stand_still = RewTerm(func=mdp.feet_contact_stand_still, weight=0.1, params={'command_name': 'base_velocity', 'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_ankle_roll_link'), 'cmd_threshold': 0.2, 'force_threshold': 10.0})
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-1.0, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_ankle_roll_link')})
    hip_pos = RewTerm(func=mdp.hip_pos_deviation, weight=-0.15, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*_hip_yaw_joint', '.*_hip_roll_joint'])})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    collision = RewTerm(func=mdp.undesired_contacts, weight=-5.0, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='(?!.*_ankle_roll_link).*'), 'threshold': 1.0})
    fly = RewTerm(func=mdp.fly, weight=-1.0, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_ankle_roll_link'), 'threshold': 1.0})
    feet_too_near = RewTerm(func=mdp.feet_too_near, weight=-2.0, params={'asset_cfg': SceneEntityCfg('robot', body_names=['left_ankle_roll_link', 'right_ankle_roll_link']), 'threshold': 0.2})
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1_always, weight=-0.2, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['waist_.*_joint', '.*_shoulder_roll_joint', '.*_shoulder_yaw_joint', '.*_shoulder_pitch_joint', '.*_elbow_joint', '.*_wrist_.*_joint'])})
    feet_air_time = RewTerm(func=mdp.feet_air_time_positive_biped, weight=0.5, params={'command_name': 'base_velocity', 'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_ankle_roll_link'), 'threshold': 0.4})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.4, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_ankle_roll_link'), 'asset_cfg': SceneEntityCfg('robot', body_names='.*_ankle_roll_link')})
    joint_deviation_ankle = RewTerm(func=mdp.joint_deviation_l1_always, weight=-0.2, params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*_ankle_.*_joint')})
    leg_ref_joint_pos = RewTerm(func=mdp.leg_ref_joint_pos, weight=0.5, params={'left_cfg': SceneEntityCfg('robot', joint_names=['left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint']), 'right_cfg': SceneEntityCfg('robot', joint_names=['right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint']), 'period': 0.8, 'scales': (-0.2, 0.4, -0.2), 'double_support_threshold': 0.1, 'command_name': 'base_velocity', 'cmd_threshold': 0.1})
    gait_phase_contact = RewTerm(func=mdp.feet_gait, weight=0.2, params={'period': 0.8, 'offset': [0.0, 0.5], 'sensor_cfg': SceneEntityCfg('contact_forces', body_names=['left_ankle_roll_link', 'right_ankle_roll_link']), 'threshold': 0.55, 'command_name': 'base_velocity'})

@configclass
class G1TSDepthTerminationsCfg(TSDepthTerminationsCfg):
    gravity_tilt = DoneTerm(func=mdp.gravity_too_horizontal, params={'threshold': -0.1})

@configclass
class G1TSDepthEnvCfg(LocomotionTSDepthEnvCfg):
    scene: G1TSDepthSceneCfg = G1TSDepthSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1TSDepthObservationsCfg = G1TSDepthObservationsCfg()
    rewards: G1TSDepthRewardsCfg = G1TSDepthRewardsCfg()
    events: G1TSDepthEventCfg = G1TSDepthEventCfg()
    terminations: G1TSDepthTerminationsCfg = G1TSDepthTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
        self.scene.height_scanner.prim_path = '{ENV_REGEX_NS}/Robot/pelvis'
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=(1.8, 0.8))
        for sensor_name in ('foot_height_scanner_L', 'foot_height_scanner_R'):
            sensor = getattr(self.scene, sensor_name, None)
            if sensor is not None:
                sensor.update_period = self.decimation * self.sim.dt
        self.events.physics_material.params['static_friction_range'] = (0.2, 1.7)
        self.events.physics_material.params['dynamic_friction_range'] = (0.2, 1.7)
        self.events.add_base_mass.params['mass_distribution_params'] = (-1.0, 2.0)
        self.events.base_com.params['com_range'] = {'x': (-0.03, 0.03), 'y': (-0.03, 0.03), 'z': (-0.03, 0.03)}
        self.events.add_base_mass.params['asset_cfg'].body_names = 'torso_link'
        self.events.base_com.params['asset_cfg'].body_names = 'torso_link'
        self.events.base_external_force_torque.params['asset_cfg'].body_names = 'torso_link'
        self.terminations.base_contact.params['sensor_cfg'].body_names = 'torso_link'
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.events.reset_base.params['pose_range']['yaw'] = (0.0, 0.0)
        self.curriculum.lin_vel_cmd_levels.params['reward_term_name'] = 'tracking_lin_vel'
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)

@configclass
class G1TSDepthEnvCfg_PLAY(G1TSDepthEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.observations.policy.enable_corruption = False
        self.observations.depth.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains = None
        self.scene.depth_scanner.debug_vis = True