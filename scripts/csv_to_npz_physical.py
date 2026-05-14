"""Physics-based CSV → NPZ preprocessor for AMP reference motions.

Unlike ``scripts/csv_to_npz.py`` which only kinematically writes the state each
frame (so the recorded ``joint_vel`` / ``body_lin_vel_w`` is just the
finite-difference of the input, not physics-integrated), this script:

1. Fixes the robot root kinematically to the reference trajectory each frame
   (``write_root_pose_to_sim`` + ``write_root_velocity_to_sim``).
2. Drives the joints through the configured PD actuators with the reference
   joint positions as targets.
3. Runs ``sim.step()`` for real PhysX integration — actuator torque → joint
   acceleration → joint velocity → joint position.
4. Reads the physics-derived state back.

The resulting ``joint_vel`` / ``body_lin_vel_w`` reflects actual actuator +
inertia dynamics, matching the distribution the policy produces at training
time.  This removes the trivial velocity-distribution gap that saturated the
AMP discriminator when using raw finite-difference CSV data.

Usage::

    python scripts/csv_to_npz_physical.py \
        --input_file path/to/walk1_subject1.csv \
        --input_fps 30 \
        --output_fps 30 \
        --headless
"""

from __future__ import annotations

import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Physics-based replay of a reference motion CSV → NPZ."
)
parser.add_argument("--input_file", "-f", type=str, required=True)
parser.add_argument("--input_fps", type=int, default=30)
parser.add_argument("--output_fps", type=int, default=30)
parser.add_argument("--output_name", type=str, default=None)
parser.add_argument(
    "--warmup_frames",
    type=int,
    default=30,
    help="Pre-integration frames to let PD actuators settle before logging.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.output_name is None:
    args_cli.output_name = args_cli.input_file.rsplit(".", 1)[0] + "_physical.npz"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Rest of imports must come after AppLauncher
# ---------------------------------------------------------------------------

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_slerp

from legged_rl_lab.assets.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ---------------------------------------------------------------------------
# Motion I/O helpers
# ---------------------------------------------------------------------------

def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return a * (1.0 - blend) + b * blend


def _slerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Batched SLERP over (B, 4) quaternions with (B,) blend."""
    out = torch.zeros_like(a)
    for i in range(a.shape[0]):
        out[i] = quat_slerp(a[i], b[i], blend[i])
    return out


def load_and_resample_motion(
    path: str, input_fps: int, output_fps: int, device: torch.device
) -> dict:
    """Load a LAFAN1-style CSV and resample to ``output_fps``.

    CSV layout (G1, 30 FPS):  ``[x, y, z, qx, qy, qz, qw,  joint_0 … joint_28]``
    Returns a dict with all per-frame tensors on ``device``.
    """
    raw = torch.from_numpy(np.loadtxt(path, delimiter=",")).float().to(device)
    root_pos_in = raw[:, :3]
    # Convert quat xyzw → wxyz
    quat_xyzw = raw[:, 3:7]
    root_quat_in = torch.stack(
        [quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]], dim=-1
    )
    joint_pos_in = raw[:, 7:]  # AMASS/MuJoCo order — will be reordered by IsaacLab

    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (raw.shape[0] - 1) * input_dt
    times = torch.arange(0, duration, output_dt, device=device, dtype=torch.float32)
    phase = times / duration
    idx0 = (phase * (raw.shape[0] - 1)).floor().long()
    idx1 = torch.minimum(idx0 + 1, torch.tensor(raw.shape[0] - 1, device=device))
    blend = phase * (raw.shape[0] - 1) - idx0

    root_pos = _lerp(root_pos_in[idx0], root_pos_in[idx1], blend.unsqueeze(1))
    root_quat = _slerp(root_quat_in[idx0], root_quat_in[idx1], blend)
    joint_pos = _lerp(joint_pos_in[idx0], joint_pos_in[idx1], blend.unsqueeze(1))

    # Finite-diff velocities only for the kinematic root write.  Joint vel
    # doesn't need to be seeded — PD actuators compute effort from target.
    root_lin_vel = torch.gradient(root_pos, spacing=output_dt, dim=0)[0]
    # Angular vel from consecutive quats via axis-angle of relative rotation
    from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul
    q_prev, q_next = root_quat[:-2], root_quat[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    ang = axis_angle_from_quat(q_rel) / (2.0 * output_dt)
    ang = torch.cat([ang[:1], ang, ang[-1:]], dim=0)

    return {
        "root_pos": root_pos,
        "root_quat": root_quat,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": ang,
        "joint_pos": joint_pos,
        "dt": output_dt,
        "num_frames": root_pos.shape[0],
    }


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run(sim: SimulationContext, scene: InteractiveScene, motion: dict) -> None:
    robot = scene["robot"]
    # Map reference joint column order (AMASS) → IsaacLab BFS index order.
    joint_idx = robot.find_joints(scene.cfg.robot.joint_sdk_names, preserve_order=True)[0]

    device = sim.device
    N = motion["num_frames"]
    dt = motion["dt"]

    # Ensure sim.dt matches motion dt (one sim.step per frame).
    if abs(sim.get_physics_dt() - dt) > 1e-6:
        print(
            f"[WARN] Sim dt ({sim.get_physics_dt():.6f}) != motion dt ({dt:.6f}). "
            "The recorded velocities will reflect sim dt integration."
        )

    # Warm-up: settle PD actuators before we start logging.
    # Write root state + joint target to frame 0, step a few times.
    for _ in range(args_cli.warmup_frames):
        root_pose0 = torch.cat([motion["root_pos"][:1], motion["root_quat"][:1]], dim=-1)
        root_vel0 = torch.cat(
            [motion["root_lin_vel"][:1], motion["root_ang_vel"][:1]], dim=-1
        )
        robot.write_root_pose_to_sim(root_pose0)
        robot.write_root_velocity_to_sim(root_vel0)
        target = robot.data.default_joint_pos.clone()
        target[:, joint_idx] = motion["joint_pos"][:1]
        # Initialize joint state so PD doesn't have to catch up from zero.
        robot.write_joint_state_to_sim(target, torch.zeros_like(target))
        robot.set_joint_position_target(target)
        sim.step(render=False)
        scene.update(dt)

    # Logging buffers
    log = {
        "fps": np.array([int(round(1.0 / dt))], dtype=np.int64),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for i in range(N):
        # Kinematic root tracking — keeps robot on reference trajectory.
        root_pose = torch.cat(
            [motion["root_pos"][i:i+1], motion["root_quat"][i:i+1]], dim=-1
        )
        root_vel = torch.cat(
            [motion["root_lin_vel"][i:i+1], motion["root_ang_vel"][i:i+1]], dim=-1
        )
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(root_vel)

        # PD target — actuators compute torque, physics integrates joint state.
        target = robot.data.default_joint_pos.clone()
        target[:, joint_idx] = motion["joint_pos"][i:i+1]
        robot.set_joint_position_target(target)

        sim.step(render=False)
        scene.update(dt)

        # Read back physics-derived joint + body state.
        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

        if (i + 1) % 500 == 0:
            print(f"[{i + 1}/{N}] frames processed…")

    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
              "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)

    np.savez(args_cli.output_name, **log)
    print(f"[INFO] Saved physics-based NPZ to: {args_cli.output_name}")
    print(f"       {N} frames @ {1.0 / dt:.1f} FPS, "
          f"joint_vel range: [{log['joint_vel'].min():.2f}, {log['joint_vel'].max():.2f}]")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] Sim ready; loading motion…")

    motion = load_and_resample_motion(
        args_cli.input_file,
        args_cli.input_fps,
        args_cli.output_fps,
        torch.device(sim.device),
    )
    print(
        f"[INFO] Loaded {motion['num_frames']} frames @ {args_cli.output_fps} FPS "
        f"(source: {args_cli.input_file})"
    )

    run(sim, scene, motion)
    simulation_app.close()


if __name__ == "__main__":
    main()
