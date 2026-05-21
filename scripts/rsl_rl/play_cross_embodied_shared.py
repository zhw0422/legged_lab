# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Play a shared G1 + Go2 cross-embodied policy in the mixed scene.

Both Unitree G1 (29 DOF) and Unitree Go2 (12 DOF) appear in the same scene.
The first ``num_envs // 2`` envs run G1; the remaining envs run Go2.
Requires ``num_envs >= 2`` so that at least one of each robot is visible.

Usage
-----
    # Basic play – 4 envs (2 G1 + 2 Go2), picks latest checkpoint automatically
    python scripts/rsl_rl/play_cross_embodied_shared.py --num_envs 4

    # Specify a checkpoint explicitly
    python scripts/rsl_rl/play_cross_embodied_shared.py \\
        --num_envs 4 \\
        --checkpoint logs/rsl_rl/cross_embodied_g1go2_flat/2026-03-23_12-00-00/model_1000.pt

    # Export JIT + ONNX policy
    python scripts/rsl_rl/play_cross_embodied_shared.py --num_envs 4 --export

    # Record video
    python scripts/rsl_rl/play_cross_embodied_shared.py --num_envs 4 --video
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Play a shared G1+Go2 cross-embodied policy (single process, mixed scene)."
)
parser.add_argument(
    "--num_envs", type=int, default=4,
    help="Total envs to spawn (must be >= 2; split equally between G1 and Go2).",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint file.")
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to load from.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--export", action="store_true", default=False, help="Export policy as JIT and ONNX.")
parser.add_argument("--real_time", action="store_true", default=False, help="Throttle loop to real-time speed.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video clip.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps.")
parser.add_argument("--ckpt", type=str, default=None, help="Name of a checkpoint file under the ckpt/ directory (e.g. 'model.pt').")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 2:
    raise ValueError("--num_envs must be >= 2 to display both G1 and Go2.")

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Suppress Go2 USD rotor-visuals warnings that flood stdout at scale.
# ---------------------------------------------------------------------------
# Suppress "Unresolved reference prim path" USD warnings from Go2 rotor visuals.
# go2_description_base.usd references visual prims absent in go2_description_physics.usd.
# set_channel_enabled(channel, False) disables the channel; passing Level.ERROR (int=3)
# is truthy and would ENABLE it — use explicit False instead.
try:
    import omni.log as _omni_log
    _omni_log.get_log().set_channel_enabled("omni.usd", False)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import legged_rl_lab  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK_ID = "LeggedRLLab-Isaac-CrossEmboided-Flat-G1Go2-Mixed-Play-v0"
_AGENT_ENTRY = "rsl_rl_cfg_entry_point"


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load cfgs
    # ------------------------------------------------------------------
    env_cfg = load_cfg_from_registry(_TASK_ID, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(_TASK_ID, _AGENT_ENTRY)

    # Override num_envs (minimum 2 for both robots to be visible)
    env_cfg.scene.num_envs = max(args_cli.num_envs, 2)
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # Disable noise / corruption for clean play
    env_cfg.observations.policy.enable_corruption = False

    # ------------------------------------------------------------------
    # 2. Resolve checkpoint
    # ------------------------------------------------------------------
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from: {log_root_path}")

    if args_cli.ckpt:
        ckpt_path = os.path.join("ckpt", args_cli.ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        resume_path = os.path.abspath(ckpt_path)
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            args_cli.load_run or agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Checkpoint: {resume_path}")

    # ------------------------------------------------------------------
    # 3. Create environment
    # ------------------------------------------------------------------
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(_TASK_ID, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video to:", video_kwargs["video_folder"])
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Print layout info
    n = env_cfg.scene.num_envs
    n_g1 = n // 2
    n_go2 = n - n_g1
    print(f"\n[INFO] Scene layout:")
    print(f"       Total envs : {n}  ({n_g1} G1  +  {n_go2} Go2)")
    print(f"       G1  env ids: 0 .. {n_g1 - 1}")
    print(f"       Go2 env ids: {n_g1} .. {n - 1}\n")

    # ------------------------------------------------------------------
    # 4. Build runner and load checkpoint
    # ------------------------------------------------------------------
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    print(f"[INFO] Loaded checkpoint: {resume_path}")

    # ------------------------------------------------------------------
    # 5. Export policy (optional)
    # ------------------------------------------------------------------
    if args_cli.export:
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        normalizer = None
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer

        export_dir = os.path.join(log_dir, "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
        print(f"[INFO] Exported JIT + ONNX to: {export_dir}")

    # ------------------------------------------------------------------
    # 6. Inference loop
    # ------------------------------------------------------------------
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    print("[INFO] Starting play loop. Close the viewer window to exit.")
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
