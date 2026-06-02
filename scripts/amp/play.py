# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Script to play/visualise a trained AMP-PPO policy."""
import uuid
uuid_value = uuid.uuid4()

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add path of scripts/rsl_rl so we can reuse cli_args
import os
import pathlib
scripts_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.insert(0, os.path.join(scripts_dir, "rsl_rl"))

# Add custom rsl_rl to sys.path to override the pip-installed version
custom_rsl_rl_dir = os.path.abspath(os.path.join(scripts_dir, "../source/legged_rl_lab/legged_rl_lab/rsl_rl"))
sys.path.insert(0, custom_rsl_rl_dir)
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained AMP agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Path to reference motion file or directory. Directories are searched recursively.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point key."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--ckpt", type=str, default=None, help="Name of a checkpoint file under the ckpt/ directory (e.g. 'model.pt').")
parser.add_argument("--export_only", action="store_true", default=False, help="Export policy to ONNX and exit.")
# append RSL-RL cli arguments (adds --checkpoint / --load_run etc.)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

# Use default linalg library selection — avoids CUSOLVER_STATUS_INTERNAL_ERROR
# that occurs when GPU memory is exhausted and cuSolver cannot initialize.
torch.backends.cuda.preferred_linalg_library("default")

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
import legged_rl_lab  # noqa: F401 - Register custom environments
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# AMP wrapper: extracts amp_obs from the "amp" observation group each step
from legged_rl_lab.envs import AmpRslRlVecEnvWrapper


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with AMP-PPO agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.ckpt:
        ckpt_path = os.path.join("ckpt", args_cli.ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        resume_path = os.path.abspath(ckpt_path)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    if hasattr(args_cli, "motion_file") and args_cli.motion_file is not None:
        env_cfg.amp_motion_files = args_cli.motion_file

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap with AMP-aware wrapper (extracts amp_obs into extras)
    env = AmpRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    agent_dict = agent_cfg.to_dict()
    
    if "policy" in agent_dict:
        policy_cfg = agent_dict.pop("policy")
        agent_dict["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("actor_obs_normalization", False),
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "std_type": policy_cfg.get("noise_std_type", "scalar"),
                "init_std": policy_cfg.get("init_noise_std", 1.0),
            },
        }
        agent_dict["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
        }

    amp_cfg = agent_dict.get("algorithm", {}).get("amp_cfg")
    if isinstance(amp_cfg, dict):
        # Play only needs policy inference; keep AMP training buffers from
        # consuming most of the GPU before the checkpoint can load.
        amp_cfg["amp_replay_buffer_size"] = 1
        amp_cfg["amp_num_preload_transitions"] = 1

    runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped_env.device)

    # extract policy nn module (rsl_rl >= 2.3)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor

    # export policy to jit / onnx for deployment
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")
    print(f"[INFO] Exported ONNX policy to: {os.path.join(export_model_dir, 'policy.onnx')}")

    if args_cli.export_only:
        env.close()
        return

    dt = env.unwrapped_env.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # real-time pacing
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
