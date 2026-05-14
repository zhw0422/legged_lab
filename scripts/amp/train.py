# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with AMP (Adversarial Motion Priors) using RSL-RL.

This script implements the AMP training loop directly, because the standard
``OnPolicyRunner.learn()`` does not call AMP-specific methods (style reward,
replay buffer, discriminator). The custom loop adds:

1. Record AMP observations before ``env.step``
2. Compute style reward via the discriminator after ``env.step``
3. Blend task and style rewards
4. Store (amp_obs, next_amp_obs) pairs in the replay buffer
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time

from isaaclab.app import AppLauncher

# add path of scripts/rsl_rl so we can reuse cli_args
import pathlib
scripts_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.insert(0, os.path.join(scripts_dir, "rsl_rl"))

# Add custom rsl_rl to sys.path to override the pip-installed version
custom_rsl_rl_dir = os.path.abspath(os.path.join(scripts_dir, "../source/legged_rl_lab/legged_rl_lab/rsl_rl"))
sys.path.insert(0, custom_rsl_rl_dir)

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with AMP using RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--motion_file", type=str, default=None, help="Path to reference motion data file or directory."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import torch

torch.backends.cuda.preferred_linalg_library("cusolver")
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner

# Inject AMPPPO into RSL-RL namespace so eval() finds it
from rsl_rl.algorithms.amp_ppo import AMPPPO
rsl_rl.runners.on_policy_runner.AMPPPO = AMPPPO

from rsl_rl.utils import check_nan

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
import legged_rl_lab  # noqa: F401 - Register custom environments
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from legged_rl_lab.envs import AmpRslRlVecEnvWrapper
from legged_rl_lab.managers import MotionLoader

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def amp_learn(runner: OnPolicyRunner, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    """AMP-aware training loop.

    This replaces ``runner.learn()`` with a loop that integrates AMP discriminator
    style rewards and replay buffer management into the standard on-policy rollout.
    """
    env = runner.env
    alg = runner.alg
    cfg = runner.cfg
    device = runner.device

    # Randomize initial episode lengths
    if init_at_random_ep_len:
        env.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )

    obs = env.get_observations().to(device)
    alg.train_mode()

    # Sync parameters for distributed training
    if runner.is_distributed:
        print(f"Synchronizing parameters for rank {runner.gpu_global_rank}...")
        alg.broadcast_parameters()

    runner.logger.init_logging_writer()

    num_steps_per_env = cfg["num_steps_per_env"]
    check_for_nan = cfg.get("check_for_nan", True)
    has_amp = alg.discriminator is not None

    start_it = runner.current_learning_iteration
    total_it = start_it + num_learning_iterations

    for it in range(start_it, total_it):
        start = time.time()

        # -- Rollout phase --
        with torch.inference_mode():
            ep_task_rewards = torch.zeros(env.num_envs, device=device)
            ep_style_rewards = torch.zeros(env.num_envs, device=device)

            for _ in range(num_steps_per_env):
                # 1. Sample actions from policy
                actions = alg.act(obs)

                # 2. Step the environment
                obs, rewards, dones, extras = env.step(actions.to(env.device))

                if check_for_nan:
                    try:
                        check_nan(obs, rewards, dones)
                    except ValueError as e:
                        import warnings
                        warnings.warn(
                            f"[rank {runner.gpu_global_rank}] NaN detected in env output: {e}. "
                            "Replacing NaN with zeros and continuing.",
                            RuntimeWarning,
                        )
                        for key in obs.keys():
                            obs[key] = torch.nan_to_num(obs[key], nan=0.0)
                        rewards = torch.nan_to_num(rewards, nan=0.0)
                        dones = torch.nan_to_num(dones, nan=1.0)

                obs, rewards, dones = (
                    obs.to(device),
                    rewards.to(device),
                    dones.to(device),
                )

                # 3. Single-window AMP processing.
                # We use the env's pre-reset AMP obs (extras["amp_obs"]) so the
                # discriminator sees the actual physics state, not post-reset
                # artifacts.  Style reward = D(amp_obs) × dt.
                if has_amp:
                    if "amp_obs" in extras and extras["amp_obs"] is not None:
                        amp_obs = extras["amp_obs"].to(device)
                    elif "amp" in obs.keys():
                        amp_obs = obs["amp"].clone()
                    else:
                        amp_obs = None

                    if amp_obs is not None:
                        # Style reward × env step_dt
                        style_rewards = alg.compute_style_reward(amp_obs, dt=env.unwrapped_env.cfg.sim.dt * env.unwrapped_env.cfg.decimation)
                        # Blend task + style
                        blended_rewards = alg.blend_rewards(rewards, style_rewards)
                        # Track for logging
                        ep_task_rewards += rewards
                        ep_style_rewards += style_rewards.view(-1)
                        # Insert single window into replay buffer
                        alg.process_amp_transition(amp_obs)
                        # Process env step with blended rewards
                        alg.process_env_step(obs, blended_rewards, dones, extras)
                    else:
                        alg.process_env_step(obs, rewards, dones, extras)
                else:
                    # No AMP, standard PPO
                    alg.process_env_step(obs, rewards, dones, extras)

                # Logging
                intrinsic_rewards = alg.intrinsic_rewards if cfg["algorithm"]["rnd_cfg"] else None
                runner.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

            stop = time.time()
            collect_time = stop - start
            start = stop

            # Compute returns (GAE)
            alg.compute_returns(obs)

        # -- Update phase --
        loss_dict = alg.update()

        stop = time.time()
        learn_time = stop - start
        runner.current_learning_iteration = it

        # -- Logging --
        runner.logger.log(
            it=it,
            start_it=start_it,
            total_it=total_it,
            collect_time=collect_time,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=alg.learning_rate,
            action_std=alg.get_policy().output_std,
            rnd_weight=alg.rnd.weight if cfg["algorithm"]["rnd_cfg"] else None,
        )

        # Print AMP-specific metrics
        if has_amp and it % 10 == 0:
            amp_keys = ["amp_loss", "grad_pen_loss", "disc_accuracy_policy", "disc_accuracy_expert"]
            amp_info = {k: f"{loss_dict[k]:.4f}" for k in amp_keys if k in loss_dict}
            # Also show average style reward magnitude for diagnostics
            if ep_style_rewards is not None:
                amp_info["mean_style_r"] = f"{ep_style_rewards.mean().item():.4f}"
            if amp_info:
                print(f"  [AMP] {amp_info}")

        # Save model
        if runner.logger.writer is not None and it % cfg["save_interval"] == 0:
            runner.save(os.path.join(runner.logger.log_dir, f"model_{it}.pt"))

    # Save final model
    if runner.logger.writer is not None:
        runner.save(os.path.join(runner.logger.log_dir, f"model_{runner.current_learning_iteration}.pt"))
        runner.logger.stop_logging_writer()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with AMP-PPO agent."""
    # override configurations
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.log_dir = log_dir

    # Override motion file path if provided via CLI
    if args_cli.motion_file is not None and hasattr(env_cfg, "amp_motion_files"):
        env_cfg.amp_motion_files = args_cli.motion_file

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Print joint order
    print("\n" + "=" * 70)
    print("ROBOT JOINT ORDER (IsaacLab BFS order)")
    print("=" * 70)
    robot = env.unwrapped.scene["robot"]
    print(f"Total joints: {robot.num_joints}")
    for i, name in enumerate(robot.data.joint_names):
        print(f"  [{i:2d}] {name}")
    print("=" * 70 + "\n")

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Use AMP-specific wrapper instead of standard RslRlVecEnvWrapper
    env = AmpRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions, amp_obs_group="amp")

    # Map policy config to actor/critic configs for custom RSL-RL
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

    # Create runner
    runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # Load reference motion data and set on algorithm
    motion_path = getattr(env_cfg, "amp_motion_files", "")
    robot_type = getattr(env_cfg, "robot_type", "g1")
    amp_history_length = getattr(
        getattr(getattr(env_cfg, "observations", None), "amp", None),
        "history_length",
        2,
    )
    if motion_path and os.path.exists(motion_path):
        print(f"[INFO] Loading reference motion data from: {motion_path}")
        motion_loader = MotionLoader(device=agent_cfg.device, robot=robot_type)
        reference_data = motion_loader.load(motion_path)
        print(f"[INFO] Loaded {motion_loader.num_frames} frames, obs_dim={motion_loader.obs_dim}, history_length={amp_history_length}")
        runner.alg.set_reference_data(reference_data, history_length=amp_history_length)
    else:
        print(f"[WARNING] No reference motion data found at: {motion_path}")
        print("[WARNING] AMP will train without style reward (task reward only).")

    # Load checkpoint
    if agent_cfg.resume:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # Dump config
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Run AMP-aware training loop
    amp_learn(runner, num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Close
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
