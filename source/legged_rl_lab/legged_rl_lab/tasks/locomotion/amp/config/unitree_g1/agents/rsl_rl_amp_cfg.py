# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL AMP-PPO agent configuration for Unitree G1 humanoid."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Import the base AMP algorithm config shared across robots
from legged_rl_lab.tasks.locomotion.amp.config.unitree_go2.agents.rsl_rl_amp_cfg import (
    RslRlAmpPpoAlgorithmCfg,
)


@configclass
class UnitreeG1AMPFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """AMP-PPO runner config for G1 humanoid flat terrain."""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "unitree_g1_amp_flat"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticCfg(
        noise_std_type="scalar",
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlAmpPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg={
            # Pair-style AMP aligned with TienKung-Lab:
            # policy / expert each provide a single-frame AMP feature vector,
            # and the discriminator sees explicit transition pairs
            # ``[s_t, s_{t+1}]``.
            "amp_discr_hidden_dims": [1024, 512],
            "amp_learning_rate": 1e-5,
            "amp_replay_buffer_size": 1000000,
            "amp_num_preload_transitions": 200000,
            "amp_reward_coef": 0.3,
            "amp_task_reward_lerp": 0.7,
            "amp_disc_gradient_penalty_coef": 5.0,
            "amp_disc_weight_decay": 0.001,
            "amp_disc_head_weight_decay": 0.1,
        },
    )
