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

    policy = RslRlPpoActorCriticCfg(
        # TienKung uses scalar init_std=1.0 — gives broad exploration across
        # all joints.  Our earlier log-std with init=0.3 was too restrictive
        # and combined with low entropy made the policy collapse locally.
        noise_std_type="scalar",
        init_noise_std=1.0,
        # Running mean/var for policy / critic obs — stabilises value training.
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
        # TienKung uses 0.005 — enough to keep exploration alive, not so much
        # that action_std explodes.
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg={
            # Design-3 single-window discriminator:
            #   per-frame = 29+29+1+6+3+3+18 = 89
            #   history_length = 2, input dim to disc = 178
            # Moderate hidden dims (we no longer feed redundant pair of 10-frame
            # windows at 1220 dim, so 178 → [512, 256] is enough).
            "amp_discriminator_hidden_dims": [512, 256],
            "amp_discriminator_activation": "relu",
            # Discriminator gets its own low LR to prevent saturation — legged_lab
            # uses 1e-4, well below policy LR of 1e-3.
            "amp_learning_rate": 1e-4,
            # Buffer holds single-window observations (not pairs).  100 iters
            # worth of rollout steps — legged_lab uses disc_obs_buffer_size=100
            # for similar effect.
            "amp_replay_buffer_size": 200000,
            # 30% of blended reward is task; 70% is style.  With style_reward
            # now being ``× dt``-scaled (see discriminator.predict_reward), the
            # per-step style magnitude is comparable to per-step task magnitude.
            "amp_task_reward_lerp": 0.3,
            # R1 grad penalty λ = 10 (legged_lab / IsaacLab default).
            "amp_disc_gradient_penalty_coef": 10.0,
            "amp_disc_logit_reg_coef": 0.0,
            "amp_disc_weight_decay": 0.0005,
            # Style reward scale in "per-second" units.  Actual per-step style
            # reward = scale × dt × clamp(..., 0, 1).  With dt=1/30 and scale=5,
            # per-step style ≤ 5/30 ≈ 0.17 — comparable to per-step task ~0.5.
            "amp_reward_scale": 5.0,
        },
    )
