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
            # TienKung-sized discriminator [1024, 512, 256] — big enough to
            # learn meaningful style differences, combined with:
            #   • no tanh on logits (raw D for LSGAN MSE)
            #   • no logit regularization
            #   • R1 grad penalty λ=10
            # This combo keeps the disc usefully bounded without saturating.
            "amp_discriminator_hidden_dims": [1024, 512, 256],
            "amp_discriminator_activation": "relu",
            # Match AC learning rate — discriminator updated alongside policy
            # in the same optimizer (weight decay differs per param group).
            "amp_learning_rate": 1e-3,
            "amp_replay_buffer_size": 1000000,
            # 30% of blended reward is task; 70% is style (TienKung default).
            # With style reward ∈ [0, reward_coef], bounded and dense, the
            # disc no longer has to carry all the signal alone.
            "amp_task_reward_lerp": 0.3,
            # R1 grad penalty λ (TienKung uses 10).
            "amp_disc_gradient_penalty_coef": 10.0,
            # Disable logit reg — TienKung uses 0.  Regularization belongs in
            # the gradient penalty, not in squashing logits toward 0 (which
            # flattens the reward surface).
            "amp_disc_logit_reg_coef": 0.0,
            "amp_disc_weight_decay": 0.0005,
            # Reward scale (TienKung "amp_reward_coef" = 0.3).  Dense per-step
            # bonus ≤0.3, well below task reward magnitude so task gradient
            # stays dominant while style nudges gait quality.
            "amp_reward_scale": 0.3,
        },
    )
