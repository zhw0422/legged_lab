from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from legged_rl_lab.rsl_rl import RslRlAttentionActorCriticCfg
from legged_rl_lab.tasks.parkour.attention.attention_env_cfg import ATTENTION_MAP_SCAN_DIM, ATTENTION_OBS_GROUPS


@configclass
class G1AttentionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "g1_parkour_attention"
    obs_groups = ATTENTION_OBS_GROUPS
    policy = RslRlAttentionActorCriticCfg(
        class_name="AttentionTerrainModel",
        init_noise_std=0.8,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        map_scan_dim=ATTENTION_MAP_SCAN_DIM,
        mha_dim=64,
        num_heads=16,
        cnn_downsample=True,
        attach_global=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
