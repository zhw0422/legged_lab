from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlTsDepthActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticTSDepth"
    num_latent_dims: int = 64
    depth_shape: tuple[int, int, int] = (2, 32, 48)
    num_student_envs: int | None = None
    privilege_encoder_hidden_dims: list[int] = [256, 128]
    cnn_input_channel: int = 2
    cnn_channel_dims: list[int] = [8, 8]
    cnn_strides: list[int] = [1, 1]
    cnn_fc_layer_dims: list[int] = [128, 64]
    combination_mlp_dims: list[int] = [128, 32]
    cnn_kernel_sizes: list[int] = [5, 3]
    rnn_type: str = "gru"
    rnn_hidden_size: int = 512
    rnn_num_layers: int = 1
    clip_actions: float = 100.0


@configclass
class RslRlTsDepthAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPO_TSDepth"
    encoder_lr: float = 2e-4
    distillation: bool = False
    teacher_checkpoint_path: str = ""


@configclass
class RslRlAttentionActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "AttentionTerrainModel"
    map_scan_dim: tuple[int, int, int] = (33, 21, 3)
    mha_dim: int = 64
    num_heads: int = 16
    cnn_downsample: bool = True
    attach_global: bool = False
