# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Cross-Procedural MDP: mixed biped+quadruped dispatch, obs encoders, action terms.

This module provides everything needed to train a **single cross-embodied
policy** on a heterogeneous mix of procedurally generated bipeds (26 DOF)
and quadrupeds (12 DOF).

Public API
----------
Data structures:
    ObsLayout, CrossEmbodiedEncoderCfg

Encoders (nn.Module):
    MaskEncoder, TransformerObsEncoder, GCNObsEncoder, build_obs_encoder()

Environment wrapper:
    CrossProceduralEnv     – thin ManagerBasedRLEnv wrapper with type annotations

rsl_rl integration:
    ActorCriticWithEncoder – marker; real class built by register_in_rsl_rl()
    register_in_rsl_rl()   – monkey-patches rsl_rl.modules

Mixed dispatch functions:
    *_mixed  (obs, reward, termination)

Action term:
    ProceduralMixedJointPosAction, ProceduralMixedJointPosActionCfg

Environment class:
    ProceduralMixedRobotEnv – mixed biped+quad env with parking
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_BIPED_DOF: int = 26
_QUAD_DOF: int = 12
_PARK_Z: float = 100.0  # parking altitude (m above env origin)

# Biped joint indices (alphabetical order within IsaacLab articulation)
# left:  ankle_pitch(0) ankle_roll(1) elbow(2) hip_pitch(3) hip_roll(4)
#        hip_yaw(5) knee(6) sh_pitch(7) sh_roll(8) sh_yaw(9)
#        wr_pitch(10) wr_roll(11) wr_yaw(12)
# right: ankle_pitch(13) ... wrist_yaw(25)
_BIPED_HIP_ROLL_IDX = [4, 17]
_BIPED_HIP_YAW_IDX = [5, 18]
_BIPED_LEG_DEV_IDX = (4, 5, 17, 18)
_BIPED_ARM_DEV_IDX = (2, 7, 8, 9, 10, 11, 12, 15, 20, 21, 22, 23, 24, 25)


# ═══════════════════════════════════════════════════════════════════════════
# ObsLayout
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ObsLayout:
    """Describes the flat-vector layout of a mixed-env observation."""

    state_dim: int = 9        # ang_vel(3) + gravity(3) + cmd(3)
    n_joints: int = _BIPED_DOF  # unified joint-space size (26)
    morph_dim: int = 11       # morphology parameter size
    height_dim: int = 0       # height-scan size (0 on flat terrain)

    @property
    def joint_pos_start(self) -> int:
        return self.state_dim

    @property
    def joint_vel_start(self) -> int:
        return self.state_dim + self.n_joints

    @property
    def action_start(self) -> int:
        return self.state_dim + 2 * self.n_joints

    @property
    def morph_start(self) -> int:
        return self.state_dim + 3 * self.n_joints

    @property
    def height_start(self) -> int:
        return self.morph_start + self.morph_dim

    @property
    def total_dim(self) -> int:
        return self.state_dim + 3 * self.n_joints + self.morph_dim + self.height_dim

    @property
    def per_joint_feature_dim(self) -> int:
        """Per-joint token dim: state + (pos, vel, act) + morph."""
        return self.state_dim + 3 + self.morph_dim


# ═══════════════════════════════════════════════════════════════════════════
# CrossEmbodiedEncoderCfg
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CrossEmbodiedEncoderCfg:
    """Configuration for the cross-embodied observation encoder."""

    type: str = "mask"         # "mask" | "transformer" | "gcn"
    latent_dim: int = 256

    # ObsLayout fields (flattened here for convenient configclass usage)
    state_dim: int = 9
    n_joints: int = _BIPED_DOF
    morph_dim: int = 11
    height_dim: int = 0

    # Transformer-specific
    tf_d_model: int = 64
    tf_n_heads: int = 4
    tf_n_layers: int = 3
    tf_dropout: float = 0.0

    # GCN-specific
    gcn_d_model: int = 64
    gcn_n_layers: int = 2

    def to_obs_layout(self) -> ObsLayout:
        return ObsLayout(
            state_dim=self.state_dim,
            n_joints=self.n_joints,
            morph_dim=self.morph_dim,
            height_dim=self.height_dim,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Encoder modules
# ═══════════════════════════════════════════════════════════════════════════

class MaskEncoder(nn.Module):
    """Identity encoder — passes observations through unchanged."""

    def __init__(self, cfg: CrossEmbodiedEncoderCfg):
        super().__init__()
        self.output_dim = cfg.to_obs_layout().total_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = x + self.attn(h, h, h, need_weights=False)[0]
        h = h + self.ff(self.norm2(h))
        return h


class TransformerObsEncoder(nn.Module):
    """Joint-token transformer encoder: tokenise → attend → pool → MLP."""

    def __init__(self, cfg: CrossEmbodiedEncoderCfg):
        super().__init__()
        layout = cfg.to_obs_layout()
        self.layout = layout
        d = cfg.tf_d_model

        self.input_proj = nn.Linear(layout.per_joint_feature_dim, d)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(d, cfg.tf_n_heads, cfg.tf_dropout)
             for _ in range(cfg.tf_n_layers)]
        )
        pool_in = d + layout.height_dim  # height appended after pooling
        self.head = nn.Sequential(nn.Linear(pool_in, cfg.latent_dim), nn.ELU())
        self.output_dim = cfg.latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.layout
        state = x[:, :L.state_dim]
        jpos = x[:, L.joint_pos_start:L.joint_vel_start]
        jvel = x[:, L.joint_vel_start:L.action_start]
        act = x[:, L.action_start:L.morph_start]
        morph = x[:, L.morph_start:L.height_start]

        # Per-joint tokens: (B, n_joints, feat_dim)
        state_exp = state.unsqueeze(1).expand(-1, L.n_joints, -1)
        morph_exp = morph.unsqueeze(1).expand(-1, L.n_joints, -1)
        tokens = torch.cat([
            state_exp,
            jpos.unsqueeze(-1), jvel.unsqueeze(-1), act.unsqueeze(-1),
            morph_exp,
        ], dim=-1)

        h = self.input_proj(tokens)
        for blk in self.blocks:
            h = blk(h)
        pooled = h.mean(dim=1)  # (B, d)

        if L.height_dim > 0:
            pooled = torch.cat([pooled, x[:, L.height_start:]], dim=-1)
        return self.head(pooled)


class _GCNLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_in), adj: (N, N) normalised adjacency
        h = self.linear(x)
        h = torch.bmm(adj.unsqueeze(0).expand(x.shape[0], -1, -1), h)
        return F.elu(self.norm(h))


def _build_biped_adjacency(n: int = _BIPED_DOF) -> torch.Tensor:
    """26-DOF biped kinematic tree adjacency (alphabetical index order)."""
    adj = torch.eye(n)
    edges = [
        # Left leg chain
        (1, 0), (0, 6), (6, 5), (5, 4), (4, 3),
        # Left arm chain
        (12, 11), (11, 10), (10, 2), (2, 9), (9, 8), (8, 7),
        # Right leg chain
        (14, 13), (13, 19), (19, 18), (18, 17), (17, 16),
        # Right arm chain
        (25, 24), (24, 23), (23, 15), (15, 22), (22, 21), (21, 20),
        # Torso connections (hip_pitch ↔ shoulder_pitch, cross-body)
        (3, 7), (3, 16), (3, 20), (7, 16), (7, 20), (16, 20),
    ]
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    deg = adj.sum(dim=1).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


def _build_quad_adjacency_padded(n: int = _BIPED_DOF) -> torch.Tensor:
    """12-DOF quad kinematic tree padded to *n* (isolated nodes beyond 12).

    IsaacLab sorts joints alphabetically, so within each leg group of 3 joints
    (prefix FL/FR/RL/RR) the order is: calf(+0) < hip(+1) < thigh(+2).
    Kinematic chain: body → hip → thigh → calf (root-to-tip).
    """
    adj = torch.eye(n)
    # Per-leg chain (alphabetical order: calf=+0, hip=+1, thigh=+2)
    # Connections: hip ↔ thigh and thigh ↔ calf
    for leg_base in (0, 3, 6, 9):
        # hip(+1) ↔ thigh(+2): adjacent in kinematic chain
        adj[leg_base + 1, leg_base + 2] = 1.0
        adj[leg_base + 2, leg_base + 1] = 1.0
        # thigh(+2) ↔ calf(+0): adjacent in kinematic chain
        adj[leg_base + 2, leg_base] = 1.0
        adj[leg_base, leg_base + 2] = 1.0
    # Base connections (all hip joints share the base body)
    hips = [1, 4, 7, 10]
    for i in range(len(hips)):
        for j in range(i + 1, len(hips)):
            adj[hips[i], hips[j]] = 1.0
            adj[hips[j], hips[i]] = 1.0
    deg = adj.sum(dim=1).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


class GCNObsEncoder(nn.Module):
    """Graph convolutional encoder with biped/quad adjacency dispatch."""

    def __init__(self, cfg: CrossEmbodiedEncoderCfg):
        super().__init__()
        layout = cfg.to_obs_layout()
        self.layout = layout
        d = cfg.gcn_d_model

        self.input_proj = nn.Linear(layout.per_joint_feature_dim, d)
        self.layers = nn.ModuleList(
            [_GCNLayer(d, d) for _ in range(cfg.gcn_n_layers)]
        )
        pool_in = d + layout.height_dim
        self.head = nn.Sequential(nn.Linear(pool_in, cfg.latent_dim), nn.ELU())
        self.output_dim = cfg.latent_dim

        self.register_buffer("adj_biped", _build_biped_adjacency())
        self.register_buffer("adj_quad", _build_quad_adjacency_padded())

    def _tokenise(self, x: torch.Tensor) -> torch.Tensor:
        L = self.layout
        state = x[:, :L.state_dim]
        jpos = x[:, L.joint_pos_start:L.joint_vel_start]
        jvel = x[:, L.joint_vel_start:L.action_start]
        act = x[:, L.action_start:L.morph_start]
        morph = x[:, L.morph_start:L.height_start]
        state_exp = state.unsqueeze(1).expand(-1, L.n_joints, -1)
        morph_exp = morph.unsqueeze(1).expand(-1, L.n_joints, -1)
        return torch.cat([
            state_exp,
            jpos.unsqueeze(-1), jvel.unsqueeze(-1), act.unsqueeze(-1),
            morph_exp,
        ], dim=-1)

    def forward(self, x: torch.Tensor, is_biped: torch.Tensor | None = None) -> torch.Tensor:
        L = self.layout
        # Auto-infer is_biped from the robot-type token in morphology params
        # (morph[0] = +1.0 for biped, -1.0 for quad — set by setup helpers)
        if is_biped is None:
            is_biped = x[:, L.morph_start] > 0
        tokens = self._tokenise(x)
        h = self.input_proj(tokens)  # (B, n_joints, d)

        if is_biped is not None and not is_biped.all() and is_biped.any():
            # Mixed batch: process biped / quad subsets separately
            out = torch.zeros(x.shape[0], h.shape[-1], device=x.device)
            bp = is_biped
            for subset, adj in [(bp, self.adj_biped), (~bp, self.adj_quad)]:
                if subset.any():
                    h_sub = h[subset]
                    for layer in self.layers:
                        h_sub = layer(h_sub, adj)
                    out[subset] = h_sub.mean(dim=1)
            pooled = out
        else:
            adj = self.adj_biped if (is_biped is None or is_biped.all()) else self.adj_quad
            for layer in self.layers:
                h = layer(h, adj)
            pooled = h.mean(dim=1)

        if L.height_dim > 0:
            pooled = torch.cat([pooled, x[:, L.height_start:]], dim=-1)
        return self.head(pooled)


def build_obs_encoder(cfg: CrossEmbodiedEncoderCfg) -> nn.Module:
    """Factory: build the observation encoder from *cfg*."""
    t = cfg.type
    if t == "mask":
        return MaskEncoder(cfg)
    elif t == "transformer":
        return TransformerObsEncoder(cfg)
    elif t == "gcn":
        return GCNObsEncoder(cfg)
    raise ValueError(f"Unknown encoder type: {t!r}")


# ═══════════════════════════════════════════════════════════════════════════
# CrossProceduralEnv  (thin wrapper for type annotations)
# ═══════════════════════════════════════════════════════════════════════════

class CrossProceduralEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with typed attributes for cross-procedural fields.

    Subclasses populate these in ``__init__`` via the setup helpers.
    """

    morphology_params_tensor: torch.Tensor | None
    morphology_params_dim: int
    is_humanoid_env: torch.Tensor


# ═══════════════════════════════════════════════════════════════════════════
# ActorCriticWithEncoder + rsl_rl registration
# ═══════════════════════════════════════════════════════════════════════════

_ACWE_CLASS = None  # lazy-built class cache


def _build_acwe_class():
    """Lazily build the real ActorCriticWithEncoder (subclass of rsl_rl ActorCritic)."""
    global _ACWE_CLASS
    if _ACWE_CLASS is not None:
        return _ACWE_CLASS

    from rsl_rl.modules import ActorCritic as _AC

    class _ActorCriticWithEncoder(_AC):
        """ActorCritic with a pluggable observation encoder for the actor."""

        def __init__(
            self,
            num_actor_obs: int,
            num_critic_obs: int,
            num_actions: int,
            actor_hidden_dims: list = [256, 256, 256],
            critic_hidden_dims: list = [256, 256, 256],
            activation: str = "elu",
            init_noise_std: float = 1.0,
            encoder_cfg=None,
            **kwargs,
        ):
            # Convert dict to dataclass if needed
            if isinstance(encoder_cfg, dict):
                encoder_cfg = CrossEmbodiedEncoderCfg(**encoder_cfg)

            if encoder_cfg is not None:
                self._obs_encoder = build_obs_encoder(encoder_cfg)
                effective_actor = self._obs_encoder.output_dim
            else:
                self._obs_encoder = None
                effective_actor = num_actor_obs

            # Critic always uses raw privileged obs (no encoding)
            super().__init__(
                effective_actor,
                num_critic_obs,
                num_actions,
                actor_hidden_dims,
                critic_hidden_dims,
                activation,
                init_noise_std,
                **kwargs,
            )
            self._raw_num_actor_obs = num_actor_obs

        def act(self, observations, **kwargs):
            if self._obs_encoder is not None:
                observations = self._obs_encoder(observations)
            return super().act(observations, **kwargs)

        def act_inference(self, observations):
            if self._obs_encoder is not None:
                observations = self._obs_encoder(observations)
            return super().act_inference(observations)

    _ACWE_CLASS = _ActorCriticWithEncoder
    return _ACWE_CLASS


class ActorCriticWithEncoder:
    """Marker class for type hints. Real class built by :func:`register_in_rsl_rl`."""
    pass


def register_in_rsl_rl() -> type:
    """Monkey-patch ``ActorCriticWithEncoder`` into ``rsl_rl.modules``."""
    import rsl_rl.modules
    cls = _build_acwe_class()
    rsl_rl.modules.ActorCriticWithEncoder = cls
    return cls


# ═══════════════════════════════════════════════════════════════════════════
# Mixed observation dispatch functions
# ═══════════════════════════════════════════════════════════════════════════
#
# Convention: ``env.scene["robot"]`` is the *biped* articulation,
# ``env.scene["quad_robot"]`` is the *quadruped* articulation.
# ``env.is_humanoid_env`` is a (N,) bool mask (True → biped active).


def base_ang_vel_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Body-frame angular velocity from the active robot.  Shape: (N, 3)."""
    b = env.scene["robot"].data.root_ang_vel_b
    q = env.scene["quad_robot"].data.root_ang_vel_b
    return torch.where(env.is_humanoid_env.unsqueeze(-1), b, q)


def projected_gravity_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Projected gravity from the active robot.  Shape: (N, 3)."""
    b = env.scene["robot"].data.projected_gravity_b
    q = env.scene["quad_robot"].data.projected_gravity_b
    return torch.where(env.is_humanoid_env.unsqueeze(-1), b, q)


def base_lin_vel_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Body-frame linear velocity (privileged).  Shape: (N, 3)."""
    b = env.scene["robot"].data.root_lin_vel_b
    q = env.scene["quad_robot"].data.root_lin_vel_b
    return torch.where(env.is_humanoid_env.unsqueeze(-1), b, q)


def joint_pos_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative joint positions, zero-padded to 26 dims.  Shape: (N, 26)."""
    biped_pos = (
        env.scene["robot"].data.joint_pos
        - env.scene["robot"].data.default_joint_pos
    )
    quad_pos = (
        env.scene["quad_robot"].data.joint_pos
        - env.scene["quad_robot"].data.default_joint_pos
    )
    quad_padded = F.pad(quad_pos, (0, _BIPED_DOF - _QUAD_DOF))
    return torch.where(env.is_humanoid_env.unsqueeze(-1), biped_pos, quad_padded)


def joint_vel_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint velocities, zero-padded to 26 dims.  Shape: (N, 26)."""
    biped_vel = env.scene["robot"].data.joint_vel
    quad_vel = env.scene["quad_robot"].data.joint_vel
    quad_padded = F.pad(quad_vel, (0, _BIPED_DOF - _QUAD_DOF))
    return torch.where(env.is_humanoid_env.unsqueeze(-1), biped_vel, quad_padded)


def last_action_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last applied action; inactive dims zeroed for quads.  Shape: (N, 26)."""
    action = env.action_manager.action.clone()
    action[~env.is_humanoid_env, _QUAD_DOF:] = 0.0
    return action


# ═══════════════════════════════════════════════════════════════════════════
# Mixed reward dispatch functions
# ═══════════════════════════════════════════════════════════════════════════


def track_lin_vel_xy_exp_mixed(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
) -> torch.Tensor:
    """Velocity-tracking reward.  Biped: yaw-frame; Quad: body-frame."""
    cmd = env.command_manager.get_command(command_name)[:, :2]

    # Biped: yaw-frame
    biped = env.scene["robot"]
    yaw_q = yaw_quat(biped.data.root_quat_w)
    biped_vel = quat_apply_inverse(yaw_q, biped.data.root_lin_vel_w[:, :3])[:, :2]
    biped_err = torch.sum(torch.square(cmd - biped_vel), dim=1)

    # Quad: body-frame
    quad_vel = env.scene["quad_robot"].data.root_lin_vel_b[:, :2]
    quad_err = torch.sum(torch.square(cmd - quad_vel), dim=1)

    error = torch.where(env.is_humanoid_env, biped_err, quad_err)
    return torch.exp(-error / std**2)


def track_ang_vel_z_exp_mixed(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.5,
) -> torch.Tensor:
    """Yaw-rate tracking.  Biped: world-frame; Quad: body-frame."""
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]

    biped_yaw = env.scene["robot"].data.root_ang_vel_w[:, 2]
    quad_yaw = env.scene["quad_robot"].data.root_ang_vel_b[:, 2]

    error = torch.where(
        env.is_humanoid_env,
        torch.square(cmd_yaw - biped_yaw),
        torch.square(cmd_yaw - quad_yaw),
    )
    return torch.exp(-error / std**2)


def ang_vel_xy_l2_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Roll/pitch rate penalty."""
    b = env.scene["robot"].data.root_ang_vel_b[:, :2]
    q = env.scene["quad_robot"].data.root_ang_vel_b[:, :2]
    w = torch.where(env.is_humanoid_env.unsqueeze(-1), b, q)
    return torch.sum(torch.square(w), dim=1)


def flat_orientation_l2_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Projected-gravity XY penalty (penalises tilting)."""
    b = env.scene["robot"].data.projected_gravity_b[:, :2]
    q = env.scene["quad_robot"].data.projected_gravity_b[:, :2]
    g = torch.where(env.is_humanoid_env.unsqueeze(-1), b, q)
    return torch.sum(torch.square(g), dim=1)


def body_roll_l2_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Roll penalty (projected gravity x-component)."""
    b_roll = torch.square(env.scene["robot"].data.projected_gravity_b[:, 0])
    q_roll = torch.square(env.scene["quad_robot"].data.projected_gravity_b[:, 0])
    return torch.where(env.is_humanoid_env, b_roll, q_roll)


def joint_torques_l2_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sum of squared joint torques for the active robot."""
    b = torch.sum(torch.square(env.scene["robot"].data.applied_torque), dim=1)
    q = torch.sum(torch.square(env.scene["quad_robot"].data.applied_torque), dim=1)
    return torch.where(env.is_humanoid_env, b, q)


def joint_vel_l2_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sum of squared joint velocities."""
    b = torch.sum(torch.square(env.scene["robot"].data.joint_vel), dim=1)
    q = torch.sum(torch.square(env.scene["quad_robot"].data.joint_vel), dim=1)
    return torch.where(env.is_humanoid_env, b, q)


def joint_acc_l2_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sum of squared joint accelerations."""
    b = torch.sum(torch.square(env.scene["robot"].data.joint_acc), dim=1)
    q = torch.sum(torch.square(env.scene["quad_robot"].data.joint_acc), dim=1)
    return torch.where(env.is_humanoid_env, b, q)


def joint_pos_limits_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Soft joint-limit penalty for the active robot."""

    def _limits(name: str) -> torch.Tensor:
        r = env.scene[name]
        pos = r.data.joint_pos
        lo = r.data.soft_joint_pos_limits[..., 0]
        hi = r.data.soft_joint_pos_limits[..., 1]
        return torch.sum(torch.clamp(pos - hi, min=0.0) + torch.clamp(lo - pos, min=0.0), dim=1)

    return torch.where(env.is_humanoid_env, _limits("robot"), _limits("quad_robot"))


def action_rate_l2_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared action-rate penalty."""
    cur = env.action_manager.action
    if not hasattr(env, "_mixed_prev_action"):
        env._mixed_prev_action = cur.clone()
    diff = cur - env._mixed_prev_action
    env._mixed_prev_action = cur.clone()
    diff[~env.is_humanoid_env, _QUAD_DOF:] = 0.0
    return torch.sum(torch.square(diff), dim=1)


def stand_still_mixed(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.06,
) -> torch.Tensor:
    """Penalise joint deviation from default when command ≈ 0."""
    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    is_still = (cmd_norm < command_threshold).float()

    b_dev = torch.sum(
        torch.abs(env.scene["robot"].data.joint_pos - env.scene["robot"].data.default_joint_pos),
        dim=1,
    )
    q_dev = torch.sum(
        torch.abs(
            env.scene["quad_robot"].data.joint_pos
            - env.scene["quad_robot"].data.default_joint_pos
        ),
        dim=1,
    )
    return torch.where(env.is_humanoid_env, b_dev, q_dev) * is_still


def undesired_contacts_mixed(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalise non-foot contacts for the active robot."""
    sensor_biped = env.scene.sensors["contact_forces"]
    sensor_quad = env.scene.sensors["contact_forces_quad"]

    # Biped: penalise hip/knee/pelvis contacts
    if not hasattr(env, "_undesired_biped_ids"):
        env._undesired_biped_ids = sensor_biped.find_bodies(".*_(hip|knee|pelvis).*")[0]
    # Quad: penalise calf/thigh contacts
    if not hasattr(env, "_undesired_quad_ids"):
        env._undesired_quad_ids = sensor_quad.find_bodies(".*_(calf|thigh)")[0]

    def _max_contact(sensor, ids):
        if len(ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        return (sensor.data.net_forces_w[:, ids, :].norm(dim=-1).max(dim=-1).values > threshold).float()

    b_pen = _max_contact(sensor_biped, env._undesired_biped_ids)
    q_pen = _max_contact(sensor_quad, env._undesired_quad_ids)
    return torch.where(env.is_humanoid_env, b_pen, q_pen)


def feet_air_time_mixed(
    env: ManagerBasedRLEnv,
    threshold: float = 0.4,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Foot air-time reward: biped ankle_roll, quad _foot."""
    sensor_biped = env.scene.sensors["contact_forces"]
    sensor_quad = env.scene.sensors["contact_forces_quad"]

    if not hasattr(env, "_air_biped_ids"):
        env._air_biped_ids = sensor_biped.find_bodies(".*ankle_roll.*")[0]
    if not hasattr(env, "_air_quad_ids"):
        env._air_quad_ids = sensor_quad.find_bodies(".*_foot")[0]

    def _air_reward(sensor, ids):
        if len(ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        fc = sensor.compute_first_contact(env.step_dt)[:, ids]
        lat = sensor.data.last_air_time[:, ids]
        return torch.sum((lat - threshold) * fc, dim=1).clamp(min=0.0)

    b_rew = _air_reward(sensor_biped, env._air_biped_ids)
    q_rew = _air_reward(sensor_quad, env._air_quad_ids)

    rew = torch.where(env.is_humanoid_env, b_rew, q_rew)
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        rew = rew * (cmd_norm > 0.1).float()
    return rew


def feet_slide_mixed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Foot-sliding penalty for the active robot."""
    sensor_biped = env.scene.sensors["contact_forces"]
    sensor_quad = env.scene.sensors["contact_forces_quad"]
    asset_biped = env.scene["robot"]
    asset_quad = env.scene["quad_robot"]

    if not hasattr(env, "_slide_biped_body_ids"):
        env._slide_biped_body_ids = asset_biped.find_bodies(".*ankle_roll.*")[0]
        env._slide_biped_sensor_ids = sensor_biped.find_bodies(".*ankle_roll.*")[0]
    if not hasattr(env, "_slide_quad_body_ids"):
        env._slide_quad_body_ids = asset_quad.find_bodies(".*_foot")[0]
        env._slide_quad_sensor_ids = sensor_quad.find_bodies(".*_foot")[0]

    def _slide(sensor, s_ids, asset, b_ids):
        if len(b_ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        contacts = (
            sensor.data.net_forces_w_history[:, :, s_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        vel = asset.data.body_lin_vel_w[:, b_ids, :2]
        return torch.sum(vel.norm(dim=-1) * contacts, dim=1)

    b = _slide(sensor_biped, env._slide_biped_sensor_ids, asset_biped, env._slide_biped_body_ids)
    q = _slide(sensor_quad, env._slide_quad_sensor_ids, asset_quad, env._slide_quad_body_ids)
    return torch.where(env.is_humanoid_env, b, q)


# ═══════════════════════════════════════════════════════════════════════════
# Mixed termination dispatch
# ═══════════════════════════════════════════════════════════════════════════


def illegal_contact_base_mixed(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate on base/torso contact for the active robot.  Returns bool (N,)."""
    sensor_biped = env.scene.sensors["contact_forces"]
    sensor_quad = env.scene.sensors["contact_forces_quad"]

    if not hasattr(env, "_illegal_biped_ids"):
        env._illegal_biped_ids = sensor_biped.find_bodies("(torso_link|pelvis)")[0]
    if not hasattr(env, "_illegal_quad_ids"):
        env._illegal_quad_ids = sensor_quad.find_bodies("base")[0]

    def _max_force(sensor, ids):
        if len(ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        return sensor.data.net_forces_w[:, ids, :].norm(dim=-1).max(dim=-1).values

    b_f = _max_force(sensor_biped, env._illegal_biped_ids)
    q_f = _max_force(sensor_quad, env._illegal_quad_ids)
    force = torch.where(env.is_humanoid_env, b_f, q_f)
    return force > threshold


# ═══════════════════════════════════════════════════════════════════════════
# ProceduralMixedJointPosAction
# ═══════════════════════════════════════════════════════════════════════════


class ProceduralMixedJointPosAction(ActionTerm):
    """Routes a 26-dim action to the active robot per environment.

    - Biped envs: all 26 dims applied to ``robot``.
    - Quad envs: first 12 dims applied to ``quad_robot``; dims 12–25 discarded.
    - Inactive robot held at default joint targets.
    """

    cfg: "ProceduralMixedJointPosActionCfg"

    def __init__(self, cfg: "ProceduralMixedJointPosActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._biped = env.scene["robot"]
        self._quad = env.scene["quad_robot"]

        self._jids_biped, _ = self._biped.find_joints(".*")
        self._jids_quad, _ = self._quad.find_joints(".*")
        self._n_biped = len(self._jids_biped)  # 26
        self._n_quad = len(self._jids_quad)    # 12

        N = env.num_envs
        self._biped_targets = torch.zeros(N, self._n_biped, device=env.device)
        self._quad_targets = torch.zeros(N, self._n_quad, device=env.device)
        self._last_scaled = torch.zeros(N, self._n_biped, device=env.device)

    @property
    def action_dim(self) -> int:
        return self._n_biped  # 26

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._last_scaled

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._last_scaled

    def process_actions(self, actions: torch.Tensor) -> None:
        is_bp = self._env.is_humanoid_env
        scale = torch.where(
            is_bp,
            torch.full((actions.shape[0],), self.cfg.scale_biped, device=actions.device),
            torch.full((actions.shape[0],), self.cfg.scale_quad, device=actions.device),
        )
        scaled = actions * scale.unsqueeze(-1)

        self._biped_targets = self._biped.data.default_joint_pos + scaled[:, :self._n_biped]
        self._quad_targets = self._quad.data.default_joint_pos + scaled[:, :self._n_quad]

        self._last_scaled = scaled.clone()
        self._last_scaled[~is_bp, self._n_quad:] = 0.0

    def apply_actions(self) -> None:
        is_bp = self._env.is_humanoid_env
        bp_ids = is_bp.nonzero(as_tuple=False).view(-1)
        qd_ids = (~is_bp).nonzero(as_tuple=False).view(-1)

        # Active robot targets
        if len(bp_ids) > 0:
            self._biped.set_joint_position_target(
                self._biped_targets[bp_ids], joint_ids=self._jids_biped, env_ids=bp_ids,
            )
        if len(qd_ids) > 0:
            self._quad.set_joint_position_target(
                self._quad_targets[qd_ids], joint_ids=self._jids_quad, env_ids=qd_ids,
            )
        # Inactive robot: hold at defaults
        if len(qd_ids) > 0:
            self._biped.set_joint_position_target(
                self._biped.data.default_joint_pos[qd_ids],
                joint_ids=self._jids_biped, env_ids=qd_ids,
            )
        if len(bp_ids) > 0:
            self._quad.set_joint_position_target(
                self._quad.data.default_joint_pos[bp_ids],
                joint_ids=self._jids_quad, env_ids=bp_ids,
            )


@configclass
class ProceduralMixedJointPosActionCfg(ActionTermCfg):
    """Configuration for :class:`ProceduralMixedJointPosAction`."""

    class_type: type = ProceduralMixedJointPosAction
    # Joint-position action targets ``robot`` (biped) as the primary asset
    asset_name: str = "robot"
    scale_biped: float = 0.5
    scale_quad: float = 0.25


# ═══════════════════════════════════════════════════════════════════════════
# Setup helpers (moved from procedural_obs – cross-procedural only)
# ═══════════════════════════════════════════════════════════════════════════


def modify_mixed_procedural_articulations(env: ManagerBasedRLEnv) -> None:
    """Adjust joint limits for the two-entity mixed env."""
    for art_key, builder_name in (("robot", "BipedBuilder"), ("quad_robot", "QuadrupedBuilder")):
        if art_key not in env.scene.articulations:
            continue
        robot = env.scene.articulations[art_key]
        try:
            from metamorphosis import builder as _b
            BuilderCls = getattr(_b, builder_name)
            instance = BuilderCls.get_instance()
            if instance is not None and len(instance.params) > 0 and hasattr(BuilderCls, "modify_articulation"):
                BuilderCls.modify_articulation(robot)
                print(f"[INFO] Modified {art_key!r} via {builder_name} ({len(instance.params)} robots)")
        except (ImportError, AttributeError, RuntimeError, KeyError):
            pass


def setup_mixed_two_entity_morphology_params(env: ManagerBasedRLEnv) -> None:
    """Set 11-dim morphology params for the two-entity mixed environment.

    Biped envs → ``robot_type=+1`` + 10 normalised biped body dims.
    Quad envs  → ``robot_type=-1`` + 7 normalised quad body dims (zero-padded).
    """
    morph_dim = 11
    tensor = torch.zeros(env.num_envs, morph_dim, device=env.device)
    ib = env.is_humanoid_env

    tensor[ib, 0] = 1.0
    tensor[~ib, 0] = -1.0

    biped_indices = ib.nonzero(as_tuple=False).view(-1)
    quad_indices = (~ib).nonzero(as_tuple=False).view(-1)

    try:
        from metamorphosis.builder import BipedBuilder
        builder = BipedBuilder.get_instance()
        if builder is not None and len(builder.params) > 0:
            params_list = [
                [p.torso_link_length, p.torso_link_width, p.torso_link_height,
                 p.pelvis_height, p.hip_spacing,
                 p.hip_pitch_link_length, p.hip_roll_link_length, p.hip_yaw_link_length,
                 p.knee_link_length, p.ankle_roll_link_length]
                for p in builder.params
            ]
            raw = torch.tensor(params_list, device=env.device, dtype=torch.float32)
            mean = torch.tensor([[0.13, 0.22, 0.11, 0.065, 0.20, 0.045, 0.045, 0.30, 0.31, 0.215]], device=env.device)
            std = torch.tensor([[0.03, 0.04, 0.03, 0.015, 0.04, 0.015, 0.015, 0.08, 0.09, 0.035]], device=env.device)
            normed = (raw - mean) / std
            n = min(len(biped_indices), len(normed))
            tensor[biped_indices[:n], 1:11] = normed[:n]
    except (ImportError, RuntimeError, KeyError):
        pass

    try:
        from metamorphosis.builder import QuadrupedBuilder
        builder = QuadrupedBuilder.get_instance()
        if builder is not None and len(builder.params) > 0:
            params_list = [
                [p.base_length, p.base_width, p.base_height,
                 p.thigh_length, p.calf_length, p.thigh_radius,
                 float(p.parallel_abduction)]
                for p in builder.params
            ]
            raw = torch.tensor(params_list, device=env.device, dtype=torch.float32)
            mean = torch.tensor([[0.75, 0.35, 0.20, 0.50, 0.50, 0.04, 0.5]], device=env.device)
            std = torch.tensor([[0.25, 0.05, 0.05, 0.30, 0.30, 0.01, 0.5]], device=env.device)
            normed = (raw - mean) / std
            n = min(len(quad_indices), len(normed))
            tensor[quad_indices[:n], 1:8] = normed[:n]
    except (ImportError, RuntimeError, KeyError):
        pass

    env.morphology_params_tensor = tensor
    env.morphology_params_dim = morph_dim


def setup_mixed_morphology_params(env: ManagerBasedRLEnv) -> None:
    """Set 11-dim morphology params; BipedBuilder envs first, then QuadrupedBuilder."""
    morph_dim = 11
    tensor = torch.zeros(env.num_envs, morph_dim, device=env.device)
    n_biped = 0

    try:
        from metamorphosis.builder import BipedBuilder
        builder = BipedBuilder.get_instance()
        if builder is not None and len(builder.params) > 0:
            params_list = [
                [p.torso_link_length, p.torso_link_width, p.torso_link_height,
                 p.pelvis_height, p.hip_spacing,
                 p.hip_pitch_link_length, p.hip_roll_link_length, p.hip_yaw_link_length,
                 p.knee_link_length, p.ankle_roll_link_length]
                for p in builder.params
            ]
            raw = torch.tensor(params_list, device=env.device, dtype=torch.float32)
            mean = torch.tensor([[0.13, 0.22, 0.11, 0.065, 0.20, 0.045, 0.045, 0.30, 0.31, 0.215]], device=env.device)
            std = torch.tensor([[0.03, 0.04, 0.03, 0.015, 0.04, 0.015, 0.015, 0.08, 0.09, 0.035]], device=env.device)
            n_biped = len(builder.params)
            tensor[:n_biped, 0] = 1.0
            tensor[:n_biped, 1:11] = (raw - mean) / std
    except (ImportError, RuntimeError, KeyError):
        pass

    try:
        from metamorphosis.builder import QuadrupedBuilder
        builder = QuadrupedBuilder.get_instance()
        if builder is not None and len(builder.params) > 0:
            params_list = [
                [p.base_length, p.base_width, p.base_height,
                 p.thigh_length, p.calf_length, p.thigh_radius,
                 float(p.parallel_abduction)]
                for p in builder.params
            ]
            raw = torch.tensor(params_list, device=env.device, dtype=torch.float32)
            mean = torch.tensor([[0.75, 0.35, 0.20, 0.50, 0.50, 0.04, 0.5]], device=env.device)
            std = torch.tensor([[0.25, 0.05, 0.05, 0.30, 0.30, 0.01, 0.5]], device=env.device)
            n_quad = len(builder.params)
            tensor[n_biped:n_biped + n_quad, 0] = -1.0
            tensor[n_biped:n_biped + n_quad, 1:8] = (raw - mean) / std
    except (ImportError, RuntimeError, KeyError):
        pass

    env.morphology_params_tensor = tensor
    env.morphology_params_dim = morph_dim
    env.is_humanoid_env = tensor[:, 0] > 0


# ═══════════════════════════════════════════════════════════════════════════
# ProceduralMixedRobotEnv
# ═══════════════════════════════════════════════════════════════════════════


class ProceduralMixedRobotEnv(CrossProceduralEnv):
    """Mixed environment with both procedural biped and quadruped robots.

    Two homogeneous ``ArticulationView`` objects are spawned in every env:

    * ``scene["robot"]``       – biped (26 DOF); active in biped envs.
    * ``scene["quad_robot"]``  – quad  (12 DOF); active in quad  envs.

    The inactive robot is parked at ``_PARK_Z`` m above the env origin
    and held at default joint positions every :meth:`step`.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        # Register ActorCriticWithEncoder in rsl_rl before runner creates policy
        register_in_rsl_rl()

        # Determine biped / quad split
        n = cfg.scene.num_envs
        ratio = getattr(cfg, "humanoid_ratio", 0.5)
        n_humanoid = int(n * ratio)
        device = cfg.sim.device

        self.is_humanoid_env = torch.zeros(n, dtype=torch.bool, device=device)
        self.is_humanoid_env[:n_humanoid] = True

        self.morphology_params_tensor = None
        self.morphology_params_dim = 0

        super().__init__(cfg, render_mode, **kwargs)

        # Initial parking
        biped_ids = self.is_humanoid_env.nonzero(as_tuple=False).view(-1)
        quad_ids = (~self.is_humanoid_env).nonzero(as_tuple=False).view(-1)
        self._park_robot("quad_robot", biped_ids)
        self._park_robot("robot", quad_ids)

        # Articulation adjustments + morphology
        modify_mixed_procedural_articulations(self)
        setup_mixed_two_entity_morphology_params(self)

    # ── Parking ──────────────────────────────────────────────────────────

    def _park_robot(self, robot_name: str, env_ids: torch.Tensor) -> None:
        """Teleport *robot_name* to altitude and hold at default joints."""
        if len(env_ids) == 0:
            return
        robot = self.scene[robot_name]
        pos = self.scene.env_origins[env_ids].clone()
        pos[:, 2] = _PARK_Z

        quat = torch.zeros(len(env_ids), 4, device=self.device)
        quat[:, 0] = 1.0
        vel = torch.zeros(len(env_ids), 6, device=self.device)

        robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        robot.write_root_velocity_to_sim(vel, env_ids=env_ids)
        robot.set_joint_position_target(robot.data.default_joint_pos[env_ids], env_ids=env_ids)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos[env_ids],
            torch.zeros_like(robot.data.default_joint_pos[env_ids]),
            env_ids=env_ids,
        )

    # ── Per-step re-parking ──────────────────────────────────────────────

    def step(self, action: torch.Tensor):
        all_ids = torch.arange(self.num_envs, device=self.device)
        bp_ids = all_ids[self.is_humanoid_env]
        qd_ids = all_ids[~self.is_humanoid_env]
        if len(bp_ids) > 0:
            self._park_robot("quad_robot", bp_ids)
        if len(qd_ids) > 0:
            self._park_robot("robot", qd_ids)
        return super().step(action)

    # ── Reset override ───────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        bp_ids = env_ids[self.is_humanoid_env[env_ids]]
        qd_ids = env_ids[~self.is_humanoid_env[env_ids]]
        if len(bp_ids) > 0:
            self._park_robot("quad_robot", bp_ids)
        if len(qd_ids) > 0:
            self._park_robot("robot", qd_ids)
