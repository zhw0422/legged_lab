"""Curriculum terms for parkour attention tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_parkour(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_distance: float = 2.0,
    move_down_distance: float = 0.6,
) -> torch.Tensor:
    """Progress parkour terrain by actual forward progress.

    The shared velocity curriculum requires traversing half of the whole terrain
    tile.  Parkour tiles include local platforms and obstacles, so G1 attention
    uses a shorter progress gate to avoid getting stuck on the first rows.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    move_up = distance > move_up_distance
    move_down = (distance < move_down_distance) & ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
