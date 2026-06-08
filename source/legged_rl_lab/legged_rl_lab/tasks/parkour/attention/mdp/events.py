"""Event terms for parkour attention tasks."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def push_by_setting_velocity_record_xy(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Push the robot and expose the sampled xy velocity delta to the critic."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)

    if not hasattr(env, "_attention_push_xy"):
        env._attention_push_xy = torch.zeros(env.scene.num_envs, 2, device=asset.device)

    vel_w = asset.data.root_vel_w[env_ids]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")]
    ranges = torch.tensor(range_list, device=asset.device)
    delta = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)

    vel_w = vel_w + delta
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
    env._attention_push_xy[env_ids, 0] = delta[:, 0]
    env._attention_push_xy[env_ids, 1] = delta[:, 1]
