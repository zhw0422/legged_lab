"""Functions to generate height fields for different terrains."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import ame_hf_terrains_cfg

from random import randint

@height_field_to_mesh
def stones_bridge_terrain(difficulty: float, cfg: ame_hf_terrains_cfg.HfStonesBridgeTerrainCfg) -> np.array:
    """Generate a terrain with stones bridge pattern.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_length = cfg.stone_length_range[1] - difficulty * (cfg.stone_length_range[1] - cfg.stone_length_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
            cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )
    stone_lateral_distance = cfg.stone_lateral_distance_range[0] + difficulty * (
        cfg.stone_lateral_distance_range[1] - cfg.stone_lateral_distance_range[0]
    )

    # switch parameters to discrete units
    # --terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # --stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_lateral_distance = int(stone_lateral_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_length = int(stone_length / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # --holes
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain with a flat platform at one side
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)

    # add the stones
    start_x = stone_distance
    while start_x < width_pixels:
        # ensure that stones stops along x-axis
        stop_x = min(width_pixels, start_x + stone_width)
        # randomly sample x-position
        start_y = (length_pixels - stone_length) // 2 + np.random.choice([-stone_lateral_distance, stone_lateral_distance])
        stop_y = start_y + stone_length
        hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
        # update y-position
        start_x = stop_x + stone_distance
    start_y = stone_distance
    while start_y < length_pixels:
        # ensure that stones stops along y-axis
        stop_y = min(length_pixels, start_y + stone_width)
        # randomly sample x-position
        start_x = (width_pixels - stone_length) // 2 + np.random.choice([-stone_lateral_distance, stone_lateral_distance])
        stop_x = start_x + stone_length
        hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
        # update y-position
        start_y = stop_y + stone_distance

    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def double_column_stakes_terrain(
    difficulty: float, cfg: ame_hf_terrains_cfg.HfDoubleColumnStakesTerrainCfg
) -> np.ndarray:
    """Generate a double-column stake heightfield extending along x/y directions."""

    # Interpolate parameters by difficulty
    stake_side = cfg.stake_side_range[1] - difficulty * (
        cfg.stake_side_range[1] - cfg.stake_side_range[0]
    )
    stake_gap = cfg.stake_gap_range[0] + difficulty * (
        cfg.stake_gap_range[1] - cfg.stake_gap_range[0]
    )
    column_gap = cfg.column_gap_range[0] + difficulty * (
        cfg.column_gap_range[1] - cfg.column_gap_range[0]
    )

    # Discretized grid parameters
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    stake_side_px = max(1, int(stake_side / cfg.horizontal_scale))
    stake_gap_px = max(0, int(stake_gap / cfg.horizontal_scale))
    column_gap_px = max(0, int(column_gap / cfg.horizontal_scale))
    column_jitter_px = max(0, int(cfg.column_jitter / cfg.horizontal_scale))

    stake_height_max_px = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
    holes_depth_px = int(cfg.holes_depth / cfg.vertical_scale)

    platform_width_px = max(1, int(cfg.platform_width / cfg.horizontal_scale))

    hf_raw = np.full((width_pixels, length_pixels), holes_depth_px, dtype=float)
    half_lower = stake_side_px // 2
    half_upper = stake_side_px - half_lower
    center_offset_px = stake_side_px + column_gap_px

    center_x = width_pixels // 2
    center_y = length_pixels // 2

    rng = np.random.default_rng()
    stake_height_values = (
        np.arange(-stake_height_max_px, stake_height_max_px + 1)
        if stake_height_max_px > 0
        else np.array([0], dtype=int)
    )

    def paint_square(cx: int, cy: int, value: int) -> None:
        if cx < 0 or cx >= width_pixels or cy < 0 or cy >= length_pixels:
            return
        x1 = max(0, cx - half_lower)
        x2 = min(width_pixels, cx + half_upper)
        y1 = max(0, cy - half_lower)
        y2 = min(length_pixels, cy + half_upper)
        hf_raw[x1:x2, y1:y2] = value

    def place_column_pair(primary_pos: int, along_x: bool) -> None:
        if along_x:
            axis_limit_low = half_lower
            axis_limit_high = length_pixels - half_upper
            base_offset = max(center_offset_px // 2, half_lower)
            for sign in (-1, 1):
                jitter = (
                    rng.integers(-column_jitter_px, column_jitter_px + 1)
                    if column_jitter_px > 0
                    else 0
                )
                cy = int(np.clip(center_y + sign * base_offset + jitter, axis_limit_low, axis_limit_high))
                height_value = int(rng.choice(stake_height_values))
                paint_square(primary_pos, cy, height_value)
        else:
            axis_limit_low = half_lower
            axis_limit_high = width_pixels - half_upper
            base_offset = max(center_offset_px // 2, half_lower)
            for sign in (-1, 1):
                jitter = (
                    rng.integers(-column_jitter_px, column_jitter_px + 1)
                    if column_jitter_px > 0
                    else 0
                )
                cx = int(np.clip(center_x + sign * base_offset + jitter, axis_limit_low, axis_limit_high))
                height_value = int(rng.choice(stake_height_values))
                paint_square(cx, primary_pos, height_value)

    def extend_from_center(along_x: bool, direction: int) -> None:
        if along_x:
            start = center_x + direction * (half_upper + stake_gap_px + stake_side_px)
            step = (stake_gap_px + stake_side_px) * direction
            while 0 <= start < width_pixels:
                if not (half_lower <= start <= width_pixels - half_upper):
                    break
                place_column_pair(int(start), along_x=True)
                start += step
        else:
            start = center_y + direction * (half_upper + stake_gap_px + stake_side_px)
            step = (stake_gap_px + stake_side_px) * direction
            while 0 <= start < length_pixels:
                if not (half_lower <= start <= length_pixels - half_upper):
                    break
                place_column_pair(int(start), along_x=False)
                start += step

    def extend_from_edge(along_x: bool) -> None:
        start = 0
        step = stake_gap_px + stake_side_px
        while 0 <= start < width_pixels:
            place_column_pair(int(start), along_x)
            start += step


    # Extend along +x/-x
    # extend_from_center(along_x=True, direction=1)
    # extend_from_center(along_x=True, direction=-1)
    extend_from_edge(along_x=True)

    # Extend along +y/-y
    # extend_from_center(along_x=False, direction=1)
    # extend_from_center(along_x=False, direction=-1)
    extend_from_edge(along_x=False)

    # add the platform in the center
    x1 = (width_pixels - platform_width_px) // 2
    x2 = (width_pixels + platform_width_px) // 2
    y1 = (length_pixels - platform_width_px) // 2
    y2 = (length_pixels + platform_width_px) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def concentric_gap_terrain(difficulty: float, cfg: ame_hf_terrains_cfg.HfConcentricGapTerrainCfg) -> np.ndarray:
    """
    Generate concentric gap terrain with a center platform.
    Gap width is difficulty-dependent and gap depth is fixed.
    """
    gap_depth = int(abs(cfg.gap_depth) / cfg.vertical_scale)
    # Gap width varies with difficulty
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    gap_width = int(gap_width / cfg.horizontal_scale)
    # Ground width varies with difficulty (narrower for harder terrains)
    ground_width = cfg.ground_width_range[0] + (1.0 - difficulty) * (cfg.ground_width_range[1] - cfg.ground_width_range[0])
    ground_width = int(ground_width / cfg.horizontal_scale)
    # Ground height
    ground_height_max = int(cfg.ground_height_max / cfg.vertical_scale)
    # Terrain dimensions
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # Platform width
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    is_gap = True
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        if is_gap:
            # Fill gap ring
            hf_raw[start_x:stop_x, start_y:stop_y] = -gap_depth
            start_x += gap_width
            stop_x -= gap_width
            start_y += gap_width
            stop_y -= gap_width
        else:
            # Fill ground ring
            hf_raw[start_x:stop_x, start_y:stop_y] = randint(-ground_height_max, ground_height_max)
            start_x += ground_width
            stop_x -= ground_width
            start_y += ground_width
            stop_y -= ground_width
        is_gap = not is_gap
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def alternate_column_stakes_terrain(
    difficulty: float, cfg: ame_hf_terrains_cfg.HfDoubleColumnStakesTerrainCfg
) -> np.ndarray:
    """Generate alternating double-column stake terrain along x/y directions."""

    # Interpolate parameters by difficulty
    stake_side = cfg.stake_side_range[1] - difficulty * (
        cfg.stake_side_range[1] - cfg.stake_side_range[0]
    )
    stake_gap = cfg.stake_gap_range[0] + difficulty * (
        cfg.stake_gap_range[1] - cfg.stake_gap_range[0]
    )
    column_gap = cfg.column_gap_range[1] - difficulty * (
        cfg.column_gap_range[1] - cfg.column_gap_range[0]
    )

    # Discretized grid parameters
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    stake_side_px = max(1, int(stake_side / cfg.horizontal_scale))
    stake_gap_px = max(0, int(stake_gap / cfg.horizontal_scale))
    column_gap_px = max(0, int(column_gap / cfg.horizontal_scale))
    column_jitter_px = max(0, int(cfg.column_jitter / cfg.horizontal_scale))

    stake_height_max_px = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
    holes_depth_px = int(cfg.holes_depth / cfg.vertical_scale)

    platform_width_px = max(1, int(cfg.platform_width / cfg.horizontal_scale))

    hf_raw = np.full((width_pixels, length_pixels), holes_depth_px, dtype=float)
    half_lower = stake_side_px // 2
    half_upper = stake_side_px - half_lower

    # Build a deterministic RNG for this sub-terrain when cfg.seed is provided.
    # We mix in quantized difficulty so each tile can still look different while
    # remaining reproducible across runs.
    if getattr(cfg, "seed", None) is not None:
        difficulty_key = int(round(float(difficulty) * 1_000_000.0))
        local_seed = (int(cfg.seed) * 1_000_003 + difficulty_key) % (2**32)
        rng = np.random.default_rng(local_seed)
    else:
        rng = np.random.default_rng()
    stake_height_values = (
        np.arange(-stake_height_max_px, stake_height_max_px + 1)
        if stake_height_max_px > 0
        else np.array([0], dtype=int)
    )

    def paint_square(cx: int, cy: int, value: int) -> None:
        if cx < 0 or cx >= width_pixels or cy < 0 or cy >= length_pixels:
            return
        x1 = max(0, cx - half_lower)
        x2 = min(width_pixels, cx + half_upper)
        y1 = max(0, cy - half_lower)
        y2 = min(length_pixels, cy + half_upper)
        hf_raw[x1:x2, y1:y2] = value

    def place_alternate_columns(start_pos: int, along_x: bool) -> None:
        offset = column_gap_px // 2  # Alternating offset
        step = stake_gap_px + stake_side_px
        while start_pos < (width_pixels if along_x else length_pixels):
            jitter = (
                rng.integers(-column_jitter_px, column_jitter_px + 1)
                if column_jitter_px > 0
                else 0
            )
            height_value = int(rng.choice(stake_height_values))

            if along_x:
                cx = start_pos
                cy = (length_pixels // 2) + offset + jitter
                paint_square(cx, cy, height_value)
            else:
                cy = start_pos
                cx = (width_pixels // 2) + offset + jitter
                paint_square(cx, cy, height_value)

            # Flip offset for alternating pattern
            offset = -offset
            start_pos += step

    # Place alternating columns along x and y
    place_alternate_columns(0, along_x=True)
    place_alternate_columns(0, along_x=False)

    # add the platform in the center
    x1 = (width_pixels - platform_width_px) // 2
    x2 = (width_pixels + platform_width_px) // 2
    y1 = (length_pixels - platform_width_px) // 2
    y2 = (length_pixels + platform_width_px) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)
