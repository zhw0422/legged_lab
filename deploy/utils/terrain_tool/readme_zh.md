# Terrain Tool (MuJoCo)

把训练用的 AME 课程拼成 MuJoCo `scene_terrain.xml`，沿 `+x` 轴从机器人 spawn 处依次排开：

```text
上楼梯 -> 一段 deck-level 障碍课程 -> 终点平台
```

默认目标机器人是 **G1**，输出文件：

- `deploy/g1_deploy/assets/scene_terrain.xml`：带 `<include g1_29dof.xml>`。
- `deploy/g1_deploy/assets/scene_terrain_only.xml`：不带机器人，纯地形，需手动重生成，见下文。

> 兼容旧用法：`--robot go2` 会改为生成 `deploy/go2_deploy/assets/scene_terrain.xml`。

---

## 1. 安装依赖

```bash
conda activate legged_rl_lab

# 默认课程只需 opencv（用于 ndarray IO）和 numpy
pip install opencv-python numpy

# 仅 AddPerlinHeighField 需要
pip install noise
```

`legged_rl_lab` 环境已经预装 `mujoco`、`numpy`、`opencv-python`。Perlin 高度场是可选的，不调用 `AddPerlinHeighField` 时不需要安装 `noise`。

---

## 2. 一键生成默认 G1 课程

```bash
python deploy/utils/terrain_tool/terrain_generator.py --robot g1
```

终端会打印：

```text
Saved terrain scene: /home/.../deploy/g1_deploy/assets/scene_terrain.xml
```

随后任何加载 `scene_terrain.xml` 的脚本（例如 `sim2sim_attention.py`）都会用到新地形。

如果你只想看「不带机器人的纯地形」，再跑一次这个小脚本：

```bash
python - <<'PY'
from pathlib import Path

src = Path('deploy/g1_deploy/assets/scene_terrain.xml').read_text()
out = src.replace('<include file="g1_29dof.xml" />', '').replace('<include file="g1_29dof.xml"/>', '')
Path('deploy/g1_deploy/assets/scene_terrain_only.xml').write_text(out)
PY
```

再用 MuJoCo 自带 viewer 查看：

```bash
python -m mujoco.viewer --mjcf=deploy/g1_deploy/assets/scene_terrain_only.xml
```

---

## 3. 课程拓扑（默认参数）

整条路从 `x = 1.5 m` 开始，所有 deck 顶面统一在 `z = 1.6 m`（10 级 `0.16 m` 楼梯爬上去后的高度）：

```text
spawn --(地面)--┬─ ame_stair_00..09        (主楼梯 10 级 x 0.16 m，宽 0.30 m)
                ├─ ame_stair_top           (顶平台 1.2 x 1.6)
                ├─ ame_connector_after_stairs
                ├─ E  concentric_gaps      (实心地块 + 沟壑交替)
                ├─ B  double_column_stakes (双列均匀石桩)
                ├─ D  stone_bridge         (单板长石桥)
                ├─ C  stepping_stones      (左右交替石块)
                ├─ A  alternate_column_stakes (左右交错石桩)
                ├─ F  stairs_up    (再升 6 级 0.16 m -> z = 2.56 m)
                ├─ G  stairs_down  (再降 6 级 0.16 m -> 回到 z = 1.6 m)
                ├─ H  radial_plank_bridge  (中心平台 + 单板)
                └─ ame_final_platform      (1.2 m 长终点平台)
```

每段之间夹一个 `ame_connector_*`（默认 `0.8 m` 长，整宽 deck），用来给 policy 一个安全过渡区。

按训练（AME finetune）课程从易到难的顺序，gap 段在最前。要换顺序，直接改 `AddAMETerrainSequence` 里的调用顺序。

---

## 4. 命令行参数

```bash
python deploy/utils/terrain_tool/terrain_generator.py [选项]
```

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--robot {g1,go2}` | `g1` | 选择输出目录：`deploy/<robot>_deploy/assets/scene_terrain.xml` |
| `--input_scene PATH` | 见下 | 起始模板 XML；不指定时按 `--robot` 自动选择 |
| `--output_scene PATH` | 见下 | 输出 XML 路径；不指定时按 `--robot` 自动选择 |
| `--seed INT` | `7` | `numpy` 随机种子，只影响 `AddRoughGround` / `AddPerlinHeighField` |
| `--side_demos` | `off` | 在主路径之外另起几个旧风格 demo 区（slope box、悬浮楼梯、rough、Perlin），仅供 debug |

`--robot g1` 时模板默认为 `deploy/g1_deploy/assets/scene_29dof.xml`（已包含 `<include g1_29dof.xml/>` 与 floor），输出到 `deploy/g1_deploy/assets/scene_terrain.xml`。

---

## 5. 关键可调参数

所有几何尺寸都在 `terrain_generator.py` 的方法签名里。下表只列**最常改的几个**。

### 5.1 主上楼梯

`AddStairsUpWithPlatform(...)`

调用点：`__main__` 中。

```python
tg.AddStairsUpWithPlatform(
    init_pos=[1.5, 0.0, 0.0],   # 起点 (x, y, z)，z=0 表示从地面起
    yaw=0.0,                    # 走廊整体朝向（0 = +x）
    width=0.30,                 # 每级踏面深度（沿 x）
    height=0.16,                # 每级高度（决定 deck 顶高）
    length=1.6,                 # 踏面横向宽度（沿 y）
    stair_nums=10,              # 级数。deck_height = stair_nums * height
    top_width=1.2,              # 顶平台沿 x 长度，0 表示不要平台
)
```

要把整条课程抬高到例如 `2 m`：把 `height` 改成 `0.20` 即可（`AddAMETerrainSequence` 会自动跟随返回的 `deck_height`）。

### 5.2 AME 段公共参数

`AddAMETerrainSequence(init_pos, yaw, start_x, deck_height, lane_width=1.6, connector_length=0.8, thickness=0.18)`

| 参数 | 含义 |
| --- | --- |
| `lane_width` | deck 整条走廊的横向宽度（沿 y，米） |
| `connector_length` | 每段之间过渡平台的沿 x 长度 |
| `thickness` | deck box 的厚度。box 顶面对齐 `deck_height`，所以 box 中心 `z = deck_height - thickness / 2` |

### 5.3 单段参数（按段速查）

| 段 | 方法 | 关键参数（默认值） |
| --- | --- | --- |
| **E concentric gaps** | `AddAMEConcentricGaps` | `segment_length=4.0`, `ground_width=0.50`, `gap_width=0.50`：地块 + 沟交替 |
| **B double column stakes** | `AddAMEDoubleColumnStakes` | `stake_side=0.20`, `stake_gap=0.30`, `column_gap=0.30`：双列均匀石桩 |
| **D stone bridge** | `AddAMEStoneBridge` | `stone_length=0.75`, `stone_width=0.35`, `stone_distance=0.22`：长石条桥 |
| **C stepping stones** | `AddAMESteppingStones` | `stone_width=0.35`, `stone_gap=0.18`：左右交替石块 |
| **A alternate stakes** | `AddAMEAlternateColumnStakes` | 同 B，但左右交错（每行只剩单边） |
| **F stairs up** | `AddAMEStairsUpSegment` | `stair_nums=6`, `step_width=0.30`, `step_height=0.16` |
| **G stairs down** | `AddAMEStairsDownSegment` | 同 F，反向；必须和 F 的高度增量相等才能闭环 |
| **H radial plank bridge** | `AddAMERadialPlankBridge` | `plank_width=0.19`, `segment_length=4.0`：中心 `0.75 m` 方台 + 长板 |

要修改某段尺寸，直接改对应方法的 default，或在 `AddAMETerrainSequence` 里调用时显式传参。

---

## 6. 自定义示例

### 6.1 只生成一条「楼梯 -> 沟壑 -> 终点」简化课程

复制 `__main__` 段，改成：

```python
tg = TerrainGenerator('g1')
stair = tg.AddStairsUpWithPlatform(stair_nums=8, height=0.18)
cursor = stair['next_x']
cursor = tg.AddAMEConcentricGaps(
    init_pos=[1.5, 0.0, 0.0],
    yaw=0.0,
    start_x=cursor,
    deck_height=stair['deck_height'],
    segment_length=5.0,
    ground_width=0.40,
    gap_width=0.40,
)
tg._add_top_box(
    [1.5, 0.0, 0.0],
    0.0,
    cursor,
    1.5,
    0.0,
    stair['lane_width'],
    stair['deck_height'],
    0.18,
    name='ame_final_platform',
)
tg.Save()
```

### 6.2 让 deck 整体升高 + 楼梯踏面更深

```python
stair = tg.AddStairsUpWithPlatform(
    init_pos=[1.5, 0.0, 0.0],
    width=0.35,
    height=0.20,
    stair_nums=10,
)
```

此时 `deck_height = 2.0 m`，所有后续段自动跟着升高。

### 6.3 把整条 yaw 旋转 90°

```python
stair = tg.AddStairsUpWithPlatform(init_pos=[1.5, 0.0, 0.0], yaw=np.deg2rad(90))
tg.AddAMETerrainSequence(
    init_pos=[1.5, 0.0, 0.0],
    yaw=np.deg2rad(90),
    start_x=stair['next_x'],
    deck_height=stair['deck_height'],
    lane_width=stair['lane_width'],
)
```

走廊会沿 `+y` 排列。`sim2sim_attention.py` 的 ray scan 也用 yaw 对齐，policy 仍能正常感知。

---

## 7. 验证生成结果

最快的两个检查：

```bash
# 1) MuJoCo 能不能解析 + 列出 78 个 geom 左右
python - <<'PY'
import mujoco

m = mujoco.MjModel.from_xml_path('deploy/g1_deploy/assets/scene_terrain.xml')
print('ngeom =', m.ngeom, 'nbody =', m.nbody)
PY

# 2) sim2sim_attention 加载 + ray scan 自检
python deploy/g1_deploy/sim2sim_attention.py \
  --config g1_attention.yaml --model g1_attention.onnx --check
```

可视化（不带机器人）：

```bash
python -m mujoco.viewer --mjcf=deploy/g1_deploy/assets/scene_terrain_only.xml
```

---

## 8. 旧的辅助函数（仍可用，不在默认课程里）

- `AddBox(position, euler, size, name)`：单个 axis-aligned / rotated box。
- `AddGeometry(position, euler, size, geo_type)`：`plane` / `sphere` / `capsule` / `ellipsoid` / `cylinder` / `box`。
- `AddSuspendStairs(...)`：“悬浮楼梯”，需要 `gap < height`。
- `AddRoughGround(...)`：随机抖动 box 网格。
- `AddPerlinHeighField(...)`：输出 PNG heightfield + 关联 `<hfield>`（依赖 `noise`）。
- `AddHeighFieldFromImage(...)`：从外部灰度图生成 hfield。

要叠加这些 demo 区，加 `--side_demos`。它们会落在 `y=2`、`y=6`、`y=4` 等远离主 `+x` 走廊的位置，不会污染 attention 课程。
