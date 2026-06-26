# G1 部署

Unitree G1 29 自由度策略部署文档，包含基于 MuJoCo 的 Sim2Sim 验证，以及在真实机器人上的 Sim2Real 部署。

---

## Sim2Sim（MuJoCo）

使用 MuJoCo 验证导出的策略，可通过 GameSir USB 手柄或键盘输入控制。

### 依赖安装

```bash
conda activate legged_rl_lab
pip install onnxruntime numpy scipy pyyaml mujoco pygame
```

### 手柄映射检测

首次使用手柄前，先运行检测脚本确认按键和摇杆索引：

```bash
python deploy/utils/test_joystick.py
```

按下手柄按键或推动摇杆时，终端会实时打印按钮索引、轴索引和 Hat 值。当前实测 GameSir 映射如下：

| 控件 | 默认索引 |
| --- | --- |
| A / B / X / Y | `button 0 / 1 / 3 / 4` |
| LB / RB | `button 6 / 7` |
| LT | `axis 5`，同时也会触发 `button 8` |
| RT | `axis 4`，同时也会触发 `button 9` |
| 视图键 / 菜单键 | `button 10 / 11` |
| Home | `button 12` |
| 左摇杆 X / Y | `axis 0 / axis 1` |
| 右摇杆 X / Y | `axis 2 / axis 3` |
| 十字键 | `Hat 0`，返回二维数组 `(x, y)` |

十字键的 `Hat 0` 值中，第一个数表示左右，左为 `-1`、右为 `1`，即 `(-1, 0)` / `(1, 0)`；第二个数表示上下，上为 `1`、下为 `-1`，即 `(0, 1)` / `(0, -1)`。

如果检测到的索引和上表不一致，需要按实际输出调整 YAML 或脚本里的 `gamepad_btn_*` / `axis_*` 配置。

### 当前 ONNX 布局

`deploy/g1_deploy/exported_policy/` 当前包含以下模型：

| ONNX | 用途 | 脚本 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| `g1_flat_1.onnx` | 平地行走、站立稳定、基础速度控制 | `sim2sim_walk.py`，也是 `sim2sim_mimic.py` 的启动策略 | `obs [1, 96]` | `actions [1, 29]` |
| `g1_walk.onnx` | AMP 行走策略 | `sim2sim_amp.py` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_run.onnx` | AMP 跑步策略 | `sim2sim_amp.py` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_dance.onnx` | 动作跟踪舞蹈策略 | `sim2sim_mimic.py` | `obs [1, 160]`，`time_step [1, 1]` | `actions [1, 29]` 以及参考状态 |
| `g1_jump.onnx` | 动作跟踪跳跃策略 | `sim2sim_mimic.py` | `obs [1, 160]`，`time_step [1, 1]` | `actions [1, 29]` 以及参考状态 |

当前 `exported_policy/` 文件夹中没有 `g1_amp.onnx` 或 `policy.onnx`。请使用上表列出的模型名称。

### 1. 行走 / 基础速度控制

使用 `g1_walk.yaml` 和 `g1_flat_1.onnx`。该策略输入为一个 96 维观测帧：

`base_ang_vel(3) + projected_gravity(3) + command(3) + joint_pos(29) + joint_vel(29) + last_action(29)`。

手柄：

```bash
python deploy/g1_deploy/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad
```

键盘：

```bash
python deploy/g1_deploy/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input keyboard
```

控制方式：

手柄：

| 输入 | 功能 |
| --- | --- |
| 左摇杆上下 | `vx` 前进/后退 |
| 左摇杆左右 | `vy` 横移 |
| 右摇杆左右 | `vyaw` 转向 |
| **RB + A** | 行走策略 |
| **RB + B/X/Y** | 策略槽 1/2/3 占位 |
| **菜单键 / Start** | 退出 |

键盘：

| 输入 | 功能 |
| --- | --- |
| **W/S** 或上下方向键 | 增加/减小 `vx` |
| **A/D** | 增加/减小 `vy` |
| **Q/E** 或左右方向键 | 增加/减小 `vyaw` |
| **Space** 或 **0** | 速度指令归零 |
| **1/2/3/4** | 切换策略槽 |
| **X** 或 **Esc** | 退出 |

### 2. AMP 行走 / 跑步速度策略

使用 `g1_amp.yaml` 和 `g1_walk.onnx` 或 `g1_run.onnx`。该策略输入为 384 维：

`history_length=4`，每一帧为 96 维，并且在 ONNX 推理前按特征组堆叠这些帧。

检查配置、ONNX 维度以及 MuJoCo 关节/执行器映射：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --check
```

运行 AMP 行走策略：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad
```

运行 AMP 跑步策略：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_run.onnx \
  --input gamepad
```

键盘：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input keyboard
```

手柄轴调试：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad \
  --debug_gamepad
```

控制方式：

手柄：

| 输入 | 功能 |
| --- | --- |
| 左摇杆向上 | `vx` 前进。当前 `g1_amp.yaml` 中禁用了后退指令。 |
| 左摇杆左右 | `vy` 横移 |
| 右摇杆左右 | `vyaw` 转向 |
| **RB + A** | AMP 策略 |
| **RB + B/X/Y** | 策略槽 1/2/3 占位 |
| **菜单键 / Start** | 退出 |

键盘：

| 输入 | 功能 |
| --- | --- |
| **W/S** 或上下方向键 | 增加/减小 `vx` |
| **A/D** | 增加/减小 `vy` |
| **Q/E** 或左右方向键 | 增加/减小 `vyaw` |
| **Space** 或 **0** | 速度指令归零 |
| **1/2/3/4** | 切换策略槽 |
| **X** 或 **Esc** | 退出 |

### 3. 动作跟踪 / 模仿

使用 `g1_mimic.yaml` 和 `g1_dance.onnx` 或 `g1_jump.onnx`。跟踪策略 ONNX 内嵌了参考动作片段。它接收当前观测和 `time_step`，然后输出动作以及参考关节/身体状态。

舞蹈：

```bash
python deploy/g1_deploy/sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_dance.onnx
```

跳跃：

```bash
python deploy/g1_deploy/sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_jump.onnx
```

键盘：

```bash
python deploy/g1_deploy/sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_dance.onnx \
  --input keyboard
```

`sim2sim_mimic.py` 会先使用 `g1_flat_1.onnx` 进行站立稳定。机器人稳定后，按 **B**（手柄）或 **2**（键盘）切换到 `--model` 指定的跟踪策略。

控制方式：

手柄：

| 输入 | 功能 |
| --- | --- |
| **A** | 平地行走稳定策略 |
| **B** | `--model` 指定的主模仿/跟踪策略 |
| **X** | `g1_jump.onnx` |
| **Y** | `g1_dance.onnx` |
| **视图键 / Select** | 退出 |

键盘：

| 输入 | 功能 |
| --- | --- |
| **1** | 平地行走稳定策略 |
| **2** | `--model` 指定的主模仿/跟踪策略 |
| **3** | `g1_jump.onnx` |
| **4** | `g1_dance.onnx` |
| **X** 或 **Esc** | 退出 |

策略切换会在终端中打印提示，例如 `[PolicySwitch] Active policy: N`。

### 策略切换

每个 Sim2Sim 脚本都在文件底部附近定义了 `policy_registry`。如需注册新的策略槽，修改其中的 YAML 路径和 ONNX 文件名即可。

当前跟踪策略注册表结构：

```python
policy_registry = {
    1: (flat_config,  'g1_flat_1.onnx'),  # 1 / A: stand / stabilize
    2: (mimic_config, args.model),        # 2 / B: main mimic model
    3: (mimic_config, 'g1_jump.onnx'),    # 3 / X
    4: (mimic_config, 'g1_dance.onnx'),   # 4 / Y
}
```

---

## Sim2Real

### 安装

```bash
conda activate legged_rl_lab
cd deploy/g1_deploy

# 1. 先安装 Cyclone DDS。unitree_sdk2_python 依赖它。
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir -p build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install -j"$(nproc)"

# 2. 如果 env_isaaclab1 中缺少 Python.h，则重新安装 Python 头文件。
conda install -n env_isaaclab1 --force-reinstall -y python=3.11.14

# 3. 安装 unitree_sdk2_python。
cd ../..
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
export CYCLONEDDS_HOME=$(pwd)/../cyclonedds/install
pip install -e .
```

### 启动流程

#### 1. 启动机器人

给 G1 上电，并保持在零力矩模式。

#### 2. 进入调试模式

![调试模式](image.png)

按 **L2 + R2** 进入调试模式。此时机器人应处于阻尼模式。

可以按 **L2 + A** 确认调试模式，然后再次按 **L2 + R2** 返回阻尼模式。

安全提示：在调试模式下，按 **L2 + B** 可立即进入阻尼模式。

#### 3. 连接机器人

使用以太网连接 PC 和机器人。USB 转以太网适配器或电脑自带网口都可以使用。

将 PC 的网络接口设置到 `192.168.123.X` 网段。推荐使用 `192.168.123.99`。

![网络设置](image-1.png)

检查网络接口地址：

![ifconfig 输出](image-2.png)

验证连通性：

```bash
ping 192.168.123.161
```

#### 4. 启动程序

假设以太网接口为 `enp108s0`。

```bash
python sim2real_walk.py
```

##### 4.1 零力矩状态

程序启动后，机器人关节处于零力矩模式。可以用手轻轻移动关节来确认该状态。

##### 4.2 默认姿态状态

在零力矩模式下，按遥控器上的 **Start**。机器人会移动到默认关节姿态。

机器人到达默认姿态后，缓慢降低保护架，直到双脚接触地面。

##### 4.3 运动控制状态

设置完成后，按遥控器上的 **A**。机器人会开始原地踏步。稳定后，逐步降低保护架，并允许机器人进行有限自由运动。

遥控器指令：

| 输入 | 功能 |
| --- | --- |
| 左摇杆前后 | X 速度 |
| 左摇杆左右 | Y 速度 |
| 右摇杆左右 | 偏航速度 |

##### 4.4 退出控制

在运动控制模式下，按遥控器上的 **Select**。机器人会进入阻尼模式，安全下落，并退出程序。也可以在终端中使用 `Ctrl+C` 停止程序。
