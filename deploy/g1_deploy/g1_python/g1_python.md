# G1 部署

Unitree G1 29 自由度策略部署文档，包含基于 MuJoCo 的 Sim2Sim 验证，以及在真实机器人上的 Sim2Real 部署。

除安装依赖步骤外，本文中的 Python 脚本命令默认在 `~/legged_rl_lab/deploy/g1_deploy/g1_python` 下执行。

## Sim2Sim（MuJoCo）

### 1. SDK2 环境配置

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy

# MuJoCo / ONNX / 手柄依赖
pip install onnxruntime scipy pyyaml mujoco pygame

# CycloneDDS C 库。若 deploy/g1_deploy/cyclonedds/install 已存在，可跳过这一段。
python -m pip install cmake
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir -p build install
cd build
python -m cmake .. -DCMAKE_INSTALL_PREFIX=../install
python -m cmake --build . --target install -j"$(nproc)"

# unitree_sdk2_python submodule
cd ~/legged_rl_lab
git submodule update --init --recursive deploy/g1_deploy/g1_python/unitree_sdk2_python

cd deploy/g1_deploy/g1_python/unitree_sdk2_python
export CYCLONEDDS_HOME=$(pwd)/../../cyclonedds/install
pip install -e .

# 防止 pip 把 IsaacLab 依赖升级到不兼容版本
pip install numpy==1.26.0 opencv-python==4.10.0.84 packaging==23.0 wheel==0.45.1
```

检查安装：

```bash
python -c "import cyclonedds, unitree_sdk2py, mujoco, pygame; print('sdk2 sim env ok')"
```

本地仿真使用 DDS domain `1`；真实机器人默认使用 domain `0`。如果你看到 `selected interface "lo" is not multicast-capable: disabling multicast`，这是 loopback 的常见提示，不是错误。

### 2. 以 Walk 为例：SDK2 Sim2Sim 闭环联调全流程

这一节用 `g1_walk.yaml` + `g1_flat_1.onnx` 作为完整例子。SDK2 Sim2Sim 联调需要两个终端：终端 1 跑 MuJoCo 假机器人和 SDK2 bridge，终端 2 跑和真机相同入口风格的 deploy controller。

普通 `sim2sim_walk.py` 是快速验证策略本身；SDK2 闭环联调则会完整走 DDS topic：

```text
sim2real_walk.py 发布 rt/lowcmd
  ↓
sim2sim_sdk2_bridge.py 订阅 LowCmd，写入 MuJoCo actuator
  ↓
MuJoCo step
  ↓
bridge 发布 rt/lowstate / rt/wirelesscontroller / rt/sportmodestate
  ↓
sim2real_walk.py 订阅 LowState，继续推理和发布 LowCmd
```

#### 2.1 可选：先跑普通 Sim2Sim

这一步不走 DDS，适合先确认 ONNX、YAML、手柄映射和策略本身正常。

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad
```

没有手柄时可用键盘：

```bash
python sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input keyboard
```

#### 2.2 终端 1：启动 SDK2 bridge

手柄输入：

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2sim_sdk2_bridge.py \
  --config g1_walk.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

没有手柄时可用键盘输入：

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2sim_sdk2_bridge.py \
  --config g1_walk.yaml \
  --net lo \
  --domain_id 1 \
  --input keyboard \
  --elastic_band \
  --debug_lowcmd
```

启动后保持 MuJoCo viewer 打开。此时如果 bridge 终端显示 `cmd_age=inf`，表示还没有 controller 发布 `rt/lowcmd`，这是正常的。

`--debug_lowcmd` 会打印 LowCmd 转 MuJoCo torque 的范围。bridge 默认会把 LowCmd 计算出的执行器力矩 clamp 到 XML 的 `ctrlrange`。如果 `[LowCmdDebug]` 里 `ctrl_raw` 很大，或者 `clipped` 长时间不为 0，优先检查观测对齐、动作缩放、默认关节姿态和 PD 增益；这通常不是 DDS 频率问题。只有调试原始输出时才使用 `--no_clamp_ctrl`。

#### 2.3 终端 2：启动 Walk deploy controller

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2real_walk.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_walk.yaml \
  --debug_policy
```

两个终端的 `--net` 和 `--domain_id` 必须一致。本地闭环推荐 `--net lo --domain_id 1`；如果 CycloneDDS 不走 `lo`，把两个终端都改成同一个网卡名，例如 `enp108s0`。

`[PolicyDebug]` 会打印策略输入和输出范围。平地站稳附近，`grav` 应接近 `[0, 0, -1]`，`cmd` 应接近 0；如果 `action`、`target_delta_max` 很大，或者 `clipped` 长时间不为 0，优先查策略输出、动作缩放和初始高度；如果 `grav` 符号或轴明显不对，优先查 MuJoCo IMU 到 SDK2 LowState 的坐标对齐。

#### 2.4 操作顺序

1. 先启动终端 1 的 bridge，并保持 MuJoCo viewer 打开。
2. 再启动终端 2 的 `sim2real_walk.py`。
3. 如果使用键盘输入，按键前先把焦点切回终端 1 的 bridge 终端；键盘控制不在 MuJoCo viewer 窗口或终端 2 生效。
4. controller 打印 Waiting for the start signal to move to default pos... 后，按手柄 Start/+。键盘 bridge 对应 Enter 或 1。
5. controller 进入 Moving to default pos.，等待机器人移动到默认姿态。
6. controller 打印 Waiting for the Button A signal to Start Control... 后，按手柄 A。键盘 bridge 对应 2。
7. 策略开始控制后，先保持挂带支撑，观察 [PolicyDebug] 和 [LowCmdDebug] 是否稳定。
8. 稳定后，在 MuJoCo viewer 中逐步放下虚拟挂带。
9. 退出时按手柄 Select/-。键盘 bridge 对应 9 或 Esc。

#### 2.5 手柄 / 键盘控制

| 输入 | 功能 |
| --- | --- |
| **Start/+**，或键盘 **Enter / 1** | 从零力矩进入默认姿态 |
| **A**，或键盘 **2** | 开始策略控制 |
| 左摇杆上下，或键盘 **W/S** | 前进速度命令 |
| 左摇杆左右，或键盘 **A/D** | 横移速度命令 |
| 右摇杆左右，或键盘 **Q/E** | 偏航速度命令 |
| 键盘 **Space / 0** | 清零速度命令 |
| **Select/-**，或键盘 **9 / Esc** | 退出 / 阻尼 |

注意：`select` 是退出键，不是进入默认姿态的键。进入默认姿态要按 **Start/+**。

#### 2.6 虚拟挂带操作

虚拟挂带按键只在 MuJoCo viewer 窗口生效：

| MuJoCo viewer 输入 | 功能 |
| --- | --- |
| **9** | 开关虚拟挂带 |
| **8** | 增加挂带长度，减少支撑，机器人逐步下放 |
| **7** | 减少挂带长度，增加支撑，机器人被抬高 |

建议不要一开始就按 **9** 直接关闭挂带。先按 **8** 一点点降低支撑，确认 controller 已经稳定输出 LowCmd 后，再关闭或继续放低。

#### 2.7 Walk 常见调参点

当前 `g1_walk.yaml` 中 SDK2 bridge 默认使用 `imu_source: "qpos_qvel"`，用于对齐普通 `sim2sim_walk.py` 的观测；如需模拟 MuJoCo IMU sensor，可改成 `"sensor"`。

`policy_ramp_time` 用于按 **A** 后平滑接管，避免 policy 第一帧目标角跳变造成大 torque。如果开始控制瞬间冲击明显，先适当增大这个值。

如果从前进切回停止、或者转弯命令变化时出现抖动，优先调 `command_deadband`、`command_smoothing_tau`、`command_rate_limit`。训练配置中 `lin_vel_x` 是 `[0, 1]`，deploy 也保持同样范围；不要给这个策略发送后退命令。

### 3. 任务脚本速查

本节按策略任务列出纯 Sim2Sim gamepad 命令，以及 SDK2 本地闭环联调的两条命令。真机 Sim2Real 命令统一放在后面的 Sim2Real 部分。

`../exported_policy/` 当前包含以下模型：

| ONNX | 用途 | 脚本 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| `g1_flat_1.onnx` | 平地行走、站立稳定、基础速度控制 | `sim2sim_walk.py`，也是 `sim2sim_mimic.py` 的启动策略 | `obs [1, 96]` | `actions [1, 29]` |
| `g1_walk.onnx` | AMP 行走策略 | `sim2sim_amp.py` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_run.onnx` | AMP 跑步策略 | `sim2sim_amp.py` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_dance.onnx` | 动作跟踪舞蹈策略 | `sim2sim_mimic.py` | `obs [1, 160]`，`time_step [1, 1]` | `actions [1, 29]` 以及参考状态 |
| `g1_jump.onnx` | 动作跟踪跳跃策略 | `sim2sim_mimic.py` | `obs [1, 160]`，`time_step [1, 1]` | `actions [1, 29]` 以及参考状态 |
| `g1_attention.onnx` | Attention 地形 / Parkour 策略 | `sim2sim_attention.py` | `obs [1, 2175]` | `actions [1, 29]` |

当前 `../exported_policy/` 文件夹中没有 `g1_amp.onnx` 或 `policy.onnx`。请使用上表列出的模型名称。

#### 3.1 Walk：平地行走 / 站立稳定

纯 Sim2Sim，gamepad 输入：

```bash
python sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad
```

SDK2 联调，终端 1：

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2sim_sdk2_bridge.py \
  --config g1_walk.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

SDK2 联调，终端 2：

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2real_walk.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_walk.yaml \
  --debug_policy
```

#### 3.2 AMP：行走 / 跑步速度策略

纯 Sim2Sim，gamepad 输入：

```bash
python sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad
```

SDK2 联调，终端 1：

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2sim_sdk2_bridge.py \
  --config g1_amp.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

SDK2 联调，终端 2：

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2real_amp.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_amp.yaml \
  --model g1_walk.onnx \
  --debug_policy
```

跑步策略把两处 `--model g1_walk.onnx` 改成 `--model g1_run.onnx`。

#### 3.3 Mimic：动作跟踪 / 舞蹈 / 跳跃

纯 Sim2Sim，gamepad 输入：

```bash
python sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_dance.onnx \
  --input gamepad
```

SDK2 联调，终端 1：

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2sim_sdk2_bridge.py \
  --config g1_mimic.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

SDK2 联调，终端 2：

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2real_mimic.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_mimic.yaml \
  --model g1_dance.onnx \
  --debug_policy
```

跳跃策略把两处 `--model g1_dance.onnx` 改成 `--model g1_jump.onnx`。`sim2sim_mimic.py` 会先使用 `g1_flat_1.onnx` 站立稳定，机器人稳定后按手柄 **B** 切换到跟踪策略。

#### 3.4 Attention：地形高度图 / Parkour 策略

注意：当前 `../exported_policy/` 列表中如果还没有 `g1_attention.onnx`，需要先从训练结果导出并放到 `deploy/g1_deploy/exported_policy/g1_attention.onnx`，或者在命令里用 `--model /path/to/your_attention.onnx` 指向实际文件。

纯 Sim2Sim，gamepad 输入：

```bash
python sim2sim_attention.py \
  --config g1_attention.yaml \
  --model g1_attention.onnx \
  --input gamepad \
  --gamepad_type gamesir \
  --show_rays
```

SDK2 联调，终端 1：

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2sim_sdk2_bridge.py \
  --config g1_attention.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

SDK2 联调，终端 2：

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2real_attention.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_attention.yaml \
  --model g1_attention.onnx \
  --debug_policy
```

`sim2real_attention.py` 当前使用 `--terrain_source flat`，即用平地高度图占位构造 attention 观测；真正上复杂地形前，需要接入真实高度图 / 深度感知来源，替换这个 flat terrain map。

### 4. SDK2 闭环架构

```text
手柄 / 键盘
  ↓
sim bridge 写入 LowState.wireless_remote，并发布 rt/wirelesscontroller
  ↓
deploy controller 订阅 rt/lowstate，解析手柄和机器人状态
  ↓
deploy controller 推理 / PD，发布 rt/lowcmd
  ↓
sim bridge 订阅 rt/lowcmd
  ↓
LowCmd → MuJoCo actuator torque
  ↓
MuJoCo step
  ↓
sim bridge 发布新的 rt/lowstate
```

bridge 对齐 `unitree_mujoco/simulate_python`：从 MuJoCo `sensordata` 读取关节位置、速度、力矩和 IMU，订阅 `rt/lowcmd` 后按
`tau + kp * (q_des - q_sensor) + kd * (dq_des - dq_sensor)` 写入 MuJoCo actuator。

## Sim2Real

### 安装

```bash
conda activate legged_rl_lab
cd deploy/g1_deploy

# 1. 先安装 CMake。当前环境如果已有 cmake，可跳过。
python -m pip install cmake

# 2. 编译安装 CycloneDDS C 库。unitree_sdk2_python 的 cyclonedds Python 包依赖它。
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir -p build install
cd build
python -m cmake .. -DCMAKE_INSTALL_PREFIX=../install
python -m cmake --build . --target install -j"$(nproc)"

# 3. 安装本仓库内置的 unitree_sdk2_python。
cd ~/legged_rl_lab
git submodule update --init --recursive deploy/g1_deploy/g1_python/unitree_sdk2_python

cd deploy/g1_deploy/g1_python/unitree_sdk2_python
export CYCLONEDDS_HOME=$(pwd)/../../cyclonedds/install
pip install -e .

# 4. 如果 pip 把 numpy / packaging 等依赖升到 IsaacLab 不兼容版本，需要恢复兼容版本。
pip install numpy==1.26.0 opencv-python==4.10.0.84 packaging==23.0 wheel==0.45.1
```

本机当前已按上述流程安装完成，并验证 `cyclonedds`、`unitree_sdk2py`、`mujoco`、`pygame` 可在 `legged_rl_lab` 环境中 import。

### 启动流程

#### 1. 启动机器人

给 G1 上电，并保持在零力矩模式。

#### 2. 进入调试模式

![调试模式](../image.png)

按 **L2 + R2** 进入调试模式。此时机器人应处于阻尼模式。

可以按 **L2 + A** 确认调试模式，然后再次按 **L2 + R2** 返回阻尼模式。

安全提示：在调试模式下，按 **L2 + B** 可立即进入阻尼模式。

#### 3. 连接机器人

使用以太网连接 PC 和机器人。USB 转以太网适配器或电脑自带网口都可以使用。

将 PC 的网络接口设置到 `192.168.123.X` 网段。推荐使用 `192.168.123.99`。

![网络设置](../image-1.png)

检查网络接口地址：

![ifconfig 输出](../image-2.png)

验证连通性：

```bash
ping 192.168.123.161
```

#### 4. 启动程序

假设以太网接口为 `enp108s0`。

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2real_walk.py
```

各策略真机命令速查：

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python

# Walk
python sim2real_walk.py \
  --net enp108s0 \
  --domain_id 0 \
  --config_path config/g1_walk.yaml

# AMP walk；跑步把 model 改成 g1_run.onnx
python sim2real_amp.py \
  --net enp108s0 \
  --domain_id 0 \
  --config_path config/g1_amp.yaml \
  --model g1_walk.onnx

# Mimic dance；跳跃把 model 改成 g1_jump.onnx
python sim2real_mimic.py \
  --net enp108s0 \
  --domain_id 0 \
  --config_path config/g1_mimic.yaml \
  --model g1_dance.onnx

# Attention
python sim2real_attention.py \
  --net enp108s0 \
  --domain_id 0 \
  --config_path config/g1_attention.yaml \
  --model g1_attention.onnx
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
