# G1 部署

Unitree G1 29 自由度策略部署文档，包含基于 MuJoCo 的 Sim2Sim 验证，以及在真实机器人上的 Sim2Real 部署。

## Sim2Sim（MuJoCo）

### 1. SDK2 环境配置

```bash
conda activate legged_rl_lab
cd /home/wzh/legged_rl_lab/deploy/g1_deploy

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

# unitree_sdk2_python
cd ../../unitree_sdk2_python
export CYCLONEDDS_HOME=$(pwd)/../cyclonedds/install
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
cd /home/wzh/legged_rl_lab
python deploy/g1_deploy/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad
```

没有手柄时可用键盘：

```bash
python deploy/g1_deploy/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input keyboard
```

#### 2.2 终端 1：启动 SDK2 bridge

手柄输入：

```bash
conda activate legged_rl_lab
cd /home/wzh/legged_rl_lab/deploy/g1_deploy
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
cd /home/wzh/legged_rl_lab/deploy/g1_deploy
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
cd /home/wzh/legged_rl_lab/deploy/g1_deploy
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
3. controller 打印 `Waiting for the start signal to move to default pos...` 后，按手柄 **Start/+**。键盘 bridge 对应 **Enter** 或 **1**。
4. controller 进入 `Moving to default pos.`，等待机器人移动到默认姿态。
5. controller 打印 `Waiting for the Button A signal to Start Control...` 后，按手柄 **A**。键盘 bridge 对应 **2**。
6. 策略开始控制后，先保持挂带支撑，观察 `[PolicyDebug]` 和 `[LowCmdDebug]` 是否稳定。
7. 稳定后，在 MuJoCo viewer 中逐步放下虚拟挂带。
8. 退出时按手柄 **Select/-**。键盘 bridge 对应 **9** 或 **Esc**。

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

本节按策略任务列出普通 Sim2Sim、SDK2 bridge 联调、以及对应 Sim2Real controller。SDK2 联调都需要两个终端：终端 1 跑 `sim2sim_sdk2_bridge.py`，终端 2 跑对应的 `sim2real_*.py`。

`deploy/g1_deploy/exported_policy/` 当前包含以下模型：

| ONNX | 用途 | 脚本 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| `g1_flat_1.onnx` | 平地行走、站立稳定、基础速度控制 | `sim2sim_walk.py`，也是 `sim2sim_mimic.py` 的启动策略 | `obs [1, 96]` | `actions [1, 29]` |
| `g1_walk.onnx` | AMP 行走策略 | `sim2sim_amp.py` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_run.onnx` | AMP 跑步策略 | `sim2sim_amp.py` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_dance.onnx` | 动作跟踪舞蹈策略 | `sim2sim_mimic.py` | `obs [1, 160]`，`time_step [1, 1]` | `actions [1, 29]` 以及参考状态 |
| `g1_jump.onnx` | 动作跟踪跳跃策略 | `sim2sim_mimic.py` | `obs [1, 160]`，`time_step [1, 1]` | `actions [1, 29]` 以及参考状态 |

当前 `exported_policy/` 文件夹中没有 `g1_amp.onnx` 或 `policy.onnx`。请使用上表列出的模型名称。

#### 3.1 Walk：平地行走 / 站立稳定

使用 `g1_walk.yaml` 和 `g1_flat_1.onnx`。该策略输入为一个 96 维观测帧：

`base_ang_vel(3) + projected_gravity(3) + command(3) + joint_pos(29) + joint_vel(29) + last_action(29)`。

普通 Sim2Sim：

```bash
python deploy/g1_deploy/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad
```

普通 Sim2Sim，键盘输入：

```bash
python deploy/g1_deploy/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input keyboard
```

SDK2 bridge 联调，终端 1：

```bash
cd /home/wzh/legged_rl_lab/deploy/g1_deploy
python sim2sim_sdk2_bridge.py \
  --config g1_walk.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

SDK2 bridge 联调，终端 2：

```bash
cd /home/wzh/legged_rl_lab/deploy/g1_deploy
python sim2real_walk.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_walk.yaml \
  --debug_policy
```

真实机器人：

```bash
python sim2real_walk.py \
  --net enp108s0 \
  --domain_id 0 \
  --config_path config/g1_walk.yaml
```

普通 Sim2Sim 控制方式：

| 输入 | 功能 |
| --- | --- |
| 左摇杆上下 / **W/S** | `vx` 前进/后退 |
| 左摇杆左右 / **A/D** | `vy` 横移 |
| 右摇杆左右 / **Q/E** | `vyaw` 转向 |
| **RB + A** / **1** | 行走策略 |
| **菜单键 / Start** 或 **X/Esc** | 退出 |

#### 3.2 AMP：行走 / 跑步速度策略

使用 `g1_amp.yaml` 和 `g1_walk.onnx` 或 `g1_run.onnx`。该策略输入为 384 维：

`history_length=4`，每一帧为 96 维，并且在 ONNX 推理前按特征组堆叠这些帧。

检查配置、ONNX 维度以及 MuJoCo 关节/执行器映射：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --check
```

普通 Sim2Sim 行走：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad
```

普通 Sim2Sim 跑步：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_run.onnx \
  --input gamepad
```

普通 Sim2Sim，键盘输入：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input keyboard
```

普通 Sim2Sim，手柄轴调试：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad \
  --debug_gamepad
```

SDK2 bridge 联调，终端 1：

```bash
cd /home/wzh/legged_rl_lab/deploy/g1_deploy
python sim2sim_sdk2_bridge.py \
  --config g1_amp.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

SDK2 bridge 联调，终端 2，AMP 行走：

```bash
cd /home/wzh/legged_rl_lab/deploy/g1_deploy
python sim2real_amp.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_amp.yaml \
  --model g1_walk.onnx \
  --debug_policy
```

SDK2 bridge 联调，终端 2，AMP 跑步：

```bash
python sim2real_amp.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_amp.yaml \
  --model g1_run.onnx \
  --debug_policy
```

真实机器人：

```bash
python sim2real_amp.py \
  --net enp108s0 \
  --domain_id 0 \
  --config_path config/g1_amp.yaml \
  --model g1_walk.onnx
```

#### 3.3 Mimic：动作跟踪 / 舞蹈 / 跳跃

使用 `g1_mimic.yaml` 和 `g1_dance.onnx` 或 `g1_jump.onnx`。跟踪策略 ONNX 内嵌参考动作片段。它接收当前观测和 `time_step`，然后输出动作以及参考关节/身体状态。

普通 Sim2Sim 舞蹈：

```bash
python deploy/g1_deploy/sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_dance.onnx \
  --input gamepad
```

普通 Sim2Sim 跳跃：

```bash
python deploy/g1_deploy/sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_jump.onnx \
  --input gamepad
```

普通 Sim2Sim，键盘输入：

```bash
python deploy/g1_deploy/sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_dance.onnx \
  --input keyboard
```

`sim2sim_mimic.py` 会先使用 `g1_flat_1.onnx` 进行站立稳定。机器人稳定后，按 **B**（手柄）或 **2**（键盘）切换到 `--model` 指定的跟踪策略。

策略切换会在终端中打印提示，例如 `[PolicySwitch] Active policy: N`。每个 Sim2Sim 脚本都在文件底部附近定义了 `policy_registry`，如需注册新的策略槽，修改其中的 YAML 路径和 ONNX 文件名即可。

当前跟踪策略注册表结构：

```python
policy_registry = {
    1: (flat_config,  'g1_flat_1.onnx'),  # 1 / A: stand / stabilize
    2: (mimic_config, args.model),        # 2 / B: main mimic model
    3: (mimic_config, 'g1_jump.onnx'),    # 3 / X
    4: (mimic_config, 'g1_dance.onnx'),   # 4 / Y
}
```

SDK2 bridge 联调，终端 1：

```bash
cd /home/wzh/legged_rl_lab/deploy/g1_deploy
python sim2sim_sdk2_bridge.py \
  --config g1_mimic.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

SDK2 bridge 联调，终端 2，舞蹈：

```bash
cd /home/wzh/legged_rl_lab/deploy/g1_deploy
python sim2real_mimic.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_mimic.yaml \
  --model g1_dance.onnx \
  --debug_policy
```

SDK2 bridge 联调，终端 2，跳跃：

```bash
python sim2real_mimic.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_mimic.yaml \
  --model g1_jump.onnx \
  --debug_policy
```

真实机器人：

```bash
python sim2real_mimic.py \
  --net enp108s0 \
  --domain_id 0 \
  --config_path config/g1_mimic.yaml \
  --model g1_dance.onnx
```

Mimic 的 Sim2Real 脚本会订阅 `rt/sportmodestate` 来近似构造 tracking policy 的 state-estimation 观测。先用 SDK2 bridge 联调确认 `[PolicyDebug]` 和 `[LowCmdDebug]` 稳定，再考虑真机。

控制方式：

| 输入 | 功能 |
| --- | --- |
| **Start** | 进入默认姿态 |
| **A** | 开始对应策略控制 |
| 左摇杆上下 / **W/S** | 前进速度命令 |
| 左摇杆左右 / **A/D** | 横移速度命令 |
| 右摇杆左右 / **Q/E** | 偏航速度命令 |
| **Space** 或 **0** | 清零速度命令 |
| **Select** 或 **Esc** | 退出 / 阻尼 |

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
cd ../..
cd unitree_sdk2_python
export CYCLONEDDS_HOME=$(pwd)/../cyclonedds/install
pip install -e .

# 4. 如果 pip 把 numpy / packaging 等依赖升到 IsaacLab 不兼容版本，需要恢复兼容版本。
pip install numpy==1.26.0 opencv-python==4.10.0.84 packaging==23.0 wheel==0.45.1
```

本机当前已按上述流程安装完成，并验证 `cyclonedds`、`unitree_sdk2py`、`mujoco`、`pygame` 可在 `legged_rl_lab` 环境中 import。

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
