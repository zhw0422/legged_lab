# G1 Deploy

Unitree G1 (29 DOF) 强化学习策略部署，支持 Sim2Sim (MuJoCo) 和 Sim2Real (真实机器人)。

---

## Sim2Sim (MuJoCo)

在 MuJoCo 中验证策略，支持 GameSir 手柄 (USB) 或键盘控制。

### 安装依赖

```bash
conda activate env_isaaclab1
pip install onnxruntime numpy scipy pyyaml mujoco pygame
```

### 启动 — Walk / 速度控制策略

手柄控制：

```bash
python deploy/g1_deploy/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad
```

键盘控制：

```bash
python deploy/g1_deploy/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input keyboard
```

### 启动 — Motion Tracking / 动作模仿策略

```bash
python deploy/g1_deploy/sim2sim_mimic.py --config g1_mimic.yaml
```

脚本启动时默认加载 **`g1_flat_1.onnx`（flat walk 站立稳定策略）**，待机器人站稳后按 **RB + B** 切换至 tracking 策略开始播放动作。

### 启动 — AMP 策略

先检查 AMP 配置、ONNX 输入输出维度、MuJoCo 关节/执行器映射：

```bash
python deploy/g1_deploy/sim2sim_amp.py --check
```

启动 AMP Sim2Sim：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_amp.onnx \
  --input gamepad
```

使用键盘控制：

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_amp.onnx \
  --input keyboard
```

`g1_amp.onnx` 当前输入为 **510 维**：`history_length=5`，每帧 `base_ang_vel(3) + projected_gravity(3) + command(3) + joint_pos(29) + joint_vel(29) + last_action(29) + gait_phase(6) = 102`。输出为 **29 维 action**，对应 G1 29DOF。

### 输入控制操作

#### Walk 策略（`sim2sim_walk.py`）

手柄模式：

| 操作           | 功能                       |
| -------------- | -------------------------- |
| 左摇杆 上/下   | vx 前进/后退               |
| 左摇杆 左/右   | vy 横移                    |
| 右摇杆 左/右   | vyaw 转向                  |
| **RB + A**     | Walk 策略（主策略）        |
| **RB + B/X/Y** | Policy 1/2/3（占位符）     |
| **Start**      | 退出                       |

键盘模式：

| 操作               | 功能                       |
| ------------------ | -------------------------- |
| **W/S** 或 上/下键 | vx 增加/减小               |
| **A/D**            | vy 增加/减小               |
| **Q/E** 或 左/右键 | vyaw 增加/减小             |
| **Space** 或 **0** | 速度命令清零               |
| **1/2/3/4**        | 切换策略                   |
| **X** 或 **Esc**   | 退出                       |

#### Tracking 策略（`sim2sim_mimic.py`）

| 操作         | 功能                                         |
| ------------ | -------------------------------------------- |
| **RB + A**   | 切换至 Flat walk 策略（站立稳定）            |
| **RB + B**   | 切换至主 Mimic / Tracking 策略（动作播放）   |
| **RB + X**   | Policy 3（占位符）                           |
| **RB + Y**   | Policy 4（占位符）                           |
| **Start**    | 退出                                         |

> 切换 policy 时终端会换行并打印 `[PolicySwitch] Active policy: N (tracking/flat)`。

#### AMP 策略（`sim2sim_amp.py`）

手柄模式：

| 操作           | 功能                       |
| -------------- | -------------------------- |
| 左摇杆 上/下   | vx 前进/后退               |
| 左摇杆 左/右   | vy 横移                    |
| 右摇杆 左/右   | vyaw 转向                  |
| **RB + A**     | AMP 策略（主策略）         |
| **RB + B/X/Y** | Policy 1/2/3（占位符）     |
| **Start**      | 退出                       |

键盘模式：

| 操作               | 功能                       |
| ------------------ | -------------------------- |
| **W/S** 或 上/下键 | vx 增加/减小               |
| **A/D**            | vy 增加/减小               |
| **Q/E** 或 左/右键 | vyaw 增加/减小             |
| **Space** 或 **0** | 速度命令清零               |
| **1/2/3/4**        | 切换策略                   |
| **X** 或 **Esc**   | 退出                       |

### 切换 Policy

`policy_registry` 在脚本底部定义，修改对应的 YAML 配置路径和 ONNX 文件名即可注册新的策略：

```python
policy_registry = {
    1: (flat_config,    'g1_flat_1.onnx'),   # RB+A
    2: (mimic_config,   'policy.onnx'),      # RB+B
    3: ('config/g1_mimic_2.yaml', 'policy_2.onnx'),  # RB+X
    4: ('config/g1_mimic_3.yaml', 'policy_3.onnx'),  # RB+Y
}
```

---

## Sim2Real


### Installation

```bash
conda activate legged_rl_lab
cd deploy/g1_deploy

# 1. Install cyclonedds first (required by unitree_sdk2_python)
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir -p build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install -j"$(nproc)"

# 2. 修复 env_isaaclab1 缺失 Python.h
conda install -n env_isaaclab1 --force-reinstall -y python=3.11.14


# 3. Install unitree_sdk2_python
cd ../..
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
export CYCLONEDDS_HOME=$(pwd)/../cyclonedds/install
pip install -e .
```


### Startup Process

#### 1. Start the robot
开机，g1进入`零力矩模式`

#### 2. Enter the debugging mode

![alt text](image.png)

按下`L2+R2` G1进入`调试模式`，此时为`阻尼模式`，

然后可以按下`L2 + A`确认是否进入`调试模式`，最后按下`L2 + R2`回到`阻尼模式`

> 注意：在调试模式遇到紧急情况，按下`L2 + B`即可进入阻尼模式

#### 3. Connect the robot
用网线连接 PC 与机器人（USB Ethernet 或直连网口）

将 PC 网卡 IP 设置为 `192.168.123.X` 网段，推荐 `192.168.123.99`

![alt text](image-1.png)

用`ifconfig`查看网口地址

![alt text](image-2.png)

验证连通性：

```bash
ping 192.168.123.161 
```

#### 4. Start the Program

假设网线连接的网卡名称为 `enp108s0`

```bash
python sim2real_walk.py
```



##### 4.1 零力矩状态

启动后，机器人关节将处于零力矩状态。你可以用手摇动机器人关节来感受和确认。

##### 4.2 默认位置状态

在零力矩状态下，按下遥控器的 Start 按钮，机器人将移动到默认关节位置状态。

机器人移动到默认关节位置后，可以缓慢降低吊装机制，让机器人的脚接触地面。

##### 4.3 运动控制模式

准备工作完成后，按下遥控器上的 A 按钮，机器人将原地踏步。机器人稳定后，可以逐渐降低吊绳，给机器人一定的活动空间。

此时，你可以使用遥控器上的摇杆来控制机器人的运动：
- **左摇杆前后**：控制机器人 X 方向的运动速度
- **左摇杆左右**：控制机器人 Y 方向的运动速度
- **右摇杆左右**：控制机器人偏航角速度

##### 4.4 退出控制

在运动控制模式下，按下遥控器上的 Select 按钮，机器人将进入阻尼模式并下落，程序将退出。或者在终端中使用 `Ctrl+C` 关闭程序。
