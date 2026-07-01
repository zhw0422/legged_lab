# G1 C++ 部署

Unitree G1 29 自由度策略 C++ 部署文档，包含基于 MuJoCo 的 Sim2Sim 验证，以及 SDK2 本地闭环联调入口。

除安装依赖步骤外，本文中的 C++ 命令默认在 `~/legged_rl_lab` 仓库根目录下执行。

## Sim2Sim（MuJoCo）

### 1. C++ / SDK2 环境配置

#### 1.1 unitree_sdk2 结构

`unitree_sdk2` 作为 submodule 放在 `deploy/g1_deploy/g1_cpp/unitree_sdk2`。首次 clone 仓库后先初始化：

```bash
cd ~/legged_rl_lab
git submodule update --init --recursive deploy/g1_deploy/g1_cpp/unitree_sdk2
```

SDK 主要目录：

| 路径 | 作用 |
| --- | --- |
| `include/unitree/robot/channel` | DDS channel publisher/subscriber/factory |
| `include/unitree/idl/hg` | G1/H1 humanoid LowCmd、LowState、IMU、MotorState 等 IDL |
| `include/unitree/robot/g1` | G1 高层 client：loco、arm、audio、agv |
| `include/unitree/dds_wrapper/robots/g1` | DDS wrapper 风格的 G1 pub/sub |
| `lib/x86_64/libunitree_sdk2.a` | x86_64 静态库 |
| `lib/aarch64/libunitree_sdk2.a` | aarch64 静态库 |
| `thirdparty` | CycloneDDS / ddscxx 头文件和库 |
| `example/g1` | G1 loco、arm、hand、low-level 示例 |

`unitree_sdk2/CMakeLists.txt` 会按 `CMAKE_SYSTEM_PROCESSOR` 在 `lib/<arch>` 中导入 `libunitree_sdk2.a`，并把 `ddsc`、`ddscxx`、`Threads::Threads` 作为接口依赖。`g1_cpp/CMakeLists.txt` 默认不构建 SDK 示例；如需把 SDK target 一并导入：

```bash
cmake -S deploy/g1_deploy/g1_cpp -B build/g1_cpp \
  -DG1_CPP_BUILD_SDK2=ON
```

#### 1.2 依赖

C++ sim2sim 需要：

```bash
cmake
yaml-cpp
mujoco C/C++ headers and library
onnxruntime C++ API
glfw3                 # 可选；没有时使用 --no_render/headless
```

Ubuntu / Debian 上可以先装系统包：

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  libyaml-cpp-dev \
  libglfw3-dev
```

MuJoCo C/C++ 库从官方 release 下载 Linux 包：

```bash
#  ~/.mujoco
mkdir -p ~/.mujoco
cd ~/.mujoco

wget https://github.com/google-deepmind/mujoco/releases/download/3.3.6/mujoco-3.3.6-linux-x86_64.tar.gz
tar -xzf mujoco-3.3.6-linux-x86_64.tar.gz

# 让 CMake 能找到 mujocoConfig.cmake
export MUJOCO_ROOT=$HOME/.mujoco/mujoco-3.3.6
export CMAKE_PREFIX_PATH=$MUJOCO_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$MUJOCO_ROOT/lib:$LD_LIBRARY_PATH
```

可以在 `g1_cpp` 下建一个软链接：

```bash
cd /home/wzh/legged_rl_lab/deploy/g1_deploy/g1_cpp
ln -s ~/.mujoco/mujoco-3.3.6 mujoco
```

ONNX Runtime C++ API 从官方 release 下载 Linux 包：

```bash
mkdir -p ~/.onnx
cd ~/.onnx

export ORT_VERSION=1.22.0
wget https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz
tar -xzf onnxruntime-linux-x64-${ORT_VERSION}.tgz

export ONNXRUNTIME_ROOT=$HOME/.onnx/onnxruntime-linux-x64-${ORT_VERSION}
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH
```

如果是在 aarch64 / Jetson / ARM 主机上，MuJoCo 和 ONNX Runtime 都要下载对应架构的包；`unitree_sdk2` 本身已经带有 `lib/aarch64/libunitree_sdk2.a`。

#### 1.3 编译

```bash
cmake -S deploy/g1_deploy/g1_cpp -B build/g1_cpp \
  -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
cmake --build build/g1_cpp -j
```

### 2. SDK2 Sim2Sim 闭环联调教程

SDK2 联调需要两个终端：终端 1 运行 C++ bridge 假机器人，终端 2 运行对应 C++ `sim2real_*` controller。两个终端的 `--net` 和 `--domain_id` 必须一致；本地闭环推荐 `--net lo --domain_id 1`。

编译 SDK2 联调入口时需要打开：

```bash
cmake -S deploy/g1_deploy/g1_cpp -B build/g1_cpp \
  -DG1_CPP_BUILD_SDK2=ON \
  -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
cmake --build build/g1_cpp -j
```

#### 2.1 终端 1：启动 SDK2 bridge

手柄输入：

```bash
build/g1_cpp/sim2sim_sdk2_bridge \
  --config g1_walk.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band
```

没有手柄时可用键盘输入：

```bash
build/g1_cpp/sim2sim_sdk2_bridge \
  --config g1_walk.yaml \
  --net lo \
  --domain_id 1 \
  --input keyboard \
  --elastic_band
```

bridge 默认打开 MuJoCo viewer；无显示器或 SSH headless 环境下加 `--no_render`。启动后保持 MuJoCo viewer 打开。

`--input gamepad` 会读取 Linux joystick 设备，默认依次尝试 `/dev/input/js0`、`/dev/input/js1`；需要指定其他设备时可设置 `G1_CPP_JOYSTICK=/dev/input/jsX`。手柄布局默认按 `--joystick_type switch`，Xbox 手柄使用 `--joystick_type xbox`。如果启动时提示没有找到 `/dev/input/js*`，先用 MuJoCo viewer 键盘或 bridge 终端键盘操作。

MuJoCo viewer 相机会持续跟随机器人根位置；鼠标左键拖动旋转视角，鼠标右键上下拖动或滚轮缩放。

#### 2.2 终端 2：启动 deploy controller

以 Walk 为例：

```bash
build/g1_cpp/sim2real_walk \
  --net lo \
  --domain_id 1 \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx
```

AMP / Mimic 时，把可执行文件、YAML 和模型替换为对应任务：

| 任务 | 终端 2 可执行文件 | YAML | 模型 |
| --- | --- | --- | --- |
| Walk | `sim2real_walk` | `g1_walk.yaml` | `g1_flat_1.onnx` |
| AMP walk | `sim2real_amp` | `g1_amp.yaml` | `g1_walk.onnx` |
| AMP run | `sim2real_amp` | `g1_amp.yaml` | `g1_run.onnx` |
| Mimic dance | `sim2real_mimic` | `g1_mimic.yaml` | `g1_dance.onnx` |
| Mimic jump | `sim2real_mimic` | `g1_mimic.yaml` | `g1_jump.onnx` |

#### 2.3 操作顺序和焦点

1. 先启动终端 1 的 bridge，并保持 MuJoCo viewer 打开。
2. 再启动终端 2 的 `sim2real_*` controller。
3. controller 打印 `Waiting for the start signal to move to default pos...` 后，按手柄 Start/+；键盘对应 `Enter` 或 `1`。
4. controller 进入 `Moving to default pos.`，等待机器人移动到默认姿态。
5. controller 打印 `Waiting for the Button A signal to Start Control...` 后，按手柄 A；键盘对应 `2`。
6. 策略开始控制后，先保持挂带支撑。
7. 稳定后，把焦点切到 MuJoCo viewer，按 `8` 逐步放下虚拟挂带。
8. 退出时按手柄 Select/-；bridge 终端键盘对应 `9`，MuJoCo viewer 对应 `Esc` 关闭窗口。

焦点规则：

| 输入方式 | 焦点放哪里 | 说明 |
| --- | --- | --- |
| 手柄 / 遥控器 | 无要求 | bridge 会把手柄状态写入 `LowState.wireless_remote` |
| MuJoCo viewer 键盘 | MuJoCo viewer 窗口 | `1/2/3`、`W/S/A/D/Q/E` 和 `7/8/9` 都在 viewer 内生效 |
| bridge 终端键盘 | 终端 1 | 用于无 viewer 或不想切到 viewer 时发送 start/A/B/速度命令 |
| controller 终端 | 终端 2 | 只看日志，不接收控制按键 |

#### 2.4 手柄 / 键盘 / 遥控器控制

| 操作 | 手柄 / 遥控器 | MuJoCo viewer 键盘 | bridge 终端键盘 |
| --- | --- | --- | --- |
| 连接后进入 default pos | Start | `Enter` or `1` | `Enter` or `1` |
| 开始控制 | A | `2` | `2` |
| 退出 / select | Select | `Esc` 关闭 viewer | `9` |
| 速度 vx | Left stick Y | `W/S` | `W/S` |
| 速度 vy | Left stick X | `A/D` | `A/D` |
| yaw | Right stick X | `Q/E` | `Q/E` |
| 速度归零 | stick 回中 | `Space` or `0` | `Space` or `0` |
| 拉带增加支撑 / 抬高 | - | `7` | - |
| 拉带减小支撑 / 放下 | - | `8` | - |
| 拉带开关 | - | `9` | - |

注意：`select` 是退出键，不是进入默认姿态的键。进入默认姿态要按 Start/+。

#### 2.5 虚拟挂带操作

虚拟挂带按键只在 MuJoCo viewer 窗口生效：

| MuJoCo viewer 输入 | 功能 |
| --- | --- |
| `9` | 开关虚拟挂带 |
| `8` | 增加挂带长度，减少支撑，机器人逐步下放 |
| `7` | 减少挂带长度，增加支撑，机器人被抬高 |

建议不要一开始就按 `9` 直接关闭挂带。先按 `8` 一点点降低支撑，确认 controller 已经稳定输出 LowCmd 后，再关闭或继续放低。

### 3. 任务脚本速查

`../exported_policy/` 当前包含以下模型：

| ONNX | 用途 | 脚本 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| `g1_flat_1.onnx` | 平地行走、站立稳定、基础速度控制 | `sim2sim_walk`，也是 `sim2sim_mimic` 的启动策略 | `obs [1, 96]` | `actions [1, 29]` |
| `g1_walk.onnx` | AMP 行走策略 | `sim2sim_amp` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_run.onnx` | AMP 跑步策略 | `sim2sim_amp` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_dance.onnx` | 动作跟踪舞蹈策略 | `sim2sim_mimic` | `obs [1, 160]`，`time_step [1, 1]` | `actions [1, 29]` 以及参考状态 |
| `g1_jump.onnx` | 动作跟踪跳跃策略 | `sim2sim_mimic` | `obs [1, 160]`，`time_step [1, 1]` | `actions [1, 29]` 以及参考状态 |
| `g1_attention1.onnx` | Attention 地形 / Parkour 策略 | `sim2sim_attention` | `obs [1, 2175]` | `actions [1, 29]` |

每个任务列出一条纯 MuJoCo Sim2Sim 命令，以及 SDK2 本地闭环联调的两条命令。

#### 3.1 Walk：平地行走 / 站立稳定

纯 Sim2Sim，gamepad 输入：

```bash
build/g1_cpp/sim2sim_walk \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad \
  --joystick_type switch
```

SDK2 联调，终端 1：

```bash
build/g1_cpp/sim2sim_sdk2_bridge \
  --config g1_walk.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band
```

SDK2 联调，终端 2：

```bash
build/g1_cpp/sim2real_walk \
  --net lo \
  --domain_id 1 \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx
```

#### 3.2 AMP：行走 / 跑步速度策略

纯 Sim2Sim，gamepad 输入：

```bash
build/g1_cpp/sim2sim_amp \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad \
  --joystick_type switch
```

SDK2 联调，终端 1：

```bash
build/g1_cpp/sim2sim_sdk2_bridge \
  --config g1_amp.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band
```

SDK2 联调，终端 2：

```bash
build/g1_cpp/sim2real_amp \
  --net lo \
  --domain_id 1 \
  --config g1_amp.yaml \
  --model g1_walk.onnx
```

跑步策略把两处 `--model g1_walk.onnx` 改成 `--model g1_run.onnx`。

#### 3.3 Mimic：动作跟踪 / 舞蹈 / 跳跃

纯 Sim2Sim，gamepad 输入：

```bash
build/g1_cpp/sim2sim_mimic \
  --config g1_mimic.yaml \
  --model g1_dance.onnx \
  --input gamepad \
  --joystick_type switch
```

SDK2 联调，终端 1：

```bash
build/g1_cpp/sim2sim_sdk2_bridge \
  --config g1_mimic.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band
```

SDK2 联调，终端 2：

```bash
build/g1_cpp/sim2real_mimic \
  --net lo \
  --domain_id 1 \
  --config g1_mimic.yaml \
  --model g1_dance.onnx
```

跳跃策略把两处 `--model g1_dance.onnx` 改成 `--model g1_jump.onnx`。`sim2sim_mimic` 会先使用 `g1_flat_1.onnx` 站立稳定，机器人稳定后按手柄 **B** 切换到跟踪策略。

#### 3.4 Attention：地形高度图 / Parkour 策略

纯 Sim2Sim，gamepad 输入：

```bash
build/g1_cpp/sim2sim_attention \
  --config g1_attention.yaml \
  --model g1_attention1.onnx \
  --input gamepad \
  --joystick_type switch
```

Attention 策略依赖地形高度图观测，当前 C++ SDK2 部署入口暂不提供真实机器人 height scan；先使用纯 Sim2Sim 验证。

无交互检查：

```bash
build/g1_cpp/sim2sim_attention --check --no_render --input const --const_vx 0.3
```

### 4. 普通 Sim2Sim 手柄 / 键盘控制

纯 Sim2Sim 的 `--input gamepad` 会读取 Linux joystick 设备，默认依次尝试 `/dev/input/js0`、`/dev/input/js1`；需要指定其他设备时可设置 `G1_CPP_JOYSTICK=/dev/input/jsX`。手柄布局默认按 `--joystick_type switch`，Xbox 手柄使用 `--joystick_type xbox`。没有手柄时使用 `--input keyboard`。

普通 Sim2Sim 的 MuJoCo viewer 同样会跟随机器人根位置；鼠标左键拖动旋转视角，鼠标右键上下拖动或滚轮缩放。

| Key | Action |
| --- | --- |
| `W/S` | vx +/- |
| `A/D` | vy +/- |
| `Q/E` | yaw +/- |
| `Space` or `0` | zero command |
| `1/2/3/4` | policy slot marker |
| `X` or `Esc` | exit |
