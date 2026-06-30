# G1 C++ Deploy

`g1_cpp` 是 `g1_python` 的 C++ 迁移入口，当前覆盖 MuJoCo sim2sim 主流程：

| Python | C++ |
| --- | --- |
| `sim2sim_walk.py` | `sim2sim_walk` |
| `sim2sim_amp.py` | `sim2sim_amp` |
| `sim2sim_mimic.py` | `sim2sim_mimic` |
| `sim2sim_attention.py` | `sim2sim_attention` |
| `sim2sim_sdk2_bridge.py` | `sim2sim_sdk2_bridge` scaffold |

## unitree_sdk2 结构

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

## 依赖

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

## 编译

```bash
cmake -S deploy/g1_deploy/g1_cpp -B build/g1_cpp \
  -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
cmake --build build/g1_cpp -j
```

## 运行

配置和模型路径规则与 Python 版一致：YAML 默认从 `deploy/g1_deploy/config` 读取，ONNX 默认从 `deploy/g1_deploy/exported_policy` 读取。

`../exported_policy/` 当前包含以下模型：

| ONNX | 用途 | 脚本 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| `g1_flat_1.onnx` | 平地行走、站立稳定、基础速度控制 | `sim2sim_walk`，也是 `sim2sim_mimic` 的启动策略 | `obs [1, 96]` | `actions [1, 29]` |
| `g1_walk.onnx` | AMP 行走策略 | `sim2sim_amp` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_run.onnx` | AMP 跑步策略 | `sim2sim_amp` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_dance.onnx` | 动作跟踪舞蹈策略 | `sim2sim_mimic` | `obs [1, 160]`，`time_step [1, 1]` | `actions [1, 29]` 以及参考状态 |
| `g1_jump.onnx` | 动作跟踪跳跃策略 | `sim2sim_mimic` | `obs [1, 160]`，`time_step [1, 1]` | `actions [1, 29]` 以及参考状态 |
| `g1_attention1.onnx` | Attention 地形 / Parkour 策略 | `sim2sim_attention` | `obs [1, 2175]` | `actions [1, 29]` |

### 任务脚本速查

每个任务列出一条纯 MuJoCo Sim2Sim 命令，以及 SDK2 本地闭环联调的两条命令。SDK2 联调需要两个终端：终端 1 运行 C++ bridge 假机器人，终端 2 暂用现有 Python `sim2real_*` controller。C++ `sim2real_*` controller 迁移完成后，再把终端 2 命令替换成对应 C++ 可执行文件。

#### Walk：平地行走 / 站立稳定

纯 Sim2Sim，gamepad 输入：

```bash
build/g1_cpp/sim2sim_walk \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad
```

SDK2 联调，终端 1：

```bash
build/g1_cpp/sim2sim_sdk2_bridge \
  --config g1_walk.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
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

#### AMP：行走 / 跑步速度策略

纯 Sim2Sim，gamepad 输入：

```bash
build/g1_cpp/sim2sim_amp \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad
```

SDK2 联调，终端 1：

```bash
build/g1_cpp/sim2sim_sdk2_bridge \
  --config g1_amp.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
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

#### Mimic：动作跟踪 / 舞蹈 / 跳跃

纯 Sim2Sim，gamepad 输入：

```bash
build/g1_cpp/sim2sim_mimic \
  --config g1_mimic.yaml \
  --model g1_dance.onnx \
  --input gamepad
```

SDK2 联调，终端 1：

```bash
build/g1_cpp/sim2sim_sdk2_bridge \
  --config g1_mimic.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
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

跳跃策略把两处 `--model g1_dance.onnx` 改成 `--model g1_jump.onnx`。`sim2sim_mimic` 会先使用 `g1_flat_1.onnx` 站立稳定，机器人稳定后按手柄 **B** 切换到跟踪策略。

#### Attention：地形高度图 / Parkour 策略

纯 Sim2Sim，gamepad 输入：

```bash
build/g1_cpp/sim2sim_attention \
  --config g1_attention.yaml \
  --model g1_attention1.onnx \
  --input gamepad
```

SDK2 联调，终端 1：

```bash
build/g1_cpp/sim2sim_sdk2_bridge \
  --config g1_attention.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
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
  --model g1_attention1.onnx \
  --debug_policy
```

无交互检查：

```bash
build/g1_cpp/sim2sim_attention --check --no_render --input const --const_vx 0.3
```

### 手柄 / 键盘控制

纯 Sim2Sim 目标输入是 `--input gamepad`。当前 C++ 代码里的原生 gamepad 映射还在对齐中，如果看到 `[input] Native C++ gamepad mapping is not implemented yet; using keyboard controls.`，则临时使用键盘控制：

| Key | Action |
| --- | --- |
| `W/S` | vx +/- |
| `A/D` | vy +/- |
| `Q/E` | yaw +/- |
| `Space` or `0` | zero command |
| `1/2/3/4` | policy slot marker |
| `X` or `Esc` | exit |

## 实现对应关系

公共 C++ 库在 `src/sim2sim_controller.cpp`：

- 解析 YAML，保持 Python 版的 `assets/`、`exported_policy/` 路径规则。
- 用 MuJoCo C API 加载 XML、映射 joint/actuator 地址。
- 按 `mujoco_to_isaac_map` / `isaac_to_mujoco_map` 做策略顺序转换。
- walk/amp 使用 96 维单帧 proprio obs，再按 Python 版 group-major history 拼接。
- mimic tracking 使用 `obs` 和 `time_step` 双输入，读取 ONNX 返回的 reference joint/body 状态。
- attention 复刻 torso yaw frame 下的地形 ray scan，并拼接 proprio + terrain map。

`sim2sim_sdk2_bridge` 的目标是对齐 Python `sim2sim_sdk2_bridge.py`：订阅 `LowCmd`，写入 MuJoCo actuator；发布 `LowState`、`SportModeState` 和 `WirelessController`，让终端 2 的 controller 像连接真机一样通过 DDS 闭环控制仿真机器人。`unitree_sdk2` 已在 `g1_cpp/unitree_sdk2` 下，可通过 `-DG1_CPP_BUILD_SDK2=ON` 导入 SDK target。
