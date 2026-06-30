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

```bash
build/g1_cpp/sim2sim_walk --config g1_walk.yaml --model g1_flat_1.onnx --input keyboard
build/g1_cpp/sim2sim_amp --config g1_amp.yaml --model g1_walk.onnx --input keyboard
build/g1_cpp/sim2sim_mimic --config g1_mimic.yaml --model g1_dance.onnx --input keyboard
build/g1_cpp/sim2sim_attention --config g1_attention.yaml --model g1_attention1.onnx --input keyboard
```

无交互检查：

```bash
build/g1_cpp/sim2sim_attention --check --no_render --input const --const_vx 0.3
```

键盘控制：

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

`sim2sim_sdk2_bridge` 当前先作为 C++ bridge scaffold：它确认 `unitree_sdk2` 已在 `g1_cpp/unitree_sdk2` 下，可通过 `-DG1_CPP_BUILD_SDK2=ON` 导入 SDK target。要做到 Python bridge 的完整 LowCmd/LowState parity，下一步需要把 `common/sdk2_mujoco_bridge.py` 中的 DDS topic、CRC、IMU/sensor 填充逻辑移植到这个入口。
