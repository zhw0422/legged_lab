# G1 Deploy

Unitree G1 29-DOF policy deployment for Sim2Sim with MuJoCo and Sim2Real on the robot.

## SDK2 Closed-Loop Sim2Sim Quick Start

This mode follows `unitree_mujoco/simulate_python`: MuJoCo behaves like a real G1, the bridge publishes `rt/lowstate`, `rt/wirelesscontroller`, and `rt/sportmodestate`, the deploy controller publishes `rt/lowcmd`, and the full SDK2/DDS loop is exercised.

Terminal 1: start the MuJoCo fake robot and SDK2 bridge.

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2sim_sdk2_bridge.py --config g1_walk.yaml --net lo --domain_id 1 --input gamepad --joystick_type switch --elastic_band
```

Keyboard input:

```bash
python sim2sim_sdk2_bridge.py --config g1_walk.yaml --net lo --domain_id 1 --input keyboard --elastic_band
```

Terminal 2: start the deploy controller against the local DDS bridge.

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2real_walk.py --net lo --domain_id 1 --config_path config/g1_walk.yaml
```

If CycloneDDS is not using `lo`, pass the same network interface to both commands, for example `enp108s0`. Local simulation uses domain `1`; the real robot uses domain `0` by default.

---

## Sim2Sim (MuJoCo)

Use MuJoCo to validate exported policies with either a GameSir USB gamepad or keyboard input.

### Dependencies

```bash
conda activate legged_rl_lab
pip install onnxruntime numpy scipy pyyaml mujoco pygame
```

### Gamepad Mapping Check

Before using a gamepad for the first time, run the mapping check script:

```bash
python deploy/utils/test_joystick.py
```

Press buttons or move joysticks. The terminal prints button indices, axis indices, and Hat values in real time. The current measured GameSir mapping is:

| Control | Default index |
| --- | --- |
| A / B / X / Y | `button 0 / 1 / 3 / 4` |
| LB / RB | `button 6 / 7` |
| LT | `axis 5`, also triggers `button 8` |
| RT | `axis 4`, also triggers `button 9` |
| View / Menu | `button 10 / 11` |
| Home | `button 12` |
| Left joystick X / Y | `axis 0 / axis 1` |
| Right joystick X / Y | `axis 2 / axis 3` |
| D-pad | `Hat 0`, returned as `(x, y)` |

For `Hat 0`, the first value is left/right: left is `-1`, right is `1`, so `(-1, 0)` / `(1, 0)`. The second value is up/down: up is `1`, down is `-1`, so `(0, 1)` / `(0, -1)`.

If the detected indices differ from this table, update the `gamepad_btn_*` / `axis_*` settings in the YAML or script to match the printed values.

### Current ONNX Layout

`deploy/g1_deploy/exported_policy/` currently contains:

| ONNX | Purpose | Script | Inputs | Outputs |
| --- | --- | --- | --- | --- |
| `g1_flat_1.onnx` | Flat walk, standing stabilization, basic velocity control | `sim2sim_walk.py`, startup policy for `sim2sim_mimic.py` | `obs [1, 96]` | `actions [1, 29]` |
| `g1_walk.onnx` | AMP walk policy | `sim2sim_amp.py` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_run.onnx` | AMP run policy | `sim2sim_amp.py` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_dance.onnx` | Motion tracking dance policy | `sim2sim_mimic.py` | `obs [1, 160]`, `time_step [1, 1]` | `actions [1, 29]` plus reference states |
| `g1_jump.onnx` | Motion tracking jump policy | `sim2sim_mimic.py` | `obs [1, 160]`, `time_step [1, 1]` | `actions [1, 29]` plus reference states |

The current `exported_policy/` folder does not contain `g1_amp.onnx` or `policy.onnx`. Use the model names listed above.

### 1. Walk / Basic Velocity Control

Use `g1_walk.yaml` with `g1_flat_1.onnx`. The policy input is one 96-dimensional observation frame:

`base_ang_vel(3) + projected_gravity(3) + command(3) + joint_pos(29) + joint_vel(29) + last_action(29)`.

Gamepad:

```bash
python deploy/g1_deploy/g1_python/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad
```

Keyboard:

```bash
python deploy/g1_deploy/g1_python/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input keyboard
```

Controls:

Gamepad:

| Input | Function |
| --- | --- |
| Left joystick up/down | `vx` forward/back |
| Left joystick left/right | `vy` strafe |
| Right joystick left/right | `vyaw` turn |
| **RB + A** | Walk policy |
| **RB + B/X/Y** | Policy slots 1/2/3 placeholders |
| **Menu / Start** | Exit |

Keyboard:

| Input | Function |
| --- | --- |
| **W/S** or up/down arrows | Increase/decrease `vx` |
| **A/D** | Increase/decrease `vy` |
| **Q/E** or left/right arrows | Increase/decrease `vyaw` |
| **Space** or **0** | Zero velocity command |
| **1/2/3/4** | Switch policy slot |
| **X** or **Esc** | Exit |

### 2. AMP Walk / Run Velocity Policies

Use `g1_amp.yaml` with `g1_walk.onnx` or `g1_run.onnx`. The policy input is 384-dimensional:

`history_length=4`, each frame is 96 dimensions, and frames are stacked by feature group before ONNX inference.

Check config, ONNX dimensions, and MuJoCo joint/actuator mapping:

```bash
python deploy/g1_deploy/g1_python/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --check
```

Run the walk AMP policy:

```bash
python deploy/g1_deploy/g1_python/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad
```

Run the run AMP policy:

```bash
python deploy/g1_deploy/g1_python/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_run.onnx \
  --input gamepad
```

Keyboard:

```bash
python deploy/g1_deploy/g1_python/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input keyboard
```

Gamepad axis debug:

```bash
python deploy/g1_deploy/g1_python/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad \
  --debug_gamepad
```

Controls:

Gamepad:

| Input | Function |
| --- | --- |
| Left joystick up | `vx` forward. Backward command is disabled in the current `g1_amp.yaml`. |
| Left joystick left/right | `vy` strafe |
| Right joystick left/right | `vyaw` turn |
| **RB + A** | AMP policy |
| **RB + B/X/Y** | Policy slots 1/2/3 placeholders |
| **Menu / Start** | Exit |

Keyboard:

| Input | Function |
| --- | --- |
| **W/S** or up/down arrows | Increase/decrease `vx` |
| **A/D** | Increase/decrease `vy` |
| **Q/E** or left/right arrows | Increase/decrease `vyaw` |
| **Space** or **0** | Zero velocity command |
| **1/2/3/4** | Switch policy slot |
| **X** or **Esc** | Exit |

### 3. Motion Tracking / Mimic

Use `g1_mimic.yaml` with `g1_dance.onnx` or `g1_jump.onnx`. The tracking ONNX embeds the reference motion clip. It takes the current observation and `time_step`, then outputs actions and reference joint/body states.

Dance:

```bash
python deploy/g1_deploy/g1_python/sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_dance.onnx
```

Jump:

```bash
python deploy/g1_deploy/g1_python/sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_jump.onnx
```

Keyboard:

```bash
python deploy/g1_deploy/g1_python/sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_dance.onnx \
  --input keyboard
```

`sim2sim_mimic.py` starts with `g1_flat_1.onnx` for standing stabilization. After the robot is stable, press **B** (gamepad) or **2** (keyboard) to switch to the tracking policy specified by `--model`.

Controls:

Gamepad:

| Input | Function |
| --- | --- |
| **A** | Flat walk stabilization policy |
| **B** | Main mimic/tracking policy specified by `--model` |
| **X** | `g1_jump.onnx` |
| **Y** | `g1_dance.onnx` |
| **View / Select** | Exit |

Keyboard:

| Input | Function |
| --- | --- |
| **1** | Flat walk stabilization policy |
| **2** | Main mimic/tracking policy specified by `--model` |
| **3** | `g1_jump.onnx` |
| **4** | `g1_dance.onnx` |
| **X** or **Esc** | Exit |

Policy switches are printed in the terminal with messages such as `[PolicySwitch] Active policy: N`.

### Policy Switching

Each Sim2Sim script defines a `policy_registry` near the bottom of the file. Edit the YAML path and ONNX filename there to register new policy slots.

Current tracking registry structure:

```python
policy_registry = {
    1: (flat_config,  'g1_flat_1.onnx'),  # 1 / A: stand / stabilize
    2: (mimic_config, args.model),        # 2 / B: main mimic model
    3: (mimic_config, 'g1_jump.onnx'),    # 3 / X
    4: (mimic_config, 'g1_dance.onnx'),   # 4 / Y
}
```

---

## SDK2 Closed-Loop Sim2Sim (MuJoCo as the DDS Robot)

The bridge follows the `unitree_mujoco/simulate_python` split:

```text
keyboard/gamepad simulation
  ↓
sim bridge writes LowState.wireless_remote
  ↓
deploy controller subscribes rt/lowstate and reads robot/gamepad state
  ↓
deploy controller runs policy/PD and publishes rt/lowcmd
  ↓
sim bridge subscribes rt/lowcmd
  ↓
LowCmd → MuJoCo actuator torque
  ↓
MuJoCo step
  ↓
sim bridge publishes the next rt/lowstate
```

SDK2 is only the DDS communication layer. The deploy controller is still responsible for parsing the remote state and generating `LowCmd`. G1/H1-2 use the `unitree_hg` IDL, so this bridge publishes/subscribes HG `LowState/LowCmd`. Like the official Python simulator, the bridge reads joint position, velocity, torque, and IMU from MuJoCo `sensordata`.

### Run Notes

Terminal 1: start the MuJoCo fake robot and SDK2 bridge.

```bash
cd deploy/g1_deploy/g1_python
python sim2sim_sdk2_bridge.py --config g1_walk.yaml --net lo --domain_id 1 --input keyboard --elastic_band
```

For pygame gamepad input:

```bash
python sim2sim_sdk2_bridge.py --config g1_walk.yaml --net lo --domain_id 1 --input gamepad --joystick_type switch --elastic_band
```

Terminal 2: start the deploy controller against the local DDS bridge.

```bash
cd deploy/g1_deploy/g1_python
python sim2real_walk.py --net lo --domain_id 1 --config_path config/g1_walk.yaml
```

If CycloneDDS is not using `lo`, pass the same network interface to both commands, for example `enp108s0`. For local closed-loop testing, start with `lo` and domain `1`.

### Bridge Keyboard Controls

| Input | Function |
| --- | --- |
| **Enter** or **1** | Simulate Start; controller moves to default posture |
| **2** | Simulate A; controller starts control |
| **3 / 4 / 5** | Simulate B / X / Y |
| **W/S** | `ly` forward/back command |
| **A/D** | `lx` lateral command |
| **Q/E** | `rx` yaw command |
| **Space** or **0** | Zero joystick axes |
| **9** or **Esc** | Select / exit |

The bridge validates the low-level SDK2 path: it subscribes `rt/lowcmd`, converts each command to MuJoCo torque with
`tau + kp * (q_des - q_sensor) + kd * (dq_des - dq_sensor)`, steps MuJoCo, and publishes `rt/lowstate`, `rt/wirelesscontroller`, and `rt/sportmodestate`. This is closer to the real robot loop than running `sim2sim_walk.py` directly, and is useful for checking DDS topics, joint order, `wireless_remote`, CRC, and deploy state-machine behavior.

Humanoids such as G1 can fall before the controller takes over. `--elastic_band` enables a virtual support band similar to the official `unitree_mujoco` simulator. In the MuJoCo viewer, press **9** to toggle the band and **7/8** to adjust its length.

---

## Sim2Real

### Installation

```bash
conda activate legged_rl_lab
cd deploy/g1_deploy

# 1. Install CMake. Skip this if cmake is already available.
python -m pip install cmake

# 2. Build and install the CycloneDDS C library.
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir -p build install
cd build
python -m cmake .. -DCMAKE_INSTALL_PREFIX=../install
python -m cmake --build . --target install -j"$(nproc)"

# 3. Install the bundled unitree_sdk2_python.
cd ../..
cd g1_python/unitree_sdk2_python
export CYCLONEDDS_HOME=$(pwd)/../../cyclonedds/install
pip install -e .

# 4. If pip upgrades numpy to 2.x, restore IsaacLab-compatible versions.
pip install numpy==1.26.0 opencv-python==4.10.0.84 packaging==23.0 wheel==0.45.1
```

### Startup Process

#### 1. Start the robot

Power on the G1 and leave it in zero-torque mode.

#### 2. Enter debug mode

![Debug mode](image.png)

Press **L2 + R2** to enter debug mode. The robot should be in damping mode.

You can press **L2 + A** to confirm debug mode, then press **L2 + R2** again to return to damping mode.

Safety note: in debug mode, press **L2 + B** to enter damping mode immediately.

#### 3. Connect the robot

Connect the PC to the robot with Ethernet. A USB Ethernet adapter or a direct Ethernet port both work.

Set the PC network interface to the `192.168.123.X` subnet. `192.168.123.99` is recommended.

![Network settings](image-1.png)

Check the network interface address:

![ifconfig output](image-2.png)

Verify connectivity:

```bash
ping 192.168.123.161
```

#### 4. Start the program

Assume the Ethernet interface is `enp108s0`.

```bash
cd deploy/g1_deploy/g1_python
python sim2real_walk.py
```

##### 4.1 Zero-torque state

After startup, the robot joints are in zero-torque mode. You can gently move the joints by hand to confirm this state.

##### 4.2 Default-position state

In zero-torque mode, press **Start** on the remote controller. The robot moves to the default joint posture.

After the robot reaches the default posture, slowly lower the support rig until the feet touch the ground.

##### 4.3 Motion-control state

After setup is complete, press **A** on the remote controller. The robot starts stepping in place. Once it is stable, gradually lower the support rig and allow limited free motion.

Remote-controller commands:

| Input | Function |
| --- | --- |
| Left joystick forward/back | X velocity |
| Left joystick left/right | Y velocity |
| Right joystick left/right | Yaw velocity |

##### 4.4 Exit control

In motion-control mode, press **Select** on the remote controller. The robot enters damping mode, drops safely, and the program exits. You can also stop the program from the terminal with `Ctrl+C`.
