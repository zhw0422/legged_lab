# G1 Deploy

Unitree G1 29-DOF policy deployment for Sim2Sim with MuJoCo and Sim2Real on the robot.

---

## Sim2Sim (MuJoCo)

Use MuJoCo to validate exported policies with either a GameSir USB gamepad or keyboard input.

### Dependencies

```bash
conda activate env_isaaclab1
pip install onnxruntime numpy scipy pyyaml mujoco pygame
```

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
python deploy/g1_deploy/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad
```

Keyboard:

```bash
python deploy/g1_deploy/sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input keyboard
```

### 2. AMP Walk / Run Velocity Policies

Use `g1_amp.yaml` with `g1_walk.onnx` or `g1_run.onnx`. The policy input is 384-dimensional:

`history_length=4`, each frame is 96 dimensions, and frames are stacked by feature group before ONNX inference.

Check config, ONNX dimensions, and MuJoCo joint/actuator mapping:

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --check
```

Run the walk AMP policy:

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad
```

Run the run AMP policy:

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_run.onnx \
  --input gamepad
```

Keyboard:

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input keyboard
```

Gamepad axis debug:

```bash
python deploy/g1_deploy/sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad \
  --debug_gamepad
```

### 3. Motion Tracking / Mimic

Use `g1_mimic.yaml` with `g1_dance.onnx` or `g1_jump.onnx`. The tracking ONNX embeds the reference motion clip. It takes the current observation and `time_step`, then outputs actions and reference joint/body states.

Dance:

```bash
python deploy/g1_deploy/sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_dance.onnx
```

Jump:

```bash
python deploy/g1_deploy/sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_jump.onnx
```

`sim2sim_mimic.py` starts with `g1_flat_1.onnx` for standing stabilization. After the robot is stable, press **RB + B** to switch to the tracking policy specified by `--model`.

### Controls

#### Walk (`sim2sim_walk.py`)

Gamepad:

| Input | Function |
| --- | --- |
| Left joystick up/down | `vx` forward/back |
| Left joystick left/right | `vy` strafe |
| Right joystick left/right | `vyaw` turn |
| **RB + A** | Walk policy |
| **RB + B/X/Y** | Policy slots 1/2/3 placeholders |
| **Start** | Exit |

Keyboard:

| Input | Function |
| --- | --- |
| **W/S** or up/down arrows | Increase/decrease `vx` |
| **A/D** | Increase/decrease `vy` |
| **Q/E** or left/right arrows | Increase/decrease `vyaw` |
| **Space** or **0** | Zero velocity command |
| **1/2/3/4** | Switch policy slot |
| **X** or **Esc** | Exit |

#### AMP (`sim2sim_amp.py`)

Gamepad:

| Input | Function |
| --- | --- |
| Left joystick up | `vx` forward. Backward command is disabled in the current `g1_amp.yaml`. |
| Left joystick left/right | `vy` strafe |
| Right joystick left/right | `vyaw` turn |
| **RB + A** | AMP policy |
| **RB + B/X/Y** | Policy slots 1/2/3 placeholders |
| **Start** | Exit |

Keyboard:

| Input | Function |
| --- | --- |
| **W/S** or up/down arrows | Increase/decrease `vx` |
| **A/D** | Increase/decrease `vy` |
| **Q/E** or left/right arrows | Increase/decrease `vyaw` |
| **Space** or **0** | Zero velocity command |
| **1/2/3/4** | Switch policy slot |
| **X** or **Esc** | Exit |

#### Motion Tracking (`sim2sim_mimic.py`)

| Input | Function |
| --- | --- |
| **RB + A** | Flat walk stabilization policy |
| **RB + B** | Main mimic/tracking policy specified by `--model` |
| **RB + X** | `g1_jump.onnx` |
| **RB + Y** | `g1_dance.onnx` |
| **Start** | Exit |

Policy switches are printed in the terminal with messages such as `[PolicySwitch] Active policy: N`.

### Policy Switching

Each Sim2Sim script defines a `policy_registry` near the bottom of the file. Edit the YAML path and ONNX filename there to register new policy slots.

Current tracking registry structure:

```python
policy_registry = {
    1: (flat_config,  'g1_flat_1.onnx'),  # RB+A: stand / stabilize
    2: (mimic_config, args.model),        # RB+B: main mimic model
    3: (mimic_config, 'g1_jump.onnx'),    # RB+X
    4: (mimic_config, 'g1_dance.onnx'),   # RB+Y
}
```

---

## Sim2Real

### Installation

```bash
conda activate env_isaaclab1
cd deploy/g1_deploy

# 1. Install Cyclone DDS first. It is required by unitree_sdk2_python.
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir -p build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install -j"$(nproc)"

# 2. Reinstall Python headers if Python.h is missing in env_isaaclab1.
conda install -n env_isaaclab1 --force-reinstall -y python=3.11.14

# 3. Install unitree_sdk2_python.
cd ../..
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
export CYCLONEDDS_HOME=$(pwd)/../cyclonedds/install
pip install -e .
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
