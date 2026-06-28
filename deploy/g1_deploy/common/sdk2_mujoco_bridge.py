import os
import struct
import sys
import time
from pathlib import Path
from threading import Lock

import mujoco
import mujoco.viewer
import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
G1_DEPLOY_DIR = THIS_DIR.parent
SDK2_PYTHON_DIR = G1_DEPLOY_DIR / "unitree_sdk2_python"

if str(SDK2_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(SDK2_PYTHON_DIR))


def _bootstrap_cyclonedds_home() -> None:
    env_home = os.environ.get("CYCLONEDDS_HOME", "")
    if env_home and (Path(env_home) / "lib" / "libddsc.so").exists():
        return

    candidate = G1_DEPLOY_DIR / "cyclonedds" / "install"
    if (candidate / "lib" / "libddsc.so").exists():
        os.environ["CYCLONEDDS_HOME"] = str(candidate)
        print(f"[CycloneDDS] Using CYCLONEDDS_HOME={candidate}")


_bootstrap_cyclonedds_home()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__SportModeState_,
    unitree_go_msg_dds__WirelessController_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, WirelessController_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from common.remote_controller import KeyMap


TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"
MOTOR_SENSOR_NUM = 3

BUTTON_BINDINGS = {
    "enter": KeyMap.start,
    "1": KeyMap.start,
    "2": KeyMap.A,
    "3": KeyMap.B,
    "4": KeyMap.X,
    "5": KeyMap.Y,
    "9": KeyMap.select,
}

JOYSTICK_MAPS = {
    "xbox": {
        "axis": {"LX": 0, "LY": 1, "RX": 3, "RY": 4, "LT": 2, "RT": 5},
        "button": {"X": 2, "Y": 3, "B": 1, "A": 0, "LB": 4, "RB": 5, "SELECT": 6, "START": 7},
    },
    "switch": {
        "axis": {"LX": 0, "LY": 1, "RX": 2, "RY": 3, "LT": 5, "RT": 4},
        "button": {"X": 3, "Y": 4, "B": 1, "A": 0, "LB": 6, "RB": 7, "SELECT": 10, "START": 11},
    },
}


class Sdk2MujocoBridge:
    """Unitree-mujoco style SDK2 bridge for G1.

    This follows unitree_mujoco/simulate_python:
    - MuJoCo sensors provide LowState motor q/dq/tau and IMU data.
    - LowCmd is converted to MuJoCo actuator control.
    - Wireless controller is exposed both in LowState.wireless_remote and
      rt/wirelesscontroller.
    """

    def __init__(
        self,
        config_path: str | os.PathLike,
        net: str = "lo",
        domain_id: int = 1,
        input_mode: str = "keyboard",
        joystick_type: str = "switch",
        lowcmd_topic: str | None = None,
        lowstate_topic: str | None = None,
        render: bool = True,
        print_rate: float = 1.0,
        print_scene: bool = False,
        elastic_band: bool = False,
        clamp_ctrl: bool = True,
        debug_lowcmd: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        if self.config.get("msg_type", "hg") != "hg":
            raise ValueError("G1 SDK2 MuJoCo bridge supports msg_type='hg'.")

        self.domain_id = int(domain_id)
        self.net = net
        self.input_mode = input_mode
        self.render = render
        self.print_rate = print_rate
        self.clamp_ctrl = clamp_ctrl
        self.debug_lowcmd = debug_lowcmd
        self._last_debug_time = 0.0
        self.lowcmd_topic = lowcmd_topic or self.config.get("lowcmd_topic", TOPIC_LOWCMD)
        self.lowstate_topic = lowstate_topic or self.config.get("lowstate_topic", TOPIC_LOWSTATE)

        self.xml_path = self._resolve_path(self.config["xml_path"], base_dir=G1_DEPLOY_DIR)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(self.config.get("sim_dt", 0.005))

        self.num_motor = self.model.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.have_frame_sensor = self._has_sensor("frame_pos")
        self.imu_source = self.config.get("imu_source", "qpos_qvel")
        self.ctrl_range = self.model.actuator_ctrlrange[: self.num_motor].copy()
        self.elastic_band = ElasticBand() if elastic_band else None
        self.elastic_body_id = self.model.body("torso_link").id if self.elastic_band is not None else None
        self.anchor_body_id = self.model.body(self.config.get("anchor_body_name", "torso_link")).id

        default_joint_pos = np.asarray(self.config["default_joint_pos"], dtype=np.float32)
        isaac_to_mujoco = np.asarray(self.config["isaac_to_mujoco_map"], dtype=np.int32)
        joint_names = list(self.config["joint_names_mujoco"])
        qpos_addrs = [self.model.jnt_qposadr[self.model.joint(name).id] for name in joint_names]
        self.data.qpos[qpos_addrs] = default_joint_pos[isaac_to_mujoco]
        self.data.qpos[2] = float(self.config.get("init_height", 0.90))
        mujoco.mj_forward(self.model, self.data)

        ChannelFactoryInitialize(self.domain_id, self.net)

        self.crc = CRC()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.low_state.tick = 1
        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.low_cmd_lock = Lock()
        self.last_cmd_time = 0.0
        self.low_cmd_received = False
        self.low_cmd_q = np.zeros(self.num_motor, dtype=np.float32)
        self.low_cmd_dq = np.zeros(self.num_motor, dtype=np.float32)
        self.low_cmd_kp = np.zeros(self.num_motor, dtype=np.float32)
        self.low_cmd_kd = np.zeros(self.num_motor, dtype=np.float32)
        self.low_cmd_tau = np.zeros(self.num_motor, dtype=np.float32)

        self.remote = SimRemoteState(input_mode=input_mode, joystick_type=joystick_type)

        self.low_state_pub = ChannelPublisher(self.lowstate_topic, LowState_)
        self.low_state_pub.Init()
        self.high_state_pub = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_pub.Init()
        self.wireless_controller_pub = ChannelPublisher(TOPIC_WIRELESS_CONTROLLER, WirelessController_)
        self.wireless_controller_pub.Init()
        self.low_cmd_sub = ChannelSubscriber(self.lowcmd_topic, LowCmd_)
        self.low_cmd_sub.Init(self.lowcmd_handler, 10)

        if print_scene:
            self.print_scene_information()

    def _load_config(self, path: Path) -> dict:
        if not path.exists():
            path = G1_DEPLOY_DIR / "config" / str(path)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_path(self, value: str, base_dir: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        direct = (base_dir / path).resolve()
        if direct.exists():
            return direct
        return (base_dir / path.name).resolve()

    def _has_sensor(self, name: str) -> bool:
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name == name:
                return True
        return False

    def lowcmd_handler(self, msg: LowCmd_) -> None:
        with self.low_cmd_lock:
            for i in range(self.num_motor):
                motor = msg.motor_cmd[i]
                self.low_cmd_q[i] = motor.q
                self.low_cmd_dq[i] = motor.dq
                self.low_cmd_kp[i] = motor.kp
                self.low_cmd_kd[i] = motor.kd
                self.low_cmd_tau[i] = motor.tau
            self.low_cmd_received = True
            self.low_state.mode_pr = msg.mode_pr
            self.low_state.mode_machine = msg.mode_machine
            self.last_cmd_time = time.time()

    def _apply_lowcmd_control(self) -> None:
        with self.low_cmd_lock:
            if not self.low_cmd_received:
                self.data.ctrl[: self.num_motor] = 0.0
                return
            q_des = self.low_cmd_q.copy()
            dq_des = self.low_cmd_dq.copy()
            kp = self.low_cmd_kp.copy()
            kd = self.low_cmd_kd.copy()
            tau = self.low_cmd_tau.copy()

        q = self.data.sensordata[: self.num_motor]
        dq = self.data.sensordata[self.num_motor : 2 * self.num_motor]
        raw_ctrl = tau + kp * (q_des - q) + kd * (dq_des - dq)

        if self.clamp_ctrl:
            self.data.ctrl[: self.num_motor] = np.clip(raw_ctrl, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        else:
            self.data.ctrl[: self.num_motor] = raw_ctrl

        if self.debug_lowcmd and time.time() - self._last_debug_time > 0.5:
            clipped = int(np.count_nonzero(np.abs(raw_ctrl - self.data.ctrl[: self.num_motor]) > 1e-5))
            max_idx = int(np.argmax(np.abs(raw_ctrl)))
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, max_idx)
            print(
                "[LowCmdDebug] "
                f"q_des=[{q_des.min():+.3f},{q_des.max():+.3f}] "
                f"kp=[{kp.min():.1f},{kp.max():.1f}] kd=[{kd.min():.1f},{kd.max():.1f}] "
                f"ctrl_raw=[{raw_ctrl.min():+.1f},{raw_ctrl.max():+.1f}] "
                f"clipped={clipped}/{self.num_motor} max={name}:{raw_ctrl[max_idx]:+.1f}"
            )
            self._last_debug_time = time.time()

    def publish_lowstate(self) -> None:
        self.low_state.tick = (self.low_state.tick + 1) & 0xFFFFFFFF

        for i in range(self.num_motor):
            motor = self.low_state.motor_state[i]
            motor.mode = 1
            motor.q = float(self.data.sensordata[i])
            motor.dq = float(self.data.sensordata[i + self.num_motor])
            motor.tau_est = float(self.data.sensordata[i + 2 * self.num_motor])

        if self.imu_source == "sensor" and self.have_frame_sensor:
            base = self.dim_motor_sensor
            self.low_state.imu_state.quaternion[:] = [float(x) for x in self.data.sensordata[base : base + 4]]
            self.low_state.imu_state.gyroscope[:] = [float(x) for x in self.data.sensordata[base + 4 : base + 7]]
            self.low_state.imu_state.accelerometer[:] = [float(x) for x in self.data.sensordata[base + 7 : base + 10]]
            self.low_state.imu_state.rpy[:] = [
                float(x) for x in self._quat_wxyz_to_rpy(self.low_state.imu_state.quaternion)
            ]
        else:
            quat = self.data.qpos[3:7]
            self.low_state.imu_state.quaternion[:] = [float(x) for x in quat]
            self.low_state.imu_state.gyroscope[:] = [float(x) for x in self.data.qvel[3:6]]
            self.low_state.imu_state.accelerometer[:] = [0.0, 0.0, 0.0]
            self.low_state.imu_state.rpy[:] = [float(x) for x in self._quat_wxyz_to_rpy(quat)]

        self.remote.poll()
        self.low_state.wireless_remote[:] = self.remote.pack_wireless_remote()
        self.low_state.crc = self.crc.Crc(self.low_state)
        self.low_state_pub.Write(self.low_state)

    def publish_highstate(self) -> None:
        anchor_pos = self.data.xpos[self.anchor_body_id]
        anchor_quat = self.data.xquat[self.anchor_body_id]
        self.high_state.position[:] = [float(x) for x in anchor_pos]
        self.high_state.imu_state.quaternion[:] = [float(x) for x in anchor_quat]
        self.high_state.velocity[:] = [float(x) for x in self.data.qvel[:3]]
        self.high_state_pub.Write(self.high_state)

    def publish_wireless_controller(self) -> None:
        self.wireless_controller.keys = int(self.remote.keys)
        self.wireless_controller.lx = float(self.remote.lx)
        self.wireless_controller.ly = float(self.remote.ly)
        self.wireless_controller.rx = float(self.remote.rx)
        self.wireless_controller.ry = float(self.remote.ry)
        self.wireless_controller_pub.Write(self.wireless_controller)

    def run(self) -> None:
        print(
            f"[SDK2 Bridge] domain={self.domain_id} net={self.net} "
            f"publish {self.lowstate_topic}/{TOPIC_WIRELESS_CONTROLLER}, subscribe {self.lowcmd_topic}"
        )
        print(f"[SDK2 Bridge] actuator ctrlrange clamp={'on' if self.clamp_ctrl else 'off'}")
        print(f"[SDK2 Bridge] imu_source={self.imu_source}")
        if self.input_mode == "keyboard":
            print("[SDK2 Bridge] axes: W/S=ly, A/D=lx, Q/E=rx, Space=zero")
            print("[SDK2 Bridge] buttons: Enter/1=start, 2=A(control), 3=B, 4=X, 5=Y, 9=select/exit")
        if self.elastic_band is not None:
            print("[SDK2 Bridge] elastic band: viewer key 9=toggle, 8=lower/less support, 7=lift/more support")

        if self.render:
            key_callback = self.elastic_band.mujoco_key_callback if self.elastic_band is not None else None
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as viewer:
                self._setup_follow_camera(viewer)
                self._run_loop(viewer)
        else:
            self._run_loop(None)

    def _setup_follow_camera(self, viewer) -> None:
        viewer.cam.lookat[:] = self.data.qpos[:3]
        viewer.cam.distance = float(self.config.get("camera_distance", 2.0))
        viewer.cam.azimuth = float(self.config.get("camera_azimuth", 90.0))
        viewer.cam.elevation = float(self.config.get("camera_elevation", -20.0))

    def _update_follow_camera(self, viewer) -> None:
        viewer.cam.lookat[:] = self.data.qpos[:3]

    def _run_loop(self, viewer) -> None:
        sim_dt = self.model.opt.timestep
        viewer_dt = 0.02
        wireless_dt = 0.01
        next_viewer_time = time.perf_counter()
        next_wireless_time = time.perf_counter()
        last_print_time = time.perf_counter()

        while viewer is None or viewer.is_running():
            step_start = time.perf_counter()
            if self.remote.exit_requested:
                break

            self.data.xfrc_applied[:] = 0.0
            if self.elastic_band is not None and self.elastic_band.enable:
                self.data.xfrc_applied[self.elastic_body_id, :3] = self.elastic_band.advance(
                    self.data.qpos[:3], self.data.qvel[:3]
                )
            self._apply_lowcmd_control()
            mujoco.mj_step(self.model, self.data)
            self.publish_lowstate()
            self.publish_highstate()

            now = time.perf_counter()
            if now >= next_wireless_time:
                self.publish_wireless_controller()
                next_wireless_time = now + wireless_dt

            if viewer is not None and now >= next_viewer_time:
                self._update_follow_camera(viewer)
                viewer.sync()
                next_viewer_time = now + viewer_dt

            if self.print_rate > 0.0 and now - last_print_time >= self.print_rate:
                cmd_age = time.time() - self.last_cmd_time if self.last_cmd_time > 0.0 else float("inf")
                print(
                    f"[SDK2 Bridge] t={self.data.time:.2f}s height={self.data.qpos[2]:.3f} "
                    f"cmd_age={cmd_age:.3f}s keys=0x{self.remote.keys:04x}"
                )
                last_print_time = now

            sleep_s = sim_dt - (time.perf_counter() - step_start)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    def _quat_wxyz_to_rpy(self, q) -> np.ndarray:
        w, x, y, z = q
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = np.sign(sinp) * np.pi / 2.0 if abs(sinp) >= 1.0 else np.arcsin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def print_scene_information(self) -> None:
        print("<<------------- Joint ------------->>")
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                print("joint_index:", i, ", name:", name)
        print("<<------------- Actuator ------------->>")
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                print("actuator_index:", i, ", name:", name)
        print("<<------------- Sensor ------------->>")
        index = 0
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                print("sensor_index:", index, ", name:", name, ", dim:", self.model.sensor_dim[i])
            index += self.model.sensor_dim[i]


class SimRemoteState:
    def __init__(self, input_mode: str = "keyboard", joystick_type: str = "switch") -> None:
        self.input_mode = input_mode
        self.joystick_type = joystick_type
        self.keys = 0
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.axis_step = 0.1
        self.button_hold_s = 0.2
        self._held_until = {}
        self.exit_requested = False
        self._keyboard = None
        self._pygame = None
        self._joystick = None
        self.axis_id = JOYSTICK_MAPS[joystick_type]["axis"]
        self.button_id = JOYSTICK_MAPS[joystick_type]["button"]

    def poll(self) -> None:
        if self.input_mode == "gamepad":
            self.poll_gamepad()
        else:
            self.poll_keyboard()

    def _ensure_keyboard(self) -> None:
        if self._keyboard is not None:
            return
        import select
        import termios
        import tty

        if not sys.stdin.isatty():
            return
        self._keyboard = (select, termios, tty, termios.tcgetattr(sys.stdin))
        tty.setcbreak(sys.stdin.fileno())

    def poll_keyboard(self) -> None:
        self._ensure_keyboard()
        if self._keyboard is None:
            self._update_held_keys()
            return
        select, _, _, _ = self._keyboard
        while select.select([sys.stdin], [], [], 0.0)[0]:
            ch = sys.stdin.read(1)
            if ch in ("\x03", "\x1b"):
                self.exit_requested = True
                return
            if ch in ("\r", "\n"):
                self._pulse_key(KeyMap.start)
            else:
                self._apply_key(ch.lower())
        self._update_held_keys()

    def _apply_key(self, ch: str) -> None:
        if ch == "w":
            self.ly = min(1.0, self.ly + self.axis_step)
        elif ch == "s":
            self.ly = max(-1.0, self.ly - self.axis_step)
        elif ch == "a":
            self.lx = min(1.0, self.lx + self.axis_step)
        elif ch == "d":
            self.lx = max(-1.0, self.lx - self.axis_step)
        elif ch == "q":
            self.rx = min(1.0, self.rx + self.axis_step)
        elif ch == "e":
            self.rx = max(-1.0, self.rx - self.axis_step)
        elif ch in (" ", "0"):
            self.lx = 0.0
            self.ly = 0.0
            self.rx = 0.0
            self.ry = 0.0
        else:
            key = BUTTON_BINDINGS.get(ch)
            if key is not None:
                self._pulse_key(key)

    def poll_gamepad(self) -> None:
        self._ensure_gamepad()
        if self._pygame is None or self._joystick is None:
            return

        pygame = self._pygame
        pygame.event.get()
        key_state = [0] * 16
        self._set_button_key(key_state, KeyMap.R1, "RB")
        self._set_button_key(key_state, KeyMap.L1, "LB")
        self._set_button_key(key_state, KeyMap.start, "START")
        self._set_button_key(key_state, KeyMap.select, "SELECT")
        self._set_axis_key(key_state, KeyMap.R2, "RT")
        self._set_axis_key(key_state, KeyMap.L2, "LT")
        self._set_button_key(key_state, KeyMap.A, "A")
        self._set_button_key(key_state, KeyMap.B, "B")
        self._set_button_key(key_state, KeyMap.X, "X")
        self._set_button_key(key_state, KeyMap.Y, "Y")

        if self._joystick.get_numhats() > 0:
            hat = self._joystick.get_hat(0)
            key_state[KeyMap.up] = int(hat[1] > 0)
            key_state[KeyMap.right] = int(hat[0] > 0)
            key_state[KeyMap.down] = int(hat[1] < 0)
            key_state[KeyMap.left] = int(hat[0] < 0)

        self.keys = sum(int(key_state[i]) << i for i in range(16))
        if key_state[KeyMap.select]:
            self.exit_requested = True

        self.lx = self._axis("LX")
        self.ly = -self._axis("LY")
        self.rx = self._axis("RX")
        self.ry = -self._axis("RY")

    def _set_button_key(self, key_state: list[int], key: int, button: str) -> None:
        btn_idx = self.button_id[button]
        if btn_idx < self._joystick.get_numbuttons():
            key_state[key] = int(self._joystick.get_button(btn_idx))

    def _set_axis_key(self, key_state: list[int], key: int, axis: str) -> None:
        axis_idx = self.axis_id[axis]
        if axis_idx < self._joystick.get_numaxes():
            key_state[key] = int(self._joystick.get_axis(axis_idx) > 0)

    def _axis(self, axis: str) -> float:
        axis_idx = self.axis_id[axis]
        if axis_idx >= self._joystick.get_numaxes():
            return 0.0
        value = self._joystick.get_axis(axis_idx)
        return 0.0 if abs(value) < 0.05 else float(np.clip(value, -1.0, 1.0))

    def _ensure_gamepad(self) -> None:
        if self._pygame is not None:
            return
        try:
            import pygame
        except ImportError:
            print("[SDK2 Bridge] pygame is not installed; gamepad input disabled.")
            return
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("[SDK2 Bridge] no gamepad detected; gamepad input disabled.")
            self._pygame = pygame
            return
        self._pygame = pygame
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        print(f"[SDK2 Bridge] gamepad: {self._joystick.get_name()} ({self.joystick_type})")

    def _pulse_key(self, key: int) -> None:
        if key == KeyMap.select:
            self.exit_requested = True
        self._held_until[key] = time.time() + self.button_hold_s

    def _update_held_keys(self) -> None:
        now = time.time()
        active_keys = 0
        for key, held_until in list(self._held_until.items()):
            if held_until >= now:
                active_keys |= 1 << key
            else:
                del self._held_until[key]
        self.keys = active_keys

    def pack_wireless_remote(self) -> list[int]:
        data = bytearray(40)
        struct.pack_into("H", data, 2, self.keys & 0xFFFF)
        struct.pack_into("f", data, 4, float(self.lx))
        struct.pack_into("f", data, 8, float(self.rx))
        struct.pack_into("f", data, 12, float(self.ry))
        struct.pack_into("f", data, 20, float(self.ly))
        return list(data)

    def close(self) -> None:
        if self._keyboard is not None:
            _, termios, _, old = self._keyboard
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            self._keyboard = None
        if self._pygame is not None:
            self._pygame.quit()


class ElasticBand:
    def __init__(self) -> None:
        self.stiffness = 200.0
        self.damping = 100.0
        self.point = np.array([0.0, 0.0, 3.0])
        self.length = 0.0
        self.enable = True

    def advance(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        delta = self.point - x
        distance = np.linalg.norm(delta)
        if distance < 1e-6:
            return np.zeros(3)
        direction = delta / distance
        velocity = np.dot(dx, direction)
        return (self.stiffness * (distance - self.length) - self.damping * velocity) * direction

    def mujoco_key_callback(self, key: int) -> None:
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        elif key == glfw.KEY_8:
            self.length += 0.1
        elif key == glfw.KEY_9:
            self.enable = not self.enable
