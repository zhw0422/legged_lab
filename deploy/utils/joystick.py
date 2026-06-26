#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一手柄控制模块

当前 Sim2Sim 仅保留 GameSir 710 (pygame 接口)。

使用方法:
    from utils.joystick import create_gamepad_controller
    gamepad = create_gamepad_controller("gamesir", vx_range=(0, 1.0), vy_range=(-0.5, 0.5), vyaw_range=(-1.0, 1.0))
    gamepad.start()
    vx, vy, vyaw = gamepad.get_velocity()
"""

import struct
import threading
import time
import numpy as np


def apply_deadzone(value, deadzone=0.08):
    """应用死区，线性缩放保持平滑过渡"""
    if abs(value) < deadzone:
        return 0.0
    sign = 1 if value > 0 else -1
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


# ============================================================================
#  底层手柄读取器 (Linux /dev/input/jsX)
# ============================================================================

class LinuxJoystickReader:
    """
    直接读取 Linux joystick 设备 (/dev/input/jsX)
    该读取器当前仅作兼容保留，Sim2Sim 默认使用 GameSir pygame 读取器。

    事件格式 (8 bytes):
    - timestamp (4 bytes unsigned int)
    - value (2 bytes signed short)
    - type (1 byte unsigned char): 0x01=button, 0x02=axis
    - number (1 byte unsigned char): 按钮/轴编号
    """

    def __init__(self, device_path='/dev/input/js0'):
        self.device_path = device_path
        self.device_file = None
        self.running = False
        self.thread = None
        self.axes = [0] * 8
        self.buttons = [0] * 16
        self.lock = threading.Lock()

        try:
            self.device_file = open(self.device_path, 'rb')
            print(f"✅ Joystick opened: {self.device_path}")
        except Exception as e:
            print(f"❌ Failed to open joystick: {e}")
            raise

    def _read_thread(self):
        while self.running:
            try:
                event_data = self.device_file.read(8)
                if len(event_data) < 8:
                    break
                timestamp, value, event_type, number = struct.unpack('IhBB', event_data)
                event_type_masked = event_type & 0x7F
                with self.lock:
                    if event_type_masked == 0x01 and number < len(self.buttons):
                        self.buttons[number] = value
                    elif event_type_masked == 0x02 and number < len(self.axes):
                        self.axes[number] = value
            except Exception:
                break

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.device_file:
            self.device_file.close()


# ============================================================================
#  GameSir 底层读取器 (pygame 接口)
# ============================================================================

class PygameJoystickReader:
    """
    使用 pygame 读取 GameSir 手柄
    pygame 事件循环在后台线程中运行
    """

    # 轴索引
    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_RIGHT_X = 2
    AXIS_RIGHT_Y = 3
    AXIS_RT = 4
    AXIS_LT = 5

    # 按钮索引
    BTN_A = 0
    BTN_B = 1
    BTN_X = 3
    BTN_Y = 4
    BTN_LB = 6
    BTN_RB = 7
    BTN_LT = 8
    BTN_RT = 9
    BTN_SELECT = 10
    BTN_START = 11
    BTN_HOME = 12

    def __init__(self):
        import pygame
        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError("未检测到手柄，请先连接手柄")

        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        print(f"✅ GameSir joystick opened: {self.js.get_name()}")
        self.num_buttons = self.js.get_numbuttons()
        self.num_axes = self.js.get_numaxes()
        self.num_hats = self.js.get_numhats()
        print(f"   Buttons: {self.num_buttons}, Axes: {self.num_axes}, Hats: {self.num_hats}")

        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # 归一化后的轴值 [-1, 1]
        self.axes = [0.0] * 6
        self.buttons = [0] * 19  # 包含虚拟 D-pad 按钮
        self.hat = (0, 0)

    def _read_thread(self):
        while self.running:
            try:
                for event in self.pygame.event.get():
                    with self.lock:
                        if event.type == self.pygame.JOYBUTTONDOWN:
                            if event.button < 15:
                                self.buttons[event.button] = 1
                        elif event.type == self.pygame.JOYBUTTONUP:
                            if event.button < 15:
                                self.buttons[event.button] = 0
                        elif event.type == self.pygame.JOYAXISMOTION:
                            if event.axis < len(self.axes):
                                self.axes[event.axis] = event.value
                        elif event.type == self.pygame.JOYHATMOTION:
                            self.hat = event.value
                            # 映射 hat 到虚拟按钮
                            self.buttons[15] = 1 if event.value[1] == 1 else 0   # UP
                            self.buttons[16] = 1 if event.value[1] == -1 else 0  # DOWN
                            self.buttons[17] = 1 if event.value[0] == -1 else 0  # LEFT
                            self.buttons[18] = 1 if event.value[0] == 1 else 0   # RIGHT
                time.sleep(0.005)
            except Exception:
                time.sleep(0.01)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.pygame.quit()


# ============================================================================
#  GamepadController 基类
# ============================================================================

class GamepadController:
    """
    手柄速度控制器基类
    所有手柄实现需要提供统一的 get_velocity() / start() / stop() 接口
    """

    def __init__(
        self,
        vx_range=(-2.0, 4.0),
        vy_range=(-1.0, 1.0),
        vyaw_range=(-1.57, 1.57),
        deadzone=0.05,
        command_slew_rate=(2.0, 4.0, 3.0),
        debug=False,
        debug_interval=0.5,
    ):
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vyaw_range = vyaw_range
        self.lock = threading.Lock()
        self.running = True
        self.exit_requested = False
        self.thread = None

        self.deadzone = float(deadzone)
        if command_slew_rate is None:
            self.command_slew_rate = np.full(3, np.inf, dtype=np.float32)
        else:
            self.command_slew_rate = np.asarray(command_slew_rate, dtype=np.float32)
            if self.command_slew_rate.size == 1:
                self.command_slew_rate = np.repeat(self.command_slew_rate, 3)
            if self.command_slew_rate.shape != (3,):
                raise ValueError("command_slew_rate must be a scalar or [vx, vy, vyaw].")
            self.command_slew_rate = np.where(self.command_slew_rate <= 0.0, np.inf, self.command_slew_rate)
        self.debug = bool(debug)
        self.debug_interval = max(float(debug_interval), 1.0e-6)
        self._last_debug_print = 0.0
        self.vx_increment = 0.1
        self.dpad_vx = 0.0
        self.dpad_last_state = {'up': False, 'down': False}
        self.walk_requested = False  # RB+A 组合触发，进入 walk policy
        # 当前激活的策略索引: 0=空闲, 1=walk(RB+A), 2=policy1(RB+B), 3=policy2(RB+X), 4=policy3(RB+Y)
        self.active_policy = 0

    def get_velocity(self):
        with self.lock:
            return self.vx, self.vy, self.vyaw

    def set_velocity(self, vx, vy, vyaw):
        with self.lock:
            self.vx = np.clip(vx, self.vx_range[0], self.vx_range[1])
            self.vy = np.clip(vy, self.vy_range[0], self.vy_range[1])
            self.vyaw = np.clip(vyaw, self.vyaw_range[0], self.vyaw_range[1])

    def set_velocity_smooth(self, vx, vy, vyaw, dt):
        target = np.array(
            [
                np.clip(vx, self.vx_range[0], self.vx_range[1]),
                np.clip(vy, self.vy_range[0], self.vy_range[1]),
                np.clip(vyaw, self.vyaw_range[0], self.vyaw_range[1]),
            ],
            dtype=np.float32,
        )
        with self.lock:
            current = np.array([self.vx, self.vy, self.vyaw], dtype=np.float32)
            max_delta = self.command_slew_rate * max(float(dt), 0.0)
            next_value = current + np.clip(target - current, -max_delta, max_delta)
            self.vx, self.vy, self.vyaw = map(float, next_value)

    def _print_debug(self, raw_axes, mapped_axes, target_velocity):
        if not self.debug:
            return
        now = time.time()
        if now - self._last_debug_print < self.debug_interval:
            return
        self._last_debug_print = now
        vx, vy, vyaw = self.get_velocity()
        raw = np.array(raw_axes, dtype=np.float32)
        mapped = np.array(mapped_axes, dtype=np.float32)
        target = np.array(target_velocity, dtype=np.float32)
        print(
            "\n[GamepadDebug] raw_axes="
            f"{np.array2string(raw, precision=3, suppress_small=False)} "
            "mapped[left_x,left_y,right_x]="
            f"{np.array2string(mapped, precision=3, suppress_small=False)} "
            "target[vx,vy,yaw]="
            f"{np.array2string(target, precision=3, suppress_small=False)} "
            f"cmd[vx,vy,yaw]=[{vx:+.3f}, {vy:+.3f}, {vyaw:+.3f}]"
        )

    def start(self):
        self.thread = threading.Thread(target=self._control_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self._stop_backend()
        if self.thread:
            self.thread.join(timeout=1.0)

    def _control_thread(self):
        raise NotImplementedError

    def _stop_backend(self):
        raise NotImplementedError


# ============================================================================
#  Logitech F710 手柄控制器
# ============================================================================

class F710GamepadController(GamepadController):
    """Logitech F710 手柄 (Linux /dev/input/jsX 原生接口)"""

    # F710 轴映射 (X模式)
    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_RIGHT_X = 3
    AXIS_RIGHT_Y = 4
    AXIS_DPAD_X = 6
    AXIS_DPAD_Y = 7
    # X模式默认按键: BTN_A=0, BTN_RB=5, BTN_START=7
    # D模式默认按键: BTN_A=1, BTN_RB=7, BTN_START=9  (通过 YAML gamepad_btn_* 覆盖)
    BTN_A = 0
    BTN_RB = 5
    BTN_START = 7

    def __init__(self, vx_range=(-2.0, 4.0), vy_range=(-1.0, 1.0), vyaw_range=(-1.57, 1.57),
                 device_path='/dev/input/js0', btn_start=None, btn_rb=None, btn_a=None,
                 axis_left_x=None, axis_left_y=None, axis_right_x=None,
                 deadzone=0.05, command_slew_rate=(2.0, 4.0, 3.0), debug=False, debug_interval=0.5):
        super().__init__(
            vx_range,
            vy_range,
            vyaw_range,
            deadzone=deadzone,
            command_slew_rate=command_slew_rate,
            debug=debug,
            debug_interval=debug_interval,
        )
        if btn_start is not None: self.BTN_START = btn_start
        if btn_rb   is not None: self.BTN_RB    = btn_rb
        if btn_a    is not None: self.BTN_A     = btn_a
        try:
            self.reader = LinuxJoystickReader(device_path)
            self.reader.start()
            print(f"✅ F710 Gamepad initialized (BTN_A={self.BTN_A}, BTN_RB={self.BTN_RB}, BTN_START={self.BTN_START})")
        except Exception as e:
            print(f"❌ F710 init failed: {e}")
            self.reader = None

    def _control_thread(self):
        if self.reader is None:
            print("Gamepad not available, using zero velocity")
            return

        update_interval = 1.0 / 33.0
        last_update = time.time()

        while self.running:
            try:
                loop_start = time.time()

                with self.reader.lock:
                    raw_axes = list(self.reader.axes)
                    raw_buttons = list(self.reader.buttons)

                # 归一化摇杆 [-1, 1]
                left_x = apply_deadzone(raw_axes[self.AXIS_LEFT_X] / 32767.0, self.deadzone)
                left_y = apply_deadzone(raw_axes[self.AXIS_LEFT_Y] / 32767.0, self.deadzone)
                right_x = apply_deadzone(raw_axes[self.AXIS_RIGHT_X] / 32767.0, self.deadzone)

                now = time.time()
                dt = now - last_update
                last_update = now

                # D-pad
                dpad_y = raw_axes[self.AXIS_DPAD_Y] if len(raw_axes) > self.AXIS_DPAD_Y else 0
                dpad_up = (dpad_y < -16000)
                dpad_down = (dpad_y > 16000)

                if dpad_up and not self.dpad_last_state['up']:
                    self.dpad_vx = min(self.dpad_vx + self.vx_increment, self.vx_range[1])
                    print(f"\n[D-pad UP] speed step: {self.dpad_vx:.1f} m/s")
                if dpad_down and not self.dpad_last_state['down']:
                    self.dpad_vx = max(self.dpad_vx - self.vx_increment, 0.0)
                    print(f"\n[D-pad DOWN] speed step: {self.dpad_vx:.1f} m/s")

                self.dpad_last_state['up'] = dpad_up
                self.dpad_last_state['down'] = dpad_down

                # 摇杆映射到速度；摇杆回中时回到 D-pad 巡航速度。
                target_vx = self.dpad_vx
                if abs(left_y) > 0.1:
                    if left_y <= 0:
                        target_vx = (-left_y) * self.vx_range[1]
                    else:
                        target_vx = (-left_y) * abs(self.vx_range[0])

                target_vy = -left_x * self.vy_range[1]
                target_vyaw = -right_x * self.vyaw_range[1]

                self.set_velocity_smooth(target_vx, target_vy, target_vyaw, dt)
                self._print_debug(raw_axes, [left_x, left_y, right_x], [target_vx, target_vy, target_vyaw])

                # RB+A 进入 walk policy
                if (self.BTN_RB < len(raw_buttons) and raw_buttons[self.BTN_RB] and
                        self.BTN_A < len(raw_buttons) and raw_buttons[self.BTN_A]):
                    if not self.walk_requested:
                        print("\n✅ [RB+A] Walk policy activated!")
                        self.walk_requested = True

                # Start 按钮退出
                if self.BTN_START < len(raw_buttons) and raw_buttons[self.BTN_START]:
                    print("\n✅ Start button pressed - exiting")
                    self.exit_requested = True
                    break

                elapsed = time.time() - loop_start
                time.sleep(max(0, update_interval - elapsed))

            except Exception as e:
                print(f"\nGamepad error: {e}")
                time.sleep(0.1)

    def _stop_backend(self):
        if self.reader:
            self.reader.stop()


# ============================================================================
#  GameSir 盖世小鸡手柄控制器
# ============================================================================

class GameSirGamepadController(GamepadController):
    """
    GameSir 盖世小鸡手柄 (pygame 接口)

    轴映射:
      左摇杆: axes[0]=X(-1左,+1右), axes[1]=Y(-1上,+1下)
      Generic X-Box SDL: axes[2]=LT, axes[3]=右摇杆X, axes[4]=右摇杆Y, axes[5]=RT
    按钮映射:
      GameSir: A/B/X/Y=0/1/3/4, LB/RB=6/7, View/Menu=10/11
      D-pad 通过 hat 事件 -> 虚拟按钮 15(UP) 16(DOWN) 17(LEFT) 18(RIGHT)
    """

    def __init__(
        self,
        vx_range=(-2.0, 4.0),
        vy_range=(-1.0, 1.0),
        vyaw_range=(-1.57, 1.57),
        axis_left_x=None,
        axis_left_y=None,
        axis_right_x=None,
        device_path=None,
        btn_start=None,
        btn_rb=None,
        btn_a=None,
        policy_switch_mode='rb_combo',
        exit_button='start',
        deadzone=0.05,
        command_slew_rate=(2.0, 4.0, 3.0),
        debug=False,
        debug_interval=0.5,
    ):
        super().__init__(
            vx_range,
            vy_range,
            vyaw_range,
            deadzone=deadzone,
            command_slew_rate=command_slew_rate,
            debug=debug,
            debug_interval=debug_interval,
        )
        try:
            self.reader = PygameJoystickReader()
            self.reader.start()
            self.axis_left_x = PygameJoystickReader.AXIS_LEFT_X if axis_left_x is None else int(axis_left_x)
            self.axis_left_y = PygameJoystickReader.AXIS_LEFT_Y if axis_left_y is None else int(axis_left_y)
            default_right_x = PygameJoystickReader.AXIS_RIGHT_X if self.reader.num_axes > 3 else 2
            self.axis_right_x = default_right_x if axis_right_x is None else int(axis_right_x)
            self.btn_start = PygameJoystickReader.BTN_START if btn_start is None else int(btn_start)
            self.btn_rb = PygameJoystickReader.BTN_RB if btn_rb is None else int(btn_rb)
            self.btn_a = PygameJoystickReader.BTN_A if btn_a is None else int(btn_a)
            self.policy_switch_mode = policy_switch_mode
            self.exit_button = exit_button
            print("✅ GameSir Gamepad initialized")
            print(
                "   Mapping: "
                f"LX axis {self.axis_left_x}, LY axis {self.axis_left_y}, RX axis {self.axis_right_x}, "
                f"RB button {self.btn_rb}, A button {self.btn_a}, Start button {self.btn_start}, "
                f"policy_switch_mode={self.policy_switch_mode}, exit_button={self.exit_button}"
            )
        except Exception as e:
            print(f"❌ GameSir init failed: {e}")
            self.reader = None

    @staticmethod
    def _axis(raw_axes, axis_index):
        if axis_index is None or axis_index < 0 or axis_index >= len(raw_axes):
            return 0.0
        return raw_axes[axis_index]

    @staticmethod
    def _button(buttons, button_index):
        return 0 <= button_index < len(buttons) and bool(buttons[button_index])

    def _control_thread(self):
        if self.reader is None:
            print("Gamepad not available, using zero velocity")
            return

        update_interval = 1.0 / 33.0
        last_update = time.time()
        _prev_combos = {1: False, 2: False, 3: False, 4: False}

        while self.running:
            try:
                loop_start = time.time()

                with self.reader.lock:
                    # pygame axes 已经是 [-1, 1]
                    raw_axes = list(self.reader.axes)
                    left_x = self._axis(raw_axes, self.axis_left_x)
                    left_y = self._axis(raw_axes, self.axis_left_y)
                    right_x = self._axis(raw_axes, self.axis_right_x)
                    buttons = list(self.reader.buttons)

                now = time.time()
                dt = now - last_update
                last_update = now

                left_x = apply_deadzone(left_x, self.deadzone)
                left_y = apply_deadzone(left_y, self.deadzone)
                right_x = apply_deadzone(right_x, self.deadzone)

                # D-pad (虚拟按钮 15=UP, 16=DOWN)
                dpad_up = bool(buttons[15])
                dpad_down = bool(buttons[16])

                if dpad_up and not self.dpad_last_state['up']:
                    self.dpad_vx = min(self.dpad_vx + self.vx_increment, self.vx_range[1])
                    print(f"\n[D-pad UP] speed step: {self.dpad_vx:.1f} m/s")
                if dpad_down and not self.dpad_last_state['down']:
                    self.dpad_vx = max(self.dpad_vx - self.vx_increment, 0.0)
                    print(f"\n[D-pad DOWN] speed step: {self.dpad_vx:.1f} m/s")

                self.dpad_last_state['up'] = dpad_up
                self.dpad_last_state['down'] = dpad_down

                # 左摇杆: 前后左右速度；摇杆回中时回到 D-pad 巡航速度。
                # Y轴: 向上(-1) -> vx正(前进), 向下(+1) -> vx负(后退)
                target_vx = self.dpad_vx
                if abs(left_y) > 0.1:
                    if left_y <= 0:
                        target_vx = (-left_y) * self.vx_range[1]
                    else:
                        target_vx = (-left_y) * abs(self.vx_range[0])
                # X轴: 向左(-1) -> vy正(左移), 向右(+1) -> vy负(右移)
                target_vy = -left_x * self.vy_range[1]

                # 右摇杆: yaw 速度
                target_vyaw = -right_x * self.vyaw_range[1]

                self.set_velocity_smooth(target_vx, target_vy, target_vyaw, dt)
                self._print_debug(raw_axes, [left_x, left_y, right_x], [target_vx, target_vy, target_vyaw])

                # 策略切换 (上升沿触发)
                rb = self._button(buttons, self.btn_rb)
                if self.policy_switch_mode == 'face_buttons':
                    combos = {
                        1: self._button(buttons, self.btn_a),
                        2: bool(buttons[PygameJoystickReader.BTN_B]),
                        3: bool(buttons[PygameJoystickReader.BTN_X]),
                        4: bool(buttons[PygameJoystickReader.BTN_Y]),
                    }
                    combo_names = {1: 'A (Policy 1)', 2: 'B (Policy 2)',
                                   3: 'X (Policy 3)', 4: 'Y (Policy 4)'}
                else:
                    combos = {
                        1: rb and self._button(buttons, self.btn_a),
                        2: rb and bool(buttons[PygameJoystickReader.BTN_B]),
                        3: rb and bool(buttons[PygameJoystickReader.BTN_X]),
                        4: rb and bool(buttons[PygameJoystickReader.BTN_Y]),
                    }
                    combo_names = {1: 'RB+A (Walk)', 2: 'RB+B (Policy1)',
                                   3: 'RB+X (Policy2)', 4: 'RB+Y (Policy3)'}
                for idx, pressed in combos.items():
                    if pressed and not _prev_combos[idx]:
                        self.active_policy = idx
                        if idx == 1:
                            self.walk_requested = True
                        print(f"\n✅ [{combo_names[idx]}] activated!")
                _prev_combos = dict(combos)

                # 退出按钮
                if self.exit_button == 'select':
                    exit_pressed = self._button(buttons, PygameJoystickReader.BTN_SELECT)
                    exit_name = 'Select'
                else:
                    exit_pressed = self._button(buttons, self.btn_start)
                    exit_name = 'Start'
                if exit_pressed:
                    print(f"\n✅ {exit_name} button pressed - exiting")
                    self.exit_requested = True
                    break

                elapsed = time.time() - loop_start
                time.sleep(max(0, update_interval - elapsed))

            except Exception as e:
                print(f"\nGamepad error: {e}")
                time.sleep(0.1)

    def _stop_backend(self):
        if self.reader:
            self.reader.stop()


# ============================================================================
#  工厂函数: 根据类型创建手柄控制器
# ============================================================================

# 支持的手柄类型映射（当前仅保留 GameSir）
GAMEPAD_TYPES = {
    'gamesir': GameSirGamepadController,
    'gamesir710': GameSirGamepadController,
    'g710': GameSirGamepadController,
}


def create_gamepad_controller(gamepad_type, vx_range=(-2.0, 4.0), vy_range=(-1.0, 1.0),
                              vyaw_range=(-1.57, 1.57), device_path='/dev/input/js0',
                              btn_start=None, btn_rb=None, btn_a=None,
                              axis_left_x=None, axis_left_y=None, axis_right_x=None,
                              policy_switch_mode='rb_combo', exit_button='start',
                              deadzone=0.05, command_slew_rate=(2.0, 4.0, 3.0),
                              debug=False, debug_interval=0.5):
    """
    根据手柄类型创建对应的控制器（当前仅支持 GameSir）

    Args:
        gamepad_type: 手柄类型 ('gamesir', 'gamesir710', 'g710')
        vx_range: 前后速度范围 (m/s)
        vy_range: 左右速度范围 (m/s)
        vyaw_range: 旋转速度范围 (rad/s)
        device_path/btn_start/btn_rb/btn_a: 兼容参数，当前 GameSir 路径不使用

    Returns:
        GamepadController 子类实例
    """
    gamepad_type = gamepad_type.lower().replace('-', '').replace('_', '')
    if gamepad_type not in GAMEPAD_TYPES:
        raise ValueError(f"不支持的手柄类型: '{gamepad_type}', 可选: {list(GAMEPAD_TYPES.keys())}")

    cls = GAMEPAD_TYPES[gamepad_type]
    return cls(
        vx_range=vx_range,
        vy_range=vy_range,
        vyaw_range=vyaw_range,
        device_path=device_path,
        btn_start=btn_start,
        btn_rb=btn_rb,
        btn_a=btn_a,
        axis_left_x=axis_left_x,
        axis_left_y=axis_left_y,
        axis_right_x=axis_right_x,
        policy_switch_mode=policy_switch_mode,
        exit_button=exit_button,
        deadzone=deadzone,
        command_slew_rate=command_slew_rate,
        debug=debug,
        debug_interval=debug_interval,
    )


# ============================================================================
#  测试代码
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="手柄测试工具")
    parser.add_argument('--type', type=str, default='gamesir', choices=list(GAMEPAD_TYPES.keys()),
                        help='手柄类型')
    parser.add_argument('--device', type=str, default='/dev/input/js0',
                        help='兼容参数，当前 GameSir 路径不使用')
    args = parser.parse_args()

    print("=" * 70)
    print(f"手柄测试 - 类型: {args.type}")
    print("=" * 70)
    print("  左摇杆: 控制 vx (前后) / vy (左右)")
    print("  右摇杆: 控制 vyaw (转向)")
    print("  D-pad 上/下: 步进调速")
    print("  Start: 退出")
    print("=" * 70 + "\n")

    gamepad = create_gamepad_controller(args.type, vx_range=(0, 2.0), vy_range=(-0.5, 0.5),
                                        vyaw_range=(-1.5, 1.5), device_path=args.device)
    gamepad.start()

    try:
        while not gamepad.exit_requested:
            vx, vy, vyaw = gamepad.get_velocity()
            print(f"\rvx={vx:+.2f} m/s | vy={vy:+.2f} m/s | vyaw={vyaw:+.2f} rad/s   ", end='', flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\n✅ Ctrl+C")
    finally:
        gamepad.stop()
        print("\n测试结束")
