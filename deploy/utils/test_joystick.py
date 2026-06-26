"""
手柄按键检测工具 - 盖世小鸡 / GameSir 适用
运行方式：
    cd /home/wzh/legged_rl_lab
    python deploy/utils/test_joystick.py

按下任意按键/推动摇杆 → 实时打印索引和值
按 Ctrl+C 退出
"""

import pygame
import sys

def main():
    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()
    if count == 0:
        print("[ERROR] 未检测到手柄，请先连接手柄再运行！")
        sys.exit(1)

    js = pygame.joystick.Joystick(0)
    js.init()

    print(f"[OK] 检测到手柄: {js.get_name()}")
    print(f"     按钮数: {js.get_numbuttons()}")
    print(f"     轴数:   {js.get_numaxes()}")
    print(f"     Hat数:  {js.get_numhats()}")
    print("=" * 50)
    print("操作手柄，实时显示事件（Ctrl+C 退出）")
    print("=" * 50)

    AXIS_DEADZONE = 0.05  # 忽略小于此值的轴偏移（消除漂移噪声）

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"[BUTTON DOWN ] 按钮索引: {event.button}")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"[BUTTON UP   ] 按钮索引: {event.button}")
                elif event.type == pygame.JOYAXISMOTION:
                    if abs(event.value) > AXIS_DEADZONE:
                        print(f"[AXIS MOTION ] 轴索引: {event.axis}  值: {event.value:+.3f}")
                elif event.type == pygame.JOYHATMOTION:
                    print(f"[HAT MOTION  ] Hat索引: {event.hat}  值: {event.value}")
    except KeyboardInterrupt:
        print("\n退出检测。")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
