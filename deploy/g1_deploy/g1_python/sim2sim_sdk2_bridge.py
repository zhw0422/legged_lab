#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common.sdk2_mujoco_bridge import Sdk2MujocoBridge


def resolve_config(name: str) -> Path:
    path = Path(name)
    if path.exists():
        return path
    if path.parts and path.parts[0] == "config":
        return DEPLOY_DIR / path
    return DEPLOY_DIR / "config" / name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bridge Unitree SDK2 LowCmd/LowState topics to a G1 MuJoCo simulation."
    )
    parser.add_argument("--config", default="g1_walk.yaml", help="Config YAML path or name under deploy/g1_deploy/config/.")
    parser.add_argument("--net", default="lo", help="CycloneDDS network interface. Unitree mujoco uses lo for simulation.")
    parser.add_argument("--domain_id", type=int, default=1, help="DDS domain id. Unitree mujoco uses 1 for simulation.")
    parser.add_argument("--input", choices=["keyboard", "gamepad"], default="keyboard", help="Input source for simulated wireless_remote.")
    parser.add_argument("--joystick_type", choices=["xbox", "switch"], default="switch", help="Gamepad layout used when --input gamepad.")
    parser.add_argument("--lowcmd_topic", default=None, help="Override LowCmd DDS topic. Default comes from YAML.")
    parser.add_argument("--lowstate_topic", default=None, help="Override LowState DDS topic. Default comes from YAML.")
    parser.add_argument("--no_render", action="store_true", help="Run without MuJoCo viewer.")
    parser.add_argument("--elastic_band", action="store_true", help="Enable a virtual support band for humanoid startup.")
    parser.add_argument(
        "--no_clamp_ctrl",
        action="store_true",
        help="Disable actuator ctrlrange clipping. Useful only for debugging raw LowCmd behavior.",
    )
    parser.add_argument(
        "--debug_lowcmd",
        action="store_true",
        help="Print LowCmd q/kp/kd/raw torque ranges and clipping counts.",
    )
    parser.add_argument("--print_rate", type=float, default=1.0, help="Status print interval in seconds; 0 disables prints.")
    parser.add_argument("--print_scene", action="store_true", help="Print MuJoCo joints, actuators, and sensors.")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "glfw")

    bridge = Sdk2MujocoBridge(
        config_path=resolve_config(args.config),
        net=args.net,
        domain_id=args.domain_id,
        input_mode=args.input,
        joystick_type=args.joystick_type,
        lowcmd_topic=args.lowcmd_topic,
        lowstate_topic=args.lowstate_topic,
        render=not args.no_render,
        print_rate=args.print_rate,
        print_scene=args.print_scene,
        elastic_band=args.elastic_band,
        clamp_ctrl=not args.no_clamp_ctrl,
        debug_lowcmd=args.debug_lowcmd,
    )
    try:
        bridge.run()
    finally:
        bridge.remote.close()


if __name__ == "__main__":
    main()
