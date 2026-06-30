import argparse
import os
import xml.etree.ElementTree as xml_et

import cv2
import numpy as np

ROBOT = "g1"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "../.."))


def default_paths(robot):
    if robot == "g1":
        asset_dir = os.path.join(_DEPLOY_DIR, "g1_deploy/assets")
        return {
            "input_scene": os.path.join(asset_dir, "scene_29dof.xml"),
            "output_scene": os.path.join(asset_dir, "scene_terrain.xml"),
            "mesh_dir": os.path.join(asset_dir, "meshes"),
        }
    if robot == "go2":
        asset_dir = os.path.join(_DEPLOY_DIR, "go2_deploy/assets")
        input_scene = os.path.join(asset_dir, "scene.xml")
        if not os.path.exists(input_scene):
            input_scene = os.path.join(_SCRIPT_DIR, "scene.xml")
        return {
            "input_scene": input_scene,
            "output_scene": os.path.join(asset_dir, "scene_terrain.xml"),
            "mesh_dir": os.path.join(asset_dir, "meshes"),
        }
    raise ValueError(f"Unsupported robot '{robot}'.")


INPUT_SCENE_PATH = default_paths(ROBOT)["input_scene"]
OUTPUT_SCENE_PATH = default_paths(ROBOT)["output_scene"]


def euler_to_quat(roll, pitch, yaw):
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def euler_to_rot(roll, pitch, yaw):
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def rot2d(x, y, yaw):
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny


def rot3d(pos, euler):
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


def list_to_str(vec):
    return " ".join(str(s) for s in vec)


class TerrainGenerator:
    def __init__(self, robot=ROBOT, input_scene_path=None, output_scene_path=None) -> None:
        paths = default_paths(robot)
        self.robot = robot
        self.input_scene_path = input_scene_path or paths["input_scene"]
        self.output_scene_path = output_scene_path or paths["output_scene"]
        self.assets_mesh_dir = paths["mesh_dir"]
        self.scene = xml_et.parse(self.input_scene_path)
        self.root = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")

    def AddBox(self, position=[1.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0], size=[0.1, 0.1, 0.1], name=None):
        geo = xml_et.SubElement(self.worldbody, "geom")
        if name is not None:
            geo.attrib["name"] = name
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(0.5 * np.array(size))
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddGeometry(self, position=[1.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0], size=[0.1, 0.1], geo_type="box"):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(0.5 * np.array(size))
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def _local_to_world(self, init_pos, local_pos, yaw):
        x, y = rot2d(local_pos[0], local_pos[1], yaw)
        return [x + init_pos[0], y + init_pos[1], local_pos[2] + init_pos[2]]

    def _add_top_box(self, init_pos, yaw, x_start, x_size, y_center, y_size, top_height, thickness, name=None):
        center_local = [x_start + 0.5 * x_size, y_center, top_height - 0.5 * thickness]
        self.AddBox(
            position=self._local_to_world(init_pos, center_local, yaw),
            euler=[0.0, 0.0, yaw],
            size=[x_size, y_size, thickness],
            name=name,
        )
        return x_start + x_size

    def AddStairsWithDown(self, init_pos=[1.0, 0.0, 0.0], yaw=0.0, width=0.8, height=0.12, length=1.5, stair_nums=10, top_width=0.8):
        local_pos = [0.0, 0.0, -0.5 * height]
        for _ in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2] + init_pos[2]], [0.0, 0.0, yaw], [width, length, height])

        if top_width > 0:
            local_pos[0] += width / 2 + top_width / 2
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2] + init_pos[2]], [0.0, 0.0, yaw], [top_width, length, height])
            local_pos[0] += top_width / 2 + width / 2
        else:
            local_pos[0] += width

        for _ in range(stair_nums):
            local_pos[2] -= height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2] + init_pos[2]], [0.0, 0.0, yaw], [width, length, height])
            local_pos[0] += width

    def AddStairsUpWithPlatform(self, init_pos=[1.0, 0.0, 0.0], yaw=0.0, width=0.30, height=0.12, length=1.6, stair_nums=10, top_width=1.2):
        cursor = 0.0
        for i in range(stair_nums):
            top_height = (i + 1) * height
            cursor = self._add_top_box(init_pos, yaw, cursor, width, 0.0, length, top_height, height, name=f"ame_stair_{i:02d}")
        if top_width > 0.0:
            cursor = self._add_top_box(init_pos, yaw, cursor, top_width, 0.0, length, stair_nums * height, height, name="ame_stair_top")
        return {"next_x": cursor, "deck_height": stair_nums * height, "lane_width": length}

    def AddSuspendStairs(self, init_pos=[1.0, 0.0, 0.0], yaw=1.0, width=0.4, height=0.1, length=1.5, gap=0.1, stair_nums=10):
        local_pos = [0.0, 0.0, -0.5 * height]
        for _ in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            solid_height = max(abs(height - gap), 1e-4)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2] + init_pos[2]], [0.0, 0.0, yaw], [width, length, solid_height])

    def AddRoughGround(self, init_pos=[1.0, 0.0, 0.0], euler=[0.0, -0.0, 0.0], nums=[10, 10], box_size=[0.5, 0.5, 0.5], box_euler=[0.0, 0.0, 0.0], separation=[0.2, 0.2], box_size_rand=[0.05, 0.05, 0.05], box_euler_rand=[0.2, 0.2, 0.2], separation_rand=[0.05, 0.05]):
        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        new_separation = np.array(separation) + np.array(separation_rand) * np.random.uniform(-1.0, 1.0, 2)
        for _ in range(nums[0]):
            local_pos[0] += new_separation[0]
            local_pos[1] = 0.0
            for _ in range(nums[1]):
                new_box_size = np.array(box_size) + np.array(box_size_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_box_euler = np.array(box_euler) + np.array(box_euler_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_separation = np.array(separation) + np.array(separation_rand) * np.random.uniform(-1.0, 1.0, 2)
                local_pos[1] += new_separation[1]
                pos = rot3d(local_pos, euler) + np.array(init_pos)
                self.AddBox(pos, new_box_euler, new_box_size)

    def _add_label_platform(self, init_pos, yaw, start_x, label, deck_height, lane_width, thickness):
        return self._add_top_box(init_pos, yaw, start_x, 0.45, 0.0, lane_width, deck_height, thickness, name=f"ame_{label}_entry")

    def AddAMEAlternateColumnStakes(self, init_pos, yaw, start_x, deck_height, lane_width=1.6, segment_length=4.0, stake_side=0.20, stake_gap=0.30, column_gap=0.30, thickness=0.18):
        cursor = self._add_label_platform(init_pos, yaw, start_x, "A_alternate_stakes", deck_height, lane_width, thickness)
        step = stake_side + stake_gap
        lateral = 0.5 * (stake_side + column_gap)
        index = 0
        while cursor + stake_side <= start_x + segment_length:
            y_center = lateral if index % 2 == 0 else -lateral
            self._add_top_box(
                init_pos,
                yaw,
                cursor,
                stake_side,
                y_center,
                stake_side,
                deck_height,
                thickness,
                name=f"ame_A_alternate_stake_{index:02d}",
            )
            cursor += step
            index += 1
        return start_x + segment_length

    def AddAMEDoubleColumnStakes(self, init_pos, yaw, start_x, deck_height, lane_width=1.6, segment_length=4.0, stake_side=0.20, stake_gap=0.30, column_gap=0.30, thickness=0.18):
        cursor = self._add_label_platform(init_pos, yaw, start_x, "B_double_stakes", deck_height, lane_width, thickness)
        step = stake_side + stake_gap
        lateral = 0.5 * (stake_side + column_gap)
        index = 0
        while cursor + stake_side <= start_x + segment_length:
            for y_center in (-lateral, lateral):
                self._add_top_box(
                    init_pos,
                    yaw,
                    cursor,
                    stake_side,
                    y_center,
                    stake_side,
                    deck_height,
                    thickness,
                    name=f"ame_B_double_stake_{index:02d}_{int(y_center > 0)}",
                )
            cursor += step
            index += 1
        return start_x + segment_length

    def AddAMESteppingStones(self, init_pos, yaw, start_x, deck_height, lane_width=1.6, segment_length=4.0, stone_width=0.35, stone_gap=0.18, thickness=0.18):
        cursor = self._add_label_platform(init_pos, yaw, start_x, "C_stepping_stones", deck_height, lane_width, thickness)
        step = stone_width + stone_gap
        index = 0
        while cursor + stone_width <= start_x + segment_length:
            y_center = 0.18 if index % 2 == 0 else -0.18
            self._add_top_box(
                init_pos,
                yaw,
                cursor,
                stone_width,
                y_center,
                stone_width,
                deck_height,
                thickness,
                name=f"ame_C_stepping_stone_{index:02d}",
            )
            cursor += step
            index += 1
        return start_x + segment_length

    def AddAMEStoneBridge(self, init_pos, yaw, start_x, deck_height, lane_width=1.6, segment_length=4.0, stone_width=0.35, stone_length=0.75, stone_distance=0.22, thickness=0.18):
        cursor = self._add_label_platform(init_pos, yaw, start_x, "D_stone_bridge", deck_height, lane_width, thickness)
        step = stone_length + stone_distance
        index = 0
        while cursor + stone_length <= start_x + segment_length:
            self._add_top_box(
                init_pos,
                yaw,
                cursor,
                stone_length,
                0.0,
                stone_width,
                deck_height,
                thickness,
                name=f"ame_D_stone_bridge_{index:02d}",
            )
            cursor += step
            index += 1
        return start_x + segment_length

    def AddAMEConcentricGaps(self, init_pos, yaw, start_x, deck_height, lane_width=1.6, segment_length=4.0, ground_width=0.50, gap_width=0.50, thickness=0.18):
        cursor = self._add_label_platform(init_pos, yaw, start_x, "E_concentric_gaps", deck_height, lane_width, thickness)
        index = 0
        end_x = start_x + segment_length
        while cursor < end_x:
            block_len = min(ground_width, end_x - cursor)
            self._add_top_box(
                init_pos,
                yaw,
                cursor,
                block_len,
                0.0,
                lane_width,
                deck_height,
                thickness,
                name=f"ame_E_gap_ground_{index:02d}",
            )
            cursor += ground_width + gap_width
            index += 1
        return end_x

    def AddAMEStairsUpSegment(self, init_pos, yaw, start_x, deck_height, lane_width=1.6, stair_nums=6, step_width=0.30, step_height=0.16, thickness=0.18):
        cursor = self._add_label_platform(init_pos, yaw, start_x, "F_stairs_up", deck_height, lane_width, thickness)
        for index in range(stair_nums):
            top_height = deck_height + (index + 1) * step_height
            cursor = self._add_top_box(
                init_pos,
                yaw,
                cursor,
                step_width,
                0.0,
                lane_width,
                top_height,
                thickness,
                name=f"ame_F_stairs_up_{index:02d}",
            )
        return cursor, deck_height + stair_nums * step_height

    def AddAMEStairsDownSegment(self, init_pos, yaw, start_x, deck_height, lane_width=1.6, stair_nums=6, step_width=0.30, step_height=0.16, thickness=0.18):
        cursor = self._add_label_platform(init_pos, yaw, start_x, "G_stairs_down", deck_height, lane_width, thickness)
        for index in range(stair_nums):
            top_height = deck_height - (index + 1) * step_height
            cursor = self._add_top_box(
                init_pos,
                yaw,
                cursor,
                step_width,
                0.0,
                lane_width,
                top_height,
                thickness,
                name=f"ame_G_stairs_down_{index:02d}",
            )
        return cursor, deck_height - stair_nums * step_height

    def AddAMERadialPlankBridge(self, init_pos, yaw, start_x, deck_height, lane_width=1.6, segment_length=4.0, plank_width=0.19, thickness=0.18):
        cursor = self._add_label_platform(init_pos, yaw, start_x, "H_radial_plank_bridge", deck_height, lane_width, thickness)
        platform_len = 0.75
        cursor = self._add_top_box(init_pos, yaw, cursor, platform_len, 0.0, platform_len, deck_height, thickness, name="ame_H_center_platform")
        cursor = self._add_top_box(init_pos, yaw, cursor, segment_length - platform_len, 0.0, plank_width, deck_height, thickness, name="ame_H_forward_plank")
        return cursor

    def AddAMETerrainSequence(self, init_pos=[1.0, 0.0, 0.0], yaw=0.0, start_x=0.0, deck_height=1.0, lane_width=1.6, connector_length=0.8, thickness=0.18):
        """Sequential parkour course laid out along +x at constant deck height.

        Layout after the initial stair climb (easiest → hardest, with a single
        difficulty bump from F/G stairs):

            connector → E (gaps)           [easiest: solid blocks with gaps]
                      → B (double stakes)  [continuous columns, narrow gaps]
                      → D (stone bridge)   [single-plank style, long stones]
                      → C (stepping stones)
                      → A (alternate stakes)
                      → F (stairs up) → G (stairs down)
                      → H (radial plank bridge)
                      → final platform
        """
        cursor = start_x
        cursor = self._add_top_box(init_pos, yaw, cursor, connector_length, 0.0, lane_width, deck_height, thickness, name="ame_connector_after_stairs")
        cursor = self.AddAMEConcentricGaps(init_pos, yaw, cursor, deck_height, lane_width=lane_width, thickness=thickness)
        cursor = self._add_top_box(init_pos, yaw, cursor, connector_length, 0.0, lane_width, deck_height, thickness, name="ame_connector_E_B")
        cursor = self.AddAMEDoubleColumnStakes(init_pos, yaw, cursor, deck_height, lane_width=lane_width, thickness=thickness)
        cursor = self._add_top_box(init_pos, yaw, cursor, connector_length, 0.0, lane_width, deck_height, thickness, name="ame_connector_B_D")
        cursor = self.AddAMEStoneBridge(init_pos, yaw, cursor, deck_height, lane_width=lane_width, thickness=thickness)
        cursor = self._add_top_box(init_pos, yaw, cursor, connector_length, 0.0, lane_width, deck_height, thickness, name="ame_connector_D_C")
        cursor = self.AddAMESteppingStones(init_pos, yaw, cursor, deck_height, lane_width=lane_width, thickness=thickness)
        cursor = self._add_top_box(init_pos, yaw, cursor, connector_length, 0.0, lane_width, deck_height, thickness, name="ame_connector_C_A")
        cursor = self.AddAMEAlternateColumnStakes(init_pos, yaw, cursor, deck_height, lane_width=lane_width, thickness=thickness)
        cursor = self._add_top_box(init_pos, yaw, cursor, connector_length, 0.0, lane_width, deck_height, thickness, name="ame_connector_A_F")
        cursor, high_deck = self.AddAMEStairsUpSegment(init_pos, yaw, cursor, deck_height, lane_width=lane_width, thickness=thickness)
        cursor = self._add_top_box(init_pos, yaw, cursor, connector_length, 0.0, lane_width, high_deck, thickness, name="ame_connector_F_G")
        cursor, deck_after_down = self.AddAMEStairsDownSegment(init_pos, yaw, cursor, high_deck, lane_width=lane_width, thickness=thickness)
        if not np.isclose(deck_after_down, deck_height):
            raise ValueError(f"AME stairs down ended at {deck_after_down}, expected {deck_height}.")
        cursor = self._add_top_box(init_pos, yaw, cursor, connector_length, 0.0, lane_width, deck_height, thickness, name="ame_connector_G_H")
        cursor = self.AddAMERadialPlankBridge(init_pos, yaw, cursor, deck_height, lane_width=lane_width, thickness=thickness)
        cursor = self._add_top_box(init_pos, yaw, cursor, 1.2, 0.0, lane_width, deck_height, thickness, name="ame_final_platform")
        return {"next_x": cursor, "deck_height": deck_height, "lane_width": lane_width}

    def AddPerlinHeighField(self, position=[1.0, 0.0, 0.0], euler=[0.0, -0.0, 0.0], size=[1.0, 1.0], height_scale=0.2, negative_height=0.2, image_width=128, img_height=128, smooth=100.0, perlin_octaves=6, perlin_persistence=0.5, perlin_lacunarity=2.0, output_hfield_image="height_field.png"):
        try:
            noise_module = __import__("noise")
        except ImportError as exc:
            raise ImportError("AddPerlinHeighField requires the optional 'noise' package.") from exc

        terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
        for y in range(img_height):
            for x in range(image_width):
                noise_value = noise_module.pnoise2(
                    x / smooth,
                    y / smooth,
                    octaves=perlin_octaves,
                    persistence=perlin_persistence,
                    lacunarity=perlin_lacunarity,
                )
                terrain_image[y, x] = int((noise_value + 1) / 2 * 255)

        os.makedirs(self.assets_mesh_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.assets_mesh_dir, output_hfield_image), terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str([size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddHeighFieldFromImage(self, position=[1.0, 0.0, 0.0], euler=[0.0, -0.0, 0.0], size=[2.0, 1.6], height_scale=0.02, negative_height=0.1, input_img=None, output_hfield_image="height_field.png", image_scale=[1.0, 1.0], invert_gray=False):
        input_image = cv2.imread(input_img)
        if input_image is None:
            raise FileNotFoundError(f"Cannot read image file: {input_img}")

        width = int(input_image.shape[1] * image_scale[0])
        height = int(input_image.shape[0] * image_scale[1])
        resized_image = cv2.resize(input_image, (width, height), interpolation=cv2.INTER_AREA)
        terrain_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        if invert_gray:
            terrain_image = 255 - terrain_image

        os.makedirs(self.assets_mesh_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.assets_mesh_dir, output_hfield_image), terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "image_hfield"
        hfield.attrib["size"] = list_to_str([size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "image_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def Save(self):
        os.makedirs(os.path.dirname(self.output_scene_path), exist_ok=True)
        self.scene.write(self.output_scene_path)
        print(f"Saved terrain scene: {self.output_scene_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["g1", "go2"], default=ROBOT)
    parser.add_argument("--input_scene", type=str, default=None)
    parser.add_argument("--output_scene", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--side_demos", action="store_true", help="Also generate the older side demo terrains away from the main AME course.")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tg = TerrainGenerator(args.robot, input_scene_path=args.input_scene, output_scene_path=args.output_scene)

    stair = tg.AddStairsUpWithPlatform(init_pos=[1.5, 0.0, 0.0], yaw=0.0, width=0.30, height=0.16, length=1.6, stair_nums=10, top_width=1.2)
    tg.AddAMETerrainSequence(
        init_pos=[1.5, 0.0, 0.0],
        yaw=0.0,
        start_x=stair["next_x"],
        deck_height=stair["deck_height"],
        lane_width=stair["lane_width"],
    )

    if args.side_demos:
        tg.AddBox(position=[2.0, 2.0, 0.31], euler=[0.0, np.deg2rad(-10), 0.0], size=[3, 1.5, 0.1])
        tg.AddSuspendStairs(init_pos=[1.0, 6.0, 0.0], yaw=0.0)
        tg.AddRoughGround(init_pos=[-2.5, 5.0, 0.0], euler=[0, 0, 0.0], nums=[10, 8])
        tg.AddPerlinHeighField(position=[-1.5, 4.0, 0.0], size=[2.0, 1.5])

    tg.Save()


if __name__ == "__main__":
    main()
