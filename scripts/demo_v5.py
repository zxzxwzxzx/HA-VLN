"""
demo_v5.py — 交互窗口 + ViCo 穿衣角色逐帧 LBS 动画
需先运行：python scripts/generate_vico_animated.py --scan <scan_id>

用法：
    conda activate havlnce
    python scripts/demo_v5.py --scan 1LXtFkjw3qL

控制：W=前进  A=左转  D=右转  T=时间+1h  Q=退出
"""

import habitat_sim
import magnum as mn
import numpy as np
import cv2
import json
import os
import argparse
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feet_offset import get_feet_y

DATA_PATH           = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
SCENE_DATASETS_PATH = os.path.join(DATA_PATH, "scene_datasets/mp3d")
ANNOT_PATH          = os.path.join(DATA_PATH, "Multi-Human-Annotations/human_motion.json")
VICO_ANIM_DIR       = os.path.join(DATA_PATH, "ViCo_animated")

VICO_FEET_LOCAL_Y = 0.029   # lowest vertex in ViCo GLB (measured)

# MP3D .house category letter → schedule room name(s)
HOUSE_CAT_TO_ROOM = {
    'b': 'bedroom', 'k': 'kitchen', 'h': 'hallway',
    'l': 'living_room', 'f': 'living_room',
    'a': 'bathroom', 't': 'bathroom',
    'd': 'dining_room', 'p': 'porch_terrace_deck_driveway',
    'o': 'office', 's': 'stairs', 'e': 'entryway',
    'c': 'closet', 'v': 'living_room', 'r': 'living_room',
}

def load_room_positions(scan_id):
    """Parse .house file → {room_name: [np.array(x,y,z), ...]} centroids."""
    house_path = os.path.join(SCENE_DATASETS_PATH, scan_id, f"{scan_id}.house")
    rooms = {}
    if not os.path.exists(house_path):
        return rooms
    with open(house_path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != 'R':
                continue
            # R  room_id level ...  category  cx cy cz  ...
            cat = parts[5]
            cx, cy, cz = float(parts[6]), float(parts[7]), float(parts[8])
            room_name = HOUSE_CAT_TO_ROOM.get(cat)
            if room_name:
                rooms.setdefault(room_name, []).append(np.array([cx, cy, cz]))
    return rooms


def make_sim(scene_id, resolution=720):
    bc = habitat_sim.SimulatorConfiguration()
    bc.gpu_device_id = 0
    bc.scene_id = scene_id
    bc.enable_physics = False

    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "color"
    cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.resolution = [resolution, resolution]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [cam]
    agent_cfg.action_space = {
        "move_forward":  habitat_sim.agent.ActionSpec("move_forward",  habitat_sim.agent.ActuationSpec(0.25)),
        "move_backward": habitat_sim.agent.ActionSpec("move_backward", habitat_sim.agent.ActuationSpec(0.25)),
        "turn_left":     habitat_sim.agent.ActionSpec("turn_left",     habitat_sim.agent.ActuationSpec(10.0)),
        "turn_right":    habitat_sim.agent.ActionSpec("turn_right",    habitat_sim.agent.ActuationSpec(10.0)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(bc, [agent_cfg]))


def _object_base_y(t):
    """annotation Y = character ground level; shift so GLB feet align to it."""
    return float(t[1]) - VICO_FEET_LOCAL_Y


def _parse_hour(time_str):
    """'07:30' → 7.5"""
    h, m = map(int, time_str.split(':'))
    return h + m / 60.0

def _slot_active(slot, sim_hour):
    """判断当前模拟时间是否在该时间段内（支持跨零点）。"""
    s = _parse_hour(slot['time_start'])
    e = _parse_hour(slot['time_end'])
    if s < e:
        return s <= sim_hour < e
    else:                       # 跨零点，如 22:00–06:00
        return sim_hour >= s or sim_hour < e


class ViCoAnimHumanManager:
    """逐帧替换 ViCo LBS-animated GLB，支持按模拟时间切换动作 clip。"""

    def __init__(self, sim, human_data, scan_id, sim_hour=12.0):
        self.sim = sim
        self.current_frame = 0
        self.sim_hour = sim_hour   # 当前模拟时刻（0–24）
        self.humans = {}
        self.room_positions = load_room_positions(scan_id)
        self._load(human_data, scan_id)

    def _load_tids(self, otm, directory, n=120):
        """从目录加载 frame000–119 的 template id 列表。"""
        tids = []
        for fi in range(n):
            cfg = os.path.join(directory, f"frame{fi:03d}.object_config.json")
            if not os.path.exists(cfg):
                break
            ids = otm.load_configs(cfg)
            if ids:
                tids.append(ids[0])
        return tids

    def _load(self, human_data, scan_id):
        otm = self.sim.get_object_template_manager()
        rom = self.sim.get_rigid_object_manager()
        scan_data = human_data.get(scan_id, {})
        if not scan_data:
            print(f"Warning: no data for scan {scan_id}")
            return

        # 加载路线 manifest（若有）
        manifest_path = os.path.join(VICO_ANIM_DIR, f"{scan_id}_route_manifest.json")
        self.manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            print(f"  Route manifest loaded: {len(self.manifest)} chars")
        else:
            print("  No route manifest found, using default clip only.")

        print(f"Loading ViCo animated humans for scan {scan_id}...")
        loaded = 0
        char_idx = 0
        for pid, v in scan_data.items():
            translations = v.get('translation', [])
            rotations    = v.get('rotation', [])
            if not translations or not rotations:
                continue

            char_key = f"{scan_id}__char{char_idx:02d}"
            char_idx += 1
            anim_dir = os.path.join(VICO_ANIM_DIR, char_key)
            if not os.path.isdir(anim_dir):
                print(f"  skip {char_key} (not generated)")
                continue

            # 默认帧（当前活动）
            default_tids = self._load_tids(otm, anim_dir)
            if not default_tids:
                continue

            # 各 clip 帧（来自 clips/ 子目录）
            clip_tids = {}   # clip_name → [template_id, ...]
            clips_dir = os.path.join(anim_dir, 'clips')
            if os.path.isdir(clips_dir):
                for clip_name in sorted(os.listdir(clips_dir)):
                    clip_path = os.path.join(clips_dir, clip_name)
                    if os.path.isdir(clip_path):
                        tids = self._load_tids(otm, clip_path)
                        if tids:
                            clip_tids[clip_name] = tids

            # 从预计算 JSON 读脚部偏移；GLB z-range 判断移动角色
            feet_local_y = get_feet_y(char_key)
            is_moving = False
            try:
                import pygltflib
                zvals = []
                for fi in [0, 20, 40, 60, 80, 100, 119]:
                    p = os.path.join(anim_dir, f"frame{fi:03d}.glb")
                    g = pygltflib.GLTF2().load(p)
                    for mesh in g.meshes:
                        for prim in mesh.primitives:
                            if prim.attributes.POSITION is not None:
                                acc = g.accessors[prim.attributes.POSITION]
                                zvals.append((acc.min[2]+acc.max[2])/2)
                                break
                        break
                if len(zvals) >= 3 and max(zvals)-min(zvals) >= 0.5:
                    is_moving = True
            except Exception:
                pass

            t0  = translations[0]
            obj = rom.add_object_by_template_id(default_tids[0])
            if obj.object_id == -1:
                continue
            obj.translation = np.array([t0[0], _object_base_y(t0), t0[2]])

            self.humans[pid] = {
                'char_key':       char_key,
                'default_tids':   default_tids,
                'clip_tids':      clip_tids,
                'is_moving':      is_moving,
                'feet_local_y':   feet_local_y,
                'current_obj_id': obj.object_id,
                'translations':   translations,
                'rotations':      rotations,
            }
            loaded += 1
            clips_info = f"  clips:{list(clip_tids.keys())}" if clip_tids else ""
            cat = v.get('category', '')[:40]
            print(f"  [{loaded:02d}] {char_key}{clips_info}  {cat}")

        print(f"Loaded {loaded} chars (sim_hour={self.sim_hour:.1f}h).")

    def _active_tids(self, h):
        """根据当前模拟时间选择对应的 template_ids。移动角色始终用 base。"""
        if h.get('is_moving'):
            return h['default_tids'], 'base'
        char_key = h['char_key']
        if char_key in self.manifest and h['clip_tids']:
            for slot in self.manifest[char_key].get('schedule', []):
                if _slot_active(slot, self.sim_hour):
                    clip = slot['motion_clip']
                    if clip == 'base':
                        return h['default_tids'], 'base'
                    if clip in h['clip_tids']:
                        return h['clip_tids'][clip], clip
        return h['default_tids'], 'default'

    def advance_sim_hour(self, delta=1.0):
        self.sim_hour = (self.sim_hour + delta) % 24
        print(f"  [Time] {self.sim_hour:05.2f}h", end='')
        for h in self.humans.values():
            _, clip = self._active_tids(h)
            char = h['char_key'].split('__')[1]
            print(f"  {char}:{clip}", end='')
        print()

    def advance_frame(self):
        self.current_frame = (self.current_frame + 1) % 120

    def update_humans(self):
        rom = self.sim.get_rigid_object_manager()
        fi  = self.current_frame
        for h in self.humans.values():
            try:
                rom.remove_object_by_id(h['current_obj_id'])
            except Exception:
                pass

            tids, _ = self._active_tids(h)
            obj = rom.add_object_by_template_id(tids[fi % len(tids)])
            h['current_obj_id'] = obj.object_id

            ts = h['translations'];  rs = h['rotations']
            t  = ts[fi % len(ts)];   r  = rs[fi % len(rs)]
            feet_y = h.get('feet_local_y', VICO_FEET_LOCAL_Y)
            obj.translation = np.array([t[0], float(t[1]) - feet_y, t[2]])  # per-char offset from feet_offset.json
            # 旋转顺序：先 Ry（朝向），再 Rx（俯仰），再 Rz（滚转）
            obj.rotation = (
                mn.Quaternion.rotation(mn.Deg(r[1]), mn.Vector3.y_axis()) *
                mn.Quaternion.rotation(mn.Deg(r[0]), mn.Vector3.x_axis()) *
                mn.Quaternion.rotation(mn.Deg(r[2]), mn.Vector3.z_axis())
            )

    def cleanup(self):
        rom = self.sim.get_rigid_object_manager()
        for h in self.humans.values():
            try:
                rom.remove_object_by_id(h['current_obj_id'])
            except Exception:
                pass
        self.humans.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan",       type=str,   default="1LXtFkjw3qL")
    parser.add_argument("--resolution", type=int,   default=720)
    parser.add_argument("--time_scale", type=float, default=60.0,
                        help="Sim-seconds per real-second (default 60 = 1 real-sec → 1 sim-min)")
    parser.add_argument("--start_hour", type=float, default=12.0,
                        help="Starting sim hour (0-24, default 12)")
    args = parser.parse_args()

    scene_filepath = os.path.join(SCENE_DATASETS_PATH, args.scan, f"{args.scan}.glb")
    print(f"Loading scene: {scene_filepath}")
    sim = make_sim(scene_filepath, args.resolution)

    navmesh_path = os.path.join(SCENE_DATASETS_PATH, args.scan, f"{args.scan}.navmesh")
    if os.path.exists(navmesh_path):
        sim.pathfinder.load_nav_mesh(navmesh_path)
        start_pos = sim.pathfinder.get_random_navigable_point()
    else:
        start_pos = np.array([0.0, 0.0, 0.0])

    st = sim.get_agent(0).get_state()
    st.position = np.round(start_pos, 2)
    sim.get_agent(0).set_state(st)
    print(f"Agent start: {np.round(start_pos, 2)}")

    with open(ANNOT_PATH) as f:
        all_human_data = json.load(f)

    human_manager = ViCoAnimHumanManager(sim, all_human_data, args.scan,
                                         sim_hour=args.start_hour)

    cv2.namedWindow("HA-VLN ViCo Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HA-VLN ViCo Demo", args.resolution, args.resolution)

    KEY_ACTIONS = {
        ord('w'): "move_forward",  ord('W'): "move_forward",
        ord('s'): "move_backward", ord('S'): "move_backward",
        ord('a'): "turn_left",     ord('A'): "turn_left",
        ord('d'): "turn_right",    ord('D'): "turn_right",
    }

    FRAME_INTERVAL = 1.0 / 25.0
    last_anim_time = time.time()
    last_render_time = time.time()

    print("\nRunning... W=前进  A=左转  D=右转  T=时间+1h  Q=退出")

    try:
        while True:
            now = time.time()

            # ── advance animation frame at 25 fps ─────────────────────────────
            if now - last_anim_time >= FRAME_INTERVAL:
                human_manager.advance_frame()
                human_manager.update_humans()
                last_anim_time = now

            # ── render at 25 fps, sleep the rest ──────────────────────────────
            if now - last_render_time < FRAME_INTERVAL:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                if key in (ord('t'), ord('T')):
                    human_manager.advance_sim_hour(1.0)
                action = KEY_ACTIONS.get(key)
                if action:
                    sim.step(action)
                continue
            last_render_time = now

            obs = sim.get_sensor_observations()
            rgb = obs.get("color")
            if rgb is None:
                continue
            bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)

            # overlay sim time
            h_int = int(human_manager.sim_hour)
            m_int = int((human_manager.sim_hour - h_int) * 60)
            time_str = f"Time: {h_int:02d}:{m_int:02d}"
            cv2.putText(bgr, time_str, (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, time_str, (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("HA-VLN ViCo Demo", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('t'), ord('T')):
                human_manager.advance_sim_hour(1.0)
                continue
            action = KEY_ACTIONS.get(key)
            if action:
                sim.step(action)

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cv2.destroyAllWindows()
        human_manager.cleanup()
        sim.close()
        print("Done.")


if __name__ == "__main__":
    main()
