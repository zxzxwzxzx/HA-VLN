"""
demo_v5.py — 交互窗口 + ViCo 穿衣角色逐帧 LBS 动画
需先运行：python scripts/generate_vico_animated.py --scan <scan_id>

用法：
    conda activate havlnce
    python scripts/demo_v5.py --scan 1LXtFkjw3qL

控制：W=前进  A=左转  D=右转  Q=退出
"""

import habitat_sim
import magnum as mn
import numpy as np
import cv2
import json
import os
import argparse
import time

DATA_PATH           = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
SCENE_DATASETS_PATH = os.path.join(DATA_PATH, "scene_datasets/mp3d")
ANNOT_PATH          = os.path.join(DATA_PATH, "Multi-Human-Annotations/human_motion.json")
VICO_ANIM_DIR       = os.path.join(DATA_PATH, "ViCo_animated")

VICO_FEET_LOCAL_Y = 0.029   # lowest vertex in ViCo GLB (measured)


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


class ViCoAnimHumanManager:
    """逐帧替换 ViCo LBS-animated GLB（穿衣 + 真实姿态）"""

    def __init__(self, sim, human_data, scan_id):
        self.sim = sim
        self.current_frame = 0
        self.humans = {}   # pid → {template_ids, current_obj_id, translations, rotations}
        self._load(human_data, scan_id)

    def _load(self, human_data, scan_id):
        otm = self.sim.get_object_template_manager()
        rom = self.sim.get_rigid_object_manager()
        scan_data = human_data.get(scan_id, {})
        if not scan_data:
            print(f"Warning: no data for scan {scan_id}")
            return

        print(f"Loading ViCo animated humans for scan {scan_id}...")
        loaded = 0
        char_idx = 0
        for pid, v in scan_data.items():
            translations = v.get('translation', [])
            rotations    = v.get('rotation', [])
            if not translations or not rotations:
                continue

            out_key = f"{scan_id}__char{char_idx:02d}"
            char_idx += 1
            anim_dir = os.path.join(VICO_ANIM_DIR, out_key)
            if not os.path.isdir(anim_dir):
                print(f"  skip {out_key} (not generated yet)")
                continue

            # Pre-load template IDs for all frames
            template_ids = []
            for fi in range(120):
                cfg = os.path.join(anim_dir, f"frame{fi:03d}.object_config.json")
                if not os.path.exists(cfg):
                    break
                ids = otm.load_configs(cfg)
                if ids:
                    template_ids.append(ids[0])

            if not template_ids:
                continue

            t0 = translations[0]
            obj = rom.add_object_by_template_id(template_ids[0])
            if obj.object_id == -1:
                continue

            obj.translation = np.array([t0[0], _object_base_y(t0), t0[2]])

            self.humans[pid] = {
                'template_ids': template_ids,
                'current_obj_id': obj.object_id,
                'translations': translations,
                'rotations': rotations,
            }
            loaded += 1
            cat = v.get('category', '')[:45]
            print(f"  [{loaded:02d}] {out_key} → {cat}")

        print(f"Loaded {loaded} ViCo animated characters.")

    def advance_frame(self):
        self.current_frame = (self.current_frame + 1) % 120

    def update_humans(self):
        rom = self.sim.get_rigid_object_manager()
        fi  = self.current_frame
        for h in self.humans.values():
            # Remove old object
            try:
                rom.remove_object_by_id(h['current_obj_id'])
            except Exception:
                pass

            # Add new pose frame
            tids = h['template_ids']
            obj  = rom.add_object_by_template_id(tids[fi % len(tids)])
            h['current_obj_id'] = obj.object_id

            # Update translation from annotation
            ts = h['translations']
            rs = h['rotations']
            t  = ts[fi % len(ts)]
            r  = rs[fi % len(rs)]
            obj.translation = np.array([t[0], _object_base_y(t), t[2]])
            obj.rotation = (
                mn.Quaternion.rotation(mn.Deg(r[0]), mn.Vector3.x_axis()) *
                mn.Quaternion.rotation(mn.Deg(r[1]), mn.Vector3.y_axis()) *
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
    parser.add_argument("--scan",       type=str, default="1LXtFkjw3qL")
    parser.add_argument("--resolution", type=int, default=720)
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

    human_manager = ViCoAnimHumanManager(sim, all_human_data, args.scan)

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

    print("\nRunning... W=前进  A=左转  D=右转  Q=退出")

    try:
        while True:
            now = time.time()
            if now - last_anim_time >= FRAME_INTERVAL:
                human_manager.advance_frame()
                human_manager.update_humans()
                last_anim_time = now

            obs = sim.get_sensor_observations()
            rgb = obs.get("color")
            if rgb is None:
                continue
            bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
            cv2.imshow("HA-VLN ViCo Demo", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
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
