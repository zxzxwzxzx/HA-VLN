"""
generate_vico_animated.py
离线预生成 ViCo 穿衣角色逐帧 GLB（LBS 蒙皮 + motion.pkl 动作）
输出格式与 HAPS2_0 相同，可直接被 demo_v5.py 使用。

用法：
    conda activate havlnce
    python scripts/generate_vico_animated.py --scan 1LXtFkjw3qL
    python scripts/generate_vico_animated.py  # 全部扫描

输出：Data/ViCo_animated/<scan>__char<idx>/frame000.glb ... frame119.glb

材质改进（v2）：
- 多 primitive 导出，每个 primitive 保留独立纹理
- 衣服 primitive 带 normal map（布料褶皱）
- 正确 PBR 参数：衣服 metallic=0, roughness=0.9；皮肤 roughness=0.65 等
- 动态计算顶点法线和切线
"""

import os, json, pickle, struct, io, argparse
import numpy as np
from scipy.spatial.transform import Rotation
from collections import deque
import pygltflib
from PIL import Image

def _wxyz_to_R(q):
    """[w,x,y,z] quaternion → 3x3 rotation matrix."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

DATA_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
VICO_MODELS = "/datadrive/havln/ViCo_assets/avatars/models"
MOTION_PKL  = "/datadrive/havln/ViCo_assets/avatars/motions/motion.pkl"
ANNOT_PATH  = os.path.join(DATA_PATH, "Multi-Human-Annotations/human_motion.json")
OUT_ROOT    = os.path.join(DATA_PATH, "ViCo_animated")

N_FRAMES = 120
FPS      = 25

# motion.pkl → MAPPING: 65 motion joints → skin joint indices (0-based into skin.joints[])
MAPPING = [0,1,2,3,4,7,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
           30,31,32,33,34,35,36,37,38,39,42,43,44,45,46,47,48,49,50,51,52,53,
           54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72]

# category keyword → motion name
CAT_TO_MOTION = {
    "sliding": "sit", "sitting": "sit", "seated": "sit",
    "walking": "walk", "walk": "walk", "running": "Running", "run": "Running",
    "clean": "reach", "vacuum": "reach", "sweep": "reach", "mop": "reach",
    "photograph": "reach", "camera": "reach", "photo": "reach",
    "meal": "pick", "cook": "pick", "prep": "pick", "chop": "pick",
    "pick": "pick", "reach": "reach", "throw": "throw",
    "sit": "sit", "stand": "stand",
    "talk": "Talking1", "convers": "Talking1", "chat": "Talking1", "argu": "StandingArguing",
    "greet": "StandingGreeting", "wave": "StandingGreeting",
    "toast": "Cheering", "cheer": "Cheering", "drink": "drink",
    "play": "play", "read": "idle", "idle": "idle",
    "put": "pick", "place": "pick",
}

# PBR material presets per mesh name keyword
# metallic, roughness, normal_scale (0=no normal map), color_size, normal_size
MATERIAL_PRESETS = {
    "outfit_top":    dict(metallic=0.0, roughness=0.90, normal_scale=1.5, color_sz=256, norm_sz=512),
    "outfit_bottom": dict(metallic=0.0, roughness=0.90, normal_scale=1.5, color_sz=256, norm_sz=512),
    "outfit_shoes":  dict(metallic=0.0, roughness=0.65, normal_scale=1.2, color_sz=256, norm_sz=512),
    "AvatarBody":    dict(metallic=0.0, roughness=0.65, normal_scale=0.8, color_sz=256, norm_sz=256),
    "AvatarHead":    dict(metallic=0.0, roughness=0.60, normal_scale=0.8, color_sz=256, norm_sz=256),
    "AvatarLeftEyeball":  dict(metallic=0.0, roughness=0.15, normal_scale=0.0, color_sz=128, norm_sz=0),
    "AvatarRightEyeball": dict(metallic=0.0, roughness=0.15, normal_scale=0.0, color_sz=128, norm_sz=0),
    "AvatarTeeth":   dict(metallic=0.0, roughness=0.50, normal_scale=0.0, color_sz=128, norm_sz=0),
    "AvatarEyelashes": dict(metallic=0.0, roughness=0.90, normal_scale=0.0, color_sz=64,  norm_sz=0),
    "glasses":       dict(metallic=0.50, roughness=0.30, normal_scale=0.0, color_sz=128, norm_sz=0),
}
DEFAULT_PRESET = dict(metallic=0.0, roughness=0.80, normal_scale=0.0, color_sz=128, norm_sz=0)

def _get_preset(mat_name):
    name = mat_name or ""
    for key, preset in MATERIAL_PRESETS.items():
        if key.lower() in name.lower():
            return preset
    return DEFAULT_PRESET


def pick_motion(category, motion_data):
    cat_lower = category.lower()
    for kw, mot in CAT_TO_MOTION.items():
        if kw in cat_lower and mot in motion_data:
            return mot
    return "idle"


# ── GLB reading helpers ──────────────────────────────────────────────────────

def read_acc(gltf, binary, idx):
    a = gltf.accessors[idx]
    bv = gltf.bufferViews[a.bufferView]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    COMP = {5120:'i1',5121:'u1',5122:'i2',5123:'u2',5125:'u4',5126:'f4'}
    SIZE = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
    dt = COMP[a.componentType]; nc = SIZE[a.type]
    return np.frombuffer(binary[off:off+a.count*nc*np.dtype(dt).itemsize], dtype=dt).reshape(a.count, nc)

def read_indices(gltf, binary, prim):
    if prim.indices is None:
        return None
    a = gltf.accessors[prim.indices]
    bv = gltf.bufferViews[a.bufferView]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    COMP = {5121:'u1',5123:'u2',5125:'u4'}
    dt = COMP[a.componentType]
    return np.frombuffer(binary[off:off+a.count*np.dtype(dt).itemsize], dtype=dt).reshape(-1,3)

def _load_raw_image(gltf, binary, img_src_idx):
    """Return raw bytes of embedded image."""
    img = gltf.images[img_src_idx]
    bv = gltf.bufferViews[img.bufferView]
    return binary[bv.byteOffset:bv.byteOffset+bv.byteLength], img.mimeType

def _resize_image_bytes(raw_bytes, target_size, fmt="JPEG", quality=88):
    """Resize image bytes and return compressed bytes."""
    pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    pil = pil.resize((target_size, target_size), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "JPEG":
        pil.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ── Skeleton / FK ────────────────────────────────────────────────────────────

def build_skeleton(gltf):
    skin = gltf.skins[0]
    skin_joints = skin.joints
    node_to_sj  = {nj: i for i, nj in enumerate(skin_joints)}
    parent_node = {}
    for ni, nd in enumerate(gltf.nodes):
        for ci in (nd.children or []):
            parent_node[ci] = ni
    return skin_joints, node_to_sj, parent_node

def read_ibm(gltf, binary):
    skin = gltf.skins[0]
    a = gltf.accessors[skin.inverseBindMatrices]
    bv = gltf.bufferViews[a.bufferView]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    raw = np.frombuffer(binary[off:off+a.count*16*4], dtype=np.float32).reshape(a.count,4,4)
    return raw.transpose(0, 2, 1)

def fk_global_transforms(gltf, skin_joints, node_to_sj,
                          mot_rot, mot_joint, mot_trans, ref_mat, ref_mat_inv):
    all_q    = np.vstack([mot_rot[None], mot_joint])
    motion_R = np.stack([_wxyz_to_R(all_q[i]) for i in range(65)])
    local_R  = np.einsum('nij,njk,nkl->nil', ref_mat_inv, motion_R, ref_mat)

    sj_to_mi = {sj: mi for mi, sj in enumerate(MAPPING)}
    n_sj     = len(skin_joints)
    global_R = np.tile(np.eye(3), (n_sj, 1, 1))
    global_T = np.zeros((n_sj, 3))

    root_sj = 0
    root_ni  = skin_joints[root_sj]
    nd0      = gltf.nodes[root_ni]
    rest_R0  = Rotation.from_quat(nd0.rotation).as_matrix() if nd0.rotation else np.eye(3)
    global_R[root_sj] = rest_R0 @ local_R[0]
    global_T[root_sj] = mot_trans

    visited = set([root_ni])
    queue   = deque([root_ni])
    while queue:
        ni   = queue.popleft()
        sj_p = node_to_sj[ni]
        for ci in (gltf.nodes[ni].children or []):
            if ci not in node_to_sj or ci in visited:
                continue
            visited.add(ci)
            sj_c    = node_to_sj[ci]
            rest_t  = np.array(gltf.nodes[ci].translation or [0.0, 0.0, 0.0], dtype=np.float64)
            global_T[sj_c] = global_T[sj_p] + global_R[sj_p] @ rest_t
            nd_c    = gltf.nodes[ci]
            rest_R_c = Rotation.from_quat(nd_c.rotation).as_matrix() if nd_c.rotation else np.eye(3)
            mi      = sj_to_mi.get(sj_c)
            combined = rest_R_c @ local_R[mi] if mi is not None else rest_R_c
            global_R[sj_c] = global_R[sj_p] @ combined
            queue.append(ci)

    G = np.zeros((n_sj, 4, 4))
    G[:, :3, :3] = global_R
    G[:, :3,  3] = global_T
    G[:,  3,  3] = 1.0
    return G


# ── LBS ─────────────────────────────────────────────────────────────────────

def lbs(verts, joints_v, weights_v, global4x4, ibm):
    skin_M = global4x4 @ ibm
    N = len(verts)
    v_h = np.ones((N, 4), dtype=np.float64)
    v_h[:, :3] = verts
    out = np.zeros((N, 3), dtype=np.float64)
    for k in range(4):
        j  = joints_v[:, k].astype(int)
        w  = weights_v[:, k].astype(np.float64)
        M  = skin_M[j]
        vt = np.einsum('nij,nj->ni', M, v_h)[:, :3]
        out += w[:, None] * vt
    return out


# ── Normals & Tangents ────────────────────────────────────────────────────────

def compute_normals(verts, faces):
    """Smooth per-vertex normals from deformed mesh."""
    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    fn = np.cross(v1-v0, v2-v0).astype(np.float64)
    fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / (fn_len + 1e-10)
    vn = np.zeros((len(verts), 3), dtype=np.float64)
    np.add.at(vn, faces[:,0], fn)
    np.add.at(vn, faces[:,1], fn)
    np.add.at(vn, faces[:,2], fn)
    vn_len = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / (vn_len + 1e-10)).astype(np.float32)

def compute_tangents(verts, normals, uvs, faces):
    """Per-vertex tangents with handedness (w) for normal mapping."""
    T = np.zeros((len(verts), 3), dtype=np.float64)
    B = np.zeros((len(verts), 3), dtype=np.float64)

    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    uv0, uv1, uv2 = uvs[faces[:,0]], uvs[faces[:,1]], uvs[faces[:,2]]

    e1, e2   = (v1 - v0).astype(np.float64), (v2 - v0).astype(np.float64)
    du1, dv1 = uv1[:,0]-uv0[:,0], uv1[:,1]-uv0[:,1]
    du2, dv2 = uv2[:,0]-uv0[:,0], uv2[:,1]-uv0[:,1]

    denom = du1*dv2 - du2*dv1
    f = np.where(np.abs(denom) > 1e-10, 1.0/denom, 0.0)

    Tf = (dv2[:,None]*e1 - dv1[:,None]*e2) * f[:,None]
    Bf = (du1[:,None]*e2 - du2[:,None]*e1) * f[:,None]

    np.add.at(T, faces[:,0], Tf); np.add.at(T, faces[:,1], Tf); np.add.at(T, faces[:,2], Tf)
    np.add.at(B, faces[:,0], Bf); np.add.at(B, faces[:,1], Bf); np.add.at(B, faces[:,2], Bf)

    n = normals.astype(np.float64)
    t = T - (T * n).sum(1, keepdims=True) * n
    t_len = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / (t_len + 1e-10)

    cross_nt = np.cross(n, t)
    w = np.where((cross_nt * B).sum(1) > 0, 1.0, -1.0)

    return np.concatenate([t, w[:,None]], axis=1).astype(np.float32)


# ── Multi-primitive PBR GLB export ───────────────────────────────────────────

def _align4(b):
    """Pad bytes to 4-byte boundary."""
    r = len(b) % 4
    return b if r == 0 else b + b'\x00' * (4 - r)

def _bv_entry(offset, length, target=None):
    e = {"buffer": 0, "byteOffset": offset, "byteLength": length}
    if target is not None:
        e["target"] = target
    return e

def export_glb_pbr(primitives, out_path):
    """
    Export multi-primitive GLB with full PBR materials.

    primitives: list of dicts with keys:
      verts    (N,3) float32  — deformed vertex positions
      normals  (N,3) float32  — smooth vertex normals
      tangents (N,4) float32 or None  — tangents+handedness
      uvs      (N,2) float32  — UV texture coords
      faces    (F,3) uint32   — triangle indices
      color_jpg bytes or None — JPEG color texture
      normal_png bytes or None — PNG normal map (only used if tangents present)
      metallic  float
      roughness float
      normal_scale float      — normal map intensity (0 = disable)
    """
    bin_chunks  = []   # list of (bytes, alignment_required)
    bv_list     = []
    acc_list    = []
    mat_list    = []
    mesh_prims  = []
    img_list    = []
    tex_list    = []

    cur_offset = 0

    def add_chunk(data, align=4, target=None):
        nonlocal cur_offset
        pad = (align - cur_offset % align) % align
        if pad:
            bin_chunks.append(b'\x00' * pad)
            cur_offset += pad
        bv = _bv_entry(cur_offset, len(data), target)
        bv_list.append(bv)
        bin_chunks.append(data)
        cur_offset += len(data)
        return len(bv_list) - 1

    def add_image(img_bytes, mime):
        bv_idx = add_chunk(img_bytes)   # no special target for images
        img_list.append({"bufferView": bv_idx, "mimeType": mime})
        tex_list.append({"source": len(img_list) - 1})
        return len(tex_list) - 1        # texture index

    for prim in primitives:
        verts    = prim['verts'].astype(np.float32)
        normals  = prim['normals'].astype(np.float32)
        uvs      = prim['uvs'].astype(np.float32)
        faces    = prim['faces'].astype(np.uint32)
        tangents = prim.get('tangents')
        V, F     = len(verts), len(faces)

        attrs = {}

        # POSITION
        bv = add_chunk(verts.tobytes(), align=4, target=34962)
        acc_list.append({"bufferView": bv, "componentType": 5126, "count": V, "type": "VEC3",
                         "min": verts.min(0).tolist(), "max": verts.max(0).tolist()})
        attrs["POSITION"] = len(acc_list) - 1

        # NORMAL
        bv = add_chunk(normals.tobytes(), align=4, target=34962)
        acc_list.append({"bufferView": bv, "componentType": 5126, "count": V, "type": "VEC3"})
        attrs["NORMAL"] = len(acc_list) - 1

        # TANGENT (needed for normal maps)
        if tangents is not None:
            bv = add_chunk(tangents.astype(np.float32).tobytes(), align=4, target=34962)
            acc_list.append({"bufferView": bv, "componentType": 5126, "count": V, "type": "VEC4"})
            attrs["TANGENT"] = len(acc_list) - 1

        # TEXCOORD_0
        bv = add_chunk(uvs.tobytes(), align=4, target=34962)
        acc_list.append({"bufferView": bv, "componentType": 5126, "count": V, "type": "VEC2"})
        attrs["TEXCOORD_0"] = len(acc_list) - 1

        # Indices
        idx_flat = faces.flatten()
        bv = add_chunk(idx_flat.tobytes(), align=4, target=34963)
        acc_list.append({"bufferView": bv, "componentType": 5125, "count": F * 3, "type": "SCALAR"})
        indices_idx = len(acc_list) - 1

        # Material
        pbr = {"metallicFactor": float(prim['metallic']),
               "roughnessFactor": float(prim['roughness'])}

        if prim.get('color_jpg'):
            tex_idx = add_image(prim['color_jpg'], "image/jpeg")
            pbr["baseColorTexture"] = {"index": tex_idx}

        mat = {"pbrMetallicRoughness": pbr, "doubleSided": False}

        ns = prim.get('normal_scale', 0.0)
        if prim.get('normal_png') and tangents is not None and ns > 0:
            tex_idx = add_image(prim['normal_png'], "image/png")
            mat["normalTexture"] = {"index": tex_idx, "scale": float(ns)}

        mat_list.append(mat)
        mesh_prims.append({
            "attributes": attrs,
            "indices": indices_idx,
            "material": len(mat_list) - 1,
            "mode": 4
        })

    # Assemble JSON
    gltf_dict = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": mesh_prims}],
        "materials": mat_list,
        "accessors": acc_list,
        "bufferViews": bv_list,
        "buffers": [{"byteLength": cur_offset}],
    }
    if img_list:
        gltf_dict["images"]   = img_list
        gltf_dict["textures"] = tex_list

    json_bytes = json.dumps(gltf_dict, separators=(',', ':')).encode()
    json_bytes = _align4(json_bytes + b' ' * ((4 - len(json_bytes) % 4) % 4))

    bin_data = _align4(b''.join(bin_chunks))

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, total))
        f.write(struct.pack('<II',  len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack('<II',  len(bin_data),   0x004E4942))
        f.write(bin_data)


# ── Per-character primitive loading ─────────────────────────────────────────

def load_char_primitives(glb_path):
    """
    Load all skinned primitives from a ViCo character GLB.

    Returns list of dicts:
      verts   (N,3) float32    rest-pose vertices
      joints  (N,4) uint8      skin joint indices
      weights (N,4) float32    blend weights
      uvs     (N,2) float32    UV texture coords
      faces   (F,3) uint32     triangle indices (0-based per primitive)
      color_jpg  bytes or None  resized JPEG color texture
      normal_png bytes or None  resized PNG normal map
      metallic, roughness, normal_scale  from preset
      skin_joints, ibm          shared skeleton (same for all)
    """
    gltf   = pygltflib.GLTF2().load(glb_path)
    binary = gltf.binary_blob()
    skin_joints, node_to_sj, _ = build_skeleton(gltf)
    ibm = read_ibm(gltf, binary)

    primitives = []

    for ni, nd in enumerate(gltf.nodes):
        if nd.mesh is None:
            continue
        mesh = gltf.meshes[nd.mesh]
        for prim in mesh.primitives:
            attrs = prim.attributes
            if attrs.JOINTS_0 is None or attrs.WEIGHTS_0 is None:
                continue

            verts   = read_acc(gltf, binary, attrs.POSITION).astype(np.float32)
            joints  = read_acc(gltf, binary, attrs.JOINTS_0).astype(np.uint8)
            weights = read_acc(gltf, binary, attrs.WEIGHTS_0).astype(np.float32)
            uvs     = (read_acc(gltf, binary, attrs.TEXCOORD_0).astype(np.float32)
                       if attrs.TEXCOORD_0 is not None
                       else np.zeros((len(verts), 2), dtype=np.float32))
            faces   = read_indices(gltf, binary, prim)
            if faces is None:
                n = len(verts)
                faces = np.arange(n, dtype=np.uint32).reshape(-1, 3)
            faces = faces.astype(np.uint32)

            # Material name & preset
            mat_name = ""
            if prim.material is not None:
                m = gltf.materials[prim.material]
                mat_name = m.name or ""
            preset = _get_preset(mat_name)

            # Color texture → JPEG
            color_jpg = None
            normal_png = None
            if prim.material is not None:
                m = gltf.materials[prim.material]
                pbr = m.pbrMetallicRoughness
                # Color texture
                if pbr and pbr.baseColorTexture is not None:
                    src = gltf.textures[pbr.baseColorTexture.index].source
                    raw, _ = _load_raw_image(gltf, binary, src)
                    sz = preset['color_sz']
                    if sz > 0:
                        color_jpg = _resize_image_bytes(raw, sz, fmt="JPEG", quality=88)
                elif pbr and pbr.baseColorFactor:
                    f = pbr.baseColorFactor
                    rgb = tuple(int(c*255) for c in f[:3])
                    img = Image.new("RGB", (4, 4), rgb)
                    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=88)
                    color_jpg = buf.getvalue()

                # Normal map
                sz_n = preset.get('norm_sz', 0)
                if m.normalTexture is not None and sz_n > 0:
                    src = gltf.textures[m.normalTexture.index].source
                    raw, _ = _load_raw_image(gltf, binary, src)
                    normal_png = _resize_image_bytes(raw, sz_n, fmt="PNG")

            primitives.append(dict(
                verts=verts, joints=joints, weights=weights, uvs=uvs, faces=faces,
                color_jpg=color_jpg, normal_png=normal_png,
                metallic=preset['metallic'],
                roughness=preset['roughness'],
                normal_scale=preset.get('normal_scale', 0.0),
                mat_name=mat_name,
            ))

    return primitives, skin_joints, ibm


# ── Main generation ──────────────────────────────────────────────────────────

def generate_character(char_name, category, glb_path, motion_data, out_dir, n_frames=N_FRAMES):
    print(f"  {char_name[:50]} → {out_dir.split('/')[-1]}")

    primitives, skin_joints, ibm = load_char_primitives(glb_path)
    gltf_obj = pygltflib.GLTF2().load(glb_path)
    skin_joints_list, node_to_sj, _ = build_skeleton(gltf_obj)

    mot_name   = pick_motion(category, motion_data)
    mot        = motion_data[mot_name]
    mot_frames = len(mot['trans'])

    ref_mat     = motion_data['idle']['mat'][0].astype(np.float64)
    ref_mat_inv = np.linalg.inv(ref_mat)

    os.makedirs(out_dir, exist_ok=True)

    for fi in range(n_frames):
        mfi      = fi % mot_frames
        mot_rot  = mot['rot'][mfi].astype(np.float64)
        mot_jnt  = mot['joint'][mfi].astype(np.float64)
        mot_trans = mot['trans'][mfi].astype(np.float64)

        G = fk_global_transforms(gltf_obj, skin_joints_list, node_to_sj,
                                  mot_rot, mot_jnt, mot_trans, ref_mat, ref_mat_inv)

        deformed_prims = []
        for prim in primitives:
            dv = lbs(prim['verts'], prim['joints'], prim['weights'], G, ibm).astype(np.float32)
            dn = compute_normals(dv, prim['faces'])
            # Tangents only if normal map is present
            dt = None
            if prim['normal_png'] is not None and prim['normal_scale'] > 0:
                dt = compute_tangents(dv, dn, prim['uvs'], prim['faces'])
            deformed_prims.append(dict(
                verts=dv, normals=dn, tangents=dt,
                uvs=prim['uvs'], faces=prim['faces'],
                color_jpg=prim['color_jpg'],
                normal_png=prim['normal_png'],
                metallic=prim['metallic'],
                roughness=prim['roughness'],
                normal_scale=prim['normal_scale'],
            ))

        glb_out = os.path.join(out_dir, f"frame{fi:03d}.glb")
        export_glb_pbr(deformed_prims, glb_out)

    # Write object_config.json
    for fi in range(n_frames):
        cfg = {
            "mass": 60.0,
            "friction_coefficient": 0.0,
            "restitution_coefficient": 0.0,
            "is_collidable": False,
            "render_asset": f"frame{fi:03d}.glb",
            "collision_asset": f"frame{fi:03d}.glb",
        }
        with open(os.path.join(out_dir, f"frame{fi:03d}.object_config.json"), 'w') as f:
            json.dump(cfg, f, indent=2)

    print(f"    → {n_frames} frames, motion={mot_name}, primitives={len(primitives)}")


def get_vico_cfgs():
    return sorted([
        os.path.join(VICO_MODELS, f)
        for f in os.listdir(VICO_MODELS)
        if f.endswith('.glb') and f.startswith('custom_')
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan",  type=str, default=None)
    parser.add_argument("--frames",type=int, default=N_FRAMES)
    parser.add_argument("--force", action="store_true", help="重新生成已存在的序列")
    args = parser.parse_args()

    with open(ANNOT_PATH) as f:
        annotations = json.load(f)
    with open(MOTION_PKL, 'rb') as f:
        motion_data = pickle.load(f)

    vico_cfgs = get_vico_cfgs()
    scans = [args.scan] if args.scan else list(annotations.keys())

    todo = []
    for scan in scans:
        scan_data = annotations.get(scan, {})
        char_idx = 0
        for pid, v in scan_data.items():
            cat  = v.get('category', '')
            if not v.get('translation'):
                continue
            glb_path = vico_cfgs[char_idx % len(vico_cfgs)]
            char_idx += 1
            cname   = os.path.splitext(os.path.basename(glb_path))[0]
            out_key = f"{scan}__char{char_idx-1:02d}"
            out_dir = os.path.join(OUT_ROOT, out_key)
            last_frame = os.path.join(out_dir, f"frame{args.frames-1:03d}.glb")
            if not args.force and os.path.exists(last_frame):
                continue
            todo.append((cname, cat, glb_path, out_dir))

    print(f"Generating {len(todo)} ViCo animated sequences (PBR materials + normal maps)...")
    for cname, cat, glb_path, out_dir in todo:
        try:
            generate_character(cname, cat, glb_path, motion_data, out_dir, args.frames)
        except Exception as e:
            print(f"  ERROR {cname}: {e}")
            import traceback; traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()
