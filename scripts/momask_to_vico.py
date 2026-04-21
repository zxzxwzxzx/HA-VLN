"""
momask_to_vico.py — 用 MoMask 生成动作并驱动 ViCo 角色

流程:
  文字描述 → MoMask → (T,263) HumanML3D features
           → recover_from_ric → (T,22,3) 关节位置
           → AvatarProfile 骨骼缩放 → scaled joint positions
           → IK on HumanML3D skeleton → (T,22) 关节四元数
           → 映射到 ViCo 65 关节格式
           → LBS → post-LBS 体型缩放 → GLB

用法:
    conda activate havlnce
    cd /datadrive/havln/HA-VLN
    # 单帧体型测试:
    python scripts/momask_to_vico.py --test_frame --height 1.1 --width 1.05
    # 生成单条文字描述的动作:
    python scripts/momask_to_vico.py --text "A person vacuuming the floor" --name "vacuum_momask"
    # 批量为 scan 里所有角色生成:
    python scripts/momask_to_vico.py --scan 1LXtFkjw3qL
"""

import sys, os, json, pickle, struct, argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation

MOMASK_DIR = "/datadrive/havln/momask-codes"
DATA_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
ANNOT_PATH = os.path.join(DATA_PATH, "Multi-Human-Annotations/human_motion.json")
MOTION_PKL = "/datadrive/havln/ViCo_assets/avatars/motions/motion.pkl"
VICO_MODELS = "/datadrive/havln/ViCo_assets/avatars/models"
OUT_ROOT   = os.path.join(DATA_PATH, "ViCo_animated")

sys.path.insert(0, MOMASK_DIR)

# ── HumanML3D 关节索引 → ViCo skin joint 索引 (来自 MAPPING 分析) ─────────────
# HumanML3D 22 关节 → ViCo 73 关节中对应的 skin joint index
HML3D_TO_VICO_SJ = {
    0:  0,   # root/pelvis → Hips
    1:  63,  # l_hip       → LeftUpLeg
    2:  68,  # r_hip       → RightUpLeg
    3:  1,   # spine1      → Spine
    4:  64,  # l_knee      → LeftLeg
    5:  69,  # r_knee      → RightLeg
    6:  2,   # spine2      → Spine1
    7:  65,  # l_ankle     → LeftFoot
    8:  70,  # r_ankle     → RightFoot
    9:  3,   # spine3      → Spine2
    10: 66,  # l_foot      → LeftToeBase
    11: 71,  # r_foot      → RightToeBase
    12: 4,   # neck        → Neck
    13: 11,  # l_collar    → LeftShoulder
    14: 37,  # r_collar    → RightShoulder
    15: 7,   # head        → Head
    16: 12,  # l_shoulder  → LeftArm
    17: 38,  # r_shoulder  → RightArm
    18: 13,  # l_elbow     → LeftForeArm
    19: 39,  # r_elbow     → RightForeArm
    20: 16,  # l_wrist     → LeftHand
    21: 42,  # r_wrist     → RightHand
}

# ViCo MAPPING: motion_joint_idx → skin_joint_idx
MAPPING = [0,1,2,3,4,7,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
           30,31,32,33,34,35,36,37,38,39,42,43,44,45,46,47,48,49,50,51,52,53,
           54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72]
# Reverse: skin_joint_idx → motion_joint_idx
VICO_SJ_TO_MI = {sj: mi for mi, sj in enumerate(MAPPING)}

# HML3D joint → motion_joint_idx in motion.pkl (via HML3D_TO_VICO_SJ → VICO_SJ_TO_MI)
HML3D_TO_MI = {h: VICO_SJ_TO_MI[v] for h, v in HML3D_TO_VICO_SJ.items()
               if v in VICO_SJ_TO_MI}


# ── MoMask model loading ─────────────────────────────────────────────────────

CKPT_DIR  = os.path.join(MOMASK_DIR, 'checkpoints')
T2M_NAME  = 't2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns'
RES_NAME  = 'tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw'
CLIP_VER  = 'ViT-B/32'

def load_momask(device='cuda'):
    """Load MoMask text-to-motion model (mirrors gen_t2m.py logic)."""
    from utils.get_opt import get_opt
    from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
    from models.vq.model import RVQVAE, LengthEstimator
    from os.path import join as pjoin

    model_opt_path = pjoin(CKPT_DIR, 't2m', T2M_NAME, 'opt.txt')
    if not os.path.exists(model_opt_path):
        raise FileNotFoundError(
            f"MoMask checkpoint not found:\n  {model_opt_path}\n"
            "Run: cd /datadrive/havln/momask-codes && bash prepare/download_models.sh"
        )

    model_opt = get_opt(model_opt_path, device=device)

    # VQ-VAE
    vq_opt_path = pjoin(CKPT_DIR, 't2m', model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=device)
    vq_opt.dim_pose = 263  # HumanML3D

    vq_model = RVQVAE(vq_opt, vq_opt.dim_pose, vq_opt.nb_code, vq_opt.code_dim,
                      vq_opt.output_emb_width, vq_opt.down_t, vq_opt.stride_t,
                      vq_opt.width, vq_opt.depth, vq_opt.dilation_growth_rate,
                      vq_opt.vq_act, vq_opt.vq_norm)
    ckpt = torch.load(pjoin(CKPT_DIR, 't2m', vq_opt.name, 'model', 'net_best_fid.tar'),
                      map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    vq_model.eval().to(device)
    print(f"Loaded VQ-VAE: {vq_opt.name}")

    model_opt.num_tokens    = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim      = vq_opt.code_dim

    # Mask Transformer
    t2m_model = MaskTransformer(
        code_dim=model_opt.code_dim, cond_mode='text',
        latent_dim=model_opt.latent_dim, ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers, num_heads=model_opt.n_heads,
        dropout=model_opt.dropout, clip_dim=512,
        cond_drop_prob=model_opt.cond_drop_prob,
        clip_version=CLIP_VER, opt=model_opt)
    ckpt = torch.load(pjoin(CKPT_DIR, 't2m', T2M_NAME, 'model', 'latest.tar'),
                      map_location='cpu')
    t2m_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing, unexpected = t2m_model.load_state_dict(ckpt[t2m_key], strict=False)
    assert len(unexpected) == 0
    assert all(k.startswith('clip_model.') for k in missing)
    t2m_model.eval().to(device)
    print(f"Loaded MaskTransformer epoch {ckpt.get('ep', '?')}")

    # Residual Transformer
    res_opt_path = pjoin(CKPT_DIR, 't2m', RES_NAME, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=device)
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens     = vq_opt.nb_code

    res_model = ResidualTransformer(
        code_dim=vq_opt.code_dim, cond_mode='text',
        latent_dim=res_opt.latent_dim, ff_size=res_opt.ff_size,
        num_layers=res_opt.n_layers, num_heads=res_opt.n_heads,
        dropout=res_opt.dropout, clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=res_opt.cond_drop_prob,
        share_weight=res_opt.share_weight,
        clip_version=CLIP_VER, opt=res_opt)
    ckpt = torch.load(pjoin(CKPT_DIR, 't2m', RES_NAME, 'model', 'net_best_fid.tar'),
                      map_location=device)
    missing, unexpected = res_model.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected) == 0
    assert all(k.startswith('clip_model.') for k in missing)
    res_model.eval().to(device)
    print(f"Loaded ResidualTransformer epoch {ckpt.get('ep', '?')}")

    # Length estimator
    len_model = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(CKPT_DIR, 't2m', 'length_estimator', 'model', 'finest.tar'),
                      map_location=device)
    len_model.load_state_dict(ckpt['estimator'])
    len_model.eval().to(device)
    print(f"Loaded LengthEstimator epoch {ckpt.get('epoch', '?')}")

    # Mean/Std for denormalization
    mean = np.load(pjoin(CKPT_DIR, 't2m', vq_opt.name, 'meta', 'mean.npy'))
    std  = np.load(pjoin(CKPT_DIR, 't2m', vq_opt.name, 'meta', 'std.npy'))

    return vq_model, t2m_model, res_model, len_model, model_opt, mean, std


def generate_motion(text, vq_model, t2m_model, res_model, len_model,
                    model_opt, mean, std, n_frames=None, device='cuda'):
    """
    Generate motion from text description.
    Returns (T, 22, 3) numpy array of joint positions.
    n_frames: if None, use length estimator; otherwise fix to this length.
    """
    import torch.nn.functional as F
    from torch.distributions.categorical import Categorical
    from utils.motion_process import recover_from_ric

    with torch.no_grad():
        captions = [text]

        if n_frames is None:
            text_emb = t2m_model.encode_text(captions)
            pred_dis  = len_model(text_emb)
            probs     = F.softmax(pred_dis, dim=-1)
            token_len = Categorical(probs).sample()   # (1,)
        else:
            token_len = torch.LongTensor([n_frames // 4]).to(device)

        m_length = token_len * 4

        mids = t2m_model.generate(captions, token_len,
                                   timesteps=18, cond_scale=4,
                                   temperature=1.0, topk_filter_thres=0.9,
                                   gsample=True)
        mids = res_model.generate(mids, captions, token_len,
                                  temperature=1, cond_scale=5)
        pred_motions = vq_model.forward_decoder(mids)   # (1, T, 263)

        pred_motions = pred_motions.detach().cpu().numpy()
        data = pred_motions * std + mean                  # denormalize

        T = int(m_length[0].item())
        joint_data = data[0, :T]                          # (T, 263)
        joint_pos  = recover_from_ric(
            torch.from_numpy(joint_data).float(), 22).numpy()  # (T, 22, 3)

    return joint_pos, joint_data   # also return raw 263-dim features


# ── AvatarProfile: 体型配置 → 骨骼 Translation 缩放比例 ──────────────────────

class AvatarProfile:
    """
    输入体型参数 {"height": 1.1, "width": 1.05}，
    输出 HumanML3D 每条骨骼的 Translation 缩放比例，
    并提供 post-LBS 顶点缩放比例。

    骨骼缩放规则（h_w + w_w 权重各自对应 height/width 的贡献）：
      纵向骨骼（脊柱、大腿、小腿）：h_w ≈ 0.9，w_w ≈ 0.1
      横向骨骼（胯宽/肩宽偏移）  ：h_w ≈ 0.1，w_w ≈ 0.9
      手臂                        ：h_w ≈ 0.5，w_w ≈ 0.5

    scale = h_w * height + w_w * width
    当 height=1.0, width=1.0 时所有 scale=1.0。
    """

    _HML_NAMES = [
        'Hips','LHip','RHip','Spine1','LKnee','RKnee','Spine2',
        'LAnkle','RAnkle','Spine3','LFoot','RFoot','Neck',
        'LCollar','RCollar','Head','LShoulder','RShoulder',
        'LElbow','RElbow','LWrist','RWrist',
    ]
    HML_PARENT = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]

    # (parent_hml, child_hml) → (height_weight, width_weight), h_w + w_w = 1
    _BONE_HW: dict = {
        # 脊柱（纵向）
        (0,  3):  (0.9, 0.1),   # Hips → Spine1
        (3,  6):  (0.9, 0.1),   # Spine1 → Spine2
        (6,  9):  (0.9, 0.1),   # Spine2 → Spine3
        (9,  12): (0.8, 0.2),   # Spine3 → Neck
        (12, 15): (0.7, 0.3),   # Neck → Head
        # 胯宽偏移（横向）
        (0,  1):  (0.1, 0.9),   # Hips → LHip
        (0,  2):  (0.1, 0.9),   # Hips → RHip
        # 腿部（纵向）
        (1,  4):  (0.9, 0.1),   # LHip → LKnee
        (2,  5):  (0.9, 0.1),   # RHip → RKnee
        (4,  7):  (0.9, 0.1),   # LKnee → LAnkle
        (5,  8):  (0.9, 0.1),   # RKnee → RAnkle
        (7,  10): (0.6, 0.4),   # LAnkle → LFoot
        (8,  11): (0.6, 0.4),   # RAnkle → RFoot
        # 肩宽偏移（横向）
        (9,  13): (0.1, 0.9),   # Spine3 → LCollar
        (9,  14): (0.1, 0.9),   # Spine3 → RCollar
        # 手臂（混合）
        (13, 16): (0.5, 0.5),   # LCollar → LShoulder
        (14, 17): (0.5, 0.5),   # RCollar → RShoulder
        (16, 18): (0.6, 0.4),   # LShoulder → LElbow
        (17, 19): (0.6, 0.4),   # RShoulder → RElbow
        (18, 20): (0.6, 0.4),   # LElbow → LWrist
        (19, 21): (0.6, 0.4),   # RElbow → RWrist
    }

    def __init__(self, config: dict):
        """config: {"height": float, "width": float}"""
        self.height = float(config.get("height", 1.0))
        self.width  = float(config.get("width",  1.0))

    def bone_scale(self, parent_hml: int, child_hml: int) -> float:
        """返回该骨骼的 Translation 缩放比例。"""
        h_w, w_w = self._BONE_HW.get((parent_hml, child_hml), (0.5, 0.5))
        return h_w * self.height + w_w * self.width

    def all_bone_scales(self) -> dict:
        """返回所有骨骼的缩放比例 {(parent, child): scale}。"""
        return {bone: self.bone_scale(*bone) for bone in self._BONE_HW}

    def apply_to_joints(self, joint_pos: np.ndarray) -> np.ndarray:
        """
        将 HML3D joint positions 按体型比例重新缩放骨骼长度。
        joint_pos: (T, 22, 3) 或 (22, 3)
        返回: 相同 shape，各骨骼向量长度按 bone_scale 缩放。
        """
        from collections import deque
        single = (joint_pos.ndim == 2)
        if single:
            joint_pos = joint_pos[np.newaxis]

        children_map = {i: [c for c in range(22) if self.HML_PARENT[c] == i]
                        for i in range(22)}
        out = joint_pos.copy()

        # BFS：从 root 向外逐骨骼缩放，保证父骨骼先被处理
        queue = deque([0])
        visited = {0}
        while queue:
            p = queue.popleft()
            for c in children_map[p]:
                if c not in visited:
                    visited.add(c)
                    scale = self.bone_scale(p, c)
                    bone_vec = out[:, c] - out[:, p]   # (T, 3)
                    out[:, c] = out[:, p] + bone_vec * scale
                    queue.append(c)

        return out[0] if single else out

    def mesh_scales(self) -> tuple:
        """返回 (height_scale, width_scale) 供 post-LBS 顶点缩放使用。"""
        return self.height, self.width

    def summary(self) -> str:
        lines = [f"AvatarProfile(height={self.height:.3f}, width={self.width:.3f})",
                 f"  {'Bone':<24} {'h_w':>4} {'w_w':>4} {'scale':>7}",
                 f"  {'-'*42}"]
        for (p, c), (hw, ww) in self._BONE_HW.items():
            s = hw * self.height + ww * self.width
            name = f"{self._HML_NAMES[p]}→{self._HML_NAMES[c]}"
            lines.append(f"  {name:<24} {hw:>4.1f} {ww:>4.1f} {s:>7.4f}")
        return '\n'.join(lines)


# ── IK + motion.pkl conversion ───────────────────────────────────────────────

def hml3d_to_motion_pkl_entry(joint_pos, gltf_obj, skin_joints_list, node_to_sj,
                               ref_mat, ref_mat_inv, profile=None):
    """
    骨骼重定向 v3: parent-drives-child bone-direction IK

    核心原理：
      ViCo FK:
        global_T[C] = global_T[P] + global_R[P] @ nd_C.translation
        global_R[P] = global_R[grandP] @ rest_R_P @ ref_mat_inv[mi_P] @ motion_R[mi_P] @ ref_mat[mi_P]

      因此 joint P 的 motion_R 控制子关节 C 的位置方向。
      令 M = global_R[grandP] @ rest_R_P @ ref_mat_inv[mi_P]，
      令 v = normalize(ref_mat[mi_P] @ nd_C.translation)，
      令 target_in_motion = normalize(M.T @ target_dir_world)，
      则 motion_R[P] = align_vectors(target_in_motion, v)。

    对每个关节 P：其 motion_R 决定其 HML3D 主要子关节 C 的骨骼方向。
    根节点（Hips）和多分支节点设为 identity。

    profile: AvatarProfile — 若指定，先对 joint_pos 做骨骼缩放，
             使 IK 目标方向匹配当前体型的肢体比例。
    """
    from collections import deque

    # 按体型比例缩放 HML3D 关节位置（改变骨骼方向目标）
    if profile is not None:
        joint_pos = profile.apply_to_joints(joint_pos)

    T = len(joint_pos)

    HML_PARENT = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
    # 每个 HML3D 关节对应的主要子关节（单链用唯一子节点；多分支用 None 或选一个）
    HML_PRIMARY_CHILD = {}
    for hp in range(22):
        children = [c for c in range(22) if HML_PARENT[c] == hp]
        if not children:
            HML_PRIMARY_CHILD[hp] = None
        elif hp == 0:
            HML_PRIMARY_CHILD[hp] = None  # root: identity, can't satisfy 3 branches
        elif hp == 9:
            HML_PRIMARY_CHILD[hp] = 12   # spine3 → neck (main axis)
        else:
            HML_PRIMARY_CHILD[hp] = children[0]

    VICO_SJ_TO_HML = {sj: h for h, sj in HML3D_TO_VICO_SJ.items()}
    sj_to_ni = {sj: ni for ni, sj in node_to_sj.items()}

    # Build children map: sj → list of child sj's
    vico_children_sj = {sj: [] for sj in sj_to_ni}
    vico_parent_sj   = {}
    for sj, ni in sj_to_ni.items():
        for child_ni in (gltf_obj.nodes[ni].children or []):
            child_sj = node_to_sj.get(child_ni)
            if child_sj is not None:
                vico_children_sj[sj].append(child_sj)
                vico_parent_sj[child_sj] = sj

    # Root is Hips (sj=0)
    root_sj = 0

    # BFS order
    bfs_order = []
    queue = deque([root_sj])
    visited = {root_sj}
    while queue:
        sj = queue.popleft()
        bfs_order.append(sj)
        for child_sj in vico_children_sj[sj]:
            if child_sj not in visited:
                visited.add(child_sj)
                queue.append(child_sj)

    # Initialize output arrays
    trans     = joint_pos[:, 0].copy()           # root translation from HML3D
    rot_out   = np.zeros((T, 4));    rot_out[:, 0]  = 1.0
    joint_out = np.zeros((T, 64, 4)); joint_out[:, :, 0] = 1.0
    mat_out   = np.tile(np.eye(3), (T, 65, 1, 1)).astype(np.float64)

    def _R_to_q(R):
        q = Rotation.from_matrix(R).as_quat()   # [x,y,z,w]
        return np.array([q[3], q[0], q[1], q[2]])  # → [w,x,y,z]

    for t in range(T):
        G_R = {}  # sj → 3x3 world rotation (accumulated)

        for sj_P in bfs_order:
            ni_P    = sj_to_ni[sj_P]
            nd_P    = gltf_obj.nodes[ni_P]
            rest_R_P = (Rotation.from_quat(nd_P.rotation).as_matrix()
                        if nd_P.rotation else np.eye(3))
            mi_P    = VICO_SJ_TO_MI.get(sj_P)

            # Parent's world rotation
            if sj_P == root_sj:
                G_R_grandP = np.eye(3)   # root has no grandparent
            else:
                G_R_grandP = G_R.get(vico_parent_sj[sj_P], np.eye(3))

            if mi_P is None:
                # Joint not in motion mapping → propagate rest rotation only
                G_R[sj_P] = G_R_grandP @ rest_R_P
                continue

            ref_m     = ref_mat[mi_P]
            ref_m_inv = ref_mat_inv[mi_P]

            # M = G_R[grandP] @ rest_R_P @ ref_mat_inv[mi_P]
            M = G_R_grandP @ rest_R_P @ ref_m_inv

            # Determine target: find the primary HML3D child
            h_P  = VICO_SJ_TO_HML.get(sj_P)
            h_C  = HML_PRIMARY_CHILD.get(h_P) if h_P is not None else None
            sj_C = HML3D_TO_VICO_SJ.get(h_C)  if h_C is not None else None

            # sj_C must be a direct ViCo child of sj_P
            if (sj_C is not None and sj_C in vico_children_sj[sj_P]):
                ni_C  = sj_to_ni[sj_C]
                nd_C  = gltf_obj.nodes[ni_C]
                rest_t_C = np.array(nd_C.translation or [0, 0, 0])
                t_len = np.linalg.norm(rest_t_C)

                if t_len > 1e-6:
                    # Effective bone direction in motion_R space
                    v = ref_m @ rest_t_C
                    v_len = np.linalg.norm(v)
                    if v_len > 1e-6:
                        v /= v_len

                    # Target direction: from HML3D parent to HML3D child
                    target_world = joint_pos[t, h_C] - joint_pos[t, h_P]
                    tw_len = np.linalg.norm(target_world)

                    if tw_len > 1e-6 and v_len > 1e-6:
                        target_world /= tw_len
                        # Transform to motion_R space: motion_R @ v ∝ M.T @ target_world
                        target_motion = M.T @ target_world
                        tm_len = np.linalg.norm(target_motion)
                        if tm_len > 1e-6:
                            target_motion /= tm_len
                            try:
                                motion_R = Rotation.align_vectors(
                                    [target_motion], [v])[0].as_matrix()
                            except Exception:
                                motion_R = np.eye(3)
                        else:
                            motion_R = np.eye(3)
                    else:
                        motion_R = np.eye(3)
                else:
                    motion_R = np.eye(3)
            else:
                # Root / branching / leaf → identity (keep rest pose)
                motion_R = np.eye(3)

            # Compute and store global rotation
            G_R[sj_P] = M @ motion_R @ ref_m

            # Store motion_R → mat_out (this is what FK reads as motion_R)
            mat_out[t, mi_P] = motion_R
            q = _R_to_q(motion_R)
            if mi_P == 0:
                rot_out[t] = q
            elif mi_P - 1 < 64:
                joint_out[t, mi_P - 1] = q

    return {'trans': trans, 'rot': rot_out, 'joint': joint_out, 'mat': mat_out}


# ── Skeleton-space body scaling (correct approach) ───────────────────────────

# Per-sj translation component scaling: (y_scale_factor, x_scale_factor)
# y_scale_factor: coefficient on (height-1) for the Y component
# x_scale_factor: coefficient on (width-1)  for the X component
# Face/eye bones = (0, 0): no scaling — prevents eye/tooth floating
_SJ_BONE_COEFF = {
    # Spine chain
    1:  (0.9, 0.0),   # Spine
    2:  (0.9, 0.0),   # Spine1
    3:  (0.9, 0.0),   # Spine2
    4:  (0.6, 0.0),   # Neck      (partial height)
    5:  (0.4, 0.0),   # Neck1
    6:  (0.2, 0.0),   # Neck2
    7:  (0.0, 0.0),   # Head      (no scale — contains face)
    8:  (0.0, 0.0),   # LeftEye   (NO SCALE — prevents floating)
    9:  (0.0, 0.0),   # RightEye
    10: (0.0, 0.0),   # HeadTop_End
    # Shoulders: X=width, Y=height(partial)
    11: (0.4, 0.8),   # LeftShoulder
    37: (0.4, 0.8),   # RightShoulder
    # Arms: Y scales with height (arm length ∝ height)
    12: (0.8, 0.0),   # LeftArm
    38: (0.8, 0.0),   # RightArm
    13: (0.8, 0.0),   # LeftForeArm
    39: (0.8, 0.0),   # RightForeArm
    16: (0.8, 0.0),   # LeftHand
    42: (0.8, 0.0),   # RightHand
    # Hip lateral offset: X=width
    63: (0.0, 0.9),   # LeftUpLeg  (X=hip half-width)
    68: (0.0, 0.9),   # RightUpLeg
    # Legs: Y scales with height
    64: (0.9, 0.0),   # LeftLeg
    69: (0.9, 0.0),   # RightLeg
    65: (0.8, 0.0),   # LeftFoot
    70: (0.8, 0.0),   # RightFoot
    66: (0.5, 0.0),   # LeftToeBase
    71: (0.5, 0.0),   # RightToeBase
    # Everything else defaults to (0, 0): fingers, toes tips, etc.
}


def rot6d_to_matrix(d):
    """
    Convert 6D rotation representation to 3×3 rotation matrix.
    d: (..., 6)  — first two columns of the rotation matrix, concatenated.
    Returns (..., 3, 3) rotation matrix (columns are the axes).
    """
    a1 = d[..., :3]
    a2 = d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)   # (..., 3, 3) columns


def hml3d_6drot_to_motion_pkl_entry(joint_data, gltf_obj, skin_joints_list,
                                     node_to_sj, ref_mat, ref_mat_inv):
    """
    Approach B: drive ViCo skeleton directly from HumanML3D 6D rotation features,
    bypassing IK entirely.

    HumanML3D 263-dim layout:
      [0:4]    root features (ang_vel_y, lin_vel_xz, height_y)
      [4:67]   relative joint positions 21×3 (unused here)
      [67:193] 6D local rotations 21×6  ← used here
      [193:259] joint velocities (unused)
      [259:263] foot contact (unused)

    Pipeline:
      1. Recover root Y-rotation and XZ-position via recover_root_rot_pos (momask)
      2. Convert 6D features to local rotation matrices (joints 1-21)
      3. SMPL FK: accumulate global rotations G_R_smpl[h] = G_R_smpl[parent] @ local_R[h]
      4. For each ViCo joint with HML3D counterpart h_P:
           target_rel  = G_R_smpl[parent(h_P)].T @ G_R_smpl[h_P]
           local_R     = rest_R.T @ target_rel
           motion_R    = ref_mat[mi] @ local_R @ ref_mat_inv[mi]

    The rest_R.T term is important. ViCo/Mixamo joints often have large rest
    rotations, especially legs and arms. FK later evaluates each joint as
    rest_R @ local_R, so storing target_rel directly causes a second rest-pose
    rotation to be applied and twists the body.
    """
    import sys; sys.path.insert(0, MOMASK_DIR)
    from utils.motion_process import recover_root_rot_pos
    import torch
    from collections import deque

    T = len(joint_data)
    data_t = torch.from_numpy(joint_data).float()
    r_rot_quat, r_pos = recover_root_rot_pos(data_t)
    r_rot_quat = r_rot_quat.numpy()   # (T,4) [w,x,y,z]
    trans = r_pos.numpy().copy()       # (T,3) root XZ position

    # 6D rotations: (T, 21, 6)
    rot_6d = joint_data[:, 67:193].reshape(T, 21, 6)

    HML_PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    hml_children = {i: [c for c in range(22) if HML_PARENT[c] == i] for i in range(22)}
    hml_bfs = []
    q_bfs = deque([0]); visited_h = {0}
    while q_bfs:
        h = q_bfs.popleft(); hml_bfs.append(h)
        for c in hml_children[h]:
            if c not in visited_h: visited_h.add(c); q_bfs.append(c)

    VICO_SJ_TO_HML = {sj: h for h, sj in HML3D_TO_VICO_SJ.items()}
    sj_to_ni = {sj: ni for ni, sj in node_to_sj.items()}

    vico_children = {sj: [] for sj in sj_to_ni}
    for sj, ni in sj_to_ni.items():
        for c_ni in (gltf_obj.nodes[ni].children or []):
            c_sj = node_to_sj.get(c_ni)
            if c_sj is not None:
                vico_children[sj].append(c_sj)

    bfs_order = []
    q_bfs = deque([0]); visited_v = {0}
    while q_bfs:
        sj = q_bfs.popleft(); bfs_order.append(sj)
        for c in vico_children[sj]:
            if c not in visited_v: visited_v.add(c); q_bfs.append(c)

    def R_to_q(R):
        q = Rotation.from_matrix(R).as_quat()
        return np.array([q[3], q[0], q[1], q[2]])

    rot_out   = np.zeros((T, 4));     rot_out[:, 0]   = 1.0
    joint_out = np.zeros((T, 64, 4)); joint_out[:, :, 0] = 1.0
    mat_out   = np.tile(np.eye(3), (T, 65, 1, 1)).astype(np.float64)

    for t in range(T):
        # ── Step 1: SMPL FK → global rotations G_R_smpl[h] ──────────────────
        # Ry(90°): aligns SMPL canonical frame (+X forward) with ViCo/Mixamo frame (-Z forward)
        _C  = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        _CT = _C.T
        G_R_smpl = {}
        q = r_rot_quat[t]
        G_R_smpl[0] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        for h in hml_bfs[1:]:
            local_R = rot6d_to_matrix(rot_6d[t, h - 1])
            G_R_smpl[h] = G_R_smpl[HML_PARENT[h]] @ local_R
        # Apply Ry(90°) correction to all global rotations
        for h in G_R_smpl:
            G_R_smpl[h] = _C @ G_R_smpl[h] @ _CT

        # ── Step 2: Map to ViCo motion_R ─────────────────────────────────────
        for sj_P in bfs_order:
            mi_P = VICO_SJ_TO_MI.get(sj_P)
            h_P  = VICO_SJ_TO_HML.get(sj_P)
            ni_P = sj_to_ni[sj_P]
            nd_P = gltf_obj.nodes[ni_P]
            rest_R = (Rotation.from_quat(nd_P.rotation).as_matrix()
                      if nd_P.rotation else np.eye(3))

            if sj_P == 0:
                if mi_P is None:
                    continue
                # FK root is rest_R @ (ref^-1 @ motion_R @ ref). Convert the
                # desired SMPL/HML root orientation into ViCo motion space.
                local_R = rest_R.T @ G_R_smpl[0]
                motion_R = ref_mat[mi_P] @ local_R @ ref_mat_inv[mi_P]
                mat_out[t, mi_P] = motion_R
                rot_out[t] = R_to_q(motion_R)
                continue

            if h_P is None or h_P == 0 or mi_P is None:
                continue

            h_par = HML_PARENT[h_P]
            G_R_par_smpl = G_R_smpl.get(h_par, np.eye(3))
            G_R_cur_smpl = G_R_smpl[h_P]

            target_rel = G_R_par_smpl.T @ G_R_cur_smpl
            local_R    = rest_R.T @ target_rel
            motion_R   = ref_mat[mi_P] @ local_R @ ref_mat_inv[mi_P]

            mat_out[t, mi_P] = motion_R
            q = R_to_q(motion_R)
            if mi_P - 1 < 64:
                joint_out[t, mi_P - 1] = q

    return {'trans': trans, 'rot': rot_out, 'joint': joint_out, 'mat': mat_out}


def hml_joints_to_motion_pkl_entry(joint_pos, joint_data, gltf_obj, skin_joints_list,
                                    node_to_sj, ref_mat, ref_mat_inv):
    """
    Convert HumanML3D joint positions (from MoMask) to motion.pkl format for ViCo FK.

    Uses position-based bone-direction IK:
      For each ViCo parent bone sj_P with HumanML3D counterpart h_P:
        - b = normalize(nd.translation[child_vico])   — child direction in sj_P's local frame
        - d = normalize(jp[h_C] - jp[h_P])            — target direction in world space
        - G_R[sj_P] = align_vectors([d], [b])         — world rotation that maps b → d
        - R_vico_anim = rest_R.T @ G_R_parent.T @ G_R[sj_P]
        - motion_R = ref_mat @ R_vico_anim @ ref_mat_inv

    This correctly handles the ViCo/Mixamo convention where leg bone translations
    point +Y (upward) while rest_R is a ~180° flip to make them point downward at rest.
    The formula is coordinate-system agnostic — it only constrains bone DIRECTIONS.

    Root Y-rotation is taken from the integrated angular velocity in joint_data.
    """
    import sys; sys.path.insert(0, MOMASK_DIR)
    from utils.motion_process import recover_root_rot_pos
    from collections import deque

    T = len(joint_pos)
    data_t  = torch.from_numpy(joint_data).float()
    r_rot_quat, r_pos = recover_root_rot_pos(data_t)
    r_rot_quat = r_rot_quat.numpy()   # (T,4) [w,x,y,z] Y-axis quaternion
    r_pos      = r_pos.numpy()        # (T,3) root world position

    # Build skeleton maps
    sj_to_ni = {sj: ni for ni, sj in node_to_sj.items()}
    VICO_SJ_TO_HML = {v: k for k, v in HML3D_TO_VICO_SJ.items()}

    # Build parent/children maps for ViCo
    vico_parent = {}
    vico_children = {sj: [] for sj in sj_to_ni}
    for sj, ni in sj_to_ni.items():
        for c_ni in (gltf_obj.nodes[ni].children or []):
            c_sj = node_to_sj.get(c_ni)
            if c_sj is not None:
                vico_children[sj].append(c_sj)
                vico_parent[c_sj] = sj

    # HumanML3D primary child: the HML child that drives each HML parent bone's direction
    HML_PARENT = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
    HML_PRIMARY_CHILD = {}
    for hp in range(22):
        children = [c for c in range(22) if HML_PARENT[c] == hp]
        HML_PRIMARY_CHILD[hp] = (12 if hp == 9 else (children[0] if children else None))
    HML_PRIMARY_CHILD[0] = None   # root: handled via r_rot_quat

    # BFS order over ViCo joints
    root_sj = 0
    bfs_order = []
    visited = {root_sj}
    queue = deque([root_sj])
    while queue:
        sj = queue.popleft()
        bfs_order.append(sj)
        for c in vico_children[sj]:
            if c not in visited:
                visited.add(c)
                queue.append(c)

    def R_to_q(R):
        q = Rotation.from_matrix(R).as_quat()   # [x,y,z,w]
        return np.array([q[3], q[0], q[1], q[2]])

    trans     = r_pos.copy()
    rot_out   = np.zeros((T, 4));     rot_out[:, 0]   = 1.0
    joint_out = np.zeros((T, 64, 4)); joint_out[:, :, 0] = 1.0
    mat_out   = np.tile(np.eye(3), (T, 65, 1, 1)).astype(np.float64)

    for t in range(T):
        G_R = {}   # sj → 3×3 world rotation matrix

        for sj_P in bfs_order:
            ni_P    = sj_to_ni[sj_P]
            nd_P    = gltf_obj.nodes[ni_P]
            rest_R  = Rotation.from_quat(nd_P.rotation).as_matrix() if nd_P.rotation else np.eye(3)
            G_R_par = G_R.get(vico_parent.get(sj_P), np.eye(3))
            mi_P    = VICO_SJ_TO_MI.get(sj_P)
            h_P     = VICO_SJ_TO_HML.get(sj_P)

            # ── Compute G_R[sj_P] ────────────────────────────────────────────
            if sj_P == root_sj:
                # Root: use Y-axis facing from recovered angular velocity
                q = r_rot_quat[t]
                G_R[sj_P] = Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()

            elif h_P is not None:
                # Has HumanML3D counterpart: align ViCo bone direction to HML target
                h_C  = HML_PRIMARY_CHILD.get(h_P)
                sj_C = HML3D_TO_VICO_SJ.get(h_C) if h_C is not None else None

                # Determine bone direction reference child:
                # Prefer the HML-mapped child if it's a direct ViCo child;
                # otherwise fall back to the first actual ViCo child (fixes Neck).
                if sj_C is not None and sj_C in vico_children[sj_P]:
                    ref_child_sj = sj_C
                elif vico_children[sj_P] and h_C is not None:
                    ref_child_sj = vico_children[sj_P][0]   # use first ViCo child for b
                else:
                    ref_child_sj = None

                if ref_child_sj is not None:
                    ni_C = sj_to_ni[ref_child_sj]
                    rest_t = np.array(gltf_obj.nodes[ni_C].translation or [0,0,0])
                    t_len  = np.linalg.norm(rest_t)

                    if t_len > 1e-6:
                        b = rest_t / t_len                              # bone direction in sj_P local frame
                        d = joint_pos[t, h_C] - joint_pos[t, h_P]
                        d_len = np.linalg.norm(d)

                        if d_len > 1e-6:
                            d /= d_len
                            # Key fix: align in PARENT-LOCAL space, not world space.
                            # This makes each joint's roll inherit from its parent,
                            # preventing the independent roll accumulation that causes
                            # knee/elbow twisting when aligning in world space.
                            d_local = G_R_par.T @ d   # target direction in parent's frame
                            try:
                                R_local = Rotation.align_vectors([d_local], [b])[0].as_matrix()

                                # ── Foot roll fix ──────────────────────────────────
                                # After the swing aligns b (toe direction) to d_local,
                                # add a twist around d_local so the foot's local +Z
                                # (top-of-foot axis) faces world +Y (upward).
                                # This prevents the sole from rolling sideways during walking.
                                if sj_P in {65, 70}:   # LeftFoot, RightFoot
                                    world_up_loc = G_R_par.T @ np.array([0., 1., 0.])
                                    up_perp = world_up_loc - np.dot(world_up_loc, d_local) * d_local
                                    up_len  = np.linalg.norm(up_perp)
                                    if up_len > 0.15:   # skip when foot direction ≈ vertical
                                        up_perp /= up_len
                                        z_loc  = R_local @ np.array([0., 0., 1.])
                                        z_perp = z_loc - np.dot(z_loc, d_local) * d_local
                                        z_len  = np.linalg.norm(z_perp)
                                        if z_len > 1e-6:
                                            z_perp /= z_len
                                            cos_a = np.clip(np.dot(z_perp, up_perp), -1., 1.)
                                            sin_a = np.dot(np.cross(z_perp, up_perp), d_local)
                                            twist = np.arctan2(sin_a, cos_a)
                                            R_twist  = Rotation.from_rotvec(d_local * twist).as_matrix()
                                            R_local  = R_twist @ R_local

                                G_R[sj_P] = G_R_par @ R_local
                            except Exception:
                                G_R[sj_P] = G_R_par @ rest_R
                        else:
                            G_R[sj_P] = G_R_par @ rest_R
                    else:
                        G_R[sj_P] = G_R_par @ rest_R
                else:
                    G_R[sj_P] = G_R_par @ rest_R
            else:
                # No HumanML3D counterpart: propagate parent rotation + rest
                G_R[sj_P] = G_R_par @ rest_R

            # ── Derive motion_R from G_R[sj_P] ───────────────────────────────
            if mi_P is None:
                continue

            # R_vico_anim = (G_R_par @ rest_R)^{-1} @ G_R[sj_P]
            R_vico_anim = rest_R.T @ G_R_par.T @ G_R[sj_P]
            motion_R    = ref_mat[mi_P] @ R_vico_anim @ ref_mat_inv[mi_P]
            mat_out[t, mi_P] = motion_R
            q = R_to_q(motion_R)
            if mi_P == 0:
                rot_out[t] = q
            elif mi_P - 1 < 64:
                joint_out[t, mi_P - 1] = q

    return {'trans': trans, 'rot': rot_out, 'joint': joint_out, 'mat': mat_out}


def _recompute_ibm(gltf_obj, skin_joints, node_to_sj):
    """
    重算 Inverse Bind Matrix——在 scale_skeleton_rest_pose 修改了 node.translation 后调用。
    等价于用缩放后的 rest-pose 做 FK（identity motion），再求逆。
    """
    from collections import deque
    n_sj = len(skin_joints)
    G_R  = np.tile(np.eye(3), (n_sj, 1, 1))
    G_T  = np.zeros((n_sj, 3))

    root_ni = skin_joints[0]
    nd0     = gltf_obj.nodes[root_ni]
    G_R[0]  = Rotation.from_quat(nd0.rotation).as_matrix() if nd0.rotation else np.eye(3)
    G_T[0]  = np.array(nd0.translation or [0, 0, 0], dtype=np.float64)

    visited = {root_ni}
    queue   = deque([root_ni])
    while queue:
        ni_p = queue.popleft()
        sj_p = node_to_sj[ni_p]
        for ci in (gltf_obj.nodes[ni_p].children or []):
            if ci not in node_to_sj or ci in visited:
                continue
            visited.add(ci)
            sj_c = node_to_sj[ci]
            nd_c = gltf_obj.nodes[ci]
            rest_t = np.array(nd_c.translation or [0, 0, 0], dtype=np.float64)
            rest_R = Rotation.from_quat(nd_c.rotation).as_matrix() if nd_c.rotation else np.eye(3)
            G_T[sj_c] = G_T[sj_p] + G_R[sj_p] @ rest_t
            G_R[sj_c] = G_R[sj_p] @ rest_R
            queue.append(ci)

    G4 = np.zeros((n_sj, 4, 4))
    G4[:, :3, :3] = G_R
    G4[:, :3,  3] = G_T
    G4[:,  3,  3] = 1.0
    return np.linalg.inv(G4)


def build_scaled_character(gltf_obj, profile, skin_joints, node_to_sj):
    """
    骨骼空间体型缩放（单次 LBS 方案，无 noodle-person 畸变）：

    只修改骨骼 rest translation，保持原始顶点(v_orig)和逆绑定矩阵(ibm_orig)不变。
    动画时仍用 lbs(v_orig, G_anim_scaled, ibm_orig)。

    缩放效果来源：G_anim_scaled 的 FK 将各骨骼放到新位置（更长/更宽），
    骨骼在 world-space 的位置改变自然带动顶点移位，无需 double-LBS。

    注意：joint 处会有 ~1cm 量级的小偏差（IBM 与新 bind-pose 不完全匹配），
    这是骨骼重定向的正常代价，不会产生扭曲。

    返回: gltf_scaled （deep copy，translation 已缩放）
    调用方继续使用 ibm_orig 和 primitives_orig 做 LBS。
    """
    import copy
    gltf_scaled = copy.deepcopy(gltf_obj)
    dh = profile.height - 1.0
    dw = profile.width  - 1.0
    for sj, ni in enumerate(skin_joints):
        nd = gltf_scaled.nodes[ni]
        if nd.translation is None:
            continue
        y_c, x_c = _SJ_BONE_COEFF.get(sj, (0.0, 0.0))
        if y_c == 0.0 and x_c == 0.0:
            continue
        t = list(nd.translation)
        t[1] = t[1] * (1.0 + y_c * dh)
        t[0] = t[0] * (1.0 + x_c * dw)
        gltf_scaled.nodes[ni].translation = t
    return gltf_scaled


# ── Direct GLB generation from joint positions ───────────────────────────────

def generate_vico_glbs_from_joints(char_name, joint_pos_120,
                                   glb_path, motion_data, out_dir, n_frames=120,
                                   profile=None):
    """
    Generate ViCo animated GLBs directly from MoMask joint positions.
    joint_pos_120: (T, 22, 3) joint positions in HumanML3D coords (Y-up, meters)
    n_frames: number of frames to generate (input will be trimmed/tiled)
    profile: AvatarProfile — 控制体型缩放（骨骼 Translation + post-LBS 顶点缩放）
    """
    T_in = len(joint_pos_120)
    if T_in >= n_frames:
        joint_pos_120 = joint_pos_120[:n_frames]
    else:
        reps = int(np.ceil(n_frames / T_in))
        joint_pos_120 = np.tile(joint_pos_120, (reps, 1, 1))[:n_frames]
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_vico_animated import (
        load_char_primitives, fk_global_transforms, lbs, export_glb_pbr,
        build_skeleton, compute_normals, compute_tangents
    )
    import pygltflib, json

    print(f"  Generating from MoMask joints: {out_dir.split('/')[-1]}")
    primitives, skin_joints, ibm = load_char_primitives(glb_path)
    gltf_obj = pygltflib.GLTF2().load(glb_path)
    skin_joints_list, node_to_sj, _ = build_skeleton(gltf_obj)

    # ref_mat from idle rest pose (used by fk_global_transforms and IK)
    ref_mat     = motion_data['idle']['mat'][0].astype(np.float64)  # (65, 3, 3)
    ref_mat_inv = np.linalg.inv(ref_mat)

    # 骨骼空间体型缩放（单次 LBS）：
    #   只缩放 bone translation → 影响 FK 位置 → 顶点随骨骼移位
    #   保持 ibm_orig 和 v_orig 不变，避免 double-LBS 导致的关节扭曲
    if profile is not None:
        gltf_scaled = build_scaled_character(gltf_obj, profile, skin_joints_list, node_to_sj)
    else:
        gltf_scaled = gltf_obj
    # 始终使用原始顶点和原始 IBM
    primitives_active = primitives
    ibm_active        = ibm

    # IK 使用缩放骨骼，使旋转目标匹配新比例
    entry = hml3d_to_motion_pkl_entry(joint_pos_120, gltf_scaled, skin_joints_list, node_to_sj,
                                      ref_mat, ref_mat_inv, profile=profile)

    os.makedirs(out_dir, exist_ok=True)

    for fi in range(n_frames):
        mot_rot   = entry['rot'][fi]
        mot_jnt   = entry['joint'][fi]
        mot_trans = entry['trans'][fi]

        G = fk_global_transforms(gltf_scaled, skin_joints_list, node_to_sj,
                                  mot_rot, mot_jnt, mot_trans,
                                  ref_mat, ref_mat_inv)

        deformed_prims = []
        for prim in primitives_active:
            dv = lbs(prim['verts'], prim['joints'], prim['weights'], G, ibm_active).astype(np.float32)
            dn = compute_normals(dv, prim['faces'])
            dt = None
            if prim['normal_png'] is not None and prim['normal_scale'] > 0:
                dt = compute_tangents(dv, dn, prim['uvs'], prim['faces'])
            deformed_prims.append(dict(
                verts=dv, normals=dn, tangents=dt,
                uvs=prim['uvs'], faces=prim['faces'],
                color_jpg=prim['color_jpg'], normal_png=prim['normal_png'],
                metallic=prim['metallic'], roughness=prim['roughness'],
                normal_scale=prim['normal_scale'],
            ))

        glb_out = os.path.join(out_dir, f"frame{fi:03d}.glb")
        export_glb_pbr(deformed_prims, glb_out)

        cfg = {"mass": 60.0, "friction_coefficient": 0.0,
               "restitution_coefficient": 0.0, "is_collidable": False,
               "render_asset": f"frame{fi:03d}.glb",
               "collision_asset": f"frame{fi:03d}.glb"}
        with open(os.path.join(out_dir, f"frame{fi:03d}.object_config.json"), 'w') as f:
            json.dump(cfg, f, indent=2)

    print(f"    → {n_frames} frames written to {out_dir}")


# ── Batch processing for a scan (motion.pkl clip-based, no IK) ───────────────

def _text_to_clip(text):
    """Map annotation text to the best matching motion.pkl clip name."""
    t = text.lower()
    if any(w in t for w in ['toast', 'cocktail', 'drink', 'cheer', 'celebrat']):
        return 'Cheering'
    if any(w in t for w in ['convers', 'talk', 'chat', 'discuss', 'couple', 'intimate', 'quiet']):
        return 'Talking1'
    if any(w in t for w in ['phone', 'call']):
        return 'TalkingOnPhone'
    if any(w in t for w in ['greet', 'hello', 'wave']):
        return 'StandingGreeting'
    if any(w in t for w in ['run', 'jog']):
        return 'Running'
    if any(w in t for w in ['walk', 'stroll', 'slide', 'banis']):
        return 'walk'
    if any(w in t for w in ['sit', 'seat', 'sofa', 'couch']):
        return 'sit'
    if any(w in t for w in ['pick', 'lift', 'grab']):
        return 'pick'
    if any(w in t for w in ['reach', 'vacuum', 'clean', 'wipe', 'mop', 'sweep',
                             'meal', 'prep', 'cook', 'photo', 'sunscreen', 'apply']):
        return 'reach'
    return 'idle'


def generate_vico_glbs_from_clip(glb_path, clip_name, motion_data, out_dir, n_frames=120,
                                  height_scale=1.0, width_scale=1.0):
    """
    Generate ViCo animated GLBs by replaying a motion.pkl clip.
    Body scaling (height/width) is done correctly at the skeleton level:
      1. Scale all bone translations by (width, height, width)
      2. Recompute IBM from the scaled rest-pose skeleton
      3. Scale bind-pose vertices by the same factors
      4. Standard LBS — joints stay connected, no distortion
    """
    import sys, copy
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_vico_animated import (
        load_char_primitives, fk_global_transforms, lbs, export_glb_pbr,
        build_skeleton, compute_normals, compute_tangents
    )
    import pygltflib

    clip = motion_data.get(clip_name, motion_data['idle'])
    T_clip = len(clip['rot'])

    primitives, skin_joints, ibm = load_char_primitives(glb_path)
    gltf_obj = pygltflib.GLTF2().load(glb_path)
    sj_list, node_to_sj, _ = build_skeleton(gltf_obj)

    # Detect cm-unit models (mixamo): root node Y >> 1 means skeleton is in cm.
    # bone_scale converts cm bone offsets → meters in FK.
    # verts and IBM are already corrected in load_char_primitives.
    root_t = gltf_obj.nodes[skin_joints[0]].translation or [0, 0, 0]
    bone_scale = 0.01 if abs(root_t[1]) > 5.0 else 1.0

    ref_mat     = motion_data['idle']['mat'][0].astype(np.float64)
    ref_mat_inv = np.linalg.inv(ref_mat)

    gltf_active       = gltf_obj
    ibm_active        = ibm
    primitives_active = primitives

    os.makedirs(out_dir, exist_ok=True)
    for fi in range(n_frames):
        ti = fi % T_clip
        G = fk_global_transforms(
            gltf_active, sj_list, node_to_sj,
            clip['rot'][ti], clip['joint'][ti], clip['trans'][ti],
            ref_mat, ref_mat_inv, bone_scale=bone_scale
        )
        # Post-LBS scaling around the root position (shared center for all
        # primitives so body parts stay connected — no arm-separation gaps).
        root_xz = np.array([clip['trans'][ti][0], clip['trans'][ti][2]], dtype=np.float32)
        root_y  = float(clip['trans'][ti][1])

        deformed = []
        for p in primitives_active:
            dv = lbs(p['verts'], p['joints'], p['weights'], G, ibm_active).astype(np.float32)
            if width_scale != 1.0:
                dv[:, 0] = root_xz[0] + (dv[:, 0] - root_xz[0]) * width_scale
                dv[:, 2] = root_xz[1] + (dv[:, 2] - root_xz[1]) * width_scale
            if height_scale != 1.0:
                dv[:, 1] = root_y + (dv[:, 1] - root_y) * height_scale
            dn = compute_normals(dv, p['faces'])
            dt = compute_tangents(dv, dn, p['uvs'], p['faces']) if p.get('normal_png') else None
            deformed.append({**p, 'verts': dv, 'normals': dn, 'tangents': dt})

        glb_out = os.path.join(out_dir, f"frame{fi:03d}.glb")
        export_glb_pbr(deformed, glb_out)
        cfg = {"mass": 60.0, "friction_coefficient": 0.0, "restitution_coefficient": 0.0,
               "is_collidable": False, "render_asset": f"frame{fi:03d}.glb",
               "collision_asset": f"frame{fi:03d}.glb"}
        with open(os.path.join(out_dir, f"frame{fi:03d}.object_config.json"), 'w') as f:
            json.dump(cfg, f)

    print(f"    → {n_frames} frames written to {out_dir}  (clip:{clip_name} H×{height_scale} W×{width_scale})")


def generate_vico_glbs_from_momask(text, glb_path, models, motion_data, out_dir,
                                    n_frames=120, height_scale=1.0, width_scale=1.0,
                                    device='cuda'):
    """
    Generate ViCo animated GLBs from text using MoMask.
    models: return value of load_momask()
    motion_data: loaded motion.pkl (for ref_mat from idle clip)
    """
    import sys, copy
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_vico_animated import (
        load_char_primitives, fk_global_transforms, lbs, export_glb_pbr,
        build_skeleton, compute_normals, compute_tangents
    )
    import pygltflib

    vq_model, t2m_model, res_model, len_model, model_opt, mean, std = models

    print(f"  MoMask generating: '{text}'")
    joint_pos, joint_data = generate_motion(
        text, vq_model, t2m_model, res_model, len_model, model_opt, mean, std,
        device=device)
    print(f"  Generated {len(joint_data)} frames")

    primitives, skin_joints, ibm = load_char_primitives(glb_path)
    gltf_obj  = pygltflib.GLTF2().load(glb_path)
    sj_list, node_to_sj, _ = build_skeleton(gltf_obj)

    ref_mat     = motion_data['idle']['mat'][0].astype(np.float64)
    ref_mat_inv = np.linalg.inv(ref_mat)

    motion_entry = hml_joints_to_motion_pkl_entry(
        joint_pos, joint_data, gltf_obj, sj_list, node_to_sj, ref_mat, ref_mat_inv)

    T_clip = len(motion_entry['rot'])
    os.makedirs(out_dir, exist_ok=True)

    for fi in range(n_frames):
        ti = fi % T_clip
        G = fk_global_transforms(
            gltf_obj, sj_list, node_to_sj,
            motion_entry['rot'][ti], motion_entry['joint'][ti], motion_entry['trans'][ti],
            ref_mat, ref_mat_inv)

        root_xz = np.array([motion_entry['trans'][ti][0], motion_entry['trans'][ti][2]], dtype=np.float32)
        root_y  = float(motion_entry['trans'][ti][1])

        deformed = []
        for p in primitives:
            dv = lbs(p['verts'], p['joints'], p['weights'], G, ibm).astype(np.float32)
            if width_scale != 1.0:
                dv[:, 0] = root_xz[0] + (dv[:, 0] - root_xz[0]) * width_scale
                dv[:, 2] = root_xz[1] + (dv[:, 2] - root_xz[1]) * width_scale
            if height_scale != 1.0:
                dv[:, 1] = root_y + (dv[:, 1] - root_y) * height_scale
            dn = compute_normals(dv, p['faces'])
            dt = compute_tangents(dv, dn, p['uvs'], p['faces']) if p.get('normal_png') else None
            deformed.append({**p, 'verts': dv, 'normals': dn, 'tangents': dt})

        glb_out = os.path.join(out_dir, f"frame{fi:03d}.glb")
        export_glb_pbr(deformed, glb_out)
        cfg = {"mass": 60.0, "friction_coefficient": 0.0, "restitution_coefficient": 0.0,
               "is_collidable": False, "render_asset": f"frame{fi:03d}.glb",
               "collision_asset": f"frame{fi:03d}.glb"}
        with open(os.path.join(out_dir, f"frame{fi:03d}.object_config.json"), 'w') as f:
            json.dump(cfg, f)

    print(f"    → {n_frames} frames written to {out_dir}  (MoMask T={T_clip})")


def process_scan(scan_id, use_momask=False, device='cuda'):
    with open(ANNOT_PATH) as f:
        annotations = json.load(f)
    with open(MOTION_PKL, 'rb') as f:
        motion_data = pickle.load(f)

    # 加载身份数据（Phase 4 avatar 分配）
    identity_path = os.path.join(DATA_PATH, 'Identity/identities.json')
    identities = {}
    if os.path.exists(identity_path):
        with open(identity_path) as f:
            identities = json.load(f)

    # fallback 模型列表（无身份数据时使用）
    fallback_models = sorted([os.path.join(VICO_MODELS, f) for f in os.listdir(VICO_MODELS)
                               if f.endswith('.glb') and not f.endswith('.glb.glb')])
    BODY_SCALES_FALLBACK = [
        (1.00, 1.08), (1.03, 1.12), (0.97, 1.05), (1.05, 1.10),
        (0.95, 1.18), (1.02, 1.15), (0.98, 1.20), (1.06, 1.08),
    ]

    scan_data = annotations.get(scan_id, {})

    if use_momask:
        print("Loading MoMask models...")
        models = load_momask(device=device)
    else:
        models = None

    char_idx = 0
    for pid, v in scan_data.items():
        translations = v.get('translation', [])
        if not translations:
            char_idx += 1; continue

        cat = v.get('category', '')
        if ':' in cat:
            _, act = cat.split(':', 1)
            text = act.replace('_', ' ').strip().rstrip('.')
        else:
            text = cat.replace('_', ' ')

        out_key = f"{scan_id}__char{char_idx:02d}"
        out_dir = os.path.join(OUT_ROOT, out_key)

        # ── 从 identity 取模型和体型，否则 fallback ──────────────────────────
        if pid in identities and 'avatar' in identities[pid]:
            av       = identities[pid]['avatar']
            glb_path = av['model_path']
            h_scale  = av['height_scale']
            w_scale  = av['width_scale']
            id_info  = f"{identities[pid]['name']} ({identities[pid]['age_group']}, {identities[pid]['gender']})"
        else:
            glb_path = fallback_models[char_idx % len(fallback_models)]
            h_scale, w_scale = BODY_SCALES_FALLBACK[char_idx % len(BODY_SCALES_FALLBACK)]
            id_info  = f"fallback"

        try:
            if use_momask:
                print(f"\n[char{char_idx:02d}] {id_info}  →  MoMask  H×{h_scale} W×{w_scale}")
                generate_vico_glbs_from_momask(text, glb_path, models, motion_data, out_dir,
                                               height_scale=h_scale, width_scale=w_scale,
                                               device=device)
            else:
                clip = _text_to_clip(text)
                print(f"\n[char{char_idx:02d}] {id_info}  →  clip:{clip}  H×{h_scale} W×{w_scale}")
                generate_vico_glbs_from_clip(glb_path, clip, motion_data, out_dir,
                                             height_scale=h_scale, width_scale=w_scale)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

        char_idx += 1


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",   type=str, default=None, help="Single text prompt (MoMask only)")
    parser.add_argument("--name",   type=str, default="momask_test")
    parser.add_argument("--scan",    type=str,  default=None, help="Batch process scan")
    parser.add_argument("--momask",  action="store_true",     help="Use MoMask (default: prerecorded clips)")
    parser.add_argument("--device",  type=str,  default="cuda")
    args = parser.parse_args()

    if args.text:
        print(f"Generating motion for: '{args.text}'")
        vq_model, t2m_model, res_model, len_model, model_opt, mean, std = load_momask(args.device)
        joint_pos, _ = generate_motion(args.text, vq_model, t2m_model, res_model, len_model,
                                       model_opt, mean, std, device=args.device)
        out = f"/tmp/{args.name}_joints.npy"
        np.save(out, joint_pos)
        print(f"Joint positions saved: {out}  shape: {joint_pos.shape}")
    elif args.scan:
        process_scan(args.scan, use_momask=args.momask, device=args.device)
    else:
        parser.print_help()


def _run_test_frame(args):  # kept for reference, no longer called
    """
    --test_frame 测试：
    1. 打印 AvatarProfile 骨骼缩放表
    2. 用 idle 姿态 joint_pos（T=1）跑完整 IK + LBS + post-scale 管线
    3. 输出单帧 GLB 到 /tmp/avatar_profile_test/frame000.glb
    """
    import sys, pickle
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_vico_animated import (
        load_char_primitives, fk_global_transforms, lbs, export_glb_pbr,
        build_skeleton, compute_normals, compute_tangents
    )
    import pygltflib

    profile = AvatarProfile({'height': args.height, 'width': args.width})
    print("\n" + profile.summary() + "\n")

    # 找第一个可用的 ViCo GLB
    vico_cfgs = sorted([os.path.join(VICO_MODELS, f) for f in os.listdir(VICO_MODELS)
                        if f.endswith('.glb') and f.startswith('custom_')])
    if not vico_cfgs:
        print(f"ERROR: no custom_*.glb found in {VICO_MODELS}")
        return
    glb_path = vico_cfgs[0]
    print(f"Using GLB: {glb_path}")

    with open(MOTION_PKL, 'rb') as f:
        motion_data = pickle.load(f)

    ref_mat     = motion_data['idle']['mat'][0].astype(np.float64)
    ref_mat_inv = np.linalg.inv(ref_mat)

    # 从 idle clip 提取 T=1 帧的 HML3D-like joint positions
    # idle trans 给 root，关节来自 FK 的 rest-pose 近似
    idle_trans = motion_data['idle']['trans'][:1]       # (1, 3)
    idle_joint = motion_data['idle']['joint'][:1]       # (1, 64, 4) quaternions
    idle_rot   = motion_data['idle']['rot'][:1]         # (1, 4)

    # 用 idle 的 FK 全局位置近似 HML3D joint_pos（只需要骨骼方向，数值可近似）
    gltf_obj = pygltflib.GLTF2().load(glb_path)
    primitives, skin_joints, ibm = load_char_primitives(glb_path)
    skin_joints_list, node_to_sj, _ = build_skeleton(gltf_obj)

    # 建一个简单的 T-pose joint_pos (22, 3)：用 ViCo rest-pose 骨骼直接 FK
    sj_to_ni = {sj: ni for ni, sj in node_to_sj.items()}
    joint_pos_1 = np.zeros((1, 22, 3))   # root at origin

    from collections import deque
    HML_PARENT = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
    hml_children = {i: [c for c in range(22) if HML_PARENT[c] == i] for i in range(22)}

    # 用 ViCo rest-pose bone translations 填充 HML3D joint positions
    vico_sj_to_hml = {sj: h for h, sj in HML3D_TO_VICO_SJ.items()}
    sj_to_pos = {0: np.zeros(3)}
    bfs = deque([0])   # root sj=0 → hml=0
    while bfs:
        sj_P = bfs.popleft()
        ni_P = sj_to_ni.get(sj_P)
        if ni_P is None: continue
        for child_ni in (gltf_obj.nodes[ni_P].children or []):
            sj_C = node_to_sj.get(child_ni)
            if sj_C is None: continue
            nd_C = gltf_obj.nodes[child_ni]
            t_C  = np.array(nd_C.translation or [0, 0, 0])
            sj_to_pos[sj_C] = sj_to_pos[sj_P] + t_C
            bfs.append(sj_C)

    for sj, pos in sj_to_pos.items():
        h = vico_sj_to_hml.get(sj)
        if h is not None:
            joint_pos_1[0, h] = pos

    print(f"T-pose joint_pos built from ViCo rest skeleton (shape {joint_pos_1.shape})")

    # 打印缩放前后的骨骼长度变化
    scaled = profile.apply_to_joints(joint_pos_1)
    print(f"\n骨骼长度变化（原始 → 缩放后）：")
    HML_NAMES = AvatarProfile._HML_NAMES
    for (p, c) in list(AvatarProfile._BONE_HW.keys())[:8]:
        orig_len  = np.linalg.norm(joint_pos_1[0, c] - joint_pos_1[0, p])
        scale_len = np.linalg.norm(scaled[0, c] - scaled[0, p])
        s = profile.bone_scale(p, c)
        print(f"  {HML_NAMES[p]}→{HML_NAMES[c]:<15}: {orig_len:.4f}m → {scale_len:.4f}m  (×{s:.4f})")

    # 运行完整管线：骨骼缩放 → IK → FK → LBS → 输出 GLB
    out_dir = "/tmp/avatar_profile_test"
    os.makedirs(out_dir, exist_ok=True)

    # Step A: 骨骼空间体型缩放（单次 LBS）
    gltf_scaled = build_scaled_character(gltf_obj, profile, skin_joints_list, node_to_sj)

    # Step B: IK（使用缩放后的骨骼）
    entry = hml3d_to_motion_pkl_entry(
        joint_pos_1, gltf_scaled, skin_joints_list, node_to_sj,
        ref_mat, ref_mat_inv, profile=profile
    )

    # Step C: FK → LBS（使用原始顶点和原始 IBM）
    G = fk_global_transforms(
        gltf_scaled, skin_joints_list, node_to_sj,
        entry['rot'][0], entry['joint'][0], entry['trans'][0],
        ref_mat, ref_mat_inv
    )

    deformed_prims = []
    for prim in primitives:
        dv = lbs(prim['verts'], prim['joints'], prim['weights'], G, ibm).astype(np.float32)
        dn = compute_normals(dv, prim['faces'])
        dt = None
        if prim['normal_png'] is not None and prim['normal_scale'] > 0:
            dt = compute_tangents(dv, dn, prim['uvs'], prim['faces'])
        deformed_prims.append(dict(
            verts=dv, normals=dn, tangents=dt,
            uvs=prim['uvs'], faces=prim['faces'],
            color_jpg=prim['color_jpg'], normal_png=prim['normal_png'],
            metallic=prim['metallic'], roughness=prim['roughness'],
            normal_scale=prim['normal_scale'],
        ))

    glb_out = os.path.join(out_dir, "frame000.glb")
    export_glb_pbr(deformed_prims, glb_out)
    print(f"\n测试帧输出: {glb_out}")
    print(f"GLB 大小: {os.path.getsize(glb_out)/1024:.1f} KB")
    print("完成。用 render_mitsuba.py 或 pyrender 可视化查看体型效果。")


if __name__ == "__main__":
    main()
