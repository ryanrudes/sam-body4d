from typing import List, Dict

def pick_best_from_run(iou_list: List[float], l: int, r: int) -> int:
    """在闭区间 [l, r] 的 1-run 内，返回 iou 最大的下标；并列取更靠近 0 的（可改）"""
    best_i = l
    best_v = iou_list[l]
    for i in range(l + 1, r + 1):
        v = iou_list[i]
        if v > best_v:
            best_v = v
            best_i = i
    return best_i

def build_zero_neighbor_dict(bin_list: List[int], iou_list: List[float]) -> Dict[int, List[int]]:
    """
    对每个 0 的位置 idx：
      - 找左侧最近的连续 1 段（run），在该段内选 iou 最大的 1 的下标
      - 找右侧最近的连续 1 段（run），同理
    返回：{zero_idx: [left_best?, right_best?]}
    """
    n = len(bin_list)
    assert n == len(iou_list), "bin_list 和 iou_list 长度必须相同"
    for v in bin_list:
        if v not in (0, 1):
            raise ValueError("bin_list 只能包含 0/1")

    out: Dict[int, List[int]] = {}

    for idx, v in enumerate(bin_list):
        if v != 0:
            continue

        picked = []

        # -------- left side: find nearest 1-run ending at idx-1 or earlier --------
        j = idx - 1
        while j >= 0 and bin_list[j] == 0:
            j -= 1
        if j >= 0 and bin_list[j] == 1:
            r = j
            l = j
            while l - 1 >= 0 and bin_list[l - 1] == 1:
                l -= 1
            left_best = pick_best_from_run(iou_list, l, r)
            picked.append(left_best)

        # -------- right side: find nearest 1-run starting at idx+1 or later --------
        j = idx + 1
        while j < n and bin_list[j] == 0:
            j += 1
        if j < n and bin_list[j] == 1:
            l = j
            r = j
            while r + 1 < n and bin_list[r + 1] == 1:
                r += 1
            right_best = pick_best_from_run(iou_list, l, r)
            picked.append(right_best)

        out[idx] = picked

    return out




KEY_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]  # key body joints for prompting

KEY_POINT_NAME = { 0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear", 5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow", 9: "left_hip", 10: "right_hip", 11: "left_knee", 12: "right_knee", 13: "left_ankle", 14: "right_ankle", 15: "left_big_toe_tip", 16: "left_small_toe_tip", 17: "left_heel", 18: "right_big_toe_tip", 19: "right_small_toe_tip", 20: "right_heel", 21: "right_thumb_tip", 22: "right_thumb_first_joint", 23: "right_thumb_second_joint", 24: "right_thumb_third_joint", 25: "right_index_tip", 26: "right_index_first_joint", 27: "right_index_second_joint", 28: "right_index_third_joint", 29: "right_middle_tip", 30: "right_middle_first_joint", 31: "right_middle_second_joint", 32: "right_middle_third_joint", 33: "right_ring_tip", 34: "right_ring_first_joint", 35: "right_ring_second_joint", 36: "right_ring_third_joint", 37: "right_pinky_tip", 38: "right_pinky_first_joint", 39: "right_pinky_second_joint", 40: "right_pinky_third_joint", 41: "right_wrist", 42: "left_thumb_tip", 43: "left_thumb_first_joint", 44: "left_thumb_second_joint", 45: "left_thumb_third_joint", 46: "left_index_tip", 47: "left_index_first_joint", 48: "left_index_second_joint", 49: "left_index_third_joint", 50: "left_middle_tip", 51: "left_middle_first_joint", 52: "left_middle_second_joint", 53: "left_middle_third_joint", 54: "left_ring_tip", 55: "left_ring_first_joint", 56: "left_ring_second_joint", 57: "left_ring_third_joint", 58: "left_pinky_tip", 59: "left_pinky_first_joint", 60: "left_pinky_second_joint", 61: "left_pinky_third_joint", 62: "left_wrist", 63: "left_olecranon", 64: "right_olecranon", 65: "left_cubital_fossa", 66: "right_cubital_fossa", 67: "left_acromion", 68: "right_acromion", 69: "neck", }

KINEMATIC_EDGES = [
    (5, 7), (7, 62),      # left shoulder-elbow-wrist
    (6, 8), (8, 41),      # right shoulder-elbow-wrist
    (9, 11), (11, 13),    # left hip-knee-ankle
    (10, 12), (12, 14),   # right hip-knee-ankle
    (5, 9), (6, 10),      # shoulder-hip (torso)
    (9, 10),              # left hip - right hip
]

def build_body_keypoint_dict(points_70x2):
    """
    Args:
        points_70x2: array-like, shape (70, 2)
            每一行是一个关键点的 (x, y)
        KEY_BODY: list[int], length = 12
            需要选取的 70 维关键点索引
        KEY_POINT_NAME: dict[int, str]
            key: 0~69, value: 关键点名称

    Returns:
        dict:
            {
                idx: {
                    "xy": [x, y],
                    "name": str
                },
                ...
            }
    """
    assert len(points_70x2) == 70, "points_70x2 must have length 70"

    out = {}
    for idx in KEY_BODY:
        xy = points_70x2[idx]
        out[idx] = {
            "xy": xy.tolist() if hasattr(xy, "tolist") else list(xy),
            "name": KEY_POINT_NAME[idx],
        }
    return out


def draw_named_keypoints_on_image(
    image_path,
    kp_dict,
    save_path="res.jpg",
    radius=4,
    font_scale=0.5,
    thickness=1,
):
    """
    Args:
        image_path: str, input jpg path
        kp_dict: dict, output of build_body_keypoint_dict(...)
            { idx: {"xy":[x,y], "name":str}, ... }
        save_path: str, output image path
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    H, W = img.shape[:2]

    for _, v in kp_dict.items():
        xy = v.get("xy", None)
        name = v.get("name", "")

        if xy is None or len(xy) < 2:
            continue

        x, y = float(xy[0]), float(xy[1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            continue

        # point
        cv2.circle(img, (xi, yi), radius, (0, 255, 0), -1)

        # text (offset) + outline
        tx, ty = xi + 6, yi - 6
        tx = max(0, min(W - 1, tx))
        ty = max(0, min(H - 1, ty))

        cv2.putText(
            img, str(name), (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
        )
        cv2.putText(
            img, str(name), (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    ok = cv2.imwrite(save_path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")
    return save_path

def draw_named_keypoints_on_image(
    image_path,
    kp_dict,
    save_path="res.jpg",
    mask_path=None,
    radius=8,
    font_scale=0.5,   # 数字更大
    thickness=2,      # 更粗
):
    """
    Args:
        image_path: str, input jpg path
        kp_dict: dict
            { idx: {"xy":[x,y], "name":str}, ... }  # 这里 name 不用了，直接画 idx
        save_path: str
        mask_path: str|None
            P 模式 mask（palette），前景像素值 == 1 才算前景；否则算背景
            若 mask 尺寸与 image 不一致，会用最近邻 resize
    """
    import cv2
    import numpy as np
    from PIL import Image

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    H, W = img.shape[:2]

    mask_arr = None
    if mask_path is not None:
        m = Image.open(mask_path)           # P 模式也能直接读
        mask_arr = np.array(m)              # 得到 uint8 索引图
        if mask_arr.ndim != 2:
            # 有些情况下会变成 (H,W,3) 之类，强制取单通道
            mask_arr = mask_arr[..., 0]
        if mask_arr.shape[:2] != (H, W):
            mask_arr = cv2.resize(mask_arr, (W, H), interpolation=cv2.INTER_NEAREST)

    for idx, v in kp_dict.items():
        xy = v.get("xy", None)
        if xy is None or len(xy) < 2:
            continue

        x, y = float(xy[0]), float(xy[1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            continue

        # mask: P 模式，==1 前景，否则背景
        if mask_arr is None:
            color = (0, 255, 0)  # 默认绿色
        else:
            color = (0, 255, 0) if int(mask_arr[yi, xi]) == 1 else (0, 0, 255)  # green / red (BGR)

        # draw point
        cv2.circle(img, (xi, yi), radius, color, -1)

        # draw index number (bigger + outline)
        tx, ty = xi + 6, yi - 6
        tx = max(0, min(W - 1, tx))
        ty = max(0, min(H - 1, ty))

        txt = str(idx)
        cv2.putText(
            img, txt, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA
        )
        cv2.putText(
            img, txt, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    ok = cv2.imwrite(save_path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")
    return save_path


def draw(image_path, points_70x2, mask_path=None, res_path='res.jpg'):
    points_dict = build_body_keypoint_dict(points_70x2)
    draw_named_keypoints_on_image(image_path, points_dict, save_path=res_path, mask_path=mask_path)

def draw_s(points_70x2, res_path='skeleton.jpg'):
    points_dict = build_body_keypoint_dict(points_70x2)
    draw_skeleton_on_white_bg(points_dict, save_path=res_path)

def draw_skeleton_on_white_bg(
    kp_dict,
    save_path="skeleton.jpg",
    canvas_size=None,          # None -> auto from points; or (H, W)
    pad=30,
    point_radius=5,
    point_thickness=-1,        # -1 filled
    line_thickness=3,
    font_scale=0.5,
    font_thickness=1,
    KINEMATIC_EDGES=None,
):
    """
    输入 keypoint 字典（{idx: {"xy":[x,y], ...}, ...}），生成白底骨架图：
      - 点：绿色
      - 边：蓝色（按 KINEMATIC_EDGES 连接）
      - 标号：idx（更大、更清晰）

    坐标默认认为是“像素坐标系”(x right, y down)。
    如果 canvas_size=None，会根据点的范围自动生成画布，并把点平移到画布内。
    """
    import numpy as np
    import cv2

    if KINEMATIC_EDGES is None:
        KINEMATIC_EDGES = [
            (5, 7), (7, 62),      # left shoulder-elbow-wrist
            (6, 8), (8, 41),      # right shoulder-elbow-wrist
            (9, 11), (11, 13),    # left hip-knee-ankle
            (10, 12), (12, 14),   # right hip-knee-ankle
            (5, 9), (6, 10),      # shoulder-hip (torso)
            (9, 10),              # left hip - right hip
        ]

    # collect valid points
    pts = {}
    for idx, v in kp_dict.items():
        xy = v.get("xy", None)
        if xy is None or len(xy) < 2:
            continue
        x, y = float(xy[0]), float(xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        pts[int(idx)] = (x, y)

    if len(pts) == 0:
        raise ValueError("No valid keypoints found in kp_dict")

    # decide canvas + coordinate shift
    if canvas_size is None:
        xs = np.array([p[0] for p in pts.values()], dtype=np.float32)
        ys = np.array([p[1] for p in pts.values()], dtype=np.float32)
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())

        W = int(np.ceil((max_x - min_x) + 2 * pad))
        H = int(np.ceil((max_y - min_y) + 2 * pad))
        W = max(W, 64)
        H = max(H, 64)

        shift_x = -min_x + pad
        shift_y = -min_y + pad
    else:
        H, W = int(canvas_size[0]), int(canvas_size[1])
        shift_x, shift_y = 0.0, 0.0

    # white background
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    def to_int_xy(x, y):
        xi = int(round(x + shift_x))
        yi = int(round(y + shift_y))
        return xi, yi

    def in_bounds(xi, yi):
        return 0 <= xi < W and 0 <= yi < H

    # draw edges first (so points are on top)
    for a, b in KINEMATIC_EDGES:
        if a not in pts or b not in pts:
            continue
        xa, ya = pts[a]
        xb, yb = pts[b]
        x1, y1 = to_int_xy(xa, ya)
        x2, y2 = to_int_xy(xb, yb)
        if in_bounds(x1, y1) and in_bounds(x2, y2):
            cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), line_thickness, cv2.LINE_AA)  # blue (BGR)

    # draw points + indices
    for idx, (x, y) in pts.items():
        xi, yi = to_int_xy(x, y)
        if not in_bounds(xi, yi):
            continue

        cv2.circle(canvas, (xi, yi), point_radius, (0, 255, 0), point_thickness, cv2.LINE_AA)  # green

        # index label (bigger + outline)
        tx, ty = xi + 8, yi - 8
        tx = max(0, min(W - 1, tx))
        ty = max(0, min(H - 1, ty))
        txt = str(idx)

        cv2.putText(
            canvas, txt, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
            font_thickness + 3, cv2.LINE_AA
        )
        cv2.putText(
            canvas, txt, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20),
            font_thickness, cv2.LINE_AA
        )

    ok = cv2.imwrite(save_path, canvas)
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")
    return save_path


import math
import torch
import torch.nn.functional as F
from PIL import Image

def find_topk_similar_points_on_a(
    a: torch.Tensor,
    b: torch.Tensor,
    point_dict: dict,
    mask_path: str = None,
    obj_id: int = None,
    H: int = 480,
    W: int = 854,
    feat_h: int = 72,
    feat_w: int = 72,
    topk: int = 3,
    eps: float = 1e-8,
):
    assert a.ndim == 4 and b.ndim == 4 and a.shape == b.shape, "a and b must have same shape (1,C,Hf,Wf)"
    assert a.shape[0] == 1, "batch must be 1"
    C = a.shape[1]
    assert C == 256, f"expected C=256, got {C}"
    assert a.shape[2] == feat_h and a.shape[3] == feat_w, f"expected (feat_h,feat_w)=({feat_h},{feat_w})"
    assert topk >= 1

    device = a.device
    dtype = a.dtype

    # --- load mask (optional) ---
    mask_img = None
    obj_id_int = None
    if mask_path is not None and obj_id is not None:
        m = Image.open(mask_path)
        if m.mode != "P":
            m = m.convert("P")
        if m.size != (W, H):
            m = m.resize((W, H), resample=Image.NEAREST)
        mask_img = m
        obj_id_int = int(obj_id)

    def is_finite_xy(x: float, y: float) -> bool:
        # python float / numpy float / torch scalar float 都能 cover
        return (x is not None and y is not None and
                math.isfinite(float(x)) and math.isfinite(float(y)))

    def is_foreground(x_img: float, y_img: float) -> bool:
        # NEW: NaN/Inf 直接判 False（跳过）
        if not is_finite_xy(x_img, y_img):
            return False
        if mask_img is None:
            return True
        xi = int(round(float(x_img)))
        yi = int(round(float(y_img)))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            return False
        return int(mask_img.getpixel((xi, yi))) == obj_id_int

    # normalize a for cosine similarity
    a_norm = a / (a.norm(dim=1, keepdim=True) + eps)          # (1,256,72,72)
    a_flat = a_norm[0].reshape(C, feat_h * feat_w)            # (256, N)

    out = {}

    def imgxy_to_featxy(x_img: float, y_img: float):
        x_img = max(0.0, min(float(W - 1), float(x_img)))
        y_img = max(0.0, min(float(H - 1), float(y_img)))
        x_feat = x_img / (W - 1) * (feat_w - 1)
        y_feat = y_img / (H - 1) * (feat_h - 1)
        return x_feat, y_feat

    def featxy_to_imgxy(x_feat: int, y_feat: int):
        x_img = x_feat / (feat_w - 1) * (W - 1)
        y_img = y_feat / (feat_h - 1) * (H - 1)
        return float(x_img), float(y_img)

    def sample_b_feature(x_feat: float, y_feat: float):
        gx = x_feat / (feat_w - 1) * 2.0 - 1.0
        gy = y_feat / (feat_h - 1) * 2.0 - 1.0
        grid = torch.tensor([[[[gx, gy]]]], device=device, dtype=dtype)  # (1,1,1,2)
        v = F.grid_sample(b, grid, mode="bilinear", align_corners=True)[0, :, 0, 0]  # (256,)
        v = v / (v.norm() + eps)
        return v

    with torch.no_grad():
        for idx, vv in point_dict.items():
            xy = vv.get("xy", None)
            if xy is None or len(xy) < 2:
                continue

            x_img, y_img = float(xy[0]), float(xy[1])

            # NEW: 先过滤 NaN/Inf（避免后面任何地方炸）
            if not is_finite_xy(x_img, y_img):
                continue

            # NEW: mask 前景过滤
            if not is_foreground(x_img, y_img):
                continue

            x_feat, y_feat = imgxy_to_featxy(x_img, y_img)

            q = sample_b_feature(x_feat, y_feat)              # (256,)
            sims = torch.matmul(a_flat.t(), q)                # (N,)
            _, top_idx = torch.topk(sims, k=min(topk, sims.numel()), largest=True)

            a_topk_xy = []
            for flat_i in top_idx.tolist():
                yy = flat_i // feat_w
                xx = flat_i % feat_w
                ax, ay = featxy_to_imgxy(xx, yy)
                a_topk_xy.append([ax, ay])

            out[int(idx)] = {
                "b_xy": [x_img, y_img],
                "a_topk_xy": a_topk_xy,
            }

    return out



# import torch
# import torch.nn.functional as F
# from PIL import Image

# def find_topk_similar_points_on_a(
#     a: torch.Tensor,                 # (1, 256, 72, 72)
#     b: torch.Tensor,                 # (1, 256, 72, 72)
#     point_dict: dict,                # {idx: {"xy":[x,y], ...}, ...} 这些点在 b 的 (H,W) 坐标系里
#     mask_path: str = None,           # 新增：mask 路径（P 模式 / 或可转为 P/L）
#     obj_id: int = None,              # 新增：前景 ID（mask == obj_id 为前景）
#     H: int = 480,
#     W: int = 854,
#     feat_h: int = 72,
#     feat_w: int = 72,
#     topk: int = 3,
#     eps: float = 1e-8,
# ):
#     """
#     对 b 的每个关键点（仅当落在 mask 前景区域内）：
#       1) 用 (H,W) 把 (x,y) 映射到 (feat_h, feat_w) 特征图上取出 b 的 256D 特征向量
#       2) 在 a 的所有 (feat_h*feat_w) 位置上做 cosine 相似度
#       3) 取 topk 最相近位置，映射回 (H,W) 坐标输出

#     注意：
#       - 如果提供了 mask_path + obj_id，则只有 mask(x,y) == obj_id 的 keypoint 才会参与匹配；
#         其他点会被跳过（不出现在输出 out 里）。

#     Returns:
#         out: dict
#             {
#               idx: {
#                 "b_xy": [x, y],                         # 原始 b 上关键点（H,W）
#                 "a_topk_xy": [[x1,y1],[x2,y2],[x3,y3]]  # a 上 topk 点（H,W）
#               }, ...
#             }
#     """
#     assert a.ndim == 4 and b.ndim == 4 and a.shape == b.shape, "a and b must have same shape (1,C,Hf,Wf)"
#     assert a.shape[0] == 1, "batch must be 1"
#     C = a.shape[1]
#     assert C == 256, f"expected C=256, got {C}"
#     assert a.shape[2] == feat_h and a.shape[3] == feat_w, f"expected (feat_h,feat_w)=({feat_h},{feat_w})"
#     assert topk >= 1

#     device = a.device
#     dtype = a.dtype

#     # --- load mask (optional) ---
#     mask_img = None
#     if mask_path is not None and obj_id is not None:
#         m = Image.open(mask_path)
#         # 你希望用 P 模式；如果不是 P，也尽量转成 P/L 后再取像素值
#         if m.mode != "P":
#             # 若原本就是单通道/灰度，L 更直观；但你说 P 模式，这里保持转为 P
#             m = m.convert("P")
#         # 确保 mask 与 (W,H) 对齐（b 的坐标系）
#         if m.size != (W, H):
#             m = m.resize((W, H), resample=Image.NEAREST)
#         mask_img = m
#         obj_id_int = int(obj_id)

#     def is_foreground(x_img: float, y_img: float) -> bool:
#         if mask_img is None:
#             return True
#         xi = int(round(x_img))
#         yi = int(round(y_img))
#         if xi < 0 or xi >= W or yi < 0 or yi >= H:
#             return False
#         # P 模式下 getpixel 返回 palette index（就是我们要的 obj_id）
#         return int(mask_img.getpixel((xi, yi))) == obj_id_int

#     # normalize a for cosine similarity
#     a_norm = a / (a.norm(dim=1, keepdim=True) + eps)          # (1,256,72,72)
#     a_flat = a_norm[0].reshape(C, feat_h * feat_w)            # (256, N)

#     out = {}

#     def imgxy_to_featxy(x_img: float, y_img: float):
#         x_img = max(0.0, min(float(W - 1), float(x_img)))
#         y_img = max(0.0, min(float(H - 1), float(y_img)))
#         x_feat = x_img / (W - 1) * (feat_w - 1)
#         y_feat = y_img / (H - 1) * (feat_h - 1)
#         return x_feat, y_feat

#     def featxy_to_imgxy(x_feat: int, y_feat: int):
#         x_img = x_feat / (feat_w - 1) * (W - 1)
#         y_img = y_feat / (feat_h - 1) * (H - 1)
#         return float(x_img), float(y_img)

#     def sample_b_feature(x_feat: float, y_feat: float):
#         gx = x_feat / (feat_w - 1) * 2.0 - 1.0
#         gy = y_feat / (feat_h - 1) * 2.0 - 1.0
#         grid = torch.tensor([[[[gx, gy]]]], device=device, dtype=dtype)  # (1,1,1,2)
#         v = F.grid_sample(b, grid, mode="bilinear", align_corners=True)[0, :, 0, 0]  # (256,)
#         v = v / (v.norm() + eps)
#         return v

#     with torch.no_grad():
#         for idx, vv in point_dict.items():
#             xy = vv.get("xy", None)
#             if xy is None or len(xy) < 2:
#                 continue

#             x_img, y_img = float(xy[0]), float(xy[1])

#             # --- NEW: only match keypoints inside foreground on mask (b coords) ---
#             if not is_foreground(x_img, y_img):
#                 continue

#             x_feat, y_feat = imgxy_to_featxy(x_img, y_img)

#             q = sample_b_feature(x_feat, y_feat)              # (256,)
#             sims = torch.matmul(a_flat.t(), q)                # (N,)
#             _, top_idx = torch.topk(sims, k=min(topk, sims.numel()), largest=True)

#             a_topk_xy = []
#             for flat_i in top_idx.tolist():
#                 yy = flat_i // feat_w
#                 xx = flat_i % feat_w
#                 ax, ay = featxy_to_imgxy(xx, yy)
#                 a_topk_xy.append([ax, ay])

#             out[int(idx)] = {
#                 "b_xy": [x_img, y_img],
#                 "a_topk_xy": a_topk_xy,
#             }

#     return out








# def find_topk_similar_points_on_a(
#     a: torch.Tensor,                 # (1, 32, 288, 288)
#     b: torch.Tensor,                 # (1, 32, 288, 288)
#     point_dict: dict,                # {idx: {"xy":[x,y], ...}, ...}  这些点在 b 的 480x854 坐标系里
#     H: int = 480,
#     W: int = 854,
#     feat_hw: int = 288,
#     topk: int = 3,
#     eps: float = 1e-8,
# ):
#     """
#     对 b 的每个关键点：
#       1) 用 (H,W) 把 (x,y) 归一化后映射到 288x288 特征图上取出 b 的 32D 特征向量
#       2) 在 a 的所有 288x288 位置上做相似度（cosine/dot on normalized features）
#       3) 取 topk=3 的最相近位置，映射回 (H,W) 坐标输出

#     Returns:
#         out: dict
#             {
#               idx: {
#                 "b_xy": [x, y],                 # 原始 b 上关键点（480x854）
#                 "a_topk_xy": [[x1,y1],[x2,y2],[x3,y3]]   # a 上 topk 点（480x854）
#               }, ...
#             }
#     """
#     assert a.ndim == 4 and b.ndim == 4 and a.shape == b.shape, "a and b must be (1,32,288,288) with same shape"
#     assert a.shape[0] == 1 and a.shape[1] == 32 and a.shape[2] == feat_hw and a.shape[3] == feat_hw
#     assert topk >= 1

#     device = a.device
#     dtype = a.dtype

#     # ---- pre-normalize a features (cosine similarity) ----
#     # a_norm: (1,32,288,288)
#     a_norm = a / (a.norm(dim=1, keepdim=True) + eps)

#     # flatten spatial dims
#     # a_flat: (32, 288*288)
#     a_flat = a_norm[0].reshape(32, feat_hw * feat_hw)

#     # We'll also normalize b on-the-fly per query vector
#     out = {}

#     # helper: b image xy (480x854) -> feature map xy (0..287)
#     def imgxy_to_featxy(x_img: float, y_img: float):
#         # clamp to image bounds
#         x_img = max(0.0, min(float(W - 1), float(x_img)))
#         y_img = max(0.0, min(float(H - 1), float(y_img)))

#         # map to [0, feat_hw-1]
#         x_feat = x_img / (W - 1) * (feat_hw - 1)
#         y_feat = y_img / (H - 1) * (feat_hw - 1)
#         return x_feat, y_feat

#     # helper: feature map xy (0..287) -> img xy (480x854)
#     def featxy_to_imgxy(x_feat: int, y_feat: int):
#         x_img = x_feat / (feat_hw - 1) * (W - 1)
#         y_img = y_feat / (feat_hw - 1) * (H - 1)
#         return float(x_img), float(y_img)

#     # sample b feature at a single (x_feat, y_feat) using bilinear grid_sample
#     # using align_corners=True so that -1/1 align with 0/(feat_hw-1)
#     def sample_b_feature(x_feat: float, y_feat: float):
#         # normalize to grid [-1, 1]
#         gx = x_feat / (feat_hw - 1) * 2.0 - 1.0
#         gy = y_feat / (feat_hw - 1) * 2.0 - 1.0
#         grid = torch.tensor([[[[gx, gy]]]], device=device, dtype=dtype)  # (1,1,1,2)
#         # (1,32,1,1) -> (32,)
#         v = F.grid_sample(b, grid, mode="bilinear", align_corners=True)[0, :, 0, 0]
#         v = v / (v.norm() + eps)
#         return v  # (32,)

#     with torch.no_grad():
#         for idx, v in point_dict.items():
#             xy = v.get("xy", None)
#             if xy is None or len(xy) < 2:
#                 continue

#             x_img, y_img = float(xy[0]), float(xy[1])
#             x_feat, y_feat = imgxy_to_featxy(x_img, y_img)

#             q = sample_b_feature(x_feat, y_feat)  # (32,)

#             # cosine sim with all positions in a: (288*288,)
#             sims = torch.matmul(a_flat.t(), q)  # (N,)
#             top_vals, top_idx = torch.topk(sims, k=min(topk, sims.numel()), largest=True)

#             a_topk_xy = []
#             for flat_i in top_idx.tolist():
#                 yy = flat_i // feat_hw
#                 xx = flat_i % feat_hw
#                 ax, ay = featxy_to_imgxy(xx, yy)
#                 a_topk_xy.append([ax, ay])

#             out[int(idx)] = {
#                 "b_xy": [x_img, y_img],
#                 "a_topk_xy": a_topk_xy,
#             }

#     return out

def visualize_topk_points_on_image(
    image_path,
    match_dict,
    save_path="vis_topk.jpg",
    radius=6,
    thickness=-1,
):
    """
    Args:
        image_path: str
            原始图片路径（480x854）
        match_dict: dict
            {
              idx: {
                "b_xy": [x, y],
                "a_topk_xy": [[x1,y1],[x2,y2],[x3,y3]]
              }, ...
            }
        save_path: str
        radius: int
        thickness: int
            -1 表示实心圆
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    H, W = img.shape[:2]

    radius = int(radius)
    thickness = int(thickness)

    # 每个 keypoint 一个确定性颜色（稳定、不随机）
    def color_from_index(idx: int):
        r = (idx * 37) % 255
        g = (idx * 59) % 255
        b = (idx * 83) % 255
        return (int(b), int(g), int(r))  # BGR

    for idx, v in match_dict.items():
        pts = v.get("a_topk_xy", [])
        if len(pts) == 0:
            continue

        color = color_from_index(int(idx))

        for (x, y) in pts:
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or xi >= W or yi < 0 or yi >= H:
                continue

            cv2.circle(
                img,
                (xi, yi),
                radius,
                color,
                thickness,
                cv2.LINE_AA,
            )

    ok = cv2.imwrite(save_path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")

    return save_path

def visualize_closest_top1_per_keypoint(
    image_path,
    point_dict,
    topk_dict,
    save_path="vis_closest_top1.jpg",
    H=480,
    W=854,
    radius=6,
    thickness=-1,
    draw_query_point=False,   # True: 也把 b_xy/原始 keypoint 画出来（同色空心）
):
    """
    对每个 keypoint（来自 point_dict），在 topk_dict[idx]["a_topk_xy"] 里选一个
    与该 keypoint (x,y) 最接近的点（在 480x854 坐标系下用欧氏距离）。
    然后在 image_path 对应的图片上画出来：每个 keypoint 一个颜色（与之前一致的 idx->color 映射）。

    Args:
        image_path: str
        point_dict: dict
            {idx: {"xy":[x,y], ...}, ...}  (keypoint 在 b 的 480x854 坐标系)
        topk_dict: dict
            {idx: {"a_topk_xy":[[x1,y1],[x2,y2],[x3,y3]], "b_xy":[x,y]}, ...}
        save_path: str
        draw_query_point: bool
            是否也画出原始 keypoint（同色空心圆）
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    imgH, imgW = img.shape[:2]
    # 不强制，但你说的应该是 480x854
    # assert (imgH, imgW) == (H, W)

    radius = int(radius)
    thickness = int(thickness)

    def color_from_index(idx: int):
        r = (idx * 37) % 255
        g = (idx * 59) % 255
        b = (idx * 83) % 255
        return (int(b), int(g), int(r))  # BGR

    def clamp_int_xy(x, y):
        xi, yi = int(round(float(x))), int(round(float(y)))
        xi = max(0, min(imgW - 1, xi))
        yi = max(0, min(imgH - 1, yi))
        return xi, yi

    out_closest = {}

    for idx, v in point_dict.items():
        idx = int(idx)
        xy = v.get("xy", None)
        if xy is None or len(xy) < 2:
            continue
        qx, qy = float(xy[0]), float(xy[1])

        cand = None
        if idx in topk_dict:
            cand = topk_dict[idx].get("a_topk_xy", None)

        if not cand:
            continue

        # pick closest candidate to (qx,qy)
        best_pt = None
        best_d2 = None
        for pt in cand:
            if pt is None or len(pt) < 2:
                continue
            x, y = float(pt[0]), float(pt[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            d2 = (x - qx) ** 2 + (y - qy) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_pt = (x, y)

        if best_pt is None:
            continue

        color = color_from_index(idx)

        # draw chosen point (filled)
        xi, yi = clamp_int_xy(best_pt[0], best_pt[1])
        cv2.circle(img, (xi, yi), radius, color, thickness, cv2.LINE_AA)

        # optionally draw query keypoint (hollow)
        if draw_query_point:
            qxi, qyi = clamp_int_xy(qx, qy)
            cv2.circle(img, (qxi, qyi), radius + 2, color, max(2, abs(thickness) if thickness != -1 else 2), cv2.LINE_AA)

        out_closest[idx] = {
            "query_xy": [qx, qy],
            "closest_xy": [float(best_pt[0]), float(best_pt[1])],
            "dist2": float(best_d2),
        }

    ok = cv2.imwrite(save_path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")

    return save_path, out_closest

def draw_keybody17_on_image_from_kp70(
    image_path: str,
    kp_xy: "np.ndarray",                      # (70, 2), xy
    save_path: str = "res.jpg",
    mask_path: str | None = None,             # P-mode, fg==1, bg==0
    radius: int = 8,
    font_scale: float = 0.6,
    thickness: int = 2,
    key_ids=None,                             # default KEY_BODY below
    draw_name: bool = False,                  # False: draw idx; True: draw name
    draw_idx_and_name: bool = False,          # if True and draw_name True -> "idx:name"
    text_margin: int = 8,                     # horizontal offset from point
):
    """
    Changes requested:
    1) Text is placed strictly LEFT/RIGHT of the point (same y as the point; OpenCV uses baseline).
    2) Left/right split is NOT by image center. Instead, compute a split by x-median of the selected points:
       - points with x < median_x => "left group" => text on LEFT
       - points with x >= median_x => "right group" => text on RIGHT
    3) mask: Image.open(mask_path).convert("P"); fg==1 -> green else red
    """
    import cv2
    import numpy as np
    from PIL import Image

    # -------- defaults --------
    KEY_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]
    KEY_POINT_NAME = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
        5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
        9: "left_hip", 10: "right_hip", 11: "left_knee", 12: "right_knee",
        13: "left_ankle", 14: "right_ankle",
        15: "left_big_toe_tip", 16: "left_small_toe_tip", 17: "left_heel",
        18: "right_big_toe_tip", 19: "right_small_toe_tip", 20: "right_heel",
        21: "right_thumb_tip", 22: "right_thumb_first_joint", 23: "right_thumb_second_joint",
        24: "right_thumb_third_joint", 25: "right_index_tip", 26: "right_index_first_joint",
        27: "right_index_second_joint", 28: "right_index_third_joint", 29: "right_middle_tip",
        30: "right_middle_first_joint", 31: "right_middle_second_joint", 32: "right_middle_third_joint",
        33: "right_ring_tip", 34: "right_ring_first_joint", 35: "right_ring_second_joint",
        36: "right_ring_third_joint", 37: "right_pinky_tip", 38: "right_pinky_first_joint",
        39: "right_pinky_second_joint", 40: "right_pinky_third_joint",
        41: "right_wrist",
        42: "left_thumb_tip", 43: "left_thumb_first_joint", 44: "left_thumb_second_joint",
        45: "left_thumb_third_joint", 46: "left_index_tip", 47: "left_index_first_joint",
        48: "left_index_second_joint", 49: "left_index_third_joint", 50: "left_middle_tip",
        51: "left_middle_first_joint", 52: "left_middle_second_joint", 53: "left_middle_third_joint",
        54: "left_ring_tip", 55: "left_ring_first_joint", 56: "left_ring_second_joint",
        57: "left_ring_third_joint", 58: "left_pinky_tip", 59: "left_pinky_first_joint",
        60: "left_pinky_second_joint", 61: "left_pinky_third_joint",
        62: "left_wrist",
        63: "left_olecranon", 64: "right_olecranon", 65: "left_cubital_fossa", 66: "right_cubital_fossa",
        67: "left_acromion", 68: "right_acromion", 69: "neck",
    }

    if key_ids is None:
        key_ids = KEY_BODY

    # -------- validate kp_xy --------
    if not isinstance(kp_xy, np.ndarray):
        raise TypeError(f"kp_xy must be np.ndarray, got {type(kp_xy)}")
    if kp_xy.ndim != 2 or kp_xy.shape[1] != 2:
        raise ValueError(f"kp_xy must have shape (N,2), got {kp_xy.shape}")
    if kp_xy.shape[0] < 70:
        raise ValueError(f"kp_xy must have at least 70 rows, got {kp_xy.shape[0]}")

    # -------- load image --------
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    H, W = img.shape[:2]

    # -------- load mask (optional): force P-mode --------
    mask_arr = None
    if mask_path is not None:
        m = Image.open(mask_path).convert("P")
        mask_arr = np.array(m)
        if mask_arr.ndim != 2:
            mask_arr = mask_arr[..., 0]
        if mask_arr.shape[:2] != (H, W):
            mask_arr = cv2.resize(mask_arr, (W, H), interpolation=cv2.INTER_NEAREST)

    # -------- determine split by x-median of VALID selected points --------
    xs = []
    for idx in key_ids:
        if 0 <= idx < kp_xy.shape[0]:
            x, y = kp_xy[idx]
            if np.isfinite(x) and np.isfinite(y):
                xi, yi = int(round(float(x))), int(round(float(y)))
                if 0 <= xi < W and 0 <= yi < H:
                    xs.append(float(x))
    median_x = float(np.median(xs)) if len(xs) > 0 else (W * 0.5)

    # helper: text size
    def get_text_wh(text: str):
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        return tw, th, baseline

    # -------- draw selected keypoints --------
    for idx in key_ids:
        if idx < 0 or idx >= kp_xy.shape[0]:
            continue

        x, y = kp_xy[idx]
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        xi, yi = int(round(float(x))), int(round(float(y)))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            continue

        # color by mask (fg==1 else bg)
        if mask_arr is None:
            color = (0, 255, 0)
        else:
            color = (0, 255, 0) if int(mask_arr[yi, xi]) == 1 else (0, 0, 255)

        cv2.circle(img, (xi, yi), radius, color, -1)

        # text content
        if draw_name:
            name = KEY_POINT_NAME.get(int(idx), str(int(idx)))
            txt = f"{idx}:{name}" if draw_idx_and_name else name
        else:
            txt = str(int(idx))

        tw, th, baseline = get_text_wh(txt)

        # LEFT/RIGHT by median_x of points (not image center)
        is_left_group = float(x) < median_x

        # "正左右": same y (baseline aligned near point center)
        ty = yi + int(th * 0.35)  # push baseline down a bit so glyph center aligns better

        if is_left_group:
            tx = xi - text_margin - tw
        else:
            tx = xi + text_margin

        # clamp inside image
        tx = max(0, min(W - 1, tx))
        ty = max(th + 1, min(H - 1, ty))

        # outline + white text
        cv2.putText(
            img, txt, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA
        )
        cv2.putText(
            img, txt, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    ok = cv2.imwrite(save_path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")
    return save_path

def draw_named_keypoints_on_image(
    image_path,
    kp_dict,
    save_path="res.jpg",
    mask_path=None,          # 保留接口，但不使用（无论如何都画绿色）
    radius=8,
    font_scale=0.5,
    thickness=2,
    text_margin=8,           # 文字离点的水平距离（正左右）
):
    """
    Args:
        image_path: str, input jpg path
        kp_dict: dict
            { idx: {"xy":[x,y], ...}, ... }  # name 不用，直接画 idx
        save_path: str
        mask_path: 保留但忽略（不读 mask；全部绿色）
    Rule:
        - 点：全绿色
        - 文字：正左右（同一水平线），并且左右划分按“点的数量”中位数 x（median_x）来分：
            x < median_x -> 文字在左边
            x >= median_x -> 文字在右边
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    H, W = img.shape[:2]

    # -------- collect valid points for median split --------
    xs = []
    for idx, v in kp_dict.items():
        xy = v.get("xy", None) if isinstance(v, dict) else None
        if xy is None or len(xy) < 2:
            continue
        x, y = float(xy[0]), float(xy[1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            xs.append(x)

    median_x = float(np.median(xs)) if len(xs) > 0 else (W * 0.5)

    # helper: text size
    def get_text_wh(text: str):
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        return tw, th, baseline

    # -------- draw --------
    for idx, v in kp_dict.items():
        xy = v.get("xy", None) if isinstance(v, dict) else None
        if xy is None or len(xy) < 2:
            continue

        x, y = float(xy[0]), float(xy[1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            continue

        # always green
        color = (0, 255, 0)

        # draw point
        cv2.circle(img, (xi, yi), radius, color, -1)

        # text: strict left/right, decided by median_x of all points
        txt = str(idx)
        tw, th, baseline = get_text_wh(txt)

        is_left_group = float(x) < median_x

        # "正左右": keep horizontal; OpenCV uses baseline, so shift a bit for visual centering
        ty = yi + int(th * 0.35)

        if is_left_group:
            tx = xi - text_margin - tw
        else:
            tx = xi + text_margin

        # clamp
        tx = max(0, min(W - 1, tx))
        ty = max(th + 1, min(H - 1, ty))

        # outline + white text
        cv2.putText(
            img, txt, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA
        )
        cv2.putText(
            img, txt, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    ok = cv2.imwrite(save_path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")
    return save_path


import numpy as np

def save_keypoints_attention_map(
    kp_dict: dict,
    save_path: str,
    H: int = 480,
    W: int = 854,
    sigma: float = 18.0,          # 白色扩散半径（像素）
    key_ids=None,                # None=用 dict 里所有点；或传 KEY_BODY
):
    """
    从 kp_dict 生成 attention / 水平集风格 map，并直接保存为图片。
    - 白色大，黑色小
    - 输出尺寸: (H, W)
    - 保存为 8-bit 灰度图
    """
    import numpy as np
    import cv2

    if not isinstance(kp_dict, dict):
        raise TypeError(f"kp_dict must be dict, got {type(kp_dict)}")

    # -------------------------
    # collect keypoints
    # -------------------------
    pts = []
    if key_ids is None:
        items = kp_dict.items()
    else:
        items = ((i, kp_dict.get(i, None)) for i in key_ids)

    for idx, v in items:
        if not isinstance(v, dict):
            continue
        xy = v.get("xy", None)
        if xy is None or len(xy) < 2:
            continue

        x, y = float(xy[0]), float(xy[1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        pts.append((x, y))

    # 没点：直接存全黑
    if len(pts) == 0:
        blank = np.zeros((H, W), dtype=np.uint8)
        cv2.imwrite(save_path, blank)
        return save_path

    # -------------------------
    # coordinate grid
    # -------------------------
    ys = np.arange(H, dtype=np.float32)[:, None]   # (H,1)
    xs = np.arange(W, dtype=np.float32)[None, :]   # (1,W)

    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    att = np.zeros((H, W), dtype=np.float32)

    # -------------------------
    # build attention (max of Gaussians)
    # -------------------------
    for (x, y) in pts:
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        g = np.exp(-d2 * inv2s2).astype(np.float32)
        att = np.maximum(att, g)

    # -------------------------
    # normalize & save
    # -------------------------
    att = np.clip(att, 0.0, 1.0)
    att_u8 = (att * 255.0).astype(np.uint8)

    ok = cv2.imwrite(save_path, att_u8)
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")

    return save_path


import numpy as np
from typing import Dict, Any, Tuple, List, Optional

KEY_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]

SKELETON_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (9, 10),
    (9, 11), (11, 13),
    (10, 12), (12, 14),
    # wrist edges (optional, since you include 41/62)
    (6, 41), (5, 62),
]

def _point_to_segment_distance(
    xx: np.ndarray, yy: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    eps: float = 1e-8,
) -> np.ndarray:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    vx, vy = x1 - x0, y1 - y0
    denom = vx * vx + vy * vy + eps

    t = ((xx - x0) * vx + (yy - y0) * vy) / denom
    t = np.clip(t, 0.0, 1.0)

    projx = x0 + t * vx
    projy = y0 + t * vy
    dx = xx - projx
    dy = yy - projy
    return np.sqrt(dx * dx + dy * dy)

def _build_level_set_phi(
    kp_dict: Dict[int, Any],
    img_hw: Tuple[int, int],
    point_radius: float = 12.0,
    limb_radius: float = 14.0,
    edges: List[Tuple[int, int]] = SKELETON_EDGES,
    key_ids: List[int] = KEY_BODY,   # ✅只用这批点来构建 phi
) -> np.ndarray:
    """
    完全参考你的模板：
      phi = min( min_i (d(x, kp_i) - point_radius),
                 min_(i,j) (d(x, seg(kp_i,kp_j)) - limb_radius) )
    """
    H, W = img_hw
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    phi = np.full((H, W), np.inf, dtype=np.float32)

    def get_xy(idx: int) -> Optional[Tuple[float, float]]:
        v = kp_dict.get(idx, None)
        if not isinstance(v, dict):
            return None
        xy = v.get("xy", None)
        if xy is None or len(xy) < 2:
            return None
        x, y = float(xy[0]), float(xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (x, y)

    # 1) point terms (只遍历 key_ids)
    for idx in key_ids:
        p = get_xy(int(idx))
        if p is None:
            continue
        x, y = p
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        phi = np.minimum(phi, d - float(point_radius))

    # 2) limb segment terms
    for i, j in edges:
        if (i not in key_ids) or (j not in key_ids):
            continue
        p0 = get_xy(int(i))
        p1 = get_xy(int(j))
        if p0 is None or p1 is None:
            continue
        d = _point_to_segment_distance(xx, yy, p0, p1)
        phi = np.minimum(phi, d - float(limb_radius))

    return phi

def build_levelset_attention_map(
    kp_dict: Dict[int, Any],
    img_hw: Tuple[int, int] = (480, 854),
    point_radius: float = 12.0,
    limb_radius: float = 14.0,
    edges: List[Tuple[int, int]] = SKELETON_EDGES,
    key_ids: List[int] = KEY_BODY,
    sigma: float = 7.0,   # ✅边缘软化：越小越“硬/细”，越大越“糊/厚”
) -> np.ndarray:
    """
    attention（白大黑小）：
      inside(phi<=0) = 1
      outside        = exp(-(phi^2)/(2*sigma^2))
    """
    phi = _build_level_set_phi(
        kp_dict=kp_dict,
        img_hw=img_hw,
        point_radius=point_radius,
        limb_radius=limb_radius,
        edges=edges,
        key_ids=key_ids,
    )
    d_out = np.maximum(phi, 0.0).astype(np.float32)
    inv2s2 = 1.0 / (2.0 * float(sigma) * float(sigma) + 1e-8)
    att = np.exp(-(d_out * d_out) * inv2s2).astype(np.float32)
    return np.clip(att, 0.0, 1.0)

def save_levelset_attention_map(
    kp_dict: Dict[int, Any],
    save_path: str = "att_levelset.png",
    img_hw: Tuple[int, int] = (480, 854),
    point_radius: float = 12.0,
    limb_radius: float = 14.0,
    sigma: float = 7.0,
) -> str:
    import cv2
    att = build_levelset_attention_map(
        kp_dict=kp_dict,
        img_hw=img_hw,
        point_radius=point_radius,
        limb_radius=limb_radius,
        sigma=sigma,
    )
    att_u8 = (att * 255.0).astype(np.uint8)
    ok = cv2.imwrite(save_path, att_u8)
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")
    return save_path
