

KEY_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]  # key body joints for prompting

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

import torch
import torch.nn.functional as F
def find_topk_similar_points_on_a(
    a: torch.Tensor,                 # (1, 256, 72, 72)
    b: torch.Tensor,                 # (1, 256, 72, 72)
    point_dict: dict,                # {idx: {"xy":[x,y], ...}, ...} 这些点在 b 的 480x854 坐标系里
    H: int = 480,
    W: int = 854,
    feat_h: int = 72,
    feat_w: int = 72,
    topk: int = 3,
    eps: float = 1e-8,
):
    """
    对 b 的每个关键点：
      1) 用 (H,W) 把 (x,y) 映射到 (feat_h, feat_w) 特征图上取出 b 的 256D 特征向量
      2) 在 a 的所有 (feat_h*feat_w) 位置上做 cosine 相似度
      3) 取 topk 最相近位置，映射回 (H,W) 坐标输出

    Returns:
        out: dict
            {
              idx: {
                "b_xy": [x, y],                         # 原始 b 上关键点（480x854）
                "a_topk_xy": [[x1,y1],[x2,y2],[x3,y3]]  # a 上 topk 点（480x854）
              }, ...
            }
    """
    assert a.ndim == 4 and b.ndim == 4 and a.shape == b.shape, "a and b must have same shape (1,C,Hf,Wf)"
    assert a.shape[0] == 1, "batch must be 1"
    C = a.shape[1]
    assert C == 256, f"expected C=256, got {C}"
    assert a.shape[2] == feat_h and a.shape[3] == feat_w, f"expected (feat_h,feat_w)=({feat_h},{feat_w})"
    assert topk >= 1

    device = a.device
    dtype = a.dtype

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

