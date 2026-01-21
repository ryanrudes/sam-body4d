import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2


# -------------------------
# Skeleton (for densifying TPS constraints)
# -------------------------
# Indices follow your KEY_POINT_NAME map:
# 0~14: COCO body, 41: right_wrist, 62: left_wrist, 69: neck (if present)
SKELETON_EDGES = [
    (5, 6),    # Lshoulder - Rshoulder
    (5, 9),    # Lshoulder - Lhip
    (6, 10),   # Rshoulder - Rhip
    (9, 10),   # Lhip - Rhip
    (5, 7),    # Lshoulder - Lelbow
    (7, 62),   # Lelbow - Lwrist
    (6, 8),    # Rshoulder - Relbow
    (8, 41),   # Relbow - Rwrist
    (9, 11),   # Lhip - Lknee
    (11, 13),  # Lknee - Lankle
    (10, 12),  # Rhip - Rknee
    (12, 14),  # Rknee - Rankle
    (0, 69),   # nose - neck (if 69 exists in your dict)
    (69, 5),   # neck - Lshoulder
    (69, 6),   # neck - Rshoulder
]


# -------------------------
# Mask I/O (robust for PNG "P" / palette)
# -------------------------
def _read_obj_mask(mask_path: str, obj_id: int) -> np.ndarray:
    """
    Robustly read a label mask PNG, including palette (mode 'P').

    Returns:
        fg: (H,W) uint8 {0,1} where label == obj_id
    """
    # Prefer PIL for palette PNGs
    try:
        from PIL import Image  # python -m pip install pillow (if needed)
        img = Image.open(mask_path)
        label = np.array(img)  # for 'P' this is palette indices (labels)

        # If somehow RGB/RGBA, this is almost surely NOT raw ids, but handle anyway.
        if label.ndim == 3:
            label = label[..., 0]

        fg = (label.astype(np.int32) == int(obj_id)).astype(np.uint8)
        return fg
    except Exception:
        # Fallback: OpenCV (works when file is true single-channel label map)
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(mask_path)
        if m.ndim == 3:
            m = m[..., 0]
        fg = (m.astype(np.int32) == int(obj_id)).astype(np.uint8)
        return fg


def _write_obj_mask_png(
    mask_bin: np.ndarray,
    obj_id: int,
    out_path: str,
    keep_palette_from: Optional[str] = None,
) -> None:
    """
    Write a label mask PNG: FG=obj_id, BG=0.

    If keep_palette_from is provided and is a palette PNG, we will save as 'P' mode
    and copy its palette. Otherwise we save a normal single-channel PNG.
    """
    label = (mask_bin > 0).astype(np.uint8) * int(obj_id)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if keep_palette_from is not None:
        try:
            from PIL import Image
            ref = Image.open(keep_palette_from)
            if ref.mode == "P" and ref.getpalette() is not None:
                out = Image.fromarray(label.astype(np.uint8), mode="P")
                out.putpalette(ref.getpalette())
                out.save(out_path)
                return
        except Exception:
            pass

    # fallback: write as single-channel
    cv2.imwrite(out_path, label)


# -------------------------
# Keypoints helpers
# -------------------------
def _kpdict_to_xy(
    kp_dict: Dict[int, Any],
    ids: Optional[List[int]] = None,
    require_finite: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Convert {idx: {"xy":[x,y], ...}} to (N,2) float32 + used_ids.
    Filters out missing / NaN / inf points.
    """
    if ids is None:
        ids = sorted(list(kp_dict.keys()))

    pts, used = [], []
    for k in ids:
        if k not in kp_dict:
            continue
        xy = kp_dict[k].get("xy", None)
        if xy is None:
            continue
        x, y = float(xy[0]), float(xy[1])
        if require_finite and (not np.isfinite(x) or not np.isfinite(y)):
            continue
        pts.append([x, y])
        used.append(k)

    if len(pts) == 0:
        return np.zeros((0, 2), np.float32), []
    return np.asarray(pts, np.float32), used


def _keep_paired_points(
    high_kp_dict: Dict[int, Any],
    low_kp_dict: Dict[int, Any],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Take intersection keys and return paired (src_pts, dst_pts) in identical order.
    """
    common_ids = sorted(set(high_kp_dict.keys()) & set(low_kp_dict.keys()))
    if not common_ids:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), []

    src_pts, used1 = _kpdict_to_xy(high_kp_dict, ids=common_ids)
    dst_pts, used2 = _kpdict_to_xy(low_kp_dict, ids=common_ids)

    used = sorted(set(used1) & set(used2))
    if not used:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), []

    src_pts, _ = _kpdict_to_xy(high_kp_dict, ids=used)
    dst_pts, _ = _kpdict_to_xy(low_kp_dict, ids=used)
    return src_pts, dst_pts, used


def densify_skeleton_points_from_kpdict(
    high_kp_dict: Dict[int, Any],
    low_kp_dict: Dict[int, Any],
    edges: List[Tuple[int, int]] = SKELETON_EDGES,
    num_interp: int = 3,          # per edge
    add_original: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Use skeleton edges to insert interpolated correspondence points for TPS.

    Returns:
        src_pts: (N,2) float32 in high frame
        dst_pts: (N,2) float32 in low frame
        used_edges: edges that were actually used (both endpoints existed & finite)
    """

    def get_xy(d: Dict[int, Any], idx: int) -> Optional[np.ndarray]:
        if idx not in d:
            return None
        xy = d[idx].get("xy", None)
        if xy is None:
            return None
        x, y = float(xy[0]), float(xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return np.array([x, y], dtype=np.float32)

    src_pts: List[np.ndarray] = []
    dst_pts: List[np.ndarray] = []
    used_edges: List[Tuple[int, int]] = []

    # 1) original paired points (intersection)
    if add_original:
        common_ids = sorted(set(high_kp_dict.keys()) & set(low_kp_dict.keys()))
        for k in common_ids:
            a = get_xy(high_kp_dict, k)
            b = get_xy(low_kp_dict, k)
            if a is None or b is None:
                continue
            src_pts.append(a)
            dst_pts.append(b)

    # 2) interpolated points along edges
    for i, j in edges:
        a0 = get_xy(high_kp_dict, i)
        a1 = get_xy(high_kp_dict, j)
        b0 = get_xy(low_kp_dict, i)
        b1 = get_xy(low_kp_dict, j)
        if a0 is None or a1 is None or b0 is None or b1 is None:
            continue

        used_edges.append((i, j))

        # uniform interpolation excluding endpoints
        for t in range(1, num_interp + 1):
            alpha = t / (num_interp + 1.0)
            src_pts.append((1 - alpha) * a0 + alpha * a1)
            dst_pts.append((1 - alpha) * b0 + alpha * b1)

    if len(src_pts) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), []

    return np.stack(src_pts).astype(np.float32), np.stack(dst_pts).astype(np.float32), used_edges


# -------------------------
# Transform estimators
# -------------------------
def _estimate_translation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Use mean translation. Return 2x3 matrix for cv2.warpAffine."""
    t = (dst - src).mean(axis=0)  # (2,)
    return np.array(
        [[1.0, 0.0, float(t[0])],
         [0.0, 1.0, float(t[1])]],
        dtype=np.float32
    )


def _estimate_affine_partial(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """
    Robust similarity/partial-affine (rotation+scale+translation, no shear).
    Needs >=3 points for stable estimation.
    """
    if src.shape[0] < 3:
        return None
    M, _ = cv2.estimateAffinePartial2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    return M


def _tps_warp_mask(
    mask: np.ndarray,           # HxW uint8 {0,1}
    src_pts: np.ndarray,        # Nx2 float32 (high frame)
    dst_pts: np.ndarray,        # Nx2 float32 (low frame)
    out_hw: Tuple[int, int],    # (H_out, W_out)
    border_value: int = 0,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """
    Thin-Plate Spline warp using OpenCV createThinPlateSplineShapeTransformer.
    Learn mapping src -> dst, then warp mask.
    """
    H_out, W_out = out_hw
    if src_pts.shape[0] < 3:
        raise ValueError("TPS needs >=3 points (>=5 recommended).")

    tps = cv2.createThinPlateSplineShapeTransformer()
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(src_pts.shape[0])]

    src_shape = src_pts.reshape(-1, 1, 2)
    dst_shape = dst_pts.reshape(-1, 1, 2)

    # estimate src -> dst
    tps.estimateTransformation(src_shape, dst_shape, matches)

    warped = tps.warpImage(mask.astype(np.uint8), borderValue=border_value)

    # warpImage keeps source size; force exact output size
    if warped.shape[:2] != (H_out, W_out):
        warped = cv2.resize(warped, (W_out, H_out), interpolation=interpolation)

    return (warped > 0).astype(np.uint8)


# -------------------------
# Post-processing
# -------------------------
def _largest_connected_component(mask_bin: np.ndarray) -> np.ndarray:
    """Keep only the largest foreground connected component."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask_bin.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return (labels == best).astype(np.uint8)


# -------------------------
# Main API
# -------------------------
def warp_high_quality_mask_to_low_frame(
    high_mask_path: str,
    low_mask_path: str,
    obj_id: int,
    high_kp_dict: Dict[int, Any],
    low_kp_dict: Dict[int, Any],
    out_mask_path: Optional[str] = None,
    prefer: str = "tps",                 # "tps" or "affine"
    min_points_tps: int = 10,            # after densify, recommend >=10
    densify: bool = True,
    densify_num_interp: int = 3,         # per edge
    densify_add_original: bool = True,
    dilate_after: int = 0,
    erode_after: int = 0,
    keep_largest_cc: bool = True,
    fail_behavior: str = "raise",        # "raise" | "return_low"
    keep_palette: bool = True,           # if True, save out as palette 'P' if low_mask is palette
) -> np.ndarray:
    """
    Warp HQ mask into LQ frame coordinate system using keypoint-driven deformation,
    with optional skeleton densification for stronger non-rigid TPS.

    Robust to missing keypoints:
      - N>=min_points_tps: TPS
      - 3<=N<min_points_tps: partial affine (similarity)
      - N==2: translation
      - N<2: fail_behavior

    Returns:
        warped_mask_bin: (H_low, W_low) uint8 {0,1}
    """
    high_bin = _read_obj_mask(high_mask_path, obj_id)  # Hh x Wh
    low_bin  = _read_obj_mask(low_mask_path, obj_id)   # Hl x Wl (target size)
    H_low, W_low = low_bin.shape[:2]

    # points
    if densify:
        src_pts, dst_pts, _ = densify_skeleton_points_from_kpdict(
            high_kp_dict, low_kp_dict,
            edges=SKELETON_EDGES,
            num_interp=densify_num_interp,
            add_original=densify_add_original,
        )
        N = int(src_pts.shape[0])
    else:
        src_pts, dst_pts, _ = _keep_paired_points(high_kp_dict, low_kp_dict)
        N = int(src_pts.shape[0])

    if N < 2:
        if fail_behavior == "return_low":
            warped = low_bin.astype(np.uint8)
        else:
            raise ValueError(f"Not enough valid common keypoints to warp: N={N}")
    else:
        prefer_ = (prefer or "tps").lower()
        warped = None

        # 1) TPS
        if prefer_ == "tps" and N >= min_points_tps:
            try:
                warped = _tps_warp_mask(
                    mask=high_bin,
                    src_pts=src_pts,
                    dst_pts=dst_pts,
                    out_hw=(H_low, W_low),
                    border_value=0,
                    interpolation=cv2.INTER_NEAREST,
                )
            except Exception:
                warped = None

        # 2) partial affine (similarity)
        if warped is None and N >= 3:
            M = _estimate_affine_partial(src_pts, dst_pts)
            if M is not None:
                warped = cv2.warpAffine(
                    high_bin.astype(np.uint8),
                    M,
                    (W_low, H_low),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                warped = (warped > 0).astype(np.uint8)

        # 3) translation
        if warped is None and N >= 2:
            M = _estimate_translation(src_pts, dst_pts)
            warped = cv2.warpAffine(
                high_bin.astype(np.uint8),
                M,
                (W_low, H_low),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped = (warped > 0).astype(np.uint8)

        if warped is None:
            if fail_behavior == "return_low":
                warped = low_bin.astype(np.uint8)
            else:
                raise ValueError("Warp failed (TPS/affine/translation all failed).")

    # morphology
    if dilate_after > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_after + 1, 2 * dilate_after + 1))
        warped = cv2.dilate(warped, k, iterations=1)
    if erode_after > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_after + 1, 2 * erode_after + 1))
        warped = cv2.erode(warped, k, iterations=1)

    if keep_largest_cc:
        warped = _largest_connected_component(warped)

    if out_mask_path is not None:
        keep_palette_from = low_mask_path if keep_palette else None
        _write_obj_mask_png(warped, obj_id, out_mask_path, keep_palette_from=keep_palette_from)

    return warped


# -------------------------
# Optional: your helper (same behavior)
# -------------------------
def build_body_keypoint_dict(points_70x2, KEY_BODY: List[int], KEY_POINT_NAME: Dict[int, str]) -> Dict[int, Any]:
    """
    Args:
        points_70x2: array-like, shape (70, 2)
        KEY_BODY: list[int]
        KEY_POINT_NAME: dict[int, str]
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


import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2

# Same skeleton as before (works even if some points missing)
SKELETON_EDGES = [
    (5, 6), (5, 9), (6, 10), (9, 10),
    (5, 7), (7, 62), (6, 8), (8, 41),
    (9, 11), (11, 13), (10, 12), (12, 14),
    (0, 69), (69, 5), (69, 6),
]


def _point_to_segment_distance(xx: np.ndarray, yy: np.ndarray, p0: Tuple[float, float], p1: Tuple[float, float]) -> np.ndarray:
    """
    xx,yy: HxW meshgrid (float32)
    p0,p1: (x,y)
    returns: HxW distances to segment
    """
    x0, y0 = p0
    x1, y1 = p1
    vx = x1 - x0
    vy = y1 - y0
    wx = xx - x0
    wy = yy - y0

    c1 = vx * wx + vy * wy
    c2 = vx * vx + vy * vy
    t = np.clip(c1 / (c2 + 1e-6), 0.0, 1.0)

    projx = x0 + t * vx
    projy = y0 + t * vy
    return np.sqrt((xx - projx) ** 2 + (yy - projy) ** 2)


def _build_level_set_phi(
    kp_dict: Dict[int, Any],
    img_hw: Tuple[int, int],
    point_radius: float = 16.0,
    limb_radius: float = 20.0,
    edges: List[Tuple[int, int]] = SKELETON_EDGES,
) -> np.ndarray:
    """
    Build a level-set function phi(x,y) from keypoints/skeleton:
      phi = min( min_i (d(x, kp_i) - point_radius),
                 min_(i,j) (d(x, seg(kp_i,kp_j)) - limb_radius) )

    phi <= 0 : inside (foreground region)
    phi > 0  : outside
    """
    H, W = img_hw
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    phi = np.full((H, W), np.inf, dtype=np.float32)

    def get_xy(idx: int) -> Optional[Tuple[float, float]]:
        v = kp_dict.get(idx, None)
        if v is None:
            return None
        xy = v.get("xy", None)
        if xy is None:
            return None
        x, y = float(xy[0]), float(xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (x, y)

    # 1) point terms
    for idx in kp_dict.keys():
        p = get_xy(idx)
        if p is None:
            continue
        x, y = p
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        phi = np.minimum(phi, d - float(point_radius))

    # 2) limb segment terms
    for i, j in edges:
        p0 = get_xy(i)
        p1 = get_xy(j)
        if p0 is None or p1 is None:
            continue
        d = _point_to_segment_distance(xx, yy, p0, p1)
        phi = np.minimum(phi, d - float(limb_radius))

    return phi


def keep_levelset_inside_to_white_outside(
    img_path: str,
    kp_dict: Dict[int, Any],
    out_path: Optional[str] = None,
    # level-set geometry
    point_radius: float = 16.0,
    limb_radius: float = 20.0,
    edges: List[Tuple[int, int]] = SKELETON_EDGES,
    # feather / gradient control (in pixels)
    feather: float = 20.0,        # 0 -> hard cut; larger -> softer transition
    gamma: float = 1.0,           # >1 makes edge whiter faster; <1 keeps more
    bg_color_bgr: Tuple[int, int, int] = (255, 255, 255),
) -> str:
    """
    Build a level-set from kp_dict on the image coordinate system.
    Keep inside region (phi<=0), make outside white.
    Edge can be feathered to white using a smooth alpha based on phi.

    Returns: out_path (jpg)
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    H, W = img.shape[:2]

    # build phi on this image size
    phi = _build_level_set_phi(
        kp_dict=kp_dict,
        img_hw=(H, W),
        point_radius=point_radius,
        limb_radius=limb_radius,
        edges=edges,
    )

    # alpha: 1 inside, 0 outside, smooth around boundary
    # We use a simple ramp based on phi:
    #   phi<=0 -> alpha=1
    #   phi>=feather -> alpha=0
    #   between -> linear then gamma
    if feather <= 0:
        alpha = (phi <= 0).astype(np.float32)
    else:
        a = 1.0 - np.clip(phi / float(feather), 0.0, 1.0)  # inside(<=0)->1, far outside->0
        if gamma != 1.0:
            a = np.power(a, float(gamma))
        alpha = a.astype(np.float32)

    # blend: out = alpha*img + (1-alpha)*white
    bg = np.empty_like(img, dtype=np.float32)
    bg[...] = np.array(bg_color_bgr, dtype=np.float32)
    out = alpha[..., None] * img.astype(np.float32) + (1.0 - alpha[..., None]) * bg
    out = np.clip(out, 0, 255).astype(np.uint8)

    # write jpg
    if out_path is None:
        base, _ = os.path.splitext(img_path)
        out_path = base + "_levelset_keep.jpg"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, out)
    return out_path
