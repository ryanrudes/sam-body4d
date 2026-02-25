from __future__ import annotations

import os
from typing import Optional

import numpy as np


def points_on_mask(
    mask_1hw: np.ndarray,
    pts_xy_70x2: np.ndarray,
    *,
    dilate_px: int = 10,
    save_path: Optional[str] = None,
    point_radius: int = 3,
    draw_index: bool = True,
) -> np.ndarray:
    """
    Args:
        mask_1hw: (1,H,W) or (H,W) np, 255=fg, 0=bg (or any >0 as fg)
        pts_xy_70x2: (N,2) xy (float/int)
        dilate_px: dilate radius in pixels (10 -> tolerant)
        save_path: if given -> save visualization
        point_radius: circle radius
        draw_index: draw index text

    Returns:
        (N,1) np, 1=in (dilated) mask, 0=out
    """
    # ---- normalize mask ----
    if mask_1hw.ndim == 3:
        mask = mask_1hw[0]
    else:
        mask = mask_1hw
    if mask.ndim != 2:
        raise ValueError(f"mask must be (H,W) or (1,H,W), got {mask_1hw.shape}")

    H, W = mask.shape

    pts = np.asarray(pts_xy_70x2)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"pts must be (N,2[+]), got {pts.shape}")

    pts_int = np.rint(pts[:, :2]).astype(int)

    # ---- dilate mask (tolerant) ----
    mask_to_check = mask
    if dilate_px and dilate_px > 0:
        import cv2

        mask_bool = (mask > 0).astype(np.uint8) * 255
        k = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_dil = cv2.dilate(mask_bool, kernel, iterations=1)
        mask_to_check = mask_dil

    inside = np.zeros((pts_int.shape[0], 1), dtype=np.uint8)

    # ---- check ----
    for i, (x, y) in enumerate(pts_int):
        if 0 <= x < W and 0 <= y < H and mask_to_check[y, x] > 0:
            inside[i, 0] = 1

    # ---- optional draw ----
    if save_path is not None:
        import cv2

        # show the dilated mask you actually used for checking
        vis_mask = (mask_to_check > 0).astype(np.uint8) * 255
        vis = np.stack([vis_mask, vis_mask, vis_mask], axis=-1)

        for i, (x, y) in enumerate(pts_int):
            if not (0 <= x < W and 0 <= y < H):
                color = (0, 0, 255)  # red
            else:
                color = (0, 255, 0) if inside[i, 0] == 1 else (0, 0, 255)

            cv2.circle(vis, (x, y), point_radius, color, -1)

            if draw_index:
                cv2.putText(
                    vis,
                    str(i),
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, vis)

    return inside


# import numpy as np

SMPL21_TO_70 = {
    0: [9],
    1: [10],
    2: [67, 68],
    3: [11],
    4: [12],
    5: [69],
    6: [13],
    7: [14],
    8: [69],
    9: [15,16,17],
    10:[18,19,20],
    11:[69],
    12:[67],
    13:[68],
    14:[0],
    15:[5],
    16:[6],
    17:[7],
    18:[8],
    19:[62],
    20:[41],
}

def smpl21_missing_from70(vis70: np.ndarray):
    """
    vis70: (70,) or (70,1)  1 visible, 0 missing
    return: np.array of missing smpl21 indices
    """
    vis70 = np.asarray(vis70).reshape(-1)

    if vis70.shape[0] != 70:
        raise ValueError("vis70 must have length 70")

    missing = []

    for smpl_idx, kp_list in SMPL21_TO_70.items():
        if any(vis70[k] == 0 for k in kp_list):
            missing.append(smpl_idx)

    return np.array(missing, dtype=int)
