# kp_warp_rgbmarker.py
import numpy as np
import cv2

def _make_spread_colors_rgb(K: int = 17, seed: int = 0) -> np.ndarray:
    """
    生成分得很开的 RGB 方向向量（单位向量），用于 cosine 匹配。
    用 fibonacci sphere 先生成 3D 单位向量，再映射到 [0,1]，再归一化回单位向量用于匹配。
    """
    rng = np.random.RandomState(seed)
    # Fibonacci sphere on unit sphere
    i = np.arange(K, dtype=np.float32)
    phi = (1 + 5 ** 0.5) * 0.5
    theta = 2 * np.pi * i / phi
    z = 1 - (2 * (i + 0.5) / K)
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    v = np.stack([x, y, z], axis=1).astype(np.float32)

    # 映射到 [0,1] 做 marker 颜色
    rgb = (v + 1) * 0.5
    # 避免过暗：抬一下底
    rgb = 0.15 + 0.85 * rgb
    # 用于匹配时我们用方向（单位向量）
    rgb_unit = rgb / (np.linalg.norm(rgb, axis=1, keepdims=True) + 1e-12)
    return rgb.astype(np.float32), rgb_unit.astype(np.float32)

def _draw_gaussian_rgb(marker: np.ndarray, x: float, y: float, color_rgb: np.ndarray,
                      sigma: float = 2.0, radius: int = 8):
    """
    marker: (H,W,3) float32 0..1
    在 (x,y) 画一个彩色高斯团（max 合成），抗插值
    """
    H, W = marker.shape[:2]
    xi, yi = int(round(float(x))), int(round(float(y)))
    if xi < 0 or xi >= W or yi < 0 or yi >= H:
        return

    x1 = max(0, xi - radius); x2 = min(W, xi + radius + 1)
    y1 = max(0, yi - radius); y2 = min(H, yi + radius + 1)
    xs = np.arange(x1, x2, dtype=np.float32)
    ys = np.arange(y1, y2, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    g = g[..., None]  # (h,w,1)

    patch = marker[y1:y2, x1:x2, :]
    colored = g * color_rgb[None, None, :]
    marker[y1:y2, x1:x2, :] = np.maximum(patch, colored)

def warp_keypoints_like_image_rgbmarker(
    img: np.ndarray,
    keypoints_xy: np.ndarray,
    warp_mat: np.ndarray,
    out_w: int,
    out_h: int,
    sigma: float = 2.0,
    radius: int = 8,
    window: int = 9,
    min_conf: float = 0.05,
    seed: int = 0,
    return_debug: bool = False,
):
    """
    用 RGB marker（高斯团）随图 warpAffine，再用 cosine 相似度“找最接近的颜色”回读 keypoints。

    Args:
        img: 原图，仅用来取 H,W
        keypoints_xy: (K,2) 原图像素坐标 (x,y)，K=17
        warp_mat: 与图像 warpAffine 同一个 2x3
        out_w,out_h: 输出尺寸（与你的 input_size 一致）
        sigma,radius: marker 点的形状（radius 建议 6~10）
        window: 峰值附近做质心的窗口（奇数）
        min_conf: 置信度太低也仍返回 argmax，但你可以用 conf 做过滤
        return_debug: True 时返回 marker_warp 可视化

    Returns:
        kp_warp: (K,2) float32
        conf: (K,) float32 0..1（越大越可靠）
        (optional) marker_warp_rgb_u8: (out_h,out_w,3) uint8 (RGB) 方便你保存看
    """
    H0, W0 = img.shape[:2]
    kps = np.asarray(keypoints_xy, dtype=np.float32).reshape(-1, 2)
    K = kps.shape[0]

    colors_rgb, colors_unit = _make_spread_colors_rgb(K, seed=seed)

    # 1) marker in RGB float32 0..1
    marker = np.zeros((H0, W0, 3), dtype=np.float32)
    for i in range(K):
        x, y = float(kps[i, 0]), float(kps[i, 1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        _draw_gaussian_rgb(marker, x, y, colors_rgb[i], sigma=sigma, radius=int(radius))

    # 2) warp marker with SAME style as image (LINEAR)
    marker_warp = cv2.warpAffine(marker, warp_mat, (int(out_w), int(out_h)), flags=cv2.INTER_LINEAR)
    marker_warp = np.clip(marker_warp, 0.0, 1.0).astype(np.float32)

    # 3) decode by cosine similarity + intensity gating
    H, W = marker_warp.shape[:2]
    v = marker_warp.reshape(-1, 3)
    v_norm = np.linalg.norm(v, axis=1) + 1e-12
    v_unit = v / v_norm[:, None]
    intensity = v_norm.reshape(H, W).astype(np.float32)  # 0..sqrt(3)

    half = window // 2
    kp_out = np.zeros((K, 2), dtype=np.float32)
    conf = np.zeros((K,), dtype=np.float32)

    for i in range(K):
        c = colors_unit[i]  # (3,)
        sim = (v_unit @ c).reshape(H, W).astype(np.float32)  # [-1,1]
        score = sim * intensity  # 抑制暗处的假匹配

        # peak
        y0, x0 = np.unravel_index(int(np.argmax(score)), score.shape)
        peak = float(score[y0, x0])
        conf[i] = max(0.0, min(1.0, peak / (np.sqrt(3) + 1e-12)))

        # local centroid (even if low conf we still compute)
        x1 = max(0, x0 - half); x2 = min(W, x0 + half + 1)
        y1 = max(0, y0 - half); y2 = min(H, y0 + half + 1)
        patch = score[y1:y2, x1:x2]
        wgt = np.maximum(patch, 0.0)

        s = float(wgt.sum())
        if s < 1e-8:
            kp_out[i] = np.array([x0, y0], dtype=np.float32)
        else:
            xs = np.arange(x1, x2, dtype=np.float32)[None, :]
            ys = np.arange(y1, y2, dtype=np.float32)[:, None]
            cx = float((wgt * xs).sum() / s)
            cy = float((wgt * ys).sum() / s)
            kp_out[i] = np.array([cx, cy], dtype=np.float32)

        # 若你想硬阈值，可在外面用 conf < min_conf 做处理
        # 这里不返回 nan，保证“找得回来”
        if conf[i] < min_conf:
            # 仍然返回，只是 conf 很低，你可在上层决定信不信
            pass

    if return_debug:
        marker_vis = (marker_warp * 255.0).astype(np.uint8)  # RGB
        return kp_out, conf, marker_vis
    return kp_out, conf

import os
import cv2
import numpy as np

def save_keypoints_with_index(
    img_bgr: np.ndarray,
    keypoints_xy: np.ndarray,
    out_path: str = "res.jpg",
    radius: int = 3,
    thickness: int = -1,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    text_offset: tuple = (4, -4),
):
    """
    Args:
        img_bgr: np.ndarray, shape (H, W, 3), dtype uint8, BGR, range [0,255]
        keypoints_xy: np.ndarray, shape (70, 2), (x, y) in pixel coords (float/int ok)
        out_path: output image path
    """
    assert isinstance(img_bgr, np.ndarray) and img_bgr.ndim == 3 and img_bgr.shape[2] == 3, \
        f"img_bgr must be HxWx3 np.ndarray, got {type(img_bgr)} {getattr(img_bgr,'shape',None)}"
    assert isinstance(keypoints_xy, np.ndarray) and keypoints_xy.shape == (70, 2), \
        f"keypoints_xy must be (70,2) np.ndarray, got {type(keypoints_xy)} {getattr(keypoints_xy,'shape',None)}"
    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

    H, W = img_bgr.shape[:2]
    vis = img_bgr.copy()

    for i, (x, y) in enumerate(keypoints_xy):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            continue

        # point
        cv2.circle(vis, (xi, yi), radius, (0, 255, 0), thickness, lineType=cv2.LINE_AA)

        # index text (with black outline for readability)
        tx = np.clip(xi + int(text_offset[0]), 0, W - 1)
        ty = np.clip(yi + int(text_offset[1]), 0, H - 1)
        label = str(i)

        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ok = cv2.imwrite(out_path, vis)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {out_path}")
    return vis
