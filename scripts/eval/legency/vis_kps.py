import os
import cv2
import numpy as np
import torch


def draw_points_with_indices(
    image_3x512x512: torch.Tensor,
    points_17x2: np.ndarray,
    out_jpg_path: str,
):
    """
    Args:
        image_3x512x512: torch.Tensor, shape (3, 512, 512), value range [0, 1]
        points_17x2: np.ndarray with shape (17, 2), each row is (x, y) in pixel coords
        out_jpg_path: output jpg path
    """

    # -------------------------
    # checks
    # -------------------------
    if not isinstance(image_3x512x512, torch.Tensor):
        raise TypeError("image_3x512x512 must be a torch.Tensor")

    if image_3x512x512.ndim != 3 or image_3x512x512.shape[0] != 3:
        raise ValueError(
            f"Expected image shape (3, H, W), got {tuple(image_3x512x512.shape)}"
        )

    if not isinstance(points_17x2, np.ndarray):
        raise TypeError("points_17x2 must be a np.ndarray")

    if points_17x2.shape != (17, 2):
        raise ValueError(f"Expected points_17x2 shape (17, 2), got {points_17x2.shape}")

    # -------------------------
    # tensor -> numpy image
    # -------------------------
    img = image_3x512x512.detach().float().clamp(0, 1)
    img = (img * 255.0).round().to(torch.uint8)   # 3 x H x W
    img = img.permute(1, 2, 0).cpu().numpy()      # H x W x 3 (RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    # OpenCV expects BGR

    H, W = img.shape[:2]
    pts = points_17x2.astype(np.float32, copy=False)

    # -------------------------
    # drawing params
    # -------------------------
    radius = max(4, int(round(min(H, W) * 0.007)))
    thickness = -1  # filled
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(1.2, min(H, W) / 900.0))
    text_thickness = max(1, int(round(min(H, W) / 600.0)))

    dx = int(round(radius * 1.4))
    dy = int(round(-radius * 1.4))

    # -------------------------
    # draw points
    # -------------------------
    for i in range(17):
        x, y = float(pts[i, 0]), float(pts[i, 1])

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        xi, yi = int(round(x)), int(round(y))

        if xi < -50 or xi > W + 50 or yi < -50 or yi > H + 50:
            continue

        xi = int(np.clip(xi, 0, W - 1))
        yi = int(np.clip(yi, 0, H - 1))

        # point
        cv2.circle(
            img,
            (xi, yi),
            radius,
            (0, 255, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

        # index label
        label = str(i)
        tx = int(np.clip(xi + dx, 0, W - 1))
        ty = int(np.clip(yi + dy, 0, H - 1))

        # black outline
        cv2.putText(
            img,
            label,
            (tx, ty),
            font,
            font_scale,
            (0, 0, 0),
            thickness=max(2, text_thickness + 2),
            lineType=cv2.LINE_AA,
        )
        # white text
        cv2.putText(
            img,
            label,
            (tx, ty),
            font,
            font_scale,
            (255, 255, 255),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    # -------------------------
    # save
    # -------------------------
    os.makedirs(os.path.dirname(out_jpg_path) or ".", exist_ok=True)
    ok = cv2.imwrite(out_jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise IOError(f"Failed to write output image: {out_jpg_path}")

    return out_jpg_path
