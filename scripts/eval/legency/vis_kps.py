import os
import cv2
import numpy as np

def draw_points_with_indices(image_path: str, points_17x2: np.ndarray, out_jpg_path: str):
    """
    Args:
        image_path: input image path
        points_17x2: np.ndarray with shape (17, 2), each row is (x, y)
        out_jpg_path: output jpg path
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not isinstance(points_17x2, np.ndarray):
        raise TypeError("points_17x2 must be a np.ndarray")

    if points_17x2.ndim != 2 or points_17x2.shape != (17, 2):
        raise ValueError(f"Expected points_17x2 shape (17, 2), got {points_17x2.shape}")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    H, W = img.shape[:2]

    pts = points_17x2.astype(np.float32, copy=False)

    # Drawing params
    radius = max(4, int(round(min(H, W) * 0.007)))     # larger points
    thickness = -1                                     # filled circle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(1.2, min(H, W) / 900.0))
    text_thickness = max(1, int(round(min(H, W) / 600.0)))

    dx = int(round(radius * 1.4))
    dy = int(round(-radius * 1.4))

    for i in range(17):
        x, y = float(pts[i, 0]), float(pts[i, 1])

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        xi, yi = int(round(x)), int(round(y))

        if xi < -50 or xi > W + 50 or yi < -50 or yi > H + 50:
            continue

        xi_clamp = int(np.clip(xi, 0, W - 1))
        yi_clamp = int(np.clip(yi, 0, H - 1))

        # draw point (green)
        cv2.circle(img, (xi_clamp, yi_clamp), radius, (0, 255, 0),
                   thickness, lineType=cv2.LINE_AA)

        # draw index with outline for readability
        label = str(i)
        tx = int(np.clip(xi_clamp + dx, 0, W - 1))
        ty = int(np.clip(yi_clamp + dy, 0, H - 1))

        cv2.putText(img, label, (tx, ty), font, font_scale,
                    (0, 0, 0), thickness=max(2, text_thickness + 2),
                    lineType=cv2.LINE_AA)
        cv2.putText(img, label, (tx, ty), font, font_scale,
                    (255, 255, 255), thickness=text_thickness,
                    lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_jpg_path) or ".", exist_ok=True)
    ok = cv2.imwrite(out_jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise IOError(f"Failed to write output image: {out_jpg_path}")

    return out_jpg_path
