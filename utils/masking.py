from painter import mask_painter
from PIL import Image
import os
import cv2
import numpy as np
from PIL import Image

import os
import cv2
import numpy as np

def paste_masked_region_from_a_to_b_np_mask(
    mask_hw: np.ndarray,          # H x W, uint8, 255 fg / 0 bg
    image_a_path: str,
    image_b_path: str,
    suffix: str = "_paste_from_a",
) -> str:
    """
    Copy pixels from image A to image B where mask == 255.

    Args:
        mask_hw: np.ndarray (H, W), uint8, foreground=255, background=0
        image_a_path: source image A
        image_b_path: target image B
        suffix: output filename suffix

    Returns:
        out_path: saved image path
    """

    # -------------------------
    # validate mask
    # -------------------------
    if not isinstance(mask_hw, np.ndarray):
        raise TypeError("mask_hw must be np.ndarray")
    if mask_hw.ndim != 2:
        raise ValueError(f"mask_hw must be HxW, got shape {mask_hw.shape}")
    if mask_hw.dtype != np.uint8:
        raise ValueError(f"mask_hw must be uint8, got {mask_hw.dtype}")

    fg = (mask_hw == 255)

    # -------------------------
    # load images
    # -------------------------
    img_a = cv2.imread(image_a_path, cv2.IMREAD_COLOR)
    img_b = cv2.imread(image_b_path, cv2.IMREAD_COLOR)

    if img_a is None:
        raise FileNotFoundError(image_a_path)
    if img_b is None:
        raise FileNotFoundError(image_b_path)

    if img_a.shape[:2] != mask_hw.shape:
        raise ValueError(f"Shape mismatch: img_a={img_a.shape[:2]} vs mask={mask_hw.shape}")
    if img_b.shape[:2] != mask_hw.shape:
        raise ValueError(f"Shape mismatch: img_b={img_b.shape[:2]} vs mask={mask_hw.shape}")

    # -------------------------
    # paste A -> B
    # -------------------------
    out = img_b.copy()
    out[fg] = img_a[fg]

    # -------------------------
    # save to image A directory
    # -------------------------
    a_dir = os.path.dirname(image_a_path) or "."
    a_base = os.path.splitext(os.path.basename(image_a_path))[0]
    out_path = os.path.join(a_dir, f"{a_base}{suffix}.jpg")

    cv2.imwrite(out_path, out)
    return out_path


init_loc = "/root/projects/sam-body4d/outputs/20260122_121732_046_079b639b/completion/21DT/images/00000015.jpg"
init_msk = "/root/projects/sam-body4d/outputs/20260122_121732_046_079b639b/completion/21DT/masks/00000015.png"
targ_loc = "/root/projects/sam-body4d/outputs/20260122_121732_046_079b639b/images/00000079.jpg"

mask = np.array(Image.open(init_msk).convert("P"))*255
loc = paste_masked_region_from_a_to_b_np_mask(
    mask,
    init_loc,
    targ_loc,
)

frame = np.array(Image.open(loc).convert("RGB"))


painted_image = mask_painter(np.array(frame, dtype=np.uint8), mask, mask_color=4+1)
zero_mask = np.zeros_like(mask)
zero_mask[mask==0] = 255
painted_image = mask_painter(np.array(frame, dtype=np.uint8), zero_mask, mask_color=1, mask_alpha=0.5)

# for k, v in RUNTIME['masks'].items():
#     if k == RUNTIME['id']:
#         continue
#     if frame_idx in v:
#         mask = v[frame_idx]
#         painted_image = mask_painter(painted_image, mask, mask_color=4+k)

frame = Image.fromarray(frame)

frame.save("xx.jpg")