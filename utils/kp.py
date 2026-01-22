from kpp_utils import build_zero_neighbor_dict, build_body_keypoint_dict, find_topk_similar_points_on_a, visualize_topk_points_on_image, KINEMATIC_EDGES, draw_named_keypoints_on_image, KEY_POINT_NAME, draw_keybody17_on_image_from_kp70, draw_named_keypoints_on_image, save_keypoints_attention_map, save_levelset_attention_map
import numpy as np
from PIL import Image

def overlay_attention_white_on_image(
    image_path: str,
    att_path: str,                 # grayscale attention image: white=fg, black=bg
    save_path: str = "overlay.jpg",
    alpha_max: float = 0.65,       # 最大叠加强度（0~1）
    threshold: int = 1,            # <=threshold 视为“黑色不叠加”
    resize_interp: str = "nearest" # "nearest" or "linear"
) -> str:
    """
    Overlay attention on image:
      - white (255) => strongest overlay
      - black (0)   => no overlay
    Implementation:
      out = img*(1-a) + white*a, where a = alpha_max * (att/255)
      and a=0 if att<=threshold
    """
    import cv2
    import numpy as np

    # ---- load image ----
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR uint8
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    H, W = img.shape[:2]

    # ---- load attention ----
    att = cv2.imread(att_path, cv2.IMREAD_UNCHANGED)
    if att is None:
        raise FileNotFoundError(f"Failed to read attention: {att_path}")

    # ensure single channel
    if att.ndim == 3:
        att = att[..., 0]
    att = att.astype(np.uint8)

    # resize to match image
    if att.shape[:2] != (H, W):
        interp = cv2.INTER_NEAREST if resize_interp == "nearest" else cv2.INTER_LINEAR
        att = cv2.resize(att, (W, H), interpolation=interp)

    # ---- build alpha map ----
    a = (att.astype(np.float32) / 255.0) * float(alpha_max)
    if threshold > 0:
        a = np.where(att.astype(np.int32) <= int(threshold), 0.0, a)

    # expand to 3ch
    a3 = a[..., None]  # (H,W,1)

    # ---- overlay with white ----
    img_f = img.astype(np.float32)
    white = 255.0
    out = img_f * (1.0 - a3) + white * a3

    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    ok = cv2.imwrite(save_path, out)
    if not ok:
        raise RuntimeError(f"Failed to write: {save_path}")
    return save_path

def overlay_background_only_by_attention(
    image_path: str,
    att_path: str,                 # grayscale: white=fg, black=bg
    save_path: str = "overlay_bg50.jpg",
    bg_alpha: float = 0.5,         # 背景叠加强度（0.5 = 50%）
    resize_interp: str = "nearest" # "nearest" or "linear"
) -> str:
    """
    Reverse overlay:
      - foreground (att=255): keep original image
      - background (att=0): overlay bg_alpha with white
      - middle gray: linear interpolation

    Formula:
      w = att / 255
      a = (1 - w) * bg_alpha
      out = img * (1 - a) + 255 * a
    """
    import cv2
    import numpy as np

    # ---- load image ----
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR uint8
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    H, W = img.shape[:2]

    # ---- load attention ----
    att = cv2.imread(att_path, cv2.IMREAD_UNCHANGED)
    if att is None:
        raise FileNotFoundError(f"Failed to read attention: {att_path}")

    # ensure single channel
    if att.ndim == 3:
        att = att[..., 0]
    att = att.astype(np.float32)

    # resize if needed
    if att.shape[:2] != (H, W):
        interp = cv2.INTER_NEAREST if resize_interp == "nearest" else cv2.INTER_LINEAR
        att = cv2.resize(att, (W, H), interpolation=interp)

    # ---- compute alpha map (background only) ----
    w = np.clip(att / 255.0, 0.0, 1.0)   # fg weight
    a = (1.0 - w) * float(bg_alpha)      # only background gets alpha

    a3 = a[..., None]                    # (H,W,1)

    # ---- blend with white ----
    img_f = img.astype(np.float32)
    out = img_f * (1.0 - a3) + 255.0 * a3

    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    ok = cv2.imwrite(save_path, out)
    if not ok:
        raise RuntimeError(f"Failed to write image: {save_path}")

    return save_path


overlay_background_only_by_attention("xx.jpg", "s.jpg", bg_alpha=0.7)

# # chosen_kp_dict = np.load('/root/projects/sam-body4d/65.npy')
# chosen_kp_dict = np.load('c.npy', allow_pickle=True)

# # draw_named_keypoints_on_image("x.jpg", chosen_kp_dict.item(), save_path="res.jpg")

# save_levelset_attention_map(chosen_kp_dict.item(), save_path="s.jpg")

# # draw_keybody17_on_image_from_kp70("/root/projects/sam-body4d/outputs/20260122_125951_554_d3c6ad3a/masks_vis/00000065.png", chosen_kp_dict, save_path="chosen65.jpg", mask_path="/root/projects/sam-body4d/outputs/20260122_125951_554_d3c6ad3a/masks/00000065.png")
