from PIL import Image
import numpy as np


def paste_mask_region(img1_path: str,
                      img2_path: str,
                      mask_path: str,
                      output_path: str = "output.jpg"):
    """
    Paste masked region from image1 onto image2 using mask.

    Parameters
    ----------
    img1_path : str
        Path to source image (foreground).
    img2_path : str
        Path to target image (background).
    mask_path : str
        Path to mask image (P mode, foreground == 1).
    output_path : str
        Output image path.
    """

    # ---------- Load ----------
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    mask = Image.open(mask_path)

    # ---------- Ensure same size ----------
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)

    if mask.size != img1.size:
        mask = mask.resize(img1.size, Image.NEAREST)

    # ---------- Convert mask ----------
    mask_np = np.array(mask)

    # foreground = 1
    mask_bool = mask_np == 1

    # ---------- Convert images ----------
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    # ---------- Paste ----------
    output_np = img2_np.copy()
    output_np[mask_bool] = img1_np[mask_bool]

    # ---------- Save ----------
    output_img = Image.fromarray(output_np)
    output_img.save(output_path, quality=95)


paste_mask_region(
    "/home/hmq/projects/sam-body4d/outputs/20260205_144808_142_729d420c/completion/UM94/images/00000000.jpg",
    "/home/hmq/projects/sam-body4d/outputs/20260205_144808_142_729d420c/images/00000000.jpg",
    "/home/hmq/projects/sam-body4d/outputs/20260205_144808_142_729d420c/completion/UM94/masks/00000000.png",
    "output.jpg"
    )

from painter import mask_painter
from PIL import Image

img = np.array(Image.open("output.jpg").convert('RGB'))
# msk = np.zeros_like(img[:, :, 0])
out_mask = np.array(Image.open("/home/hmq/projects/sam-body4d/outputs/20260205_144808_142_729d420c/completion/UM94/masks/00000000.png").convert("P"))
mask = (out_mask == 0).astype(np.uint8) * 255
img = mask_painter(img, mask, mask_color=0, mask_alpha=0.8, contour_width=17)
img_vis = Image.fromarray(img).convert('RGB')
img_vis.save("abc.jpg")