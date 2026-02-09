import os
import numpy as np
from PIL import Image, ImageDraw

def draw_bbox_and_save(bbox, image_path, output_path=None, color=(255, 0, 0), width=3):
    """
    bbox: numpy array or list [x1, y1, x2, y2]
    image_path: 输入图片路径
    output_path: 保存路径（默认自动生成）
    color: bbox 颜色 (R,G,B)
    width: 线宽

    return: 保存后的文件路径
    """

    bbox = np.array(bbox).astype(int)
    x1, y1, x2, y2 = bbox.tolist()

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_bbox.jpg"

    img.save(output_path, quality=95)

    return output_path
