import cv2
import torch

def draw_and_save_points(img_path, tensor_path, out_path="res.jpg", radius=4):
    """
    img_path: str, image path
    tensor_path: str, torch tensor path (.pt)
                 shape: (1, 12, 3) -> (x, y, label)
    out_path: str, output image path
    """
    # load image
    img = cv2.imread(img_path)
    assert img is not None, f"Failed to load image: {img_path}"

    # load tensor
    points = torch.load(tensor_path, map_location="cpu")
    assert points.ndim == 3 and points.shape[-1] == 3

    pts = points.squeeze(0)

    for p in pts:
        x, y, label = p.tolist()
        x, y = int(round(x)), int(round(y))
        label = int(label)

        # draw point
        cv2.circle(img, (x, y), radius, (0, 255, 0), -1)

        # draw label
        cv2.putText(
            img,
            str(label),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(out_path, img)

draw_and_save_points("/root/projects/sam-body4d/outputs/20260118_131723_148_0a830095/images/00000064.jpg", "/root/projects/sam-body4d/x.pt")
