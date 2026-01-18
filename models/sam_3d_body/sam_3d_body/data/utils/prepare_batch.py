# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.utils.data import default_collate


class NoCollate:
    def __init__(self, data):
        self.data = data

from PIL import Image

def save_tensor_as_jpg(tensor, path):
    """
    tensor: torch.Size([1, 1, 3, H, W]), value in [0,1]
    """
    assert isinstance(tensor, torch.Tensor)

    img = tensor[0, 0]              # -> [3, H, W]
    img = img.clamp(0, 1)
    img = (img * 255).byte()        # uint8
    img = img.permute(1, 2, 0)      # [H, W, 3]
    img = img.cpu().numpy()

    Image.fromarray(img).save(path, quality=95)

import cv2

def draw_and_save_keypoints(img_tensor, keypoints, save_path="point.jpg",
                            radius=4, thickness=2):
    """
    img_tensor: torch.Tensor, shape (1, 3, H, W)
    keypoints: (17, 2), xy format, torch.Tensor or np.ndarray
    save_path: output image path
    """

    # ---- image ----
    img = img_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    img = img.copy()

    # ---- keypoints ----
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()

    # ---- draw ----
    for i, (x, y) in enumerate(keypoints):
        x, y = int(round(x)), int(round(y))

        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), radius, (0, 255, 0), -1)
            cv2.putText(
                img,
                str(i),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

    # ---- save ----
    cv2.imwrite(save_path, img)


def prepare_batch(
    img,
    transform,
    boxes,
    masks=None,
    masks_score=None,
    kps=None,
    cam_int=None,
    img_com_dict=None,  # optional dict of complementary images for each box index
):
    """A helper function to prepare data batch for SAM 3D Body model inference."""
    height, width = img.shape[:2]

    # construct batch data samples
    data_list = []
    for idx in range(boxes.shape[0]):
        if (img_com_dict is not None) and idx in img_com_dict:
            data_info = dict(img=img_com_dict[idx])    
        else:
            data_info = dict(img=img)
        data_info["bbox"] = boxes[idx]  # shape (4,)
        data_info["bbox_format"] = "xyxy"

        if kps is not None:
            if kps[idx].shape[-1] == 3:
                data_info["keypoints_2d"] = kps[idx][:,:2]
            else:    
                data_info["keypoints_2d"] = kps[idx]

        if masks is not None:
            data_info["mask"] = masks[idx].copy()
            if masks_score is not None:
                data_info["mask_score"] = masks_score[idx]
            else:
                data_info["mask_score"] = np.array(1.0, dtype=np.float32)
        else:
            data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
            data_info["mask_score"] = np.array(0.0, dtype=np.float32)

        data_list.append(transform(data_info))

    batch = default_collate(data_list)

    max_num_person = batch["img"].shape[0]
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, max_num_person))

    if cam_int is not None:
        batch["cam_int"] = cam_int.to(batch["img"])
    else:
        # Default camera intrinsics according image size
        batch["cam_int"] = torch.tensor(
            [
                [
                    [(height**2 + width**2) ** 0.5, 0, width / 2.0],
                    [0, (height**2 + width**2) ** 0.5, height / 2.0],
                    [0, 0, 1],
                ]
            ],
        ).to(batch["img"])

    batch["img_ori"] = [NoCollate(img)]
    return batch
