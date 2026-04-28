# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from pathlib import Path

import numpy as np
import torch


class HumanDetector:
    def __init__(self, name="vitdet", device="cuda", **kwargs):
        self.device = device

        if name == "vitdet":
            try:
                self.detector = load_detectron2_vitdet(**kwargs)
                self.detector_func = run_detectron2_vitdet
                print("########### Using human detector: ViTDet (detectron2)...")
            except ImportError:
                print(
                    "########### detectron2 not available; using torchvision Faster R-CNN (person) fallback..."
                )
                self.detector = load_torchvision_person_detector(self.device)
                self.detector_func = run_torchvision_person_detector

            self.detector = self.detector.to(self.device)
            self.detector.eval()
        else:
            raise NotImplementedError

    def run_human_detection(self, img, **kwargs):
        return self.detector_func(self.detector, img, **kwargs)


def load_torchvision_person_detector(device):
    """COCO-pretrained Faster R-CNN; person class id = 1."""
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model


def run_torchvision_person_detector(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = True,
):
    """Match ViTDet output: xyxy boxes in original image coordinates (numpy)."""
    import cv2
    import torch

    height, width = img.shape[:2]
    # ViTDet uses category 0 for person; COCO in torchvision uses 1.
    coco_person_id = 1 if det_cat_id == 0 else det_cat_id

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb.astype("float32") / 255.0).permute(2, 0, 1)
    t = t.to(next(detector.parameters()).device)
    with torch.no_grad():
        det_out = detector([t])[0]

    labels = det_out["labels"]
    scores = det_out["scores"]
    boxes = det_out["boxes"]
    valid = (labels == coco_person_id) & (scores > bbox_thr)
    if valid.sum() == 0 and default_to_full_image:
        boxes_np = np.array([0, 0, width, height]).reshape(1, 4)
    else:
        boxes_np = boxes[valid].cpu().numpy()

    sorted_indices = np.lexsort(
        (boxes_np[:, 3], boxes_np[:, 2], boxes_np[:, 1], boxes_np[:, 0])
    )
    boxes_np = boxes_np[sorted_indices]
    return boxes_np


def load_detectron2_vitdet(path=""):
    """
    Load vitdet detector similar to 4D-Humans demo.py approach.
    Checkpoint is automatically downloaded from the hardcoded URL.
    """
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate, LazyConfig

    # Get config file from tools directory (same folder as this file)
    cfg_path = Path(__file__).parent / "cascade_mask_rcnn_vitdet_h_75ep.py"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. "
            "Make sure cascade_mask_rcnn_vitdet_h_75ep.py exists in the tools directory."
        )

    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        if path == ""
        else os.path.join(path, "model_final_f05665.pkl")
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = instantiate(detectron2_cfg.model)
    checkpointer = DetectionCheckpointer(detector)
    checkpointer.load(detectron2_cfg.train.init_checkpoint)

    detector.eval()
    return detector


def run_detectron2_vitdet(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = True,
):
    import detectron2.data.transforms as T

    height, width = img.shape[:2]

    IMAGE_SIZE = 1024
    transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)
    img_transformed = transforms(T.AugInput(img)).apply_image(img)
    img_transformed = torch.as_tensor(
        img_transformed.astype("float32").transpose(2, 0, 1)
    )
    inputs = {"image": img_transformed, "height": height, "width": width}

    with torch.no_grad():
        det_out = detector([inputs])

    det_instances = det_out[0]["instances"]
    valid_idx = (det_instances.pred_classes == det_cat_id) & (
        det_instances.scores > bbox_thr
    )
    if valid_idx.sum() == 0 and default_to_full_image:
        boxes = np.array([0, 0, width, height]).reshape(1, 4)
    else:
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

    # Sort boxes to keep a consistent output order
    sorted_indices = np.lexsort(
        (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
    )  # shape: [len(boxes),]
    boxes = boxes[sorted_indices]
    return boxes
