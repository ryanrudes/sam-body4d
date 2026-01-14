import os
ROOT = os.path.dirname(os.path.dirname(__file__))

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(top_dir)
sys.path.append(os.path.join(top_dir, 'models', 'sam_3d_body'))
sys.path.append(os.path.join(top_dir, 'models', 'diffusion_vas'))

import uuid
from datetime import datetime

def gen_id():
    t = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    u = uuid.uuid4().hex[:8]
    return f"{t}_{u}"

import time
import cv2
import glob
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from utils import draw_point_marker, mask_painter, images_to_mp4, DAVIS_PALETTE, jpg_folder_to_mp4, is_super_long_or_wide, keep_largest_component, is_skinny_mask, bbox_from_mask, gpu_profile, resize_mask_with_unique_label

from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from models.sam_3d_body.tools.vis_utils import visualize_sample_together, visualize_sample
from models.diffusion_vas.demo import init_amodal_segmentation_model, init_rgb_model, init_depth_model, load_and_transform_masks, load_and_transform_rgbs, rgb_to_depth

import torch
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def build_sam3_from_config(cfg):
    """
    Construct and return your SAM-3 model from config.
    You replace this with your real init code.
    """
    from models.sam3.sam3.model_builder import build_sam3_video_model

    sam3_model = build_sam3_video_model(checkpoint_path=cfg.sam3['ckpt_path'])
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    return sam3_model, predictor


def build_sam3_3d_body_config(cfg):
    mhr_path = cfg.sam_3d_body['mhr_path']
    fov_path = cfg.sam_3d_body['fov_path']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        cfg.sam_3d_body['ckpt_path'], device=device, mhr_path=mhr_path
    )
    
    human_detector, human_segmentor, fov_estimator = None, None, None
    from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator
    fov_estimator = FOVEstimator(name='moge2', device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    return estimator


def build_diffusion_vas_config(cfg):
    model_path_mask = cfg.completion['model_path_mask']
    model_path_rgb = cfg.completion['model_path_rgb']
    depth_encoder = cfg.completion['depth_encoder']
    model_path_depth = cfg.completion['model_path_depth']
    max_occ_len = min(cfg.completion['max_occ_len'], cfg.sam_3d_body['batch_size'])

    generator = torch.manual_seed(23)

    pipeline_mask = init_amodal_segmentation_model(model_path_mask)
    pipeline_rgb = init_rgb_model(model_path_rgb)
    depth_model = init_depth_model(model_path_depth, depth_encoder)

    return pipeline_mask, pipeline_rgb, depth_model, max_occ_len, generator


class offline_app:
    def __init__(self, config_path: str = os.path.join(ROOT, "configs", "body4d.yaml"), refine_occlusion=False):
        """Initialize CONFIG, SAM3_MODEL, and global RUNTIME dict."""
        self.CONFIG = OmegaConf.load(config_path)
        # self.sam3_model, self.predictor = build_sam3_from_config(self.CONFIG)
        self.sam3_3d_body_model = build_sam3_3d_body_config(self.CONFIG)

        if self.CONFIG.completion.get('enable', False) and refine_occlusion:
            self.pipeline_mask, self.pipeline_rgb, self.depth_model, self.max_occ_len, self.generator = build_diffusion_vas_config(self.CONFIG)
        else:
            self.pipeline_mask, self.pipeline_rgb, self.depth_model, self.max_occ_len, self.generator = None, None, None, None, None
        
        self.OUTPUT_DIR = ''

        self.RUNTIME = {}  # clear any old state
        self.RUNTIME['batch_size'] = self.CONFIG.sam_3d_body.get('batch_size', 1)
        self.RUNTIME['detection_resolution'] = self.CONFIG.completion.get('detection_resolution', [256, 512])
        self.RUNTIME['completion_resolution'] = self.CONFIG.completion.get('completion_resolution', [512, 1024])
        self.RUNTIME['smpl_export'] = self.CONFIG.runtime.get('smpl_export', False)
        self.RUNTIME['bboxes'] = None

    def on_4d_generation(self, images_list, box_list):
        """
        Placeholder for 4D generation.
        Later:
        - run sam3_3d_body_model on per-frame images + masks
        - render 4D visualization video
        For now, just log and return None.
        """
        print("[DEBUG] 4D Generation button clicked.")

        os.makedirs(f"{self.OUTPUT_DIR}/rendered_frames", exist_ok=True)
        os.makedirs(f"{self.OUTPUT_DIR}/mhr_params", exist_ok=True)
        for obj_id in range(len(box_list[0])):
            os.makedirs(f"{self.OUTPUT_DIR}/mesh_4d_individual/{obj_id+1}", exist_ok=True)
            os.makedirs(f"{self.OUTPUT_DIR}/rendered_frames_individual/{obj_id+1}", exist_ok=True)

        batch_size = 1
        n = len(images_list)

        for i in tqdm(range(0, n, batch_size)):
            # batch_images = images_list[i:i + batch_size]
            batch_images = images_list[i]

            batch_boxes = [bboxes[i:i + batch_size] for bboxes in box_list]
            # Process with external mask
            mask_outputs = self.sam3_3d_body_model.process_one_image(batch_images, torch.concat(batch_boxes, dim=0))
            
            for frame_id in range(len(batch_images)):
                image_path = batch_images#[frame_id]
                mask_output = mask_outputs#[frame_id]

                id_current = [i+1 for i in range(len(mask_output))]
                img = cv2.imread(image_path)
                rend_img = visualize_sample_together(img, mask_output, self.sam3_3d_body_model.faces, id_current)
                cv2.imwrite(
                    f"{self.OUTPUT_DIR}/rendered_frames/{os.path.basename(image_path)[:-4]}.jpg",
                    rend_img.astype(np.uint8),
                )
                
                np.savez_compressed(f"{self.OUTPUT_DIR}/mhr_params/{os.path.basename(image_path)[:-4]}_data.npz", data=mask_output)
                np.savez_compressed(f"{self.OUTPUT_DIR}/mhr_params/{os.path.basename(image_path)[:-4]}_id.npz", data=id_current)

        out_4d_path = os.path.join(self.OUTPUT_DIR, f"4d_{time.time():.0f}.mp4")

        return out_4d_path


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Offline 4D Body Generation for long videos")
#     parser.add_argument("--output_dir", type=str, default="path to outputs/20251207_043551_865_21ed56bf",
#                         help="Path to the output directory")
#     args = parser.parse_args()
#     # Check dir
#     if not os.path.isdir(args.output_dir):
#         raise FileNotFoundError(f"Output directory not found: {args.output_dir}")

#     init_runtime(output_dir=args.output_dir)
#     on_4d_generation()
