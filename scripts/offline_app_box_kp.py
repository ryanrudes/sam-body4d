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
from models.sam_3d_body.notebook.utils import process_image_with_mask, save_mesh_results, process_image_with_bbox
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
        self.sam3_model, self.predictor = build_sam3_from_config(self.CONFIG)
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
        self.RUNTIME['kps'] = None

    def on_4d_generation(self, images_list, box_list, kps_list=None, flip=False):
        """
        Placeholder for 4D generation.
        Later:
        - run sam3_3d_body_model on per-frame images + masks
        - render 4D visualization video
        For now, just log and return None.
        """
        os.makedirs(f"{self.OUTPUT_DIR}/rendered_frames", exist_ok=True)
        os.makedirs(f"{self.OUTPUT_DIR}/mhr_params", exist_ok=True)
        for obj_id in range(len(box_list[0])):
            os.makedirs(f"{self.OUTPUT_DIR}/mesh_4d_individual/{obj_id+1}", exist_ok=True)
            os.makedirs(f"{self.OUTPUT_DIR}/rendered_frames_individual/{obj_id+1}", exist_ok=True)

        batch_size = self.RUNTIME['batch_size']
        n = len(images_list)
        
        # Optional, detect occlusions
        pred_res = self.RUNTIME['detection_resolution']
        pred_res_hi = self.RUNTIME['completion_resolution']
        modal_pixels_list = []
        if self.pipeline_mask is not None:
            for obj_id in range(len(box_list[0])):
                modal_pixels, ori_shape = load_and_transform_masks(self.OUTPUT_DIR + "/masks", resolution=pred_res, obj_id=obj_id+1)
                modal_pixels_list.append(modal_pixels)
            rgb_pixels, _, raw_rgb_pixels = load_and_transform_rgbs(self.OUTPUT_DIR + "/images", resolution=pred_res)
            depth_pixels = rgb_to_depth(rgb_pixels, self.depth_model)

        mhr_shape_scale_dict = {}   # each element is a list storing input parameters for mhr_forward
        obj_ratio_dict = {}         # avoid fake completion by obj ratio on the first frame

        # same cam_int across ALL frames
        input_image = np.array(Image.open(images_list[0])).astype('uint8')
        cam_int = self.sam3_3d_body_model.fov_estimator.get_cam_intrinsics(input_image)

        for i in tqdm(range(0, n, batch_size)):
            batch_images = images_list[i:i + batch_size]

            # Optional, detect occlusions
            idx_dict = {}
            idx_path = {}
            occ_dict = {}
            if len(modal_pixels_list) > 0:
                print("detect occlusions ...")
                pred_amodal_masks_dict = {}
                for (modal_pixels, obj_id) in zip(modal_pixels_list, self.RUNTIME['out_obj_ids']):
                    # detect occlusions for each object
                    # predict amodal masks (amodal segmentation)
                    pred_amodal_masks = self.pipeline_mask(
                        modal_pixels[:, i:i + batch_size, :, :, :],
                        depth_pixels[:, i:i + batch_size, :, :, :],
                        height=pred_res[0],
                        width=pred_res[1],
                        num_frames=modal_pixels[:, i:i + batch_size, :, :, :].shape[1],
                        decode_chunk_size=8,
                        motion_bucket_id=127,
                        fps=8,
                        noise_aug_strength=0.02,
                        min_guidance_scale=1.5,
                        max_guidance_scale=1.5,
                        generator=self.generator,
                    ).frames[0]

                    # for completion
                    pred_amodal_masks_com = [np.array(img.resize((pred_res_hi[1], pred_res_hi[0]))) for img in pred_amodal_masks]
                    pred_amodal_masks_com = np.array(pred_amodal_masks_com).astype('uint8')
                    pred_amodal_masks_com = (pred_amodal_masks_com.sum(axis=-1) > 600).astype('uint8')
                    pred_amodal_masks_com = [keep_largest_component(pamc) for pamc in pred_amodal_masks_com]
                    # for iou
                    pred_amodal_masks = [np.array(img.resize((W, H))) for img in pred_amodal_masks]
                    pred_amodal_masks = np.array(pred_amodal_masks).astype('uint8')
                    pred_amodal_masks = (pred_amodal_masks.sum(axis=-1) > 600).astype('uint8')
                    pred_amodal_masks = [keep_largest_component(pamc) for pamc in pred_amodal_masks]    # avoid small noisy masks
                    # compute iou
                    masks = [(np.array(Image.open(bm).convert('P'))==obj_id).astype('uint8') for bm in batch_masks]
                    ious = []
                    masks_margin_shrink = [bm.copy() for bm in masks]
                    mask_H, mask_W = masks_margin_shrink[0].shape
                    for bi, (a, b) in enumerate(zip(masks, pred_amodal_masks)):
                        # mute objects near margin
                        zero_mask_cp = np.zeros_like(masks_margin_shrink[bi])
                        zero_mask_cp[masks_margin_shrink[bi]==1] = 255
                        mask_binary_cp = zero_mask_cp.astype(np.uint8)
                        mask_binary_cp[:int(mask_H*0.05), :] = mask_binary_cp[-int(mask_H*0.05):, :] = mask_binary_cp[:, :int(mask_W*0.05)] = mask_binary_cp[:, -int(mask_W*0.05):] = 0
                        if mask_binary_cp.max() == 0:   # margin objects
                            ious.append(1.0)
                            continue
                        area_a = (a > 0).sum()
                        area_b = (b > 0).sum()
                        if area_a == 0 and area_b == 0:
                            ious.append(1.0)
                        elif area_a > area_b:
                            ious.append(1.0)
                        else:
                            inter = np.logical_and(a > 0, b > 0).sum()
                            uni = np.logical_or(a > 0, b > 0).sum()
                            obj_iou = inter / (uni + 1e-6)
                            ious.append(obj_iou)

                        if i == 0 and bi == 0:
                            if ious[0] < 0.7:
                                obj_ratio_dict[obj_id] = bbox_from_mask(b)
                            else:
                                obj_ratio_dict[obj_id] = bbox_from_mask(a)

                    # remove fake completions (empty or from MARGINs)
                    for pi, pamc in enumerate(pred_amodal_masks_com):
                        # zero predictions, back to original masks
                        if masks[pi].sum() > pred_amodal_masks[pi].sum():
                            ious[pi] = 1.0
                            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
                        # elif len(obj_ratio_dict)>0 and not are_bboxes_similar(bbox_from_mask(pred_amodal_masks[pi]), obj_ratio_dict[obj_id]):
                        #     ious[pi] = 1.0
                        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
                        elif is_super_long_or_wide(pred_amodal_masks[pi], obj_id):
                            ious[pi] = 1.0
                            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
                        elif is_skinny_mask(pred_amodal_masks[pi]):
                            ious[pi] = 1.0
                            pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)
                        # elif masks[pi].sum() == 0: # TODO: recover empty masks in future versions (to avoid severe fake completion)
                        #     ious[pi] = 1.0
                        #     pred_amodal_masks_com[pi] = resize_mask_with_unique_label(masks[pi], pred_res_hi[0], pred_res_hi[1], obj_id)

                    pred_amodal_masks_dict[obj_id] = pred_amodal_masks_com

                    # confirm occlusions & save masks (for HMR)
                    start, end = (idxs := [ix for ix,x in enumerate(ious) if x < 0.7]) and (idxs[0], idxs[-1]) or (None, None)

                    occ_dict[obj_id] = [1 if ix > 0.7 else 0 for ix in ious]

                    if start is not None and end is not None:
                        start = max(0, start-2)
                        end = min(modal_pixels[:, i:i + batch_size, :, :, :].shape[1]-1, end+2)
                        idx_dict[obj_id] = (start, end)
                        completion_path = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
                        completion_image_path = f'{self.OUTPUT_DIR}/completion/{completion_path}/images'
                        completion_masks_path = f'{self.OUTPUT_DIR}/completion/{completion_path}/masks'
                        os.makedirs(completion_image_path, exist_ok=True)
                        os.makedirs(completion_masks_path, exist_ok=True)
                        idx_path[obj_id] = {'images': completion_image_path, 'masks': completion_masks_path}
                        # save completion masks
                        for idx_ in range(start, end):
                            mask_idx_ = pred_amodal_masks[idx_].copy()
                            mask_idx_[mask_idx_ > 0] = obj_id
                            mask_idx_ = Image.fromarray(mask_idx_).convert('P')
                            mask_idx_.putpalette(DAVIS_PALETTE)
                            mask_idx_.save(os.path.join(completion_masks_path, f"{idx_:08d}.png"))

                # completion
                for obj_id, (start, end) in idx_dict.items(): 
                    completion_image_path = idx_path[obj_id]['images']
                    # prepare inputs
                    modal_pixels_current, ori_shape = load_and_transform_masks(self.OUTPUT_DIR + "/masks", resolution=pred_res_hi, obj_id=obj_id)
                    rgb_pixels_current, _, raw_rgb_pixels_current = load_and_transform_rgbs(self.OUTPUT_DIR + "/images", resolution=pred_res_hi)
                    modal_pixels_current = modal_pixels_current[:, i:i + batch_size, :, :, :]
                    modal_pixels_current = modal_pixels_current[:, start:end]
                    pred_amodal_masks_current = pred_amodal_masks_dict[obj_id][start:end]
                    modal_mask_union = (modal_pixels_current[0, :, 0, :, :].cpu().numpy() > 0).astype('uint8')
                    pred_amodal_masks_current = np.logical_or(pred_amodal_masks_current, modal_mask_union).astype('uint8')
                    pred_amodal_masks_tensor = torch.from_numpy(np.where(pred_amodal_masks_current == 0, -1, 1)).float().unsqueeze(0).unsqueeze(
                        2).repeat(1, 1, 3, 1, 1)

                    rgb_pixels_current = rgb_pixels_current[:, i:i + batch_size, :, :, :][:, start:end]
                    modal_obj_mask = (modal_pixels_current > 0).float()
                    modal_background = 1 - modal_obj_mask
                    rgb_pixels_current = (rgb_pixels_current + 1) / 2
                    modal_rgb_pixels = rgb_pixels_current * modal_obj_mask + modal_background
                    modal_rgb_pixels = modal_rgb_pixels * 2 - 1

                    print("content completion by diffusion-vas ...")
                    # predict amodal rgb (content completion)
                    pred_amodal_rgb = self.pipeline_rgb(
                        modal_rgb_pixels,
                        pred_amodal_masks_tensor,
                        height=pred_res_hi[0], # my_res[0]
                        width=pred_res_hi[1],  # my_res[1]
                        num_frames=end-start,
                        decode_chunk_size=8,
                        motion_bucket_id=127,
                        fps=8,
                        noise_aug_strength=0.02,
                        min_guidance_scale=1.5,
                        max_guidance_scale=1.5,
                        generator=self.generator,
                    ).frames[0]

                    pred_amodal_rgb = [np.array(img) for img in pred_amodal_rgb]

                    # save pred_amodal_rgb
                    pred_amodal_rgb = np.array(pred_amodal_rgb).astype('uint8')
                    pred_amodal_rgb_save = np.array([cv2.resize(frame, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)
                                                    for frame in pred_amodal_rgb])
                    idx_ = start
                    for img in pred_amodal_rgb_save:
                        cv2.imwrite(os.path.join(completion_image_path, f"{idx_:08d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        idx_ += 1

            else:
                for obj_id in range(len(box_list[0])):
                    occ_dict[obj_id+1] = [1] * len(batch_images)

            batch_boxes = [bboxes[i:i + batch_size] for bboxes in box_list]
            batch_kps = None if kps_list is None else [kps[i:i + batch_size] for kps in kps_list]

            # Process with external mask
            mask_outputs, id_batch, empty_frame_list = process_image_with_bbox(self.sam3_3d_body_model, batch_images, batch_boxes, idx_path, idx_dict, mhr_shape_scale_dict, occ_dict, batch_kps, flip=flip, cam_int=cam_int)
            
            num_empth_ids = 0
            for frame_id in range(len(batch_images)):
                image_path = batch_images[frame_id]
                if frame_id in empty_frame_list:
                    mask_output = None
                    id_current = None
                    num_empth_ids += 1
                else:
                    mask_output = mask_outputs[frame_id-num_empth_ids]
                    id_current = id_batch[frame_id-num_empth_ids]
                
                # img = cv2.imread(image_path)
                
                # rend_img = visualize_sample_together(img, mask_output, self.sam3_3d_body_model.faces, id_current)

                # out = rend_img.copy()

                # for boxi in range(len(mask_output)):
                #     x1, y1, x2, y2 = batch_boxes[boxi][frame_id]
                #     out = out.copy()
                #     color = [(0, 165, 255), (0, 255, 255)][boxi]
                #     cv2.rectangle(
                #         out,
                #         (int(x1.item()), int(y1.item())),
                #         (int(x2.item()), int(y2.item())),
                #         color=color,  # BGR
                #         thickness=3
                #     )

                # cv2.imwrite(
                #     f"{self.OUTPUT_DIR}/rendered_frames/{os.path.basename(image_path)[:-4]}.jpg",
                #     out.astype(np.uint8),
                # )

                np.savez_compressed(f"{self.OUTPUT_DIR}/mhr_params/{os.path.basename(image_path)[:-4]}_data.npz", data=mask_output)
                np.savez_compressed(f"{self.OUTPUT_DIR}/mhr_params/{os.path.basename(image_path)[:-4]}_id.npz", data=id_current)


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
