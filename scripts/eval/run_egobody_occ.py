import argparse
import os, sys, glob, re, json
from tqdm import tqdm
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
import pandas as pd

from offline_app_mask_kp import *
from utils.draw_utils import draw_bbox_and_save
# from eval_utils.dataset_egobody_occ import DataModuleEgoBody

def inference(args):
    # init configs and cover with cmd options
    predictor = OfflineApp()

    # init data
    # cfg = {
    #     'name': 'egobody', 'bm_path': 'body_models', 'model_type': 'smplx', 'clip_len': 500, 
    #     'normalize': False, 'canonical': True, 'batch_size': 1, 'num_workers': 1, 
    #     'num_workers_val': 1, 'spacing': 1, 'recording': None, 'view': None, 'body_idx': None, 
    #     'crop_res': [256, 256], 'flip_prob': 0.5, 'trans_factor': 0.1, 'rot_factor': 30.0, 'scale_factor': 0.3, 
    #     'rot_aug_prob': 0.6, 'extreme_crop_aug': True, 'extreme_crop_aug_prob': 0.2, 'extreme_crop_lvl': 2, 
    #     'extreme_crop_seq_ratio': 0.2, 'contact_hand': False, 'contact_vel_thres': 0.3, 
    #     'dataset_root': args.data_dir, 'overlap_len': 120, 'test_idx': 14, 'gt_bbox': False
    # }
    # datamodule = DataModuleEgoBody(cfg)
    # Load dataset information
    df = pd.read_csv(
        f"{args.data_dir}/egobody_occ_info.csv"
    )
    data_dict = {
        "view": dict(zip(df["recording_name"], df["view"])),
        "body_idx": dict(zip(df["recording_name"], df["target_idx"])),
        "scene_name": dict(zip(df["recording_name"], df["scene_name"])),
        "gender_gt": dict(zip(df["recording_name"], df["target_gender"])),
        "body_idx_fpv": dict(zip(df["recording_name"], df["body_idx_fpv"])),
        "target_start_frame": dict(zip(df["recording_name"], df["target_start_frame"])),
        "target_end_frame": dict(zip(df["recording_name"], df["target_end_frame"])),
    }

    for seq_name in tqdm(data_dict["view"].keys()):
        target_start_frame = data_dict["target_start_frame"][seq_name]
        target_end_frame = data_dict["target_end_frame"][seq_name]
        view = data_dict["view"][seq_name]
        body_idx = data_dict["body_idx"][seq_name]

        output_dir = os.path.join(args.output_dir, seq_name)
        predictor.OUTPUT_DIR = output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)
        frame_list = glob.glob(os.path.join(args.data_dir, 'kinect_color', seq_name, view, '*.jpg'))
        frame_list.sort()
        # frame_list = frame_list[target_start_frame:target_end_frame+1]
        frame_list = sorted(
            [
                f for f in frame_list
                if target_start_frame
                <= int(re.findall(r"\d+", f)[-1])
                <= target_end_frame
            ],
            key=lambda f: int(re.findall(r"\d+", f)[-1])
        )
        keypo_list = glob.glob(os.path.join(args.data_dir, 'keypoints_cleaned', seq_name, view, '*.json'))
        keypo_list.sort()
        # keypo_list = keypo_list[target_start_frame:target_end_frame+1]
        keypo_list = sorted(
            [
                f for f in keypo_list
                if target_start_frame
                <= int(re.findall(r"\d+", f)[-1])
                <= target_end_frame
            ],
            key=lambda f: int(re.findall(r"\d+", f)[-1])
        )
        one_frame = Image.open(frame_list[0]).convert('RGB')
        width, height = one_frame.size

        assert len(keypo_list) == len(frame_list), f"num of boxes {len(keypo_list)} != num of frames {len(frame_list)}"

        max_cover_frame_idx = 0
        batch_frames = [Image.open(bf).convert("RGB") for bf in frame_list]
        resized_batch_frames = resize_images_longest_side(batch_frames)
        ratio = resized_batch_frames[0].size[-1] / batch_frames[0].size[-1]

        # keypoint extraction (only on the 1st frame, for bbox extraction - confirming the target human)
        with open(
            os.path.join(args.data_dir, "kinect_cam_params", "kinect_{}".format(view), "Color.json"), "r") as f:
            color_cam = json.load(f)
        K = np.array(color_cam["camera_mtx"]).astype(np.float32)  # K includes f and c
        dist_coeffs = np.array(color_cam["k"]).astype(np.float32)
        keypoints_list = []
        for kps_path in keypo_list:
            with open(kps_path, 'r') as f:
                kps_data = json.load(f)
            body_kps = np.array(kps_data["people"][int(body_idx)]["pose_keypoints_2d"]).astype(np.float32).reshape((-1, 3))
            valid = body_kps[:, 2] > 0.2
            body_kps[~valid] = 0 # remove unreliable keypoints
            if valid.any():
                body_kps[valid, :2] = cv2.undistortImagePoints(
                    body_kps[valid, :2].reshape(-1, 1, 2), K, dist_coeffs).reshape(-1, 2)
            keypoints_list.append(body_kps)
        keypoints_list = np.stack(keypoints_list).astype(np.float32)

        # initialise and reset predictor state
        response = predictor.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=resized_batch_frames,
            )
        )
        predictor.RUNTIME['session_id'] = response["session_id"]
        predictor.RUNTIME['out_obj_ids'] = []

        # load bbox (start frame) 
        prompt_text_str = "person"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            response = predictor.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=predictor.RUNTIME['session_id'],
                    frame_index=0,
                    text=prompt_text_str,
                )
            )
            outputs_per_frame = propagate_in_video(predictor.predictor, predictor.RUNTIME['session_id'])
            # only focus on target person
            obj_dict = {}   # key: inference_id (start from 1), value: sam_id
            obj_list = []
            for obj_id in range(1):
                pts = keypoints_list[0]*ratio # 17 x 3
                valid = (pts[:, 0] != 0) & (pts[:, 1] != 0)
                pts = pts[valid]
                bbox_obj = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
                for out_obj_id in outputs_per_frame[max_cover_frame_idx]['out_obj_ids']:
                    mapped_id = np.where(outputs_per_frame[max_cover_frame_idx]['out_obj_ids'] == out_obj_id)[0].item()
                    if bbox_similar_to_mask_bbox(bbox_obj, outputs_per_frame[max_cover_frame_idx]['out_binary_masks'][mapped_id]):
                        obj_dict[obj_id+1] = out_obj_id.item()
                        obj_list.append(out_obj_id.item())
                predictor.RUNTIME['out_obj_ids'].append(obj_id+1)
        # 3. save masks
        predictor.save_masks(
            start_frame_idx=0, 
            outputs_per_frame=outputs_per_frame, 
            obj_dict=obj_dict, 
            resized_batch_frames=resized_batch_frames,
            original_size=(width, height),
            frame_list=frame_list,
        )
        # 4. hmr upon masks
        if predictor.RUNTIME['session_id'] is not None:
            _ = predictor.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=predictor.RUNTIME['session_id'],
                )
            )

        with torch.autocast("cuda", enabled=False):
            predictor.on_4d_generation(frame_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on EgoBody-OCC")
    parser.add_argument("--data_dir", type=str, default="path to EgoBody-OCC data",
        help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="path to output",
        help="Path to the output directory")
    parser.add_argument("--refine_occlusion", action="store_true",
        help="Whether to use occlusion-aware refinement (default False)")
    args = parser.parse_args()

    inference(args)
