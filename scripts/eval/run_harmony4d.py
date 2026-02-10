import argparse
import os, sys, glob
from tqdm import tqdm
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from offline_app_mask_kp import *
from utils.draw_utils import draw_bbox_and_save


def inference(args):
    # init configs and cover with cmd options
    predictor = OfflineApp()

    # init data
    test_seq_name_list = {}
    seq_list_1 = glob.glob(os.path.join(args.data_dir, '*'))
    seq_list_1.sort()

    for seq_1 in tqdm(seq_list_1):
        if os.path.isdir(seq_1):
            seq_list_2 = glob.glob(os.path.join(seq_1, '*'))
            seq_list_2.sort()
            for seq_2 in seq_list_2:
                if os.path.isdir(seq_2):
                    camera_list = glob.glob(os.path.join(seq_2, 'exo', '*'))
                    camera_list.sort()
                    # for each cam
                    for cam in tqdm(camera_list):
                        cam_name = os.path.basename(cam)
                        # 0. init outputs
                        # if cam_name!='cam22':
                        #     continue
                        
                        output_dir = os.path.join(args.output_dir, os.path.basename(seq_1), os.path.basename(seq_2), cam_name)
                        predictor.OUTPUT_DIR = output_dir
                        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)
                        frame_list = glob.glob(os.path.join(seq_2, 'exo', cam_name, 'images', '*.jpg'))
                        seq_path = os.path.join(os.path.join(seq_2, 'exo', 'images', cam_name))
                        frame_list.sort()
                        one_frame = Image.open(frame_list[0]).convert('RGB')
                        width, height = one_frame.size
                        box_list = glob.glob(os.path.join(seq_2, 'processed_data', 'bbox', cam_name, '*.npy'))
                        box_list.sort()

                        assert len(box_list) == len(frame_list), f"num of boxes {len(box_list)} != num of frames {len(frame_list)}"

                        # explore the num of objects and only perform text segmentation on frame with 2 persons
                        num_frames = len(frame_list)
                        
                        cover_ratio_dict = {}
                        for start_frame_idx in range(num_frames):
                            bbox_start = np.load(box_list[start_frame_idx], allow_pickle=True).item()
                            num_objects = len(bbox_start)
                            if num_objects == 2:
                                cover_ratio_dict[start_frame_idx] = iou_over_threshold(bbox_start['aria01'], bbox_start['aria02'])
                        max_cover_frame_idx = max(cover_ratio_dict, key=cover_ratio_dict.get)
                        bbox_start = np.load(box_list[max_cover_frame_idx], allow_pickle=True).item()

                        batch_frames = [Image.open(bf).convert("RGB") for bf in frame_list]
                        resized_batch_frames = resize_images_longest_side(batch_frames)
                        ratio = resized_batch_frames[0].size[-1] / batch_frames[0].size[-1]
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
                            # out = response["outputs"]
                            # # only focus on target person
                            # obj_dict = {}   # key: inference_id (start from 1), value: sam_id
                            # obj_list = []
                            # for obj_id, bbox_obj in bbox_start.items():
                            #     bbox_obj = bbox_obj*ratio # 17 x 3
                            #     for out_obj_id in out['out_obj_ids']:
                            #         if bbox_similar_to_mask_bbox(bbox_obj, out['out_binary_masks'][out_obj_id]):
                            #             obj_dict[int(obj_id[-1])] = out_obj_id.item()
                            #             obj_list.append(out_obj_id.item())
                            #     predictor.RUNTIME['out_obj_ids'].append(int(obj_id[-1]))
                            # # remove other person
                            # for out_id in out['out_obj_ids']:
                            #     if out_id.item() in obj_list:
                            #         continue
                            #     response = predictor.predictor.handle_request(
                            #         request=dict(
                            #             type="remove_object",
                            #             session_id=predictor.RUNTIME['session_id'],
                            #             obj_id=out_id.item(),
                            #         )
                            #     )
                                
                            outputs_per_frame = propagate_in_video(predictor.predictor, predictor.RUNTIME['session_id'])

                            # only focus on target person
                            obj_dict = {}   # key: inference_id (start from 1), value: sam_id
                            obj_list = []
                            for obj_id, bbox_obj in bbox_start.items():
                                bbox_obj = bbox_obj*ratio # 17 x 3
                                for out_obj_id in outputs_per_frame[max_cover_frame_idx]['out_obj_ids']:
                                    mapped_id = np.where(outputs_per_frame[max_cover_frame_idx]['out_obj_ids'] == out_obj_id)[0].item()
                                    if bbox_similar_to_mask_bbox(bbox_obj, outputs_per_frame[max_cover_frame_idx]['out_binary_masks'][mapped_id]):
                                        obj_dict[int(obj_id[-1])] = out_obj_id.item()
                                        obj_list.append(out_obj_id.item())
                                predictor.RUNTIME['out_obj_ids'].append(int(obj_id[-1]))
                        
                        # 3. save masks
                        predictor.save_masks(
                            start_frame_idx=0, 
                            outputs_per_frame=outputs_per_frame, 
                            obj_dict=obj_dict, 
                            resized_batch_frames=resized_batch_frames,
                            original_size=(width, height),
                        )
                        # 4. hmr upon masks

                        # kps_list = []
                        # for obj_id in range(3):
                        #     try:
                        #         seq_name_with_id = f'{seq}_{obj_id}'
                        #         kps_list.append(kp[seq_name_with_id])
                        #     except:
                        #         break
                        # if len(kps_list) == 0:
                        #     kps_list = None

                        if predictor.RUNTIME['session_id'] is not None:
                            # _ = predictor.predictor.handle_request(
                            #     request=dict(
                            #         type="reset_session",
                            #         session_id=predictor.RUNTIME['session_id'],
                            #     )
                            # )
                            _ = predictor.predictor.handle_request(
                                request=dict(
                                    type="close_session",
                                    session_id=predictor.RUNTIME['session_id'],
                                )
                            )

                        with torch.autocast("cuda", enabled=False):
                            predictor.on_4d_generation(frame_list, seq_path=seq_path, kps_list=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on Harmony4D")
    parser.add_argument("--data_dir", type=str, default="path to Harmony4D data",
        help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="path to output",
        help="Path to the output directory")
    parser.add_argument("--refine_occlusion", action="store_true",
        help="Whether to use occlusion-aware refinement (default False)")
    args = parser.parse_args()

    inference(args)
