import argparse
import os, sys, glob
from tqdm import tqdm
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# from offline_app import inference configs
from offline_app_mask_kp import *

from eval.eval_utils.emdb.utils import EMDB1_LIST, EMDB1_NAMES


def inference(args):
    # init configs and cover with cmd options
    predictor = OfflineApp()

    # init data
    test_seq_root = os.path.join(args.data_dir, 'EMDB_ROOT')
    test_seq_path_list = [os.path.join(test_seq_root, '/'.join(tn.split('/')[:2])) for tn in EMDB1_LIST]
    test_seq_path_list.sort()
    test_seq_name_list = ['_'.join(tn.rstrip('/').split(os.sep)[-2:]) for tn in test_seq_path_list]
    
    labels = torch.load(f"{args.data_dir}/hmr4d_support/emdb_vit_v4.pt")
    batch_size = 1000

    # inference
    for seq, seq_path in tqdm(zip(test_seq_name_list, test_seq_path_list)):

        bbx_xys = labels[seq]['bbx_xys']
        bboxes = torch.cat(
            [bbx_xys[:, :2] - bbx_xys[:, 2:3] / 2,
            bbx_xys[:, :2] + bbx_xys[:, 2:3] / 2],
            dim=1
        )
        kp = labels[seq]['kp2d']

        # 0. init outputs
        output_dir = os.path.join(args.output_dir, seq)
        predictor.OUTPUT_DIR = output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)
        frame_list = glob.glob(os.path.join(seq_path, 'images', '*.jpg'))
        frame_list.sort()

        box_list = []
        kp_list = []
        for obj_id in range(1):
            try:
                box_list.append(bboxes)
                kp_list.append(kp)
            except:
                break
            # predictor.RUNTIME['bboxes'] = bboxes[seq_name_with_id]['bbx_xyxy']
        predictor.RUNTIME['bboxes'] = box_list

        one_frame = Image.open(frame_list[0]).convert('RGB')
        width, height = one_frame.size
        batch_size = len(frame_list)

        for i in range(0, len(frame_list), batch_size):
            batch_frames = frame_list[i:i + batch_size]
            batch_frames = [Image.open(bf).convert("RGB") for bf in batch_frames]
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
            predictor.RUNTIME['out_obj_ids'] = [1]
            num_objects = 1
            ann_frame_idx = i

            if i == 0:    
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
                    out = response["outputs"]
                    
                    # only focus on target person
                    obj_dict = {}   # key: inference_id (start from 1), value: sam_id
                    obj_list = []
                    for obj_id in range(num_objects):
                        seq_name_with_id = f'{seq}_{obj_id}'
                        kp_obj_id = kp[0].numpy()*ratio # 17 x 3
                        for out_obj_id in out['out_obj_ids']:
                            if majority_keypoints_in_mask(kp_obj_id, out['out_binary_masks'][out_obj_id]):
                                obj_dict[obj_id+1] = out_obj_id.item()
                                obj_list.append(out_obj_id.item())
                        predictor.RUNTIME['out_obj_ids'].append(obj_id+1)
                
                    # # segment on all frames
                    for out_id in out['out_obj_ids']:
                        if out_id.item() in obj_list:
                            continue
                        response = predictor.predictor.handle_request(
                            request=dict(
                                type="remove_object",
                                session_id=predictor.RUNTIME['session_id'],
                                obj_id=out_id.item(),
                            )
                        )
                        
                    outputs_per_frame = propagate_in_video(predictor.predictor, predictor.RUNTIME['session_id'], max_num_objects=num_objects)
            else:
                # use previous frame masks as box for other frames
                pass

            # 3. save masks
            predictor.save_masks(
                start_frame_idx=i, 
                outputs_per_frame=outputs_per_frame, 
                obj_dict=obj_dict, 
                resized_batch_frames=resized_batch_frames,
                original_size=(width, height),
            )
        
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
        
        # 4. hmr upon masks
        with torch.autocast("cuda", enabled=False):
            predictor.on_4d_generation(frame_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on EMDB-split-1")
    parser.add_argument("--data_dir", type=str, default="path to EMDB data",
        help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="path to output",
        help="Path to the output directory")
    parser.add_argument("--refine_occlusion", action="store_true",
        help="Whether to use occlusion-aware refinement (default False)")
    parser.add_argument("--flip", action="store_true",
        help="Whether to conduct inference with horizontal flip (default False)")
    args = parser.parse_args()

    inference(args)
