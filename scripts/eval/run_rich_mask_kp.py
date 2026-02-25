import argparse
import os, sys, glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from offline_app_mask_kp import *


def inference(args):
    # init configs and cover with cmd options
    predictor = OfflineApp()

    # init data
    # Load evaluation protocol from WHAM labels
    rich_dir = Path(f"{args.data_dir}/hmr4d_support")
    labels = torch.load(rich_dir / "rich_test_labels.pt", weights_only=False)
    preproc_data = torch.load(rich_dir / "rich_test_preproc.pt", weights_only=False)
    vids = list(labels.keys())
    vids.sort()

    # inference
    for seq in tqdm(vids):
        seq_id = seq.split('_')[-1]
        frame_list = labels[seq]['frame_id']
        frame_list = [f"{args.data_dir}/{seq}/{fi:05d}_{seq_id}.jpeg" for fi in frame_list]
        frame_list.sort()
        
        if not os.path.exists(frame_list[0]):
        # if True:
            print(frame_list[0] + " does not exist")
            frame_file_list = glob.glob(f"{args.data_dir}/{seq}/*.jpeg")
            frame_file_list.sort()
            frame_list = [frame_file_list[i-1] for i in labels[seq]['frame_id'].tolist()]

        kp = preproc_data[seq]['kp2d']

        # 0. init outputs
        output_dir = os.path.join(args.output_dir, seq)
        predictor.OUTPUT_DIR = output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)

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

        kp_list = []
        for obj_id in range(1):
            try:
                kp_list.append(kp)
            except:
                break
        if len(kp_list) == 0:
            kp_list = None

        with torch.autocast("cuda", enabled=False):
            predictor.on_4d_generation(frame_list, kps_list=kp_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on RICH")
    parser.add_argument("--data_dir", type=str, default="path to RICH data",
        help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="path to output",
        help="Path to the output directory")
    parser.add_argument("--refine_occlusion", action="store_true",
        help="Whether to use occlusion-aware refinement (default False)")
    parser.add_argument("--flip", action="store_true",
        help="Whether to conduct inference with horizontal flip (default False)")
    args = parser.parse_args()

    inference(args)
