import argparse
import os, sys, glob
from tqdm import tqdm
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# from offline_app import inference configs
from offline_app_mask_kp import *
# from eval.eval_utils.kp_in_mask import majority_keypoints_in_mask


def inference(args):
    # init configs and cover with cmd options
    predictor = OfflineApp()

    # init data
    test_seq_name_list = glob.glob(os.path.join(args.data_dir, 'sequenceFiles', 'test', '*'))
    test_seq_name_list.sort()
    test_seq_name_list = [os.path.splitext(os.path.basename(tn))[0] for tn in test_seq_name_list]
    bboxes = torch.load(os.path.join(args.data_dir, 'body4d_3dpw_bbx_xyxy_uint16.pt'))
    kp = torch.load(os.path.join(args.data_dir, 'body4d_3dpw_vid2kp2d.pt'))

    # inference
    for seq in tqdm(test_seq_name_list):
        # 0. init outputs
        # if seq!='downtown_walking_00':
        #     continue
        output_dir = os.path.join(args.output_dir, seq)
        predictor.OUTPUT_DIR = output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)
        frame_list = glob.glob(os.path.join(args.data_dir, 'imageFiles', seq, '*.jpg'))
        seq_path = os.path.join(args.data_dir, 'imageFiles', seq)
        frame_list.sort()
        one_frame = Image.open(frame_list[0]).convert('RGB')
        width, height = one_frame.size

        # explore the num of objects
        num_objects = 0
        num_frames = len(frame_list)
        box_list = []
        for obj_id in range(3):
            seq_name_with_id = f'{seq}_{obj_id}'
            try:
                bbox = bboxes[seq_name_with_id]['bbx_xyxy'][0].numpy()
                box_list.append(bbox)
                num_objects += 1
            except:
                break
        kps_list = []
        for obj_id in range(num_objects):
            try:
                seq_name_with_id = f'{seq}_{obj_id}'
                kps_list.append(kp[seq_name_with_id])
            except:
                break
        if len(kps_list) == 0:
            kps_list = None

        # TODO: avoid oom
        mid = num_frames // 2
        if seq == 'downtown_runForBus_01':
            mid = 500
        if seq == 'downtown_walking_00':
            mid = num_frames - 100
        frame_lista = frame_list[:mid]
        frame_listb = frame_list[mid:]

        # first half
        batch_frames = frame_lista
        batch_frames = [Image.open(bf).convert("RGB") for bf in batch_frames]
        resized_batch_frames = resize_images_longest_side(batch_frames)
        ratio = resized_batch_frames[0].size[-1] / batch_frames[0].size[-1]
        
        # 1. load bbox (first frame)
        prompt_text_str = "person"
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # initialise and reset predictor state
            response = predictor.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=resized_batch_frames[:mid],
                )
            )
            predictor.RUNTIME['session_id'] = response["session_id"]
            predictor.RUNTIME['out_obj_ids'] = []
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
            obj_dicta = {}   # key: inference_id (start from 1), value: sam_id
            obj_list = []
            for obj_id in range(num_objects):
                seq_name_with_id = f'{seq}_{obj_id}'
                kp_obj_id = kp[seq_name_with_id][0].numpy()*ratio # 17 x 3
                for out_obj_id in out['out_obj_ids']:
                    if majority_keypoints_in_mask(kp_obj_id, out['out_binary_masks'][out_obj_id]):
                        obj_dicta[obj_id+1] = out_obj_id.item()
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
            outputs_per_framea = propagate_in_video(predictor.predictor, predictor.RUNTIME['session_id'], max_num_objects=num_objects)
            if predictor.RUNTIME['session_id'] is not None:
                _ = predictor.predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=predictor.RUNTIME['session_id'],
                    )
                )
            # predictor.save_masks(
            #     start_frame_idx=0, 
            #     outputs_per_frame=outputs_per_framea, 
            #     obj_dict=obj_dicta, 
            #     resized_batch_frames=resized_batch_frames[:mid],
            #     original_size=(width, height),
            # )
            # save frame feats


            # second half
            frame_listb = frame_listb[::-1]
            batch_frames = frame_listb
            batch_frames = [Image.open(bf).convert("RGB") for bf in batch_frames]
            resized_batch_frames = resize_images_longest_side(batch_frames)
            ratio = resized_batch_frames[0].size[-1] / batch_frames[0].size[-1]
            response = predictor.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=resized_batch_frames,
                )
            )
            predictor.RUNTIME['session_id'] = response["session_id"]
            predictor.RUNTIME['out_obj_ids'] = []
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
            obj_dictb = {}   # key: inference_id (start from 1), value: sam_id
            obj_list = []
            for obj_id in range(num_objects):
                seq_name_with_id = f'{seq}_{obj_id}'
                kp_obj_id = kp[seq_name_with_id][-1].numpy()*ratio # 17 x 3
                for out_obj_id in out['out_obj_ids']:
                    if majority_keypoints_in_mask(kp_obj_id, out['out_binary_masks'][out_obj_id]):
                        obj_dictb[obj_id+1] = out_obj_id.item()
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
            outputs_per_frameb = propagate_in_video(predictor.predictor, predictor.RUNTIME['session_id'], max_num_objects=num_objects)
            outputs_per_frameb = {num_frames - k - 1: v for k, v in reversed(outputs_per_frameb.items())}
            if predictor.RUNTIME['session_id'] is not None:
                _ = predictor.predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=predictor.RUNTIME['session_id'],
                    )
                )

            # 3. save masks
            # predictor.save_masks(
            #     start_frame_idx=mid, 
            #     outputs_per_frame=outputs_per_frameb, 
            #     obj_dict=obj_dictb, 
            #     resized_batch_frames=resized_batch_frames[::-1],
            #     original_size=(width, height),
            # )

            feats_a = [o['feature_cache'] for _, o in outputs_per_framea.items()]
            feats_b = [o['feature_cache'] for _, o in reversed(outputs_per_frameb.items())]
            np.savez_compressed(os.path.join(args.output_dir, seq, 'feature_cache.npz'), data=np.concatenate(feats_a + feats_b, axis=0))

        # 4. hmr upon masks

        # with torch.autocast("cuda", enabled=False):
        #     predictor.on_4d_generation(frame_list, seq_path=seq_path, kps_list=kps_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on 3DPW")
    parser.add_argument("--data_dir", type=str, default="path to 3DPW data",
        help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="path to output",
        help="Path to the output directory")
    parser.add_argument("--refine_occlusion", action="store_true",
        help="Whether to use occlusion-aware refinement (default False)")
    args = parser.parse_args()

    inference(args)
