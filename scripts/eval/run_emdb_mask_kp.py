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
    predictor = OfflineApp(use_detector=True)

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

        for i in range(0, len(frame_list), batch_size):
            batch_frames = frame_list[i:i + batch_size]
            inference_state = predictor.predictor.init_state(video_path=batch_frames)
            predictor.predictor.clear_all_points_in_video(inference_state)
            predictor.RUNTIME['inference_state'] = inference_state
            predictor.RUNTIME['out_obj_ids'] = []

            ann_frame_idx = i

            # 1. load bbox (first frame)
            for obj_id in range(1):
                seq_name_with_id = f'{seq}_{obj_id}'
                try:
                    # only consider the first frame bbox
                    bbox = bboxes[ann_frame_idx].numpy()

                    one_frame = np.array(Image.open(frame_list[i]).convert('RGB'))
                    # image = np.array(read_frame_at(args.input_video, starting_frame_idx))
                    outputs = predictor.sam3_3d_body_model.process_one_image(one_frame, bbox_thr=0.6,)
                    # 1. load bbox (first frame)
                    # select the largest box if multiple boxes are detected
                    best_output = max(
                        outputs,
                        key=lambda o: (o['bbox'][2] - o['bbox'][0]) * (o['bbox'][3] - o['bbox'][1])
                    )
                    xmin, ymin, xmax, ymax = best_output['bbox']
                    rel_box = np.array(
                        [[xmin / width, ymin / height, xmax / width, ymax / height]],
                        dtype=np.float32
                    )

                    # # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
                    # box = np.array([bbox], dtype=np.float32)
                    # rel_box = [[xmin / width, ymin / height, xmax / width, ymax / height] for xmin, ymin, xmax, ymax in box]
                    # rel_box = np.array(rel_box, dtype=np.float32)
                    _, predictor.RUNTIME['out_obj_ids'], low_res_masks, video_res_masks = predictor.predictor.add_new_points_or_box(
                        inference_state=predictor.RUNTIME['inference_state'],
                        frame_idx=0,
                        obj_id=obj_id+1,
                        box=rel_box,
                    )
                except:
                    break

            # 3. tracking
            predictor.on_mask_generation(start_frame_idx=i)
        # 4. hmr upon masks

        if len(kp_list) == 0:
            kp_list = None

        with torch.autocast("cuda", enabled=False):
            predictor.on_4d_generation(frame_list, kps_list=None)


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
