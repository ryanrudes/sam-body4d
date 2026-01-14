import argparse
import os, sys, glob
from tqdm import tqdm
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# from offline_app import inference configs
from offline_app_box_kp import *


def inference(args):
    # init configs and cover with cmd options
    predictor = offline_app(refine_occlusion=args.refine_occlusion)

    # init data
    test_seq_name_list = glob.glob(os.path.join(args.data_dir, 'sequenceFiles', 'test', '*'))
    test_seq_name_list.sort()
    # test_seq_name_list.sort(reverse=True)
    test_seq_name_list = [os.path.splitext(os.path.basename(tn))[0] for tn in test_seq_name_list]
    bboxes = torch.load(os.path.join(args.data_dir, 'body4d_3dpw_bbx_xyxy_uint16.pt'))
    kp = torch.load(os.path.join(args.data_dir, 'body4d_3dpw_vid2kp2d.pt'))

    # inference
    for seq in tqdm(test_seq_name_list):

        # 0. init outputs
        output_dir = os.path.join(args.output_dir, seq)
        predictor.OUTPUT_DIR = output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)
        frame_list = glob.glob(os.path.join(args.data_dir, 'imageFiles', seq, '*.jpg'))
        frame_list.sort()

        box_list = []
        kp_list = []
        for obj_id in range(3):
            try:
                seq_name_with_id = f'{seq}_{obj_id}'
                box_list.append(bboxes[seq_name_with_id]['bbx_xyxy'])
                kp_list.append(kp[seq_name_with_id])
            except:
                break
            # predictor.RUNTIME['bboxes'] = bboxes[seq_name_with_id]['bbx_xyxy']
        predictor.RUNTIME['bboxes'] = box_list
        predictor.RUNTIME['kps'] = kp_list
        with torch.autocast("cuda", enabled=False):
            predictor.on_4d_generation(frame_list, box_list, kp_list, flip=args.flip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on 3DPW")
    parser.add_argument("--data_dir", type=str, default="path to 3DPW data",
        help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="path to output",
        help="Path to the output directory")
    parser.add_argument("--refine_occlusion", action="store_true",
        help="Whether to use occlusion-aware refinement (default False)")
    parser.add_argument("--flip", action="store_true",
        help="Whether to use occlusion-aware refinement (default False)")
    args = parser.parse_args()

    inference(args)
