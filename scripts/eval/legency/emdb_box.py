import argparse
import os, sys, glob
from tqdm import tqdm
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# from offline_app import inference configs
from offline_app_box_kp import *

from eval.eval_utils.emdb.utils import EMDB1_LIST, EMDB1_NAMES


def inference(args):
    # init configs and cover with cmd options
    predictor = offline_app(refine_occlusion=args.refine_occlusion)

    # init data
    test_seq_root = os.path.join(args.data_dir, 'EMDB_ROOT')
    test_seq_path_list = [os.path.join(test_seq_root, '/'.join(tn.split('/')[:2])) for tn in EMDB1_LIST]
    test_seq_path_list.sort()
    test_seq_name_list = ['_'.join(tn.rstrip('/').split(os.sep)[-2:]) for tn in test_seq_path_list]
    
    labels = torch.load(f"{args.data_dir}/hmr4d_support/emdb_vit_v4.pt")

    # bboxes = torch.load(os.path.join(args.data_dir, 'body4d_3dpw_bbx_xyxy_uint16.pt'))
    # kp = torch.load(os.path.join(args.data_dir, 'body4d_3dpw_vid2kp2d.pt'))

    # inference
    for seq, seq_path in tqdm(zip(test_seq_name_list, test_seq_path_list)):

        bbx_xys = labels[seq]['bbx_xys']
        bboxes = torch.cat(
            [bbx_xys[:, :2] - bbx_xys[:, 2:3] / 2,
            bbx_xys[:, :2] + bbx_xys[:, 2:3] / 2],
            dim=1
        )

        # 0. init outputs
        output_dir = os.path.join(args.output_dir, seq)
        predictor.OUTPUT_DIR = output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)
        frame_list = glob.glob(os.path.join(seq_path, 'images', '*.jpg'))
        frame_list.sort()
        one_frame = Image.open(frame_list[0]).convert('RGB')

        box_list = []
        for obj_id in range(1):
            try:
                seq_name_with_id = f'{seq}_{obj_id}'
                box_list.append(bboxes)
            except:
                break
            # predictor.RUNTIME['bboxes'] = bboxes[seq_name_with_id]['bbx_xyxy']
        predictor.RUNTIME['bboxes'] = box_list
        with torch.autocast("cuda", enabled=False):
            predictor.on_4d_generation(frame_list, box_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on EMDB-split-1")
    parser.add_argument("--data_dir", type=str, default="path to EMDB data",
        help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="path to output",
        help="Path to the output directory")
    parser.add_argument("--refine_occlusion", action="store_true",
        help="Whether to use occlusion-aware refinement (default False)")
    args = parser.parse_args()

    inference(args)
