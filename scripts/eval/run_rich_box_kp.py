import argparse
import os, sys, glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# from offline_app import inference configs
from offline_app_box_kp import *


def inference(args):
    # init configs and cover with cmd options
    predictor = offline_app(refine_occlusion=args.refine_occlusion)

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

        bbx_xys = preproc_data[seq]['bbx_xys']
        bboxes = torch.cat(
            [bbx_xys[:, :2] - bbx_xys[:, 2:3] / 2,
            bbx_xys[:, :2] + bbx_xys[:, 2:3] / 2],
            dim=1
        )
        kp = preproc_data[seq]['kp2d']

        # 0. init outputs
        output_dir = os.path.join(args.output_dir, seq)
        predictor.OUTPUT_DIR = output_dir
        os.makedirs(predictor.OUTPUT_DIR, exist_ok=True)

        box_list = []
        kp_list = []
        for obj_id in range(1):
            try:
                box_list.append(bboxes)
                kp_list.append(kp)
            except:
                break
        
        predictor.RUNTIME['bboxes'] = box_list
        with torch.autocast("cuda", enabled=False):
            predictor.on_4d_generation(frame_list, box_list, kp_list, args.flip)


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
