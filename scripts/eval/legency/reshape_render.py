import os, glob, sys
import torch
import smplx
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm
from typing import Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(os.path.join(os.path.dirname(current_dir), 'mhr_smpl_conversion'))
sys.path.append(os.path.join(current_dir, 'eval_utils'))

from eval.eval_utils.dataset_3dpw import ThreedpwSmplFullSeqDataset

result_root = "results/hmr/3DPW-box-real"
# os.makedirs(save_root, exist_ok=True)

# init dataset and metric evaluator
dataset_3dpw = ThreedpwSmplFullSeqDataset()


for i in tqdm(range(len(dataset_3dpw)), desc="Processing 3DPW"):
    meta_data = dataset_3dpw[i]
    seq_name, obj_id = meta_data['meta']['vid'].rsplit("_", 1)
    
    frame_list = glob.glob(os.path.join(result_root, seq_name, "rendered_frames", "*.jpg"))
    frame_list.sort()

    # A: shape [N]，bool tensor（True/False）
    # file_list: len == N，每个是文件路径（str 或 Path）
    A = meta_data['mask'].bool().cpu()
    assert A.ndim == 1
    assert len(frame_list) == int(A.numel())

    src_paths = [Path(p) for p in frame_list]

    # 只保留 True 对应的文件
    keep_paths = [p for p, m in zip(src_paths, A.tolist()) if m]

    out_dir = Path(os.path.join(result_root, seq_name, 'reorder', obj_id))  # 改成你的目标目录
    out_dir.mkdir(parents=True, exist_ok=True)

    # 复制并重命名：00000000.ext / 00000001.ext ...
    for new_idx, src in enumerate(keep_paths):
        if not src.exists():
            raise FileNotFoundError(f"Missing: {src}")

        ext = src.suffix  # 保留原扩展名
        dst = out_dir / f"{new_idx:04d}-{src.stem}{ext}"
        shutil.copy2(src, dst)  # 若想移动用 shutil.move(src, dst)
