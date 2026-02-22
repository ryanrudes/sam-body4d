import os, glob, sys
import torch
import smplx

import numpy as np
from tqdm import tqdm
from typing import Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(os.path.join(os.path.dirname(current_dir), 'mhr_smpl_conversion'))
sys.path.append(os.path.join(current_dir, 'eval_utils'))

from mhr_smpl_conversion.mhr.mhr import MHR
from mhr_smpl_conversion.conversion import Conversion
from pathlib import Path

from eval_utils.metric_harmony4d import MetricMocap
from eval_utils.std import suppress_stdout_stderr

from eval_utils.smooth import postprocess_smpl_params


def stack_dict_list(dict_list):
    """
    输入:
        dict_list: List[Dict[str, np.ndarray]]
            每个字典包含相同的 4 个 key，每个 value 是 numpy array

    输出:
        result: Dict[str, np.ndarray]
            每个 key 对应所有 value 的 np.stack 结果
    """
    if len(dict_list) == 0:
        return {}

    keys = dict_list[0].keys()

    result = {}
    for k in keys:
        if k in ['global_orient', 'transl', 'body_pose', 'betas']:
            result[k] = torch.from_numpy(np.stack([d[k] for d in dict_list], axis=0)).cuda()

    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        type=str,
        default="",
        help="Root directory of input results"
    )
    parser.add_argument(
        "--result_path_flip",
        type=str,
        default="",
        help="Root directory of input results"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Root directory to save outputs"
    )
    parser.add_argument(
        "--body_model_path",
        type=str,
        default="",
        help="Root directory of the mhr/smpl models"
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="",
        help="Root directory of the labels"
    )

    args = parser.parse_args()

    result_path = args.result_path
    save_path = args.save_path
    if args.save_path == '':
        save_path = None

    metric_harmony4d = MetricMocap(body_model_path=args.body_model_path)
    # init smpl & mhr models for conversion
    mhr_model = MHR.from_files(folder=Path(f"{args.body_model_path}/assets"), lod=1, device=torch.device("cuda"))
    smpl_model = smplx.SMPLX(model_path=f"{args.body_model_path}/smplx", gender="neutral").cuda()
    converter = Conversion(
        mhr_model=mhr_model, smpl_model=smpl_model, method="pytorch"
    )

    # init dataset and metric evaluator
    seq_list_1 = glob.glob(os.path.join(args.label_path, '*'))
    seq_list_1.sort()
    for seq_1 in tqdm(seq_list_1):
        seq_1_name = os.path.basename(seq_1)
        if os.path.isdir(seq_1):
            seq_list_2 = glob.glob(os.path.join(seq_1, '*'))
            seq_list_2.sort()
            for seq_2 in seq_list_2:
                seq_2_name = os.path.basename(seq_2)
                if os.path.isdir(seq_2):
                    np_target_smpl_path_list = glob.glob(os.path.join(seq_2, 'processed_data', 'smpl', '*'))
                    np_target_smpl_path_list.sort()
                    # for each cam
                    smpl_target_list_01 = []
                    smpl_target_list_02 = []
                    for np_target_smpl_path in np_target_smpl_path_list:
                        smpl_target_list_01.append(np.load(np_target_smpl_path, allow_pickle=True).item()['aria01'])
                        smpl_target_list_02.append(np.load(np_target_smpl_path, allow_pickle=True).item()['aria02'])

                    smpl_target_01 = stack_dict_list(smpl_target_list_01)
                    smpl_target_02 = stack_dict_list(smpl_target_list_02)
                    smpl_target = [smpl_target_01, smpl_target_02]

                    # Load predictions for each cam
                    camera_list = glob.glob(os.path.join(args.result_path, seq_1_name, seq_2_name, '*'))
                    camera_list.sort()
                    # for each cam
                    for cam in tqdm(camera_list):
                        cam_name = os.path.basename(cam)

                        mhr_list = glob.glob(os.path.join(f"{cam}/mhr_params", "*data.npz"))
                        mhr_list.sort()

                        z = np.load(mhr_list[0], allow_pickle=True)
                        zdata = z["data"]
                        num_objs = len(zdata)

                        for obj_id in range(num_objs):
                            mhr_vertices_list = []
                            for j in range(len(mhr_list)):
                                z = np.load(mhr_list[j], allow_pickle=True)
                                try:
                                    data = z["data"][int(obj_id)]
                                    # If pred_vertices is not available, use the mhr_model_params to compute the target vertices
                                    mhr_parameters = {}
                                    concatenated_sam3d_outputs = data
                                    mhr_parameters["lbs_model_params"] = torch.from_numpy(np.expand_dims(concatenated_sam3d_outputs[
                                        "mhr_model_params"
                                    ], axis=0)).cuda()
                                    mhr_parameters["identity_coeffs"] = torch.from_numpy(np.expand_dims(concatenated_sam3d_outputs[
                                        "shape_params"
                                    ], axis=0)).cuda()
                                    mhr_parameters["face_expr_coeffs"] = torch.from_numpy(np.expand_dims(concatenated_sam3d_outputs[
                                        "expr_params"
                                    ], axis=0)).cuda()
                                    mhr_verts, _ = converter._mhr_model(
                                        identity_coeffs=mhr_parameters["identity_coeffs"],
                                        model_parameters=mhr_parameters["lbs_model_params"],
                                        face_expr_coeffs=mhr_parameters["face_expr_coeffs"],
                                        apply_correctives=True,
                                    )
                                    mhr_vertices = mhr_verts.detach().cpu().numpy()
                                    mhr_vertices[..., [1, 2]] *= -1  # Camera system difference in SAM3D-Body
                                    mhr_vertices += 100.0 * concatenated_sam3d_outputs["pred_cam_t"]
                                    mhr_vertices_list.append(mhr_vertices[0])
                                except Exception as e:
                                    # print(e)
                                    mhr_vertices_list.append(mhr_vertices_list[-1])
                        
                            mhr_vertices = np.stack(mhr_vertices_list, axis=0)

                            with suppress_stdout_stderr():
                                conversion_results = converter.convert_mhr2smpl(
                                    # mhr_vertices=mhr_vertices_kept,
                                    mhr_vertices=mhr_vertices,
                                    # mhr_parameters=mhr_params,
                                    single_identity=False,
                                    is_tracking=False,
                                    return_smpl_meshes=False,
                                    return_smpl_parameters=True,
                                    return_smpl_vertices=False,
                                    return_fitting_errors=False,
                                    # batch_size=len(mhr_vertices),
                                    batch_size=256,
                                )
                                smpl_params = conversion_results.result_parameters

                            del smpl_params['left_hand_pose']
                            del smpl_params['right_hand_pose']
                            del smpl_params['expression']

                            smpl_params = postprocess_smpl_params(
                                smpl_params,
                            )

                            # torch.save(smpl_params, f'{save_path}/{seq_name}_{obj_id}_tensor_dict.pth')
                            metric_harmony4d.evaluate(smpl_params, smpl_target[obj_id], None, None, save_path, None, vid=f"{seq_1_name}_{seq_2_name}_{cam_name}_{obj_id}")
