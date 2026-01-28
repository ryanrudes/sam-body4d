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

from mhr_smpl_conversion.smpl2obj import smpl2obj

from eval_utils.dataset_3dpw import ThreedpwSmplFullSeqDataset
from eval_utils.metric_3dpw import MetricMocap
from eval_utils.geo.flip_utils import flip_smplx_params, avg_smplx_aa
from eval_utils.std import suppress_stdout_stderr
# from eval.flip_mhr import decode_joint_params

from eval_utils.smooth import postprocess_smpl_params
from eval.eval_utils.smooth_utils.geometry import (
    rot6d_to_rotation_matrix,
    smooth_with_savgol,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_rot6d,
    rotation_matrix_to_axis_angle,
    smooth_with_slerp,
)


def dict_numpy_to_torch(
    x: Dict[str, object],
    *,
    device=None,
    dtype_map=None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}

    for k, v in x.items():
        # ndarray
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            if dtype_map and v.dtype in dtype_map:
                t = t.to(dtype=dtype_map[v.dtype])

        # numpy scalar: np.float32 / np.int64 / np.float_
        elif isinstance(v, np.generic):
            t = torch.tensor(v)   # 0-dim tensor
            if dtype_map and v.dtype in dtype_map:
                t = t.to(dtype=dtype_map[v.dtype])

        else:
            raise TypeError(
                f"Key '{k}': unsupported type {type(v)}, expected np.ndarray or np.generic"
            )

        if device is not None:
            t = t.to(device)

        out[k] = t

    return out


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
    # os.makedirs(save_path, exist_ok=True)

    # init dataset and metric evaluator
    dataset_3dpw = ThreedpwSmplFullSeqDataset(label_path=args.label_path)
    metric_3dpw = MetricMocap(body_model_path=args.body_model_path)

    # init smpl & mhr models for conversion
    mhr_model = MHR.from_files(folder=Path(f"{args.body_model_path}/assets"), lod=1, device=torch.device("cuda"))

    smpl_model = smplx.SMPLX(model_path=f"{args.body_model_path}/smplx", gender="neutral").cuda()
    converter = Conversion(
        mhr_model=mhr_model, smpl_model=smpl_model, method="pytorch"
    )

    # smpl_model = make_smplx("smpl", gender="male")
    # converter = Conversion(
    #     mhr_model=mhr_model, smpl_model=smpl_model.bm, method="pytorch"
    # )

    for i in tqdm(range(len(dataset_3dpw)), desc="Processing 3DPW"):
        meta_data = dataset_3dpw[i]
        seq_name, obj_id = meta_data['meta']['vid'].rsplit("_", 1)
        for k, v in meta_data.items():
            if isinstance(v, torch.Tensor):
                meta_data[k] = v.cuda()
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, torch.Tensor):
                        v[kk] = vv.cuda()

        # if 'downtown_cafe' not in seq_name or obj_id != '0':
        #     continue

        # for seq in tqdm(seq_list):
        mhr_list = glob.glob(os.path.join(f"{result_path}/{seq_name}/mhr_params", "*data.npz"))
        mhr_list.sort()

        z = np.load(mhr_list[0], allow_pickle=True)
        zdata = z["data"]
        num_objs = len(zdata)

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
        
        # mhr_cam_t = np.stack(mhr_cam_t_list, axis=0)
        # verts = mhr_vertices  # [B, V, 3], unit: cm
        # y_max = verts[:, :, 1].max(axis=1)
        # y_min = verts[:, :, 1].min(axis=1)
        # mhr_height = y_max - y_min  # [B], cm
        # mhr_vertices_kept = mhr_vertices[meta_data['mask'].cpu().numpy()]

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

        # # Boolean mask of shape (T,)
        # mask = meta_data["mask"]
        # if not torch.is_tensor(mask):
        #     mask = torch.as_tensor(mask)
        # mask = mask.to(dtype=torch.bool, device=mhr_vertices.device)

        # T = mask.numel()
        # i = 0

        # # Will store the full (T, N) tensors for each parameter key
        # full_result_parameters = None  # dict[str, Tensor]

        # # Progress bar over time index
        # pbar = tqdm(
        #     total=T,
        #     desc="Converting contiguous mask segments (MHR → SMPL)",
        #     unit="frame",
        #     dynamic_ncols=True,
        # )

        # while i < T:
        #     # Skip frames where mask is False
        #     if not mask[i]:
        #         i += 1
        #         pbar.update(1)
        #         continue

        #     # Find a contiguous True segment [s, e]
        #     s = i
        #     while i < T and mask[i]:
        #         i += 1
        #     e = i - 1  # inclusive

        #     # Update progress by the length of this segment
        #     pbar.update(e - s + 1)

        #     # Run conversion only on this contiguous segment
        #     with suppress_stdout_stderr():
        #         res = converter.convert_mhr2smpl(
        #             mhr_vertices=mhr_vertices[s:e+1],   # (K, V, 3), K = e - s + 1
        #             single_identity=False,
        #             is_tracking=False,
        #             return_smpl_meshes=False,
        #             return_smpl_parameters=True,
        #             return_smpl_vertices=True,
        #             return_fitting_errors=True,
        #             batch_size=256,
        #         )

        #     seg_params = res.result_parameters
        #     K = e - s + 1

        #     # Initialize the full-size result dict on the first segment
        #     if full_result_parameters is None:
        #         full_result_parameters = {}
        #         for key, value in seg_params.items():
        #             # Allocate (T, N) and fill with zeros for mask == False frames
        #             N = value.shape[1]
        #             full_result_parameters[key] = torch.zeros(
        #                 (T, N), device=value.device, dtype=value.dtype
        #             )

        #     # Copy this segment's results back to their original time positions
        #     for key, value in seg_params.items():
        #         assert value.shape[0] == K
        #         full_result_parameters[key][s:e+1] = value

        # # Build the final result:
        # # reuse other fields from the last segment result,
        # # but replace result_parameters with the full (T, N) version
        # smpl_params = full_result_parameters

        # smpl_params = conversion_results.result_parameters
        # smpl_vertices = conversion_results.result_vertices

        # B, V, C = smpl_vertices.shape
        # data_filled = np.zeros((len(mhr_vertices), V, C), dtype=smpl_vertices.dtype)
        # data_filled[meta_data['mask'].cpu().numpy()] = smpl_vertices

        del smpl_params['left_hand_pose']
        del smpl_params['right_hand_pose']
        del smpl_params['expression']

        if args.result_path_flip != "":
            # process flip results
            result_path_flip = args.result_path_flip
            mhr_list_flip = glob.glob(os.path.join(f"{result_path_flip}/{seq_name}/mhr_params", "*data.npz"))
            mhr_list_flip.sort()
            mhr_vertices_list_flip = []
            for j in range(len(mhr_list_flip)):
                z = np.load(mhr_list_flip[j], allow_pickle=True)
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
                    mhr_vertices_list_flip.append(mhr_vertices[0])
                except Exception as e:
                    print(e)
                    mhr_vertices_list_flip.append(mhr_vertices_list_flip[-1])
        
            mhr_vertices_flip = np.stack(mhr_vertices_list_flip, axis=0)
            
            # i = 0
            # full_result_parameters = None  # dict[str, Tensor]
            # while i < T:
            #     # Skip frames where mask is False
            #     if not mask[i]:
            #         i += 1
            #         pbar.update(1)
            #         continue

            #     # Find a contiguous True segment [s, e]
            #     s = i
            #     while i < T and mask[i]:
            #         i += 1
            #     e = i - 1  # inclusive

            #     # Update progress by the length of this segment
            #     pbar.update(e - s + 1)

            #     # Run conversion only on this contiguous segment
            #     with suppress_stdout_stderr():
            #         res = converter.convert_mhr2smpl(
            #             mhr_vertices=mhr_vertices_flip[s:e+1],   # (K, V, 3), K = e - s + 1
            #             single_identity=False,
            #             is_tracking=False,
            #             return_smpl_meshes=False,
            #             return_smpl_parameters=True,
            #             return_smpl_vertices=True,
            #             return_fitting_errors=True,
            #             batch_size=256,
            #         )

            #     seg_params = res.result_parameters
            #     K = e - s + 1

            #     # Initialize the full-size result dict on the first segment
            #     if full_result_parameters is None:
            #         full_result_parameters = {}
            #         for key, value in seg_params.items():
            #             # Allocate (T, N) and fill with zeros for mask == False frames
            #             N = value.shape[1]
            #             full_result_parameters[key] = torch.zeros(
            #                 (T, N), device=value.device, dtype=value.dtype
            #             )

            #     # Copy this segment's results back to their original time positions
            #     for key, value in seg_params.items():
            #         assert value.shape[0] == K
            #         full_result_parameters[key][s:e+1] = value

            # smpl_params2 = full_result_parameters

            with suppress_stdout_stderr():
                conversion_results = converter.convert_mhr2smpl(
                    mhr_vertices=mhr_vertices_flip,
                    single_identity=False,
                    is_tracking=False,
                    return_smpl_meshes=False,
                    return_smpl_parameters=True,
                    return_smpl_vertices=False,
                    return_fitting_errors=False,
                    batch_size=256,
                    # batch_size=len(mhr_vertices),
                )
                smpl_params2 = conversion_results.result_parameters
            # smpl_vertices = conversion_results.result_vertices
            

            del smpl_params2['left_hand_pose']
            del smpl_params2['right_hand_pose']
            del smpl_params2['expression']

            smpl_params2 = flip_smplx_params(smpl_params2)

            smpl_params_avg = smpl_params.copy()
            smpl_params_avg["betas"] = (smpl_params["betas"] + smpl_params2["betas"]) / 2
            smpl_params_avg["body_pose"] = avg_smplx_aa(smpl_params["body_pose"], smpl_params2["body_pose"])
            smpl_params_avg["global_orient"] = avg_smplx_aa(
                smpl_params["global_orient"], smpl_params2["global_orient"]
            )
            smpl_params = smpl_params_avg

        smpl_params = postprocess_smpl_params(
            smpl_params,
        )

        # torch.save(smpl_params, f'{save_path}/{seq_name}_{obj_id}_tensor_dict.pth')
        metric_3dpw.evaluate(smpl_params, meta_data, None, None, save_path, smpl_model)
        # metric_3dpw.evaluate(smpl_params, meta_data, mhr_height, data_filled)
        # smpl2obj(smpl_model, smpl_params['body_pose'], 'mhr2smpl1.obj'), verify the conversion
