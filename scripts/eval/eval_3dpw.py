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

from eval_utils.smooth import smpl_to_smpl_decode_o6dp
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
    os.makedirs(save_path, exist_ok=True)

    # init dataset and metric evaluator
    dataset_3dpw = ThreedpwSmplFullSeqDataset(label_path=args.label_path)
    metric_3dpw = MetricMocap(body_model_path=args.body_model_path)

    # init smpl & mhr models for conversion
    mhr_model = MHR.from_files(folder=Path(f"{args.body_model_path}/assets"), lod=1, device=torch.device("cuda"))

    smpl_model = smplx.SMPLX(model_path=f"{args.body_model_path}/smplx", gender="neutral")
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
                print(e)
                mhr_vertices_list.append(mhr_vertices_list[-1])
    
        mhr_vertices = np.stack(mhr_vertices_list, axis=0)
        
        # mhr_cam_t = np.stack(mhr_cam_t_list, axis=0)
        # verts = mhr_vertices  # [B, V, 3], unit: cm
        # y_max = verts[:, :, 1].max(axis=1)
        # y_min = verts[:, :, 1].min(axis=1)
        # mhr_height = y_max - y_min  # [B], cm
        # mhr_vertices_kept = mhr_vertices[meta_data['mask'].cpu().numpy()]

        # conversion_results = converter.convert_mhr2smpl(
        #     # mhr_vertices=mhr_vertices_kept,
        #     mhr_vertices=mhr_vertices,
        #     # mhr_parameters=mhr_params,
        #     single_identity=False,
        #     is_tracking=False,
        #     return_smpl_meshes=False,
        #     return_smpl_parameters=True,
        #     return_smpl_vertices=False,
        #     return_fitting_errors=False,
        #     # batch_size=len(mhr_vertices),
        #     batch_size=256,
        # )

        # Boolean mask of shape (T,)
        mask = meta_data["mask"]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.to(dtype=torch.bool, device=mhr_vertices.device)

        T = mask.numel()
        i = 0

        # Will store the full (T, N) tensors for each parameter key
        full_result_parameters = None  # dict[str, Tensor]

        # Progress bar over time index
        pbar = tqdm(
            total=T,
            desc="Converting contiguous mask segments (MHR → SMPL)",
            unit="frame",
            dynamic_ncols=True,
        )

        while i < T:
            # Skip frames where mask is False
            if not mask[i]:
                i += 1
                pbar.update(1)
                continue

            # Find a contiguous True segment [s, e]
            s = i
            while i < T and mask[i]:
                i += 1
            e = i - 1  # inclusive

            # Update progress by the length of this segment
            pbar.update(e - s + 1)

            # Run conversion only on this contiguous segment
            with suppress_stdout_stderr():
                res = converter.convert_mhr2smpl(
                    mhr_vertices=mhr_vertices[s:e+1],   # (K, V, 3), K = e - s + 1
                    single_identity=False,
                    is_tracking=False,
                    return_smpl_meshes=False,
                    return_smpl_parameters=True,
                    return_smpl_vertices=True,
                    return_fitting_errors=True,
                    batch_size=256,
                )

            seg_params = res.result_parameters
            K = e - s + 1

            # Initialize the full-size result dict on the first segment
            if full_result_parameters is None:
                full_result_parameters = {}
                for key, value in seg_params.items():
                    # Allocate (T, N) and fill with zeros for mask == False frames
                    N = value.shape[1]
                    full_result_parameters[key] = torch.zeros(
                        (T, N), device=value.device, dtype=value.dtype
                    )

            # Copy this segment's results back to their original time positions
            for key, value in seg_params.items():
                assert value.shape[0] == K
                full_result_parameters[key][s:e+1] = value

        # Build the final result:
        # reuse other fields from the last segment result,
        # but replace result_parameters with the full (T, N) version
        smpl_params = full_result_parameters

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
            # out = mhr_vertices_flip.copy()
            # out[..., 0] = -out[..., 0]
            # mhr_vertices_flip = out

            # mhr_vertices = 0.5 * (mhr_vertices + mhr_vertices_flip)
            # mhr_vertices = mhr_vertices_flip

            # average the original and flip results
            # mhr_vertices_final = (mhr_vertices + mhr_vertices_flip[:, :, [0, 1, 2]] * np.array([[-1, 1, 1]])) / 2.0
            i = 0
            full_result_parameters = None  # dict[str, Tensor]
            while i < T:
                # Skip frames where mask is False
                if not mask[i]:
                    i += 1
                    pbar.update(1)
                    continue

                # Find a contiguous True segment [s, e]
                s = i
                while i < T and mask[i]:
                    i += 1
                e = i - 1  # inclusive

                # Update progress by the length of this segment
                pbar.update(e - s + 1)

                # Run conversion only on this contiguous segment
                with suppress_stdout_stderr():
                    res = converter.convert_mhr2smpl(
                        mhr_vertices=mhr_vertices_flip[s:e+1],   # (K, V, 3), K = e - s + 1
                        single_identity=False,
                        is_tracking=False,
                        return_smpl_meshes=False,
                        return_smpl_parameters=True,
                        return_smpl_vertices=True,
                        return_fitting_errors=True,
                        batch_size=256,
                    )

                seg_params = res.result_parameters
                K = e - s + 1

                # Initialize the full-size result dict on the first segment
                if full_result_parameters is None:
                    full_result_parameters = {}
                    for key, value in seg_params.items():
                        # Allocate (T, N) and fill with zeros for mask == False frames
                        N = value.shape[1]
                        full_result_parameters[key] = torch.zeros(
                            (T, N), device=value.device, dtype=value.dtype
                        )

                # Copy this segment's results back to their original time positions
                for key, value in seg_params.items():
                    assert value.shape[0] == K
                    full_result_parameters[key][s:e+1] = value

            # conversion_results = converter.convert_mhr2smpl(
            #     mhr_vertices=mhr_vertices_flip,
            #     single_identity=False,
            #     is_tracking=False,
            #     return_smpl_meshes=False,
            #     return_smpl_parameters=True,
            #     return_smpl_vertices=False,
            #     return_fitting_errors=False,
            #     batch_size=256,
            #     # batch_size=len(mhr_vertices),
            # )
            # smpl_params2 = conversion_results.result_parameters
            # smpl_vertices = conversion_results.result_vertices
            smpl_params2 = full_result_parameters

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

        # smooth_results = smpl_to_smpl_decode_o6dp(smpl_params['global_orient'].unsqueeze(0),smpl_params['body_pose'].unsqueeze(0),smpl_params['transl'].unsqueeze(0))
        # smpl_params['global_orient'] = smooth_results['global_orient'].squeeze(0)
        # smpl_params['body_pose'] = smooth_results['body_pose'].squeeze(0)
        # smpl_params['transl'] = smooth_results['transl'].squeeze(0)

        import math

        # Inputs:
        # mask: (L,) bool
        # smpl_params['global_orient']: (L, 3)
        # smpl_params['body_pose']:     (L, 63)
        # smpl_params['transl']:        (L, 3)
        # axis_angle_to_rotation_matrix
        # smpl_to_smpl_decode_o6dp

        SOFT_THR = 10.0   # degrees
        HARD_THR = 20.0   # degrees

        mask = meta_data["mask"].bool()
        L = mask.shape[0]

        # work on detached copies
        go = smpl_params["global_orient"].detach().clone()
        bp = smpl_params["body_pose"].detach().clone()
        tr = smpl_params["transl"].detach().clone()

        # --------------------------------------------------
        # 1) split into contiguous True segments
        # --------------------------------------------------
        segments = []
        start = None
        for i in range(L):
            if mask[i] and start is None:
                start = i
            elif (not mask[i]) and start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, L - 1))

        # --------------------------------------------------
        # helper: per-frame root rotation change (degrees)
        # --------------------------------------------------
        def root_angle_deg(go_seg):
            """
            go_seg: (T, 3) axis-angle
            return: (T-1,) degrees
            """
            R = axis_angle_to_rotation_matrix(go_seg)          # (T,3,3)
            R_rel = R[:-1].transpose(-1, -2) @ R[1:]           # (T-1,3,3)
            trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
            cos = (trace - 1.0) / 2.0
            cos = torch.clamp(cos, -1.0, 1.0)
            return torch.acos(cos) * (180.0 / math.pi)

        # --------------------------------------------------
        # 2) detect spikes in each segment and copy previous
        # --------------------------------------------------
        with torch.no_grad():
            for s, e in segments:
                T = e - s + 1
                if T < 3:
                    continue

                ang = root_angle_deg(go[s:e+1])   # (T-1,)

                for t in range(1, T - 1):
                    a_prev = ang[t - 1].item()
                    a_next = ang[t].item()

                    # spike pattern: one large jump, neighbor small
                    if (a_prev > HARD_THR and a_next < SOFT_THR) or \
                    (a_next > HARD_THR and a_prev < SOFT_THR):

                        abs_t = s + t
                        go[abs_t] = go[abs_t - 1]
                        bp[abs_t] = bp[abs_t - 1]
                        tr[abs_t] = tr[abs_t - 1]

        # write back de-spiked sequence
        smpl_params["global_orient"] = go
        smpl_params["body_pose"] = bp
        smpl_params["transl"] = tr

        # --------------------------------------------------
        # 3) final smoothing (your original 4 lines)
        # --------------------------------------------------
        with torch.no_grad():
            for s, e in segments:
                T = e - s + 1
                if T < 2:
                    continue

                # smpl_to_smpl_decode_o6dp internally uses savgol(window_length=11),
                # so only enable smoothing when the segment is long enough.
                use_smooth = (T >= 11)
                if not use_smooth:
                    continue

                smooth_results = smpl_to_smpl_decode_o6dp(
                    smpl_params["global_orient"][s:e+1].unsqueeze(0),
                    smpl_params["body_pose"][s:e+1].unsqueeze(0),
                    smpl_params["transl"][s:e+1].unsqueeze(0),
                    should_apply_smooothing=use_smooth,
                )

                smpl_params["global_orient"][s:e+1] = smooth_results["global_orient"].squeeze(0)
                smpl_params["body_pose"][s:e+1] = smooth_results["body_pose"].squeeze(0)
                smpl_params["transl"][s:e+1] = smooth_results["transl"].squeeze(0)

        # torch.save(smpl_params, f'{save_path}/{seq_name}_{obj_id}_tensor_dict.pth')
        metric_3dpw.evaluate(smpl_params, meta_data, None, None)
        # metric_3dpw.evaluate(smpl_params, meta_data, mhr_height, data_filled)
        # smpl2obj(smpl_model, smpl_params['body_pose'], 'mhr2smpl1.obj'), verify the conversion
