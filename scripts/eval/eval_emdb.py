import os, glob, sys
import torch
import smplx

import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(os.path.join(os.path.dirname(current_dir), 'mhr_smpl_conversion'))
sys.path.append(os.path.join(current_dir, 'eval_utils'))

from mhr_smpl_conversion.mhr.mhr import MHR
from mhr_smpl_conversion.conversion import Conversion
from pathlib import Path

from eval_utils.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
from eval_utils.metric_emdb import MetricMocap

from eval_utils.geo.flip_utils import flip_smplx_params, avg_smplx_aa

from eval_utils.smooth import smpl_to_smpl_decode_o6dp
from eval.eval_utils.smooth_utils.geometry import (
    rot6d_to_rotation_matrix,
    smooth_with_savgol,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_rot6d,
    rotation_matrix_to_axis_angle,
    smooth_with_slerp,
)

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
    dataset_emdb = EmdbSmplFullSeqDataset(label_path=args.label_path)
    metric_emdb = MetricMocap(body_model_path=args.body_model_path)

    # init smpl & mhr models for conversion
    mhr_model = MHR.from_files(folder=Path(f"{args.body_model_path}/assets"), lod=1, device=torch.device("cuda"))
    smpl_model = smplx.SMPLX(model_path=f"{args.body_model_path}/smplx", gender="neutral")
    converter = Conversion(
        mhr_model=mhr_model, smpl_model=smpl_model, method="pytorch"
    )

    for i in tqdm(range(len(dataset_emdb)), desc="Processing EMDB-split-1"):
        meta_data = dataset_emdb[i]
        seq_name, obj_id = meta_data['meta']['vid'], '1'
        for k, v in meta_data.items():
            if isinstance(v, torch.Tensor):
                meta_data[k] = v.cuda()
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, torch.Tensor):
                        v[kk] = vv.cuda()

        # if seq_name != 'P1_14_outdoor_climb': 
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
                data = z["data"][0]
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

        conversion_results = converter.convert_mhr2smpl(
            mhr_vertices=mhr_vertices,
            single_identity=False,
            is_tracking=False,
            return_smpl_meshes=False,
            return_smpl_parameters=True,
            return_smpl_vertices=True,
            return_fitting_errors=False,
            batch_size=256,
        )
        smpl_params = conversion_results.result_parameters

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
                    data = z["data"][0]
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

            conversion_results = converter.convert_mhr2smpl(
                mhr_vertices=mhr_vertices_flip,
                single_identity=False,
                is_tracking=False,
                return_smpl_meshes=False,
                return_smpl_parameters=True,
                return_smpl_vertices=True,
                return_fitting_errors=False,
                batch_size=256,
            )
            smpl_params2 = conversion_results.result_parameters

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

        # betas = smpl_params['betas']          # (N, 10)
        # betas_fixed = betas.clone()
        # betas_fixed[1:] = betas_fixed[0]      # 1..N-1 -> 0
        # smpl_params['betas'] = betas_fixed

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


        metric_emdb.evaluate(smpl_params, meta_data, args.body_model_path, None, None)
