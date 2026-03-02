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

from eval_utils.rich.rich_motion_test import RichSmplFullSeqDataset
from eval_utils.metric_rich import MetricMocap

from eval_utils.geo.flip_utils import flip_smplx_params, avg_smplx_aa
from eval_utils.std import suppress_stdout_stderr
from eval_utils.smooth import postprocess_smpl_params

from torch.utils.data import default_collate

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

    # init dataset and metric evaluator
    dataset_rich = RichSmplFullSeqDataset(label_path=args.label_path)
    metric_rich = MetricMocap(body_model_path=args.body_model_path)

    # init smpl & mhr models for conversion
    mhr_model = MHR.from_files(folder=Path(f"{args.body_model_path}/assets"), lod=1, device=torch.device("cuda"))
    smpl_model = smplx.SMPLX(model_path=f"{args.body_model_path}/smplx", gender="neutral")
    converter = Conversion(
        mhr_model=mhr_model, smpl_model=smpl_model, method="pytorch"
    )

    for i in tqdm(range(len(dataset_rich)), desc="Processing RICH"):
        meta_data = dataset_rich[i]
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
        try:
            num_objs = len(zdata)
        except:
            a = 1

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

        with suppress_stdout_stderr():
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

            with suppress_stdout_stderr():
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

        smpl_params = postprocess_smpl_params(
            smpl_params,
        )

        meta_data = [meta_data]
        return_dict = {}
        for k in meta_data[0].keys():
            if k.startswith("meta"):  # data information, do not batch
                return_dict[k] = [d[k] for d in meta_data]
            else:
                return_dict[k] = default_collate([d[k] for d in meta_data])
        return_dict["B"] = 1

        seq_name = seq_name.split('/')[0]
        torch.save(smpl_params, f'{seq_name}_{obj_id}_tensor_dict.pth')
        # metric_rich.evaluate(smpl_params, meta_data, None, None, save_path, smpl_model)
        metric_rich.evaluate(smpl_params, return_dict, None, None, save_path, None)
        # metric_3dpw.evaluate(smpl_params, meta_data, None, None, save_path, None)
