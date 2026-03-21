import argparse
import os
import sys
import glob
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import smplx

from mhr.mhr import MHR
from conversion import Conversion
from smooth import postprocess_smpl_params


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(os.path.join(os.path.dirname(current_dir), "mhr_smpl_conversion"))


@contextmanager
def suppress_stdout_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)


def mhr2smpl(body_model_path, mhr_param_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mhr_model = MHR.from_files(
        folder=Path(f"{body_model_path}/assets"),
        lod=1,
        device=device,
    )
    smpl_model = smplx.SMPLX(
        model_path=f"{body_model_path}/smplx",
        gender="neutral",
    ).to(device)

    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smpl_model,
        method="pytorch",
    )

    mhr_list = sorted(glob.glob(os.path.join(mhr_param_path, "*data.npz")))
    if len(mhr_list) == 0:
        raise FileNotFoundError(f"No '*data.npz' files found in: {mhr_param_path}")

    first_frame_data = np.load(mhr_list[0], allow_pickle=True)["data"]
    num_objs = len(first_frame_data)

    smpl_dict = {}

    for obj_id in range(1, num_objs + 1):
        empty_ids = []
        mhr_vertices_list = []

        for frame_id in range(len(mhr_list)):
            mhr_data = np.load(mhr_list[frame_id], allow_pickle=True)["data"]

            try:
                data = mhr_data[obj_id - 1]

                mhr_parameters = {
                    "lbs_model_params": torch.from_numpy(
                        np.expand_dims(data["mhr_model_params"], axis=0)
                    ).float().to(device),
                    "identity_coeffs": torch.from_numpy(
                        np.expand_dims(data["shape_params"], axis=0)
                    ).float().to(device),
                    "face_expr_coeffs": torch.from_numpy(
                        np.expand_dims(data["expr_params"], axis=0)
                    ).float().to(device),
                }

                mhr_verts, _ = converter._mhr_model(
                    identity_coeffs=mhr_parameters["identity_coeffs"],
                    model_parameters=mhr_parameters["lbs_model_params"],
                    face_expr_coeffs=mhr_parameters["face_expr_coeffs"],
                    apply_correctives=True,
                )

                mhr_vertices = mhr_verts.detach().cpu().numpy()
                mhr_vertices[..., [1, 2]] *= -1  # Camera system difference in SAM3D-Body
                mhr_vertices += 100.0 * data["pred_cam_t"]
                mhr_vertices_list.append(mhr_vertices[0])

            except Exception:
                empty_ids.append(frame_id)
                if len(mhr_vertices_list) == 0:
                    raise RuntimeError(
                        f"Object {obj_id} in frame {frame_id} failed, "
                        "and there is no previous frame to copy from."
                    )
                mhr_vertices_list.append(mhr_vertices_list[-1])

        mhr_vertices = np.stack(mhr_vertices_list, axis=0)

        with suppress_stdout_stderr():
            conversion_results = converter.convert_mhr2smpl(
                mhr_vertices=mhr_vertices,
                single_identity=False,
                is_tracking=False,
                return_smpl_meshes=False,
                return_smpl_parameters=True,
                return_smpl_vertices=False,
                return_fitting_errors=False,
                batch_size=256,
            )
            smpl_params = conversion_results.result_parameters

        smpl_params = postprocess_smpl_params(smpl_params)

        smpl_params['empty_ids'] = empty_ids

        smpl_dict[obj_id] = smpl_params

    return smpl_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run MHR -> SMPLX via backend env")

    parser.add_argument(
        "--body_model_path",
        type=str,
        default="/home/data/hmq/checkpoints/sam-body4d",
        # required=True,
        help="Path to body model root, e.g. /path/to/models",
    )

    parser.add_argument(
        "--mhr_path",
        type=str,
        default="/home/hmq/projects/hmr/sam-body4d/outputs/20260322_023212_319_b0599abc/mhr_params",
        # required=True,
        help="Path to MHR params directory",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output path for saved smplx params .pt. "
             "Default: parent directory of mhr_path + /smplx_params.pt",
    )

    args = parser.parse_args()

    body_model_path = args.body_model_path
    mhr_path = args.mhr_path
    save_path = (
        args.save_path
        if args.save_path is not None
        else os.path.join(os.path.dirname(mhr_path), "smplx_params.npz")
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    smpl_dict = mhr2smpl(
        body_model_path=body_model_path,
        mhr_param_path=mhr_path,
    )
    flat_dict = {}
    for obj_id, param_dict in smpl_dict.items():
        for param_name, value in param_dict.items():
            key = f"{obj_id}::{param_name}"
            if isinstance(value, torch.Tensor):
                flat_dict[key] = value.detach().cpu().numpy()
            else:
                flat_dict[key] = np.asarray(value)
    np.savez_compressed(save_path, **flat_dict)
    print(f"SMPLX parameters saved to: {save_path}")
