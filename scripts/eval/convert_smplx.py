import os, glob, sys
import torch
import smplx

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(os.path.join(os.path.dirname(current_dir), 'mhr_smpl_conversion'))
sys.path.append(os.path.join(current_dir, 'eval_utils'))

from mhr_smpl_conversion.mhr.mhr import MHR
from mhr_smpl_conversion.conversion import Conversion
from pathlib import Path

from eval_utils.metric_3dpw import MetricMocap
from eval_utils.std import suppress_stdout_stderr

from eval_utils.smooth import postprocess_smpl_params
from pathlib import Path
from smplx_utils import make_smplx
from eval_utils.video_io_utils import read_video_np, save_video
from eval_utils.vis.renderer_utils import simple_render_mesh_background
import numpy as np
import cv2
from PIL import Image


class VIS:
    def __init__(self, body_model_path):
        # SMPLX and SMPL
        self.smplx = make_smplx("supermotion_EVAL3DPW", body_model_path=body_model_path, )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.smpl = {"male": make_smplx("smpl", body_model_path=body_model_path, gender="male"), "female": make_smplx("smpl", body_model_path=body_model_path, gender="female")}
        self.smplx2smpl = torch.load(f"{script_dir}/eval_utils/body_model/smplx2smpl_sparse.pt")
        self.faces_smplx = self.smplx.faces
        self.faces_smpl = self.smpl["male"].faces

    # ================== Batch-based Computation  ================== #
    def evaluate(self, outputs, batch, mhr_height, smplx_vertices, save_path=None, smplx=None):
        """The behaviour is the same for val/test/predict"""
        # Move to cuda if not
        if smplx_vertices is not None:
            smplx_vertices = torch.tensor(smplx_vertices).cuda()
        self.smplx = self.smplx.cuda()
        self.smpl["male"] = self.smpl["male"].cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        smpl_out = self.smplx(**outputs)

        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        
        K = torch.tensor([[1.6529e+03, 0.0000e+00, 5.4000e+02],
        [0.0000e+00, 1.6529e+03, 9.6000e+02],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]])

        if True:  # Render incam (simple)
            images = glob.glob(os.path.join(save_path, 'images', '*.jpg'))
            images.sort()
            images = np.array([np.array(Image.open(img)) for img in images])
            render_dict = {
                "K": K[None],  # only support batch size 1
                "faces": self.smpl["male"].faces,
                "verts": pred_c_verts,
                "background": images,
            }
            img_overlay = simple_render_mesh_background(render_dict)
            output_fn = Path(f"{save_path}") / f"smpl.mp4"
            
            # imgs: numpy array of shape (L, H, W, 3)
            out_dir = "output_images"
            os.makedirs(out_dir, exist_ok=True)

            for i in range(img_overlay.shape[0]):
                img = img_overlay[i]  # H x W x 3, RGB
                img_bgr = img[:, :, ::-1]  # RGB -> BGR (for OpenCV)
                cv2.imwrite(os.path.join(out_dir, f"{i:04d}.JPG"), img_bgr)

            save_video(img_overlay, output_fn, crf=28)

        del smpl_out  # Prevent OOM


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
    save_path = result_path

    metric_3dpw = VIS(body_model_path=args.body_model_path)

    # init smpl & mhr models for conversion
    mhr_model = MHR.from_files(folder=Path(f"{args.body_model_path}/assets"), lod=1, device=torch.device("cuda"))

    smpl_model = smplx.SMPLX(model_path=f"{args.body_model_path}/smplx", gender="neutral").cuda()
    converter = Conversion(
        mhr_model=mhr_model, smpl_model=smpl_model, method="pytorch"
    )

    mhr_list = glob.glob(os.path.join(f"{result_path}/mhr_params", "*data.npz"))
    mhr_list.sort()

    z = np.load(mhr_list[0], allow_pickle=True)
    zdata = z["data"]
    num_objs = 1
    obj_id = 0

    mhr_vertices_list = []
    for j in range(len(mhr_list)):
        z = np.load(mhr_list[j], allow_pickle=True)
        try:
            if int(obj_id) == 1:
                data = z["data"][0]
            else:
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
            mhr_vertices=mhr_vertices,
            single_identity=True,
            is_tracking=True,
            return_smpl_meshes=False,
            return_smpl_parameters=True,
            return_smpl_vertices=False,
            return_fitting_errors=False,
            batch_size=256,
        )
        smpl_params = conversion_results.result_parameters

        

        del smpl_params['left_hand_pose']
        del smpl_params['right_hand_pose']
        del smpl_params['expression']

        # smpl_params = postprocess_smpl_params(
        #     smpl_params,
        # )

        metric_3dpw.evaluate(smpl_params, None, None, None, save_path, None)
