import os
import torch
from einops import einsum
from pathlib import Path
from smplx_utils import make_smplx
from eval_utils.eval_tools import compute_camcoord_metrics, as_np_array
from eval_utils.video_io_utils import read_video_np, save_video
from eval_utils.vis.renderer_utils import simple_render_mesh_background
from geo_transform import apply_T_on_points
import numpy as np
import cv2


class MetricMocap:
    def __init__(self, body_model_path):
        # super().__init__()
        # vid->result
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "mpjpe": {},
            "pve": {},
            "accel": {},
        }

        # SMPLX and SMPL
        self.smplx = make_smplx("supermotion_EVAL3DPW", body_model_path=body_model_path, )
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        self.smpl = {"male": make_smplx("smpl", body_model_path=body_model_path, gender="male"), "female": make_smplx("smpl", body_model_path=body_model_path, gender="female")}
        self.J_regressor = torch.load(f"{script_dir}/body_model/smpl_3dpw14_J_regressor_sparse.pt").to_dense()
        self.J_regressor24 = torch.load(f"{script_dir}/body_model/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load(f"{script_dir}/body_model/smplx2smpl_sparse.pt")
        self.faces_smplx = self.smplx.faces
        self.faces_smpl = self.smpl["male"].faces

    # ================== Batch-based Computation  ================== #
    def evaluate(self, outputs, batch, mhr_height, smplx_vertices, save_path=None, smplx=None):
        """The behaviour is the same for val/test/predict"""
        dataset_id = batch["meta"]["dataset_id"]
        if dataset_id != "3DPW":
            return

        # Move to cuda if not
        # smplx_vertices = torch.tensor(smplx_vertices).cuda()
        self.smplx = self.smplx.cuda()
        # for g in ["male", "female", "neutral"]:
        self.smpl["male"] = self.smpl["male"].cuda()
        #     self.smpl[g] = self.smpl[g].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.J_regressor24 = self.J_regressor24.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        vid = batch["meta"]["vid"]
        # gender = batch["gender"]
        gender = "male"
        T_w2c = batch["T_w2c"]
        mask = batch["mask"]

        # Groundtruth (cam)
        target_w_params = {k: v for k, v in batch["smpl_params"].items()}
        target_w_output = self.smpl[gender](**target_w_params)
        target_w_verts = target_w_output.vertices
        target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
        target_c_j3d = torch.matmul(self.J_regressor, target_c_verts)

        # + Prediction -> Metric
        if smplx is not None:
            B = outputs["global_orient"].shape[0]
            # 1. 先保证你已有的都是 contiguous
            outputs = {k: v.contiguous() for k, v in outputs.items()}
            # 2. 自动把 smplx 里存在、但 outputs 里没传的 tensor buffer 补进来
            for k in dir(smplx):
                if k.startswith("_"):
                    continue
                if k in outputs:
                    continue
                v = getattr(smplx, k)
                if hasattr(v, "shape") and v.shape[0] == 1:
                    outputs[k] = v.expand(B, *v.shape[1:]).contiguous()
            smpl_out = smplx(**outputs)
        else:
            smpl_out = self.smplx(**outputs)
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        # pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smplx_vertices])
        pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")
        
        # smpl_out = self.smpl[gender](**outputs)
        # pred_c_verts = smpl_out.vertices
        
        # verts = pred_c_verts  # [B, V, 3], unit: meter
        # y_max = verts[:, :, 1].max(dim=1).values
        # y_min = verts[:, :, 1].min(dim=1).values
        # smpl_height_smpl = y_max - y_min  # [B], m

        # # verts = smpl_out.vertices  # [B, V, 3], unit: meter
        # # y_max = verts[:, :, 1].max(dim=1).values
        # # y_min = verts[:, :, 1].min(dim=1).values
        # # smpl_height_smplx = y_max - y_min  # [B], m

        # verts = smplx_vertices  # [B, V, 3], unit: meter
        # y_max = verts[:, :, 1].max(dim=1).values
        # y_min = verts[:, :, 1].min(dim=1).values
        # smpl_height_smplx_ = y_max - y_min  # [B], m

        # pred_c_verts = pred_c_verts * ((mhr_height[0].item())/(smpl_height[0].item()*100))
        # pred_c_j3d = torch.matmul(self.J_regressor, pred_c_verts)

        # Metric of current sequence
        batch_eval = {
            "pred_j3d": pred_c_j3d,
            "target_j3d": target_c_j3d,
            "pred_verts": pred_c_verts,
            "target_verts": target_c_verts,
        }
        camcoord_metrics = compute_camcoord_metrics(batch_eval, mask=mask, pelvis_idxs=[2, 3])
        for k in camcoord_metrics:
            self.metric_aggregator[k][vid] = as_np_array(camcoord_metrics[k])

        avg = camcoord_metrics['pa_mpjpe'].mean()
        print(f"{vid}: {avg}")

        if save_path is not None:
            import csv
            mode = "a" if os.path.exists(save_path) else "w"
            with open(save_path, mode, newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([vid, 'pa_mpjpe', camcoord_metrics['pa_mpjpe'].mean()] + list(camcoord_metrics['pa_mpjpe']))
                writer.writerow([vid, 'mpjpe', camcoord_metrics['mpjpe'].mean()] + list(camcoord_metrics['mpjpe']))
                writer.writerow([vid, 'pve', camcoord_metrics['pve'].mean()] + list(camcoord_metrics['pve']))
                writer.writerow([vid, 'accel', camcoord_metrics['accel'].mean()] + list(camcoord_metrics['accel']))

        metrics_avg = {k: np.concatenate(list(v.values())).mean() for k, v in self.metric_aggregator.items()}
        print(metrics_avg)

        # if True:  # Render incam (simple)
        #     meta_render = batch["meta_render"]
        #     images = read_video_np(meta_render["video_path"], scale=meta_render["ds"])
        #     render_dict = {
        #         "K": meta_render["K"][None],  # only support batch size 1
        #         "faces": self.smpl["male"].faces,
        #         "verts": target_c_verts,
        #         "background": images,
        #     }
        #     img_overlay = simple_render_mesh_background(render_dict)
        #     output_fn = Path("outputs/3DPW_render_pred_flip") / f"{vid}-bs1.mp4"
            
        #     # imgs: numpy array of shape (L, H, W, 3)
        #     out_dir = "output_images"
        #     os.makedirs(out_dir, exist_ok=True)

        #     for i in range(img_overlay.shape[0]):
        #         img = img_overlay[i]  # H x W x 3, RGB
        #         img_bgr = img[:, :, ::-1]  # RGB -> BGR (for OpenCV)
        #         cv2.imwrite(os.path.join(out_dir, f"{i:04d}.JPG"), img_bgr)

        #     save_video(img_overlay, output_fn, crf=28)

        del smpl_out  # Prevent OOM