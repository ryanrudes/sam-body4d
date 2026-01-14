import torch
import torch.nn.functional as F
import numpy as np
import smplx
import pickle
from smplx import SMPL, SMPLX, SMPLXLayer
from body_model import BodyModelSMPLH, BodyModelSMPLX
from body_model.smplx_lite import SmplxLiteCoco17, SmplxLiteV437Coco17, SmplxLiteSmplN24


def make_smplx(type="neu_fullpose", body_model_path="body_models", **kwargs):
    if type == "neu_fullpose":
        model = smplx.create(
            model_path="inputs/models/smplx/SMPLX_NEUTRAL.npz", use_pca=False, flat_hand_mean=True, **kwargs
        )
    elif type == "supermotion":
        # SuperMotion is trained on BEDLAM dataset, the smplx config is the same except only 10 betas are used
        bm_kwargs = {
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "flat_hand_mean": False,
        }
        bm_kwargs.update(kwargs)
        model = BodyModelSMPLX(model_path=body_model_path, **bm_kwargs)
    elif type == "supermotion_EVAL3DPW":
        # SuperMotion is trained on BEDLAM dataset, the smplx config is the same except only 10 betas are used
        bm_kwargs = {
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "flat_hand_mean": True,
        }
        bm_kwargs.update(kwargs)
        model = BodyModelSMPLX(model_path=body_model_path, **bm_kwargs)
    elif type == "supermotion_coco17":
        # Fast but only predicts 17 joints
        model = SmplxLiteCoco17()
    elif type == "supermotion_v437coco17":
        # Predicts 437 verts and 17 joints
        model = SmplxLiteV437Coco17()
    elif type == "supermotion_smpl24":
        model = SmplxLiteSmplN24()
    elif type == "rich-smplx":
        # https://github.com/paulchhuang/rich_toolkit/blob/main/smplx2images.py
        bm_kwargs = {
            "model_type": "smplx",
            "gender": kwargs.get("gender", "male"),
            "num_pca_comps": 12,
            "flat_hand_mean": False,
            # create_expression=True, create_jaw_pose=Ture
        }
        # A /smplx folder should exist under the model_path
        model = BodyModelSMPLX(model_path=body_model_path, **bm_kwargs)
    elif type == "rich-smplh":
        bm_kwargs = {
            "model_type": "smplh",
            "gender": kwargs.get("gender", "male"),
            "use_pca": False,
            "flat_hand_mean": True,
        }
        model = BodyModelSMPLH(model_path="inputs/checkpoints/body_models", **bm_kwargs)

    elif type in ["smplx-circle", "smplx-groundlink"]:
        # don't use hand
        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models",
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 16,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type == "smplx-motionx":
        layer_args = {
            "create_global_orient": False,
            "create_body_pose": False,
            "create_left_hand_pose": False,
            "create_right_hand_pose": False,
            "create_jaw_pose": False,
            "create_leye_pose": False,
            "create_reye_pose": False,
            "create_betas": False,
            "create_expression": False,
            "create_transl": False,
        }

        bm_kwargs = {
            "model_type": "smplx",
            "model_path": "inputs/checkpoints/body_models",
            "gender": "neutral",
            "use_pca": False,
            "use_face_contour": True,
            **layer_args,
        }
        model = smplx.create(**bm_kwargs)

    elif type == "smplx-samp":
        # don't use hand
        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models",
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 10,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type == "smplx-bedlam":
        # don't use hand
        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models",
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 11,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type in ["smplx-layer", "smplx-fit3d"]:
        # Use layer
        if type == "smplx-fit3d":
            assert (
                kwargs.get("gender") == "neutral"
            ), "smplx-fit3d use neutral model: https://github.com/sminchisescu-research/imar_vision_datasets_tools/blob/e8c8f83ffac23cc36adf8ec8d0fd1c55679484ef/util/smplx_util.py#L15C34-L15C34"

        bm_kwargs = {
            "model_path": "inputs/checkpoints/body_models/smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 10,
            "num_expression": 10,
        }
        model = SMPLXLayer(**bm_kwargs)

    elif type == "smpl":
        bm_kwargs = {
            "model_path": body_model_path,
            "model_type": "smpl",
            "gender": "neutral",
            "num_betas": 10,
            "create_body_pose": False,
            "create_betas": False,
            "create_global_orient": False,
            "create_transl": False,
        }
        bm_kwargs.update(kwargs)
        # model = SMPL(**bm_kwargs)
        model = BodyModelSMPLH(**bm_kwargs)
    elif type == "smplh":
        bm_kwargs = {
            "model_type": "smplh",
            "gender": kwargs.get("gender", "male"),
            "use_pca": False,
            "flat_hand_mean": False,
        }
        model = BodyModelSMPLH(model_path="inputs/checkpoints/body_models", **bm_kwargs)

    else:
        raise NotImplementedError

    return model
