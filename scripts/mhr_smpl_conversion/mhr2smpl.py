import torch
from mhr.mhr import MHR
from conversion import Conversion
from pathlib import Path
import smplx
import numpy as np


def mhr2smpl(
    mhr_vertices: dict[str, torch.Tensor] | None = None,
    batch_size: int = 256,
    ):

    conversion_results = converter.convert_mhr2smpl(
        mhr_vertices=mhr_vertices,
        mhr_parameters=None,
        single_identity=True,
        return_smpl_meshes=False,
        return_smpl_parameters=True,
        return_smpl_vertices=False,
        return_fitting_errors=True,
        batch_size=batch_size,
    )

    # print("Conversion errors:")
    # print(conversion_results.result_errors)

    return conversion_results.result_parameters
