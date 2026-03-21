import torch
import trimesh
import numpy as np

def smpl2obj(smpl_model, smpl_params, output_obj_path, idx):
    """
    Convert SMPL body_pose to OBJ using a pre-loaded SMPL instance.
    Fully handles device mismatch and 69-dim body_pose.

    Parameters
    ----------
    smpl_model : smplx.SMPL
        Pre-loaded SMPL model instance (your existing code)
    smpl_params : dict
        Required:
            - 'body_pose': (69,) or (1,69), axis-angle
        Optional:
            - 'betas': (10,)
            - 'transl': (3,)
    output_obj_path : str
        Path to save the OBJ
    """

    # --- Device ---
    device = next(smpl_model.parameters()).device

    # --- Body pose ---
    body_pose = torch.as_tensor(smpl_params["body_pose"], dtype=torch.float32, device=device)[idx:idx+1]
    if body_pose.ndim == 1:
        body_pose = body_pose.unsqueeze(0)
    if body_pose.shape[1] != 69:
        raise ValueError(f"body_pose must be 69-dim, got {body_pose.shape[1]}")

    # --- Global orientation (auto zero) ---
    global_orient = torch.zeros((body_pose.shape[0], 3), dtype=torch.float32, device=device)

    # --- Shape ---
    betas = torch.as_tensor(smpl_params.get("betas", np.zeros(10)), dtype=torch.float32, device=device)[idx:idx+1]
    if betas.ndim == 1:
        betas = betas.unsqueeze(0)

    # --- Translation ---
    transl = torch.as_tensor(smpl_params.get("transl", np.zeros(3)), dtype=torch.float32, device=device)[idx:idx+1]
    if transl.ndim == 1:
        transl = transl.unsqueeze(0)

    # --- Forward ---
    output = smpl_model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        transl=transl
    )

    # --- Extract vertices and faces ---
    vertices = output.vertices[0].detach().cpu().numpy()
    faces = smpl_model.faces

    # --- Export OBJ ---
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output_obj_path)
