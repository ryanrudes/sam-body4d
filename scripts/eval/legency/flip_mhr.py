import torch
import numpy as np

@torch.no_grad()
def decode_joint_params(character_torch, model_parameters_204: torch.Tensor):
    """
    model_parameters_204: (B, 204) float tensor
    returns joint_parameters: (B, 127, 7) or (B, 889) depending on implementation
    """
    B = model_parameters_204.shape[0]

    # This matches your forward(): pad with (expr + identity) zeros to satisfy the transform size
    pad_dim = character_torch.get_num_face_expression_blendshapes() + character_torch.get_num_identity_blendshapes()
    model_padding = torch.zeros(B, pad_dim, device=model_parameters_204.device, dtype=model_parameters_204.dtype)

    x = torch.cat([model_parameters_204, model_padding], dim=1)  # (B, 204+pad_dim)
    joint_parameters = character_torch.model_parameters_to_joint_parameters(x)

    # Some code returns (B, 127, 7); some returns (B, 889). Normalize to (B, 889).
    if joint_parameters.dim() == 3:
        joint_parameters = joint_parameters.reshape(B, -1)  # (B, 127*7)
    return joint_parameters  # (B, 889)

@torch.no_grad()
def recover_Tp(character_torch, device="cuda", dtype=torch.float32):
    """
    Recovers Tp in: theta_j_flat = Tp @ theta_p + bias
    where:
      theta_p is (204,)
      theta_j_flat is (889,)
    """
    dim_p = 204
    nj7 = 127 * 7

    # y0 = f(0)
    z0 = torch.zeros(1, dim_p, device=device, dtype=dtype)
    y0 = decode_joint_params(character_torch, z0)[0].detach().cpu().numpy().astype(np.float64)  # (889,)

    Tp = np.zeros((nj7, dim_p), dtype=np.float64)

    # Probe each basis vector e_i
    for i in range(dim_p):
        zi = torch.zeros(1, dim_p, device=device, dtype=dtype)
        zi[0, i] = 1.0
        yi = decode_joint_params(character_torch, zi)[0].detach().cpu().numpy().astype(np.float64)
        Tp[:, i] = (yi - y0)

    bias = y0  # in case f(x) = Tp x + bias
    return Tp.astype(np.float32), bias.astype(np.float32)
