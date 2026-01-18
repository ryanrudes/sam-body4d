import torch
from eval.eval_utils.smooth_utils.geometry import (
    rot6d_to_rotation_matrix,
    smooth_with_savgol,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_rot6d,
    rotation_matrix_to_axis_angle,
    smooth_with_slerp,
)

# @torch.no_grad()
# def smpl_to_smpl_decode_o6dp(
#     global_orient: torch.Tensor,  # (B, 3) or (B, L, 3) axis-angle
#     body_pose: torch.Tensor,      # (B, 63) or (B, L, 63) or (B, L, 21, 3)
#     transl: torch.Tensor,         # (B, 3) or (B, L, 3)
#     rel_trans: bool = False,
#     should_apply_smooothing: bool = True,
# ):
#     """
#     SMPL -> SMPL with the same *behavior* as _decode_o6dp:
#       - convert AA->6D
#       - (optional) temporal smoothing (slerp + savgol)
#       - (optional) SMPL forward + ground alignment
#       - convert 6D->AA back

#     Returns:
#       dict with keys: global_orient, body_pose, transl
#       Shapes follow your input convention: if input had no L, output has no L.
#     """

#     # -------- normalize shapes to (B, L, *) --------
#     def ensure_BL(x, last_dim):
#         if x.ndim == 2 and x.shape[-1] == last_dim:
#             return x[:, None, :], True  # added L
#         return x, False

#     global_orient_bl, go_added_L = ensure_BL(global_orient, 3)
#     transl_bl, tr_added_L = ensure_BL(transl, 3)

#     if body_pose.ndim == 2 and body_pose.shape[-1] == 63:
#         body_pose_bl = body_pose[:, None, :]
#         bp_added_L = True
#     elif body_pose.ndim == 3 and body_pose.shape[-1] == 63:
#         body_pose_bl = body_pose
#         bp_added_L = False
#     elif body_pose.ndim == 4 and body_pose.shape[-2:] == (21, 3):
#         body_pose_bl = body_pose.reshape(body_pose.shape[0], body_pose.shape[1], 63)
#         bp_added_L = False
#     else:
#         raise ValueError(
#             f"Unsupported body_pose shape {tuple(body_pose.shape)}; expected (B,63) or (B,L,63) or (B,L,21,3)."
#         )

#     # sanity: all (B,L,*) should agree on B,L
#     B, L = global_orient_bl.shape[:2]
#     if transl_bl.shape[:2] != (B, L) or body_pose_bl.shape[:2] != (B, L):
#         raise ValueError(
#             f"B,L mismatch: global_orient {tuple(global_orient_bl.shape[:2])}, "
#             f"transl {tuple(transl_bl.shape[:2])}, body_pose {tuple(body_pose_bl.shape[:2])}"
#         )

#     device = transl_bl.device

#     # -------- 1) SMPL axis-angle -> rot6d (root + body) --------
#     root_rotmat = axis_angle_to_rotation_matrix(
#         global_orient_bl.reshape(-1, 3)
#     ).reshape(B, L, 3, 3)
#     root6d = rotation_matrix_to_rot6d(root_rotmat).reshape(B, L, 1, 6)

#     body_pose_21x3 = body_pose_bl.reshape(B, L, 21, 3)
#     body_rotmat = axis_angle_to_rotation_matrix(
#         body_pose_21x3.reshape(-1, 3)
#     ).reshape(B, L, 21, 3, 3)
#     body6d = rotation_matrix_to_rot6d(body_rotmat)  # (B, L, 21, 6)

#     rot6d = torch.cat([root6d, body6d], dim=2)  # (B, L, 22, 6)

#     # -------- 2) transl handling (match _decode_o6dp semantics) --------
#     # if rel_trans:
#     #     delta = torch.zeros_like(transl_bl)
#     #     delta[:, 0] = transl_bl[:, 0] * self.output_mesh_fps
#     #     delta[:, 1:] = (transl_bl[:, 1:] - transl_bl[:, :-1]) * self.output_mesh_fps
#     #     transl_dec = torch.cumsum(delta, dim=1) / self.output_mesh_fps
#     # else:
#     transl_dec = transl_bl.clone()

#     # -------- 3) smoothing (same behavior) --------
#     # if should_apply_smooothing:
#     #     # only apply slerp smoothing to the first 22 joints (non-finger joints)
#     #     rot6d_body = rot6d[:, :, :22, :]
#     #     rot6d_fingers = rot6d[:, :, 22:, :]
#     #     rot6d_body_smooth = smooth_with_slerp(rot6d_body, sigma=1.0)
#     #     rot6d_smooth = torch.cat([rot6d_body_smooth, rot6d_fingers], dim=2)
#     #     transl_smooth = smooth_with_savgol(transl_dec.detach(), window_length=11, polyorder=5)
#     # else:
#     #     rot6d_smooth = rot6d
#     #     transl_smooth = transl_dec
#     if should_apply_smooothing:
#         # Keep root rotation unchanged (avoid 180° flip)
#         root6d_keep = rot6d[:, :, 0:1, :]              # (B,L,1,6)

#         # Smooth only body joints (21 joints)
#         body6d = rot6d[:, :, 1:22, :]                  # (B,L,21,6)
#         body6d_smooth = smooth_with_slerp(body6d, sigma=1.0)

#         rot6d_smooth = torch.cat([root6d_keep, body6d_smooth], dim=2)

#         transl_smooth = smooth_with_savgol(transl_dec.detach(), window_length=11, polyorder=5)
#     else:
#         rot6d_smooth = rot6d
#         transl_smooth = transl_dec

#     # # -------- 4) SMPL forward + ground alignment (same behavior) --------
#     # if self.body_model is not None:
#     #     with torch.no_grad():
#     #         verts_all = []
#     #         for b in range(B):
#     #             out = self.body_model.forward({"rot6d": rot6d_smooth[b], "trans": transl_smooth[b]})
#     #             verts_all.append(out["vertices"])
#     #         vertices = torch.stack(verts_all, dim=0)  # (B, L, V, 3) typically

#     #     min_y = vertices[..., 1].amin(dim=(1, 2), keepdim=True)  # (B,1,1)
#     #     transl_smooth = transl_smooth.clone()
#     #     transl_smooth[..., 1] -= min_y.squeeze(-1).to(device)

#     # -------- 5) rot6d -> SMPL axis-angle --------
#     root_rotmat_smooth = rot6d_to_rotation_matrix(rot6d_smooth[:, :, 0, :])  # (B, L, 3, 3)
#     global_orient_out = rotation_matrix_to_axis_angle(root_rotmat_smooth).reshape(B, L, 3)

#     body_rotmat_smooth = rot6d_to_rotation_matrix(rot6d_smooth[:, :, 1:, :].reshape(-1, 6)).reshape(B, L, 21, 3, 3)
#     body_pose_out = rotation_matrix_to_axis_angle(body_rotmat_smooth.reshape(-1, 3, 3)).reshape(B, L, 63)

#     # -------- restore original "has L?" convention --------
#     if go_added_L and tr_added_L and bp_added_L:
#         return {
#             "global_orient": global_orient_out[:, 0],
#             "body_pose": body_pose_out[:, 0],
#             "transl": transl_smooth[:, 0],
#         }
#     else:
#         return {
#             "global_orient": global_orient_out,
#             "body_pose": body_pose_out,
#             "transl": transl_smooth,
#         }

# import torch

# @torch.no_grad()
# def smpl_to_smpl_decode_o6dp(
#     global_orient: torch.Tensor,  # (B, 3) or (B, L, 3) axis-angle
#     body_pose: torch.Tensor,      # (B, 63) or (B, L, 63) or (B, L, 21, 3)
#     transl: torch.Tensor,         # (B, 3) or (B, L, 3)
#     rel_trans: bool = False,
#     should_apply_smooothing: bool = True,
# ):
#     """
#     SMPL -> SMPL (same behavior as before) EXCEPT:
#       - DO NOT smooth global_orient (root orientation). Return it as-is.
#       - Smooth only body_pose (21 joints) with slerp (AA->rot6d->slerp->AA).
#       - Smooth translation with Savgol if L >= 11.

#     Returns: dict {global_orient, body_pose, transl}
#     """

#     # -------- normalize shapes to (B, L, *) --------
#     def ensure_BL(x, last_dim):
#         if x.ndim == 2 and x.shape[-1] == last_dim:
#             return x[:, None, :], True  # added L
#         return x, False

#     global_orient_bl, go_added_L = ensure_BL(global_orient, 3)
#     transl_bl, tr_added_L = ensure_BL(transl, 3)

#     if body_pose.ndim == 2 and body_pose.shape[-1] == 63:
#         body_pose_bl = body_pose[:, None, :]
#         bp_added_L = True
#     elif body_pose.ndim == 3 and body_pose.shape[-1] == 63:
#         body_pose_bl = body_pose
#         bp_added_L = False
#     elif body_pose.ndim == 4 and body_pose.shape[-2:] == (21, 3):
#         body_pose_bl = body_pose.reshape(body_pose.shape[0], body_pose.shape[1], 63)
#         bp_added_L = False
#     else:
#         raise ValueError(
#             f"Unsupported body_pose shape {tuple(body_pose.shape)}; expected (B,63) or (B,L,63) or (B,L,21,3)."
#         )

#     # sanity: all (B,L,*) should agree on B,L
#     B, L = global_orient_bl.shape[:2]
#     if transl_bl.shape[:2] != (B, L) or body_pose_bl.shape[:2] != (B, L):
#         raise ValueError(
#             f"B,L mismatch: global_orient {tuple(global_orient_bl.shape[:2])}, "
#             f"transl {tuple(transl_bl.shape[:2])}, body_pose {tuple(body_pose_bl.shape[:2])}"
#         )

#     # -------- 1) BODY: axis-angle -> rot6d --------
#     body_pose_21x3 = body_pose_bl.reshape(B, L, 21, 3)
#     body_rotmat = axis_angle_to_rotation_matrix(body_pose_21x3.reshape(-1, 3)).reshape(B, L, 21, 3, 3)
#     body6d = rotation_matrix_to_rot6d(body_rotmat)  # (B, L, 21, 6)

#     # -------- 2) transl handling --------
#     transl_dec = transl_bl.clone()

#     # -------- 3) smoothing --------
#     if should_apply_smooothing:
#         # Smooth ONLY body joints
#         body6d_smooth = smooth_with_slerp(body6d, sigma=1.0)

#         # Smooth translation only if long enough
#         if L >= 11:
#             transl_smooth = smooth_with_savgol(transl_dec.detach(), window_length=11, polyorder=5)
#         else:
#             transl_smooth = transl_dec
#     else:
#         body6d_smooth = body6d
#         transl_smooth = transl_dec

#     # -------- 4) BODY: rot6d -> axis-angle --------
#     body_rotmat_smooth = rot6d_to_rotation_matrix(body6d_smooth.reshape(-1, 6)).reshape(B, L, 21, 3, 3)
#     body_pose_out = rotation_matrix_to_axis_angle(body_rotmat_smooth.reshape(-1, 3, 3)).reshape(B, L, 63)

#     # -------- 5) ROOT: return global_orient as-is (NO smoothing) --------
#     global_orient_out = global_orient_bl  # unchanged values

#     # -------- restore original "has L?" convention --------
#     if go_added_L and tr_added_L and bp_added_L:
#         return {
#             "global_orient": global_orient_out[:, 0],
#             "body_pose": body_pose_out[:, 0],
#             "transl": transl_smooth[:, 0],
#         }
#     else:
#         return {
#             "global_orient": global_orient_out,
#             "body_pose": body_pose_out,
#             "transl": transl_smooth,
#         }


import torch
import math

@torch.no_grad()
def smpl_to_smpl_decode_o6dp(
    global_orient: torch.Tensor,  # (B, 3) or (B, L, 3) axis-angle
    body_pose: torch.Tensor,      # (B, 63) or (B, L, 63) or (B, L, 21, 3)
    transl: torch.Tensor,         # (B, 3) or (B, L, 3)
    rel_trans: bool = False,
    should_apply_smooothing: bool = True,
    # ---- post-fix controls ----
    fix_root_flip: bool = True,
    flip_deg_thr: float = 20.0,   # if angle between consecutive root rotations > this, treat as bad
    copy_all_on_fix: bool = True, # if True, also copy body_pose/transl when fixing root
):
    """
    SMPL -> SMPL:
      - AA -> 6D
      - (optional) smoothing (slerp + savgol)
      - 6D -> AA
      - (optional) post-fix for root flips: if smoothed root becomes discontinuous, overwrite bad frames.

    This keeps your original behavior, but adds a robust "safety net" after smoothing.
    """

    # -------- normalize shapes to (B, L, *) --------
    def ensure_BL(x, last_dim):
        if x.ndim == 2 and x.shape[-1] == last_dim:
            return x[:, None, :], True  # added L
        return x, False

    global_orient_bl, go_added_L = ensure_BL(global_orient, 3)
    transl_bl, tr_added_L = ensure_BL(transl, 3)

    if body_pose.ndim == 2 and body_pose.shape[-1] == 63:
        body_pose_bl = body_pose[:, None, :]
        bp_added_L = True
    elif body_pose.ndim == 3 and body_pose.shape[-1] == 63:
        body_pose_bl = body_pose
        bp_added_L = False
    elif body_pose.ndim == 4 and body_pose.shape[-2:] == (21, 3):
        body_pose_bl = body_pose.reshape(body_pose.shape[0], body_pose.shape[1], 63)
        bp_added_L = False
    else:
        raise ValueError(
            f"Unsupported body_pose shape {tuple(body_pose.shape)}; expected (B,63) or (B,L,63) or (B,L,21,3)."
        )

    # sanity
    B, L = global_orient_bl.shape[:2]
    if transl_bl.shape[:2] != (B, L) or body_pose_bl.shape[:2] != (B, L):
        raise ValueError(
            f"B,L mismatch: global_orient {tuple(global_orient_bl.shape[:2])}, "
            f"transl {tuple(transl_bl.shape[:2])}, body_pose {tuple(body_pose_bl.shape[:2])}"
        )

    # -------- 1) AA -> rot6d (root + body) --------
    root_rotmat = axis_angle_to_rotation_matrix(global_orient_bl.reshape(-1, 3)).reshape(B, L, 3, 3)
    root6d = rotation_matrix_to_rot6d(root_rotmat).reshape(B, L, 1, 6)

    body_pose_21x3 = body_pose_bl.reshape(B, L, 21, 3)
    body_rotmat = axis_angle_to_rotation_matrix(body_pose_21x3.reshape(-1, 3)).reshape(B, L, 21, 3, 3)
    body6d = rotation_matrix_to_rot6d(body_rotmat)  # (B, L, 21, 6)

    rot6d = torch.cat([root6d, body6d], dim=2)  # (B, L, 22, 6)

    # -------- 2) translation --------
    transl_dec = transl_bl.clone()

    # -------- 3) smoothing --------
    if should_apply_smooothing:
        # smooth all 22 joints (root + 21 body)
        rot6d_body = rot6d[:, :, :22, :]
        rot6d_fingers = rot6d[:, :, 22:, :]
        rot6d_body_smooth = smooth_with_slerp(rot6d_body, sigma=1.0)
        rot6d_smooth = torch.cat([rot6d_body_smooth, rot6d_fingers], dim=2)

        # savgol needs L>=11
        if L >= 11:
            transl_smooth = smooth_with_savgol(transl_dec.detach(), window_length=11, polyorder=5)
        else:
            transl_smooth = transl_dec
    else:
        rot6d_smooth = rot6d
        transl_smooth = transl_dec

    # -------- 4) rot6d -> AA --------
    root_rotmat_smooth = rot6d_to_rotation_matrix(rot6d_smooth[:, :, 0, :])  # (B, L, 3, 3)
    global_orient_out = rotation_matrix_to_axis_angle(root_rotmat_smooth).reshape(B, L, 3)

    body_rotmat_smooth = rot6d_to_rotation_matrix(rot6d_smooth[:, :, 1:, :].reshape(-1, 6)).reshape(B, L, 21, 3, 3)
    body_pose_out = rotation_matrix_to_axis_angle(body_rotmat_smooth.reshape(-1, 3, 3)).reshape(B, L, 63)

    # -------- 5) post-fix: detect & correct bad root flips --------
    if fix_root_flip and L >= 2:
        # angle between consecutive root rotations (degrees)
        R = root_rotmat_smooth  # (B,L,3,3)
        R_rel = R[:, :-1].transpose(-1, -2) @ R[:, 1:]  # (B,L-1,3,3)
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        cos = (trace - 1.0) / 2.0
        cos = torch.clamp(cos, -1.0, 1.0)
        ang_deg = torch.acos(cos) * (180.0 / math.pi)  # (B,L-1)

        # bad at time t means transition (t-1)->t is too large, so fix frame t
        bad = ang_deg > flip_deg_thr  # (B,L-1)

        # in-place fix on outputs (safe under no_grad)
        for b in range(B):
            bad_t = (bad[b].nonzero(as_tuple=False).squeeze(-1) + 1).tolist()
            for t in bad_t:
                # conservative: copy previous frame
                global_orient_out[b, t] = global_orient_out[b, t - 1]
                if copy_all_on_fix:
                    body_pose_out[b, t] = body_pose_out[b, t - 1]
                    transl_smooth[b, t] = transl_smooth[b, t - 1]

    # -------- restore original "has L?" convention --------
    if go_added_L and tr_added_L and bp_added_L:
        return {
            "global_orient": global_orient_out[:, 0],
            "body_pose": body_pose_out[:, 0],
            "transl": transl_smooth[:, 0],
        }
    else:
        return {
            "global_orient": global_orient_out,
            "body_pose": body_pose_out,
            "transl": transl_smooth,
        }
