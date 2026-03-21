# smooth.py  (single-file, self-contained)
import math
import torch
import torch.nn.functional as F

# ============================================================
# Geometry helpers (prefer pytorch3d if available; else fallback)
# ============================================================

try:
    from pytorch3d.transforms import (
        axis_angle_to_matrix as axis_angle_to_rotation_matrix,
        matrix_to_axis_angle as rotation_matrix_to_axis_angle,
        matrix_to_rotation_6d as rotation_matrix_to_rot6d,
        rotation_6d_to_matrix as rot6d_to_rotation_matrix,
    )
except Exception:
    axis_angle_to_rotation_matrix = None
    rotation_matrix_to_axis_angle = None
    rotation_matrix_to_rot6d = None
    rot6d_to_rotation_matrix = None


def _axis_angle_to_rotation_matrix_fallback(aa: torch.Tensor) -> torch.Tensor:
    """
    aa: (..., 3) axis-angle
    return: (..., 3, 3)
    """
    eps = 1e-8
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(eps)
    axis = aa / theta
    x, y, z = axis.unbind(dim=-1)

    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(aa.shape[:-1] + (3, 3))

    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(aa.shape[:-1] + (3, 3))
    ct = torch.cos(theta)[..., None]
    st = torch.sin(theta)[..., None]
    return I + st * K + (1.0 - ct) * (K @ K)


def _rotation_matrix_to_axis_angle_fallback(R: torch.Tensor) -> torch.Tensor:
    """
    R: (..., 3, 3)
    return: (..., 3) axis-angle
    """
    eps = 1e-8
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = (trace - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0, 1.0)
    theta = torch.acos(cos)

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    r = torch.stack([rx, ry, rz], dim=-1)

    sin_theta = torch.sin(theta).clamp_min(eps)[..., None]
    axis = r / (2.0 * sin_theta)
    aa = axis * theta[..., None]

    small = theta < 1e-4
    if small.any():
        aa_small = 0.5 * r
        aa = torch.where(small[..., None], aa_small, aa)
    return aa


def _rotation_matrix_to_rot6d_fallback(R: torch.Tensor) -> torch.Tensor:
    """
    R: (...,3,3) -> (...,6) first two columns
    """
    return torch.stack(
        [
            R[..., 0, 0], R[..., 1, 0], R[..., 2, 0],
            R[..., 0, 1], R[..., 1, 1], R[..., 2, 1],
        ],
        dim=-1,
    )


def _rot6d_to_rotation_matrix_fallback(x: torch.Tensor) -> torch.Tensor:
    """
    x: (...,6) -> (...,3,3)
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _resolve_geom_fns():
    global axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle
    global rotation_matrix_to_rot6d, rot6d_to_rotation_matrix
    if axis_angle_to_rotation_matrix is None:
        axis_angle_to_rotation_matrix = _axis_angle_to_rotation_matrix_fallback
    if rotation_matrix_to_axis_angle is None:
        rotation_matrix_to_axis_angle = _rotation_matrix_to_axis_angle_fallback
    if rotation_matrix_to_rot6d is None:
        rotation_matrix_to_rot6d = _rotation_matrix_to_rot6d_fallback
    if rot6d_to_rotation_matrix is None:
        rot6d_to_rotation_matrix = _rot6d_to_rotation_matrix_fallback


# ============================================================
# Quaternion helpers (for rotation smoothing)
# ============================================================

def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    R: (...,3,3) -> q: (...,4) (w,x,y,z)
    """
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    def _norm(q):
        q = F.normalize(q, dim=-1)
        return torch.where(q[..., :1] < 0, -q, q)  # canonicalize w>=0

    t = trace + 1.0
    s = torch.sqrt(torch.clamp(t, min=1e-8)) * 2.0
    qw = 0.25 * s
    qx = (R[..., 2, 1] - R[..., 1, 2]) / s
    qy = (R[..., 0, 2] - R[..., 2, 0]) / s
    qz = (R[..., 1, 0] - R[..., 0, 1]) / s
    q_trace = _norm(torch.stack([qw, qx, qy, qz], dim=-1))

    cond1 = (m00 > m11) & (m00 > m22)
    cond2 = (m11 > m22)

    s1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8)) * 2.0
    qw1 = (R[..., 2, 1] - R[..., 1, 2]) / s1
    qx1 = 0.25 * s1
    qy1 = (R[..., 0, 1] + R[..., 1, 0]) / s1
    qz1 = (R[..., 0, 2] + R[..., 2, 0]) / s1
    q1 = _norm(torch.stack([qw1, qx1, qy1, qz1], dim=-1))

    s2 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-8)) * 2.0
    qw2 = (R[..., 0, 2] - R[..., 2, 0]) / s2
    qx2 = (R[..., 0, 1] + R[..., 1, 0]) / s2
    qy2 = 0.25 * s2
    qz2 = (R[..., 1, 2] + R[..., 2, 1]) / s2
    q2 = _norm(torch.stack([qw2, qx2, qy2, qz2], dim=-1))

    s3 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-8)) * 2.0
    qw3 = (R[..., 1, 0] - R[..., 0, 1]) / s3
    qx3 = (R[..., 0, 2] + R[..., 2, 0]) / s3
    qy3 = (R[..., 1, 2] + R[..., 2, 1]) / s3
    qz3 = 0.25 * s3
    q3 = _norm(torch.stack([qw3, qx3, qy3, qz3], dim=-1))

    q_else = torch.where(cond2[..., None], q2, q3)
    q_else = torch.where(cond1[..., None], q1, q_else)
    use_trace = trace > 0.0
    return torch.where(use_trace[..., None], q_trace, q_else)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: (...,4) (w,x,y,z) -> R: (...,3,3)
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(dim=-1)

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = torch.stack(
        [
            ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy),
            2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx),
            2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return R


def _gaussian_weights(radius: int, sigma: float, device, dtype):
    d = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    w = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return w / w.sum()


def _quat_weighted_mean_markley(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    q: (K,4), w: (K,)
    """
    q0 = q[0]
    sign = torch.sign((q * q0).sum(dim=-1, keepdim=True))
    sign = torch.where(sign >= 0, torch.ones_like(sign), -torch.ones_like(sign))
    q = q * sign

    A = (w[:, None, None] * (q[:, :, None] @ q[:, None, :])).sum(dim=0)  # (4,4)
    evals, evecs = torch.linalg.eigh(A)
    qm = evecs[:, -1]
    qm = F.normalize(qm, dim=0)
    if qm[0] < 0:
        qm = -qm
    return qm


@torch.no_grad()
def smooth_with_slerp(rot6d_body: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    rot6d_body: (B,L,J,6)
    output:     (B,L,J,6)
    (Note: name kept to match your decode func; implementation is quaternion-window mean.)
    """
    if rot6d_body.ndim != 4 or rot6d_body.shape[-1] != 6:
        raise ValueError(f"smooth_with_slerp expects (B,L,J,6), got {tuple(rot6d_body.shape)}")
    B, L, J, _ = rot6d_body.shape
    if L <= 1:
        return rot6d_body

    device, dtype = rot6d_body.device, rot6d_body.dtype
    radius = max(1, int(math.ceil(3.0 * sigma)))
    w = _gaussian_weights(radius, sigma, device=device, dtype=dtype)
    K = 2 * radius + 1

    R = rot6d_to_rotation_matrix(rot6d_body.reshape(-1, 6)).reshape(B, L, J, 3, 3)
    q = _rotmat_to_quat(R.reshape(-1, 3, 3)).reshape(B, L, J, 4)

    q_pad = torch.cat(
        [q[:, :1].expand(B, radius, J, 4), q, q[:, -1:].expand(B, radius, J, 4)],
        dim=1,
    )  # (B, L+2r, J, 4)

    q_out = torch.empty_like(q)
    for t in range(L):
        q_win = q_pad[:, t:t + K]  # (B,K,J,4)
        for b in range(B):
            for j in range(J):
                q_out[b, t, j] = _quat_weighted_mean_markley(q_win[b, :, j], w)

    R_out = _quat_to_rotmat(q_out.reshape(-1, 4)).reshape(B, L, J, 3, 3)
    out = rotation_matrix_to_rot6d(R_out.reshape(-1, 3, 3)).reshape(B, L, J, 6)
    return out


def _savgol_coeffs(window_length: int, polyorder: int, device, dtype) -> torch.Tensor:
    if window_length % 2 != 1 or window_length < 3:
        raise ValueError("window_length must be odd and >= 3")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")
    m = window_length // 2
    x = torch.arange(-m, m + 1, device=device, dtype=dtype)
    A = torch.stack([x ** p for p in range(polyorder + 1)], dim=-1)  # (W,P+1)
    pinv = torch.linalg.pinv(A)  # (P+1,W)
    return pinv[0]  # (W,)


@torch.no_grad()
def smooth_with_savgol(x: torch.Tensor, window_length: int = 11, polyorder: int = 5) -> torch.Tensor:
    """
    x: (B,L,C)
    """
    if x.ndim != 3:
        raise ValueError(f"smooth_with_savgol expects (B,L,C), got {tuple(x.shape)}")
    B, L, C = x.shape
    if L < window_length:
        return x
    device, dtype = x.device, x.dtype
    c = _savgol_coeffs(window_length, polyorder, device=device, dtype=dtype)  # (W,)
    m = window_length // 2

    x_pad = torch.cat([x[:, :1].expand(B, m, C), x, x[:, -1:].expand(B, m, C)], dim=1)  # (B,L+2m,C)
    xp = x_pad.transpose(1, 2)  # (B,C,L+2m)
    weight = c.view(1, 1, -1).expand(C, 1, -1)  # (C,1,W)
    y = F.conv1d(xp, weight, padding=0, groups=C)  # (B,C,L)
    return y.transpose(1, 2)


# ============================================================
# Your smpl_to_smpl_decode_o6dp (kept same structure)
# ============================================================

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
    """
    _resolve_geom_fns()

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
        rot6d_body = rot6d[:, :, :22, :]
        rot6d_fingers = rot6d[:, :, 22:, :]
        rot6d_body_smooth = smooth_with_slerp(rot6d_body, sigma=1.0)
        rot6d_smooth = torch.cat([rot6d_body_smooth, rot6d_fingers], dim=2)

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
        R = root_rotmat_smooth  # (B,L,3,3)
        R_rel = R[:, :-1].transpose(-1, -2) @ R[:, 1:]  # (B,L-1,3,3)
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
        cos = (trace - 1.0) / 2.0
        cos = torch.clamp(cos, -1.0, 1.0)
        ang_deg = torch.acos(cos) * (180.0 / math.pi)  # (B,L-1)

        bad = ang_deg > flip_deg_thr  # (B,L-1)
        for b in range(B):
            bad_t = (bad[b].nonzero(as_tuple=False).squeeze(-1) + 1).tolist()
            for t in bad_t:
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


# ============================================================
# Exact de-spike + per-segment smoothing flow (as you first posted)
# ============================================================

SOFT_THR = 10.0
HARD_THR = 20.0


def _split_true_segments(mask: torch.Tensor):
    L = mask.shape[0]
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
    return segments


def _root_angle_deg(go_seg: torch.Tensor):
    _resolve_geom_fns()
    R = axis_angle_to_rotation_matrix(go_seg)          # (T,3,3)
    R_rel = R[:-1].transpose(-1, -2) @ R[1:]           # (T-1,3,3)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos = (trace - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.acos(cos) * (180.0 / math.pi)


@torch.no_grad()
def postprocess_smpl_params(smpl_params: dict):
    """
    Single entry:
      input/output are both smpl_params

    Required keys:
      - global_orient: (L,3)
      - body_pose:     (L,63)
      - transl:        (L,3)
    Optional:
      - mask:          (L,) bool  (if absent: all True)
    """
    _resolve_geom_fns()

    mask = smpl_params.get("mask", None)
    go0 = smpl_params["global_orient"]
    bp0 = smpl_params["body_pose"]
    tr0 = smpl_params["transl"]

    L = go0.shape[0]
    if mask is None:
        mask = torch.ones(L, dtype=torch.bool, device=go0.device)
    else:
        mask = mask.bool().to(go0.device)

    soft_thr = float(smpl_params.get("_soft_thr_deg", SOFT_THR))
    hard_thr = float(smpl_params.get("_hard_thr_deg", HARD_THR))

    # work on detached copies
    go = go0.detach().clone()
    bp = bp0.detach().clone()
    tr = tr0.detach().clone()

    # 1) split into contiguous True segments
    segments = _split_true_segments(mask)

    # 2) detect spikes and copy previous
    for s, e in segments:
        T = e - s + 1
        if T < 3:
            continue

        ang = _root_angle_deg(go[s:e + 1])  # (T-1,)

        for t in range(1, T - 1):
            a_prev = ang[t - 1].item()
            a_next = ang[t].item()

            if (a_prev > hard_thr and a_next < soft_thr) or (a_next > hard_thr and a_prev < soft_thr):
                abs_t = s + t
                go[abs_t] = go[abs_t - 1]
                bp[abs_t] = bp[abs_t - 1]
                tr[abs_t] = tr[abs_t - 1]

    # write back de-spiked
    out = dict(smpl_params)
    out["global_orient"] = go
    out["body_pose"] = bp
    out["transl"] = tr

    # 3) final smoothing (segment-wise)
    for s, e in segments:
        T = e - s + 1
        if T < 2:
            continue
        use_smooth = (T >= 11)
        if not use_smooth:
            continue

        smooth_results = smpl_to_smpl_decode_o6dp(
            out["global_orient"][s:e + 1].unsqueeze(0),
            out["body_pose"][s:e + 1].unsqueeze(0),
            out["transl"][s:e + 1].unsqueeze(0),
            should_apply_smooothing=use_smooth,
        )

        out["global_orient"][s:e + 1] = smooth_results["global_orient"].squeeze(0)
        out["body_pose"][s:e + 1] = smooth_results["body_pose"].squeeze(0)
        out["transl"][s:e + 1] = smooth_results["transl"].squeeze(0)

    return out

# optional quick smoke-test
if __name__ == "__main__":
    L = 30
    smpl_params = {
        "global_orient": torch.zeros(L, 3),
        "body_pose": torch.zeros(L, 63),
        "transl": torch.zeros(L, 3),
        "mask": torch.ones(L, dtype=torch.bool),
    }
    out = postprocess_smpl_params(smpl_params)
    print(out["global_orient"].shape, out["body_pose"].shape, out["transl"].shape)
