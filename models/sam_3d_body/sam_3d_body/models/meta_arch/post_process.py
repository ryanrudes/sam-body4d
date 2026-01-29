# smooth_human_lb.py
# Single-file, self-contained.
#
# Goal:
#   Same workflow as your SMPL version, but your tensors are flattened as (L*B, ...),
#   where the flatten order is timestep-major:
#       idx = t * B + b
#   i.e. reshape to (L, B, ...) is valid.
#
# Public API:
#   postprocess_human_params(params: dict, batch_size: int) -> dict
#
# Expected params keys (all flattened as (L*B, ...)):
#   - "global_rot":  (L*B, 3) axis-angle
#   - "pred_cam_t":  (L*B, 3) translation
# Optional:
#   - "repr":        (L*B, D) any other human representation to copy/smooth like body_pose
#   - "mask":        (L*B,) bool, True=valid. If absent, treat all True.
#
# Behavior (per batch element b, per contiguous True segment):
#   1) despike using root rotation jump pattern (same as your SMPL code)
#   2) if segment length >= 11:
#        - smooth global_rot in rotation space (quat-window mean)
#        - smooth pred_cam_t with savgol
#        - if "repr" exists: smooth repr with savgol (Euclidean) + copy on spikes
#
# Notes:
#   - global_rot smoothing is rotation-aware; repr smoothing is Euclidean (safe default for unknown D).
#   - To match your SMPL logic: spike fix copies BOTH global_rot and pred_cam_t (and repr if present).

import math
import torch
import torch.nn.functional as F


# ============================================================
# Axis-angle <-> rotmat
# ============================================================

def axis_angle_to_rotation_matrix(aa: torch.Tensor) -> torch.Tensor:
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


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
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


# ============================================================
# Rotmat <-> quaternion (w,x,y,z) + window mean
# ============================================================

def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    R: (...,3,3) -> (...,4) (w,x,y,z), normalized, canonicalized w>=0
    """
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    def _norm(q):
        q = F.normalize(q, dim=-1)
        return torch.where(q[..., :1] < 0, -q, q)

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
    q: (...,4) (w,x,y,z) -> (...,3,3)
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
    q: (K,4) quats, w: (K,) weights
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
def smooth_axis_angle_quat_mean(aa: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    aa: (T,3) axis-angle
    return: (T,3) smoothed in rotation space
    """
    T = aa.shape[0]
    if T <= 1:
        return aa

    device, dtype = aa.device, aa.dtype
    radius = max(1, int(math.ceil(3.0 * sigma)))
    w = _gaussian_weights(radius, sigma, device=device, dtype=dtype)
    K = 2 * radius + 1

    R = axis_angle_to_rotation_matrix(aa)            # (T,3,3)
    q = _rotmat_to_quat(R)                           # (T,4)
    q_pad = torch.cat([q[:1].expand(radius, 4), q, q[-1:].expand(radius, 4)], dim=0)  # (T+2r,4)

    q_out = torch.empty_like(q)
    for t in range(T):
        q_win = q_pad[t:t + K]  # (K,4)
        q_out[t] = _quat_weighted_mean_markley(q_win, w)

    R_out = _quat_to_rotmat(q_out)                   # (T,3,3)
    return rotation_matrix_to_axis_angle(R_out)      # (T,3)


# ============================================================
# Savitzky–Golay smoothing (no scipy)
# ============================================================

def _savgol_coeffs(window_length: int, polyorder: int, device, dtype) -> torch.Tensor:
    if window_length % 2 != 1 or window_length < 3:
        raise ValueError("window_length must be odd and >=3")
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
    x: (T,C) or (T,3) etc. Smooth along T.
    """
    if x.ndim != 2:
        raise ValueError(f"smooth_with_savgol expects (T,C), got {tuple(x.shape)}")
    T, C = x.shape
    if T < window_length:
        return x

    device, dtype = x.device, x.dtype
    c = _savgol_coeffs(window_length, polyorder, device=device, dtype=dtype)  # (W,)
    m = window_length // 2

    x_pad = torch.cat([x[:1].expand(m, C), x, x[-1:].expand(m, C)], dim=0)  # (T+2m,C)
    xp = x_pad.t().unsqueeze(0)  # (1,C,T+2m)

    weight = c.view(1, 1, -1).expand(C, 1, -1)  # (C,1,W)
    y = F.conv1d(xp, weight, padding=0, groups=C)  # (1,C,T)
    return y.squeeze(0).t()  # (T,C)


# ============================================================
# Your original spike metric (degrees) + segment split
# ============================================================

def _split_true_segments(mask_1d: torch.Tensor):
    """
    mask_1d: (L,) bool
    return list[(s,e)] inclusive, same logic as your SMPL snippet
    """
    L = mask_1d.shape[0]
    segments = []
    start = None
    for i in range(L):
        if mask_1d[i] and start is None:
            start = i
        elif (not mask_1d[i]) and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, L - 1))
    return segments


def _root_angle_deg(go_seg: torch.Tensor):
    """
    go_seg: (T,3) axis-angle
    return: (T-1,) degrees
    """
    R = axis_angle_to_rotation_matrix(go_seg)          # (T,3,3)
    R_rel = R[:-1].transpose(-1, -2) @ R[1:]           # (T-1,3,3)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos = (trace - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.acos(cos) * (180.0 / math.pi)


# ============================================================
# Public API (flattened (L*B, ...) <-> processed <-> flattened)
# ============================================================

@torch.no_grad()
def postprocess_human_params(
    params: dict,
    batch_size: int,
    soft_thr_deg: float = 10.0,
    hard_thr_deg: float = 20.0,
    smooth_min_len: int = 11,
    rot_sigma: float = 1.0,
    savgol_window: int = 11,
    savgol_poly: int = 5,
):
    """
    params keys:
      required:
        - global_rot: (LB,3) axis-angle
        - pred_cam_t: (LB,3)
      optional:
        - repr:       (LB,D)
        - mask:       (LB,) bool

    flatten order must be t-major: idx = t*B + b
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    if "global_rot" not in params or "pred_cam_t" not in params:
        raise KeyError("params must contain 'global_rot' and 'pred_cam_t'")

    global_rot = params["global_rot"]
    pred_cam_t = params["pred_cam_t"]

    if global_rot.ndim != 2 or global_rot.shape[1] != 3:
        raise ValueError(f"global_rot expected (LB,3), got {tuple(global_rot.shape)}")
    if pred_cam_t.ndim != 2 or pred_cam_t.shape[1] != 3:
        raise ValueError(f"pred_cam_t expected (LB,3), got {tuple(pred_cam_t.shape)}")

    LB = global_rot.shape[0]
    if pred_cam_t.shape[0] != LB:
        raise ValueError(f"LB mismatch: global_rot {LB}, pred_cam_t {pred_cam_t.shape[0]}")
    if LB % batch_size != 0:
        raise ValueError(f"LB={LB} is not divisible by batch_size={batch_size}")

    L = LB // batch_size
    B = batch_size
    device = global_rot.device

    mask = params.get("mask", None)
    if mask is None:
        mask = torch.ones(LB, dtype=torch.bool, device=device)
    else:
        mask = mask.bool().to(device=device)
        if mask.numel() != LB:
            raise ValueError(f"mask length {mask.numel()} != LB {LB}")

    repr_x = params.get("repr", None)
    if repr_x is not None:
        if repr_x.ndim != 2 or repr_x.shape[0] != LB:
            raise ValueError(f"repr expected (LB,D), got {tuple(repr_x.shape)}")

    # reshape to (L,B,*)
    go = global_rot.detach().clone().view(L, B, 3)
    tr = pred_cam_t.detach().clone().view(L, B, 3)
    mk = mask.view(L, B)

    if repr_x is not None:
        D = repr_x.shape[1]
        rx = repr_x.detach().clone().view(L, B, D)
    else:
        rx = None

    # process per batch element
    for b in range(B):
        segments = _split_true_segments(mk[:, b])
        if not segments:
            continue

        # -------- 1) despike (copy previous for go+tr (+repr)) --------
        for s, e in segments:
            T = e - s + 1
            if T < 3:
                continue

            ang = _root_angle_deg(go[s:e + 1, b])  # (T-1,)
            for t in range(1, T - 1):
                a_prev = ang[t - 1].item()
                a_next = ang[t].item()
                if (a_prev > hard_thr_deg and a_next < soft_thr_deg) or (a_next > hard_thr_deg and a_prev < soft_thr_deg):
                    abs_t = s + t
                    go[abs_t, b] = go[abs_t - 1, b]
                    tr[abs_t, b] = tr[abs_t - 1, b]
                    if rx is not None:
                        rx[abs_t, b] = rx[abs_t - 1, b]

        # -------- 2) smoothing (only if long enough) --------
        for s, e in segments:
            T = e - s + 1
            if T < smooth_min_len:
                continue

            # global_rot: rotation-aware smoothing
            go_seg = go[s:e + 1, b]  # (T,3)
            go[s:e + 1, b] = smooth_axis_angle_quat_mean(go_seg, sigma=rot_sigma)

            # pred_cam_t: savgol
            tr_seg = tr[s:e + 1, b]  # (T,3)
            tr[s:e + 1, b] = smooth_with_savgol(tr_seg, window_length=savgol_window, polyorder=savgol_poly)

            # repr: savgol in Euclidean space
            if rx is not None:
                rx_seg = rx[s:e + 1, b]  # (T,D)
                rx[s:e + 1, b] = smooth_with_savgol(rx_seg, window_length=savgol_window, polyorder=min(savgol_poly, savgol_window - 1))

    # flatten back to (LB,*)
    out = dict(params)
    out["global_rot"] = go.view(LB, 3)
    out["pred_cam_t"] = tr.view(LB, 3)
    if rx is not None:
        out["repr"] = rx.view(LB, -1)
    return out


# ============================================================
# Minimal call example (optional)
# ============================================================
if __name__ == "__main__":
    L, B = 20, 4
    LB = L * B
    params = {
        "global_rot": torch.zeros(LB, 3),
        "pred_cam_t": torch.zeros(LB, 3),
        "repr": torch.randn(LB, 133) * 0.01,
        "mask": torch.ones(LB, dtype=torch.bool),
    }
    out = postprocess_human_params(params, batch_size=B)
    print(out["global_rot"].shape, out["pred_cam_t"].shape, out["repr"].shape)
