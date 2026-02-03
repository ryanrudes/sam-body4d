# smooth_human_lb_ultra.py
# Single-file, self-contained.
#
# ULTRA / OVERKILL STRONG SMOOTHING VERSION
# - For “super aggressive” smoothing of large jitter.
# - You can dial back later.
#
# Key changes vs normal:
#   1) Much more aggressive despike:
#        - original "jump-return" pattern
#        - + absolute jump threshold (any big inter-frame jump -> copy prev)
#   2) Rotations:
#        - big sigma + multiple passes
#        - optional extra "local slerp" relaxation pass
#   3) Translation / repr:
#        - heavy Gaussian conv (large sigma, multiple passes)
#        - optional EMA blend to further crush jitter
#
# Public API:
#   postprocess_human_params(params: dict, batch_size: int) -> dict
#
# Flatten order must be t-major: idx = t*B + b

import math
import torch
import torch.nn.functional as F


# ============================================================
# Axis-angle <-> rotmat
# ============================================================

def axis_angle_to_rotation_matrix(aa: torch.Tensor) -> torch.Tensor:
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
# Rotmat <-> quaternion (w,x,y,z) + Markley mean
# ============================================================

def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
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
def smooth_axis_angle_quat_mean_ultra(aa: torch.Tensor, sigma: float = 5.0, passes: int = 4) -> torch.Tensor:
    """
    aa: (T,3)
    ULTRA: sigma=5.0, passes=4  (yes, it will smear motion)
    """
    T = aa.shape[0]
    if T <= 1:
        return aa

    out = aa
    for _ in range(max(1, int(passes))):
        device, dtype = out.device, out.dtype
        radius = max(1, int(math.ceil(3.0 * sigma)))
        w = _gaussian_weights(radius, sigma, device=device, dtype=dtype)
        K = 2 * radius + 1

        R = axis_angle_to_rotation_matrix(out)
        q = _rotmat_to_quat(R)
        q_pad = torch.cat([q[:1].expand(radius, 4), q, q[-1:].expand(radius, 4)], dim=0)

        q_out = torch.empty_like(q)
        for t in range(T):
            q_win = q_pad[t:t + K]
            q_out[t] = _quat_weighted_mean_markley(q_win, w)

        out = rotation_matrix_to_axis_angle(_quat_to_rotmat(q_out))
    return out


# ============================================================
# Optional extra relaxation: local slerp to neighbors (ULTRA)
# ============================================================

def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """
    q0,q1: (...,4) unit quats (wxyz)
    """
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(-1.0, 1.0)

    # if very close, lerp
    close = dot > 0.9995
    q = q0 + t * (q1 - q0)
    q = F.normalize(q, dim=-1)

    theta0 = torch.acos(dot)
    sin0 = torch.sin(theta0).clamp_min(1e-8)
    theta = theta0 * t
    s0 = torch.sin(theta0 - theta) / sin0
    s1 = torch.sin(theta) / sin0
    qs = s0 * q0 + s1 * q1
    qs = F.normalize(qs, dim=-1)

    return torch.where(close, q, qs)


@torch.no_grad()
def relax_quat_local(q: torch.Tensor, alpha: float = 0.65, passes: int = 3) -> torch.Tensor:
    """
    q: (T,4) wxyz
    Each pass: pull each frame toward mean of neighbors via slerp.
    alpha: how strong the pull is (0..1)
    """
    T = q.shape[0]
    if T <= 2:
        return q

    out = q
    for _ in range(max(1, int(passes))):
        prev = out[:-2]
        mid = out[1:-1]
        nxt = out[2:]
        # neighbor "mean" approx by slerp(prev,nxt,0.5)
        nn = _slerp(prev, nxt, 0.5)
        mid2 = _slerp(mid, nn, float(alpha))
        out = torch.cat([out[:1], mid2, out[-1:]], dim=0)
    return out


@torch.no_grad()
def smooth_axis_angle_ultra_overkill(aa: torch.Tensor) -> torch.Tensor:
    """
    aa: (T,3) -> (T,3)
    Overkill pipeline: Markley mean (huge) + local relax
    """
    if aa.shape[0] <= 1:
        return aa
    R = axis_angle_to_rotation_matrix(aa)
    q = _rotmat_to_quat(R)
    # ultra mean
    aa1 = smooth_axis_angle_quat_mean_ultra(aa, sigma=5.0, passes=4)
    R1 = axis_angle_to_rotation_matrix(aa1)
    q1 = _rotmat_to_quat(R1)
    # extra relax
    q2 = relax_quat_local(q1, alpha=0.65, passes=3)
    return rotation_matrix_to_axis_angle(_quat_to_rotmat(q2))


# ============================================================
# Strong Gaussian conv smoothing (Euclidean) + EMA crush
# ============================================================

@torch.no_grad()
def smooth_with_gaussian_conv_ultra(x: torch.Tensor, sigma: float = 6.0, passes: int = 4) -> torch.Tensor:
    """
    x: (T,C) smooth along T by 1D Gaussian conv (replicate padding)
    ULTRA: sigma=6.0, passes=4
    """
    if x.ndim != 2:
        raise ValueError(f"smooth_with_gaussian_conv_ultra expects (T,C), got {tuple(x.shape)}")
    T, C = x.shape
    if T <= 1:
        return x

    out = x
    for _ in range(max(1, int(passes))):
        device, dtype = out.device, out.dtype
        radius = max(1, int(math.ceil(3.0 * sigma)))
        w = _gaussian_weights(radius, sigma, device=device, dtype=dtype)
        m = radius

        out_pad = torch.cat([out[:1].expand(m, C), out, out[-1:].expand(m, C)], dim=0)
        xp = out_pad.t().unsqueeze(0)  # (1,C,T+2m)

        weight = w.view(1, 1, -1).expand(C, 1, -1)
        y = F.conv1d(xp, weight, padding=0, groups=C)  # (1,C,T)
        out = y.squeeze(0).t()
    return out


@torch.no_grad()
def smooth_with_ema_ultra(x: torch.Tensor, alpha: float = 0.90, passes: int = 2) -> torch.Tensor:
    """
    x: (T,C)
    EMA: y[t] = alpha*y[t-1] + (1-alpha)*x[t]
    ULTRA: alpha=0.90, passes=2
    """
    if x.ndim != 2:
        raise ValueError(f"smooth_with_ema_ultra expects (T,C), got {tuple(x.shape)}")
    T, C = x.shape
    if T <= 1:
        return x

    out = x
    for _ in range(max(1, int(passes))):
        y = torch.empty_like(out)
        y[0] = out[0]
        for t in range(1, T):
            y[t] = alpha * y[t - 1] + (1.0 - alpha) * out[t]
        out = y
    return out


# ============================================================
# Spike metric + segment split
# ============================================================

def _split_true_segments(mask_1d: torch.Tensor):
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
    R = axis_angle_to_rotation_matrix(go_seg)
    R_rel = R[:-1].transpose(-1, -2) @ R[1:]
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos = (trace - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.acos(cos) * (180.0 / math.pi)


def _l2_jump(x: torch.Tensor):
    """
    x: (T,C) -> (T-1,)
    """
    d = x[1:] - x[:-1]
    return torch.linalg.norm(d, dim=-1)


# ============================================================
# Public API
# ============================================================

@torch.no_grad()
def postprocess_human_params(
    params: dict,
    batch_size: int,

    # --- despike thresholds (ULTRA sensitive) ---
    soft_thr_deg: float = 6.0,
    hard_thr_deg: float = 12.0,
    # any big inter-frame rot jump -> copy prev
    abs_jump_deg: float = 25.0,

    # any big translation jump -> copy prev (units depend on your scale)
    abs_jump_trans: float = 0.20,

    # repr jump threshold in L2 (tune if your repr scale differs)
    abs_jump_repr: float = 0.25,

    # --- smoothing ---
    smooth_min_len: int = 5,  # even short segments get smoothed
):
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

    for b in range(B):
        segments = _split_true_segments(mk[:, b])
        if not segments:
            continue

        # ===== 1) ULTRA DESPIKE =====
        for s, e in segments:
            T = e - s + 1
            if T < 2:
                continue

            go_seg = go[s:e + 1, b]  # (T,3)
            tr_seg = tr[s:e + 1, b]  # (T,3)
            ang = _root_angle_deg(go_seg)  # (T-1,)

            # rotation absolute jump
            for t in range(1, T):
                if ang[t - 1].item() > abs_jump_deg:
                    abs_t = s + t
                    go[abs_t, b] = go[abs_t - 1, b]
                    tr[abs_t, b] = tr[abs_t - 1, b]
                    if rx is not None:
                        rx[abs_t, b] = rx[abs_t - 1, b]

            # original jump-return pattern (more sensitive)
            if T >= 3:
                ang2 = _root_angle_deg(go[s:e + 1, b])
                for t in range(1, T - 1):
                    a_prev = ang2[t - 1].item()
                    a_next = ang2[t].item()
                    if (a_prev > hard_thr_deg and a_next < soft_thr_deg) or (a_next > hard_thr_deg and a_prev < soft_thr_deg):
                        abs_t = s + t
                        go[abs_t, b] = go[abs_t - 1, b]
                        tr[abs_t, b] = tr[abs_t - 1, b]
                        if rx is not None:
                            rx[abs_t, b] = rx[abs_t - 1, b]

            # translation absolute jump
            tj = _l2_jump(tr[s:e + 1, b])
            for t in range(1, T):
                if tj[t - 1].item() > abs_jump_trans:
                    abs_t = s + t
                    go[abs_t, b] = go[abs_t - 1, b]
                    tr[abs_t, b] = tr[abs_t - 1, b]
                    if rx is not None:
                        rx[abs_t, b] = rx[abs_t - 1, b]

            # repr absolute jump
            if rx is not None:
                rj = _l2_jump(rx[s:e + 1, b])
                for t in range(1, T):
                    if rj[t - 1].item() > abs_jump_repr:
                        abs_t = s + t
                        go[abs_t, b] = go[abs_t - 1, b]
                        tr[abs_t, b] = tr[abs_t - 1, b]
                        rx[abs_t, b] = rx[abs_t - 1, b]

        # ===== 2) ULTRA SMOOTH =====
        for s, e in segments:
            T = e - s + 1
            if T < smooth_min_len:
                continue

            # rotations: overkill rotation smoothing
            go[s:e + 1, b] = smooth_axis_angle_ultra_overkill(go[s:e + 1, b])

            # translations: huge gaussian + EMA crush
            tr_seg = tr[s:e + 1, b]
            tr_sm = smooth_with_gaussian_conv_ultra(tr_seg, sigma=6.0, passes=4)
            tr_sm = smooth_with_ema_ultra(tr_sm, alpha=0.90, passes=2)
            tr[s:e + 1, b] = tr_sm

            # repr: huge gaussian + EMA crush
            if rx is not None:
                rx_seg = rx[s:e + 1, b]
                rx_sm = smooth_with_gaussian_conv_ultra(rx_seg, sigma=6.0, passes=4)
                rx_sm = smooth_with_ema_ultra(rx_sm, alpha=0.90, passes=2)
                rx[s:e + 1, b] = rx_sm

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
    L, B = 60, 4
    LB = L * B
    params = {
        "global_rot": torch.zeros(LB, 3),
        "pred_cam_t": torch.zeros(LB, 3),
        "repr": torch.randn(LB, 133) * 0.01,
        "mask": torch.ones(LB, dtype=torch.bool),
    }

    out = postprocess_human_params(params, batch_size=B)
    print(out["global_rot"].shape, out["pred_cam_t"].shape, out["repr"].shape)
