# post_process.py
# Drop-in postprocess for (L, D), robust to missing frames (all-NaN rows).
# Works for D=133 (body_pose) and D=108 (hand), single-person or multi-person (L=T*N).
#
# Key features:
#  - NEVER crash on missing frames / NaN rows
#  - spike detect (scalar) + linear inpaint (gap-limited)
#  - velocity-domain gaussian smoothing (continuous, reduces "顿挫")
#  - optional tiny position gaussian to shave micro-steps
#  - strong constraint for specific 3-dim Euler groups (quat spike -> slerp inpaint -> quat EMA)
#  - index-free "extra-strong topK dims" smoothing (often stabilizes elbow/knee/etc without knowing indices)
#
# Backward-compat:
#  - accepts enable_strong_fix as alias of enable_strong_groups
#  - accepts enable_wrist_fix / wrist_* as alias of strong_* for D==133 (if you still pass those)

from __future__ import annotations
from typing import Dict, Optional, List, Tuple
import math
import torch
import torch.nn.functional as F
import roma  # already in your project


# -----------------------------
# helpers: reshape LxD <-> TxNxD
# -----------------------------
def _to_tnd(x: torch.Tensor, num_person: int) -> torch.Tensor:
    assert x.dim() == 2, f"expect (L,D), got {tuple(x.shape)}"
    L, D = x.shape
    assert num_person >= 1
    assert L % num_person == 0, f"L({L}) must be divisible by num_person({num_person})"
    T = L // num_person
    return x.view(T, num_person, D)


def _from_tnd(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 3
    T, N, D = x.shape
    return x.reshape(T * N, D)


def _to_tn(mask: torch.Tensor, num_person: int) -> torch.Tensor:
    assert mask.dim() == 1
    L = mask.numel()
    assert num_person >= 1
    assert L % num_person == 0, f"L({L}) must be divisible by num_person({num_person})"
    T = L // num_person
    return mask.view(T, num_person)


def _finite_row_mask(x: torch.Tensor) -> torch.Tensor:
    """(L,D) -> (L,) True if the whole row is finite."""
    if x.numel() == 0:
        return torch.zeros((0,), device=x.device, dtype=torch.bool)
    return torch.isfinite(x).all(dim=-1)


# -----------------------------
# robust window predictor (exclude current frame)
# -----------------------------
def _robust_pred_excluding_center(x_t: torch.Tensor) -> torch.Tensor:
    """
    x_t: (W, ...) where W is odd and includes center element.
    Return robust prediction using median of neighbors excluding center.
    """
    W = x_t.shape[0]
    assert W >= 3 and (W % 2 == 1), "window W should be odd and >=3"
    c = W // 2
    neigh = torch.cat([x_t[:c], x_t[c + 1 :]], dim=0)
    return neigh.median(dim=0).values


# -----------------------------
# spike detection for scalar dims
# -----------------------------
def _detect_spikes_scalar(
    x: torch.Tensor,            # (T,N,D)
    vis: torch.Tensor,          # (T,N) bool
    spike_w: int = 7,           # odd window
    ratio_thr: float = 3.0,
    abs_thr_deg: float = 10.0,
    expand: int = 2,
) -> torch.Tensor:
    """
    Return spike_mask: (T,N,D) bool
    NOTE: x may contain NaNs at invisible frames; we nan_to_num + gate by vis.
    """
    T, N, D = x.shape
    if T < 3:
        return torch.zeros((T, N, D), dtype=torch.bool, device=x.device)

    W = spike_w + (spike_w % 2 == 0)
    W = max(W, 3)
    pad = W // 2

    x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_pad = torch.cat([x_safe[:1].repeat(pad, 1, 1), x_safe, x_safe[-1:].repeat(pad, 1, 1)], dim=0)

    spike = torch.zeros((T, N, D), dtype=torch.bool, device=x.device)
    abs_thr = abs_thr_deg * (math.pi / 180.0)

    for t in range(T):
        if not vis[t].any():
            continue
        wchunk = x_pad[t : t + W]  # (W,N,D)
        pred = _robust_pred_excluding_center(wchunk)  # (N,D)
        cur = x_safe[t]

        diff = (cur - pred).abs()
        neigh = torch.cat([wchunk[:pad], wchunk[pad + 1 :]], dim=0)  # (W-1,N,D)
        scale = (neigh - pred.unsqueeze(0)).abs().median(dim=0).values
        scale = torch.clamp(scale, min=1e-6)

        ratio = diff / scale
        bad = (ratio > ratio_thr) & (diff > abs_thr)
        bad = bad & vis[t].unsqueeze(-1)
        spike[t] = bad

    if expand > 0:
        s = spike.permute(1, 2, 0).float()  # (N,D,T)
        k = 2 * expand + 1
        s = F.max_pool1d(s, kernel_size=k, stride=1, padding=expand)
        spike = (s > 0).permute(2, 0, 1).contiguous()

    return spike


# -----------------------------
# inpaint (gap-limited), optionally fill missing too (internal)
# -----------------------------
def _inpaint_scalar_linear(
    x: torch.Tensor,                # (T,N,D)
    need_fill: torch.Tensor,        # (T,N,D) bool  True => replace by interpolation
    valid: torch.Tensor,            # (T,N) bool    True => this frame is a "real observation" (anchor)
    max_gap: int = 96,
) -> torch.Tensor:
    """
    Linear inpaint per-dim:
      - anchors are frames where valid==True and need_fill==False
      - frames with need_fill==True will be interpolated between nearest anchors (gap-limited)
    This is safe for NaNs because we should pass x_safe.
    """
    T, N, D = x.shape
    out = x.clone()

    if T < 2:
        return out

    idx = torch.arange(T, device=x.device)

    for n in range(N):
        v_n = valid[:, n]
        if v_n.sum() < 2:
            continue

        for d in range(D):
            fill_td = need_fill[:, n, d]
            if not fill_td.any():
                continue

            # anchors: valid and not needing fill
            anchor = v_n & (~fill_td)
            if anchor.sum() < 2:
                continue

            anchor_idx = idx[anchor]
            for t in idx[fill_td].tolist():
                prev = anchor_idx[anchor_idx < t]
                nxt = anchor_idx[anchor_idx > t]
                if prev.numel() == 0 or nxt.numel() == 0:
                    continue
                t0 = int(prev[-1].item())
                t1 = int(nxt[0].item())
                if (t1 - t0) > max_gap:
                    continue
                v0 = out[t0, n, d]
                v1 = out[t1, n, d]
                a = (t - t0) / max(1, (t1 - t0))
                out[t, n, d] = (1 - a) * v0 + a * v1

    return out


# -----------------------------
# gaussian helpers
# -----------------------------
def _gaussian_kernel1d(sigma: float, radius: int, device, dtype) -> torch.Tensor:
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (xs / max(sigma, 1e-6)) ** 2)
    return k / (k.sum() + 1e-12)


# -----------------------------
# velocity-domain gaussian smoothing (continuous / reduce staccato)
# -----------------------------
def _velocity_gaussian(
    x: torch.Tensor,          # (T,N,D) (assumed finite)
    sigma: float,
    passes: int = 1,
) -> torch.Tensor:
    """
    Smooth in velocity domain:
      v[t] = x[t]-x[t-1]
      v_s = Gaussian(v)
      x'[t] = x[0] + cumsum(v_s)
    """
    T, N, D = x.shape
    if T < 3 or sigma <= 0:
        return x

    radius = int(math.ceil(3 * sigma))
    k = _gaussian_kernel1d(sigma, radius, x.device, x.dtype)
    K = k.numel()

    y = x.permute(1, 2, 0).contiguous().view(N * D, 1, T)

    for _ in range(max(1, passes)):
        v = y[:, :, 1:] - y[:, :, :-1]  # (ND,1,T-1)
        v_pad = F.pad(v, (radius, radius), mode="reflect")
        v_s = F.conv1d(v_pad, k.view(1, 1, K))
        y2 = y.clone()
        y2[:, :, 1:] = y2[:, :, :1] + torch.cumsum(v_s, dim=-1)
        y = y2

    out = y.view(N, D, T).permute(2, 0, 1).contiguous()
    return out


def _pos_gaussian(
    x: torch.Tensor,      # (T,N,D) (assumed finite)
    sigma: float,
    passes: int = 1,
) -> torch.Tensor:
    T, N, D = x.shape
    if T < 3 or sigma <= 0:
        return x
    radius = int(math.ceil(3 * sigma))
    k = _gaussian_kernel1d(sigma, radius, x.device, x.dtype)
    K = k.numel()

    y = x.permute(1, 2, 0).contiguous().view(N * D, 1, T)
    for _ in range(max(1, passes)):
        y_pad = F.pad(y, (radius, radius), mode="reflect")
        y = F.conv1d(y_pad, k.view(1, 1, K))
    out = y.view(N, D, T).permute(2, 0, 1).contiguous()
    return out


# -----------------------------
# strong constraint for 3D Euler groups (quat spike -> slerp inpaint -> quat EMA)
# -----------------------------
def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=0.999999)
    omega = torch.acos(dot)
    so = torch.sin(omega).clamp(min=1e-8)
    s0 = torch.sin((1 - t) * omega) / so
    s1 = torch.sin(t * omega) / so
    return s0 * q0 + s1 * q1


def _ema_smooth_quat(
    q: torch.Tensor,        # (T,N,4)
    vis: torch.Tensor,      # (T,N) -- only update on visible frames
    alpha: float = 0.88,
    passes: int = 5,
) -> torch.Tensor:
    T, N, _ = q.shape
    if T < 2:
        return q
    out = q.clone()

    def _forward(inp: torch.Tensor) -> torch.Tensor:
        y = inp.clone()
        for n in range(N):
            first = None
            for t in range(T):
                if vis[t, n]:
                    first = t
                    break
            if first is None:
                continue
            for t in range(first + 1, T):
                if not vis[t, n]:
                    continue
                y[t, n] = _quat_slerp(y[t - 1, n], y[t, n], 1 - alpha)
        return y

    for _ in range(max(1, passes)):
        out = _forward(out)
        out = torch.flip(_forward(torch.flip(out, dims=[0])), dims=[0])

    out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return out


def _euler_group_fix_and_smooth(
    x: torch.Tensor,                    # (T,N,133) Euler scalars (radians)
    vis: torch.Tensor,                  # (T,N)
    groups: List[Tuple[int, int, int]],
    order: str = "XZY",
    spike_w: int = 9,
    ratio_thr: float = 2.2,
    abs_thr_deg: float = 6.0,
    expand: int = 4,
    ema_alpha: float = 0.88,
    ema_passes: int = 5,
    missing_max_gap: int = 96,
) -> torch.Tensor:
    """
    For each group (a,b,c): treat as 3D Euler triplet:
      - convert to quat
      - detect spikes on quat (angle to robust pred)
      - treat (spike OR missing) as "need repair" (gap-limited)
      - slerp inpaint
      - EMA smooth in quat space
      - back to euler
    """
    T, N, D = x.shape
    if T < 3:
        return x

    out = x.clone()
    out_safe = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    for (a, b, c) in groups:
        if not (0 <= a < D and 0 <= b < D and 0 <= c < D):
            continue

        e = out_safe[:, :, [a, b, c]]  # (T,N,3)
        R = roma.euler_to_rotmat(order, e.reshape(-1, 3)).view(T, N, 3, 3)
        q = roma.rotmat_to_unitquat(R.reshape(-1, 3, 3)).view(T, N, 4)

        # spike detect on quat (angle to robust pred)
        W = spike_w + (spike_w % 2 == 0)
        W = max(W, 3)
        pad = W // 2
        q_pad = torch.cat([q[:1].repeat(pad, 1, 1), q, q[-1:].repeat(pad, 1, 1)], dim=0)

        abs_thr = abs_thr_deg * (math.pi / 180.0)
        bad = torch.zeros((T, N), dtype=torch.bool, device=x.device)

        for t in range(T):
            if not vis[t].any():
                continue
            wq = q_pad[t : t + W]  # (W,N,4)
            pred = _robust_pred_excluding_center(wq)
            pred = pred / pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            cur = q[t]

            dot = (cur * pred).sum(dim=-1).abs().clamp(max=0.999999)
            ang = 2.0 * torch.acos(dot)

            neigh = torch.cat([wq[:pad], wq[pad + 1 :]], dim=0)
            neigh = neigh / neigh.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            dotn = (neigh * pred.unsqueeze(0)).sum(dim=-1).abs().clamp(max=0.999999)
            angn = 2.0 * torch.acos(dotn)
            scale = angn.median(dim=0).values.clamp(min=1e-6)
            ratio = ang / scale

            bad[t] = (ratio > ratio_thr) & (ang > abs_thr) & vis[t]

        if expand > 0:
            s = bad.permute(1, 0).float().unsqueeze(1)  # (N,1,T)
            k = 2 * expand + 1
            s = F.max_pool1d(s, kernel_size=k, stride=1, padding=expand)
            bad = (s.squeeze(1) > 0).permute(1, 0).contiguous()

        # inpaint by slerp between nearest good frames (also covers missing)
        q2 = q.clone()
        idx = torch.arange(T, device=x.device)

        for n in range(N):
            v_n = vis[:, n]
            if v_n.sum() < 2:
                continue

            # need repair on spikes OR missing frames
            need = bad[:, n] | (~v_n)

            # anchors: visible AND not need
            anchor = v_n & (~need)
            if anchor.sum() < 2:
                continue
            anchor_idx = idx[anchor]

            for t in idx[need].tolist():
                prev = anchor_idx[anchor_idx < t]
                nxt = anchor_idx[anchor_idx > t]
                if prev.numel() == 0 or nxt.numel() == 0:
                    continue
                t0 = int(prev[-1].item())
                t1 = int(nxt[0].item())
                if (t1 - t0) > missing_max_gap:
                    continue
                tt = (t - t0) / max(1, (t1 - t0))
                q2[t, n] = _quat_slerp(q2[t0, n], q2[t1, n], float(tt))

        q2 = _ema_smooth_quat(q2, vis, alpha=ema_alpha, passes=ema_passes)

        R2 = roma.unitquat_to_rotmat(q2.reshape(-1, 4)).view(T, N, 3, 3)
        e2 = roma.rotmat_to_euler(order, R2.reshape(-1, 3, 3)).view(T, N, 3)

        out[:, :, [a, b, c]] = e2

    # keep invisible frames as original (don’t invent)
    out = torch.where(vis.unsqueeze(-1), out, x)
    return out


# -----------------------------
# index-free "extra strong topK dims" smoothing
# -----------------------------
def _extra_strong_topk_dims(
    x_filled: torch.Tensor,     # (T,N,D) finite
    vis: torch.Tensor,          # (T,N)
    topk: int = 24,
    sigma: float = 6.5,
    passes: int = 2,
) -> torch.Tensor:
    """
    Pick topK jittery dims (median velocity magnitude) and apply stronger velocity smoothing.
    This often stabilizes elbows/knees/etc without knowing indices.
    """
    T, N, D = x_filled.shape
    if T < 3 or topk <= 0 or D <= 0:
        return x_filled

    # velocity and mask where consecutive frames visible
    v = (x_filled[1:] - x_filled[:-1]).abs()  # (T-1,N,D)
    vm = (vis[1:] & vis[:-1]).unsqueeze(-1)   # (T-1,N,1)
    v = v * vm

    # robust jitter score per dim (avg over persons)
    score = v.float().median(dim=0).values.mean(dim=0)  # (D,)
    k = min(int(topk), D)
    idx = torch.topk(score, k=k, largest=True).indices

    out = x_filled.clone()
    sel = out[:, :, idx]  # (T,N,k)
    sel = _velocity_gaussian(sel, sigma=float(sigma), passes=int(passes))
    out[:, :, idx] = sel
    return out


# -----------------------------
# main API
# -----------------------------
@torch.no_grad()
def postprocess_human_params(
    params: Dict[str, torch.Tensor],
    batch_size: int,
    order: str = "auto",

    # ---- spike detect ----
    spike_w: int = 8,
    spike_topk: int = 8,                # kept for compatibility; not used here as "topk dims"
    spike_ratio_thr: float = 3.2,
    spike_abs_thr_deg: float = 14.0,
    spike_expand: int = 2,

    # ---- spike/missing repair ----
    pred_w: int = 12,                   # kept for compatibility; we use missing_max_gap instead
    missing_max_gap: int = 96,

    # ---- smoothing (continuous) ----
    base_sigma: float = 3.0,
    smooth_passes: int = 3,

    # ---- optional final pos gaussian ----
    enable_pos_gaussian: bool = True,
    pos_sigma: float = 1.0,
    pos_passes: int = 1,

    # ---- strong constraint groups (recommended) ----
    enable_strong_groups: bool = True,
    strong_groups_body133: Optional[List[Tuple[int, int, int]]] = None,
    strong_spike_w: int = 11,
    strong_ratio_thr: float = 2.0,
    strong_abs_thr_deg: float = 5.0,
    strong_expand: int = 6,
    strong_ema_alpha: float = 0.92,
    strong_ema_passes: int = 7,

    # ---- backward compat aliases ----
    enable_strong_fix: Optional[bool] = None,          # alias -> enable_strong_groups
    enable_wrist_fix: Optional[bool] = None,           # alias -> enable_strong_groups (only when D==133)
    wrist_groups_body133: Optional[List[Tuple[int, int, int]]] = None,
    wrist_spike_w: Optional[int] = None,
    wrist_ratio_thr: Optional[float] = None,
    wrist_abs_thr_deg: Optional[float] = None,
    wrist_expand: Optional[int] = None,
    wrist_ema_alpha: Optional[float] = None,
    wrist_ema_passes: Optional[int] = None,

    # ---- index-free extra strong dims (helps elbow a lot) ----
    enable_extra_strong_topk: bool = True,
    extra_strong_topk: int = 24,
    extra_strong_sigma: float = 6.5,
    extra_strong_passes: int = 2,
) -> Dict[str, torch.Tensor]:
    """
    params:
      - "repr": (L,D) may contain all-NaN rows for missing frames
      - "mask": (L,) bool optional. If missing, we infer from finiteness.

    Output:
      - "repr": (L,D) smoothed (missing frames stay as original NaNs)
      - "mask": (L,) final valid mask (user mask & finite-row)
    """
    assert "repr" in params and params["repr"] is not None
    x = params["repr"]
    assert x.dim() == 2, f"repr should be (L,D), got {tuple(x.shape)}"
    device = x.device
    L, D = x.shape

    # alias handling
    if enable_strong_fix is not None:
        enable_strong_groups = bool(enable_strong_fix)

    # user mask
    if ("mask" in params) and (params["mask"] is not None):
        mask = params["mask"].to(device=device, dtype=torch.bool)
    else:
        mask = torch.ones((L,), device=device, dtype=torch.bool)

    # crucial: NaN rows => invalid (missing)
    mask = mask & _finite_row_mask(x)

    N = int(batch_size)
    if L == 0 or N <= 0 or (L % N != 0):
        return {"repr": x, "mask": mask}

    x_tnd = _to_tnd(x, N)         # (T,N,D) may have NaNs
    vis_tn = _to_tn(mask, N)      # (T,N)
    T = x_tnd.shape[0]

    # quick exits (avoid T-1==0 stats crash)
    if T < 2 or vis_tn.sum().item() < 2:
        return {"repr": x, "mask": mask}

    order_use = "XZY" if order == "auto" else order

    # make a safe working copy (finite) for all computations
    x_safe = torch.nan_to_num(x_tnd, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- strong constraint for D==133 (wrist / elbow groups etc) ----
    # support wrist_* alias (only for body_pose)
    if D == 133:
        if enable_wrist_fix is not None:
            enable_strong_groups = bool(enable_wrist_fix)
        if wrist_groups_body133 is not None:
            strong_groups_body133 = wrist_groups_body133
        if wrist_spike_w is not None:
            strong_spike_w = int(wrist_spike_w)
        if wrist_ratio_thr is not None:
            strong_ratio_thr = float(wrist_ratio_thr)
        if wrist_abs_thr_deg is not None:
            strong_abs_thr_deg = float(wrist_abs_thr_deg)
        if wrist_expand is not None:
            strong_expand = int(wrist_expand)
        if wrist_ema_alpha is not None:
            strong_ema_alpha = float(wrist_ema_alpha)
        if wrist_ema_passes is not None:
            strong_ema_passes = int(wrist_ema_passes)

    if enable_strong_groups and (D == 133):
        if strong_groups_body133 is None:
            # default: wrists only (you验证过可行)
            strong_groups_body133 = [(31, 32, 33), (41, 42, 43)]

        x_safe = _euler_group_fix_and_smooth(
            x_safe,
            vis_tn,
            groups=strong_groups_body133,
            order=order_use,
            spike_w=strong_spike_w,
            ratio_thr=strong_ratio_thr,
            abs_thr_deg=strong_abs_thr_deg,
            expand=strong_expand,
            ema_alpha=strong_ema_alpha,
            ema_passes=strong_ema_passes,
            missing_max_gap=missing_max_gap,
        )
        # x_safe is still finite

    # ---- spike detect (scalar) on x_safe, gated by vis ----
    spike_mask = _detect_spikes_scalar(
        x_safe, vis_tn,
        spike_w=spike_w,
        ratio_thr=spike_ratio_thr,
        abs_thr_deg=spike_abs_thr_deg,
        expand=spike_expand,
    )

    # ---- inpaint (internal) ----
    # Fill BOTH spikes and missing frames to allow continuous smoothing,
    # but later we will restore missing frames to original NaNs.
    need_fill = spike_mask | (~vis_tn.unsqueeze(-1))  # (T,N,D)

    # anchors are frames that are visible and not need_fill
    x_filled = _inpaint_scalar_linear(
        x_safe,
        need_fill=need_fill,
        valid=vis_tn,
        max_gap=missing_max_gap,
    )

    # ---- main continuous smoothing (velocity gaussian) ----
    x_s = _velocity_gaussian(x_filled, sigma=float(base_sigma), passes=int(smooth_passes))

    # ---- extra-strong topK jittery dims (index-free; usually stabilizes elbow) ----
    if enable_extra_strong_topk and extra_strong_topk > 0:
        x_s = _extra_strong_topk_dims(
            x_s,
            vis=vis_tn,
            topk=int(extra_strong_topk),
            sigma=float(extra_strong_sigma),
            passes=int(extra_strong_passes),
        )

    # ---- optional tiny pos gaussian to shave micro-steps ----
    if enable_pos_gaussian and pos_sigma > 0:
        x_s = _pos_gaussian(x_s, sigma=float(pos_sigma), passes=int(pos_passes))

    # restore missing frames to original (NaN rows stay NaN)
    x_out_tnd = torch.where(vis_tn.unsqueeze(-1), x_s, x_tnd)

    y = _from_tnd(x_out_tnd)
    return {"repr": y, "mask": mask}


# ======================================================================================
# Example CALL (copy-paste) — body_pose + hand
# ======================================================================================
def example_call(pose_output: Dict[str, Dict[str, torch.Tensor]]) -> None:
    # -------------------------
    # body_pose: (L,133)
    # -------------------------
    if ("mhr" in pose_output) and ("body_pose" in pose_output["mhr"]) and (pose_output["mhr"]["body_pose"] is not None):
        body_pose = pose_output["mhr"]["body_pose"]

        params = {
            "repr": body_pose,  # may contain all-NaN rows for missing frames
            # OPTIONAL: if you already have valid mask, pass it; otherwise omit it.
            # "mask": your_mask_bool_L,
        }

        out = postprocess_human_params(
            params,
            batch_size=1,  # 单人
            order="auto",

            # spike detect（身体别太敏感）
            spike_w=8,
            spike_ratio_thr=3.2,
            spike_abs_thr_deg=14.0,
            spike_expand=2,

            # gap-limited fill for missing/spikes (internal only)
            missing_max_gap=96,

            # 连续平滑（核心：velocity gaussian）
            base_sigma=3.0,
            smooth_passes=3,

            # 肘未知时：这招通常最有效（把最抖的 topK 维度二次加重平滑）
            enable_extra_strong_topk=True,
            extra_strong_topk=24,
            extra_strong_sigma=6.5,
            extra_strong_passes=2,

            # 强约束组：默认只锁 wrist；如果你以后知道肘 index，就加进去
            enable_strong_groups=True,
            strong_groups_body133=[(31, 32, 33), (41, 42, 43)],

            # 最后一点点 position gaussian 抹掉“细小台阶”
            enable_pos_gaussian=True,
            pos_sigma=1.0,
            pos_passes=1,
        )
        pose_output["mhr"]["body_pose"] = out["repr"]

    # -------------------------
    # hand: (L,108)
    # -------------------------
    if ("mhr" in pose_output) and ("hand" in pose_output["mhr"]) and (pose_output["mhr"]["hand"] is not None):
        hand = pose_output["mhr"]["hand"]
        params_h = {
            "repr": hand,  # may contain all-NaN rows too
            # "mask": your_mask_bool_L,
        }

        out_h = postprocess_human_params(
            params_h,
            batch_size=1,
            order="auto",

            # spike detect（手更敏感）
            spike_w=10,
            spike_ratio_thr=2.0,
            spike_abs_thr_deg=6.0,
            spike_expand=4,

            missing_max_gap=96,

            # 连续平滑更强
            base_sigma=3.6,
            smooth_passes=4,

            enable_extra_strong_topk=True,
            extra_strong_topk=24,
            extra_strong_sigma=7.0,
            extra_strong_passes=2,

            # hand 不是 133，这里强约束组不会触发（安全）
            enable_strong_groups=True,

            enable_pos_gaussian=True,
            pos_sigma=1.2,
            pos_passes=1,
        )
        pose_output["mhr"]["hand"] = out_h["repr"]
