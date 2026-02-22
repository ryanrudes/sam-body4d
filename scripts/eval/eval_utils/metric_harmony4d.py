import os
import torch
import numpy as np
from einops import einsum

from smplx_utils import make_smplx
from eval_utils.eval_tools import compute_camcoord_metrics, as_np_array


# ============================================================
# 1) Utilities
# ============================================================

def rotate_points(points: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """points: (B,N,3), R:(3,3) -> points @ R^T"""
    return torch.einsum("bni,ji->bnj", points, R.T)

def _apply_R(points: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bni,ji->bnj", points, R.T)

def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=1, keepdim=True)

def _kabsch_R(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A,B: (N,3) centered. Find R (3,3) s.t. A @ R^T ≈ B, det(R)=+1.
    """
    H = A.T @ B
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

def _R_pi(axis: int, device, dtype):
    if axis == 0:
        return torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]], device=device, dtype=dtype)
    if axis == 1:
        return torch.tensor([[-1,0,0],[0,1,0],[0,0,-1]], device=device, dtype=dtype)
    if axis == 2:
        return torch.tensor([[-1,0,0],[0,-1,0],[0,0,1]], device=device, dtype=dtype)
    raise ValueError("axis must be 0/1/2")

@torch.no_grad()
def estimate_constant_R_from_many_points(
    pred_j3d: torch.Tensor,      # (B,J,3)
    tgt_j3d: torch.Tensor,       # (B,J,3)
    pred_verts: torch.Tensor,    # (B,V,3)
    tgt_verts: torch.Tensor,     # (B,V,3)
    vert_sample_idx: torch.Tensor = None,  # (K,) long on cuda
    num_verts: int = 2048,
) -> torch.Tensor:
    """
    Returns constant R (3,3) that best aligns target->pred using MANY points.
    """
    device = pred_j3d.device
    if vert_sample_idx is None:
        V = pred_verts.shape[1]
        K = min(num_verts, V)
        vert_sample_idx = torch.linspace(0, V - 1, steps=K, device=device).long()

    P = torch.cat([pred_j3d, pred_verts[:, vert_sample_idx]], dim=1)  # (B,N,3)
    T = torch.cat([tgt_j3d,  tgt_verts[:,  vert_sample_idx]], dim=1)

    P0 = _center(P).reshape(-1, 3)
    T0 = _center(T).reshape(-1, 3)

    R = _kabsch_R(T0, P0)  # T0 @ R^T ≈ P0
    return R

@torch.no_grad()
def refine_R_by_pi_flips_on_many_points(
    pred_j3d: torch.Tensor,
    tgt_j3d: torch.Tensor,
    pred_verts: torch.Tensor,
    tgt_verts: torch.Tensor,
    R_init: torch.Tensor,
    vert_sample_idx: torch.Tensor = None,
    num_verts: int = 2048,
):
    """
    Try {I,Rx(pi),Ry(pi),Rz(pi)} @ R_init and pick min MSE on mean-centered many-points.
    """
    device, dtype = pred_j3d.device, pred_j3d.dtype
    I  = torch.eye(3, device=device, dtype=dtype)
    Rx = _R_pi(0, device, dtype)
    Ry = _R_pi(1, device, dtype)
    Rz = _R_pi(2, device, dtype)
    flips = [("I", I), ("Rx", Rx), ("Ry", Ry), ("Rz", Rz)]

    if vert_sample_idx is None:
        V = pred_verts.shape[1]
        K = min(num_verts, V)
        vert_sample_idx = torch.linspace(0, V - 1, steps=K, device=device).long()

    P = torch.cat([pred_j3d, pred_verts[:, vert_sample_idx]], dim=1)
    T = torch.cat([tgt_j3d,  tgt_verts[:,  vert_sample_idx]], dim=1)
    P0 = _center(P)

    best_R = R_init
    best_err = None
    best_name = None

    for name, F in flips:
        R = F @ R_init
        Trot = _apply_R(T, R)
        T0 = _center(Trot)
        err = (P0 - T0).pow(2).sum(dim=-1).mean()
        if best_err is None or err < best_err:
            best_err = err
            best_R = R
            best_name = name

    return best_R, {"pi_flip": best_name, "pi_flip_err": float(best_err.item())}

@torch.no_grad()
def translate_align_mean(pred_j3d: torch.Tensor, tgt_j3d: torch.Tensor, tgt_verts: torch.Tensor):
    delta = pred_j3d.mean(dim=1, keepdim=True) - tgt_j3d.mean(dim=1, keepdim=True)
    return tgt_j3d + delta, tgt_verts + delta


# ============================================================
# 2) Adjustable "small rotation around hips-axis ([1,2])"
#    + Auto-estimated yaw around that axis (optional)
# ============================================================

def _normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

@torch.no_grad()
def axis_angle_to_R(axis: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    """
    axis: (3,) unit vector
    angle_rad: scalar tensor
    returns R: (3,3)
    Rodrigues.
    """
    device, dtype = axis.device, axis.dtype
    axis = _normalize(axis)

    x, y, z = axis[0], axis[1], axis[2]
    O = torch.zeros((), device=device, dtype=dtype)
    K = torch.stack([
        torch.stack([ O, -z,  y]),
        torch.stack([ z,  O, -x]),
        torch.stack([-y,  x,  O]),
    ], dim=0)

    I = torch.eye(3, device=device, dtype=dtype)
    c = torch.cos(angle_rad)
    s = torch.sin(angle_rad)
    return I + s * K + (1 - c) * (K @ K)

@torch.no_grad()
def hips_defined_axis(j3d_24: torch.Tensor, use_neck_up: bool = True,
                     lhip: int = 1, rhip: int = 2, neck: int = 12) -> torch.Tensor:
    """
    Define an "up" axis from hips midpoint. Returns (3,) unit axis.
    """
    hips_mid = j3d_24[:, [lhip, rhip]].mean(dim=1)  # (B,3)
    if use_neck_up:
        up = (j3d_24[:, neck] - hips_mid).mean(dim=0)  # (3,)
    else:
        pelvis = j3d_24[:, 0]
        spine3 = j3d_24[:, 9]
        up = (spine3 - pelvis).mean(dim=0)
    return _normalize(up)

@torch.no_grad()
def apply_manual_rotation_around_hips_axis(
    j3d_24: torch.Tensor,
    verts: torch.Tensor,
    angle_deg: float,
    axis_source: str = "given",   # keep as given by default
    axis_given: torch.Tensor = None,  # (3,) if axis_source=="given"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rotate (j3d_24, verts) around a provided axis by user angle (degrees).
    Returns rotated_j3d, rotated_verts, R_manual.
    """
    device, dtype = j3d_24.device, j3d_24.dtype
    if axis_source != "given":
        raise ValueError("For safety, only 'given' axis_source is supported in this integrated version.")
    assert axis_given is not None and axis_given.numel() == 3
    axis = axis_given.to(device=device, dtype=dtype)
    axis = _normalize(axis)

    angle_rad = torch.tensor(angle_deg * np.pi / 180.0, device=device, dtype=dtype)
    R = axis_angle_to_R(axis, angle_rad)  # (3,3)

    j3d_rot = rotate_points(j3d_24, R)
    v_rot = rotate_points(verts, R)
    return j3d_rot, v_rot, R

@torch.no_grad()
def estimate_small_yaw_angle_deg_using_hips_forward(pred_j3d_24: torch.Tensor,
                                                   tgt_j3d_24: torch.Tensor,
                                                   up_axis_vec: torch.Tensor,
                                                   head_idx: int = 15,
                                                   lhip: int = 1, rhip: int = 2,
                                                   clamp_deg: float = 10.0) -> float:
    """
    Estimate a small yaw angle (deg) around the given up-axis vector that aligns
    pred forward to tgt forward, using hips midpoint -> head vector projected to plane orthogonal to up-axis.
    """
    device, dtype = pred_j3d_24.device, pred_j3d_24.dtype

    def proj_to_plane(v, n):  # v: (B,3), n:(3,)
        return v - (v @ n).unsqueeze(-1) * n.unsqueeze(0)

    n = _normalize(up_axis_vec.to(device=device, dtype=dtype))  # (3,)

    hips_p = pred_j3d_24[:, [lhip, rhip]].mean(dim=1)
    hips_t = tgt_j3d_24[:,  [lhip, rhip]].mean(dim=1)
    head_p = pred_j3d_24[:, head_idx]
    head_t = tgt_j3d_24[:,  head_idx]

    fwd_p = _normalize(head_p - hips_p)  # (B,3)
    fwd_t = _normalize(head_t - hips_t)

    fp = _normalize(proj_to_plane(fwd_p, n))
    ft = _normalize(proj_to_plane(fwd_t, n))

    cross = torch.cross(fp, ft, dim=-1)                   # (B,3)
    sinv = (cross * n.unsqueeze(0)).sum(dim=-1).mean()    # scalar
    cosv = (fp * ft).sum(dim=-1).mean()                   # scalar
    ang = torch.atan2(sinv, cosv)                         # rad

    ang_deg = float((ang * 180.0 / np.pi).item())
    if clamp_deg is not None:
        ang_deg = max(-clamp_deg, min(clamp_deg, ang_deg))
    return ang_deg


# ============================================================
# 2.5) STRONG yaw alignment using hips + shoulders (what you asked)
# ============================================================

def _project_to_plane_batch(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    # v:(B,3), n:(3,)
    return v - (v @ n).unsqueeze(-1) * n.unsqueeze(0)

@torch.no_grad()
def estimate_yaw_R_from_hips_shoulders(
    pred_j3d_24: torch.Tensor,   # (B,24,3)
    tgt_j3d_24: torch.Tensor,    # (B,24,3)
    up_axis_vec: torch.Tensor,   # (3,) unit
    lhip: int = 1, rhip: int = 2,
    lsho: int = 16, rsho: int = 17,
    robust: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Estimate an extra constant yaw-only rotation around up_axis_vec that best maps TARGET -> PRED.
    Uses two cues:
      - hips left-right axis (Rhip-Lhip)
      - torso direction (shoulder_mid - hip_mid)
    """
    device, dtype = pred_j3d_24.device, pred_j3d_24.dtype
    n = _normalize(up_axis_vec.to(device=device, dtype=dtype))  # (3,)

    # centers
    hp = pred_j3d_24[:, [lhip, rhip]].mean(dim=1)  # (B,3)
    ht = tgt_j3d_24[:,  [lhip, rhip]].mean(dim=1)

    # hips LR axis
    xp = pred_j3d_24[:, rhip] - pred_j3d_24[:, lhip]   # (B,3)
    xt = tgt_j3d_24[:,  rhip] - tgt_j3d_24[:,  lhip]

    # torso axis proxy
    sp = pred_j3d_24[:, [lsho, rsho]].mean(dim=1) - hp  # (B,3)
    st = tgt_j3d_24[:,  [lsho, rsho]].mean(dim=1) - ht

    # project to yaw plane
    xp2 = _normalize(_project_to_plane_batch(xp, n))
    xt2 = _normalize(_project_to_plane_batch(xt, n))
    sp2 = _normalize(_project_to_plane_batch(sp, n))
    st2 = _normalize(_project_to_plane_batch(st, n))

    def signed_angle(a, b, n):
        # rotate a -> b around n
        cross = torch.cross(a, b, dim=-1)                 # (B,3)
        sinv = (cross * n.unsqueeze(0)).sum(dim=-1)       # (B,)
        cosv = (a * b).sum(dim=-1)                        # (B,)
        return torch.atan2(sinv, cosv)                    # (B,)

    # need R such that (T @ R^T) matches P, so angle is tgt->pred
    ang_torso = signed_angle(st2, sp2, n)  # (B,)
    ang_hips  = signed_angle(xt2, xp2, n)  # (B,)

    if robust:
        ang_pair = torch.stack([ang_torso, ang_hips], dim=0)  # (2,B)
        ang_per_frame = ang_pair.median(dim=0).values         # (B,)
        ang0 = ang_per_frame.median()                         # scalar
    else:
        ang0 = 0.5 * (ang_torso.mean() + ang_hips.mean())

    R_yaw = axis_angle_to_R(n, ang0)
    return R_yaw, {"yaw_deg": float((ang0 * 180.0 / np.pi).item())}


# ============================================================
# 3) Metric class (FULL integrated) + angle interface
# ============================================================

class MetricMocap:
    def __init__(self, body_model_path):
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "mpjpe": {},
            "pve": {},
            "accel": {},
        }

        self.smplx = make_smplx("supermotion_EVAL3DPW", body_model_path=body_model_path)
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        self.smpl = {
            "male": make_smplx("smpl", body_model_path=body_model_path, gender="male"),
            "female": make_smplx("smpl", body_model_path=body_model_path, gender="female"),
            "neutral": make_smplx("smpl", body_model_path=body_model_path, gender="neutral"),
        }

        self.J_regressor = torch.load(f"{script_dir}/body_model/smpl_neutral_J_regressor.pt")
        self.J_regressor24 = torch.load(f"{script_dir}/body_model/smpl_neutral_J_regressor.pt")
        self.smplx2smpl = torch.load(f"{script_dir}/body_model/smplx2smpl_sparse.pt")

        self.faces_smplx = self.smplx.faces
        self.faces_smpl = self.smpl["male"].faces

    @torch.no_grad()
    def evaluate(self,
                 outputs,
                 target_c_params,
                 mhr_height,
                 smplx_vertices,
                 save_path=None,
                 smplx=None,
                 vid=None,
                 # ===== old interface =====
                 manual_hips_axis_deg: float = 0.0,   # rotate PRED around hips-up axis by this deg (optional)
                 auto_hips_axis: bool = False,        # small auto correction around up axis (optional)
                 auto_clamp_deg: float = 10.0,
                 # ===== NEW: strong yaw matching using hips+shoulders =====
                 strong_yaw_align: bool = True,
                 strong_yaw_lsho: int = 16,
                 strong_yaw_rsho: int = 17,
                 ):
        """
        Key behavior:
          - First, find a constant R_align (target->pred) using MANY points (J+V) + optional pi flips (you said up/down fixed).
          - Then, (Harmony4D-useful) optionally do a SECOND constant yaw-only alignment using hips+shoulders to fix left/right rotation.
          - Finally, optional manual rotation on PRED around an up axis (keep if you still want a knob).
        """

        self.smplx = self.smplx.cuda()
        self.smpl["neutral"] = self.smpl["neutral"].cuda()
        self.J_regressor = self.J_regressor.cuda()
        self.J_regressor24 = self.J_regressor24.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

        # ---------- pred (SMPLX -> SMPL -> joints) ----------
        smpl_out = self.smplx(**outputs)
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])  # (B,V,3)
        pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")                   # (B,24,3)

        # ---------- target (SMPL -> joints) ----------
        target_c_output = self.smpl["neutral"](**target_c_params)
        target_c_verts0 = target_c_output.vertices                                                  # (B,V,3)
        target_c_j3d0 = torch.matmul(self.J_regressor, target_c_verts0)                              # (B,24,3)

        # ============================================================
        # A) constant 3D rotation using MANY points (J+V)
        # ============================================================
        V = pred_c_verts.shape[1]
        K = min(2048, V)
        vert_idx = torch.linspace(0, V - 1, steps=K, device=pred_c_verts.device).long()

        R_align = estimate_constant_R_from_many_points(
            pred_j3d=pred_c_j3d,
            tgt_j3d=target_c_j3d0,
            pred_verts=pred_c_verts,
            tgt_verts=target_c_verts0,
            vert_sample_idx=vert_idx,
            num_verts=2048,
        )

        R_align, info_flip = refine_R_by_pi_flips_on_many_points(
            pred_j3d=pred_c_j3d,
            tgt_j3d=target_c_j3d0,
            pred_verts=pred_c_verts,
            tgt_verts=target_c_verts0,
            R_init=R_align,
            vert_sample_idx=vert_idx,
            num_verts=2048,
        )

        # apply to target
        target_c_j3d = rotate_points(target_c_j3d0, R_align)
        target_c_verts = rotate_points(target_c_verts0, R_align)

        # translation align (mean)
        target_c_j3d, target_c_verts = translate_align_mean(pred_c_j3d, target_c_j3d, target_c_verts)

        # ============================================================
        # A2) STRONG yaw-only alignment using hips + shoulders
        #     Fix "left/right rotation" (yaw) without touching pitch/roll.
        # ============================================================
        if strong_yaw_align:
            # define yaw axis (up). If your coordinate system has a known up axis, you can replace this with a fixed one.
            up_axis = hips_defined_axis(pred_c_j3d, use_neck_up=True, lhip=1, rhip=2, neck=12)  # (3,)

            R_yaw, info_yaw = estimate_yaw_R_from_hips_shoulders(
                pred_j3d_24=pred_c_j3d,
                tgt_j3d_24=target_c_j3d,
                up_axis_vec=up_axis,
                lhip=1, rhip=2,
                lsho=strong_yaw_lsho,
                rsho=strong_yaw_rsho,
                robust=True,
            )

            target_c_j3d = rotate_points(target_c_j3d, R_yaw)
            target_c_verts = rotate_points(target_c_verts, R_yaw)

            # re-translate align
            target_c_j3d, target_c_verts = translate_align_mean(pred_c_j3d, target_c_j3d, target_c_verts)
        else:
            info_yaw = {"yaw_deg": 0.0}

        # ============================================================
        # B) Optional: rotate PRED around an "up axis" (old knob)
        # ============================================================
        axis_vec = hips_defined_axis(pred_c_j3d, use_neck_up=True, lhip=1, rhip=2, neck=12)  # (3,)

        if auto_hips_axis:
            auto_deg = estimate_small_yaw_angle_deg_using_hips_forward(
                pred_j3d_24=pred_c_j3d,
                tgt_j3d_24=target_c_j3d,
                up_axis_vec=axis_vec,
                head_idx=15,
                lhip=1, rhip=2,
                clamp_deg=auto_clamp_deg,
            )
            manual_hips_axis_deg = manual_hips_axis_deg + auto_deg

        if abs(manual_hips_axis_deg) > 1e-9:
            pred_c_j3d, pred_c_verts, R_manual = apply_manual_rotation_around_hips_axis(
                j3d_24=pred_c_j3d,
                verts=pred_c_verts,
                angle_deg=manual_hips_axis_deg,
                axis_source="given",
                axis_given=axis_vec,
            )

        # ---------- Metrics ----------
        batch_eval = {
            "pred_j3d": pred_c_j3d,
            "target_j3d": target_c_j3d,
            "pred_verts": pred_c_verts,
            "target_verts": target_c_verts,
        }
        camcoord_metrics = compute_camcoord_metrics(batch_eval)
        for k in camcoord_metrics:
            self.metric_aggregator[k][vid] = as_np_array(camcoord_metrics[k])

        avg = camcoord_metrics["pa_mpjpe"].mean()
        print(f"{vid}: {avg:.4f} | pi_flip={info_flip.get('pi_flip', None)} | yaw_deg={info_yaw.get('yaw_deg', 0.0):.2f}")

        if save_path is not None:
            import csv
            mode = "a" if os.path.exists(save_path) else "w"
            with open(save_path, mode, newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([vid, "pa_mpjpe", camcoord_metrics["pa_mpjpe"].mean()] + list(camcoord_metrics["pa_mpjpe"]))
                writer.writerow([vid, "mpjpe", camcoord_metrics["mpjpe"].mean()] + list(camcoord_metrics["mpjpe"]))
                writer.writerow([vid, "pve", camcoord_metrics["pve"].mean()] + list(camcoord_metrics["pve"]))
                writer.writerow([vid, "accel", camcoord_metrics["accel"].mean()] + list(camcoord_metrics["accel"]))

        metrics_avg = {k: np.concatenate(list(v.values())).mean() for k, v in self.metric_aggregator.items()}
        print(metrics_avg)

        del smpl_out  # Prevent OOM
