import os
import numpy as np

def save_smpl_overlap_ply_single(verts_a, verts_b, idx, ply_path):
    """
    Save two SMPL vertex sets (one batch index) as a colored PLY point cloud.

    Colors:
      - verts_a: red  (255, 0, 0)
      - verts_b: blue (0, 0, 255)

    Fix:
      - DO NOT flip only one axis (reflection) because it will swap left/right.
      - If you want to fix the “upside-down” issue while keeping correct left/right,
        flip TWO axes (a proper 180° rotation), e.g. Y and Z.
    """

    def to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    A = to_numpy(verts_a)
    B = to_numpy(verts_b)

    if A.shape != B.shape or A.ndim != 3 or A.shape[-1] != 3:
        raise ValueError(f"Expected both inputs as (B, N, 3), got {A.shape} and {B.shape}")

    if not (0 <= idx < A.shape[0]):
        raise IndexError(f"idx {idx} out of range for batch size {A.shape[0]}")

    pts_a = A[idx].copy()
    pts_b = B[idx].copy()

    # ---- coordinate fix WITHOUT left-right mirroring ----
    # Old (bad): flip only Y -> reflection -> left/right swapped
    # New (good): flip Y and Z -> rotation 180° around X -> handedness preserved
    pts_a[:, 1] *= -1.0
    pts_a[:, 2] *= -1.0
    pts_b[:, 1] *= -1.0
    pts_b[:, 2] *= -1.0

    N = pts_a.shape[0]

    # colors: A = red, B = blue
    col_a = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (N, 1))
    col_b = np.tile(np.array([[0, 0, 255]], dtype=np.uint8), (N, 1))

    pts = np.concatenate([pts_a, pts_b], axis=0)
    cols = np.concatenate([col_a, col_b], axis=0)

    os.makedirs(os.path.dirname(ply_path) or ".", exist_ok=True)

    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
