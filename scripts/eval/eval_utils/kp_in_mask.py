import numpy as np

def majority_keypoints_in_mask(keypoints: np.ndarray, mask: np.ndarray) -> bool:
    """
    Return True if more than half of valid keypoints lie inside the mask.
    """
    H, W = mask.shape
    pts = keypoints[:, :2]

    x = np.rint(pts[:, 0]).astype(int)  # column (x)
    y = np.rint(pts[:, 1]).astype(int)  # row (y)

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    num_valid = np.sum(valid)
    if num_valid == 0:
        return False

    num_inside = np.sum(mask[y[valid], x[valid]])
    return num_inside > (num_valid / 2)
