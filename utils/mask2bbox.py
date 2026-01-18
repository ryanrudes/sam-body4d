from __future__ import annotations

from typing import Tuple, Union
import numpy as np
from PIL import Image


def mask_png_to_bbox_xyxy(
    mask_path: str,
    obj_id: Union[int, np.integer],
    *,
    inclusive: bool = False,
    return_none_if_empty: bool = True,
) -> Tuple[int, int, int, int] | None:
    """
    Read a PNG mask and compute bbox (x1, y1, x2, y2) for pixels == obj_id.

    Args:
        mask_path: Path to a PNG mask image.
        obj_id: Foreground object id. Foreground pixels are those with value == obj_id.
        inclusive: If False (default), returns half-open bbox: x2/y2 are exclusive (like slicing).
                  If True, returns inclusive bbox: x2/y2 are max indices (common in some datasets).
        return_none_if_empty: If True, return None when no pixels match; else return (0,0,0,0).

    Returns:
        (x1, y1, x2, y2) or None if empty and return_none_if_empty=True.
    """
    # Load as-is; for typical ID masks this will be mode 'L' (8-bit) or 'I;16' (16-bit)
    mask = np.array(Image.open(mask_path))

    if mask.ndim == 3:
        # If someone saved a colored mask, fall back to using the first channel
        mask = mask[..., 0]

    fg = (mask == int(obj_id))
    if not np.any(fg):
        return None if return_none_if_empty else (0, 0, 0, 0)

    ys, xs = np.where(fg)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    if inclusive:
        return (x1, y1, x2, y2)
    else:
        # half-open: [x1, x2+1), [y1, y2+1)
        return (x1, y1, x2 + 1, y2 + 1)
