import os
import glob
import cv2

def jpg_folder_to_mp4(folder: str, output_filename: str, fps: int = 25, long_edge: int = 1080):
    """
    Convert JPG images in a folder into an MP4 video, sorted by filename.
    Resize images so that the long edge equals `long_edge`.

    Parameters
    ----------
    folder : str
        Path to the folder containing JPG images.
    output_filename : str
        Output MP4 file path.
    fps : int, default 25
        Video frame rate.
    long_edge : int, default 1080
        Target size for the longer edge.
    """
    patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    img_paths = []
    for p in patterns:
        img_paths.extend(glob.glob(os.path.join(folder, p)))

    if not img_paths:
        raise ValueError(f"No JPG images found in folder: {folder}")

    img_paths = sorted(img_paths)

    # Read first image
    first_img = cv2.imread(img_paths[0])
    if first_img is None:
        raise ValueError(f"Failed to read image: {img_paths[0]}")

    h, w = first_img.shape[:2]

    # Compute resize scale (keep aspect ratio)
    scale = long_edge / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Resize first frame
    first_img = cv2.resize(first_img, (new_w, new_h))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (new_w, new_h))

    writer.write(first_img)

    for path in img_paths[1:]:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: skipped unreadable image {path}")
            continue

        img = cv2.resize(img, (new_w, new_h))
        writer.write(img)

    writer.release()
    print(f"Saved video to: {output_filename} ({new_w}x{new_h})")


jpg_folder_to_mp4(
    "/home/data/hmq/datasets/hmr/sub_3",
    "example1.mp4"
)
