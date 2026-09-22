"""Import a Stray Scanner capture (rgb.mp4, depth/*.png in mm, confidence/*.png, odometry.csv, camera_matrix.csv)
into the RoomCapture session layout so the same pipeline can run on it. Test data only: no audio, no notes, no
country (you must pass one), and its stills are ordinary video frames."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def quat_to_mat(x, y, z, w) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    x, y, z, w = (v / np.sqrt(n) for v in (x, y, z, w))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def import_stray(src: str | Path, dest: str | Path, country: str, every: int = 15, still_every: int = 4,
                 rgb_scale: float = 0.5) -> Path:
    """`every`: keep every Nth frame (45fps source). `still_every`: every Nth kept frame is also saved as a
    full-size still. RGB frames are downscaled by `rgb_scale` (intrinsics scaled to match)."""
    import cv2

    src, dest = Path(src), Path(dest)
    (dest / "frames").mkdir(parents=True, exist_ok=True)
    (dest / "stills").mkdir(exist_ok=True)
    K = np.array([[float(v) for v in r] for r in csv.reader(open(src / "camera_matrix.csv"))])
    odo = list(csv.reader(open(src / "odometry.csv")))[1:]
    cap = cv2.VideoCapture(str(src / "rgb.mp4"))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rw, rh = int(W * rgb_scale), int(H * rgb_scale)
    Ks = K.copy()
    Ks[:2] *= rgb_scale

    frames, stills, kept = [], [], 0
    for i, row in enumerate(odo):
        ok, img = cap.read()
        if not ok:
            break
        if i % every:
            continue
        x, y, z, qx, qy, qz, qw = (float(v) for v in row[2:9])
        T = np.eye(4)
        # Stray Scanner poses use the OpenCV camera convention (x right, y down, z forward); the pipeline (like the
        # RoomCapture app) uses ARKit's (x right, y up, looks down -z). Verified empirically: adjacent-frame depth
        # clouds agree to 0.7cm this way vs 15.9cm unconverted.
        T[:3, :3], T[:3, 3] = quat_to_mat(qx, qy, qz, qw) @ np.diag([1.0, -1.0, -1.0]), [x, y, z]
        idx, name = kept, f"{kept:06d}"
        cv2.imwrite(str(dest / "frames" / f"{name}_rgb.jpg"), cv2.resize(img, (rw, rh)), [cv2.IMWRITE_JPEG_QUALITY, 88])
        d = np.array(Image.open(src / "depth" / f"{i:06d}.png"), dtype=np.float32) / 1000.0
        c = np.array(Image.open(src / "confidence" / f"{i:06d}.png"), dtype=np.uint8)
        d.tofile(dest / "frames" / f"{name}_depth.bin")
        c.tofile(dest / "frames" / f"{name}_conf.bin")
        ts = float(row[0])
        frames.append({"index": idx, "ar_timestamp": ts, "rgb": f"frames/{name}_rgb.jpg", "depth": f"frames/{name}_depth.bin",
                       "confidence": f"frames/{name}_conf.bin", "transform": T.tolist(), "intrinsics": Ks.tolist(),
                       "tracking_state": "normal", "quality": None})
        if kept % still_every == 0:
            cv2.imwrite(str(dest / "stills" / f"{name}_still.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            stills.append({"index": idx, "ar_timestamp": ts, "file": f"stills/{name}_still.jpg", "width": W, "height": H,
                           "transform": T.tolist(), "intrinsics": K.tolist(), "tracking_state": "normal", "trigger": "auto"})
        kept += 1
    dh, dw = d.shape
    manifest = {"session_id": f"stray_{src.name}", "device_model": "stray-scanner", "ios_version": "n/a",
                "started_at_unix": 0.0, "country": country, "relocalized_against": None,
                "relocalized_at_ar_timestamp": None, "faces_blurred": False,
                "audio": {"file": "", "sample_rate": 0, "channels": 0, "start_ar_timestamp": 0.0},
                "depth_width": dw, "depth_height": dh, "rgb_width": rw, "rgb_height": rh, "frames": frames,
                "stills": stills, "notes": [], "world_map": None, "dropped_frames": 0, "complete": True,
                "ended_at_unix": None}
    (dest / "manifest.json").write_text(json.dumps(manifest))
    return dest
