"""Import a Record3D .r3d capture (zip: metadata, rgbd/N.jpg, rgbd/N.depth + N.conf LZFSE-compressed, sound.m4a)
into the RoomCapture session layout. Test data only: needs `pip install pyliblzfse`.

Pose convention: Record3D poses are [qx, qy, qz, qw, tx, ty, tz] and, unlike Stray Scanner, already use ARKit's camera
convention. Verified against ground truth: this mapping gave a 2.77m-tall, 3.9x4.9m cloud for a room measured at
2.6 x 3.54 x 3.77m; every alternative axis mapping smeared the room to a 7m+ footprint.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from .stray import quat_to_mat


def import_r3d(src: str | Path, dest: str | Path, country: str, every: int = 20, still_every: int = 2) -> Path:
    import liblzfse

    src, dest = Path(src), Path(dest)
    (dest / "frames").mkdir(parents=True, exist_ok=True)
    (dest / "stills").mkdir(exist_ok=True)
    z = zipfile.ZipFile(src)
    md = json.loads(z.read("metadata"))
    w, h, dw, dh = md["w"], md["h"], md["dw"], md["dh"]
    ts, poses, intr = md["frameTimestamps"], md["poses"], md["perFrameIntrinsicCoeffs"]

    audio = {"file": "", "sample_rate": 0, "channels": 0, "start_ar_timestamp": 0.0}
    if "sound.m4a" in z.namelist():
        (dest / "audio.m4a").write_bytes(z.read("sound.m4a"))
        audio = {"file": "audio.m4a", "sample_rate": 0, "channels": 0, "start_ar_timestamp": float(ts[0])}

    frames, stills, kept = [], [], 0
    for i in range(0, len(poses), every):
        name = f"{kept:06d}"
        jpg = z.read(f"rgbd/{i}.jpg")
        (dest / "frames" / f"{name}_rgb.jpg").write_bytes(jpg)
        d = np.frombuffer(liblzfse.decompress(z.read(f"rgbd/{i}.depth")), dtype=np.float32).reshape(dh, dw)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        c = np.frombuffer(liblzfse.decompress(z.read(f"rgbd/{i}.conf")), dtype=np.uint8).reshape(dh, dw)
        d.tofile(dest / "frames" / f"{name}_depth.bin")
        c.tofile(dest / "frames" / f"{name}_conf.bin")
        qx, qy, qz, qw, tx, ty, tz = poses[i]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = quat_to_mat(qx, qy, qz, qw), [tx, ty, tz]
        fx, fy, cx, cy = intr[i]
        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        frames.append({"index": kept, "ar_timestamp": float(ts[i]), "rgb": f"frames/{name}_rgb.jpg",
                       "depth": f"frames/{name}_depth.bin", "confidence": f"frames/{name}_conf.bin",
                       "transform": T.tolist(), "intrinsics": K, "tracking_state": "normal", "quality": None})
        if kept % still_every == 0:
            (dest / "stills" / f"{name}_still.jpg").write_bytes(jpg)
            stills.append({"index": kept, "ar_timestamp": float(ts[i]), "file": f"stills/{name}_still.jpg", "width": w,
                           "height": h, "transform": T.tolist(), "intrinsics": K, "tracking_state": "normal", "trigger": "auto"})
        kept += 1
    manifest = {"session_id": f"r3d_{src.stem}", "device_model": "record3d", "ios_version": "n/a", "started_at_unix": 0.0,
                "country": country, "relocalized_against": None, "relocalized_at_ar_timestamp": None,
                "faces_blurred": False, "audio": audio, "depth_width": dw, "depth_height": dh, "rgb_width": w,
                "rgb_height": h, "frames": frames, "stills": stills, "notes": [], "world_map": None,
                "dropped_frames": 0, "complete": True, "ended_at_unix": None}
    (dest / "manifest.json").write_text(json.dumps(manifest))
    return dest
