"""Reads a RoomCapture session folder (layout defined by the iOS app's README)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def mat4(rows) -> np.ndarray:
    return np.array(rows, dtype=np.float64).reshape(4, 4)


def mat3(rows) -> np.ndarray:
    return np.array(rows, dtype=np.float64).reshape(3, 3)


@dataclass
class Session:
    root: Path
    manifest: dict

    @classmethod
    def load(cls, root: str | Path) -> "Session":
        root = Path(root)
        return cls(root, json.loads((root / "manifest.json").read_text()))

    # -- identity -----------------------------------------------------------
    @property
    def id(self) -> str:
        return self.manifest["session_id"]

    @property
    def country(self) -> str:
        return self.manifest["country"]

    @property
    def complete(self) -> bool:
        return bool(self.manifest.get("complete"))

    @property
    def relocalized_against(self) -> str | None:
        return self.manifest.get("relocalized_against")

    @property
    def frames(self) -> list[dict]:
        return self.manifest["frames"]

    @property
    def stills(self) -> list[dict]:
        return self.manifest["stills"]

    @property
    def notes(self) -> list[dict]:
        return self.manifest["notes"]

    # -- pixel data ---------------------------------------------------------
    def rgb(self, frame: dict) -> Image.Image:
        return Image.open(self.root / frame["rgb"]).convert("RGB")

    def still(self, still: dict) -> Image.Image:
        return Image.open(self.root / still["file"]).convert("RGB")

    def depth(self, frame: dict) -> np.ndarray:
        h, w = self.manifest["depth_height"], self.manifest["depth_width"]
        return np.fromfile(self.root / frame["depth"], dtype=np.float32).reshape(h, w)

    def confidence(self, frame: dict) -> np.ndarray:
        h, w = self.manifest["depth_height"], self.manifest["depth_width"]
        return np.fromfile(self.root / frame["confidence"], dtype=np.uint8).reshape(h, w)

    # -- geometry -----------------------------------------------------------
    def unproject_depth(self, frame: dict, min_conf: int = 2, stride: int = 2) -> np.ndarray:
        """Depth image -> Nx3 world points. ARKit camera: x right, y up, looks down -z;
        intrinsics are for the RGB image so they are rescaled to the depth grid."""
        d = self.depth(frame)
        conf = self.confidence(frame)
        dh, dw = d.shape
        K = mat3(frame["intrinsics"])
        sx = dw / self.manifest["rgb_width"]
        sy = dh / self.manifest["rgb_height"]
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
        v, u = np.mgrid[0:dh:stride, 0:dw:stride]
        z = d[::stride, ::stride]
        ok = (z > 0.1) & (z < 6.0) & np.isfinite(z) & (conf[::stride, ::stride] >= min_conf)
        u, v, z = u[ok], v[ok], z[ok]
        cam = np.stack([(u - cx) / fx * z, -(v - cy) / fy * z, -z, np.ones_like(z)], axis=1)
        return (mat4(frame["transform"]) @ cam.T).T[:, :3]


def pixel_ray_point(frame: dict, u: float, v: float, depth: float, rgb_wh: tuple[int, int]) -> np.ndarray:
    """RGB-image pixel + metric depth -> world point."""
    K = mat3(frame["intrinsics"])
    cam = np.array([(u - K[0, 2]) / K[0, 0] * depth, -(v - K[1, 2]) / K[1, 1] * depth, -depth, 1.0])
    return (mat4(frame["transform"]) @ cam)[:3]


def project_world(frame: dict, pts: np.ndarray) -> np.ndarray:
    """World Nx3 -> Nx3 (u, v, depth) in the RGB image of `frame`. depth<=0 means behind camera."""
    K = mat3(frame["intrinsics"])
    inv = np.linalg.inv(mat4(frame["transform"]))
    cam = (inv @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    z = -cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = -K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([u, v, z], axis=1)
