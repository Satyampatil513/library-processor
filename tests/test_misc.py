import numpy as np

from libpipe.config import load_env
from libpipe.pricing import Pricer
from libpipe.regions import normalise
from libpipe.stray import quat_to_mat


def test_quaternion_identity_and_90deg_about_z():
    assert np.allclose(quat_to_mat(0, 0, 0, 1), np.eye(3))
    R = quat_to_mat(0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])


def test_opencv_to_arkit_pose_conversion_flips_y_and_z_axes():
    F = np.diag([1.0, -1.0, -1.0])
    cv_forward_cam = np.array([0, 0, 1.0])            # OpenCV: z forward
    arkit_cam = F @ cv_forward_cam                    # same ray in ARKit camera coords: looks down -z
    assert np.allclose(arkit_cam, [0, 0, -1])


def test_env_loader_strips_comments_and_ignores_empty(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("A_KEY=abc123   # note\nB_EMPTY=      # nothing\n#C=1\nD='quoted'\n")
    for k in ("A_KEY", "B_EMPTY", "D"):
        monkeypatch.delenv(k, raising=False)
    load_env(f)
    import os
    assert os.environ["A_KEY"] == "abc123" and "B_EMPTY" not in os.environ and os.environ["D"] == "quoted"
    for k in ("A_KEY", "D"):
        monkeypatch.delenv(k)


def test_keepa_is_optional(monkeypatch):
    monkeypatch.delenv("KEEPA_API_KEY", raising=False)
    assert Pricer().keepa is None
    monkeypatch.setenv("KEEPA_API_KEY", "x")
    assert Pricer().keepa is not None


def test_is_object_defaults_true_and_is_kept_when_false():
    raw = {"objects": [{"piece_ids": ["p1"], "is_book": False},
                       {"piece_ids": ["p2"], "is_book": False, "is_object": False}]}
    out = normalise(raw, ["p1", "p2"])
    assert [o["is_object"] for o in out] == [True, False]
