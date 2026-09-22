import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient


def manifest(sid: str) -> str:
    return json.dumps({"session_id": sid, "country": "IN", "complete": True, "frames": [], "stills": [], "notes": [],
                       "depth_width": 1, "depth_height": 1, "rgb_width": 1, "rgb_height": 1, "device_model": "x",
                       "ios_version": "x", "started_at_unix": 0, "relocalized_against": None,
                       "relocalized_at_ar_timestamp": None, "faces_blurred": False,
                       "audio": {"file": "", "sample_rate": 0, "channels": 0, "start_ar_timestamp": 0},
                       "world_map": None, "dropped_frames": 0, "ended_at_unix": None})


def zip_bytes(entries: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in entries.items():
            z.writestr(name, content)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBPIPE_DATA", str(tmp_path))
    monkeypatch.delenv("UPLOAD_TOKEN", raising=False)
    import importlib

    from libpipe import config, server
    importlib.reload(config)
    importlib.reload(server)
    return TestClient(server.app), server, tmp_path


def test_wrapped_zip_matches_nsfilecoordinator_shape(client):
    c, server, data_dir = client
    body = zip_bytes({"my_session/manifest.json": manifest("my_session")})
    r = c.put("/upload", content=body, headers={"X-Session-Name": "my_session"})
    assert r.status_code == 200 and r.json()["session_id"] == "my_session"
    assert (data_dir / "sessions" / "my_session" / "manifest.json").exists()


def test_flat_zip_is_actually_tolerated(client):
    """Regression test: the code used to claim flat archives were tolerated but they were not - extractall put
    manifest.json at the sessions root, never at sessions/<name>/manifest.json, so this always 400'd."""
    c, server, data_dir = client
    body = zip_bytes({"manifest.json": manifest("flat_session")})
    r = c.put("/upload", content=body, headers={"X-Session-Name": "flat_session"})
    assert r.status_code == 200 and r.json()["session_id"] == "flat_session"
    assert (data_dir / "sessions" / "flat_session" / "manifest.json").exists()
    assert not (data_dir / "sessions" / "manifest.json").exists()   # moved into the session folder, not left at root


def test_flat_zip_with_multiple_top_level_entries(client):
    """A flat archive with several root-level files/dirs (audio.wav, frames/...) must all move, each only once."""
    c, server, data_dir = client
    body = zip_bytes({"manifest.json": manifest("multi"), "audio.wav": "x",
                      "frames/000000_rgb.jpg": "x", "frames/000001_rgb.jpg": "x"})
    r = c.put("/upload", content=body, headers={"X-Session-Name": "multi"})
    assert r.status_code == 200
    d = data_dir / "sessions" / "multi"
    assert (d / "manifest.json").exists() and (d / "audio.wav").exists()
    assert (d / "frames" / "000000_rgb.jpg").exists() and (d / "frames" / "000001_rgb.jpg").exists()


def test_missing_manifest_rejected(client):
    c, server, data_dir = client
    body = zip_bytes({"session_x/notes.txt": "hello"})
    r = c.put("/upload", content=body, headers={"X-Session-Name": "session_x"})
    assert r.status_code == 400


def test_bad_session_name_rejected(client):
    c, server, _ = client
    body = zip_bytes({"manifest.json": manifest("x")})
    r = c.put("/upload", content=body, headers={"X-Session-Name": "../../etc"})
    assert r.status_code == 400


def test_not_a_zip_rejected(client):
    c, server, _ = client
    r = c.put("/upload", content=b"not a zip file", headers={"X-Session-Name": "ok_name"})
    assert r.status_code == 400


def test_upload_token_enforced(client, monkeypatch):
    c, server, data_dir = client
    monkeypatch.setenv("UPLOAD_TOKEN", "secret123")
    body = zip_bytes({"session_y/manifest.json": manifest("session_y")})
    r = c.put("/upload", content=body, headers={"X-Session-Name": "session_y"})
    assert r.status_code == 401
    r2 = c.put("/upload", content=body, headers={"X-Session-Name": "session_y", "Authorization": "Bearer secret123"})
    assert r2.status_code == 200
