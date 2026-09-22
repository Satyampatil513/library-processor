"""Step 2: receives the app's `PUT <url>` of a zipped session with header X-Session-Name."""
from __future__ import annotations

import io
import os
import re
import zipfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from . import hitl_ui
from .config import Config
from .db import DB

cfg = Config().ensure()
app = FastAPI(title="libpipe ingest")
app.include_router(hitl_ui.router)
app.mount("/img", StaticFiles(directory=str(cfg.output_dir)), name="img")  # serves data/output/crops/... for the HITL page


@app.put("/upload")
async def upload(request: Request):
    token = os.environ.get("UPLOAD_TOKEN")
    if token and request.headers.get("authorization") != f"Bearer {token}":
        raise HTTPException(401, "bad token")
    name = request.headers.get("x-session-name", "")
    if not re.fullmatch(r"[\w\-.]+", name):
        raise HTTPException(400, "bad X-Session-Name")
    body = await request.body()
    dest = cfg.sessions_dir / name
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            for m in z.namelist():  # zip-slip guard
                if os.path.isabs(m) or ".." in m.split("/"):
                    raise HTTPException(400, "unsafe zip")
            z.extractall(cfg.sessions_dir)
    except zipfile.BadZipFile:
        raise HTTPException(400, "not a zip")
    # NSFileCoordinator zips contain the folder itself (-> sessions_dir/<name>/manifest.json). A flat archive
    # (manifest.json at the zip root) was previously claimed to be tolerated but never actually was - fixed:
    # move its extracted contents into dest so both shapes land in the same place.
    if not (dest / "manifest.json").exists() and (cfg.sessions_dir / "manifest.json").exists():
        dest.mkdir(parents=True, exist_ok=True)
        tops = {m.split("/")[0] for m in zipfile.ZipFile(io.BytesIO(body)).namelist()}
        for top in tops:
            src = cfg.sessions_dir / top
            if src.exists() and src != dest:
                src.rename(dest / top)
    if not (dest / "manifest.json").exists():
        raise HTTPException(400, "manifest.json missing")
    import json
    man = json.loads((dest / "manifest.json").read_text())
    DB(cfg.db_path).upsert("sessions", {"id": man["session_id"], "path": str(dest),
                                        "country": man["country"], "status": "uploaded"})
    return {"ok": True, "session_id": man["session_id"], "complete": man.get("complete", False)}
