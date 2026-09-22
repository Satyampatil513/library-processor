"""Central config: env keys and every tunable threshold from the pipeline diagram."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    """Minimal .env reader (no dependency). Real environment variables win; empty values are ignored;
    trailing `# comments` are stripped."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = re.sub(r"\s+#.*$", "", v).strip().strip('"').strip("'")
        if v and k.strip() not in os.environ:
            os.environ[k.strip()] = v


load_env()


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("LIBPIPE_DATA", "./data")))

    # models
    fable_model: str = "claude-fable-5-1"
    astra_model: str = "gpt-6-astra"
    whisper_model: str = "whisper-1"  # needs word timestamps (verbose_json)
    jev_url: str = "https://api.typesafe.ai/v1/systemone"
    jev_model: str = "jev-latest"

    # step 4 stitch: overlap must agree within 2-3cm
    stitch_tolerance_m: float = 0.03
    # step 6 find pieces
    merge_iou: float = 0.3
    # library-insurance scope includes furniture, not just books - these bound what counts as "one object";
    # see pieces.find_pieces docstring for why they were raised from an earlier book-only default
    piece_max_frame_frac: float = 0.6
    piece_max_dim_m: float = 3.0
    # step 7 regions
    region_size: int = 15
    # step 9 settle
    agree_sample_rate: float = 0.05  # 5% of agreements sampled to Jev [F12]
    jev_min_confidence: float = 0.6
    shift_tolerance: int = 1  # +-1 partition shift auto-fixed
    # step 13 drift pause
    drift_pause_agreement: float = 0.85

    def key(self, name: str) -> str | None:
        return os.environ.get(name) or None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "libpipe.db"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    def ensure(self) -> "Config":
        for p in (self.data_dir, self.sessions_dir, self.output_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self
