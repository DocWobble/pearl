"""Small append-only event logger for the experimental SM120 run contract."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def job_id(header: bytes) -> str:
    return hashlib.sha256(header).hexdigest()


class Sm120EventLogger:
    def __init__(self) -> None:
        configured = os.environ.get("PEARL_SM120_EVENT_LOG")
        self._path = Path(configured) if configured else None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def emit(self, event: str, generation: int, job: bytes | None, **fields: Any) -> None:
        if self._path is None:
            return
        record: dict[str, Any] = {
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "wallclock_utc": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
            "job_id": job_id(job) if job is not None else "",
            **fields,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
