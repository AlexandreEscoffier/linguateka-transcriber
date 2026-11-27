"""Journal JSONL des extractions effectuées par le service."""
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit

try:
    YT_DLP_VERSION = version("yt-dlp")
except PackageNotFoundError:
    YT_DLP_VERSION = "unknown"

_lock = threading.Lock()
_active = 0


def _emit(event, **fields):
    now = datetime.now(timezone.utc)
    record = dict(event=event, timestamp=now.isoformat(), pid=os.getpid(),
                  replica=os.getenv("RAILWAY_REPLICA_ID"), yt_dlp=YT_DLP_VERSION,
                  active_in_worker=_active, **fields)
    line = json.dumps(record, separators=(",", ":"))
    print(line, flush=True)
    try:
        directory = Path(os.getenv("YT_DLP_LOG_DIR") or tempfile.gettempdir())
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"yt-dlp-{now:%Y-%m-%d}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    except OSError as exc:
        print(f"yt-dlp journal: file write failed (errno={exc.errno})", flush=True)


def run_logged(cmd, operation, url):
    global _active
    try:
        parsed = urlsplit(url)
        match = re.fullmatch(r"/(?:p|reels?)/([\w-]+)/?", parsed.path, re.ASCII)
        media_id = match[1] if match and parsed.hostname in {
            "instagram.com", "www.instagram.com", "m.instagram.com"
        } else None
    except ValueError:
        media_id = None
    fields = dict(call_id=uuid.uuid4().hex, operation=operation, media_id=media_id)
    start = time.monotonic()
    outcome = {"result": "exception"}
    with _lock:
        _active += 1
        _emit("yt_dlp_start", **fields)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        statuses = sorted(set(re.findall(r"HTTP(?:\s+Error)?\s+(\d{3})\b", result.stderr, re.I)))
        outcome = dict(result="error" if result.returncode else "ok",
                       exit_code=result.returncode, http_codes=statuses)
        return result
    except Exception as exc:
        outcome["exception_type"] = type(exc).__name__
        raise
    finally:
        with _lock:
            _active -= 1
            _emit("yt_dlp_finish", **fields, **outcome,
                  duration_ms=round((time.monotonic() - start) * 1000))
