"""Results on disk: each piece of work gets a folder app_data/results/<id>/ with result.json."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "oga", "aif", "aiff"}
SCORE_EXTENSIONS = {"musicxml", "xml", "mxl", "mid", "midi", "abc"}
ID_RE = re.compile(r"^[0-9a-f]{16}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")

_lock = threading.Lock()


class Storage:
    def __init__(self, root: str | os.PathLike, max_results: int = 200):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results

    # ids and paths -------------------------------------------------------
    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:16]

    def folder(self, rid: str) -> Path:
        if not ID_RE.match(rid or ""):
            raise KeyError(rid)
        return self.root / rid

    def file_path(self, rid: str, name: str) -> Path:
        """Path of a file that belongs to a finished result, or KeyError."""
        result = self.load(rid)
        if name not in result.get("files", {}).values() or not NAME_RE.match(name):
            raise KeyError(name)
        return self.folder(rid) / name

    # uploads ---------------------------------------------------------------
    @staticmethod
    def extension(filename: str) -> str:
        return filename.rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""

    def save_upload(self, rid: str, upload, stem: str, allowed: set[str]) -> tuple[Path, str]:
        """Save a werkzeug FileStorage under a fixed name. Returns (path, original filename)."""
        original = os.path.basename(upload.filename or "")
        ext = self.extension(original)
        if ext not in allowed:
            raise ValueError(f"“{original or 'This file'}” is not a supported file. "
                             f"Use {', '.join('.' + e for e in sorted(allowed))}.")
        folder = self.folder(rid)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stem}.{ext}"
        upload.save(path)
        if path.stat().st_size == 0:
            raise ValueError(f"“{original}” is empty.")
        return path, original

    # results -------------------------------------------------------------
    def save(self, rid: str, result: dict) -> dict:
        folder = self.folder(rid)
        folder.mkdir(parents=True, exist_ok=True)
        result = {"id": rid, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result}
        tmp = folder / "result.json.tmp"
        tmp.write_text(json.dumps(result), encoding="utf-8")
        tmp.replace(folder / "result.json")
        self.prune()
        return result

    def load(self, rid: str) -> dict:
        path = self.folder(rid) / "result.json"
        if not path.exists():
            raise KeyError(rid)
        return json.loads(path.read_text(encoding="utf-8"))

    def exists(self, rid: str) -> bool:
        try:
            return (self.folder(rid) / "result.json").exists()
        except KeyError:
            return False

    def list(self, kind: str | None = None, limit: int | None = None) -> list[dict]:
        items = []
        for f in self.root.iterdir():
            p = f / "result.json"
            if not p.exists():
                continue
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if kind and r.get("kind") != kind:
                continue
            items.append({k: r.get(k) for k in ("id", "kind", "title", "created", "summary")})
        items.sort(key=lambda r: r.get("created") or "", reverse=True)
        return items[:limit] if limit else items

    def delete(self, rid: str) -> None:
        folder = self.folder(rid)
        if folder.exists():
            shutil.rmtree(folder)

    def prune(self) -> None:
        """Keep at most max_results results; remove abandoned folders older than a day."""
        with _lock:
            items = self.list()
            for r in items[self.max_results:]:
                self.delete(r["id"])
            cutoff = time.time() - 86400
            for f in self.root.iterdir():
                if f.is_dir() and not (f / "result.json").exists() and f.stat().st_mtime < cutoff:
                    shutil.rmtree(f, ignore_errors=True)
