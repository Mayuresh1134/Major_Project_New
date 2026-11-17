"""Run slow work in a background thread and report progress to the page."""
from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)


class UserError(Exception):
    """A problem with the user's input, shown to them as is."""


class Jobs:
    def __init__(self, workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, job_id: str, kind: str, fn, *args, **kwargs) -> dict:
        job = {"id": job_id, "kind": kind, "status": "queued", "progress": 0.0,
               "stage": "Waiting to start", "error": None, "started": time.time()}
        with self._lock:
            self._jobs[job_id] = job
            self._forget_old()

        def report(fraction=None, stage=None):
            with self._lock:
                if fraction is not None:
                    job["progress"] = round(max(job["progress"], min(float(fraction), 1.0)), 3)
                if stage:
                    job["stage"] = stage

        def run():
            job["status"] = "running"
            try:
                fn(*args, report=report, **kwargs)
                job.update(status="done", progress=1.0, stage="Done")
            except UserError as e:
                job.update(status="error", error=str(e))
            except Exception as e:  # keep the server alive, tell the user what failed
                log.error("Job %s failed:\n%s", job_id, traceback.format_exc())
                job.update(status="error", error=f"Something went wrong while processing ({type(e).__name__}: {e}).")

        self._pool.submit(run)
        return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def _forget_old(self):
        cutoff = time.time() - 3600
        for k in [k for k, j in self._jobs.items() if j["status"] in ("done", "error") and j["started"] < cutoff]:
            del self._jobs[k]
