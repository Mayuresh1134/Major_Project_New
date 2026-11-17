"""MuseAssist: a music study and creation app (Flask)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask

from .jobs import Jobs
from .storage import Storage

ROOT = Path(__file__).resolve().parent.parent


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.config.update(
        DATA_DIR=os.environ.get("MUSE_DATA_DIR", str(ROOT / "app_data")),
        MAX_CONTENT_LENGTH=int(os.environ.get("MUSE_MAX_UPLOAD_MB", "60")) * 1024 * 1024,
        MAX_RESULTS=int(os.environ.get("MUSE_MAX_RESULTS", "200")),
        JOB_WORKERS=2,
    )
    if config:
        app.config.update(config)
    app.extensions["muse_storage"] = Storage(Path(app.config["DATA_DIR"]) / "results", app.config["MAX_RESULTS"])
    app.extensions["muse_jobs"] = Jobs(app.config["JOB_WORKERS"])
    from .web import bp
    app.register_blueprint(bp)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return app
