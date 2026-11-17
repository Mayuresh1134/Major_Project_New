import io
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from museassist import create_app, notation, synth  # noqa: E402


@pytest.fixture()
def app(tmp_path):
    return create_app({"DATA_DIR": str(tmp_path / "data")})


@pytest.fixture()
def client(app):
    return app.test_client()


def tune_notes(key="minuet"):
    return notation.score_to_notes(notation.parse_abc(notation.EXAMPLES[key][1]))


def wav_bytes(notes, voice="piano") -> io.BytesIO:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, synth.render_notes(notes, voice), synth.SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


def wait_for_job(client, response, timeout=240):
    assert response.status_code == 202, response.get_json()
    body = response.get_json()
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(body["status_url"]).get_json()
        if job["status"] in ("done", "error"):
            return job, body
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def note_f1(ref, est, tol=0.05):
    """Onset-only note F1: same pitch, start within tol seconds, one-to-one."""
    used = set()
    hits = 0
    for r in ref:
        for i, e in enumerate(est):
            if i not in used and e["pitch"] == r["pitch"] and abs(e["start"] - r["start"]) <= tol:
                used.add(i)
                hits += 1
                break
    p = hits / max(len(est), 1)
    r = hits / max(len(ref), 1)
    return 0.0 if hits == 0 else 2 * p * r / (p + r)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
