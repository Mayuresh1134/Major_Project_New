"""The web app: every tool end to end, uploads, files and the library."""
import io

from conftest import tune_notes, wait_for_job, wav_bytes


def test_pages_load(client):
    for url in ["/", "/analyze", "/audio-to-sheet", "/sheet-to-audio", "/compose", "/style-transfer",
                "/practice", "/library"]:
        assert client.get(url).status_code == 200, url
    assert client.get("/results/0123456789abcdef").status_code == 404


def test_analyze_flow(client):
    r = client.post("/api/analyze", data={"audio": (wav_bytes(tune_notes("minuet")), "minuet.wav")})
    job, body = wait_for_job(client, r)
    assert job["status"] == "done", job
    page = client.get(body["result_url"])
    assert page.status_code == 200 and b"G major" in page.data


def test_transcribe_then_render_then_compose(client):
    r = client.post("/api/transcribe", data={"file": (wav_bytes(tune_notes("ode")), "ode.wav"),
                                             "melody_only": "on", "grid": "4", "meter": "4/4"})
    job, body = wait_for_job(client, r)
    assert job["status"] == "done", job
    rid = body["job"]
    result = client.get(f"/results/{rid}")
    assert b"Sheet music" in result.data and b"D major" in result.data
    job, _ = wait_for_job(client, client.post("/api/render", data={"source": "result", "result_id": rid, "voice": "flute"}))
    assert job["status"] == "done", job
    job, body = wait_for_job(client, client.post("/api/compose", data={
        "source": "result", "result_id": rid, "length": "16", "stay_in_key": "on", "voice": "piano"}))
    assert job["status"] == "done", job
    assert client.get(body["result_url"]).status_code == 200


def test_render_from_typed_abc_and_example(client):
    job, body = wait_for_job(client, client.post("/api/render", data={
        "source": "abc", "abc": "T:Scale\nK:C\nC D E F | G A B c |", "voice": "organ", "transpose": "-3", "tempo_pct": "80"}))
    assert job["status"] == "done", job
    page = client.get(body["result_url"]).data
    assert b"Scale" in page and b"80% speed" in page
    job, _ = wait_for_job(client, client.post("/api/render", data={"source": "example", "example": "greensleeves", "voice": "auto"}))
    assert job["status"] == "done"


def test_bad_abc_reports_a_message(client):
    job, _ = wait_for_job(client, client.post("/api/render", data={"source": "abc", "abc": ""}))
    assert job["status"] == "error" and "ABC" in job["error"]


def test_style_flow(client):
    job, body = wait_for_job(client, client.post("/api/style", data={
        "content": (wav_bytes(tune_notes("minuet")), "a.wav"),
        "style": (wav_bytes(tune_notes("twinkle"), "flute"), "b.wav"),
        "seconds": "3", "quality": "quick", "strength": "80"}))
    assert job["status"] == "done", job
    assert b"Texture matched" in client.get(body["result_url"]).data


def test_upload_type_is_checked(client):
    r = client.post("/api/analyze", data={"audio": (io.BytesIO(b"MZ..."), "../../evil.exe")})
    assert r.status_code == 400 and "not a supported file" in r.get_json()["error"]
    r = client.post("/api/analyze", data={})
    assert r.status_code == 400


def test_out_of_range_options_are_rejected(client):
    r = client.post("/api/compose", data={"source": "example", "example": "ode", "length": "100000"})
    assert r.status_code == 400 and "from 8 to 256" in r.get_json()["error"]
    r = client.post("/api/render", data={"source": "example", "example": "nope"})
    assert r.status_code == 400


def test_files_are_served_safely(client, app):
    job, body = wait_for_job(client, client.post("/api/render", data={"source": "example", "example": "twinkle"}))
    rid = body["job"]
    res = app.extensions["muse_storage"].load(rid)
    f = client.get(f"/files/{rid}/{res['files']['sheet_1']}")
    assert f.status_code == 200 and "default-src 'none'" in f.headers["Content-Security-Policy"]
    assert client.get(f"/files/{rid}/result.json").status_code == 404           # not a listed output
    assert client.get(f"/files/{rid}/..%2F..%2Fapp.py").status_code == 404
    assert client.get("/files/not-an-id/x.wav").status_code == 404
    d = client.get(f"/files/{rid}/{res['files']['midi']}?download=1")
    assert "attachment" in d.headers["Content-Disposition"] and "Twinkle" in d.headers["Content-Disposition"]


def test_library_and_delete(client):
    job, body = wait_for_job(client, client.post("/api/render", data={"source": "example", "example": "ode"}))
    rid = body["job"]
    assert b"Ode to Joy" in client.get("/library").data
    assert client.post(f"/api/results/{rid}/delete").status_code == 200
    assert client.get(f"/results/{rid}").status_code == 404
    assert b"Ode to Joy" not in client.get("/library").data


def test_reuse_audio_from_an_earlier_result(client):
    job, body = wait_for_job(client, client.post("/api/analyze", data={"audio": (wav_bytes(tune_notes("twinkle")), "t.wav")}))
    job, _ = wait_for_job(client, client.post("/api/transcribe", data={"from_result": body["job"], "melody_only": "on"}))
    assert job["status"] == "done", job
