"""Pages and API routes."""
from __future__ import annotations

from pathlib import Path

from flask import (Blueprint, abort, current_app, jsonify, render_template, request, send_file,
                   url_for)
from werkzeug.exceptions import RequestEntityTooLarge

from . import genre, notation, pipelines, synth
from .jobs import UserError
from .storage import AUDIO_EXTENSIONS, SCORE_EXTENSIONS

bp = Blueprint("main", __name__)

TOOLS = [
    # endpoint, menu label, what goes in, what comes out
    ("main.analyze", "Analyze", "a recording", "tempo, key, chords, genre and a spectrogram"),
    ("main.transcribe_page", "Audio to sheet", "a recording or MIDI file", "sheet music, MIDI and MusicXML"),
    ("main.play", "Sheet to audio", "MusicXML, MIDI or ABC notation", "an engraved score you can hear in 9 instruments"),
    ("main.compose", "Compose", "a melody to start from", "a continuation written by the melody model"),
    ("main.style", "Style transfer", "two recordings", "the first, with the texture of the second"),
    ("main.practice", "Practice", "your ears and hands", "piano, chords, scales, ear training, tuner and metronome"),
]

KIND_LABELS = {"analyze": "Analysis", "transcribe": "Audio to sheet", "render": "Sheet to audio",
               "compose": "Composition", "style": "Style transfer"}


def storage():
    return current_app.extensions["muse_storage"]


def jobs():
    return current_app.extensions["muse_jobs"]


@bp.app_context_processor
def inject_globals():
    return {"tools": TOOLS, "kind_labels": KIND_LABELS, "voices": synth.VOICES}


# Form helpers ----------------------------------------------------------------
def number(name, default, lo, hi, cast=float):
    raw = request.form.get(name, "").strip()
    if raw == "":
        return cast(default)
    try:
        value = cast(float(raw))
    except ValueError:
        raise UserError(f"“{raw}” is not a number.")
    if not lo <= value <= hi:
        raise UserError(f"Choose a value from {lo} to {hi} for {name.replace('_', ' ')}.")
    return value


def checkbox(name):
    return request.form.get(name) in ("on", "true", "1", "yes")


def choice(name, options, default):
    value = request.form.get(name, default)
    if value not in options:
        raise UserError(f"Unknown {name}: {value}")
    return value


def existing_result(field="from_result"):
    rid = request.form.get(field, "").strip()
    if not rid:
        return None
    if not storage().exists(rid):
        raise UserError("That earlier result no longer exists.")
    return rid


def start(kind, fn, rid, *args):
    storage().folder(rid).mkdir(parents=True, exist_ok=True)
    job = jobs().submit(rid, kind, fn, storage(), rid, *args)
    return jsonify({"job": job["id"], "status_url": url_for("main.job_status", job_id=job["id"]),
                    "result_url": url_for("main.result", rid=rid)}), 202


@bp.errorhandler(UserError)
def user_error(e):
    return jsonify({"error": str(e)}), 400


@bp.app_errorhandler(RequestEntityTooLarge)
def too_large(e):
    mb = current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    if request.path.startswith("/api/"):
        return jsonify({"error": f"The file is too large. The limit is {mb} MB."}), 413
    return render_template("error.html", message=f"The file is too large. The limit is {mb} MB."), 413


@bp.app_errorhandler(404)
def not_found(e):
    if request.path.startswith(("/api/", "/files/")):
        return jsonify({"error": "Not found."}), 404
    return render_template("error.html", message="This page doesn't exist, or the result was deleted."), 404


def audio_upload(rid, field="audio"):
    """The uploaded audio file, or the audio of an earlier result."""
    from_id = existing_result()
    if from_id:
        return pipelines.copy_result_audio(storage(), from_id, rid)
    f = request.files.get(field)
    if not f or not f.filename:
        raise UserError("Choose an audio file first.")
    try:
        return storage().save_upload(rid, f, "input", AUDIO_EXTENSIONS)
    except ValueError as e:
        raise UserError(str(e))


# Pages -------------------------------------------------------------------------
@bp.get("/")
def home():
    return render_template("home.html", recent=storage().list(limit=5))


@bp.get("/analyze")
def analyze():
    return render_template("analyze.html", recent=storage().list("analyze", 4), report=genre.model_report())


@bp.get("/audio-to-sheet")
def transcribe_page():
    return render_template("transcribe.html", recent=storage().list("transcribe", 4))


@bp.get("/sheet-to-audio")
def play():
    return render_template("play.html", examples=notation.EXAMPLES, recent=storage().list("render", 4),
                           scores=[r for r in storage().list() if r["kind"] in ("transcribe", "compose")][:8],
                           fluidsynth=synth.fluidsynth_available())


@bp.get("/compose")
def compose():
    sources = [r for r in storage().list() if r["kind"] in ("transcribe", "render", "compose")][:10]
    return render_template("compose.html", examples=notation.EXAMPLES, sources=sources,
                           recent=storage().list("compose", 4))


@bp.get("/style-transfer")
def style():
    return render_template("style.html", recent=storage().list("style", 4))


@bp.get("/practice")
def practice():
    return render_template("practice.html")


@bp.get("/library")
def library():
    return render_template("library.html", items=storage().list())


@bp.get("/results/<rid>")
def result(rid):
    try:
        r = storage().load(rid)
    except KeyError:
        job = jobs().get(rid)
        if job and job["status"] in ("queued", "running"):
            return render_template("waiting.html", job=job)
        if job and job["status"] == "error":
            return render_template("error.html", message=job["error"]), 400
        abort(404)
    files = {k: url_for("main.file", rid=rid, name=v) for k, v in r.get("files", {}).items()}
    report = genre.model_report() if r["kind"] == "analyze" else None
    return render_template(f"result_{r['kind']}.html", r=r, d=r.get("data", {}), files=files, report=report)


@bp.get("/results/<rid>/print")
def print_sheet(rid):
    try:
        r = storage().load(rid)
    except KeyError:
        abort(404)
    pages = [url_for("main.file", rid=rid, name=r["files"][f"sheet_{i}"])
             for i in range(1, r.get("data", {}).get("pages", 0) + 1) if f"sheet_{i}" in r["files"]]
    return render_template("print.html", r=r, pages=pages)


@bp.get("/files/<rid>/<name>")
def file(rid, name):
    try:
        path = storage().file_path(rid, name)
    except KeyError:
        abort(404)
    download = request.args.get("download") == "1"
    nice = None
    if download:
        title = storage().load(rid).get("title") or "musetrack"
        safe = "".join(c if c.isalnum() or c in " -_()" else "_" for c in title).strip() or "musetrack"
        nice = f"{safe}{Path(name).suffix}"
    resp = send_file(path, as_attachment=download, download_name=nice, max_age=3600)
    # results are data, never pages: no scripts even if an SVG were opened directly
    resp.headers["Content-Security-Policy"] = "default-src 'none'; style-src 'unsafe-inline'; font-src data:; img-src data:"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


# API -----------------------------------------------------------------------------
@bp.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = jobs().get(job_id)
    if job is None:
        if storage().exists(job_id):
            return jsonify({"id": job_id, "status": "done", "progress": 1.0, "stage": "Done"})
        abort(404)
    return jsonify({k: job[k] for k in ("id", "status", "progress", "stage", "error")})


@bp.post("/api/analyze")
def api_analyze():
    rid = storage().new_id()
    path, name = audio_upload(rid)
    return start("analyze", pipelines.run_analyze, rid, path, name)


@bp.post("/api/transcribe")
def api_transcribe():
    rid = storage().new_id()
    options = {
        "sensitivity": number("sensitivity", 0.5, 0, 1),
        "melody_only": checkbox("melody_only"),
        "grid": int(choice("grid", {"2", "4", "3"}, "4")),
        "meter": choice("meter", {"4/4", "3/4", "2/4", "6/8"}, "4/4"),
        "tempo": number("tempo", 0, 0, 240) or None,
    }
    rng = choice("range", {"all", "low", "mid", "high"}, "all")
    options["min_pitch"], options["max_pitch"] = {"all": (None, None), "low": (21, 60),
                                                  "mid": (48, 84), "high": (60, 108)}[rng]
    from_id = existing_result()
    f = request.files.get("file")
    if from_id:
        path, name = pipelines.copy_result_audio(storage(), from_id, rid)
    elif f and f.filename:
        try:
            path, name = storage().save_upload(rid, f, "input", AUDIO_EXTENSIONS | {"mid", "midi"})
        except ValueError as e:
            raise UserError(str(e))
    else:
        raise UserError("Choose a recording or a MIDI file first.")
    return start("transcribe", pipelines.run_transcribe, rid, path, name, options)


def score_source(rid, allow_result=True):
    kind = request.form.get("source", "file")
    if kind == "file":
        f = request.files.get("score")
        if not f or not f.filename:
            raise UserError("Choose a score file first.")
        try:
            path, name = storage().save_upload(rid, f, "input", SCORE_EXTENSIONS)
        except ValueError as e:
            raise UserError(str(e))
        return {"type": "file", "path": path, "name": Path(name).stem}
    if kind == "abc":
        text = request.form.get("abc", "")
        if len(text) > 50_000:
            raise UserError("The ABC text is too long (50,000 characters at most).")
        return {"type": "abc", "text": text}
    if kind == "example":
        key = choice("example", set(notation.EXAMPLES), "ode")
        return {"type": "example", "key": key, "name": notation.EXAMPLES[key][0]}
    if kind == "result" and allow_result:
        rid_from = existing_result("result_id")
        if not rid_from:
            raise UserError("Choose one of your earlier results.")
        return {"type": "result", "id": rid_from}
    raise UserError("Choose where the music comes from.")


@bp.post("/api/render")
def api_render():
    rid = storage().new_id()
    options = {
        "voice": choice("voice", set(synth.VOICES), "piano"),
        "tempo_pct": number("tempo_pct", 100, 25, 300, int),
        "transpose": number("transpose", 0, -12, 12, int),
    }
    return start("render", pipelines.run_render, rid, score_source(rid), options)


@bp.post("/api/compose")
def api_compose():
    rid = storage().new_id()
    seed = number("seed", 0, 0, 1_000_000, int)
    options = {
        "length": number("length", 48, 8, 256, int),
        "temperature": number("temperature", 1.0, 0.2, 2.0),
        "stay_in_key": checkbox("stay_in_key"),
        "smooth": checkbox("smooth"),
        "snap": checkbox("snap"),
        "voice": choice("voice", set(synth.VOICES) - {"auto"}, "piano"),
        "seed": seed or None,
    }
    return start("compose", pipelines.run_compose, rid, score_source(rid), options)


@bp.post("/api/style")
def api_style():
    rid = storage().new_id()
    quality = choice("quality", {"quick", "standard", "thorough"}, "standard")
    options = {
        "seconds": number("seconds", 15, 3, 30, int),
        "start": number("start", 0, 0, 600),
        "strength": number("strength", 70, 0, 100) / 100,
        "steps": {"quick": 60, "standard": 150, "thorough": 300}[quality],
        "quality": quality,
    }
    content_path, content_name = audio_upload(rid, "content")
    f = request.files.get("style")
    if not f or not f.filename:
        raise UserError("Choose the style clip too.")
    try:
        style_path, style_name = storage().save_upload(rid, f, "style", AUDIO_EXTENSIONS)
    except ValueError as e:
        raise UserError(str(e))
    return start("style", pipelines.run_style, rid, content_path, style_path, (content_name, style_name), options)


@bp.post("/api/results/<rid>/delete")
def api_delete(rid):
    try:
        storage().folder(rid)
    except KeyError:
        abort(404)
    storage().delete(rid)
    return jsonify({"deleted": rid})
