"""The work behind each tool. Every function saves its output with Storage and reports progress."""
from __future__ import annotations

import shutil
from pathlib import Path

import librosa
import numpy as np
from matplotlib import pyplot as plt

from . import analysis, generate, genre, notation, synth, style_transfer, transcribe
from .jobs import UserError
from .theory import estimate_key, midi_to_name, note_pitch_class_weights

BROWSER_AUDIO = {"wav", "mp3", "ogg", "oga", "flac"}
TRANSCRIBE_MAX_SECONDS = 300
SHEET_MAX_SECONDS = 180


def _load_audio(path, max_seconds):
    try:
        return analysis.load_audio(path, max_seconds)
    except Exception as e:
        raise UserError("This file could not be read as audio. Try a WAV or MP3 file.") from e


def _playable_copy(folder: Path, source: Path, y, name="original") -> str:
    """A file the browser can play: the upload itself, or a WAV of the decoded audio."""
    ext = source.suffix.lstrip(".").lower()
    if ext in BROWSER_AUDIO:
        return source.name
    synth.write_wav(folder / f"{name}.wav", y, analysis.SR)
    return f"{name}.wav"


def _sheet(folder: Path, score, files: dict, report=None) -> int:
    """MusicXML + engraved SVG pages for a score. Returns the number of pages."""
    if report:
        report(stage="Engraving the sheet music")
    notation.write_musicxml(score, folder / "score.musicxml")
    files["musicxml"] = "score.musicxml"
    pages = notation.engrave(folder / "score.musicxml")
    for i, svg in enumerate(pages, start=1):
        (folder / f"sheet-{i}.svg").write_text(svg, encoding="utf-8")
        files[f"sheet_{i}"] = f"sheet-{i}.svg"
    return len(pages)


def _note_stats(notes) -> dict:
    if not notes:
        return {"count": 0}
    pitches = [n["pitch"] for n in notes]
    return {"count": len(notes), "lowest": midi_to_name(min(pitches)), "highest": midi_to_name(max(pitches)),
            "seconds": round(max(n["end"] for n in notes), 1)}


def _compact(notes):
    """Notes as short lists for the page: [pitch, start, end, velocity, part]."""
    return [[int(n["pitch"]), round(float(n["start"]), 3), round(float(n["end"]), 3),
             int(n.get("velocity", 90)), int(n.get("part", 0))] for n in notes]


# Analyze ---------------------------------------------------------------------
def run_analyze(storage, rid, audio_path: Path, original_name: str, report):
    folder = storage.folder(rid)
    report(0.02, "Reading the audio")
    y, info = _load_audio(audio_path, analysis.MAX_SECONDS)
    report(0.08, "Finding the beat, key and chords")
    data = analysis.analyse(y, progress=lambda f: report(0.08 + 0.5 * f))
    report(0.6, "Recognising the genre")
    try:
        data["genre"] = genre.classify(y, progress=lambda f: report(0.6 + 0.3 * f))
    except ValueError as e:
        data["genre"] = {"error": str(e)}
    report(0.92, "Drawing the spectrogram")
    analysis.save_spectrogram(y, folder / "spectrogram.png")
    files = {"audio": _playable_copy(folder, audio_path, y), "spectrogram": "spectrogram.png"}
    data["info"] = info
    top = data["genre"].get("top")
    summary = f"{data['tempo']['bpm']:.0f} BPM, {data['key']['name']}" + (f", {top}" if top else "")
    storage.save(rid, {"kind": "analyze", "title": Path(original_name).stem or original_name, "summary": summary,
                       "files": files, "data": data, "source": {"audio": audio_path.name}})


# Audio to sheet --------------------------------------------------------------
def run_transcribe(storage, rid, input_path: Path, original_name: str, options: dict, report):
    folder = storage.folder(rid)
    files = {}
    ext = input_path.suffix.lstrip(".").lower()
    title = Path(original_name).stem or "Transcription"
    if ext in ("mid", "midi"):
        report(0.1, "Reading the MIDI file")
        try:
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(str(input_path))
        except Exception as e:
            raise UserError("This MIDI file could not be read.") from e
        notes = [n for n in synth.notes_from_midi(pm) if not n["is_drum"]]
        tempo = float(pm.get_tempo_changes()[1][0]) if len(pm.get_tempo_changes()[1]) else 120.0
        files["input"] = input_path.name
        audio_seconds = None
    else:
        report(0.03, "Reading the audio")
        y, info = _load_audio(input_path, TRANSCRIBE_MAX_SECONDS)
        files["audio"] = _playable_copy(folder, input_path, y)
        report(0.08, "Listening for notes")
        notes = transcribe.transcribe(
            y, sensitivity=options["sensitivity"], melody_only=options["melody_only"],
            min_pitch=options.get("min_pitch"), max_pitch=options.get("max_pitch"),
            progress=lambda f: report(0.08 + 0.5 * f))
        report(0.6, "Finding the tempo")
        tempo = analysis.tempo_and_beats(y)["bpm"] or 120.0
        audio_seconds = info["analysed_seconds"]
    if not notes:
        raise UserError("No notes were found. Try a recording with a clear instrument or voice, "
                        "or raise the sensitivity.")
    sheet_offset = 0.0
    if options.get("tempo"):
        tempo = float(options["tempo"])
    elif audio_seconds is not None:
        report(0.62, "Fitting the beat grid")
        tempo, sheet_offset = notation.fit_beat_grid(notes, tempo)
    tempo = float(np.clip(tempo, 40, 220))
    synth_notes = [{**n, "velocity": n.get("velocity", 90)} for n in notes]
    notation.notes_to_midi(synth_notes, folder / "notes.mid", tempo=tempo)
    files["midi"] = "notes.mid"
    report(0.68, "Playing back the notes")
    synth.write_wav(folder / "notes.wav", synth.render_notes(synth_notes, "piano"))
    files["notes_audio"] = "notes.wav"
    # the sheet starts at the grid line nearest the first note
    sheet_notes = [{**n, "start": max(n["start"] - sheet_offset, 0.0), "end": max(n["end"] - sheet_offset, 0.02)}
                   for n in notes if n["start"] < SHEET_MAX_SECONDS]
    sheet_notes = notation.tidy_for_sheet(sheet_notes, tempo, options["grid"])
    report(0.75, "Writing the score")
    key = estimate_key(note_pitch_class_weights(notes))
    split = not options["melody_only"] and min(n["pitch"] for n in sheet_notes) < 57
    score = notation.notes_to_score(sheet_notes, tempo, title, grid=options["grid"], split_hands=split,
                                    key=key, time_signature=options["meter"])
    pages = _sheet(folder, score, files, report)
    data = {"notes": _compact(notes), "tempo": round(tempo, 1), "key": key, "pages": pages,
            "stats": _note_stats(notes), "audio_seconds": audio_seconds,
            "sheet_truncated": len(sheet_notes) < len(notes), "options": options}
    storage.save(rid, {"kind": "transcribe", "title": title, "files": files, "data": data,
                       "summary": f"{len(notes)} notes, {key['name']}",
                       "source": {"input": input_path.name}})


# Sheet to audio --------------------------------------------------------------
def load_score_source(storage, source: dict):
    """A music21 score from an upload, typed ABC, a built-in example or an earlier result."""
    try:
        if source["type"] == "file":
            return notation.load_score(source["path"])
        if source["type"] == "abc":
            return notation.parse_abc(source["text"])
        if source["type"] == "example":
            title, abc = notation.EXAMPLES[source["key"]]
            return notation.parse_abc(abc)
        if source["type"] == "result":
            r = storage.load(source["id"])
            if "musicxml" in r.get("files", {}):
                return notation.load_score(storage.folder(source["id"]) / r["files"]["musicxml"])
            raise UserError("That result has no score to use.")
    except notation.ScoreError as e:
        raise UserError(str(e)) from e
    raise UserError("Choose a score first.")


def run_render(storage, rid, source: dict, options: dict, report):
    folder = storage.folder(rid)
    files = {}
    report(0.05, "Reading the score")
    score = load_score_source(storage, source)
    if options["transpose"]:
        report(0.15, "Transposing")
        score = notation.transpose_score(score, options["transpose"])
    fallback = source.get("name") or ("Your ABC tune" if source["type"] == "abc" else "Untitled score")
    title = notation.score_title(score, fallback)
    pages = _sheet(folder, score, files, report)
    report(0.5, "Playing the score")
    notes = notation.score_to_notes(score)
    if not notes:
        raise UserError("The score has no notes to play.")
    speed = options["tempo_pct"] / 100.0
    for n in notes:
        n["start"] = round(n["start"] / speed, 4)
        n["end"] = round(n["end"] / speed, 4)
    tempo = notation.score_tempo(score) * speed
    voice = options["voice"]
    program = synth.VOICES[voice][1]
    if program is None:  # "as written": keep the score's own instrument
        program = notes[0].get("program", 0)
    notation.notes_to_midi(notes, folder / "score.mid", program=program, tempo=tempo)
    files["midi"] = "score.mid"
    report(0.65)
    if voice == "auto":
        synth.write_wav(folder / "audio.wav", synth.render_notes(notes, "auto"))
        engine = "built-in"
    else:
        engine = synth.render_midi_file(folder / "score.mid", str(folder / "audio.wav"), voice)
    files["audio"] = "audio.wav"
    key = estimate_key(note_pitch_class_weights(notes))
    data = {"notes": _compact(notes), "tempo": round(tempo, 1), "key": key, "pages": pages,
            "stats": _note_stats(notes), "options": options, "engine": engine}
    storage.save(rid, {"kind": "render", "title": title, "files": files, "data": data,
                       "summary": f"{synth.VOICES[options['voice']][0]}, {round(tempo)} BPM",
                       "source": {k: v for k, v in source.items() if k in ("type", "key", "id", "name")}})


# Compose ---------------------------------------------------------------------
def seed_from_source(storage, source: dict) -> tuple[list[dict], float, str, str]:
    """Seed notes (melody line), tempo, title and time signature."""
    meter = "4/4"
    if source["type"] == "result":
        r = storage.load(source["id"])
        raw = r.get("data", {}).get("notes")
        if not raw:
            raise UserError("That result has no notes to continue.")
        notes = [{"pitch": p, "start": s, "end": e, "velocity": v} for p, s, e, v, *_ in raw]
        tempo = float(r["data"].get("tempo") or 120)
        title = r.get("title") or "Melody"
        meter = (r["data"].get("options") or {}).get("meter") or r["data"].get("meter") or "4/4"
    elif source["type"] == "file" and source["path"].suffix.lower() in (".mid", ".midi"):
        import pretty_midi
        try:
            pm = pretty_midi.PrettyMIDI(str(source["path"]))
        except Exception as e:
            raise UserError("This MIDI file could not be read.") from e
        notes = [n for n in synth.notes_from_midi(pm) if not n["is_drum"]]
        changes = pm.get_tempo_changes()[1]
        tempo = float(changes[0]) if len(changes) else 120.0
        title = source.get("name") or "Melody"
    else:
        score = load_score_source(storage, source)
        notes = notation.score_to_notes(score)
        tempo = notation.score_tempo(score)
        title = notation.score_title(score, source.get("name") or "Melody")
        meter = notation.score_meter(score)
    if not notes:
        raise UserError("The seed has no notes.")
    melody = transcribe.keep_top_line(notes)
    melody = [n for n in melody if n["start"] < 120]  # the first two minutes are plenty of context
    return melody, tempo, title, meter


def snap(notes, tempo: float, start_at: float, division: int = 4):
    """Move note starts and ends onto a grid of `division` steps per beat."""
    step = 60.0 / tempo / division
    out = []
    for n in notes:
        s = start_at + round((n["start"] - start_at) / step) * step
        e = start_at + max(round((n["end"] - start_at) / step) * step, s - start_at + step)
        out.append({**n, "start": round(s, 4), "end": round(e, 4)})
    # drop notes that landed on the same start, keep the first
    seen, result = set(), []
    for n in out:
        if n["start"] in seen:
            continue
        seen.add(n["start"])
        result.append(n)
    for a, b in zip(result, result[1:]):
        a["end"] = min(a["end"], b["start"])
    return result


def run_compose(storage, rid, source: dict, options: dict, report):
    folder = storage.folder(rid)
    files = {}
    report(0.05, "Reading your melody")
    seed, tempo, title, meter = seed_from_source(storage, source)
    report(0.2, "Writing the continuation")
    out = generate.continue_melody(seed, n_notes=options["length"], temperature=options["temperature"],
                                   stay_in_key=options["stay_in_key"], max_leap=9 if options["smooth"] else None,
                                   rng_seed=options.get("seed"))
    new = out["notes"]
    if options["snap"]:
        beat = 60.0 / tempo
        start_at = seed[0]["start"]
        first = start_at + np.ceil((new[0]["start"] - start_at) / (beat / 4)) * (beat / 4)
        shift = first - new[0]["start"]
        new = snap([{**n, "start": n["start"] + shift, "end": n["end"] + shift} for n in new], tempo, start_at)
    seed_end = max(n["end"] for n in seed)
    if new and new[0]["start"] < seed_end - 0.01:  # never overlap the seed's last note
        shift = seed_end - new[0]["start"]
        new = [{**n, "start": round(n["start"] + shift, 4), "end": round(n["end"] + shift, 4)} for n in new]
    combined = [{**n, "part": 0} for n in seed] + [{**n, "part": 1} for n in new]
    voice = options["voice"]
    program = synth.VOICES[voice][1] or 0
    report(0.5, "Saving the MIDI file")
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for name, part in (("Your melody", seed), ("Continuation", new)):
        inst = pretty_midi.Instrument(program=program, name=name)
        for n in part:
            inst.notes.append(pretty_midi.Note(int(n.get("velocity", 90)), int(n["pitch"]),
                                               float(n["start"]), float(n["end"])))
        pm.instruments.append(inst)
    pm.write(str(folder / "composition.mid"))
    files["midi"] = "composition.mid"
    report(0.6, "Playing it back")
    synth.write_wav(folder / "composition.wav", synth.render_notes(combined, voice if voice != "auto" else "piano"))
    files["audio"] = "composition.wav"
    report(0.7, "Writing the score")
    score = notation.notes_to_score(combined, tempo, f"{title} (continued)", grid=4, split_hands=False,
                                    key=out["key"], time_signature=meter)
    pages = _sheet(folder, score, files, report)
    data = {"notes": _compact(combined), "tempo": round(tempo, 1), "key": out["key"], "pages": pages,
            "seed_count": len(seed), "new_count": len(new), "continuation_starts": new[0]["start"] if new else None,
            "model_in_key_rate": out["model_in_key_rate"], "options": options, "meter": meter,
            "stats": _note_stats(combined)}
    storage.save(rid, {"kind": "compose", "title": f"{title} (continued)", "files": files, "data": data,
                       "summary": f"{len(new)} new notes, {out['key']['name']}",
                       "source": {k: v for k, v in source.items() if k in ("type", "key", "id", "name")}})


# Style transfer --------------------------------------------------------------
def _spectro_png(spec, path, vmax):
    img = spec[:420][::-1]  # 0 to about 4.5 kHz, low notes at the bottom
    plt.imsave(str(path), img, cmap=analysis.SPECTRO_CMAP, vmin=0, vmax=vmax)


def run_style(storage, rid, content_path: Path, style_path: Path, names: tuple[str, str], options: dict, report):
    folder = storage.folder(rid)
    report(0.02, "Reading both clips")
    content, _ = _load_audio(content_path, options["start"] + options["seconds"] + 1)
    style, _ = _load_audio(style_path, 120)
    start = int(options["start"] * style_transfer.SR)
    if start >= len(content) - style_transfer.SR:
        raise UserError("The start time is past the end of the content clip.")
    content = content[start:]
    report(0.05, "Matching the texture")
    try:
        out = style_transfer.transfer(content, style, seconds=options["seconds"], steps=options["steps"],
                                      strength=options["strength"],
                                      progress=lambda f: report(0.05 + 0.9 * f,
                                                                "Matching the texture" if f < 0.85 else "Rebuilding the sound"))
    except ValueError as e:
        raise UserError(str(e)) from e
    files = {}
    for name, audio in (("content", out["content"]), ("style", out["style"]), ("result", out["audio"])):
        synth.write_wav(folder / f"{name}.wav", audio, style_transfer.SR)
        files[name] = f"{name}.wav"
    C, S, X = out["spectrograms"]
    vmax = float(np.percentile(np.concatenate([C[:420].ravel(), S[:420].ravel(), X[:420].ravel()]), 99.7))
    for name, spec in (("content", C), ("style", S), ("result", X)):
        _spectro_png(spec, folder / f"{name}-spectrogram.png", vmax)
        files[f"{name}_spectrogram"] = f"{name}-spectrogram.png"
    data = {"history": out["history"], "style_change": out["style_change"], "options": options,
            "names": {"content": names[0], "style": names[1]}}
    storage.save(rid, {"kind": "style", "title": f"{Path(names[0]).stem} in the style of {Path(names[1]).stem}",
                       "summary": f"{options['seconds']} s, strength {round(options['strength'] * 100)}%",
                       "files": files, "data": data})


def copy_result_audio(storage, from_id: str, rid: str, stem: str = "input") -> tuple[Path, str]:
    """Reuse the audio uploaded for an earlier result."""
    r = storage.load(from_id)
    src_name = (r.get("source") or {}).get("audio") or (r.get("source") or {}).get("input")
    if not src_name:
        raise UserError("That result has no audio to reuse.")
    src = storage.folder(from_id) / src_name
    if not src.exists():
        raise UserError("The audio for that result is no longer available.")
    dest_folder = storage.folder(rid)
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest = dest_folder / f"{stem}{src.suffix}"
    shutil.copyfile(src, dest)
    return dest, r.get("title") or src.name
