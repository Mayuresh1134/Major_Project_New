"""Scores: reading MusicXML / MIDI / ABC, writing MIDI and MusicXML, and engraving sheet music.

music21 reads and converts scores; Verovio engraves them to SVG. Neither needs MuseScore.
"""
from __future__ import annotations

import copy
import os
import tempfile
import warnings
from pathlib import Path

import music21 as m21
import numpy as np
import pretty_midi
import verovio

SCORE_EXTENSIONS = {"musicxml", "xml", "mxl", "mid", "midi", "abc"}
MAX_PAGES = 40

# A few public-domain tunes, written in ABC notation, used as examples and seeds.
EXAMPLES = {
    "ode": ("Ode to Joy (Beethoven)", """X:1
T:Ode to Joy
C:Ludwig van Beethoven
M:4/4
L:1/4
Q:1/4=110
K:D
F F G A | A G F E | D D E F | F3/2 E/ E2 |
F F G A | A G F E | D D E F | E3/2 D/ D2 |
E E F D | E F/G/ F D | E F/G/ F E | D E A,2 |
F F G A | A G F E | D D E F | E3/2 D/ D2 |]"""),
    "greensleeves": ("Greensleeves (traditional)", """X:1
T:Greensleeves
C:Traditional
M:6/8
L:1/8
Q:3/8=60
K:Em
E | G2 A B>cB | A2 F D>EF | G2 E E>^DE | F2 ^D B,2 E |
G2 A B>cB | A2 F D>EF | G>FE ^D>^CD | E3 E2 |]"""),
    "twinkle": ("Twinkle, Twinkle, Little Star (traditional)", """X:1
T:Twinkle, Twinkle, Little Star
C:Traditional
M:4/4
L:1/4
Q:1/4=100
K:C
C C G G | A A G2 | F F E E | D D C2 |
G G F F | E E D2 | G G F F | E E D2 |
C C G G | A A G2 | F F E E | D D C2 |]"""),
    "minuet": ("Minuet in G (Petzold)", """X:1
T:Minuet in G
C:Christian Petzold
M:3/4
L:1/8
Q:1/4=120
K:G
d2 G A B c | d2 G2 G2 | e2 c d e f | g2 G2 G2 |
c2 d c B A | B2 c B A G | F2 G A B G | A6 |
d2 G A B c | d2 G2 G2 | e2 c d e f | g2 G2 G2 |
c2 d c B A | B2 c B A G | A2 B A G F | G6 |]"""),
}


class ScoreError(ValueError):
    """The file could not be read as music."""


def load_score(path: str | os.PathLike, fmt: str | None = None) -> m21.stream.Score:
    path = Path(path)
    ext = (fmt or path.suffix.lstrip(".")).lower()
    if ext not in SCORE_EXTENSIONS:
        raise ScoreError("Use a MusicXML (.musicxml, .mxl), MIDI (.mid) or ABC (.abc) file.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if ext in ("mid", "midi"):
                s = m21.converter.parse(str(path), format="midi", quantizePost=True,
                                        quarterLengthDivisors=(4, 3))
            elif ext == "abc":
                s = m21.converter.parse(path.read_text(encoding="utf-8", errors="replace"), format="abc")
            else:
                s = m21.converter.parse(str(path))
    except Exception as e:  # music21 raises many different exception types
        raise ScoreError(f"This file could not be read as music ({type(e).__name__}).") from e
    return _as_score(s)


def parse_abc(text: str) -> m21.stream.Score:
    text = (text or "").strip()
    if not text:
        raise ScoreError("Type or paste some ABC notation first.")
    lines = text.splitlines()
    has = lambda field: any(l.strip().startswith(field) for l in lines)  # noqa: E731
    if not has("K:"):  # bare notes: add a minimal header
        lines = ["M:4/4", "L:1/4", "K:C"] + lines
    else:
        # music21 needs the meter and default note length, which ABC lets you leave out
        k = next(i for i, l in enumerate(lines) if l.strip().startswith("K:"))
        extra = ([] if has("M:") else ["M:4/4"]) + ([] if has("L:") else ["L:1/8"])
        lines = lines[:k] + extra + lines[k:]
    if not lines[0].strip().startswith("X:"):
        lines = ["X:1"] + lines
    text = "\n".join(lines)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = m21.converter.parse(text, format="abc")
    except Exception as e:
        raise ScoreError(f"The ABC notation could not be read ({type(e).__name__}). "
                         "Check the K: line and the bar lines.") from e
    s = _as_score(s)
    if not s.flatten().notes:
        raise ScoreError("No notes were found in the ABC notation.")
    return s


def _as_score(s) -> m21.stream.Score:
    if isinstance(s, m21.stream.Opus):  # several tunes: keep the first
        s = s.scores[0] if s.scores else m21.stream.Score()
    if not isinstance(s, m21.stream.Score):
        sc = m21.stream.Score()
        sc.insert(0, s)
        s = sc
    if not s.flatten().notes:
        raise ScoreError("No notes were found in this file.")
    return s


def score_title(s: m21.stream.Score, fallback: str = "Untitled") -> str:
    md = s.metadata
    title = (md.bestTitle if md and md.bestTitle else None) or fallback
    return str(title)[:120]


def score_tempo(s: m21.stream.Score) -> float:
    marks = s.flatten().getElementsByClass(m21.tempo.MetronomeMark)
    for mm in marks:
        qpm = mm.getQuarterBPM()
        if qpm:
            return float(qpm)
    return 120.0


def score_meter(s: m21.stream.Score) -> str:
    ts = next(iter(s.recurse().getElementsByClass(m21.meter.TimeSignature)), None)
    if ts is not None and ts.ratioString in ("2/4", "3/4", "4/4", "6/8", "3/8", "2/2", "9/8", "12/8"):
        return ts.ratioString
    return "4/4"


def score_to_notes(s: m21.stream.Score) -> list[dict]:
    """Notes with times in seconds, using the score's own tempo marks."""
    with tempfile.TemporaryDirectory() as tmp:
        mid = os.path.join(tmp, "s.mid")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.write("midi", fp=mid)
        pm = pretty_midi.PrettyMIDI(mid)
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append({"pitch": int(n.pitch), "start": round(float(n.start), 4),
                          "end": round(float(n.end), 4), "velocity": int(n.velocity),
                          "program": int(inst.program), "is_drum": bool(inst.is_drum)})
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def notes_to_midi(notes, path, program: int = 0, tempo: float = 120.0, split_hands: bool = False,
                  time_signature: str | None = None, key: dict | None = None):
    """Write note dicts to a MIDI file. With split_hands, notes below middle C go to a second track."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    if time_signature:
        num, den = (int(x) for x in time_signature.split("/"))
        pm.time_signature_changes.append(pretty_midi.TimeSignature(num, den, 0))
    if key:
        pm.key_signature_changes.append(
            pretty_midi.KeySignature(int(key["tonic"]) + (12 if key["mode"] == "minor" else 0), 0))
    if split_hands:
        tracks = {"Right hand": [n for n in notes if n["pitch"] >= 60],
                  "Left hand": [n for n in notes if n["pitch"] < 60]}
    else:
        tracks = {"Melody": notes}
    for name, group in tracks.items():
        if not group:
            continue
        inst = pretty_midi.Instrument(program=int(program), name=name)
        for n in group:
            start, end = float(n["start"]), float(n["end"])
            if end - start < 0.02:
                end = start + 0.02
            inst.notes.append(pretty_midi.Note(int(n.get("velocity", 90)), int(n["pitch"]), start, end))
        pm.instruments.append(inst)
    pm.write(str(path))
    return pm


def _respell(score: m21.stream.Score, key: m21.key.Key) -> None:
    """Spell pitches as the key would: D# rather than Eb in E minor."""
    names = {p.pitchClass: p.name for p in key.getScale("major" if key.mode == "major" else "minor").pitches}
    if key.mode == "minor":
        lead = key.tonic.transpose("-m2")
        names.setdefault(lead.pitchClass, lead.name)
    sharps = key.sharps >= 0
    for n in score.recurse().notes:
        for p in (n.pitches if n.isChord else [n.pitch]):
            name = names.get(p.pitchClass)
            if name is None:
                if (p.accidental is not None) and ((p.accidental.alter > 0) != sharps):
                    p.getEnharmonic(inPlace=True)
                continue
            if p.name != name:
                midi = p.midi
                p.name = name
                p.octave += round((midi - p.midi) / 12)


def fit_beat_grid(notes, tempo: float, spread: float = 0.08, division: int = 4):
    """Refine a tempo estimate so note starts land on a sixteenth-note grid.

    Tries tempos within +-spread of `tempo` and returns (best_tempo, offset_seconds), where
    offset is the time of the grid line the music should be shifted to start from.
    """
    onsets = np.array(sorted({round(float(n["start"]), 3) for n in notes}))
    if len(onsets) < 4:
        return tempo, float(onsets[0]) if len(onsets) else 0.0
    best = (-1.0, tempo, 0.0)
    for t in np.arange(tempo * (1 - spread), tempo * (1 + spread), 0.1):
        step = 60.0 / t / division
        z = np.exp(2j * np.pi * onsets / step).mean()
        score = abs(z)
        if score > best[0]:
            phase = (np.angle(z) / (2 * np.pi)) * step
            best = (score, float(t), float(phase))
    _, t, phase = best
    step = 60.0 / t / division
    first = onsets[0]
    offset = phase + np.round((first - phase) / step) * step  # the grid line nearest the first note
    return t, float(offset)


def tidy_for_sheet(notes, tempo: float, division: int = 4):
    """Make performed note lengths readable: a note ends where the next one starts.

    Recordings let notes ring over the next one and leave tiny gaps, which engrave as ties,
    dots and short rests. Notes that start together (chords) keep a shared length.
    """
    step = 60.0 / tempo / division
    onsets = sorted({round(float(n["start"]), 4) for n in notes})
    out = []
    for n in notes:
        start, end = float(n["start"]), float(n["end"])
        later = [o for o in onsets if o >= start + step * 0.75]
        if later:
            nxt = later[0]
            if end > nxt or nxt - end < 2 * step:  # overlaps the next note, or leaves a tiny gap
                end = nxt
        out.append({**n, "end": max(end, start + step)})
    return out


def notes_to_score(notes, tempo: float, title: str, grid: int = 4, split_hands: bool = True,
                   key: dict | None = None, time_signature: str = "4/4") -> m21.stream.Score:
    """Quantize performed notes (seconds) onto a beat grid and build a score.

    `grid` is the number of divisions per quarter note (4 = sixteenth notes).
    """
    with tempfile.TemporaryDirectory() as tmp:
        mid = os.path.join(tmp, "n.mid")
        notes_to_midi(notes, mid, tempo=tempo, split_hands=split_hands, time_signature=time_signature, key=key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = m21.converter.parse(mid, format="midi", quantizePost=True,
                                    quarterLengthDivisors=(grid,))
    s = _as_score(s)
    parts = list(s.parts)
    for el in list(s.recurse().getElementsByClass(m21.tempo.MetronomeMark)):
        el.activeSite.remove(el)  # replaced by one rounded mark below
    for p in parts:
        p.partName = ""
        p.partAbbreviation = ""
        for inst in p.recurse().getElementsByClass(m21.instrument.Instrument):
            inst.partName = ""
            inst.partAbbreviation = ""
            inst.instrumentName = ""
            inst.instrumentAbbreviation = ""
        lowest = min((n.pitch.midi for n in p.flatten().notes if hasattr(n, "pitch")), default=60)
        if split_hands and len(parts) > 1 and lowest < 60 and p is parts[-1]:
            p.insert(0, m21.clef.BassClef())
    if key:
        k = m21.key.Key(m21.pitch.Pitch(int(key["tonic"])).name, key["mode"])
        for p in parts:
            if not list(p.recurse().getElementsByClass(m21.key.KeySignature)):
                measures = p.getElementsByClass(m21.stream.Measure)
                (measures[0] if len(measures) else p).insert(0, m21.key.Key(k.tonic.name, k.mode))
        _respell(s, k)
    s.insert(0, m21.metadata.Metadata())
    s.metadata.title = title
    s.metadata.composer = ""
    first = parts[0].getElementsByClass(m21.stream.Measure)
    target = first[0] if len(first) else parts[0]
    target.insert(0, m21.tempo.MetronomeMark(number=int(round(tempo))))
    return s


def write_musicxml(s: m21.stream.Score, path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.write("musicxml", fp=str(path))


def engrave(musicxml_path, page_width: int = 2100, scale: int = 40) -> list[str]:
    """Engrave a MusicXML file to SVG pages with Verovio."""
    tk = verovio.toolkit(False)  # fonts are loaded from the resource path below
    # Verovio only finds its fonts by itself on the main thread; jobs run in worker threads.
    tk.setResourcePath(os.path.join(os.path.dirname(verovio.__file__), "data"))
    tk.setOptions({
        "pageWidth": page_width, "pageHeight": int(page_width * 1.414), "scale": scale,
        "pageMarginTop": 60, "pageMarginBottom": 60, "pageMarginLeft": 60, "pageMarginRight": 60,
        "footer": "none", "header": "auto", "breaks": "auto", "adjustPageHeight": False,
        "svgViewBox": True, "svgRemoveXlink": True, "smuflTextFont": "embedded",
    })
    if not tk.loadFile(str(musicxml_path)):
        raise ScoreError("The score could not be engraved.")
    pages = min(tk.getPageCount(), MAX_PAGES)
    svgs = [tk.renderToSVG(i) for i in range(1, pages + 1)]
    if svgs:  # the last page only needs to be as tall as its music
        tk.setOptions({"adjustPageHeight": True})
        tk.redoLayout()
        svgs[-1] = tk.renderToSVG(min(tk.getPageCount(), pages))
    return svgs


def transpose_score(s: m21.stream.Score, semitones: int) -> m21.stream.Score:
    if not semitones:
        return s
    t = copy.deepcopy(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return t.transpose(int(semitones), inPlace=False)
