"""Measure how well audio-to-notes transcription recovers known notes.

Bach chorales from music21's bundled corpus (four voices, public domain) are rendered to
audio with FluidSynth and a General MIDI SoundFont, a different sound engine from the
app's own synthesizer, then transcribed. A note counts as found when its pitch matches
and its start is within 50 ms (mir_eval, onset-only). Also checks the melody-only mode
against the soprano line.

    python -m ml.evaluate_transcription path/to/soundfont.sf2 [number_of_chorales]

Needs: pip install mir_eval pyfluidsynth, plus the FluidSynth library.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import music21 as m21
import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from museassist import transcribe as T  # noqa: E402
from museassist.notation import score_to_notes  # noqa: E402

warnings.filterwarnings("ignore")
PROGRAMS = {"piano": 0, "strings": 48, "organ": 19}


def score(ref, est):
    import mir_eval
    if not est:
        return 0.0, 0.0, 0.0
    f = lambda ns: (np.array([[n["start"], n["end"]] for n in ns]),  # noqa: E731
                    np.array([440 * 2 ** ((n["pitch"] - 69) / 12) for n in ns]))
    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(*f(ref), *f(est), offset_ratio=None)
    return p, r, f1


def main(sf2: str, count: int = 20):
    paths = sorted(m21.corpus.getComposer("bach"), key=str)[: int(count)]
    rows = {k: [] for k in PROGRAMS}
    melody_rows = []
    for path in paths:
        s = m21.corpus.parse(path)
        notes = score_to_notes(s)
        if not notes:
            continue
        for name, program in PROGRAMS.items():
            pm = pretty_midi.PrettyMIDI(initial_tempo=90)
            inst = pretty_midi.Instrument(program=program)
            for n in notes:
                inst.notes.append(pretty_midi.Note(90, n["pitch"], n["start"], n["end"]))
            pm.instruments.append(inst)
            y = pm.fluidsynth(fs=T.SR, sf2_path=sf2).astype(np.float32)
            y /= np.max(np.abs(y)) or 1
            est = T.transcribe(y)
            rows[name].append(score(notes, est))
            if name == "piano":
                soprano = score_to_notes(m21.stream.Score([s.parts[0]]))
                melody_rows.append(score(soprano, T.transcribe(y, melody_only=True)))
        print(Path(str(path)).name, {k: round(v[-1][2], 2) for k, v in rows.items()}, flush=True)
    report = {"pieces": len(rows["piano"]), "source": "music21 corpus Bach chorales, rendered with FluidSynth",
              "match": "pitch and start within 50 ms", "results": {}}
    for name, vals in rows.items():
        p, r, f = np.mean(vals, axis=0)
        report["results"][name] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
    p, r, f = np.mean(melody_rows, axis=0)
    report["results"]["melody only (piano, against soprano)"] = {
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
    out = Path(__file__).resolve().parent.parent / "models" / "transcription_metrics.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], *(sys.argv[2:3] or [20]))
