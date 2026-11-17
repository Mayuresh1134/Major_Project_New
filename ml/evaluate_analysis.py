"""Check tempo, key and chord detection on band mixes with known answers.

Random progressions of major and minor triads are played with piano chords, a bass line
and drums by the built-in synthesizer at random tempos, then analysed. Reports how often
the tempo is within 4% (also allowing double or half), the key is right, and the chord at the middle of each beat is right.

    python -m ml.evaluate_analysis [number_of_pieces]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from museassist import analysis, synth  # noqa: E402
from museassist.theory import chord_templates  # noqa: E402


STYLES = ("band", "no drums", "half-time")


def piece(rng, style="band"):
    names, _ = chord_templates()
    tonic = int(rng.integers(0, 12))
    # chords from the major key: I ii iii IV V vi; start on I and end with V-I so the key is clear
    degrees = [(0, 0), (2, 1), (4, 1), (5, 0), (7, 0), (9, 1)]
    bpm = float(rng.uniform(70, 150))
    beat = 60 / bpm
    notes, truth, t = [], [], 0.0
    plan = [0] + [int(rng.integers(0, 6)) for _ in range(5)] + [4, 0]
    for d in plan:
        step, minor = degrees[d]
        root = (tonic + step) % 12
        third = 3 if minor else 4
        for b in range(4):
            for p in (60 + root, 60 + root + third, 60 + root + 7):
                notes.append({"pitch": p if p < 72 else p - 12, "start": t, "end": t + beat * 0.95,
                              "velocity": 70, "program": 0})
            notes.append({"pitch": 36 + root, "start": t, "end": t + beat * 0.9, "velocity": 90, "program": 33})
            if style == "band":
                notes.append({"pitch": 36 if b % 2 == 0 else 38, "start": t, "end": t + 0.1, "velocity": 100, "is_drum": True})
                notes.append({"pitch": 42, "start": t + beat / 2, "end": t + beat / 2 + 0.05, "velocity": 80, "is_drum": True})
            elif style == "half-time":
                if b in (0, 2):
                    notes.append({"pitch": 36 if b == 0 else 38, "start": t, "end": t + 0.1, "velocity": 100, "is_drum": True})
                for h in (0, 0.5):
                    notes.append({"pitch": 42, "start": t + h * beat, "end": t + h * beat + 0.05, "velocity": 70, "is_drum": True})
            truth.append((t + beat / 2, names[root * 2 + minor]))
            t += beat
    return notes, truth, bpm, tonic


def main(count: int = 10, seed: int = 1):
    """seed 0 was used while developing the tempo and key methods; report on another seed."""
    count = int(count)
    rng = np.random.default_rng(int(seed))
    tempo_ok = tempo_octave_ok = key_ok = chord_ok = beats = 0
    for i in range(count):
        notes, truth, bpm, tonic = piece(rng, STYLES[i % len(STYLES)])
        r = analysis.analyse(synth.render_notes(notes, "auto"))
        found = r["tempo"]["bpm"]
        tempo_ok += abs(found - bpm) / bpm < 0.04
        tempo_octave_ok += any(abs(found * m - bpm) / bpm < 0.04 for m in (1, 2, 0.5))
        key_ok += r["key"]["tonic"] == tonic and r["key"]["mode"] == "major"
        for mid, label in truth:
            found = next((c["chord"] for c in r["chords"] if c["start"] <= mid < c["end"]), "N")
            chord_ok += found == label
            beats += 1
    report = {"pieces": int(count), "beats": beats, "tempo_within_4_percent": round(tempo_ok / count, 3),
              "tempo_within_4_percent_or_double_or_half": round(tempo_octave_ok / count, 3),
              "key_correct": round(key_ok / count, 3), "chord_correct_per_beat": round(chord_ok / beats, 3)}
    print(json.dumps(report, indent=2))
    (Path(__file__).resolve().parent.parent / "models" / "analysis_metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(*(sys.argv[1:3] or [10]))
