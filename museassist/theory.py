"""Small music-theory helpers shared by the analysis, generation and practice tools."""
from __future__ import annotations

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]  # natural minor

# Krumhansl-Kessler key profiles (probe-tone ratings).
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def midi_to_name(pitch: int) -> str:
    """60 -> 'C4'."""
    pitch = int(pitch)
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def estimate_key(pitch_class_weights) -> dict:
    """Estimate the key from 12 pitch-class weights (chroma energy or note durations).

    Correlates the weights with the 24 rotated Krumhansl-Kessler profiles and returns
    the best key, its correlation, and the runner-up.
    """
    w = np.asarray(pitch_class_weights, dtype=float)
    if w.shape != (12,) or not np.any(w > 0):
        return {"tonic": 0, "mode": "major", "name": "C major", "score": 0.0, "runner_up": None}
    scores = []
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = np.corrcoef(w, np.roll(profile, tonic))[0, 1]
            scores.append((0.0 if np.isnan(r) else float(r), tonic, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    return {
        "tonic": best[1],
        "mode": best[2],
        "name": f"{NOTE_NAMES[best[1]]} {best[2]}",
        "score": round(best[0], 3),
        "runner_up": f"{NOTE_NAMES[second[1]]} {second[2]}",
        "runner_up_score": round(second[0], 3),
    }


def scale_pitch_classes(tonic: int, mode: str) -> set[int]:
    steps = MAJOR_STEPS if mode == "major" else MINOR_STEPS
    pcs = {(tonic + s) % 12 for s in steps}
    if mode == "minor":
        pcs.add((tonic + 11) % 12)  # raised 7th of harmonic minor is common in minor melodies
    return pcs


def note_pitch_class_weights(notes) -> np.ndarray:
    """Duration-weighted pitch-class histogram of note dicts with pitch/start/end."""
    w = np.zeros(12)
    for n in notes:
        w[int(n["pitch"]) % 12] += max(float(n["end"]) - float(n["start"]), 0.05)
    return w


# Chord templates for chord recognition: 12 major + 12 minor triads.
def chord_templates():
    names, templates = [], []
    for root in range(12):
        for quality, intervals in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            t = np.zeros(12)
            for i in intervals:
                t[(root + i) % 12] = 1.0
            names.append(NOTE_NAMES[root] + quality)
            templates.append(t / np.linalg.norm(t))
    return names, np.array(templates)
