"""Continue a melody with the project's trained LSTM.

The model (models/melody_lstm.h5) follows TensorFlow's "Generate music with an RNN"
tutorial: it reads the last 25 notes as (pitch / 128, step, duration) and predicts the
next note's pitch (128 logits), step (seconds since the previous note started) and
duration. It is run here with numpy, so TensorFlow is not needed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

from .theory import estimate_key, note_pitch_class_weights, scale_pitch_classes

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "melody_lstm.h5"
SEQ_LENGTH = 25


class MelodyLSTM:
    def __init__(self, path: str | Path = MODEL_PATH):
        with h5py.File(path, "r") as f:
            g = f["model_weights"]
            self.kernel = g["lstm/lstm/lstm_cell/kernel"][()].astype(np.float64)
            self.recurrent = g["lstm/lstm/lstm_cell/recurrent_kernel"][()].astype(np.float64)
            self.bias = g["lstm/lstm/lstm_cell/bias"][()].astype(np.float64)
            self.heads = {name: (g[f"{name}/{name}/kernel"][()].astype(np.float64),
                                 g[f"{name}/{name}/bias"][()].astype(np.float64))
                          for name in ("pitch", "step", "duration")}
        self.units = self.recurrent.shape[0]

    def predict(self, seq: np.ndarray):
        """seq: (25, 3) array of (pitch/128, step, duration). Returns pitch logits, step, duration."""
        h = np.zeros(self.units)
        c = np.zeros(self.units)
        for x in seq:
            z = x @ self.kernel + h @ self.recurrent + self.bias
            i, f, g, o = np.split(z, 4)  # Keras gate order: input, forget, cell, output
            i, f, o = (1 / (1 + np.exp(-v)) for v in (i, f, o))
            c = f * c + i * np.tanh(g)
            h = o * np.tanh(c)
        out = {k: h @ W + b for k, (W, b) in self.heads.items()}
        return out["pitch"], float(out["step"][0]), float(out["duration"][0])


@lru_cache(maxsize=1)
def get_model() -> MelodyLSTM:
    return MelodyLSTM()


def seed_sequence(notes) -> np.ndarray:
    notes = sorted(notes, key=lambda n: (n["start"], n["pitch"]))[-SEQ_LENGTH:]
    rows, prev = [], notes[0]["start"] if notes else 0.0
    for n in notes:
        rows.append([n["pitch"] / 128.0, n["start"] - prev, n["end"] - n["start"]])
        prev = n["start"]
    seq = np.array(rows, dtype=np.float64).reshape(-1, 3)
    if len(seq) < SEQ_LENGTH:
        seq = np.vstack([np.zeros((SEQ_LENGTH - len(seq), 3)), seq])
    return seq


def continue_melody(seed_notes, n_notes: int = 64, temperature: float = 1.0,
                    stay_in_key: bool = True, keep_range: bool = True, max_leap: int | None = 9,
                    match_pace: bool = True, rng_seed: int | None = None,
                    model: MelodyLSTM | None = None) -> dict:
    """Generate `n_notes` new notes that follow `seed_notes` (dicts with pitch/start/end).

    stay_in_key: only allow pitches from the seed's estimated key.
    keep_range: only allow pitches within a fifth of the seed's lowest and highest notes.
    max_leap: largest allowed jump, in semitones, from the previous note (None = any).
    match_pace: stretch the new notes' timing so their typical gap matches the seed's.
    """
    if not seed_notes:
        raise ValueError("The seed has no notes.")
    model = model or get_model()
    rng = np.random.default_rng(rng_seed)
    temperature = float(np.clip(temperature, 0.1, 3.0))
    seed = sorted(seed_notes, key=lambda n: (n["start"], n["pitch"]))
    key = estimate_key(note_pitch_class_weights(seed))

    allowed = np.ones(128, dtype=bool)
    if stay_in_key:
        pcs = scale_pitch_classes(key["tonic"], key["mode"])
        allowed &= np.array([p % 12 in pcs for p in range(128)])
    if keep_range:
        lo = min(n["pitch"] for n in seed) - 7
        hi = max(n["pitch"] for n in seed) + 7
        allowed &= (np.arange(128) >= max(lo, 21)) & (np.arange(128) <= min(hi, 108))
    if not allowed.any():
        allowed[:] = True

    seq = seed_sequence(seed)
    prev_start = seed[-1]["start"]
    generated, unconstrained_in_key = [], 0
    scale = scale_pitch_classes(key["tonic"], key["mode"])
    for _ in range(int(n_notes)):
        logits, step, duration = model.predict(seq)
        raw_top = int(np.argmax(logits))
        unconstrained_in_key += raw_top % 12 in scale
        z = logits / temperature
        mask = allowed.copy()
        if max_leap:
            last = generated[-1]["pitch"] if generated else seed[-1]["pitch"]
            near = np.abs(np.arange(128) - last) <= max_leap
            if (mask & near).any():
                mask &= near
        z[~mask] = -np.inf
        p = np.exp(z - np.max(z))
        p /= p.sum()
        pitch = int(rng.choice(128, p=p))
        step = float(np.clip(step, 0.0, 2.0))
        duration = float(np.clip(duration, 0.05, 3.0))
        start = prev_start + step
        generated.append({"pitch": pitch, "start": round(start, 4), "end": round(start + duration, 4),
                          "velocity": 90})
        seq = np.vstack([seq[1:], [pitch / 128.0, step, duration]])
        prev_start = start
    if match_pace and len(seed) > 2 and generated:
        seed_steps = np.diff([n["start"] for n in seed])
        new_steps = np.diff([seed[-1]["start"]] + [n["start"] for n in generated])
        a, b = np.median(seed_steps[seed_steps > 0.02]) if (seed_steps > 0.02).any() else 0, \
            np.median(new_steps[new_steps > 0.02]) if (new_steps > 0.02).any() else 0
        if a > 0 and b > 0:
            ratio = float(np.clip(a / b, 0.25, 4.0))
            t0 = seed[-1]["start"]
            for n in generated:
                length = (n["end"] - n["start"]) * ratio
                n["start"] = round(t0 + (n["start"] - t0) * ratio, 4)
                n["end"] = round(n["start"] + length, 4)
    # a melody plays one note at a time: end each note when the next one starts
    for a_, b_ in zip(generated, generated[1:]):
        if b_["start"] > a_["start"] + 0.05:
            a_["end"] = min(a_["end"], b_["start"])
    return {
        "notes": generated,
        "key": key,
        "model_in_key_rate": round(unconstrained_in_key / max(len(generated), 1), 3),
    }
