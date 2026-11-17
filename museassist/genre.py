"""Music genre classification.

Each 3-second slice of a song is described by 57 numbers (chroma, loudness, spectral
shape, zero-crossing rate, harmonic and percussive energy, tempo and 20 MFCCs, as means
and variances) — the same features as the GTZAN "features_3_sec.csv" table the model
was trained on. Three small neural networks score every slice; the song's genre is the
average over slices. Weights are stored in models/genre_mlp.npz and run with numpy.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS / "genre_mlp.npz"
SR = 22050
SEGMENT = 66149          # samples per slice, as in the training table
MAX_SEGMENTS = 12

FEATURES = (
    ["chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
     "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean",
     "spectral_bandwidth_var", "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean",
     "zero_crossing_rate_var", "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var",
     "tempo"]
    + [f"mfcc{i}_{s}" for i in range(1, 21) for s in ("mean", "var")]
)


def segment_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """The 57 features for one slice of audio, in FEATURES order."""
    def mv(a):
        return [float(np.mean(a)), float(np.var(a))]
    f = []
    f += mv(librosa.feature.chroma_stft(y=y, sr=sr))
    f += mv(librosa.feature.rms(y=y))
    f += mv(librosa.feature.spectral_centroid(y=y, sr=sr))
    f += mv(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    f += mv(librosa.feature.spectral_rolloff(y=y, sr=sr))
    f += mv(librosa.feature.zero_crossing_rate(y))
    harmonic, percussive = librosa.effects.hpss(y)
    f += mv(harmonic) + mv(percussive)
    tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
    f.append(float(np.atleast_1d(tempo)[0]))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for row in mfcc:
        f += mv(row)
    return np.array(f, dtype=np.float64)


def pick_segments(y: np.ndarray, max_segments: int = MAX_SEGMENTS) -> list[tuple[float, np.ndarray]]:
    """Up to `max_segments` evenly spread 3-second slices, skipping near-silent ones."""
    n = len(y) // SEGMENT
    if n == 0:
        return [(0.0, np.pad(y, (0, SEGMENT - len(y))))] if len(y) > SR else []
    idx = np.unique(np.linspace(0, n - 1, min(n, max_segments)).round().astype(int))
    out = []
    for i in idx:
        seg = y[i * SEGMENT:(i + 1) * SEGMENT]
        if np.sqrt(np.mean(seg ** 2)) > 1e-3:
            out.append((i * SEGMENT / SR, seg))
    return out


class GenreModel:
    def __init__(self, path: str | Path = MODEL_PATH):
        d = np.load(path, allow_pickle=False)
        self.classes = [str(c) for c in d["classes"]]
        self.mean, self.scale = d["mean"], d["scale"]
        self.nets = []
        for k in range(int(d["n_nets"])):
            layers, j = [], 0
            while f"net{k}_W{j}" in d:
                layers.append((d[f"net{k}_W{j}"], d[f"net{k}_b{j}"]))
                j += 1
            self.nets.append(layers)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = (np.atleast_2d(X) - self.mean) / self.scale
        probs = []
        for layers in self.nets:
            h = X
            for i, (W, b) in enumerate(layers):
                h = h @ W + b
                if i < len(layers) - 1:
                    h = np.maximum(h, 0)
            h = np.exp(h - h.max(axis=1, keepdims=True))
            probs.append(h / h.sum(axis=1, keepdims=True))
        return np.mean(probs, axis=0)


@lru_cache(maxsize=1)
def get_model() -> GenreModel:
    return GenreModel()


def model_report() -> dict:
    p = MODELS / "genre_metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def classify(y: np.ndarray, sr: int = SR, progress=None) -> dict:
    """Genre probabilities for a whole recording (mono float audio at 22,050 Hz)."""
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    segments = pick_segments(y)
    if not segments:
        raise ValueError("The recording is too short or too quiet to classify (it needs at least 3 seconds).")
    model = get_model()
    rows = []
    for i, (_, seg) in enumerate(segments):
        rows.append(segment_features(seg))
        if progress:
            progress((i + 1) / len(segments))
    per_segment = model.predict_proba(np.array(rows))
    mean = per_segment.mean(axis=0)
    order = np.argsort(-mean)
    agree = float(np.mean(per_segment.argmax(axis=1) == order[0]))
    return {
        "genres": [{"genre": model.classes[i], "probability": round(float(mean[i]), 4)} for i in order],
        "top": model.classes[order[0]],
        "segments": [{"start": round(t, 1), "genre": model.classes[int(np.argmax(p))]}
                     for (t, _), p in zip(segments, per_segment)],
        "agreement": round(agree, 3),
    }
