"""Describe a recording: tempo and beats, key, chords over time, loudness and tone."""
from __future__ import annotations

import librosa
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .theory import NOTE_NAMES, chord_templates, estimate_key  # noqa: E402

SR = 22050
HOP = 512
MAX_SECONDS = 600

SPECTRO_CMAP = LinearSegmentedColormap.from_list(
    "ink_brass", ["#14132a", "#2b2860", "#6b3f7a", "#b8643f", "#e0ae52", "#fbf3d6"])


def load_audio(path, max_seconds: float = MAX_SECONDS):
    """Mono audio at 22,050 Hz plus facts about the original file."""
    info = {}
    try:
        import soundfile as sf
        i = sf.info(str(path))
        info = {"original_sample_rate": i.samplerate, "channels": i.channels}
    except Exception:
        pass
    y, sr = librosa.load(str(path), sr=SR, mono=True, duration=max_seconds)
    if y.size == 0:
        raise ValueError("The file contains no audio.")
    full = librosa.get_duration(path=str(path))
    info["duration"] = round(float(full), 2)
    info["analysed_seconds"] = round(len(y) / SR, 2)
    return y.astype(np.float32), info


def waveform_peaks(y: np.ndarray, bins: int = 1200) -> list[list[float]]:
    """[min, max] pairs for drawing the waveform."""
    bins = min(bins, len(y))
    edges = np.linspace(0, len(y), bins + 1).astype(int)
    return [[round(float(y[a:b].min()), 3), round(float(y[a:b].max()), 3)]
            for a, b in zip(edges[:-1], edges[1:]) if b > a]


def save_spectrogram(y: np.ndarray, path, width: int = 1400, height: int = 280) -> None:
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=height // 2, hop_length=HOP, fmax=11025)
    S = librosa.power_to_db(S, ref=np.max, top_db=80)
    if S.shape[1] > width:  # average neighbouring frames down to the image width
        edges = np.linspace(0, S.shape[1], width + 1).astype(int)
        S = np.stack([S[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    plt.imsave(str(path), S[::-1], cmap=SPECTRO_CMAP, vmin=-80, vmax=0)


def estimate_tempo(onset_env: np.ndarray) -> float:
    """Tempo from the onset envelope's autocorrelation.

    Each candidate tempo is scored by how well the envelope repeats at one beat, plus half
    credit for repeating at half a beat (beats are usually subdivided), weighted by a broad
    preference for tempos near 115 BPM. librosa's beat tracker alone often locks onto 2/3
    or 3/2 of the tempo; this choice is then handed to it as a fixed tempo.
    """
    fps = SR / HOP
    env = onset_env - onset_env.mean()
    ac = librosa.autocorrelate(env, max_size=int(fps * 2.2))
    if ac[0] <= 0:
        return 120.0
    ac = ac / ac[0]
    lags = np.arange(len(ac))
    cand = np.arange(50, 220, 0.25)
    support = np.interp(60 * fps / cand, lags, ac) + 0.5 * np.interp(30 * fps / cand, lags, ac)
    prior = np.exp(-0.5 * (np.log2(cand / 115) / 0.6) ** 2)
    return float(cand[np.argmax(np.clip(support, 0, None) * prior)])


def tempo_and_beats(y: np.ndarray, onset_env=None) -> dict:
    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    guess = estimate_tempo(onset_env)
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=HOP, bpm=guess)
    beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    gaps = np.diff(beat_times)
    tempo = guess
    if len(gaps) > 3:
        refined = 60.0 / float(np.median(gaps))
        if abs(refined - guess) / guess < 0.03:  # use the beats' own spacing when it agrees
            tempo = refined
    # how steady is the pulse: spread of the gaps between beats
    steadiness = float(1 - min(np.std(gaps) / np.mean(gaps), 1)) if len(gaps) > 3 else 0.0
    return {"bpm": round(tempo, 1), "beats": [round(float(t), 3) for t in beat_times],
            "steadiness": round(steadiness, 2)}


def detect_chords(chroma: np.ndarray, beat_frames: np.ndarray, duration: float,
                  self_prob: float = 0.8, sharpness: float = 30.0) -> list[dict]:
    """Major/minor triads per beat, smoothed with a simple hidden Markov model (Viterbi).

    chroma: (12, n_frames) harmonic chroma. Returns segments {start, end, chord}.
    """
    names, templates = chord_templates()
    n_frames = chroma.shape[1]
    bounds = np.unique(np.concatenate([[0], beat_frames[beat_frames < n_frames], [n_frames]])).astype(int)
    if len(bounds) < 3:  # no usable beats: use half-second blocks
        bounds = np.unique(np.append(np.arange(0, n_frames, int(0.5 * SR / HOP)), n_frames))
    sync = np.stack([chroma[:, a:b].mean(axis=1) for a, b in zip(bounds[:-1], bounds[1:])], axis=1)
    energy = sync.sum(axis=0)
    norms = np.linalg.norm(sync, axis=0) + 1e-9
    sim = templates @ (sync / norms)  # (24, n_beats) cosine similarity
    no_chord = np.full((1, sim.shape[1]), 0.55)
    sim = np.vstack([sim, no_chord])
    sim[-1, energy < 0.1 * np.median(energy)] = 1.5  # silence
    labels = names + ["N"]
    log_emit = sharpness * sim
    n_states = len(labels)
    log_trans = np.full((n_states, n_states), np.log((1 - self_prob) / (n_states - 1)))
    np.fill_diagonal(log_trans, np.log(self_prob))
    score = log_emit[:, 0].copy()
    back = np.zeros((n_states, sim.shape[1]), dtype=int)
    for t in range(1, sim.shape[1]):
        cand = score[:, None] + log_trans
        back[:, t] = np.argmax(cand, axis=0)
        score = cand[back[:, t], np.arange(n_states)] + log_emit[:, t]
    path = [int(np.argmax(score))]
    for t in range(sim.shape[1] - 1, 0, -1):
        path.append(back[path[-1], t])
    path = path[::-1]
    times = librosa.frames_to_time(bounds, sr=SR, hop_length=HOP)
    times[-1] = duration
    segments = []
    for i, state in enumerate(path):
        label = labels[state]
        if segments and segments[-1]["chord"] == label:
            segments[-1]["end"] = round(float(times[i + 1]), 3)
        else:
            segments.append({"start": round(float(times[i]), 3), "end": round(float(times[i + 1]), 3),
                             "chord": label})
    return segments


def settle_key(key: dict, chords: list[dict], margin: float = 0.08) -> dict:
    """Break a near-tie between the two best keys with the chord the music ends (or starts) on.

    The pitch-profile method often confuses a key with its neighbours (the dominant, the
    relative minor); listeners hear "home" where a piece settles.
    """
    if not key.get("runner_up") or key["score"] - key.get("runner_up_score", 0) > margin:
        return key
    named = [c for c in chords if c["chord"] != "N" and c["end"] - c["start"] > 0.3]
    if not named:
        return key
    def home_chord(name):
        tonic, mode = name.split()
        return tonic + ("m" if mode == "minor" else "")
    for c in (named[-1], named[0]):
        if c["chord"] == home_chord(key["runner_up"]) and c["chord"] != home_chord(key["name"]):
            tonic_name, mode = key["runner_up"].split()
            return {**key, "tonic": NOTE_NAMES.index(tonic_name), "mode": mode, "name": key["runner_up"],
                    "score": key["runner_up_score"], "runner_up": key["name"], "runner_up_score": key["score"],
                    "settled_by_chord": c["chord"]}
        if c["chord"] == home_chord(key["name"]):
            return key
    return key


def describe_tone(y: np.ndarray, harmonic: np.ndarray, percussive: np.ndarray) -> dict:
    centroid = float(np.median(librosa.feature.spectral_centroid(y=y, sr=SR)))
    perc_share = float(np.sum(percussive ** 2) / (np.sum(harmonic ** 2) + np.sum(percussive ** 2) + 1e-12))
    rms = librosa.feature.rms(y=y)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    loud = rms_db[rms_db > -70]
    dyn = float(np.percentile(loud, 95) - np.percentile(loud, 10)) if loud.size else 0.0
    return {
        "brightness_hz": round(centroid),
        "brightness": "bright" if centroid > 2800 else "balanced" if centroid > 1500 else "warm",
        "percussive_share": round(perc_share, 2),
        "rhythm": "drum-led" if perc_share > 0.45 else "mixed" if perc_share > 0.2 else "mostly sustained",
        "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)), 1),
        "peak_db": round(float(20 * np.log10(np.max(np.abs(y)) + 1e-9)), 1),
        "dynamic_range_db": round(dyn, 1),
        "loudness_curve": [round(float(v), 1) for v in
                           np.interp(np.linspace(0, len(rms_db) - 1, 300), np.arange(len(rms_db)), rms_db)],
    }


def analyse(y: np.ndarray, progress=None) -> dict:
    """Everything the Analyze page shows except genre and the spectrogram image."""
    duration = len(y) / SR
    step = (lambda f: progress(f)) if progress else (lambda f: None)
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    tb = tempo_and_beats(y, onset_env)
    step(0.25)
    harmonic, percussive = librosa.effects.hpss(y)
    step(0.5)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=SR, hop_length=HOP)
    step(0.75)
    profile = chroma.sum(axis=1)
    key = estimate_key(profile)
    beat_frames = librosa.time_to_frames(tb["beats"], sr=SR, hop_length=HOP)
    chords = detect_chords(chroma, np.asarray(beat_frames), duration)
    key = settle_key(key, chords)
    counts = {}
    for c in chords:
        if c["chord"] != "N":
            counts[c["chord"]] = counts.get(c["chord"], 0) + c["end"] - c["start"]
    common = sorted(counts, key=counts.get, reverse=True)[:6]
    tone = describe_tone(y, harmonic, percussive)
    step(1.0)
    return {
        "tempo": tb,
        "key": key,
        "pitch_profile": [{"note": NOTE_NAMES[i], "weight": round(float(v / (profile.max() or 1)), 3)}
                          for i, v in enumerate(profile)],
        "chords": chords,
        "common_chords": common,
        "tone": tone,
        "peaks": waveform_peaks(y),
        "duration": round(duration, 2),
    }
