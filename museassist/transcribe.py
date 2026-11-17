"""Audio to notes with Spotify's Basic Pitch model.

The model file (models/basic_pitch_nmp.onnx) and the decoding below come from
https://github.com/spotify/basic-pitch (Apache License 2.0, see models/BASIC_PITCH_LICENSE).
The `basic-pitch` package itself pins old library versions, so this module runs the
ONNX model with onnxruntime and carries a trimmed copy of its note decoding
(`output_to_notes_polyphonic`, pitch bends left out).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import scipy.signal

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "basic_pitch_nmp.onnx"

SR = 22050
FFT_HOP = 256
ANNOTATIONS_FPS = SR // FFT_HOP                      # 86 frames per second
AUDIO_N_SAMPLES = SR * 2 - FFT_HOP                   # 43,844 samples per window
ANNOT_N_FRAMES = ANNOTATIONS_FPS * 2                 # 172 frames per window
OVERLAP_FRAMES = 30
OVERLAP_LEN = OVERLAP_FRAMES * FFT_HOP
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN
MIDI_OFFSET = 21
MAX_FREQ_IDX = 87
MAGIC_ALIGNMENT_OFFSET = 0.0018


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"])


def run_model(y: np.ndarray, progress=None) -> dict:
    """Frame-level note, onset and pitch-contour activations for mono audio at 22,050 Hz."""
    sess = _session()
    original_length = len(y)
    y = np.concatenate([np.zeros(OVERLAP_LEN // 2, dtype=np.float32), y.astype(np.float32)])
    starts = list(range(0, len(y), HOP_SIZE))
    outputs = {"note": [], "onset": [], "contour": []}
    batch = 8
    for b in range(0, len(starts), batch):
        windows = []
        for s in starts[b:b + batch]:
            w = y[s:s + AUDIO_N_SAMPLES]
            if len(w) < AUDIO_N_SAMPLES:
                w = np.pad(w, (0, AUDIO_N_SAMPLES - len(w)))
            windows.append(w)
        x = np.stack(windows)[:, :, None]
        note, onset, contour = sess.run(
            ["StatefulPartitionedCall:1", "StatefulPartitionedCall:2", "StatefulPartitionedCall:0"],
            {"serving_default_input_2:0": x})
        outputs["note"].append(note)
        outputs["onset"].append(onset)
        outputs["contour"].append(contour)
        if progress:
            progress(min(1.0, (b + batch) / len(starts)))
    n_olap = OVERLAP_FRAMES // 2
    n_frames = int(original_length / HOP_SIZE * (ANNOT_N_FRAMES - OVERLAP_FRAMES))
    result = {}
    for k, parts in outputs.items():
        a = np.concatenate(parts)[:, n_olap:-n_olap, :]
        result[k] = a.reshape(-1, a.shape[2])[:n_frames]
    return result


def frames_to_time(n_frames: int) -> np.ndarray:
    original = np.arange(n_frames) * FFT_HOP / SR
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    offset = (FFT_HOP / SR) * (ANNOT_N_FRAMES - AUDIO_N_SAMPLES / FFT_HOP) + MAGIC_ALIGNMENT_OFFSET
    return original - offset * window_numbers


def _inferred_onsets(onsets, frames, n_diff=2):
    diffs = []
    for n in range(1, n_diff + 1):
        padded = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(padded[n:, :] - padded[:-n, :])
    diff = np.min(diffs, axis=0)
    diff[diff < 0] = 0
    diff[:n_diff, :] = 0
    if diff.max() > 0:
        diff = onsets.max() * diff / diff.max()
    return np.max([onsets, diff], axis=0)


def decode_notes(frames, onsets, onset_thresh=0.5, frame_thresh=0.3, min_note_len=11,
                 min_pitch=None, max_pitch=None, energy_tol=11, melodia_trick=True):
    """Note events (start_frame, end_frame, midi_pitch, amplitude) from model activations."""
    frames = frames.copy()
    onsets = onsets.copy()
    n_frames = frames.shape[0]
    if min_pitch is not None:
        lo = max(int(min_pitch) - MIDI_OFFSET, 0)
        frames[:, :lo] = 0
        onsets[:, :lo] = 0
    if max_pitch is not None:
        hi = min(int(max_pitch) - MIDI_OFFSET + 1, frames.shape[1])
        frames[:, hi:] = 0
        onsets[:, hi:] = 0
    onsets = _inferred_onsets(onsets, frames)

    peak = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak[peaks] = onsets[peaks]
    t_idx, f_idx = np.where(peak >= onset_thresh)
    remaining = frames.copy()
    events = []

    def clear(i0, i1, f):
        remaining[i0:i1, f] = 0
        if f < MAX_FREQ_IDX:
            remaining[i0:i1, f + 1] = 0
        if f > 0:
            remaining[i0:i1, f - 1] = 0

    for start, f in zip(t_idx[::-1], f_idx[::-1]):  # go backwards in time
        if start >= n_frames - 1:
            continue
        i, k = start + 1, 0
        while i < n_frames - 1 and k < energy_tol:
            k = k + 1 if remaining[i, f] < frame_thresh else 0
            i += 1
        i -= k
        if i - start <= min_note_len:
            continue
        clear(start, i, f)
        events.append((int(start), int(i), int(f + MIDI_OFFSET), float(np.mean(frames[start:i, f]))))

    if melodia_trick:  # pick up sustained notes whose onsets were missed
        while np.max(remaining) > frame_thresh:
            i_mid, f = np.unravel_index(np.argmax(remaining), remaining.shape)
            remaining[i_mid, f] = 0
            i, k = i_mid + 1, 0
            while i < n_frames - 1 and k < energy_tol:
                k = k + 1 if remaining[i, f] < frame_thresh else 0
                clear(i, i + 1, f)
                i += 1
            i_end = i - 1 - k
            i, k = i_mid - 1, 0
            while i > 0 and k < energy_tol:
                k = k + 1 if remaining[i, f] < frame_thresh else 0
                clear(i, i + 1, f)
                i -= 1
            i_start = i + 1 + k
            if i_end - i_start <= min_note_len:
                continue
            events.append((int(i_start), int(i_end), int(f + MIDI_OFFSET),
                           float(np.mean(frames[i_start:i_end, f]))))
    return events


def keep_top_line(notes: list[dict]) -> list[dict]:
    """Reduce polyphony to a melody: at any moment keep only the highest sounding note."""
    notes = sorted(notes, key=lambda n: (n["start"], -n["pitch"]))
    melody = []
    for n in notes:
        if melody and n["start"] < melody[-1]["end"] - 0.03:
            prev = melody[-1]
            if n["pitch"] > prev["pitch"]:
                prev["end"] = n["start"]  # a higher note takes over
                if prev["end"] - prev["start"] < 0.05:
                    melody.pop()
                melody.append(dict(n))
            continue
        melody.append(dict(n))
    return melody


def transcribe(y: np.ndarray, sr: int = SR, sensitivity: float = 0.5, min_note_ms: float = 127.7,
               min_pitch: int | None = None, max_pitch: int | None = None, melody_only: bool = False,
               progress=None) -> list[dict]:
    """Notes (pitch, start, end in seconds, velocity) heard in a recording.

    sensitivity 0..1: higher finds quieter notes (and more false ones).
    """
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    out = run_model(y, progress)
    s = float(np.clip(sensitivity, 0.0, 1.0))
    onset_thresh = 0.7 - 0.4 * s      # 0.5 at the default sensitivity
    frame_thresh = 0.45 - 0.3 * s     # 0.3 at the default
    min_len = int(round(min_note_ms / 1000 * SR / FFT_HOP))
    events = decode_notes(out["note"], out["onset"], onset_thresh, frame_thresh, min_len,
                          min_pitch, max_pitch)
    times = frames_to_time(out["note"].shape[0])
    notes = [{"pitch": p, "start": round(float(times[a]), 4), "end": round(float(times[b]), 4),
              "velocity": int(np.clip(round(127 * amp), 30, 127))} for a, b, p, amp in events]
    notes = [n for n in notes if n["end"] > n["start"]]
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    if melody_only:
        notes = keep_top_line(notes)
    return notes
