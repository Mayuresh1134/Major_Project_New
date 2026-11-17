"""Turn notes into audio.

A small built-in synthesizer renders every voice with numpy, so the app needs no
SoundFont or external program. If the `fluidsynth` program is installed and a `.sf2`
SoundFont is placed in `soundfonts/`, it is used instead for General MIDI sound.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import fftconvolve, lfilter

SR = 22050
MAX_SECONDS = 600  # never render more than 10 minutes

# key -> label shown in the app, General MIDI program used when writing MIDI
VOICES = {
    "auto": ("As written", None),
    "piano": ("Piano", 0),
    "epiano": ("Electric piano", 4),
    "organ": ("Organ", 19),
    "strings": ("Strings", 48),
    "flute": ("Flute", 73),
    "guitar": ("Nylon guitar", 24),
    "bass": ("Bass", 33),
    "marimba": ("Marimba", 12),
    "lead": ("Synth lead", 80),
}


def voice_for_program(program: int) -> str:
    """Map a General MIDI program number to the nearest built-in voice."""
    p = int(program)
    if 4 <= p <= 5:
        return "epiano"
    if p <= 7:
        return "piano"
    if p <= 15:
        return "marimba"
    if p <= 23:
        return "organ"
    if p <= 31:
        return "guitar"
    if p <= 39:
        return "bass"
    if p <= 55 or 88 <= p <= 95:
        return "strings"
    if p <= 71 or 80 <= p <= 87:
        return "lead"
    if p <= 79:
        return "flute"
    return "piano"


def _hz(pitch: float) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def _additive(f0, t, amps, decays=None, inharm=0.0, vib=0.0, vib_rate=5.5, vib_delay=0.25):
    """Sum of partials. `decays` are per-partial time constants in seconds (None = sustained)."""
    out = np.zeros_like(t)
    if vib:
        ramp = np.clip((t - vib_delay) / 0.4, 0, 1)
        # frequency wobbles by +-vib Hz; this is its integral, as phase
        phase_mod = -vib * ramp * np.cos(2 * np.pi * vib_rate * t) / vib_rate
    else:
        phase_mod = 0.0
    nyq = SR / 2
    for k, a in enumerate(amps, start=1):
        fk = k * f0 * np.sqrt(1 + inharm * k * k)
        if fk >= nyq * 0.95 or a == 0:
            continue
        partial = a * np.sin(2 * np.pi * fk * t + k * phase_mod)
        if decays is not None:
            partial *= np.exp(-t / decays[k - 1])
        out += partial
    return out


@lru_cache(maxsize=512)
def _tone(voice: str, pitch: int, seconds: float) -> np.ndarray:
    """A note of `voice` at `pitch`, `seconds` long, before its note-off release."""
    n = max(int(seconds * SR), 1)
    t = np.arange(n) / SR
    f0 = _hz(pitch)
    rng = np.random.default_rng(pitch)
    register = 2 ** (-(pitch - 60) / 24)  # low notes ring longer

    if voice == "piano":
        tau = 2.6 * register
        amps = [1 / k ** 1.25 for k in range(1, 13)]
        decays = [tau / (1 + 0.55 * (k - 1)) for k in range(1, 13)]
        y = _additive(f0, t, amps, decays, inharm=0.00035)
        y += 0.15 * rng.standard_normal(n) * np.exp(-t / 0.006)  # hammer
        attack = 0.004
    elif voice == "epiano":
        index = 2.2 * np.exp(-t / (0.35 * register))
        y = np.sin(2 * np.pi * f0 * t + index * np.sin(2 * np.pi * f0 * t))
        y *= np.exp(-t / (2.2 * register))
        y += 0.25 * np.sin(2 * np.pi * 14 * f0 * t) * np.exp(-t / 0.02)  # tine click
        attack = 0.003
    elif voice == "organ":
        amps = [1.0, 0.75, 0.0, 0.55, 0.0, 0.35, 0.0, 0.3]
        y = _additive(f0, t, amps)
        attack = 0.012
    elif voice == "strings":
        amps = [1 / k for k in range(1, 15)]
        y = _additive(f0, t, amps, vib=0.004 * f0, vib_rate=5.2)
        y += 0.6 * _additive(f0 * 1.003, t, amps[:8])
        attack = 0.14
    elif voice == "flute":
        y = _additive(f0, t, [1.0, 0.35, 0.12, 0.05], vib=0.005 * f0, vib_rate=5.0, vib_delay=0.2)
        breath = lfilter([0.08], [1, -0.9], rng.standard_normal(n))
        y += breath * (0.25 + 0.75 * np.exp(-t / 0.05))
        attack = 0.06
    elif voice == "guitar":
        period = SR / f0
        N = int(period)
        frac = period - N
        burst = np.zeros(n)
        burst[: N + 1] = rng.uniform(-1, 1, min(N + 1, n))
        g = 0.996 ** (1 + (pitch - 40) / 24)
        a = np.zeros(N + 2)
        a[0] = 1
        a[N] = -g * (1 - frac)
        a[N + 1] = -g * frac
        y = lfilter([1.0], a, burst)
        y = lfilter([0.5, 0.5], [1.0], y)
        attack = 0.002
    elif voice == "bass":
        amps = [1.0, 0.5, 0.25, 0.12, 0.06]
        decays = [1.8 * register / (1 + 0.4 * k) for k in range(5)]
        y = _additive(f0, t, amps, decays)
        attack = 0.006
    elif voice == "marimba":
        y = _additive(f0, t, [1.0, 0.0, 0.0, 0.25], [0.5 * register] * 3 + [0.06], inharm=0.0)
        y += 0.12 * np.sin(2 * np.pi * 9.2 * f0 * t) * np.exp(-t / 0.015)
        attack = 0.002
    elif voice == "lead":
        amps = [1 / k if k % 2 else 0.0 for k in range(1, 16)]
        y = _additive(f0, t, amps, vib=0.003 * f0, vib_rate=6.0)
        y *= 0.75 + 0.25 * np.exp(-t / 0.3)
        attack = 0.01
    else:
        raise ValueError(f"Unknown voice: {voice}")

    na = max(int(attack * SR), 1)
    y[:na] *= np.linspace(0, 1, na)
    peak = np.max(np.abs(y)) or 1.0
    return (y / peak).astype(np.float32)


_RELEASE = {"piano": 0.18, "epiano": 0.2, "organ": 0.06, "strings": 0.3, "flute": 0.12,
            "guitar": 0.15, "bass": 0.12, "marimba": 0.25, "lead": 0.08}


def _drum(pitch: int, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    rng = np.random.default_rng(pitch)
    if pitch in (35, 36):  # kick
        f = 50 + 90 * np.exp(-t / 0.03)
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18)).astype(np.float32)
    noise = rng.standard_normal(n)
    if pitch in (37, 38, 39, 40):  # snare / clap
        return (0.6 * noise * np.exp(-t / 0.09) + 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)).astype(np.float32)
    hp = lfilter([1, -1], [1, -0.7], noise)  # cymbals and others
    tau = 0.25 if pitch in (46, 49, 51, 52, 55, 57, 59) else 0.04
    return (0.4 * hp * np.exp(-t / tau)).astype(np.float32)


def _reverb_ir(seconds=0.9):
    rng = np.random.default_rng(7)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / (seconds / 5))
    ir = lfilter([1.0], [1, -0.6], ir)  # darker tail
    ir[0] = 0
    return (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)


_IR = None


def render_notes(notes, voice: str = "piano", reverb: float = 0.18) -> np.ndarray:
    """Render note dicts (pitch, start, end, velocity, optional program/is_drum) to mono audio."""
    global _IR
    if not notes:
        return np.zeros(SR // 2, dtype=np.float32)
    end = min(max(float(n["end"]) for n in notes) + 1.5, MAX_SECONDS)
    out = np.zeros(int(end * SR) + 1, dtype=np.float32)
    for n in notes:
        start = float(n["start"])
        if start >= end:
            continue
        s0 = int(start * SR)
        vel = float(n.get("velocity", 90)) / 127
        gain = 0.25 + 0.75 * vel
        if n.get("is_drum"):
            seg = _drum(int(n["pitch"]), int(0.6 * SR))
            seg_len = min(len(seg), len(out) - s0)
            out[s0:s0 + seg_len] += 0.5 * gain * seg[:seg_len]
            continue
        v = voice if voice != "auto" else voice_for_program(n.get("program", 0))
        pitch = int(np.clip(n["pitch"], 12, 115))
        dur = max(float(n["end"]) - start, 0.03)
        rel = _RELEASE[v]
        # cache tones in 0.5 s steps so repeated notes reuse work
        held = min(np.ceil(dur * 2) / 2, 12.0)
        tone = _tone(v, pitch, held + rel)
        nd = min(int(dur * SR), len(tone))
        nr = int(rel * SR)
        seg = tone[: nd + nr].copy()
        env = np.ones(len(seg), dtype=np.float32)
        env[nd:] = np.linspace(1, 0, len(seg) - nd) ** 2
        seg *= env
        seg_len = min(len(seg), len(out) - s0)
        out[s0:s0 + seg_len] += gain * seg[:seg_len] * 0.35
    if reverb > 0:
        if _IR is None:
            _IR = _reverb_ir()
        wet = fftconvolve(out, _IR)[: len(out)]
        out = (1 - reverb) * out + reverb * wet
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.89
    return out.astype(np.float32)


def notes_from_midi(pm: pretty_midi.PrettyMIDI) -> list[dict]:
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append({"pitch": n.pitch, "start": n.start, "end": n.end, "velocity": n.velocity,
                          "program": inst.program, "is_drum": inst.is_drum})
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def find_soundfont() -> Path | None:
    folder = Path(os.environ.get("MUSE_SOUNDFONTS", "soundfonts"))
    if not folder.is_dir():
        return None
    fonts = sorted(folder.glob("*.sf2"))
    return fonts[0] if fonts else None


def fluidsynth_available() -> bool:
    return shutil.which("fluidsynth") is not None and find_soundfont() is not None


def render_midi_file(midi_path: str, wav_path: str, voice: str = "auto") -> str:
    """Render a MIDI file to WAV. Returns which engine was used."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    if fluidsynth_available():
        if voice != "auto":
            program = VOICES[voice][1]
            for inst in pm.instruments:
                if not inst.is_drum:
                    inst.program = program
        with tempfile.TemporaryDirectory() as tmp:
            tmp_mid = os.path.join(tmp, "in.mid")
            pm.write(tmp_mid)
            try:
                subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.8", "-F", wav_path, "-r", str(SR),
                                str(find_soundfont()), tmp_mid], check=True, capture_output=True, timeout=300)
                return "fluidsynth"
            except (subprocess.SubprocessError, OSError):
                pass  # fall back to the built-in synthesizer
    audio = render_notes(notes_from_midi(pm), voice)
    sf.write(wav_path, audio, SR, subtype="PCM_16")
    return "built-in"


def write_wav(path, audio, sr=SR):
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")
