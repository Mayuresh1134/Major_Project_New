"""Theory, synthesis, notation and transcription."""
import threading

import numpy as np
import pytest

from conftest import note_f1, tune_notes
from museassist import notation, synth, theory, transcribe


def test_key_of_c_major_scale():
    weights = np.zeros(12)
    for pc in (0, 2, 4, 5, 7, 9, 11):
        weights[pc] = 1
    weights[0] = weights[7] = 2
    assert theory.estimate_key(weights)["name"] == "C major"


def test_key_of_a_minor_melody():
    notes = [{"pitch": p, "start": i, "end": i + 1} for i, p in enumerate([57, 60, 64, 57, 59, 60, 62, 64, 57, 52, 57])]
    assert theory.estimate_key(theory.note_pitch_class_weights(notes))["name"] == "A minor"


@pytest.mark.parametrize("voice", [v for v in synth.VOICES if v != "auto"])
def test_every_voice_renders(voice):
    notes = [{"pitch": 60 + i, "start": i * 0.25, "end": i * 0.25 + 0.3, "velocity": 90} for i in range(8)]
    audio = synth.render_notes(notes, voice)
    assert np.isfinite(audio).all()
    assert 0.5 < np.max(np.abs(audio)) <= 0.9
    assert abs(len(audio) / synth.SR - (2.05 + 1.5)) < 0.1


@pytest.mark.parametrize("voice", ["piano", "guitar", "flute", "strings", "lead"])
def test_voices_are_in_tune(voice):
    """Rendering then transcribing gives the same pitches back (a vibrato bug failed this)."""
    ref = tune_notes("minuet")
    est = transcribe.transcribe(synth.render_notes(ref, voice))
    assert note_f1(ref, est) > 0.8


def test_transcribe_piano_accuracy():
    ref = tune_notes("ode")
    est = transcribe.transcribe(synth.render_notes(ref, "piano"))
    assert note_f1(ref, est) > 0.9


def test_melody_only_keeps_top_line():
    notes = [{"pitch": 60, "start": 0, "end": 1}, {"pitch": 67, "start": 0, "end": 1},
             {"pitch": 64, "start": 1, "end": 2}]
    melody = transcribe.keep_top_line(notes)
    assert [n["pitch"] for n in melody] == [67, 64]


def test_abc_examples_parse_and_play():
    for key, (title, abc) in notation.EXAMPLES.items():
        score = notation.parse_abc(abc)
        notes = notation.score_to_notes(score)
        assert len(notes) > 20, key
        assert notation.score_title(score) != "Untitled"


def test_bad_abc_is_a_clear_error():
    with pytest.raises(notation.ScoreError):
        notation.parse_abc("   ")


def test_engraving_works_in_a_worker_thread(tmp_path):
    """Verovio could not find its fonts outside the main thread."""
    score = notation.parse_abc(notation.EXAMPLES["twinkle"][1])
    notation.write_musicxml(score, tmp_path / "s.musicxml")
    pages = []
    t = threading.Thread(target=lambda: pages.extend(notation.engrave(tmp_path / "s.musicxml")))
    t.start()
    t.join()
    assert pages and pages[0].lstrip().startswith("<svg") or "<svg" in pages[0][:300]


def test_key_aware_spelling():
    # E minor melody with a D#: should be written D#, not Eb, under a one-sharp key signature
    notes = [{"pitch": p, "start": i * 0.5, "end": i * 0.5 + 0.5} for i, p in enumerate([64, 67, 71, 63, 64, 66, 67, 64])]
    key = theory.estimate_key(theory.note_pitch_class_weights(notes))
    score = notation.notes_to_score(notes, 120, "t", split_hands=False, key=key)
    names = [n.pitch.name for n in score.recurse().notes]
    assert "D#" in names and "E-" not in names


def test_beat_grid_fit_recovers_tempo():
    notes = tune_notes("minuet")  # 120 BPM
    tempo, offset = notation.fit_beat_grid([{**n, "start": n["start"] + 0.137} for n in notes], 116.0)
    assert abs(tempo - 120) < 0.3
    assert abs(offset - 0.137) < 0.01


def test_transpose_score():
    score = notation.parse_abc(notation.EXAMPLES["twinkle"][1])
    up = notation.score_to_notes(notation.transpose_score(score, 2))
    assert up[0]["pitch"] == notation.score_to_notes(score)[0]["pitch"] + 2


def test_transcribed_sheet_has_clean_rhythm():
    """Audio -> notes -> sheet should give the minuet's quarters and eighths, not ties and dots."""
    from museassist import analysis
    ref = tune_notes("minuet")
    y = synth.render_notes(ref, "piano")
    est = transcribe.transcribe(y)
    tempo, offset = notation.fit_beat_grid(est, analysis.tempo_and_beats(y)["bpm"])
    shifted = [{**n, "start": max(n["start"] - offset, 0), "end": n["end"] - offset} for n in est]
    score = notation.notes_to_score(notation.tidy_for_sheet(shifted, tempo), tempo, "t",
                                    split_hands=False, time_signature="3/4")
    lengths = [float(n.quarterLength) for n in score.recurse().notes]
    clean = sum(q in (0.5, 1.0, 2.0, 3.0) for q in lengths) / len(lengths)
    assert abs(tempo - 120) < 0.5
    assert clean > 0.85
