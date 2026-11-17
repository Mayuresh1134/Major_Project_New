"""The three trained models and the analysis and style-transfer maths."""
import numpy as np
import pytest

from conftest import tune_notes
from museassist import analysis, generate, genre, style_transfer, synth, theory


# Melody model -------------------------------------------------------------------
def test_generation_is_repeatable_with_a_seed():
    seed = tune_notes("ode")
    a = generate.continue_melody(seed, 20, rng_seed=5)["notes"]
    b = generate.continue_melody(seed, 20, rng_seed=5)["notes"]
    assert a == b and len(a) == 20


def test_stay_in_key_and_leap_limit():
    seed = tune_notes("minuet")
    out = generate.continue_melody(seed, 64, temperature=1.5, stay_in_key=True, max_leap=9, rng_seed=1)
    scale = theory.scale_pitch_classes(out["key"]["tonic"], out["key"]["mode"])
    pitches = [seed[-1]["pitch"]] + [n["pitch"] for n in out["notes"]]
    assert all(p % 12 in scale for p in pitches[1:])
    assert max(abs(b - a) for a, b in zip(pitches, pitches[1:])) <= 9


def test_continuation_follows_the_seed_in_time():
    seed = tune_notes("twinkle")
    notes = generate.continue_melody(seed, 30, rng_seed=2)["notes"]
    starts = [n["start"] for n in notes]
    assert starts[0] >= seed[-1]["start"]
    assert starts == sorted(starts)
    assert all(n["end"] > n["start"] for n in notes)


def test_lstm_prediction_shapes():
    model = generate.get_model()
    logits, step, duration = model.predict(generate.seed_sequence(tune_notes("ode")))
    assert logits.shape == (128,) and np.isfinite(logits).all()
    assert np.isfinite(step) and np.isfinite(duration)


# Genre model ----------------------------------------------------------------------
def test_genre_model_loads():
    m = genre.get_model()
    assert m.classes == ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    p = m.predict_proba(np.tile(m.mean, (3, 1)))
    assert p.shape == (3, 10) and np.allclose(p.sum(axis=1), 1)


def test_features_match_training_columns():
    y = synth.render_notes(tune_notes("minuet"), "piano")[: genre.SEGMENT]
    assert genre.segment_features(y).shape == (len(genre.FEATURES),) == (57,)


def test_classify_returns_sorted_probabilities():
    y = synth.render_notes(tune_notes("minuet"), "piano")
    r = genre.classify(y)
    probs = [g["probability"] for g in r["genres"]]
    assert probs == sorted(probs, reverse=True) and abs(sum(probs) - 1) < 0.01
    assert r["top"] == r["genres"][0]["genre"]


def test_classify_rejects_tiny_clips():
    with pytest.raises(ValueError):
        genre.classify(np.zeros(1000, dtype=np.float32))


def test_genre_report_numbers():
    report = genre.model_report()
    assert report["songs"] == 1000
    assert report["comparison"][-1]["accuracy"] > report["comparison"][0]["accuracy"]


# Analysis -----------------------------------------------------------------------------
def test_tempo_key_and_chords_of_a_progression():
    chords = {"C": [60, 64, 67, 48], "G": [59, 62, 67, 43], "Am": [60, 64, 69, 45], "F": [60, 65, 69, 41]}
    notes, t, beat = [], 0.0, 0.6  # 100 BPM
    for _ in range(3):
        for c in ("C", "G", "Am", "F"):
            for _b in range(4):
                notes += [{"pitch": p, "start": t, "end": t + beat * 0.9, "velocity": 80} for p in chords[c]]
                t += beat
    r = analysis.analyse(synth.render_notes(notes, "piano"))
    assert abs(r["tempo"]["bpm"] - 100) < 3
    assert r["key"]["name"] == "C major"
    sequence = [c["chord"] for c in r["chords"] if c["end"] - c["start"] > 1]
    assert sequence[:4] == ["C", "G", "Am", "F"]


# Style transfer -------------------------------------------------------------------------
def test_style_gradient_matches_numerical():
    rng = np.random.default_rng(1)
    net = style_transfer.TextureNet(n_bins=7, filters=5, seed=3)
    net.W = net.W.astype(np.float64)
    x, S, C = rng.random((7, 6)), rng.random((7, 6)), rng.random((7, 6))
    sg, cf = net.gram(net.features(S)), net.features(C)

    def loss(v):
        s, c, _ = style_transfer.losses_and_grad(v, net, sg, cf, 0.7)
        return s + 0.7 * c
    _, _, g = style_transfer.losses_and_grad(x, net, sg, cf, 0.7)
    num = np.zeros_like(x)
    for i in range(7):
        for j in range(6):
            d = np.zeros_like(x)
            d[i, j] = 1e-6
            num[i, j] = (loss(x + d) - loss(x - d)) / 2e-6
    assert np.max(np.abs(num - g)) / np.max(np.abs(num)) < 1e-5


def test_style_transfer_reduces_style_loss_with_unequal_lengths():
    content = synth.render_notes(tune_notes("minuet"), "piano")
    style = synth.render_notes(tune_notes("twinkle"), "flute")[: synth.SR * 2]  # much shorter
    out = style_transfer.transfer(content, style, seconds=3, steps=25, strength=1.0, filters=256)
    assert len(out["audio"]) == 3 * synth.SR
    assert out["history"][-1]["style_loss"] < 0.5 * out["history"][0]["style_loss"]
    assert np.isfinite(out["audio"]).all()


def test_key_tie_is_settled_by_the_final_chord():
    key = {"tonic": 7, "mode": "major", "name": "G major", "score": 0.80,
           "runner_up": "C major", "runner_up_score": 0.78}
    chords = [{"start": 0, "end": 2, "chord": "G"}, {"start": 2, "end": 4, "chord": "C"}]
    assert analysis.settle_key(key, chords)["name"] == "C major"
    clear = {**key, "runner_up_score": 0.5}
    assert analysis.settle_key(clear, chords)["name"] == "G major"


def test_tempo_estimate_avoids_two_thirds_errors():
    """The beat tracker alone often reported about 2/3 of these tempos."""
    for bpm in (128.0, 141.0):
        beat = 60 / bpm
        notes = []
        for i in range(48):
            t = i * beat
            notes += [{"pitch": 36 if i % 2 == 0 else 38, "start": t, "end": t + 0.1, "velocity": 100, "is_drum": True},
                      {"pitch": 42, "start": t + beat / 2, "end": t + beat / 2 + 0.05, "velocity": 80, "is_drum": True},
                      {"pitch": 48, "start": t, "end": t + beat * 0.9, "velocity": 80, "program": 33}]
        found = analysis.tempo_and_beats(synth.render_notes(notes, "auto"))["bpm"]
        # double or half the tempo is a matter of taste; 2/3 or 3/2 is simply wrong
        assert any(abs(found * m - bpm) / bpm < 0.04 for m in (1, 2, 0.5)), (bpm, found)
