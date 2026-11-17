"""Audio style transfer on spectrograms (after Ulyanov & Lebedev, 2016).

A single 1-D convolution with fixed random weights turns a log-magnitude spectrogram
into "texture" features. The output starts as the content clip and is adjusted with
Adam so its feature correlations (Gram matrix) match the style clip's, while a content
term keeps its features near the original. Phase is rebuilt with Griffin-Lim.

The first version ran this with PyTorch for 10 steps and clamped the spectrogram to
0..1 (a step copied from image style transfer), which threw away most of the sound.
This version computes the gradients directly with numpy.
"""
from __future__ import annotations

import librosa
import numpy as np

SR = 22050
N_FFT = 2048
HOP = 512


def log_spectrogram(y: np.ndarray):
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    return np.log1p(np.abs(S)).astype(np.float32), np.angle(S)


class TextureNet:
    """Conv1d over time (kernel 3, 'same' padding) from 1025 frequency bins to `filters` channels."""

    def __init__(self, n_bins: int = N_FFT // 2 + 1, filters: int = 1024, seed: int = 0):
        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / (3 * n_bins))
        self.W = (rng.standard_normal((filters, 3 * n_bins)) * std).astype(np.float32)
        self.n_bins = n_bins

    @staticmethod
    def unfold(x):
        """(C, T) -> (3C, T): each frame with its left and right neighbours."""
        pad = np.pad(x, ((0, 0), (1, 1)))
        return np.concatenate([pad[:, :-2], pad[:, 1:-1], pad[:, 2:]], axis=0)

    def fold_grad(self, g):
        """Adjoint of unfold: (3C, T) gradient -> (C, T)."""
        C = self.n_bins
        a, b, c = g[:C], g[C:2 * C], g[2 * C:]
        out = b.copy()
        out[:, :-1] += a[:, 1:]   # x[t] sat in the "left neighbour" block at frame t + 1
        out[:, 1:] += c[:, :-1]   # and in the "right neighbour" block at frame t - 1
        return out

    def features(self, x):
        return np.maximum(self.W @ self.unfold(x), 0)  # ReLU

    @staticmethod
    def gram(F):
        return (F @ F.T) / F.shape[1]


def losses_and_grad(x, net: TextureNet, style_gram, content_feats, content_weight: float):
    """Style loss (Gram mismatch) + content_weight * content loss, and d(loss)/dx."""
    pre = net.W @ net.unfold(x)
    F = np.maximum(pre, 0)
    T = F.shape[1]
    G = (F @ F.T) / T
    diff = G - style_gram
    style_loss = float(np.mean(diff ** 2))
    K = G.shape[0]
    dF = (4.0 / (K * K * T)) * (diff @ F)
    content_loss = 0.0
    if content_weight > 0:
        cdiff = F - content_feats
        content_loss = float(np.mean(cdiff ** 2))
        dF += content_weight * (2.0 / cdiff.size) * cdiff
    dpre = dF * (pre > 0)
    dx = net.fold_grad(net.W.T @ dpre)
    return style_loss, content_loss, dx


def fit_length(a: np.ndarray, n: int) -> np.ndarray:
    """Loop or trim audio to exactly n samples."""
    if len(a) >= n:
        return a[:n]
    reps = int(np.ceil(n / max(len(a), 1)))
    return np.tile(a, reps)[:n]


def transfer(content: np.ndarray, style: np.ndarray, seconds: float = 15.0, steps: int = 150,
             strength: float = 0.7, filters: int = 1024, seed: int = 0, progress=None) -> dict:
    """Restyle `content` with the texture of `style` (both mono at 22,050 Hz).

    strength 0..1: 0 keeps the content as is, 1 lets the style take over.
    """
    n = int(min(seconds, len(content) / SR) * SR)
    if n < SR:
        raise ValueError("The content clip must be at least 1 second long.")
    content = content[:n]
    style = fit_length(style, n)  # the style clip may be shorter or longer
    C, phase = log_spectrogram(content)
    S, _ = log_spectrogram(style)
    net = TextureNet(C.shape[0], filters, seed)
    style_gram = net.gram(net.features(S))
    content_feats = net.features(C)
    strength = float(np.clip(strength, 0.0, 1.0))
    # Put the two losses on a common scale: the style loss of the untouched content clip,
    # and the content loss of a clip whose features were the style clip's.
    s0 = float(np.mean((net.gram(content_feats) - style_gram) ** 2))
    c_ref = float(np.mean((net.features(S) - content_feats) ** 2))
    content_weight = (1 - strength) / max(strength, 1e-3) * s0 / max(c_ref, 1e-12)

    x = C.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    lr, b1, b2, eps = 0.03, 0.9, 0.999, 1e-8
    history = []
    for t in range(1, steps + 1):
        sl, cl, g = losses_and_grad(x, net, style_gram, content_feats, content_weight)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        np.maximum(x, 0, out=x)  # log1p of a magnitude is never negative
        if t == 1 or t % 10 == 0 or t == steps:
            history.append({"step": t, "style_loss": sl, "content_loss": cl})
        if progress:
            progress(0.85 * t / steps)
    magnitude = np.expm1(x)
    y = librosa.griffinlim(magnitude, n_iter=64, hop_length=HOP, n_fft=N_FFT, momentum=0.99,
                           init=None, length=n)
    if progress:
        progress(1.0)
    peak = np.max(np.abs(y)) or 1.0
    return {"audio": (y / peak * 0.9).astype(np.float32), "content": content, "style": style,
            "spectrograms": (C, S, x), "history": history,
            "style_change": round(1 - history[-1]["style_loss"] / history[0]["style_loss"], 3)}
