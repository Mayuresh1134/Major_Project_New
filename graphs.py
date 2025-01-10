import librosa
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

script, content_audio_name, style_audio_name, output_audio_name, output_image_name = argv

N_FFT = 2048

def read_audio_spectrum(filename):
    x, fs = librosa.load(filename, duration=58.04)
    S = librosa.stft(x, n_fft=N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))  
    return S, fs

style_audio, style_sr = read_audio_spectrum(style_audio_name)
content_audio, content_sr = read_audio_spectrum(content_audio_name)
output_audio, output_sr = read_audio_spectrum(output_audio_name)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Content')
plt.imshow(content_audio[:500, :500])
plt.subplot(1, 3, 2)
plt.title('Style')
plt.imshow(style_audio[:500, :500])
plt.subplot(1, 3, 3)
plt.title('Result')
plt.imshow(output_audio[:500, :500])

# Save the plot as an image
plt.savefig(output_image_name)
