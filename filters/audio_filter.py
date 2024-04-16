"""Filtrar audio."""

import matplotlib.pyplot as plt
import numpy as np
from playsound import playsound
from scipy.io import wavfile


def audio_filter():
    """Funcion de filtrado de Audio."""
    playsound("filters/MB_Song.wav")

    sampFreq, sound = wavfile.read("filters/MB_Song.wav")
    print(sound.dtype, sampFreq)

    sound = sound / 2.0**15

    sound = sound[:, 0]

    length_in_s = sound.shape[0] / sampFreq
    print("Audio length", length_in_s)

    plt.plot(sound[:], "r")
    plt.xlabel("Sound signal")
    plt.tight_layout()
    plt.show()

    time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s

    plt.plot(time, sound[:], "r")
    plt.xlabel("time, signal")
    plt.tight_layout()
    plt.show()

    yerr = (
        0.005 * np.sin(2 * np.pi * 6000.0 * time)
        + 0.008 * np.sin(2 * np.pi * 8000.0 * time)
        + 0.006 * np.sin(2 * np.pi * 2500.0 * time)
    )
    signal = sound + yerr

    plt.plot(time[6000:7000], signal[6000:7000])
    plt.xlabel("time s")
    plt.show()

    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1.0 / sampFreq)
    print("Fourier spectrum", fft_spectrum)
    fft_spectrum_abs = np.abs(fft_spectrum)

    plt.plot(freq, fft_spectrum_abs)
    plt.xlabel("frquency, Hz")
    plt.ylabel("Amplitude, units")
    plt.show()

    for i, f in enumerate(freq):
        if f > 5900 and f < 6100:
            fft_spectrum[i] = 0.0

    noiseless_signal = np.fft.irfft(fft_spectrum)

    plt.plot(time, noiseless_signal, "r")
    plt.xlabel("time, signal")
    plt.tight_layout()
    plt.show()

    wavfile.write("Noisy Audio.wav", sampFreq, signal)
    wavfile.write("Noiseless Audio.wav", sampFreq, noiseless_signal)
    playsound("Noisy Audio.wav")
    playsound("Noiseless Audio.wav")


if __name__ == "__main__":
    audio_filter()
