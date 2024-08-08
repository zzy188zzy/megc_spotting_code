import scipy.fftpack as fftpack
import numpy as np


def temporal_ideal_filter(tensor, low, high, fps, axis=0):

    fft = fftpack.fft(tensor, axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[bound_high:-bound_high] = 0
    iff = fftpack.ifft(fft, axis=axis)

    return np.abs(iff)