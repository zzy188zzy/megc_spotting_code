import numpy as np 
import scipy.fftpack as fftpack
from PyEMD import EMD


def do_emd(yuan, s, path, xuhao, fs):
    t = np.arange(len(s) / fs)
    s = np.array(s)
    IMF = EMD().emd(s, t)
    N = IMF.shape[0]

    imf_sum = np.zeros(IMF.shape[1])
    imf_sum1 = np.zeros(IMF.shape[1])
    for n, imf in enumerate(IMF):
        if (n != N - 1):
            imf_sum1 = np.add(imf_sum1, imf)
        if (n != 0):
            imf_sum = np.add(imf_sum, imf)

    return imf_sum, imf_sum1