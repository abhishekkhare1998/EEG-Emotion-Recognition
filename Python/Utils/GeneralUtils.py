import os
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plotter

class GeneralUtils:
    def __init__(self):
        pass

    def plot_signal(self, signal):
        """
        |plot the given signal
        :param signal: 1D input signal
        :return: None
        """
        plt.plot(signal, color="r")
        plt.show()

    def perform_fft(self, signal):
        """
        | performs FFT for input signal
        :param signal: 1D signal input
        :return: None
        """

        sampling_freq = 128

        fourierTransform = np.fft.fft(signal) / len(signal)  # Normalize amplitude
        fourierTransform = fourierTransform[range(int(len(signal) / 2))]  # Exclude sampling frequency
        tpCount = len(signal)
        values = np.arange(int(tpCount / 2))
        timePeriod = tpCount / sampling_freq
        frequencies = values / timePeriod

        # Frequency domain representation
        figure, axis = plt.subplots()

        axis.set_title("FFT of given signal")
        axis.plot(frequencies, abs(fourierTransform))

        axis.set_xlabel('Frequency')

        axis.set_ylabel('Amplitude')



        """
        
        yf = fft(signal)
        sample_rate = 128
        xf = fftfreq(len(signal), 1 / sample_rate)

        plt.plot(xf, np.abs(yf))
        plt.show()
        """
        plotter.show()

