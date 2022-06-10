from matplotlib import pyplot as plt


class ComputePsd:
    def __init__(self, signal):
        self.signal = signal

    def compute(self):
        """
        | compute psd
        :return:
        """
        from scipy import signal
        freqs, times, spectrogram = signal.spectrogram(self.signal)

        plt.figure(figsize=(5, 4))
        plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
        plt.title('Spectrogram')
        plt.ylabel('Frequency band')
        plt.xlabel('Time window')
        plt.tight_layout()

        freqs, psd = signal.welch(self.signal)

        plt.figure(figsize=(5, 4))
        plt.semilogx(freqs, psd)
        plt.title('PSD: power spectral density')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.tight_layout()

        plt.show()

