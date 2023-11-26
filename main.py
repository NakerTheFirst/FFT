from matplotlib import pyplot as plt
import numpy as np


class FFT:
    def __init__(self):
        # TODO: Delete the other lists after refactoring the io
        self.__data = []
        self.__data_noisy = []
        self.__data_2d = []
        self.__dft_frequencies = []
        self.__amplitudes = []
        self.__fft_frequencies = []

    def run(self):
        self.__scrape_data()
        self.__dft_frequencies = self.__dft(self.__data)
        self.__fft_frequencies = self.__fft(self.__data)
        self.__calc_amplitudes(self.__data)

    def __scrape_data(self):
        files = ["dane_02.in", "dane_02_a.in", "dane2_02.in"]
        lists = [self.__data, self.__data_noisy, self.__data_2d]

        for file, data_list in zip(files, lists):
            with open(file) as f:
                lines = f.read().splitlines()

            if file == "dane2_02.in":
                for line in lines[2:]:
                    row_data = list(map(float, line.split()))
                    data_list.append(row_data)
                continue

            for line in lines[2:]:
                try:
                    data_list.append(float(line))
                except ValueError:
                    print(f"Could not convert {line} to float")

    def __dft (self, x):
        N = len(x)
        X = np.zeros(N, dtype=complex)

        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

        return X

    def __fft(self, x):
        N = len(x)
        if N <= 1:
            return x

        # Even/odd element splitting
        even = self.__fft(x[0::2])
        odd = self.__fft(x[1::2])

        # Combine
        T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
        return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

    def __count_operations(self):
        pass

    def __calc_amplitudes(self, data):
        for x in range(len(data)):
            amplitude = np.abs(self.__fft_frequencies[x])
            self.__amplitudes.append(amplitude)

    def plot_amplitudes(self):
        sample_num = range(len(self.__amplitudes))
        plt.plot(sample_num, self.__amplitudes)
        plt.title("FFT Amplitudes")
        plt.xlabel("Sample number")
        plt.ylabel("Amplitudes")
        plt.grid(True)
        plt.show()

    # TODO: This doesn't work, unsure what to do - move on, get back later
    def plot_magnitude_spectrum(self):
        plt.scatter(self.__fft_frequencies, self.__amplitudes)
        plt.title("FFT Magnitude Spectrum")
        plt.xlabel("Frequencies")
        plt.ylabel("Amplitudes")
        plt.grid(True)
        plt.show()

    def plot_signal(self):
        plt.scatter(range(len(self.__data)), self.__data, s=15)
        plt.title("Discrete Signal in Time Domain")
        plt.xlabel("Sample number")
        plt.ylabel("Values")
        plt.grid(True)
        plt.show()

    def get_dft_frequencies(self):
        return self.__dft_frequencies

    def get_fft_frequencies(self):
        return self.__fft_frequencies

    def get_amplitudes(self):
        return self.__amplitudes


def main():

    f = FFT()
    f.run()
    f.plot_amplitudes()
    # f.plot_magnitude_spectrum()
    f.plot_signal()

    return 0


if __name__ == "__main__":
    main()
