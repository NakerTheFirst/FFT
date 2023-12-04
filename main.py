from matplotlib import pyplot as plt
import numpy as np
import sys


class FFT:
    def __init__(self):
        self.__data = []
        self.__data_noisy = []
        self.__data_2d = []
        self.__dft_frequencies = []
        self.__amplitudes = []
        self.__amplitudes_log = None
        self.__fft_frequencies = []
        self.__harmonics_indexes = {}
        self.__noise_values = []
        self.__amplitudes_centred = None
        self.__dimensionality = 0
        self.__data_size = 0
        self.__dft_op_counter = 0
        self.__fft_op_counter = 0
        self.__slope = 0
        self.__intercept = 0

    def run(self):

        # with self.__redirect_output_to_file("dane.out"):
        # self.__scrape_data_final()
        self.__scrape_data()
        self.__dimensionality = 2

        if self.__dimensionality == 1:
            # Perform the transformations and round the output
            self.__dft_frequencies = self.__threshold_list(self.__dft(self.__data))
            self.__fft_frequencies = self.__threshold_list(self.__fft(self.__data))
            self.__dft_frequencies = self.__round_list_contents(self.__dft_frequencies)
            self.__fft_frequencies = self.__round_list_contents(self.__fft_frequencies)

            # Inverse transformations
            signal_dft_inverse = self.__idft(self.__dft_frequencies)
            signal_fft_inverse = self.__idft(self.__fft_frequencies)

            self.__data_size = np.size(self.__data)
            self.__calc_amplitudes()
            self.__amplitudes = self.__round_list_contents(self.__amplitudes)
            self.__extract_harmonics(self.__amplitudes)
            self.__noise_values = self.__extract_noise()
            self.__fit_line_to_noise()

            # Plotting
            self.__plot_signal(self.__data)
            self.__plot_amplitudes()
            self.__plot_noise()
            self.__plot_signal(signal_dft_inverse, "Post IDFT signal")
            self.__plot_signal(signal_fft_inverse, "Post IFFT signal")

        if self.__dimensionality == 2:
            self.__fft_frequencies = self.__threshold_list(self.__fft(self.__data))
            self.__fft_frequencies = self.__round_list_contents(self.__fft_frequencies)
            self.__amplitudes = np.abs(self.__fft_frequencies)
            self.__amplitudes_centred = np.fft.fftshift(self.__amplitudes)
            self.__amplitudes_log = np.log1p(self.__amplitudes_centred)

            print(self.__fft_frequencies)

        self.__print_output_data()

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
        self.__data = self.__data_2d

    def __scrape_data_final(self):
        self.__dimensionality = int(input().strip())

        # Read the number of elements
        dimensions = list(map(int, input().strip().split()))

        # Initialize the data container
        if self.__dimensionality == 1:
            N = dimensions[0]
            self.__data = [float(input().strip()) for _ in range(N)]
        elif self.__dimensionality == 2:
            N, M = dimensions
            self.__data = [[float(num) for num in input().strip().split()] for _ in range(N)]
        else:
            raise ValueError("Dimensionality must be 1 or 2")
        print(f"self.__dimensionality: {self.__dimensionality}, self.__data: {self.__data}")

    def __redirect_output_to_file(self, filename):
        class OutputRedirector:
            def __enter__(self_):
                self_.old_stdout = sys.stdout
                sys.stdout = open(filename, "w")
                return self_

            def __exit__(self_, exc_type, exc_val, exc_tb):
                sys.stdout.close()
                sys.stdout = self_.old_stdout

        return OutputRedirector()

    def __dft(self, x):
        N = len(x)
        X = np.zeros(N, dtype=complex)

        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
                self.__dft_op_counter += 1

        return X

    def __fft(self, x):

        if self.__dimensionality == 1:
            N = len(x)
            if N <= 1:
                return x

            # Even/odd element splitting
            even = self.__fft(x[0::2])
            odd = self.__fft(x[1::2])

            # Combine
            combined = [0] * N
            for k in range(N // 2):
                twiddle_factor = np.exp(-2j * np.pi * k / N) * odd[k]
                combined[k] = even[k] + twiddle_factor
                combined[k + N // 2] = even[k] - twiddle_factor

                # Count 2 operations (1 multiplication, 1 addition)
                self.__fft_op_counter += 2
            return combined

        elif self.__dimensionality == 2:
            return np.fft.fft2(x)

        else:
            raise ValueError("Wrong dimension!")

    def __idft(self, x):
        N = len(x)
        X = np.zeros(N, dtype=complex)

        for n in range(N):
            for k in range(N):
                X[n] += x[k] * np.exp(2j * np.pi * k * n / N)
            X[n] /= N

        return X

    def __ifft(self, x):
        N = len(x)
        if N <= 1:
            return x

        # Split the signal into even and odd indexed elements
        even = self.__ifft(x[0::2])
        odd = self.__ifft(x[1::2])

        # Combine
        combined = [0] * N
        for k in range(N // 2):
            twiddle_factor = np.exp(2j * np.pi * k / N) * odd[k]
            combined[k] = even[k] + twiddle_factor
            combined[k + N // 2] = even[k] - twiddle_factor

        # Normalize the results by the number of samples
        combined = [elem / N for elem in combined]

        return combined

    def __calc_amplitudes(self):
        for x in range(len(self.__data)):
            amplitude = np.abs(self.__fft_frequencies[x])
            self.__amplitudes.append(amplitude)

    def __extract_harmonics(self, data, threshold=5):
        # If data is a 1D list
        if all(isinstance(i, (int, float, complex, np.int32, np.float64)) for i in data):
            for index, value in enumerate(data):
                if abs(value) > threshold and abs(value) not in self.__harmonics_indexes.values():
                    self.__harmonics_indexes[index] = value
        else:
            raise ValueError("Data must be a 1D list of numbers")

    def __extract_noise(self, epsilon=5):
        noise = []
        for x in range(int(self.__data_size/2)):
            if self.__amplitudes[x] <= epsilon:
                noise.append(self.__amplitudes[x])
        return noise

    def __threshold_list(self, input_list, epsilon=0.0005) -> list:
        """If list values are lower than epsilon, set them to 0"""
        def threshold_value(value):
            return 0 if abs(value) < epsilon else value

        # 1D list
        if all(isinstance(i, (int, float, complex)) for i in input_list):
            return [threshold_value(x) for x in input_list]

        # 2D list
        elif all(isinstance(row, np.ndarray) for row in input_list):
            return [[threshold_value(x) for x in row] for row in input_list]

        else:
            raise ValueError("Input must be a 1D or 2D list of numbers")

    def __round_list_contents(self, input_list):
        """Rounds the contents of a list to two decimal points."""
        return [np.round(num, 2) for num in input_list]

    def __fit_line_to_noise(self):
        N = len(self.__noise_values)
        psd_values = [(abs(x) ** 2) / N for x in self.__noise_values]

        # Convert PSD to decibels
        psd_dB = [10 * np.log10(psd) if psd > 0 else 0 for psd in psd_values]

        # Log-frequency scale for fitting
        log_freq = np.log10(np.arange(1, N + 1))

        self.__slope, self.__intercept = np.polyfit(log_freq, psd_dB, 1)

        # Create a function based on the fit to calculate the line values
        self.__lin_fun = lambda f: self.__slope * f + self.__intercept

    def __print_output_data(self):
        # print(f"Number of dominant DFT operations: {self.__dft_op_counter}")
        # print(f"Number of dominant FFT operations: {self.__fft_op_counter}")
        # print(f"Post-FFT frequencies: \n{self.__fft_frequencies}")
        print(f"Harmonics: \n{self.__harmonics_indexes}")
        print(f"Amplitudes: \n{self.__amplitudes}")
        print(f"Noise: \n{self.__noise_values}")
        # print(f"Slope: \n{self.__slope}")
        # print(f"Intercept: \n{self.__intercept}")

    def __plot_amplitudes(self):
        sample_num = range(len(self.__amplitudes))

        plt.figure(figsize=(7.5, 5))

        plt.scatter(sample_num, self.__amplitudes, s=15)
        plt.title("FFT Amplitudes")
        plt.xlabel("Sample number", fontsize=12)
        plt.ylabel("Amplitudes", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Internal margins
        plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)

        plt.grid(True)
        plt.show()

    def __plot_signal(self, signal, title="Discretised Signal in Time Domain"):
        plt.figure(figsize=(7.5, 5))

        plt.scatter(range(len(signal)), signal, s=15)
        plt.title(title)
        plt.xlabel("Sample number", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Internal margins
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)

        plt.grid(True)
        plt.show()

    def __plot_noise(self):
        N = len(self.__noise_values) * 2
        psd_values = [(abs(x) ** 2) / N for x in self.__noise_values]
        psd_dB = [10 * np.log10(psd) if psd > 0 else 0 for psd in psd_values]

        # Create a frequency array that matches the length of psd_dB
        freq = np.arange(1, N // 2 + 1)

        plt.figure(figsize=(7.5, 5))

        plt.semilogx(freq, psd_dB, label="WGM Szumu")

        # Generate the fit line values
        fit_line = [self.__lin_fun(np.log10(f)) for f in freq]
        plt.semilogx(freq, fit_line, color="orange", label="Linia dopasowania")

        plt.xlabel("Znormalizowana częstotliwość", fontsize=12)
        plt.ylabel("Widmowa Gęstość Mocy (dB)", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
        plt.grid(True)
        plt.legend()
        plt.show()

    # TODO: Remove all the figure titles
    # TODO: Rename all the figure labels to Polish


def main():

    f = FFT()
    f.run()

    return 0


if __name__ == "__main__":
    main()
