import numpy as np


class FFT:
    def __init__(self):
        self.__data = []
        self.__data_noisy = []
        self.__data_2d = []

    def run(self):
        self.__scrape_data()

        data_dft = self.__dft(self.__data)
        print(f"data_dft: {data_dft}")

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

    def __dft(self, x):
        N = len(x)
        X = np.zeros(N, dtype=complex)

        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

        return X

    def __count_operations(self):
        pass

    def get_data(self):
        return self.__data

    def get_data_noisy(self):
        return self.__data_noisy

    def get_data_2d(self):
        return self.__data_2d


def main():

    f = FFT()
    f.run()

    print(f"Data: {f.get_data()}")
    print(f"Data noise: {f.get_data_noisy()}")
    print(f"Data 2D: {f.get_data_2d()}")

    return 0


if __name__ == "__main__":
    main()
