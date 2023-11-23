class FFT:
    def __init__(self, signal):
        self.__signal = signal
        self.__data = []
        self.__data_noise = []
        self.__data_2d = []

    def scrape_data(self):
        files = ["dane_02.in", "dane_02_a.in", "dane2_02.in"]
        lists = [self.__data, self.__data_noise, self.__data_2d]

        for j in files:
            with open(j) as f:
                lines = f.read().splitlines()

        for j in range(len(lines)):
            self.__data.append(float(lines[j]))


def main():

    data = []
    f = FFT([1, 2, 3])
    f.scrape_data()

    return 0


if __name__ == "__main__":
    main()
