class FFT:
    def __init__(self, signal):
        self.__signal = signal
        self.__data = []
        self.__data_noisy = []
        self.__data_2d = []

    def scrape_data(self):
        files = ["dane_02.in", "dane_02_a.in", "dane2_02.in"]
        lists = [self.__data, self.__data_noisy, self.__data_2d]

        for file, data_list in zip(files, lists):
            with open(file) as f:
                lines = f.read().splitlines()

            if file == "dane2_02.in":
                for line in lines[2:]:
                    row_data = list(map(float, line.split()))
                    data_list.append(row_data)
            else:
                for line in lines:
                    try:
                        data_list.append(float(line))
                    except ValueError:
                        print(f"Could not convert {line} to float")

    def get_data(self):
        return self.__data

    def get_data_noisy(self):
        return self.__data_noisy

    def get_data_2d(self):
        return self.__data_2d


def main():

    data = []
    f = FFT([1, 2, 3])
    f.scrape_data()

    print(f"Data: {f.get_data()}")
    print(f"Data noise: {f.get_data_noisy()}")
    print(f"Data 2D: {f.get_data_2d()}")

    return 0


if __name__ == "__main__":
    main()
