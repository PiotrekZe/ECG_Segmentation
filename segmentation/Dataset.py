import os
from sklearn.model_selection import train_test_split
import wfdb
import numpy as np

symbols_list = ['j', 'x', 'V', '/', '[', 'R', 'S', 'a', '!', 'N', 'f', 'L', 'Q', 'J', 'E', ']', 'e', '~', '+', '"', '|',
                'A', 'F']
sybols_dict = dict(zip(symbols_list, range(len(symbols_list))))


class Dataset:
    def __init__(self, path):
        self.path = path

    def segment_dataset(self, data):
        # tutaj podzielić na 360 każdy i zrobić drugą wersję z tym dzieleniem na fixed size
        n = 1803  # for 360
        # n = 6490 #for 100
        n = 360  # for 1800 (5 sec)
        n = 600  # for 1080 (3 sec)
        new_data = []
        for i in range(n):
            new_data.append(data[i * 1080:(i + 1) * 1080])
        return new_data
        # pass

    def segment_dataset1(self, data, idxs):
        # pass
        new_data = []
        tmp = idxs[1]
        idxs = idxs[1:] - tmp
        for i in range(len(idxs) - 1):
            new_data.append(data[idxs[i]:idxs[i + 1]])
            if idxs[i + 1] == idxs[-1]:
                new_data.append(data[idxs[i + 1]:len(data)])
        return new_data

    def read_dataset(self):
        f = open("D:/Databases/21/mit-bih-arrhythmia-database-1.0.0/RECORDS", "r")
        files = f.read().replace("\n", " ").split()

        inputs, targets = [], []
        ins, outs = [], []

        for file in files:
            tmp_path = os.path.join("D:/Databases/21/mit-bih-arrhythmia-database-1.0.0", file)
            record = wfdb.rdrecord(tmp_path)
            annotation = wfdb.rdann(tmp_path, 'atr')

            file_labels = []
            for i in range(len(annotation.sample) - 1):
                if i == 0 and annotation.symbol[i] == "+":
                    continue
                file_labels.extend(
                    [sybols_dict[annotation.symbol[i]]] * (annotation.sample[i + 1] - annotation.sample[i]))
                if annotation.sample[i + 1] == annotation.sample[-1]:
                    file_labels.extend(
                        [sybols_dict[annotation.symbol[i + 1]]] * (len(record.p_signal) - annotation.sample[i + 1]))

            segmented_inputs = self.segment_dataset(record.p_signal[annotation.sample[1]:])
            segmented_outputs = self.segment_dataset(file_labels)

            segmented_in = self.segment_dataset1(record.p_signal[annotation.sample[1]:], annotation.sample)
            segmented_out = self.segment_dataset1(file_labels, annotation.sample)
            print(len(segmented_in), len(segmented_out), len(annotation.sample))

            inputs.extend(segmented_inputs)
            targets.extend(segmented_outputs)
            ins.extend(segmented_in)
            outs.extend(segmented_out)
            # print(len(inputs), len(targets))

        X_train, X_test, y_train, y_test = train_test_split(np.array(inputs), np.array(targets), test_size=0.2,
                                                            random_state=42)
        # X_train1, X_test1, y_train1, y_test1 = train_test_split(np.array(ins), np.array(outs), test_size=0.2, random_state=42)
        # return X_train.transpose(0,2,1), X_test.transpose(0,2,1), y_train, y_test, ins, outs
        return X_train.transpose(0, 2, 1), X_test.transpose(0, 2, 1), y_train, y_test

    def read_dataset_peaks(self):
        f = open("D:/Databases/21/mit-bih-arrhythmia-database-1.0.0/RECORDS", "r")
        files = f.read().replace("\n", " ").split()

        inputs, targets = [], []

        for file in files:
            tmp_path = os.path.join("D:/Databases/21/mit-bih-arrhythmia-database-1.0.0", file)
            record = wfdb.rdrecord(tmp_path)
            annotation = wfdb.rdann(tmp_path, 'atr')

            size = len(record.p_signal)
            file_labels = [0] * size
            for i, idx in enumerate(annotation.sample):
                if annotation.symbol[i] != "+" and annotation.symbol[i] != "|" and annotation.symbol[i] != "\"" and \
                        annotation.symbol[i] != "~":
                    # tu można  raczej wyrzucić to co nie jest peakiem samym w sobie jak komentarz, czy zmiana jakości imo
                    file_labels[idx] = 1

            segmented_inputs = self.segment_dataset(record.p_signal)
            segmented_outputs = self.segment_dataset(file_labels)

            inputs.extend(segmented_inputs)
            targets.extend(segmented_outputs)

        X_train, X_test, y_train, y_test = train_test_split(np.array(inputs), np.array(targets), test_size=0.2,
                                                            random_state=42)
        return X_train.transpose(0, 2, 1), X_test.transpose(0, 2, 1), y_train, y_test
