import os
import wfdb
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, path, length=1800):
        self.path = path
        self.length = length
        pass

    def __segmentate_data(self, signal, peaks, labels):
        start, end = 0, self.length
        signals_tab, peaks_tab, labels_tab, peaks_idx = [], [], [], []
        while end < signal.shape[0] and start < signal.shape[0]:
            signals_tab.append(signal[start:end])
            peaks_tab.append(peaks[start:end])
            labels_tab.append(labels[start:end])

            idx = np.where(peaks_tab[-1] == 1)[0]
            peaks_idx.append(idx[-1])
            start += idx[-1]
            end = start + self.length
        return np.array(signals_tab), np.array(peaks_tab), np.array(labels_tab), peaks_idx

    def read_dataset(self):
        f = open(os.path.join(self.path, 'RECORDS'), 'r')
        files = f.read().replace('\n', ' ').split()

        peaks, targets, inputs, peaks_idx = [], [], [], []
        for file in files:

            # something has to be done with that file and similar situations
            if file == '232':
                continue

            file_path = os.path.join(self.path, file)

            record = wfdb.rdrecord(file_path)  # ECG signal
            # symbols and coords of R-peaks
            annotation = wfdb.rdann(file_path, 'atr')

            size = record.p_signal.shape[0]

            file_peaks = np.zeros(size)  # coords of R-peaks
            file_labels = np.zeros(size)  # labels of each time step
            tmp_sample, tmp_symbol = [], []  # coors and symbols of R-peaks

            # get rid of noninformative samples
            for i in range(len(annotation.symbol)):
                # idk if that list is the same for other data
                if annotation.symbol[i] not in ['+', '~', 'x', '|']:
                    tmp_sample.append(annotation.sample[i])
                    tmp_symbol.append(annotation.symbol[i])
                    file_peaks[annotation.sample[i]] = 1

            # creating labels list based on the
            for i in range(len(tmp_symbol)-1):
                if tmp_symbol[i] in ['N', 'L', 'R', 'e', 'j']:
                    file_labels[tmp_sample[i]:tmp_sample[i+1]] = 0
                elif tmp_symbol[i] in ['A', 'a', 'J', 'S']:
                    file_labels[tmp_sample[i]:tmp_sample[i+1]] = 1
                elif tmp_symbol[i] in ['V', 'E']:
                    file_labels[tmp_sample[i]:tmp_sample[i+1]] = 2
                # difference between my approach and the other published, they dont even have that label '!'
                elif tmp_symbol[i] in ['F', '!']:
                    file_labels[tmp_sample[i]:tmp_sample[i+1]] = 3
                elif tmp_symbol[i] in ['/', 'f', 'Q']:
                    file_labels[tmp_sample[i]:tmp_sample[i+1]] = 4

            # get rid of time steps behind first classified R-peak, and after the last one
            file_signals = record.p_signal[tmp_sample[0]:tmp_sample[-1]]
            file_labels = file_labels[tmp_sample[0]:tmp_sample[-1]]
            file_peaks = file_peaks[tmp_sample[0]:tmp_sample[-1]]

            # segmentate signals, labels, peaks into n sec segments
            segmented_signals, segmented_peaks, segmented_outputs, peak_idx = self.__segmentate_data(
                file_signals, file_peaks, file_labels)

            peaks.extend(segmented_peaks)
            inputs.extend(segmented_signals)
            targets.extend(segmented_outputs)
            peaks_idx.extend(peak_idx)

        X_train, X_test, y_train, y_test, peaks_train, peaks_test, peaks_idx_train, peaks_idx_test = train_test_split(
            np.array(inputs), np.array(targets), np.array(peaks), np.array(peaks_idx), test_size=0.2, random_state=42)

        return X_train.transpose(0, 2, 1), X_test.transpose(0, 2, 1), peaks_train, peaks_test, y_train, y_test, peaks_idx_train, peaks_idx_test
