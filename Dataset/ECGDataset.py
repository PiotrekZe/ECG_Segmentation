from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np


class ECGDataset(Dataset):
    def __init__(self, inputs, targets, peaks, peaks_idx):
        self.inputs = inputs
        self.targets = targets
        self.peaks = peaks
        self.peaks_idx = peaks_idx
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        peak = self.peaks[index]
        peak_idx = self.peaks_idx[index]

        input = self.transform(input)

        target = torch.tensor(target, dtype=torch.long)
        peak = torch.tensor(peak, dtype=torch.long)
        peak_idx = torch.tensor(peak_idx, dtype=torch.long)

        return input[0], target, peak, peak_idx
