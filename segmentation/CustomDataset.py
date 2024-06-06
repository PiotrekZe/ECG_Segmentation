import torch
from torchvision import transforms
from torch.utils.data import Dataset


class ECG(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        segment = self.inputs[index]
        label = self.targets[index]

        segment = self.transform(segment)
        label = torch.tensor(label, dtype=torch.long)

        return segment[0], label
