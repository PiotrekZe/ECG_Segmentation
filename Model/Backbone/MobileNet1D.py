import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(in_channels * expansion_factor)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        if expansion_factor == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv1d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv1d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv1d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv1d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv1d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNet1D(nn.Module):
    def __init__(self, input_channels, width_mult=1.):
        super(MobileNet1D, self).__init__()
        self.first_channel = 32
        self.last_channel = 1280

        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # block = InvertedResidualBlock
        block = InvertedResidual

        # First layer
        self.features = [nn.Sequential(
            nn.Conv1d(input_channels, self.first_channel, 3, 2, 1, bias=False),
            nn.BatchNorm1d(self.first_channel),
            nn.ReLU6(inplace=True)
        )]

        # Inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(self.first_channel, output_channel, s, expansion_factor=t))
                else:
                    self.features.append(
                        block(self.first_channel, output_channel, 1, expansion_factor=t))
                self.first_channel = output_channel

        # Last layer
        self.features.append(nn.Sequential(
            nn.Conv1d(self.first_channel, self.last_channel,
                      1, 1, 0, bias=False),
            nn.BatchNorm1d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))

        self.features = nn.ModuleList(self.features)

    def forward(self, x, return_layers=True):
        features = []
        for i, block in enumerate(self.features):
            x = block(x)
            features.append(x)
        if return_layers:
            return x, features
        else:
            return x
