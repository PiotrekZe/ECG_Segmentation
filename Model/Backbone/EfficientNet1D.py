import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None, activation=nn.SiLU):
        super(ConvBlock, self).__init__()
        layers = []
        if padding is None:
            padding = kernel_size // 2
        layers.extend([
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm1d(out_channels)
        ])
        if activation is not None:
            if 'inplace' in activation.__init__.__code__.co_varnames:
                layers.extend([activation(inplace=True)])
            else:
                layers.extend([activation()])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class SEBlock1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor):
        super(MBConv, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expansion_factor != 1:
            layers.extend([ConvBlock(in_channels=in_channels, out_channels=hidden_dim,
                          kernel_size=kernel_size, stride=1, groups=1)])
        layers.extend([
            ConvBlock(in_channels=hidden_dim, out_channels=hidden_dim,
                      kernel_size=kernel_size, stride=stride, groups=hidden_dim),
            SEBlock1D(in_channels=hidden_dim),
            ConvBlock(in_channels=hidden_dim, out_channels=out_channels,
                      kernel_size=kernel_size, stride=1, groups=1, activation=None)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        input = x
        if self.use_residual:
            for i in self.conv:
                x = i(x)
            return x + input
        else:
            for i in self.conv:
                x = i(x)
            return x


class EfficientNet1D(nn.Module):
    def __init__(self, in_channel):
        super(EfficientNet1D, self).__init__()
        self.last_channel = 1280
        self.stages = [
            # [Operator(f), Channels(c), Layers(l), Kernel(k), Stride(s), Expansion(exp)]
            [ConvBlock, 32, 1, 3, 2, 1],
            [MBConv, 16, 1, 3, 1, 1],
            [MBConv, 24, 2, 3, 2, 6],
            [MBConv, 40, 2, 5, 2, 6],
            [MBConv, 80, 3, 3, 2, 6],
            [MBConv, 112, 3, 5, 1, 6],
            [MBConv, 192, 4, 5, 2, 6],
            [MBConv, 320, 1, 3, 1, 6],
            [ConvBlock, 1280, 1, 1, 1, 0]
        ]
        layers = []
        last_channel = in_channel
        for i in self.stages:
            block, channel, num_layers, kernel, stride, expansion = i
            if block == ConvBlock:
                layers.extend([
                    block(in_channels=last_channel, out_channels=channel,
                          kernel_size=kernel, stride=stride)
                ])
                last_channel = channel
            elif block == MBConv:
                for j in range(num_layers):
                    if j == 0:
                        layers.extend([
                            block(in_channels=last_channel, out_channels=channel,
                                  kernel_size=kernel, stride=stride, expansion_factor=expansion)
                        ])
                    else:
                        layers.extend([
                            block(in_channels=last_channel, out_channels=channel,
                                  kernel_size=kernel, stride=1, expansion_factor=expansion)
                        ])

                    last_channel = channel

        self.conv = nn.Sequential(*layers)

    def forward(self, x, return_layers=True):
        features = []
        for i, block in enumerate(self.conv):
            x = block(x)
            features.append(x)
        if return_layers:
            return x, features
        else:
            return x
