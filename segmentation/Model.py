import torch.nn as nn
import torch
from CBAM import ChannelAttention, SpatialAttention


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attention1 = ChannelAttention(out_channels)
        self.attention2 = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.attention1(x) * x
        x = self.attention2(x) * x
        return x


class BottleneckLSTM(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(BottleneckLSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(2)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.relu1(self.batch1(x))
        # print(x.shape)
        # x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x, _ = self.lstm(x)
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.conv2(x)
        x = self.relu2(self.batch2(x))
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)

        return x


class UNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(UNet, self).__init__()
        self.encoder1 = ConvBlock1D(num_channels, 16)
        self.pool1 = nn.MaxPool1d(2)
        # self.attn1 = SimpleAttentionLayer(64, num_heads=8)
        self.attention1 = ChannelAttention(16)
        self.attention2 = SpatialAttention()

        self.encoder2 = ConvBlock1D(16, 32)
        self.pool2 = nn.MaxPool1d(2)
        # self.attn2 = SimpleAttentionLayer(128, num_heads=8)
        self.attention3 = ChannelAttention(32)
        self.attention4 = SpatialAttention()

        self.encoder3 = ConvBlock1D(32, 64)
        self.pool3 = nn.MaxPool1d(2)
        # self.attn3 = SimpleAttentionLayer(256, num_heads=8)
        self.attention5 = ChannelAttention(64)
        self.attention6 = SpatialAttention()

        self.encoder4 = ConvBlock1D(64, 128)
        self.pool4 = nn.MaxPool1d(2)
        # self.attn4 = SimpleAttentionLayer(512, num_heads=8)
        self.attention7 = ChannelAttention(128)
        self.attention8 = SpatialAttention()

        self.bottleneck = ConvBlock1D(128, 256)
        # self.bottleneck = BottleneckLSTM(128,256)
        self.attention9 = ChannelAttention(256)
        self.attention10 = SpatialAttention()

        self.upconv4 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock1D(256, 128)
        self.upconv3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock1D(128, 64)
        self.upconv2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock1D(64, 32)
        self.upconv1 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock1D(32, 16)

        self.final_conv = nn.Conv1d(16, num_classes, kernel_size=1)

    def pad_and_concat(self, upsampled, bypass, padding_value=0):
        diff = bypass.size(2) - upsampled.size(2)
        if diff > 0:
            # Pad upsampled tensor with the specified padding value
            upsampled = nn.functional.pad(upsampled, (0, diff), value=padding_value)
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # print("input", x.shape)
        enc1 = self.encoder1(x)
        # print("enc1", enc1.shape)

        enc2 = self.encoder2(self.pool1(enc1))
        # print("enc2", enc2.shape)

        enc3 = self.encoder3(self.pool2(enc2))
        # print("enc3", enc3.shape)

        enc4 = self.encoder4(self.pool3(enc3))
        # print("enc4", enc4.shape)

        bottleneck = self.bottleneck(self.pool4(enc4))
        # bottleneck = self.bottleneck(enc4)

        bottleneck = self.attention9(bottleneck) * bottleneck
        bottleneck = self.attention10(bottleneck) * bottleneck
        # print("bootleneck", bottleneck.shape)

        dec4 = self.upconv4(bottleneck)
        dec4 = self.pad_and_concat(dec4, enc4)
        dec4 = self.decoder4(dec4)
        # print("dec4", dec4.shape)

        dec3 = self.upconv3(dec4)
        enc3 = self.attention5(enc3) * enc3
        enc3 = self.attention6(enc3) * enc3
        dec3 = self.pad_and_concat(dec3, enc3)
        dec3 = self.decoder3(dec3)
        # print("dec3", dec3.shape)

        dec2 = self.upconv2(dec3)
        enc2 = self.attention3(enc2) * enc2
        enc2 = self.attention4(enc2) * enc2
        dec2 = self.pad_and_concat(dec2, enc2)
        dec2 = self.decoder2(dec2)
        # print("dec2", dec2.shape)

        dec1 = self.upconv1(dec2)
        enc1 = self.attention1(enc1) * enc1
        enc1 = self.attention2(enc1) * enc1
        dec1 = self.pad_and_concat(dec1, enc1)
        dec1 = self.decoder1(dec1)
        # print("dec1", dec1.shape)

        output = self.final_conv(dec1)
        # print("output", output.shape)
        return output
