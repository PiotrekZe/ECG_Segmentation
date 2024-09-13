import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Backbone.MobileNet1D import MobileNet1D
from Model.Backbone.EfficientNet1D import EfficientNet1D
from Model.Backbone.MobileNet1D import InvertedResidual as InvertedResidualBlock
import math
from Model.TransUNet.decoder import Decoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2048, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers)

    def forward(self, src):
        src = self.pos_encoder(src)
        return self.transformer_encoder(src)


class TransUNet(nn.Module):
    def __init__(self, num_classes, input_channel, d_model, n_head, num_transformer_layers, input_signal, widt_mult=1., encoder_name='EfficientNet'):
        super(TransUNet, self).__init__()

        # here i should check if user gave right encoder

        # that part, with the upconv part should be customizable - idk how to make it
        # self.backbone = MobileNet1D(input_channels=input_channel)
        self.backbone = EfficientNet1D(in_channel=input_channel)

        # here is bottleneck transformer
        self.conv_trans = InvertedResidualBlock(
            in_channels=self.backbone.last_channel, out_channels=d_model, stride=1, expansion_factor=1)
        self.transformer = TransformerEncoder(
            d_model=d_model, nhead=n_head, num_layers=num_transformer_layers)
        self.trans_conv = InvertedResidualBlock(
            in_channels=d_model, out_channels=self.backbone.last_channel, stride=1, expansion_factor=1)

        self.encoder_name = encoder_name
        if encoder_name == 'MobileNet':
            in_channels_tab = [1280, 64, 48, 64]
            out_channels_tab = [32, 24, 32, num_classes]
        elif encoder_name == 'EfficientNet':
            in_channels_tab = [1280, 80, 48, 64]
            out_channels_tab = [40, 24, 32, num_classes]
        else:
            # here could be other encoder versions
            pass

        self.decoder_cnn = Decoder(
            input_signal, in_channels_tab, out_channels_tab)

    def forward(self, x):
        output, list_layers = self.backbone(x)

        output = self.conv_trans(output)
        output = output.permute(2, 0, 1)
        output = self.transformer(output)
        output = output.permute(1, 2, 0)
        output = self.trans_conv(output)

        if self.encoder_name == 'MobileNet':
            list_features = [list_layers[0], list_layers[2], list_layers[6]]
        elif self.encoder_name == 'EfficientNet':
            list_features = [list_layers[0], list_layers[2], list_layers[5]]
        output = self.decoder_cnn(output, list_features)
        return output
