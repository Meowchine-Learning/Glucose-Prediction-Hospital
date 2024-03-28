# 2018, An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
# https://arxiv.org/abs/1803.01271
# https://github.com/locuslab/TCN

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels:list, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)  # [16, 2]
        for i in range(num_levels): # 3 
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HybridTCN(nn.Module):
    def __init__(self, num_inputs_static, num_inputs_seq, num_channels_seq, kernel_size=2, dropout=0.2):
        super(HybridTCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs_seq, num_channels_seq, kernel_size, dropout)
        self.conv = nn.Conv1d(num_channels_seq[-1], 1, 1)    # 16 -> 1
        self.linear1 = nn.Linear(num_inputs_static, 64)
        self.linear2 = nn.Linear(64 + num_channels_seq[-1]*num_inputs_seq, 1) # 64+len(x_seq_latent).flatten -> 1

    def forward(self, x_seq, x_static):
        # HybridTCN combines TCN with a Fully Connected Network for static features
        x_seq = x_seq.permute(0, 2, 1)      # (batch_size, seq_len, channels) -> (batch_size, channels, seq_len)
        seq_output = self.tcn(x_seq)        # Pass sequential features through TCN
        # output of TCN is (batch_size, num_channels_seq[-1], seq_len=3), flatten it for FC layer
        seq_output = seq_output.view(seq_output.size(0), -1)
        static_output = self.linear1(x_static)
        combined_output = torch.cat((seq_output, static_output), dim=1)
        final_output = self.linear2(combined_output)
        return final_output