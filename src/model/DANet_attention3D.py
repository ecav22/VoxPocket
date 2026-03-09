import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.channels = channels

    def forward(self, x):
        b, c, d, h, w = x.shape
        proj = x.view(b, c, -1)
        energy = torch.bmm(proj, proj.transpose(1, 2))
        attn = F.softmax(energy, dim=-1)
        out = torch.bmm(attn, proj).view(b, c, d, h, w)
        return self.gamma * out + x


class PositionAttention3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        inter = max(1, channels // 2)
        self.query_conv = nn.Conv3d(channels, inter, kernel_size=1)
        self.key_conv = nn.Conv3d(channels, inter, kernel_size=1)
        self.value_conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, d, h, w = x.shape
        n = d * h * w

        query = self.query_conv(x).view(b, -1, n).permute(0, 2, 1)
        key = self.key_conv(x).view(b, -1, n)
        energy = torch.bmm(query, key)
        attn = F.softmax(energy, dim=-1)

        value = self.value_conv(x).view(b, c, n)
        out = torch.bmm(value, attn.permute(0, 2, 1)).view(b, c, d, h, w)

        return self.gamma * out + x


class DANet3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention3D(channels)
        self.position_att = PositionAttention3D(channels)

    def forward(self, x):
        x1 = self.channel_att(x)
        x2 = self.position_att(x)
        return x1 + x2
