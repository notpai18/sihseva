"""Tiny depthwise-separable CNN for wake-word detection -- the same family of
architecture (DS-CNN) used in real low-latency/on-device keyword spotters
(e.g. the "Hello Edge" reference models). Kept small on purpose: fewer params
means less compute per inference, which is the entire point of a low-latency
wake-word detector.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                    padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class TinyWakeWordNet(nn.Module):
    """Input: (batch, 1, n_mels, n_frames) log-mel spectrogram.
    Output: (batch,) raw logit -- sigmoid for P(wake word)."""

    def __init__(self, n_mels: int = 40, base_channels: int = 24):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=(10, 4), stride=(2, 2), padding=(4, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(c, c * 2, stride=2),
            DepthwiseSeparableConv(c * 2, c * 2, stride=1),
            DepthwiseSeparableConv(c * 2, c * 3, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c * 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
