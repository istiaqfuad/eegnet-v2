#!/usr/bin/env python3
"""Minimal test - just test the model architecture"""
import torch

# Just test the model import and forward pass
import sys
sys.path.insert(0, '/home/istiaqfuad/Desktop/last-bci')

from model import EEGNetMultiScale

print("Creating model...")
model = EEGNetMultiScale(n_classes=4, n_channels=22)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Test forward pass
print("Testing forward pass...")
x = torch.randn(2, 22, 1000)
out = model(x)
print(f"Output shape: {out.shape}")
print("SUCCESS!")