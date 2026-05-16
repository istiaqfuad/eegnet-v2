#!/usr/bin/env python3
"""Quick test run for one subject"""
import numpy as np
import torch
from model import EEGNetMultiScale
from dataset import load_bci_iva_dataset, prepare_subject_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
X, y, meta = load_bci_iva_dataset()
print(f"Dataset: {X.shape}")

# Train on subject 1 only
subject = 1
X_train, X_test, y_train, y_test = prepare_subject_data(
    X, y, meta, subject, evaluation_mode='session_based'
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Quick training test
from dataset import BCIDataset
from torch.utils.data import DataLoader
import torch.nn as nn

train_dataset = BCIDataset(X_train, y_train, augment=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = EEGNetMultiScale(n_classes=4, n_channels=22).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train a few batches
model.train()
for i, (X_batch, y_batch) in enumerate(train_loader):
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    out = model(X_batch)
    loss = criterion(out, y_batch)
    loss.backward()
    optimizer.step()
    if i >= 2:
        break
print("Training test passed!")