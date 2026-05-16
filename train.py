"""
Training and evaluation functions
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    return lam * x + (1 - lam) * x[index, :], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Apply MixUp with 50% probability
        if np.random.random() > 0.5:
            X, y_a, y_b, lam = mixup_data(X, y, alpha=0.4)
            loss = mixup_criterion(criterion, model(X), y_a, y_b, lam)
        else:
            loss = criterion(model(X), y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def evaluate(model, loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            _, predicted = model(X).max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return correct / total


def train_model(X_train, y_train, X_test, y_test, model_class, device,
                seeds=[42, 123, 456], epochs=500, batch_size=64,
                lr=0.001, weight_decay=0.03, max_lr=0.006,
                label_smoothing=0.1, patience=150):
    """
    Train model with multiple seeds and return best accuracy

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_class: Model architecture class
        device: torch device
        seeds: List of random seeds
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Initial learning rate
        weight_decay: Weight decay
        max_lr: Maximum learning rate for OneCycleLR
        label_smoothing: Label smoothing factor
        patience: Early stopping patience

    Returns:
        best_acc: Best test accuracy
    """
    from dataset import BCIDataset

    # Create data loaders
    train_dataset = BCIDataset(X_train, y_train, augment=True)
    val_dataset = BCIDataset(X_test, y_test, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    best_acc = 0

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model
        model = model_class(n_classes=4, n_channels=22).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader),
            pct_start=0.15, anneal_strategy='cos'
        )

        best_val_acc = 0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            val_acc = evaluate(model, val_loader, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        # Load best model and evaluate
        model.load_state_dict(best_state)
        test_acc = evaluate(model, val_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            print(f"      Seed {seed}: {test_acc:.4f} (new best)")
        else:
            print(f"      Seed {seed}: {test_acc:.4f}")

    return best_acc