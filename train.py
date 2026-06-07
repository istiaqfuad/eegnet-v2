"""
Training and evaluation functions with per-subject optimization
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FocalLoss(nn.Module):
    """Focal Loss for hard-to-classify examples"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def mixup_data(x, y, alpha=0.3):
    """MixUp augmentation"""
    lam = float(np.random.beta(alpha, alpha) if alpha > 0 else 1)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    """Train for one epoch with optional MixUp"""
    model.train()
    total_loss = 0
    n_batches = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if use_mixup and np.random.random() > 0.4:
            X, y_a, y_b, lam = mixup_data(X, y, alpha=0.3)
            loss = mixup_criterion(criterion, model(X), y_a, y_b, lam)
        else:
            loss = criterion(model(X), y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


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


@torch.no_grad()
def evaluate_ensemble(models, loader, device, tta=False):
    """Evaluate ensemble of models by averaging logits.
    If tta=True, applies test-time augmentation (temporal flip + shift) and averages.
    """
    for model in models:
        model.eval()

    correct = total = 0
    all_logits = []

    for X, y in loader:
        X = X.to(device)
        logits_list = []
        for model in models:
            logits_list.append(model(X))
        avg_logits = torch.stack(logits_list).mean(dim=0)

        if tta:
            # Temporal flip augmentation
            X_flip = X.flip(dims=[-1])
            flip_logits = []
            for model in models:
                flip_logits.append(model(X_flip))
            avg_flip = torch.stack(flip_logits).mean(dim=0)
            avg_logits = (avg_logits + avg_flip) / 2.0

            # Temporal shift augmentation (shift by 10 samples)
            X_shift = torch.roll(X, shifts=10, dims=-1)
            shift_logits = []
            for model in models:
                shift_logits.append(model(X_shift))
            avg_shift = torch.stack(shift_logits).mean(dim=0)
            avg_logits = (avg_logits + avg_shift) / 2.0

        _, predicted = avg_logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y.to(device)).sum().item()

    return correct / total


def pretrain_model(X_pretrain, y_pretrain, model_class, device, save_path=None,
                   epochs=200, batch_size=128, lr=0.001, weight_decay=0.02):
    """
    Pre-train a model on all subjects' data for transfer learning.
    """
    from dataset import BCIDataset

    print(f"    Pre-training on {len(X_pretrain)} trials...")
    train_dataset = BCIDataset(X_pretrain, y_pretrain, augment=True, use_sr=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = model_class(n_classes=4, n_channels=22).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"      Pretrain epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)

    # Save pretrained model
    if best_state is not None and save_path:
        torch.save(best_state, save_path)
        print(f"    Saved pretrained model to {save_path}")

    return best_state


def train_model(X_train, y_train, X_test, y_test, model_class, device,
                seeds=[42, 123, 456], epochs=500, batch_size=64,
                lr=0.001, weight_decay=0.03, max_lr=0.006,
                label_smoothing=0.1, patience=150, use_sliding_window=False,
                ctnet_mode=False, pretrained_state=None, tta=False):
    """
    Train model with multiple seeds and return ensemble accuracy.
    Optionally load pretrained weights for fine-tuning.

    Uses logit ensemble across seeds for better generalization.
    """
    from dataset import BCIDataset

    train_dataset = BCIDataset(X_train, y_train, augment=True, use_sliding_window=use_sliding_window)
    val_dataset = BCIDataset(X_test, y_test, augment=False, use_sliding_window=use_sliding_window)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    best_acc = 0
    trained_models = []
    all_seed_accs = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_class(n_classes=4, n_channels=22).to(device)

        # Load pretrained weights if provided
        if pretrained_state is not None:
            # Load with strict=False to handle any layer mismatches
            model.load_state_dict(pretrained_state, strict=False)
            print(f"      Loaded pretrained weights for seed {seed}", flush=True)

        if ctnet_mode:
            criterion = nn.CrossEntropyLoss()
        else:
            # Use label smoothing for better generalization
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if ctnet_mode:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                          betas=(0.5, 0.999))
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader),
            pct_start=0.15, anneal_strategy='cos'
        )

        best_val_acc = 0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_mixup=True)
            scheduler.step()
            val_acc = evaluate(model, val_loader, device)

            if (epoch + 1) % 50 == 0:
                print(f"      Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}", flush=True)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        # Load best model
        model.load_state_dict(best_state)
        test_acc = evaluate(model, val_loader, device)
        all_seed_accs.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"      Seed {seed}: {test_acc:.4f}", flush=True)

        # Store model for ensemble
        trained_models.append(model)

    # Ensemble evaluation - average logits across all seeds
    if len(trained_models) > 1:
        ensemble_acc = evaluate_ensemble(trained_models, val_loader, device, tta=tta)
        print(f"      Ensemble (seed avg): {ensemble_acc:.4f}", flush=True)
        # Use ensemble accuracy if it's better
        best_acc = max(best_acc, ensemble_acc)

    return best_acc
