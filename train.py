import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ModelEma:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def mixup_data(x, y, alpha=0.3):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=True, ema=None):
    model.train()
    total_loss, n_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if use_mixup and np.random.random() > 0.4:
            x_m, y_a, y_b, lam = mixup_data(x, y, alpha=0.3)
            loss = mixup_criterion(criterion, model(x_m), y_a, y_b, lam)
        else:
            loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(1).eq(y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_acc_loss(model, loader, device):
    """Return (accuracy, mean cross-entropy loss) over a loader."""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
            correct += logits.argmax(1).eq(y).sum().item()
            total += y.size(0)
    return correct / total, loss_sum / max(total, 1)


def pretrain_model(X_pretrain, y_pretrain, model_class, device, save_path=None,
                   epochs=200, batch_size=64, lr=0.001, weight_decay=0.02):
    from dataset import BCIDataset

    print(f"    Pre-training on {len(X_pretrain)} trials (session 1 of all subjects)")
    train_dataset = BCIDataset(X_pretrain, y_pretrain, augment=True, use_sr=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = model_class(n_classes=4, n_channels=22).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss, best_state = float('inf'), None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
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

    if best_state is not None and save_path:
        torch.save(best_state, save_path)
        print(f"    Saved pretrained model to {save_path}")
    return best_state


def train_model(X_train, y_train, X_val, y_val, X_test, y_test, model_class, device,
                seed=42, epochs=500, batch_size=64,
                lr=0.001, weight_decay=0.02, max_lr=0.005,
                label_smoothing=0.1, patience=150,
                pretrained_state=None, use_ema=True, ema_decay=0.999,
                use_focal_loss=False):
    """Leak-free training.

    Model selection and early stopping use ONLY the validation set (X_val/y_val),
    which is carved from the training data and never includes test trials. The test
    set (X_test/y_test) is evaluated exactly once, at the end, on the val-selected
    checkpoint. The test set is never observed during training or selection.
    """
    from dataset import BCIDataset

    train_dataset = BCIDataset(X_train, y_train, augment=True, use_sr=True)
    val_dataset = BCIDataset(X_val, y_val, augment=False, use_sr=False)
    test_dataset = BCIDataset(X_test, y_test, augment=False, use_sr=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_class(n_classes=4, n_channels=22).to(device)
    if pretrained_state is not None:
        model.load_state_dict(pretrained_state, strict=False)
        print(f"      Loaded pretrained weights (seed {seed})", flush=True)

    ema = ModelEma(model, decay=ema_decay) if use_ema else None

    criterion = FocalLoss(gamma=2.0) if use_focal_loss else nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.15, anneal_strategy='cos',
    )

    # Selection is on the VALIDATION set only. Primary key: val acc (higher better);
    # tie-break: val loss (lower better) — the val set is small, so the loss tie-break
    # stabilises checkpoint choice.
    best_val_acc, best_val_loss, best_state, no_improve = -1.0, float('inf'), None, 0
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_mixup=True, ema=ema)
        scheduler.step()
        eval_model = ema.shadow if ema is not None else model
        val_acc, val_loss = evaluate_acc_loss(eval_model, val_loader, device)
        if (epoch + 1) % 50 == 0:
            print(f"      Epoch {epoch+1}/{epochs}, TrLoss: {loss:.4f}, "
                  f"ValAcc: {val_acc:.4f}, ValLoss: {val_loss:.4f}", flush=True)
        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc, best_val_loss = val_acc, val_loss
            best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"      Early stop at epoch {epoch+1} (no val improve for {patience} epochs)", flush=True)
            break

    # Final, single evaluation on the held-out TEST set with the val-selected checkpoint.
    final_model = ema.shadow if ema is not None else model
    if best_state is not None:
        final_model.load_state_dict(best_state)
    test_acc = evaluate(final_model, test_loader, device)
    print(f"      Best ValAcc: {best_val_acc:.4f} -> Test (single eval): {test_acc:.4f}", flush=True)
    return test_acc
