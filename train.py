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


def _build_optim_sched(model, lr, weight_decay, max_lr, n_epochs, steps_per_epoch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, epochs=max(1, n_epochs), steps_per_epoch=steps_per_epoch,
        pct_start=0.15, anneal_strategy='cos',
    )
    return optimizer, scheduler


def tent_adapt(model, loader, device, steps=1, lr=1e-3, div_weight=1.0):
    """Label-free source-free test-time adaptation via Information Maximization (SHOT-style).

    Adapts ONLY the BatchNorm affine params (gamma, beta) on the target subject's
    UNLABELLED trials; BN uses batch statistics. The loss is
        L = E[H(p)]  -  div_weight * H(E[p])
    i.e. minimise per-sample (conditional) entropy to sharpen predictions, while
    MAXIMISING the batch marginal entropy so the model cannot collapse all trials to
    one class (the failure mode of plain entropy-min / Tent on hard subjects).
    div_weight=0 reduces to plain Tent (entropy-only) — used for the ablation. No labels
    are used -> honest, transductive. Natural for LOSO. Returns an adapted deep copy.
    """
    model = copy.deepcopy(model)
    model.train()
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.track_running_stats = False           # use batch statistics
            m.running_mean = None
            m.running_var = None
            if m.affine:
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
                params += [m.weight, m.bias]
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
    if not params:
        return model
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        for x, _ in loader:
            x = x.to(device)
            p = model(x).softmax(1)
            cond_ent = -(p * p.clamp_min(1e-8).log()).sum(1).mean()   # minimise
            pbar = p.mean(0)
            div = -(pbar * pbar.clamp_min(1e-8).log()).sum()          # maximise (anti-collapse)
            loss = cond_ent - div_weight * div                        # div_weight=0 -> plain Tent
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_model(X_train, y_train, X_val, y_val, X_test, y_test, model_class, device,
                seed=42, epochs=500, batch_size=64,
                lr=0.001, weight_decay=0.02, max_lr=0.005,
                label_smoothing=0.1, patience=150,
                pretrained_state=None, use_ema=True, ema_decay=0.999,
                use_focal_loss=False, expand=1, refit=False, tta_steps=0, tta_lr=1e-3,
                tta_div=1.0):
    """Leak-free training.

    Model selection and early stopping use ONLY the validation set (X_val/y_val),
    which is carved from the training data and never includes test trials. The test
    set (X_test/y_test) is evaluated exactly once, at the end.

    expand: replicate augmented training epochs (more aug variety / steps per epoch).
    refit: after picking the best epoch on val, refit a fresh model on train+val for
           that many epochs (no val peeking), then test once. Uses all session-1 data
           for the final fit while keeping epoch-count selection leak-free.
    """
    from dataset import BCIDataset

    train_dataset = BCIDataset(X_train, y_train, augment=True, use_sr=True, expand=expand)
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
    optimizer, scheduler = _build_optim_sched(model, lr, weight_decay, max_lr, epochs, len(train_loader))

    # Selection is on the VALIDATION set only. Primary key: val acc (higher better);
    # tie-break: val loss (lower better) — the val set is small, so the loss tie-break
    # stabilises checkpoint choice.
    best_val_acc, best_val_loss, best_state, best_epoch, no_improve = -1.0, float('inf'), None, 0, 0
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
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"      Early stop at epoch {epoch+1} (no val improve for {patience} epochs)", flush=True)
            break

    if refit:
        # Refit a fresh model on train+val for the val-selected #epochs, then test once.
        n_ep = best_epoch + 1
        Xtv = np.concatenate([X_train, X_val], axis=0)
        ytv = np.concatenate([y_train, y_val], axis=0)
        refit_ds = BCIDataset(Xtv, ytv, augment=True, use_sr=True, expand=expand)
        refit_loader = DataLoader(refit_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        rmodel = model_class(n_classes=4, n_channels=22).to(device)
        if pretrained_state is not None:
            rmodel.load_state_dict(pretrained_state, strict=False)
        rcrit = FocalLoss(gamma=2.0) if use_focal_loss else nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        ropt, rsched = _build_optim_sched(rmodel, lr, weight_decay, max_lr, n_ep, len(refit_loader))
        for _ in range(n_ep):
            train_one_epoch(rmodel, refit_loader, ropt, rcrit, device, use_mixup=True, ema=None)
            rsched.step()
        test_acc = evaluate(rmodel, test_loader, device)
        print(f"      Best ValAcc: {best_val_acc:.4f} @ep{best_epoch+1} "
              f"-> REFIT(train+val,{n_ep}ep) Test (single eval): {test_acc:.4f}", flush=True)
        return test_acc

    # Final, single evaluation on the held-out TEST set with the val-selected checkpoint.
    final_model = ema.shadow if ema is not None else model
    if best_state is not None:
        final_model.load_state_dict(best_state)
    test_acc = evaluate(final_model, test_loader, device)
    msg = f"      Best ValAcc: {best_val_acc:.4f} -> Test (single eval): {test_acc:.4f}"

    if tta_steps > 0:
        # Label-free test-time adaptation on the held-out subject (transductive).
        adapted = tent_adapt(final_model, test_loader, device, steps=tta_steps, lr=tta_lr,
                             div_weight=tta_div)
        tta_acc = evaluate(adapted, test_loader, device)
        msg += f" | +TTA({tta_steps},div{tta_div}): {tta_acc:.4f}"
        test_acc = tta_acc
    print(msg, flush=True)
    return test_acc
