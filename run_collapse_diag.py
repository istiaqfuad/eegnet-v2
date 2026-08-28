"""E5 — class-collapse diagnostics for the TTA objective ablation.

For chosen held-out subjects, train a LOSO model (default: Riemannian alignment),
then run tent_adapt with log_history=True under two objectives:
  - plain Tent (div_weight=0)  -> entropy-only, prone to class collapse
  - IM-TTA   (div_weight=1)    -> entropy + diversity

Saves per-step conditional entropy, marginal entropy, loss, and the predicted-class
histogram for both objectives -> the paper's collapse figure. Training is done once
per subject; both objectives adapt the SAME trained model.

Usage:
    python run_collapse_diag.py --subjects 2,5,4 --seed 42 --out collapse_diag
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_dataset, prepare_loso_data, BCIDataset
from model import UnifiedEEGNet
from train import _build_optim_sched, tent_adapt, evaluate


def train_loso_model(X_tr, y_tr, X_va, y_va, device, seed, n_classes, n_channels,
                     epochs=500, patience=150):
    """Same recipe as train_model (AdamW + OneCycleLR + label smoothing + augmentation),
    returning the val-selected model itself."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = UnifiedEEGNet(n_classes=n_classes, n_channels=n_channels).to(device)
    tr_ld = DataLoader(BCIDataset(X_tr, y_tr, augment=True, use_sr=True),
                       batch_size=64, shuffle=True)
    va_ld = DataLoader(BCIDataset(X_va, y_va, augment=False, use_sr=False),
                       batch_size=64, shuffle=False)
    opt, sched = _build_optim_sched(model, 0.001, 0.02, 0.005, epochs, len(tr_ld))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, best_state, no_imp = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        va_acc = evaluate(model, va_ld, device)
        if va_acc > best_acc:
            best_acc, best_state, no_imp = va_acc, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', default='2,5,4')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--align', default='ra')
    ap.add_argument('--out', default='collapse_diag')
    ap.add_argument('--tta_steps', type=int, default=5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    subjects = [int(s) for s in args.subjects.split(',') if s.strip()]

    X, y, meta, n_classes, n_channels = load_dataset('iv2a', fmin=0.5, fmax=100.0)

    for subject in subjects:
        X_tr, X_va, X_te, y_tr, y_va, y_te = prepare_loso_data(
            X, y, meta, subject, val_frac=0.1, seed=args.seed, align=args.align)
        te_ld = DataLoader(BCIDataset(X_te, y_te, augment=False, use_sr=False),
                           batch_size=64, shuffle=False)

        print(f"\n[diag] subject {subject}: training LOSO model (align={args.align})...")
        model = train_loso_model(X_tr, y_tr, X_va, y_va, device, args.seed,
                                 n_classes, n_channels)
        no_tta_acc = evaluate(model, te_ld, device)

        record = {'subject': subject, 'align': args.align, 'seed': args.seed,
                  'no_tta_acc': no_tta_acc, 'n_classes': n_classes, 'histories': {}}
        for name, div in [('tent', 0.0), ('im', 1.0)]:
            adapted = tent_adapt(model, te_ld, device, steps=args.tta_steps,
                                 div_weight=div, log_history=True)
            acc_after = evaluate(adapted, te_ld, device)   # separate inference pass
            record['histories'][name] = {
                'history': getattr(adapted, '_tta_history', []),
                'acc_after': acc_after,
            }
            print(f"[diag] subject {subject} {name}: no-TTA {no_tta_acc:.4f} -> "
                  f"{acc_after:.4f}")
        path = os.path.join(args.out, f'diag_s{subject}_seed{args.seed}.json')
        with open(path, 'w') as f:
            json.dump(record, f, indent=1)
        print(f"[diag] saved {path}")


if __name__ == '__main__':
    main()
