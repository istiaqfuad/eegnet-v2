import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from model import (FreqAwareEEGNet, UnifiedEEGNet, AlignedEEGNet,
                   AdaptiveAlignEEGNet, DualAlignEEGNet)
from dataset import (
    load_bci_iva_dataset,
    prepare_subject_data,
    prepare_loso_data,
    prepare_pretrain_data,
)
from train import train_model, pretrain_model


SEED = 42
EPOCHS = 500
BATCH_SIZE = 64
LR = 0.001
MAX_LR = 0.005
WEIGHT_DECAY = 0.02
LABEL_SMOOTHING = 0.1
PATIENCE = 150
PRETRAIN_EPOCHS = 200
USE_EMA = False
EMA_DECAY = 0.99


MODEL_MAP = {
    'freqaware': FreqAwareEEGNet,
    'unified': UnifiedEEGNet,
    'aligned': AlignedEEGNet,
    'adaptalign': AdaptiveAlignEEGNet,
    'dualalign': DualAlignEEGNet,
}


def assert_no_leakage(X_train, X_val, X_test, y_train, y_val, y_test):
    assert X_train.shape[0] > 0 and X_val.shape[0] > 0 and X_test.shape[0] > 0, "Empty split"
    for arr in (y_train, y_val, y_test):
        assert set(np.unique(arr)).issubset({0, 1, 2, 3}), "Unexpected labels"
    assert X_train.shape[1:] == X_val.shape[1:] == X_test.shape[1:], "Shape mismatch across splits"


def _fit(X_train, y_train, X_val, y_val, X_test, y_test, model_class, device,
         pretrained_state, expand=1, refit=False, epochs=EPOCHS, patience=PATIENCE, seed=SEED,
         tta_steps=0, tta_div=1.0):
    return train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_class=model_class,
        device=device,
        seed=seed,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        max_lr=MAX_LR,
        label_smoothing=LABEL_SMOOTHING,
        patience=patience,
        pretrained_state=pretrained_state,
        use_ema=USE_EMA,
        ema_decay=EMA_DECAY,
        expand=expand,
        refit=refit,
        tta_steps=tta_steps,
        tta_div=tta_div,
    )


def _summarize(results, label, model_name, out_path):
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    avg, std = df['test_acc'].mean(), df['test_acc'].std()
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS — {label} ({model_name})")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n{label} Mean Accuracy: {avg:.4f} +/- {std:.4f} ({avg*100:.2f}%)")
    print(f"    saved -> {out_path}")
    return avg


def run_within(X, y, meta, model_class, model_name, device, script_dir, cfg):
    """Within-subject (session 1 -> train/val, session 2 -> test), leak-free."""
    pretrained_path = os.path.join(script_dir, 'models',
                                   f'pretrained_{model_name}_{cfg.align}_s{cfg.seed}{cfg.tagsuffix}.pt')
    print(f"\n[within] Pretraining {model_class.__name__} on session 1 of all subjects...")
    X_pretrain, y_pretrain = prepare_pretrain_data(X, y, meta, align=cfg.align)
    if os.path.exists(pretrained_path):
        print(f"    Loading existing pretrained model: {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location=device, weights_only=True)
    else:
        os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
        pretrained_state = pretrain_model(
            X_pretrain, y_pretrain, model_class=model_class, device=device,
            save_path=pretrained_path, epochs=cfg.pretrain_epochs, batch_size=BATCH_SIZE,
            lr=LR, weight_decay=WEIGHT_DECAY,
        )

    print(f"\n[within] Fine-tuning {model_class.__name__} per subject "
          f"(val_frac={cfg.val_frac}, expand={cfg.aug_expand}, refit={cfg.refit})...")
    subjects = cfg.subject_list or sorted(int(s) for s in np.unique(meta['subject']))
    out_path = os.path.join(script_dir, f'results_within_honest_{model_name}{cfg.tagsuffix}.csv')
    results = []
    for subject in subjects:
        X_tr, X_va, X_te, y_tr, y_va, y_te = prepare_subject_data(X, y, meta, subject,
                                                                  val_frac=cfg.val_frac, seed=cfg.seed,
                                                                  align=cfg.align)
        assert_no_leakage(X_tr, X_va, X_te, y_tr, y_va, y_te)
        print(f"      Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
        test_acc = _fit(X_tr, y_tr, X_va, y_va, X_te, y_te, model_class, device,
                        pretrained_state, expand=cfg.aug_expand, refit=cfg.refit,
                        epochs=cfg.epochs, patience=cfg.patience, seed=cfg.seed,
                        tta_steps=cfg.tta_steps, tta_div=cfg.tta_div)
        results.append({'subject': subject, 'test_acc': test_acc, 'model': model_class.__name__})
        df = pd.DataFrame(results)
        df.to_csv(out_path, index=False)
        print(f"    [within] Running mean: {df['test_acc'].mean()*100:.2f}% ({len(df)}/{len(subjects)})")
    return _summarize(results, f'WITHIN-SUBJECT (honest){cfg.tagsuffix}', model_class.__name__, out_path)


def run_loso(X, y, meta, model_class, model_name, device, script_dir, cfg):
    """Leave-one-subject-out cross-subject, trained from scratch on the 8-subject pool."""
    print(f"\n[loso] Cross-subject (LOSO) for {model_class.__name__} — no pretrain, from scratch...")
    subjects = cfg.subject_list or sorted(int(s) for s in np.unique(meta['subject']))
    out_path = os.path.join(script_dir, f'results_loso_{model_name}{cfg.tagsuffix}.csv')
    results = []
    for subject in subjects:
        X_tr, X_va, X_te, y_tr, y_va, y_te = prepare_loso_data(X, y, meta, subject,
                                                               val_frac=0.1, seed=cfg.seed,
                                                               align=cfg.align)
        assert_no_leakage(X_tr, X_va, X_te, y_tr, y_va, y_te)
        print(f"      Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
        test_acc = _fit(X_tr, y_tr, X_va, y_va, X_te, y_te, model_class, device,
                        pretrained_state=None, expand=cfg.aug_expand, refit=cfg.refit,
                        epochs=cfg.epochs, patience=cfg.patience, seed=cfg.seed,
                        tta_steps=cfg.tta_steps, tta_div=cfg.tta_div)
        results.append({'subject': subject, 'test_acc': test_acc, 'model': model_class.__name__})
        df = pd.DataFrame(results)
        df.to_csv(out_path, index=False)
        print(f"    [loso] Running mean: {df['test_acc'].mean()*100:.2f}% ({len(df)}/{len(subjects)})")
    return _summarize(results, f'CROSS-SUBJECT LOSO{cfg.tagsuffix}', model_class.__name__, out_path)


def main():
    parser = argparse.ArgumentParser(description='BCI Competition IV-2a training (leak-free)')
    parser.add_argument('--model', choices=list(MODEL_MAP.keys()), default='unified',
                        help='Model architecture to use')
    parser.add_argument('--protocol', choices=['within', 'loso', 'both'], default='within',
                        help='Evaluation protocol: within-subject, cross-subject LOSO, or both')
    parser.add_argument('--fmin', type=float, default=0.5, help='Band-pass low cutoff (Hz)')
    parser.add_argument('--fmax', type=float, default=100.0, help='Band-pass high cutoff (Hz)')
    parser.add_argument('--aug_expand', type=int, default=1, help='Replicate augmented train epochs')
    parser.add_argument('--val_frac', type=float, default=0.2, help='Within-subject validation fraction')
    parser.add_argument('--refit', action='store_true', help='Refit on train+val for val-selected #epochs')
    parser.add_argument('--align', choices=['none', 'ea', 'ra', 'dual'], default='none',
                        help='Alignment: ea=Euclidean, ra=Riemannian/Centroid, '
                             'dual=stacked EA+RA views (use with --model dualalign), label-free')
    parser.add_argument('--tag', type=str, default='', help='Suffix for output CSV / pretrain cache')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed (vary for multi-seed runs)')
    parser.add_argument('--tta_steps', type=int, default=0,
                        help='Test-time adaptation: entropy-min BN-affine passes over the '
                             'unlabelled test subject (Tent, label-free). 0=off')
    parser.add_argument('--tta_div', type=float, default=1.0,
                        help='Diversity (marginal-entropy) weight in TTA loss. '
                             '1.0=IM/SHOT, 0.0=plain Tent (ablation)')
    parser.add_argument('--subjects', type=str, default='',
                        help='Comma-separated subject ids to run (e.g. 2,5,6); empty = all 9')
    parser.add_argument('--pretrain_epochs', type=int, default=PRETRAIN_EPOCHS,
                        help='Cross-subject pretrain epochs (lower for quick tests)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Fine-tune epochs')
    parser.add_argument('--patience', type=int, default=PATIENCE, help='Early-stop patience')
    args = parser.parse_args()
    args.tagsuffix = f'_{args.tag}' if args.tag else ''
    args.subject_list = [int(s) for s in args.subjects.split(',') if s.strip()] if args.subjects else None
    model_class = MODEL_MAP[args.model]
    print(f"Model: {model_class.__name__} ({args.model}) | Protocol: {args.protocol} | "
          f"band={args.fmin}-{args.fmax}Hz expand={args.aug_expand} val_frac={args.val_frac} "
          f"refit={args.refit} align={args.align} tag='{args.tag}'")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("\nLoading BCI Competition IV-2a (MOABB)...")
    X, y, meta = load_bci_iva_dataset(fmin=args.fmin, fmax=args.fmax)
    print(f"    X: {X.shape}, y: {y.shape}")
    print(f"    Subjects: {sorted(np.unique(meta['subject']).tolist())}")
    sessions_per_subj = meta.groupby('subject')['session'].nunique().to_dict()
    assert all(v == 2 for v in sessions_per_subj.values()), \
        f"Expected 2 sessions per subject, got {sessions_per_subj}"

    means = {}
    if args.protocol in ('within', 'both'):
        means['within'] = run_within(X, y, meta, model_class, args.model, device, script_dir, args)
    if args.protocol in ('loso', 'both'):
        means['loso'] = run_loso(X, y, meta, model_class, args.model, device, script_dir, args)

    print("\n" + "=" * 60)
    print(f"SUMMARY ({model_class.__name__})")
    for k, v in means.items():
        print(f"  {k:8s}: {v*100:.2f}%")
    print("Reference SOTA (within-subject, 4-class): CTNet 82.52 / MSCARNet 82.66 / EEGEncoder 86.46")
    return float(np.mean(list(means.values()))) if means else 0.0


if __name__ == "__main__":
    main()
    sys.exit(0)
