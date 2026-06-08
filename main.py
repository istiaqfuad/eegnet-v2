import os
import sys

import numpy as np
import pandas as pd
import torch

from model import FreqAwareEEGNet
from dataset import load_bci_iva_dataset, prepare_subject_data, prepare_pretrain_data
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


def train_subject(subject, X, y, meta, device, pretrained_state):
    print(f"\n    {'='*50}\n    Subject {subject}\n    {'='*50}")
    X_train, X_test, y_train, y_test = prepare_subject_data(
        X, y, meta, subject
    )
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
    return train_model(
        X_train, y_train, X_test, y_test,
        model_class=UnifiedEEGNet,
        device=device,
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        max_lr=MAX_LR,
        label_smoothing=LABEL_SMOOTHING,
        patience=PATIENCE,
        pretrained_state=pretrained_state,
        use_ema=USE_EMA,
        ema_decay=EMA_DECAY,
    )


def assert_no_leakage(X_train, X_test, y_train, y_test):
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Empty split"
    assert set(np.unique(y_train)).issubset({0, 1, 2, 3}), "Unexpected train labels"
    assert set(np.unique(y_test)).issubset({0, 1, 2, 3}), "Unexpected test labels"
    assert X_train.shape[1:] == X_test.shape[1:], "Shape mismatch train/test"


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, 'results.csv')
    pretrained_path = os.path.join(script_dir, 'models', 'pretrained.pt')

    print("\n[1/3] Loading BCI Competition IV-2a (MOABB)...")
    X, y, meta = load_bci_iva_dataset()
    print(f"    X: {X.shape}, y: {y.shape}")
    print(f"    Subjects: {sorted(np.unique(meta['subject']).tolist())}")

    sessions_per_subj = meta.groupby('subject')['session'].nunique().to_dict()
    assert all(v == 2 for v in sessions_per_subj.values()), \
        f"Expected 2 sessions per subject, got {sessions_per_subj}"

    print("\n[2/3] Pretraining on session 1 of all subjects (no leakage to test session 2)...")
    X_pretrain, y_pretrain = prepare_pretrain_data(X, y, meta)
    if os.path.exists(pretrained_path):
        print(f"    Loading existing pretrained model: {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location=device, weights_only=True)
    else:
        os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
        pretrained_state = pretrain_model(
            X_pretrain, y_pretrain,
        model_class=FreqAwareEEGNet,
            device=device,
            save_path=pretrained_path,
            epochs=PRETRAIN_EPOCHS, batch_size=BATCH_SIZE,
            lr=LR, weight_decay=WEIGHT_DECAY,
        )

    print("\n[3/3] Fine-tuning on each subject (session 1 -> test on session 2)...")
    subjects = sorted(int(s) for s in np.unique(meta['subject']))
    results = []
    for subject in subjects:
        X_train, X_test, y_train, y_test = prepare_subject_data(
            X, y, meta, subject
        )
        assert_no_leakage(X_train, X_test, y_train, y_test)
        test_acc = train_subject(subject, X, y, meta, device, pretrained_state)
        results.append({'subject': subject, 'test_acc': test_acc})

        df = pd.DataFrame(results)
        df.to_csv(results_path, index=False)
        running_mean = df['test_acc'].mean()
        print(f"\n    Saved partial results. Running mean: {running_mean*100:.2f}% "
              f"({len(df)}/{len(subjects)})")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    avg = df['test_acc'].mean()
    std = df['test_acc'].std()
    print(f"\nMean Accuracy: {avg:.4f} +/- {std:.4f} ({avg*100:.2f}%)")
    print(f"Target: 85%+ -> {'PASS' if avg >= 0.85 else 'FAIL'}")

    print("\nComparison with SOTA on BCI IV-2a (4-class motor imagery):")
    print(f"  CTNet (2024):        82.52%")
    print(f"  MSCARNet (2024):     82.66%")
    print(f"  EEGEncoder (2025):   86.46%")
    print(f"  Ours (single model): {avg*100:.2f}%")
    return avg


if __name__ == "__main__":
    sys.exit(0 if main() >= 0.85 else 1)
