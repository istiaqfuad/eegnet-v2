# Reproducibility Report

## Overview

This document specifies exact steps to reproduce every reported result from this codebase. Two model architectures are supported:

| Model | Params | Best Mean Accuracy | Novelty |
|-------|--------|-------------------|---------|
| `UnifiedEEGNet` | 34,132 | **83.56%** ± 9.96% | Pure CNN baseline |
| `FreqAwareEEGNet` | 37,392 | **83.06%** | Gated Frequency Attention |

Both are selectable via `--model unified` or `--model freqaware` (default).

---

## Quick Start

```bash
# Activate environment
uv sync

# Run FreqAwareEEGNet (Gated FreqAttn v3) — default
python3 main.py

# Run UnifiedEEGNet (baseline)
python3 main.py --model unified

# Or via run.sh
./run.sh          # default (freqaware)
./run.sh unified  # baseline
```

---

## Expected Results

### `--model unified` (UnifiedEEGNet, 34K params)

| Subject | Expected Accuracy |
|---------|-----------------|
| S1 | 90.97% |
| S2 | 66.67% |
| S3 | 95.14% |
| S4 | 84.38% |
| S5 | 72.57% |
| S6 | 73.96% |
| S7 | 89.58% |
| S8 | 89.58% |
| S9 | 89.24% |
| **Mean** | **83.56%** ± 9.96% |

### `--model freqaware` (FreqAwareEEGNet, 37K params)

| Subject | Expected Accuracy |
|---------|-----------------|
| S1 | 90.97% |
| S2 | 68.06% |
| S3 | 94.79% |
| S4 | 82.99% |
| S5 | 73.61% |
| S6 | 73.96% |
| S7 | 88.89% |
| S8 | 87.15% |
| S9 | 87.15% |
| **Mean** | **83.06%** |

Expected results assume the same fixed seed (42), same hardware class, and same MOABB version. Exact values may vary ±1–2pp due to GPU nondeterminism across different hardware.

---

## Hyperparameters (Both Models)

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.02 |
| Batch size | 64 |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| EMA | Disabled |
| Pretrain epochs | 200 |
| Fine-tune epochs | 500 (early stop patience 150) |
| Scheduler | CosineAnnealingLR (T_max=200 pretrain, T_max=500 fine-tune) |
| Mixup | α=0.3, 60% probability |

### Data Split

- **Pretrain**: All subjects, session 1 only (2592 trials, 4 classes)
- **Train per subject**: Session 2, first run
- **Test per subject**: Session 2, second run
- **No leakage**: Train and test sets share no trials. Train statistics only (no test data seen during normalization).

### Augmentation (Train Only)

| Augmentation | Parameters |
|-------------|-----------|
| S&R (Segmentation & Recombination) | 4 segments |
| Frequency masking | 15% prob, 0.2 max ratio |
| Channel dropout | 10% prob |
| Gaussian noise | 20% prob, σ=0.05 |
| Temporal shift | 20% prob, max shift=50 |
| Channel permutation | 15% prob |
| Sliding window | 2 windows (750 length, 375 step) |

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.11+ |
| PyTorch | 2.5.1+cu121 |
| MOABB | 1.3.0 |
| GPU | GTX 1050 Ti 4GB VRAM (CUDA 12.1) |
| Dataset | BCI Competition IV-2a (BNCI2014-001) |
| Preprocessing | MOABB `MotorImagery` paradigm, 0.5–100 Hz bandpass |

Dependencies are locked in `uv.lock`. To recreate:

```bash
uv sync
uv run python main.py --model unified   # baseline
uv run python main.py --model freqaware  # gated
```

### Runtime

- **Pretrain** (200 epochs on 2592 trials): ~12 minutes on GTX 1050 Ti
- **Fine-tune** (500 epochs per subject, ~150 actual due to early stop): ~5–8 minutes per subject
- **Total**: ~60–80 minutes for full 9-subject run

---

## Data Leakage Verification

The pipeline enforces strict session-based splitting:

```
Subject 1:
  Session 1 → ignored (used for pretrain across all subjects)
  Session 2:
    Run 1 → TRAIN
    Run 2 → TEST (never seen during training)
```

The assertion `assert_no_leakage()` in `main.py` validates:
- No empty splits
- All labels are within {0, 1, 2, 3}
- Train and test shapes match

Pretrain uses session 1 across **all** subjects, making it a cross-subject pretraining signal. The fine-tune test set (session 2, run 2) is never used during pretrain or training.

---

## Architecture Definitions

### UnifiedEEGNet (Baseline)

```
Input [B, 22, 1000]
  → unsqueeze → [B, 1, 22, 1000]
  → Conv2D(1→8, (1,64), pad=31)
  → BN → DepthwiseConv2D(8→16, (22,1), groups=8) → BN
  → ELU → AvgPool(1,4) → Dropout(0.35)
  → Conv2D(16→32, (1,16), pad=7) → BN → ELU
  → AvgPool(1,8) → Dropout(0.35)
  → Conv2D(32→32, (1,8), pad=3) → BN → ELU
  → AvgPool(1,4) → Dropout(0.35)
  → AdaptiveAvgPool(1,8) → Flatten → FC(256→64) → GELU → Dropout(0.5) → FC(64→4)
```

### FreqAwareEEGNet (Gated FreqAttn v3)

Same as UnifiedEEGNet backbone **plus** a residual FreqAttn side-path with gating:

```
Side-path (on raw input [B,1,22,1000]):
  → 3× parallel Conv2D(1→16, (1,k)) at k=7,15,31
  → Stack → MLP frequency attention → Softmax over bands
  → Weighted sum → Depthwise spatial(16→16, (22,1), groups=16) → ELU
  → 1×1 Conv(16→16)
  
Gate:
  → Main path after pool1: global avg pool → Linear(16→16) → Sigmoid
  → Gated residual: x = main_path + gate × freq_path
  
Total: +3,260 params (FreqAttn + gate) over UnifiedEEGNet
```

---

## Reproducibility Caveats

1. **MOABB dataset download**: First run downloads the dataset (~200 MB). Requires internet. Subsequent runs use local cache.
2. **GPU nondeterminism**: cuDNN convolution algorithms are non-deterministic by default. Expect ±1–2pp variance across different GPU hardware. Results above were on GTX 1050 Ti (CUDA 12.1).
3. **Pretrained weights cache**: `models/pretrained.pt` is cached after first pretrain run. Delete this file to force re-pretraining with the selected model architecture.
4. **Previous results.csv**: The file tracks all architecture results in rows. Running a new experiment appends to it. Back up before starting if you want to preserve.
5. **Seed 42 is fixed** in `main.py`. Changing it will produce different results.
6. **S2, S5, S6 are intrinsically hard subjects** in BCI IV-2a literature. Their performance is the primary bottleneck to 85%.

---

## Version History (Git)

| Commit | Description | Mean Acc |
|--------|-------------|----------|
| `6031934` | MTSEEGFormer (364K params) | 79.28% |
| `8430897` | UnifiedEEGNet (34K params) | **83.56%** |
| `ed040ce` | FreqAwareEEGNet v3 (37K params) | 83.06% |
| `a49edd3` | Cleanup test artifacts | — |
| (current) | Both models in main.py, argparse selectable | — |
